//! Chunk administration (Stage 3, 14.1-14.2).
//!
//! - 14.1 Chunk visualisation and manual intervention: list a document's
//!   chunks (preview), split one chunk at a line, merge adjacent chunks, or
//!   replace the whole boundary list. Boundaries persist as a per-document
//!   override (`chunk_overrides`); the indexer applies the override on the
//!   next index instead of the configured chunking method, so an edit takes
//!   effect after reindexing.
//! - 14.2 Document metadata (author/date/source): stored per document and
//!   usable as a retrieval filter (`search_with_metadata_filter`).
use anyhow::{bail, Result};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};

use crate::store::Store;
use crate::text::Chunk;

/// One chunk boundary in an override: line range + heading (14.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkBoundary {
    pub start_line: usize,
    pub end_line: usize,
    pub heading: String,
}

/// Preview a document's indexed chunks (14.1).
pub fn list_chunks(store: &Store, path: &str) -> Result<Vec<Chunk>> {
    let document_id: i64 = store
        .connection()
        .query_row("SELECT id FROM documents WHERE path=?", [path], |r| r.get(0))
        .optional()?
        .ok_or_else(|| anyhow::anyhow!("文档未索引：{path}"))?;
    let db = store.connection();
    let mut stmt = db.prepare("SELECT heading,ranges,spans FROM chunks WHERE document_id=? ORDER BY id")?;
    let rows = stmt.query_map([document_id], |r| {
        let heading: String = r.get(0)?;
        let ranges: Vec<(usize, usize)> = serde_json::from_str(&r.get::<_, String>(1)?).unwrap_or_default();
        let spans: Vec<crate::types::SourceSpan> = serde_json::from_str(&r.get::<_, String>(2)?).unwrap_or_default();
        Ok((heading, ranges, spans))
    })?.collect::<rusqlite::Result<Vec<_>>>()?;
    // Content is read from the file on demand; here we return the stored
    // ranges/spans with an empty content placeholder filled by the caller.
    // For preview purposes the heading + line ranges are the stable part.
    Ok(rows
        .into_iter()
        .map(|(heading, ranges, spans)| Chunk { heading, content: String::new(), ranges, spans })
        .collect())
}

/// Read a document's current chunk boundaries (14.1). A saved manual
/// override is the current working state; otherwise the indexed chunks.
pub fn current_boundaries(store: &Store, path: &str) -> Result<Vec<ChunkBoundary>> {
    if let Some(boundaries) = store.chunk_boundaries(path)? {
        return Ok(boundaries);
    }
    let chunks = list_chunks(store, path)?;
    Ok(chunks
        .into_iter()
        .map(|c| ChunkBoundary {
            start_line: c.ranges.first().map(|r| r.0).unwrap_or(1),
            end_line: c.ranges.last().map(|r| r.1).unwrap_or(1),
            heading: c.heading,
        })
        .collect())
}

/// Split one chunk at a line (14.1): the chunk covering `at_line` is divided
/// into [start, at_line-1] and [at_line, end]. Both halves keep the heading.
pub fn split_chunk(store: &Store, path: &str, at_line: usize) -> Result<Vec<ChunkBoundary>> {
    let mut boundaries = current_boundaries(store, path)?;
    let Some(idx) = boundaries.iter().position(|b| b.start_line <= at_line && at_line <= b.end_line) else {
        bail!("行 {at_line} 不在任何块内")
    };
    let boundary = boundaries[idx].clone();
    if at_line <= boundary.start_line || at_line > boundary.end_line {
        bail!("只能在块内部按行拆分")
    }
    let second = ChunkBoundary { start_line: at_line, end_line: boundary.end_line, heading: boundary.heading.clone() };
    let first = ChunkBoundary { end_line: at_line - 1, ..boundary };
    boundaries.splice(idx..=idx, [first, second]);
    store.set_chunk_boundaries(path, &boundaries)?;
    Ok(boundaries)
}

/// Merge adjacent chunks (14.1): chunks `index` and `index+1` become one,
/// keeping the first chunk's heading.
pub fn merge_chunks(store: &Store, path: &str, index: usize) -> Result<Vec<ChunkBoundary>> {
    let mut boundaries = current_boundaries(store, path)?;
    if index + 1 >= boundaries.len() {
        bail!("没有可合并的相邻块")
    }
    let second = boundaries.remove(index + 1);
    boundaries[index].end_line = second.end_line;
    store.set_chunk_boundaries(path, &boundaries)?;
    Ok(boundaries)
}

/// Document metadata (14.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub author: String,
    pub date: String,
    pub source: String,
}

/// Metadata filter (14.2): empty fields match everything.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetadataFilter {
    pub author: String,
    pub date: String,
    pub source: String,
}

impl MetadataFilter {
    pub fn matches(&self, metadata: &DocumentMetadata) -> bool {
        (self.author.is_empty() || metadata.author.contains(&self.author))
            && (self.date.is_empty() || metadata.date.contains(&self.date))
            && (self.source.is_empty() || metadata.source.contains(&self.source))
    }
}

/// Search with a metadata filter (14.2): documents whose metadata matches
/// the filter, intersected with the given scope, then BM25 chunk search.
pub fn search_with_metadata_filter(
    store: &Store,
    query: &str,
    scope: &[String],
    filter: &MetadataFilter,
    limit: usize,
) -> Result<Vec<crate::types::Candidate>> {
    let matching = store.documents_matching_metadata(filter)?;
    // An empty match set means "no documents", not "no scope restriction":
    // search_files treats an empty scope as "everything".
    if matching.is_empty() {
        return Ok(vec![]);
    }
    let mut scoped: Vec<String> = matching;
    if !scope.is_empty() {
        scoped.retain(|p| scope.iter().any(|s| p.starts_with(s.trim_end_matches('/'))));
        if scoped.is_empty() {
            return Ok(vec![]);
        }
    }
    store.search_files(query, &scoped, limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn setup() -> (tempfile::TempDir, Store, String) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("kb");
        fs::create_dir(&root).unwrap();
        let store = Store::open(&dir.path().join("app/db.sqlite")).unwrap();
        let root_key = store.add_root(&root).unwrap();
        let doc = root.canonicalize().unwrap().join("guide.md");
        fs::write(&doc, "# 部署\n\n第一行。\n第二行。\n第三行。\n").unwrap();
        store.scan(&root_key, |_| {}).unwrap();
        let path = crate::store::path_text(&doc).unwrap();
        (dir, store, path)
    }

    #[test]
    fn chunk_preview_split_and_merge_round_trip() {
        let (_dir, store, path) = setup();
        let boundaries = current_boundaries(&store, &path).unwrap();
        assert!(!boundaries.is_empty());
        // Split the first chunk that spans more than one line.
        let multi = boundaries.iter().find(|b| b.end_line > b.start_line).unwrap().clone();
        let split_at = multi.start_line + 1;
        let after_split = split_chunk(&store, &path, split_at).unwrap();
        assert_eq!(after_split.len(), boundaries.len() + 1);
        assert!(after_split.iter().any(|b| b.end_line == split_at - 1));
        assert!(after_split.iter().any(|b| b.start_line == split_at));
        // 14.1: the override takes effect after reindexing.
        let roots = store.prepare_reindex().unwrap();
        for root in roots {
            crate::indexer::apply_request(&store, crate::indexer::IndexRequest::Scan { root, verify: true }, |_| {}).unwrap();
        }
        let indexed = current_boundaries(&store, &path).unwrap();
        assert_eq!(indexed.len(), boundaries.len() + 1, "reindexed chunks must follow the override");
        assert_eq!(indexed, after_split);
        // Merge them back.
        let idx = after_split.iter().position(|b| b.end_line == split_at - 1).unwrap();
        let after_merge = merge_chunks(&store, &path, idx).unwrap();
        assert_eq!(after_merge.len(), boundaries.len());
        assert_eq!(after_merge, boundaries);
    }

    #[test]
    fn metadata_filter_narrows_search_to_matching_documents() {
        let (_dir, store, path) = setup();
        store
            .set_document_metadata(
                &path,
                &DocumentMetadata { author: "小晚".into(), date: "2026-01-01".into(), source: "小红书".into() },
            )
            .unwrap();
        let got = store.document_metadata(&path).unwrap();
        assert_eq!(got.author, "小晚");
        let hit = search_with_metadata_filter(&store, "第一行", &[], &MetadataFilter { author: "小晚".into(), ..Default::default() }, 10).unwrap();
        assert_eq!(hit.len(), 1, "author filter must keep the match");
        let miss = search_with_metadata_filter(&store, "第一行", &[], &MetadataFilter { author: "别人".into(), ..Default::default() }, 10).unwrap();
        assert!(miss.is_empty(), "author filter must exclude the non-match");
    }
}
