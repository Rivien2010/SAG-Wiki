//! Document-shape chunking (Stage 2, 10.1-10.5).
//!
//! The default chunker (`text::chunks_for_index`) splits on sentences and
//! keeps the nearest Markdown heading. These builders add shape-aware
//! splitting on top of the same `Chunk` struct, so retrieval, fusion and
//! citations keep working unchanged:
//! - 10.1 Hierarchy: heading tree, each chunk carries the full parent path
//!   (`Part 1 > Chapter 3 > Section 2`).
//! - 10.2 Table: one row per chunk, header plus an N-token context window
//!   before/after the row.
//! - 10.3 Presentation: one slide per chunk.
//! - 10.4 Parent/child: small child chunks for retrieval, each linked to a
//!   larger parent chunk that supplies full context on a hit.
//! - 10.5 The method is chosen per document (`set_chunk_method` in the
//!   store) and otherwise auto-detected by shape (`.md` -> Hierarchy,
//!   tables -> Table, presentations -> Presentation, else General).
use crate::text::{self, Chunk, ResourceLimit};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Chunking methods (10.5). Only Hierarchy + Table + Presentation are new
/// in this stage; the rest of the plan's list (Q&A, Manual, Paper, Book,
/// Laws, Picture, One, Audio, Email) is a future subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChunkMethod {
    General,
    Hierarchy,
    Table,
    Presentation,
    ParentChild,
}

impl Default for ChunkMethod {
    fn default() -> Self {
        ChunkMethod::General
    }
}

impl ChunkMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChunkMethod::General => "general",
            ChunkMethod::Hierarchy => "hierarchy",
            ChunkMethod::Table => "table",
            ChunkMethod::Presentation => "presentation",
            ChunkMethod::ParentChild => "parent-child",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "general" => Some(ChunkMethod::General),
            "hierarchy" => Some(ChunkMethod::Hierarchy),
            "table" => Some(ChunkMethod::Table),
            "presentation" => Some(ChunkMethod::Presentation),
            "parent-child" => Some(ChunkMethod::ParentChild),
            _ => None,
        }
    }
}

/// Auto-detect the chunking method by file shape (10.5, the default since
/// 2026-11-14): `.md`/`.markdown` -> Hierarchy, table files
/// (`.csv`/`.tsv`/`.xls`/`.xlsx`/`.ods`) -> Table, presentations
/// (`.ppt`/`.pptx`/`.key`/`.odp`) -> Presentation, everything else ->
/// General. A per-document override (`set_chunk_method`) always wins.
pub fn method_for_path(path: &Path) -> ChunkMethod {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "csv" | "tsv" | "xls" | "xlsx" | "ods" => ChunkMethod::Table,
        "ppt" | "pptx" | "key" | "odp" => ChunkMethod::Presentation,
        "md" | "markdown" => ChunkMethod::Hierarchy,
        _ => ChunkMethod::General,
    }
}

/// Same structural guard as the General chunker (131_072 lines/sentences).
const MAX_STRUCTURE_ITEMS: usize = 131_072;

fn check_lines(content: &str) -> Result<(), ResourceLimit> {
    if content.matches('\n').count() > MAX_STRUCTURE_ITEMS {
        return Err(ResourceLimit("换行数量不得超过 131072".into()));
    }
    Ok(())
}

/// Indexable fragments of a byte range: intersect [start, end) with the
/// document's `text::indexable_ranges` (embedded `data:` payloads and
/// `<svg>` blocks are excluded for .md/.html, exactly like the General
/// chunker). Other extensions return the range unchanged.
fn indexable_fragments(content: &str, path: &Path, start: usize, end: usize) -> Vec<(usize, usize)> {
    text::indexable_ranges(content, path)
        .into_iter()
        .filter_map(|(s, e)| {
            let s = s.max(start);
            let e = e.min(end);
            if s < e {
                Some((s, e))
            } else {
                None
            }
        })
        .collect()
}

/// The indexable text of one line: its fragments joined with nothing (they
/// are pieces of the same line).
fn indexable_line_text(content: &str, path: &Path, s: usize, e: usize) -> String {
    indexable_fragments(content, path, s, e)
        .iter()
        .map(|&(s, e)| &content[s..e])
        .collect()
}

fn chunk_from_range(content: &str, lines: &text::Lines, heading: String, start: usize, end: usize, path: &Path) -> Chunk {
    // Build only over indexable fragments, so an embedded resource payload
    // (data: URI, <svg> block) never lands in the chunk content, its spans
    // or its line ranges — hydration re-slices the source by span, so a raw
    // slice here would re-expose the payload on retrieval.
    let mut text = String::new();
    let mut spans = vec![];
    let mut ranges = vec![];
    for (s, e) in indexable_fragments(content, path, start, end) {
        let span = lines.span(content, s, e);
        let last_line = if e > 0 && content.as_bytes()[e - 1] == b'\n' {
            span.end_line.saturating_sub(1).max(span.start_line)
        } else {
            span.end_line
        };
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&content[s..e]);
        ranges.push((span.start_line, last_line));
        spans.push(span);
    }
    Chunk {
        heading,
        content: text,
        ranges,
        spans,
    }
}

/// Split a byte range into chunks of at most `max_chars` at line boundaries.
/// `line_starts` holds the byte offset where each line of the range starts.
fn split_lines(content: &str, lines: &text::Lines, heading: String, start: usize, end: usize, max_chars: usize, path: &Path) -> Vec<Chunk> {
    let mut out = vec![];
    let mut cursor = start;
    let mut length = 0usize;
    // Line starts inside [start, end): reuse the global line spans.
    for &(s, e) in lines.spans.iter() {
        if e <= start || s >= end {
            continue;
        }
        let line_len = content[s..e.min(end)].chars().count() + 1; // + newline
        if length > 0 && length + line_len > max_chars {
            let chunk = chunk_from_range(content, lines, heading.clone(), cursor, s, path);
            if !chunk.content.trim().is_empty() {
                out.push(chunk);
            }
            cursor = s;
            length = 0;
        }
        length += line_len;
    }
    if cursor < end {
        let chunk = chunk_from_range(content, lines, heading, cursor, end, path);
        if !chunk.content.trim().is_empty() {
            out.push(chunk);
        }
    }
    out
}

/// Parse a heading line (10.1). Levels 1-5:
/// - Markdown `#`..`#####` (deeper `#` lines cap at 5).
/// - Chinese numbered: `第X章`/`第X节`/`第X条`/`第X款`/`第X部分` (X = Chinese
///   numeral or digits), mapped to levels 2/3/4/4/1.
/// - `PART N` (1), `Chapter N` (2), `Section N` (3), `Article N` (4).
pub fn parse_heading(line: &str) -> Option<(u8, String)> {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix('#') {
        let hashes = 1 + rest.chars().take_while(|c| *c == '#').count();
        if hashes > 6 {
            return None;
        }
        let title = rest.trim_start_matches('#').trim().trim_end_matches('#').trim();
        if title.is_empty() {
            return None;
        }
        return Some((hashes.min(5) as u8, title.to_string()));
    }
    let upper = trimmed.to_ascii_uppercase();
    for (prefix, level) in [("PART ", 1u8), ("CHAPTER ", 2), ("SECTION ", 3), ("ARTICLE ", 4)] {
        if let Some(rest) = upper.strip_prefix(prefix) {
            let rest = rest.trim();
            if !rest.is_empty() && rest.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                return Some((level, trimmed.to_string()));
            }
        }
    }
    // Chinese numbered headings: 第X章/节/条/款/部分. The character after the
    // marker must be a separator (space, punctuation) or end of line, so a
    // body sentence starting with 第一章... is not mistaken for a heading.
    if let Some(rest) = trimmed.strip_prefix('第') {
        let mut numeral = String::new();
        for c in rest.chars() {
            if "一二三四五六七八九十百千万零0123456789".contains(c) {
                numeral.push(c);
            } else {
                break;
            }
        }
        if numeral.is_empty() {
            return None;
        }
        let after = &rest[numeral.len()..];
        let (level, marker_len) = if after.starts_with("部分") {
            (1u8, "部分".len())
        } else {
            match after.chars().next() {
                Some('章') => (2, '章'.len_utf8()),
                Some('节') => (3, '节'.len_utf8()),
                Some('条') | Some('款') => (4, '条'.len_utf8()),
                _ => return None,
            }
        };
        let tail = &after[marker_len..];
        if tail.is_empty() || tail.starts_with(|c: char| c.is_whitespace() || "：:、.．。-—".contains(c)) {
            return Some((level, trimmed.to_string()));
        }
        return None;
    }
    None
}

/// Hierarchy chunking (10.1): walk headings, keep the full parent path
/// (`Part 1 > Chapter 3 > Section 2`) on every chunk. Text before the first
/// heading becomes the first chunk (global context) when `first_chunk_global`
/// is set; its heading is the document's first H1 title, or empty.
pub fn hierarchy_chunks(content: &str, max_chars: usize, first_chunk_global: bool, path: &Path) -> Vec<Chunk> {
    let lines = text::Lines::new(content);
    let mut stack: Vec<(u8, String)> = vec![];
    let mut sections: Vec<(String, usize, usize)> = vec![]; // (heading path, start byte, end byte)
    let mut section_start: Option<usize> = None;
    let mut first_h1: Option<String> = None;
    let mut pre_start: Option<usize> = None;
    for (i, &(s, e)) in lines.spans.iter().enumerate() {
        let line = &content[s..e];
        if let Some((level, title)) = parse_heading(line) {
            if level == 1 && first_h1.is_none() {
                first_h1 = Some(title.clone());
            }
            if let Some(start) = section_start.take() {
                let end = s;
                if end > start {
                    sections.push((path_of(&stack), start, end));
                }
            } else if first_chunk_global {
                if let Some(ps) = pre_start {
                    if s > ps {
                        sections.push((first_h1.clone().unwrap_or_default(), ps, s));
                    }
                }
            }
            while stack.last().is_some_and(|(l, _)| *l >= level) {
                stack.pop();
            }
            stack.push((level, title));
            section_start = Some(e + if content.as_bytes().get(e) == Some(&b'\n') { 1 } else { 0 });
            let _ = i;
        } else if section_start.is_none() && first_chunk_global && !line.trim().is_empty() {
            // Pre-heading text starts at the first non-empty line.
            if pre_start.is_none() {
                pre_start = Some(s);
            }
        }
    }
    if let Some(start) = section_start {
        if content.len() > start {
            sections.push((path_of(&stack), start, content.len()));
        }
    } else if first_chunk_global {
        if let Some(ps) = pre_start {
            if content.len() > ps && !content[ps..].trim().is_empty() {
                sections.push((first_h1.unwrap_or_default(), ps, content.len()));
            }
        }
    }
    let mut out = vec![];
    for (heading, start, end) in sections {
        if content[start..end].trim().is_empty() {
            continue;
        }
        out.extend(split_lines(content, &lines, heading, start, end, max_chars, path));
    }
    out
}

fn path_of(stack: &[(u8, String)]) -> String {
    stack.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>().join(" > ")
}

/// Count tokens: one whitespace-separated word, or one CJK character, is a token.
fn token_count(s: &str) -> usize {
    let mut count = 0;
    let mut in_word = false;
    for c in s.chars() {
        if c.is_whitespace() {
            in_word = false;
        } else if c.is_ascii_alphanumeric() || c == '_' {
            if !in_word {
                count += 1;
                in_word = true;
            }
        } else {
            // CJK / punctuation: one token each.
            count += 1;
            in_word = false;
        }
    }
    count
}

/// Keep at most `n` tokens from the end of `s` (char-boundary safe).
fn take_last_tokens(s: &str, n: usize) -> &str {
    if token_count(s) <= n {
        return s;
    }
    let mut kept = 0;
    let mut cut = s.len();
    let mut in_word = false;
    for (i, c) in s.char_indices().rev() {
        if c.is_whitespace() {
            in_word = false;
        } else if c.is_ascii_alphanumeric() || c == '_' {
            if !in_word {
                kept += 1;
                in_word = true;
                if kept > n {
                    cut = i + c.len_utf8();
                    break;
                }
            }
        } else {
            kept += 1;
            in_word = false;
            if kept > n {
                cut = i + c.len_utf8();
                break;
            }
        }
    }
    &s[cut..]
}

/// Keep at most `n` tokens from the start of `s` (char-boundary safe).
fn take_first_tokens(s: &str, n: usize) -> &str {
    if token_count(s) <= n {
        return s;
    }
    let mut kept = 0;
    let mut cut = s.len();
    let mut in_word = false;
    for (i, c) in s.char_indices() {
        if c.is_whitespace() {
            in_word = false;
        } else if c.is_ascii_alphanumeric() || c == '_' {
            if !in_word {
                kept += 1;
                in_word = true;
                if kept > n {
                    cut = i;
                    break;
                }
            }
        } else {
            kept += 1;
            in_word = false;
            if kept > n {
                cut = i;
                break;
            }
        }
    }
    &s[..cut]
}

/// Table chunking (10.2): the first non-empty line is the header; each data
/// row is one chunk (long rows split at `max_chars`). Every chunk carries the
/// header plus an N-token context window from the previous and next rows, so
/// a buried number keeps its meaning. A header with no data rows still
/// yields one chunk (the header line itself) instead of dropping the
/// document — e.g. a converted file the user edited down to one line.
pub fn table_chunks(content: &str, max_chars: usize, context_tokens: usize, path: &Path) -> Vec<Chunk> {
    let lines = text::Lines::new(content);
    let rows: Vec<(usize, usize)> = lines
        .spans
        .iter()
        .copied()
        .filter(|&(s, e)| !content[s..e].trim().is_empty())
        .collect();
    if rows.is_empty() {
        return vec![];
    }
    let (hs, he) = rows[0];
    let header = indexable_line_text(content, path, hs, he).trim().to_string();
    if rows.len() == 1 {
        // Header only: keep the document searchable instead of emitting
        // zero chunks.
        let chunk = chunk_from_range(content, &lines, header, hs, he, path);
        return if chunk.content.trim().is_empty() { vec![] } else { vec![chunk] };
    }
    let mut out = vec![];
    for (idx, &(s, e)) in rows.iter().enumerate().skip(1) {
        let row = indexable_line_text(content, path, s, e);
        let before_text = if idx > 1 {
            let (ps, pe) = rows[idx - 1];
            indexable_line_text(content, path, ps, pe)
        } else {
            String::new()
        };
        let before = take_last_tokens(before_text.trim(), context_tokens);
        let after_text = if idx + 1 < rows.len() {
            let (ns, ne) = rows[idx + 1];
            indexable_line_text(content, path, ns, ne)
        } else {
            String::new()
        };
        let after = take_first_tokens(after_text.trim(), context_tokens);
        let mut body = String::new();
        body.push_str(&header);
        if !before.is_empty() {
            body.push('\n');
            body.push_str(before);
        }
        body.push('\n');
        body.push_str(row.trim());
        if !after.is_empty() {
            body.push('\n');
            body.push_str(after);
        }
        // Long rows: split the row text itself; each part keeps the header.
        if body.chars().count() > max_chars {
            let width = max_chars.saturating_sub(header.chars().count() + 2).max(1);
            let chars: Vec<char> = row.trim().chars().collect();
            for part_chars in chars.chunks(width) {
                let part: String = part_chars.iter().collect();
                let mut part_body = String::new();
                part_body.push_str(&header);
                part_body.push('\n');
                part_body.push_str(&part);
                let mut chunk = chunk_from_range(content, &lines, header.clone(), s, e, path);
                chunk.content = part_body;
                out.push(chunk);
            }
        } else {
            let mut chunk = chunk_from_range(content, &lines, header.clone(), s, e, path);
            chunk.content = body;
            out.push(chunk);
        }
    }
    out
}

/// Slide delimiter (10.3): form feed, Markdown horizontal rule (not at the
/// very start), or a slide marker line (`幻灯片 N` / `Slide N` / `第 N 页`).
fn is_slide_break(line: &str, is_first_line: bool) -> bool {
    let t = line.trim();
    if t.contains('\x0c') {
        return true;
    }
    if !is_first_line && matches!(t, "---" | "***" | "___") {
        return true;
    }
    let upper = t.to_ascii_uppercase();
    if upper.starts_with("SLIDE") || t.starts_with("幻灯片") || t.starts_with("投影片") {
        return true;
    }
    if t.starts_with('第') {
        let rest = &t[3..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() && rest[digits.len()..].starts_with(['页', '張', '张']) {
            return true;
        }
    }
    false
}

/// Presentation chunking (10.3): one slide per chunk; the slide heading is
/// its first non-empty line. Over-long slides split at line boundaries.
pub fn slide_chunks(content: &str, max_chars: usize, path: &Path) -> Vec<Chunk> {
    let lines = text::Lines::new(content);
    let mut slides: Vec<(usize, usize)> = vec![];
    let mut start: Option<usize> = None;
    for (i, &(s, e)) in lines.spans.iter().enumerate() {
        let line = &content[s..e];
        if is_slide_break(line, i == 0) {
            if let Some(st) = start.take() {
                if e > st {
                    slides.push((st, s));
                }
            }
            start = Some(e + if content.as_bytes().get(e) == Some(&b'\n') { 1 } else { 0 });
        } else if start.is_none() && !line.trim().is_empty() {
            start = Some(s);
        }
    }
    if let Some(st) = start {
        if content.len() > st {
            slides.push((st, content.len()));
        }
    }
    let mut out = vec![];
    for (s, e) in slides {
        if content[s..e].trim().is_empty() {
            continue;
        }
        let title = content[s..e]
            .lines()
            .find(|l| !l.trim().is_empty())
            .map(|l| {
                let t: String = l.trim().chars().take(80).collect();
                t
            })
            .unwrap_or_default();
        out.extend(split_lines(content, &lines, title, s, e, max_chars, path));
    }
    out
}

/// Parent/child chunking (10.4). Parents are line-grouped to at most
/// `parent_chars`; children subdivide each parent to at most `child_chars`.
/// Retrieval uses children; a hit expands to its parent for full context.
#[derive(Clone, Debug)]
pub struct ParentChildPair {
    pub child: Chunk,
    pub parent: Chunk,
}

pub fn parent_child_chunks(content: &str, child_chars: usize, parent_chars: usize, path: &Path) -> Vec<ParentChildPair> {
    let lines = text::Lines::new(content);
    let mut pairs = vec![];
    let mut p_start: Option<usize> = None;
    let mut p_len = 0usize;
    let flush_parent = |pairs: &mut Vec<ParentChildPair>, lines: &text::Lines, p_start: usize, p_end: usize| {
        let parent = chunk_from_range(content, lines, String::new(), p_start, p_end, path);
        // Children: line-group inside the parent.
        let mut c_start = p_start;
        let mut c_len = 0usize;
        for &(s, e) in lines.spans.iter() {
            if e <= p_start || s >= p_end {
                continue;
            }
            let line_len = content[s..e.min(p_end)].chars().count() + 1;
            if c_len > 0 && c_len + line_len > child_chars {
                let child = chunk_from_range(content, lines, String::new(), c_start, s, path);
                if !child.content.trim().is_empty() {
                    pairs.push(ParentChildPair { child, parent: parent.clone() });
                }
                c_start = s;
                c_len = 0;
            }
            c_len += line_len;
        }
        if c_start < p_end {
            let child = chunk_from_range(content, lines, String::new(), c_start, p_end, path);
            if !child.content.trim().is_empty() {
                pairs.push(ParentChildPair { child, parent });
            }
        }
    };
    for &(s, e) in lines.spans.iter() {
        let line = &content[s..e];
        if p_start.is_none() {
            if line.trim().is_empty() {
                continue;
            }
            p_start = Some(s);
            p_len = 0;
        }
        let line_len = line.chars().count() + 1;
        if p_len > 0 && p_len + line_len > parent_chars {
            let st = p_start.take().unwrap();
            flush_parent(&mut pairs, &lines, st, s);
            p_start = Some(s);
            p_len = 0;
        }
        p_len += line_len;
    }
    if let Some(st) = p_start {
        flush_parent(&mut pairs, &lines, st, content.len());
    }
    pairs
}

/// Expand a retrieved child chunk to its parent (10.4): the pair whose
/// parent byte range contains the child's first span.
pub fn expand_child_to_parent<'a>(child: &Chunk, pairs: &'a [ParentChildPair]) -> Option<&'a Chunk> {
    let start = child.spans.first()?.start_byte;
    pairs
        .iter()
        .find(|p| p.child.spans.first().is_some_and(|c| c.start_byte == start))
        .map(|p| &p.parent)
}

/// Dispatcher (10.5): General delegates to the sentence chunker; the new
/// methods use the shape builders. ParentChild returns the child chunks.
pub fn chunk_document(content: &str, max_chars: usize, path: &Path, method: ChunkMethod) -> Vec<Chunk> {
    match method {
        ChunkMethod::General => text::chunks_for_file(content, max_chars, path),
        ChunkMethod::Hierarchy => hierarchy_chunks(content, max_chars, true, path),
        ChunkMethod::Table => table_chunks(content, max_chars, 50, path),
        ChunkMethod::Presentation => slide_chunks(content, max_chars, path),
        ChunkMethod::ParentChild => parent_child_chunks(content, max_chars / 2, max_chars, path)
            .into_iter()
            .map(|p| p.child)
            .collect(),
    }
}

/// Build chunks from explicit line boundaries (14.1): an override saved by
/// `set_chunk_boundaries`. Ranges are 1-based line numbers; a boundary
/// outside the document is dropped.
pub fn chunks_from_boundaries(content: &str, boundaries: &[crate::chunk_admin::ChunkBoundary], path: &Path) -> Vec<Chunk> {
    let lines = text::Lines::new(content);
    let total = lines.spans.len();
    let mut out = vec![];
    for boundary in boundaries {
        if boundary.start_line == 0 || boundary.start_line > total || boundary.end_line < boundary.start_line {
            continue;
        }
        let end_line = boundary.end_line.min(total);
        let start = lines.spans[boundary.start_line - 1].0;
        let end = lines.spans[end_line - 1].1;
        if end <= start {
            continue;
        }
        let chunk = chunk_from_range(content, &lines, boundary.heading.clone(), start, end, path);
        if !chunk.content.trim().is_empty() {
            out.push(chunk);
        }
    }
    out
}

/// Index-path entry (10.5): like `text::chunks_for_index` but method-aware.
/// The General path keeps the resource guard; the shape builders guard on
/// line count instead.
pub(crate) fn chunks_for_index_with_method(
    content: &str,
    max_chars: usize,
    path: &Path,
    method: ChunkMethod,
    byte_limit: usize,
) -> Result<Vec<Chunk>, ResourceLimit> {
    match method {
        ChunkMethod::General => text::chunks_for_index(content, max_chars, path, byte_limit),
        _ => {
            check_lines(content)?;
            Ok(chunk_document(content, max_chars, path, method))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "引言\n这是开头。\n\n# Part 1 总论\n总论内容。\n\n## Chapter 1 第一章\n第一章内容。\n\n### Section 1.1 背景\n背景内容在这里。\n\n## Chapter 2 第二章\n第二章内容。\n";

    #[test]
    fn parse_heading_recognises_markdown_chinese_and_english() {
        assert_eq!(parse_heading("# 总论"), Some((1, "总论".into())));
        assert_eq!(parse_heading("### 背景"), Some((3, "背景".into())));
        assert_eq!(parse_heading("第一章 总则"), Some((2, "第一章 总则".into())));
        assert_eq!(parse_heading("第一节 适用范围"), Some((3, "第一节 适用范围".into())));
        assert_eq!(parse_heading("第3条 定义"), Some((4, "第3条 定义".into())));
        assert_eq!(parse_heading("Part 1 总论"), Some((1, "Part 1 总论".into())));
        assert_eq!(parse_heading("Chapter 2 Methods"), Some((2, "Chapter 2 Methods".into())));
        assert_eq!(parse_heading("正文内容"), None);
        assert_eq!(parse_heading(""), None);
        // Body sentences starting with 第一章... are not headings.
        assert_eq!(parse_heading("第一章内容。"), None);
        assert_eq!(parse_heading("第二章内容。"), None);
    }

    #[test]
    fn hierarchy_chunks_carry_full_parent_paths() {
        let chunks = hierarchy_chunks(DOC, 512, true, Path::new("a.md"));
        assert!(!chunks.is_empty());
        // First chunk is the pre-heading global context.
        assert_eq!(chunks[0].content.trim(), "引言\n这是开头。");
        assert_eq!(chunks[0].heading, "Part 1 总论");
        let paths: Vec<&str> = chunks.iter().map(|c| c.heading.as_str()).collect();
        assert!(paths.iter().any(|h| *h == "Part 1 总论 > Chapter 1 第一章 > Section 1.1 背景"), "paths: {paths:?}");
        assert!(paths.iter().any(|h| *h == "Part 1 总论 > Chapter 2 第二章"), "paths: {paths:?}");
        // Section content stays with its own path.
        let bg = chunks.iter().find(|c| c.heading.ends_with("Section 1.1 背景")).unwrap();
        assert!(bg.content.contains("背景内容在这里"));
    }

    #[test]
    fn table_chunks_keep_header_and_context_windows() {
        let csv = "型号,价格,库存\nBCD-215,1999,12\nBCD-300,2999,5\nBCD-400,3999,0\n";
        let chunks = table_chunks(csv, 512, 50, Path::new("a.csv"));
        assert_eq!(chunks.len(), 3, "one chunk per data row");
        assert!(chunks.iter().all(|c| c.heading == "型号,价格,库存"));
        assert!(chunks[0].content.contains("BCD-215,1999,12"));
        assert!(chunks[0].content.contains("型号,价格,库存"), "header must ride along");
        assert!(chunks[0].content.contains("BCD-300,2999,5"), "next row is the after-window");
        assert!(chunks[1].content.contains("BCD-215,1999,12"), "previous row is the before-window");
    }

    #[test]
    fn slide_chunks_split_per_slide() {
        let deck = "封面\n产品介绍\n---\n第二页\n功能列表\n---\n第三页\n价格\n";
        let chunks = slide_chunks(deck, 512, Path::new("a.md"));
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].heading, "封面");
        assert!(chunks[0].content.contains("产品介绍"));
        assert_eq!(chunks[1].heading, "第二页");
        assert!(chunks[2].content.contains("价格"));
    }

    #[test]
    fn parent_child_chunks_expand_hits_to_full_context() {
        let content = (0..20).map(|i| format!("第{i}行内容。")).collect::<Vec<_>>().join("\n");
        let pairs = parent_child_chunks(&content, 40, 120, Path::new("a.txt"));
        assert!(pairs.len() >= 4, "children: {}", pairs.len());
        assert!(pairs.iter().all(|p| p.child.content.chars().count() <= 44));
        assert!(pairs.iter().all(|p| p.parent.content.chars().count() > p.child.content.chars().count() || p.parent.content == p.child.content));
        // Every child expands to a parent that contains the child's text.
        for pair in &pairs {
            let found = expand_child_to_parent(&pair.child, &pairs).unwrap();
            assert!(found.content.contains(pair.child.content.trim()));
        }
    }

    #[test]
    fn method_for_path_suggests_by_shape() {
        assert_eq!(method_for_path(Path::new("a.csv")), ChunkMethod::Table);
        assert_eq!(method_for_path(Path::new("a.pptx")), ChunkMethod::Presentation);
        assert_eq!(method_for_path(Path::new("a.md")), ChunkMethod::Hierarchy);
        assert_eq!(method_for_path(Path::new("a.txt")), ChunkMethod::General);
        assert_eq!(method_for_path(Path::new("a.png")), ChunkMethod::General);
    }

    #[test]
    fn table_chunks_keep_header_only_document_searchable() {
        // A converted file the user edited down to one line must not vanish.
        let chunks = table_chunks("usereditedcacheevidence", 512, 50, Path::new("a.md"));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("usereditedcacheevidence"));
    }

    #[test]
    fn hierarchy_chunks_exclude_embedded_resource_payloads() {
        // Same rule as the General chunker: data: URIs and <svg> blocks are
        // never indexed, in content, spans or line ranges.
        let doc = "正文marker\n<svg><text>svgpayloadmarker</text></svg>\n![图片说明](data:image/png;base64,imagepayloadmarker)\n[附件](data:application/pdf;base64,attachmentpayloadmarker)\n";
        let chunks = hierarchy_chunks(doc, 512, true, Path::new("a.md"));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("正文marker"));
        assert!(!chunks[0].content.contains("payloadmarker"));
        assert!(chunks[0].spans.iter().all(|s| !doc[s.start_byte..s.end_byte].contains("payloadmarker")));
    }

    #[test]
    fn chunk_document_dispatches_and_parentchild_returns_children() {
        let csv = "型号,价格\nBCD-215,1999\n";
        let table = chunk_document(csv, 512, Path::new("a.csv"), ChunkMethod::Table);
        assert_eq!(table.len(), 1);
        let general = chunk_document(csv, 512, Path::new("a.csv"), ChunkMethod::General);
        assert!(!general.is_empty());
        let pc = chunk_document(DOC, 512, Path::new("a.md"), ChunkMethod::ParentChild);
        assert!(!pc.is_empty());
    }
}
