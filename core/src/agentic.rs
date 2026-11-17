//! Multi-round agentic retrieval, post-hoc citation check, enrichment
//! (Stage 3, 15.1-15.3).
//!
//! - 15.1 Multi-round tiers: fast = single-round SAG2; accurate = multi-round
//!   + graph + evidence verification, with a decision layer that stops the
//!   loop (breaker: max rounds; judge: evidence sufficient; filter: no new
//!   evidence). The loop never runs unbounded.
//! - 15.2 Post-hoc citation check: after generation, every `[N]` in the
//!   answer is mechanically checked against the real chunk ids (via
//!   `models::check_citations`); bad citations are marked `[N?]` in the
//!   returned text and listed in the report.
//! - 15.3 Optional enrichment (default off): an LLM tags each chunk and
//!   generates keywords and questions for it, widening recall. Tags are
//!   stored per chunk and consulted by `search_with_enrichment`; with
//!   enrichment off, search behaviour is unchanged.
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::models::{check_citations, CitationStatus};
use crate::store::Store;
use crate::types::Candidate;

// ---------------------------------------------------------------- 15.1

/// Multi-round retrieval config (15.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiRoundConfig {
    /// Hard stop: never run more rounds than this (breaker).
    pub max_rounds: usize,
    /// Evidence is sufficient once this many distinct chunks are held (judge).
    pub min_evidence: usize,
}

impl Default for MultiRoundConfig {
    fn default() -> Self {
        MultiRoundConfig { max_rounds: 3, min_evidence: 5 }
    }
}

/// Why the multi-round loop stopped (15.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Round budget exhausted (decision_breaker).
    MaxRounds,
    /// Enough evidence to answer (decision_judge).
    Sufficient,
    /// The latest round added no new evidence (decision_filter).
    NoNewEvidence,
}

/// Decision layer (15.1): returns the stop reason, or `None` to continue.
/// Checked in order: breaker, judge, filter.
pub fn decide_stop(round: usize, config: &MultiRoundConfig, new_evidence: usize, total_evidence: usize) -> Option<StopReason> {
    if round >= config.max_rounds {
        return Some(StopReason::MaxRounds);
    }
    if total_evidence >= config.min_evidence {
        return Some(StopReason::Sufficient);
    }
    if round > 0 && new_evidence == 0 {
        return Some(StopReason::NoNewEvidence);
    }
    None
}

// ---------------------------------------------------------------- 15.2

/// One citation check in the post-hoc report (15.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CitationReportEntry {
    /// The citation as written, e.g. `[3]`.
    pub citation: String,
    /// Whether it resolves to a real chunk.
    pub valid: bool,
    /// Real chunk id when valid.
    pub chunk_id: Option<i64>,
}

/// Post-hoc citation report (15.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CitationReport {
    pub entries: Vec<CitationReportEntry>,
    /// Answer with bad citations marked as `[N?]`.
    pub marked_answer: String,
    pub bad_count: usize,
}

/// Check every citation in a generated answer against the evidence (15.2).
/// Valid citations pass through; out-of-range or malformed ones are marked
/// `[N?]` in `marked_answer` and listed as invalid in `entries`.
pub fn verify_citations(answer: &str, evidence: &[Candidate]) -> CitationReport {
    let checks = check_citations(answer, evidence);
    let mut entries = vec![];
    let mut marked = String::new();
    let mut cursor = 0;
    for check in &checks {
        marked.push_str(&answer[cursor..check.range.start]);
        let written = &answer[check.range.clone()];
        match check.status {
            CitationStatus::Valid { source_number: _, chunk_id } => {
                entries.push(CitationReportEntry { citation: written.to_string(), valid: true, chunk_id: Some(chunk_id) });
                marked.push_str(written);
            }
            CitationStatus::OutOfRange { .. } | CitationStatus::Malformed => {
                entries.push(CitationReportEntry { citation: written.to_string(), valid: false, chunk_id: None });
                if written.ends_with(']') {
                    marked.push_str(&format!("{}?]", &written[..written.len() - 1]));
                } else {
                    marked.push_str(&format!("{written}?"));
                }
            }
        }
        cursor = check.range.end;
    }
    marked.push_str(&answer[cursor..]);
    let bad_count = entries.iter().filter(|e| !e.valid).count();
    CitationReport { entries, marked_answer: marked, bad_count }
}

// ---------------------------------------------------------------- 15.3

/// Enrichment config (15.3). Off by default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnrichmentConfig {
    pub enabled: bool,
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        EnrichmentConfig { enabled: false }
    }
}

/// LLM enrichment for one chunk (15.3).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkEnrichment {
    pub chunk_id: i64,
    pub tags: Vec<String>,
    pub keywords: Vec<String>,
    pub questions: Vec<String>,
}

/// Enrichment transport (15.3): implemented by the MiniMax caller; tests
/// use a stub. Enrichment is explicitly LLM-backed and slow, so it is
/// off by default and only runs on demand.
pub trait EnrichmentTransport {
    fn enrich(&self, chunk_id: i64, content: &str) -> Result<ChunkEnrichment>;
}

/// Search that also consults stored enrichment (15.3). Plain BM25 results
/// come first; chunks whose tags/keywords/questions match the query are
/// appended (deduplicated). With no enrichment stored, this is identical
/// to `store.search_files`.
pub fn search_with_enrichment(
    store: &Store,
    query: &str,
    scope: &[String],
    limit: usize,
) -> Result<Vec<Candidate>> {
    let mut results = store.search_files(query, scope, limit)?;
    if results.len() >= limit {
        return Ok(results);
    }
    let extra_ids = store.chunks_matching_enrichment(query, scope)?;
    let seen: std::collections::HashSet<i64> = results.iter().map(|c| c.id).collect();
    for id in extra_ids {
        if results.len() >= limit || seen.contains(&id) {
            continue;
        }
        if let Some(candidate) = store.candidate_by_id(id, scope, &[])? {
            results.push(candidate);
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: i64, content: &str) -> Candidate {
        Candidate {
            id,
            document_id: id,
            path: format!("/kb/doc{id}.md"),
            title: format!("doc{id}"),
            heading: String::new(),
            content: content.into(),
            start_line: 1,
            end_line: 1,
            source_ranges: vec![(1, 1)],
            source_spans: vec![],
            bm25: 0.0,
            score: 1.0,
            previous_score: None,
            order: 0,
            hash: "h".into(),
        }
    }

    #[test]
    fn decision_layer_stops_on_breaker_judge_and_filter() {
        let config = MultiRoundConfig { max_rounds: 3, min_evidence: 5 };
        assert_eq!(decide_stop(3, &config, 2, 2), Some(StopReason::MaxRounds));
        assert_eq!(decide_stop(1, &config, 2, 5), Some(StopReason::Sufficient));
        assert_eq!(decide_stop(1, &config, 0, 2), Some(StopReason::NoNewEvidence));
        assert_eq!(decide_stop(1, &config, 2, 2), None);
        // Round 0 with no new evidence does not stop on the filter: the
        // first round always runs.
        assert_eq!(decide_stop(0, &config, 0, 0), None);
    }

    #[test]
    fn citation_check_marks_bad_citations_and_keeps_valid_ones() {
        let evidence = vec![candidate(11, "alpha"), candidate(22, "beta")];
        let answer = "Alpha 在 [1]，Beta 在 [2]，还有 [5] 和 [x]。";
        let report = verify_citations(answer, &evidence);
        assert_eq!(report.bad_count, 2);
        assert_eq!(report.entries.len(), 4);
        assert!(report.entries[0].valid && report.entries[0].chunk_id == Some(11));
        assert!(report.entries[1].valid && report.entries[1].chunk_id == Some(22));
        assert!(!report.entries[2].valid);
        assert!(!report.entries[3].valid);
        assert!(report.marked_answer.contains("[1]") && report.marked_answer.contains("[2]"));
        assert!(report.marked_answer.contains("[5?]"), "marked: {}", report.marked_answer);
        assert!(report.marked_answer.contains("[x?]"), "marked: {}", report.marked_answer);
    }

    #[test]
    fn enrichment_search_widens_recall_without_changing_plain_search() {
        use std::fs;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("kb");
        fs::create_dir(&root).unwrap();
        let store = Store::open(&dir.path().join("app/db.sqlite")).unwrap();
        let root_key = store.add_root(&root).unwrap();
        fs::write(root.join("guide.md"), "# 服务\n\n把服务跑起来。\n").unwrap();
        store.scan(&root_key, |_| {}).unwrap();
        // Plain search cannot find "部署": the word is not in the content.
        let plain = store.search_files("部署", &[], 10).unwrap();
        assert!(plain.is_empty());
        // Enrich the chunk with the keyword, then the enriched search finds it.
        let chunk_id = store.search_files("跑起来", &[], 10).unwrap()[0].id;
        store
            .save_chunk_enrichment(&ChunkEnrichment {
                chunk_id,
                tags: vec!["部署".into()],
                keywords: vec!["部署".into()],
                questions: vec!["如何部署服务？".into()],
            })
            .unwrap();
        let enriched = search_with_enrichment(&store, "部署", &[], 10).unwrap();
        assert_eq!(enriched.len(), 1);
        assert_eq!(enriched[0].id, chunk_id);
        // With no enrichment stored, enriched search equals plain search.
        let plain2 = search_with_enrichment(&store, "不存在的词", &[], 10).unwrap();
        assert!(plain2.is_empty());
    }
}
