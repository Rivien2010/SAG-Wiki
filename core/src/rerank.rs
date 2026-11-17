//! Optional reranking stage (Stage 3, 11.1-11.2).
//!
//! Reranking runs after multi-leg fusion, on the ranked candidate pool
//! (64-256, default 128), and reorders it by a cross-encoder score before
//! the final top-k cut. It is OFF by default and must stay off unless a
//! reranker model is configured: the stage is wired, the model is not
//! bundled.
//!
//! Measured result on the reference corpus (Python, bge-reranker-v2-m3,
//! bf16 CPU): full-pool reranking of 128 blocks scored 3.0/5 and reranking
//! the three-leg top-50 union scored 2.5/5, both BELOW quota fusion A1
//! (4.0/5) — reranking cannot rescue evidence outside the pool and, without
//! an author quota, pushes minority-author evidence out of the top 10. It is
//! also slow on CPU (a 128-block pool took ~199s in the Python run). That is
//! why this stage defaults to off. Enabling it without a configured scorer
//! fails loudly instead of silently reordering by something else.
use anyhow::{bail, Result};

use crate::retrieval::SearchOutput;
use crate::stage_config::StageConfig;
use crate::types::Candidate;

/// Pool bounds for the rerank stage (11.1).
pub const RERANK_POOL_MIN: usize = 64;
pub const RERANK_POOL_MAX: usize = 256;
pub const RERANK_POOL_DEFAULT: usize = 128;

/// A cross-encoder scorer: one relevance score per candidate, higher is
/// better. Implementations load their model from a user-provided path; the
/// bge-reranker-v2-m3 ONNX scorer is not bundled with the app.
pub trait Reranker {
    fn score(&self, query: &str, candidates: &[Candidate]) -> Result<Vec<f64>>;
}

/// Clamp a configured pool into [64, 256].
pub fn clamp_pool(pool: usize) -> usize {
    pool.clamp(RERANK_POOL_MIN, RERANK_POOL_MAX)
}

/// Reorder the ranked fusion output by reranker scores (11.1).
/// - `ranked` is the fusion-ranked candidate list (already confidence
///   filtered and document-weighted); only the first `pool` entries are
///   scored, the rest are dropped.
/// - Scores sort descending; ties keep fusion order (stable sort).
/// - `order` is reassigned 0..N after reranking.
pub fn rerank_candidates(
    query: &str,
    ranked: Vec<Candidate>,
    pool: usize,
    reranker: &dyn Reranker,
) -> Result<Vec<Candidate>> {
    let pool = clamp_pool(pool);
    let mut pool_candidates: Vec<Candidate> = ranked.into_iter().take(pool).collect();
    let scores = reranker.score(query, &pool_candidates)?;
    if scores.len() != pool_candidates.len() {
        bail!("重排模型返回的分数数量与候选数量不一致");
    }
    let mut scored: Vec<(f64, usize, Candidate)> = pool_candidates
        .drain(..)
        .enumerate()
        .map(|(i, c)| (scores[i], i, c))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    Ok(scored
        .into_iter()
        .enumerate()
        .map(|(order, (_, _, mut candidate))| {
            candidate.order = order;
            candidate
        })
        .collect())
}

/// Apply the rerank stage to a fused `SearchOutput` (11.1).
/// - Disabled in config: output returned unchanged (Stage 1 behaviour).
/// - Enabled with a scorer: the pool is reranked.
/// - Enabled without a scorer: hard error — the user asked for reranking
///   and no model is configured; never silently keep fusion order.
pub fn apply_rerank(
    query: &str,
    mut output: SearchOutput,
    config: &StageConfig,
    reranker: Option<&dyn Reranker>,
) -> Result<SearchOutput> {
    if !config.rerank_enabled {
        return Ok(output);
    }
    let Some(reranker) = reranker else {
        bail!("重排已开启，但未配置重排模型；请先配置 bge-reranker-v2-m3 模型路径，或关闭重排")
    };
    output.results = rerank_candidates(query, std::mem::take(&mut output.results), config.rerank_candidate_pool, reranker)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Retrieval;

    /// Deterministic term-overlap scorer, test-only: proves the stage
    /// mechanics (pool truncation, reordering, order reassignment) without a
    /// model. It is NOT a production reranker.
    struct OverlapReranker;

    impl Reranker for OverlapReranker {
        fn score(&self, query: &str, candidates: &[Candidate]) -> Result<Vec<f64>> {
            let terms: Vec<&str> = query.split_whitespace().collect();
            Ok(candidates
                .iter()
                .map(|c| {
                    terms.iter().filter(|t| c.content.contains(**t)).count() as f64
                })
                .collect())
        }
    }

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

    fn config(enabled: bool, pool: usize) -> StageConfig {
        let mut c = StageConfig::from_retrieval(&Retrieval::default());
        c.rerank_enabled = enabled;
        c.rerank_candidate_pool = pool;
        c
    }

    #[test]
    fn rerank_disabled_returns_fusion_order_unchanged() {
        let ranked = vec![candidate(1, "alpha"), candidate(2, "beta"), candidate(3, "gamma")];
        let out = apply_rerank("beta", SearchOutput { results: ranked, trace: vec![] }, &config(false, 128), Some(&OverlapReranker)).unwrap();
        assert_eq!(out.results.iter().map(|c| c.id).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn rerank_enabled_reorders_by_score_and_reassigns_order() {
        let ranked = vec![candidate(1, "alpha"), candidate(2, "beta beta"), candidate(3, "gamma")];
        let out = apply_rerank("beta", SearchOutput { results: ranked, trace: vec![] }, &config(true, 128), Some(&OverlapReranker)).unwrap();
        assert_eq!(out.results.iter().map(|c| c.id).collect::<Vec<_>>(), vec![2, 1, 3]);
        assert_eq!(out.results.iter().map(|c| c.order).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn rerank_enabled_without_scorer_fails_loudly() {
        let ranked = vec![candidate(1, "alpha")];
        let result = apply_rerank("alpha", SearchOutput { results: ranked, trace: vec![] }, &config(true, 128), None);
        let err = match result {
            Ok(_) => panic!("expected a missing-model error"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("未配置重排模型"), "{err}");
    }

    #[test]
    fn rerank_pool_is_clamped_to_64_256() {
        assert_eq!(clamp_pool(0), 64);
        assert_eq!(clamp_pool(100), 100);
        assert_eq!(clamp_pool(9999), 256);
        // Pool truncation: only the first `pool` candidates are scored.
        let ranked: Vec<Candidate> = (0..100).map(|i| candidate(i, "x")).collect();
        let out = rerank_candidates("x", ranked, 64, &OverlapReranker).unwrap();
        assert_eq!(out.len(), 64);
    }
}
