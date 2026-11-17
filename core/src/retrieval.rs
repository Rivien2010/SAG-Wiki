pub use sag_protocol::{Term,Trace};
use crate::{
    store::Store,
    text::{match_query, quote, tokenize},
    types::{Candidate, Retrieval},
};
use anyhow::{bail, Result};
use fancy_regex::Regex;
use icu_collator::{Collator, CollatorOptions};
use icu_locid::locale;
use indexmap::IndexMap;
use std::{
    collections::{HashMap, HashSet},
    future::Future,
    sync::LazyLock,
};
use tokio_util::sync::CancellationToken;

static NAMES: LazyLock<Regex> = LazyLock::new(|| {
    let word = r"(?:\p{Lu}\.|\p{Lu}[\p{L}\p{M}]*(?:[-’']\p{Lu}[\p{L}\p{M}]*)*)";
    Regex::new(&format!(
        r"(?<![\p{{L}}\p{{N}}]){word}(?:[ \t]+{word})+(?![\p{{L}}\p{{N}}])"
    ))
    .unwrap()
});
static SENTENCES: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?<!\b[A-Z])[.!?]\s+|[。！？]+\s*|\n+").unwrap());

pub fn name_phrases(s: &str) -> Result<Vec<(usize, usize, String)>> {
    NAMES
        .find_iter(s)
        .map(|r| {
            let m = r?;
            Ok((m.start(), m.end(), tokenize(m.as_str()).join(" ")))
        })
        .collect()
}
fn sentences(text: &str) -> Result<Vec<&str>> {
    let mut out = vec![];
    let mut end = 0;
    for m in SENTENCES.find_iter(text) {
        let m = m?;
        out.push(&text[end..m.start()]);
        end = m.end()
    }
    out.push(&text[end..]);
    Ok(out)
}
pub fn expansion_terms(
    store: &Store,
    query: &str,
    seeds: &[Candidate],
    limit: usize,
) -> Result<Vec<Term>> {
    let query_words = tokenize(query);
    let query_set: HashSet<_> = query_words.iter().cloned().collect();
    let max_score = seeds.iter().map(|s| s.score).fold(0., f64::max);
    let mut passages = vec![];
    let mut vocabulary: HashSet<String> = query_set.clone();
    for seed in seeds {
        let mut parts = vec![];
        for sentence in sentences(&seed.content)? {
            let names = name_phrases(sentence)?;
            let mut ungrouped = String::new();
            let mut end = 0;
            for (start, stop, _) in &names {
                ungrouped.push_str(&sentence[end..*start]);
                ungrouped.push(' ');
                end = *stop;
            }
            ungrouped.push_str(&sentence[end..]);
            let mut words = tokenize(&ungrouped);
            words.extend(names.iter().map(|(_, _, term)| term.clone()));
            vocabulary.extend(words.iter().cloned());
            parts.push((sentence.to_owned(), words, names));
        }
        passages.push((seed, parts));
    }
    let total = store.chunk_count()? as f64;
    let vocabulary: Vec<_> = vocabulary.into_iter().collect();
    let frequencies: HashMap<_, _> = store.frequencies(&vocabulary)?.into_iter()
        .map(|(term, count)| (term, count as f64)).collect();
    let idf = |term: &str| {
        let df = frequencies[term];
        (1. + (total - df + 0.5) / (df + 0.5)).ln()
    };
    let mut ranked: HashMap<String, Term> = HashMap::new();
    for (seed, parts) in passages {
        let weight = 0.5
            + 0.5
                * if max_score > 0. {
                    seed.score / max_score
                } else {
                    0.
                };
        let mut counts: IndexMap<String, usize> = IndexMap::new();
        let mut contexts: HashMap<String, f64> = HashMap::new();
        let mut name_terms = HashSet::new();
        for (sentence, words, names) in &parts {
            let mut unique = HashSet::new();
            let mut context = 0.;
            for word in tokenize(sentence) {
                if unique.insert(word.clone()) && query_set.contains(&word) {
                    context += idf(&word)
                }
            }
            for word in words {
                *counts.entry(word.clone()).or_default() += 1;
                let old = contexts.entry(word.clone()).or_default();
                *old = old.max(context);
            }
            name_terms.extend(names.iter().map(|(_, _, term)| term.clone()));
        }
        for (term, tf) in counts {
            if tokenize(&term).iter().all(|w| query_set.contains(w)) || frequencies[&term] == 0. {
                continue;
            }
            let score = (1. + (tf as f64).ln()) * idf(&term) * (1. + contexts[&term]) * weight;
            if !ranked.contains_key(&term) || score > ranked[&term].score {
                ranked.insert(
                    term.clone(),
                    Term {
                        kind: if name_terms.contains(&term) {
                            "name_phrase"
                        } else {
                            "word"
                        }
                        .into(),
                        term,
                        score,
                    },
                );
            }
        }
    }
    let collator = Collator::try_new(&locale!("en").into(), CollatorOptions::new())
        .map_err(|e| anyhow::anyhow!("词项排序初始化失败：{e:?}"))?;
    let mut terms: Vec<Term> = ranked.into_values().collect();
    terms.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| collator.compare(&a.term, &b.term))
    });
    terms.truncate(limit);
    Ok(terms)
}
pub fn interleave(lists: Vec<Vec<Candidate>>, limit: usize) -> Vec<Candidate> {
    let mut seen = HashSet::new();
    let mut result = vec![];
    for rank in 0..limit {
        for list in &lists {
            if let Some(c) = list.get(rank) {
                if seen.insert(c.id) {
                    let mut c = c.clone();
                    c.order = result.len();
                    result.push(c);
                    if result.len() == limit {
                        return result;
                    }
                }
            }
        }
    }
    result
}
pub(crate) fn validate_knobs(config: &crate::stage_config::StageConfig) -> Result<()> {
    if !config.keyword_weight.is_finite() || config.keyword_weight < 0.
        || !config.vector_weight.is_finite() || config.vector_weight < 0.
        || !config.similarity_threshold.is_finite() || !(0.0..=1.0).contains(&config.similarity_threshold) {
        bail!("检索权重须为有限非负数，相似度阈值须为 0–1")
    }
    if config.rerank_candidate_pool < config.rerank_candidate_pool_min
        || config.rerank_candidate_pool > config.rerank_candidate_pool_max {
        bail!("重排候选池须为 64–256")
    }
    Ok(())
}

/// Exact literal hits retain store match order; BM25 remains a separate leg.
pub fn grep_leg(store: &Store, query: &str, scope: &[String], references: &[String], limit: usize) -> Result<Vec<i64>> {
    Ok(store.literal_candidates(query, scope, references, limit)?.into_iter().map(|c| c.id).collect())
}

fn rank(mut candidates: Vec<Candidate>) -> Vec<Candidate> {
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.order.cmp(&b.order))
            .then(a.id.cmp(&b.id))
    });
    candidates
}

#[derive(serde::Serialize)]
pub struct SearchOutput {
    pub results: Vec<Candidate>,
    pub trace: Vec<Trace>,
}
pub async fn search<F, Fut>(
    store: &Store,
    query: &str,
    scope: &[String],
    options: &Retrieval,
    cancel: &CancellationToken,
    score: F,
    progress: impl Fn(&str, usize),
) -> Result<SearchOutput>
where
    F: FnMut(Vec<Candidate>, Vec<Candidate>) -> Fut,
    Fut: Future<Output = Result<Vec<Candidate>>>,
{
    let mut options = options.clone();
    options.candidate_k = 100;
    retrieve_inner(store, query, scope, &[], &options, cancel, Some(100), true, score, progress).await
}
pub async fn retrieve<F, Fut>(
    store: &Store,
    query: &str,
    scope: &[String],
    references: &[String],
    options: &Retrieval,
    cancel: &CancellationToken,
    score: F,
    progress: impl Fn(&str, usize),
) -> Result<SearchOutput>
where
    F: FnMut(Vec<Candidate>, Vec<Candidate>) -> Fut,
    Fut: Future<Output = Result<Vec<Candidate>>>,
{
    retrieve_inner(store, query, scope, references, options, cancel, None, true, score, progress).await
}
/// Mode-aware entry point for callers supplying ranked vector hits. Vector
/// legs must already reflect the enabled models; missing models supply no leg.
/// Fast mode and accurate mode without vector legs use configured SAG2 retrieval.
pub async fn retrieve_fused<F, Fut>(
    store: &Store,
    query: &str,
    scope: &[String],
    references: &[String],
    options: &Retrieval,
    vector_legs: &[Vec<i64>],
    cancel: &CancellationToken,
    score: F,
    progress: impl Fn(&str, usize),
) -> Result<SearchOutput>
where
    F: FnMut(Vec<Candidate>, Vec<Candidate>) -> Fut,
    Fut: Future<Output = Result<Vec<Candidate>>>,
{
    if options.mode == crate::types::RetrievalMode::Fast || vector_legs.is_empty() {
        return retrieve(store, query, scope, references, options, cancel, score, progress).await;
    }
    let config = crate::stage_config::StageConfig::from_retrieval(options);
    validate_knobs(&config)?;
    let mut sag_options = options.clone();
    sag_options.top_k = config.top_k_per_leg;
    let sag = retrieve_inner(store, query, scope, references, &sag_options, cancel, None, false, score, &progress).await;
    let mut output = match sag {
        Ok(output) => output,
        Err(error) => {
            // Cancellation remains an explicit stop, never a reason to continue.
            if cancel.is_cancelled() { return Err(error); }
            progress("SAG2 检索不可用，使用其余检索结果", 0);
            SearchOutput { results: vec![], trace: vec![] }
        }
    };
    let mut legs = Vec::new();
    let mut weights = Vec::new();
    if !output.results.is_empty() {
        legs.push(output.results.iter().map(|c| c.id).collect());
        weights.push(config.keyword_weight);
    }
    let mut candidates: HashMap<i64, Candidate> = output.results.drain(..).map(|c| (c.id, c)).collect();
    let mut resolved: HashSet<i64> = candidates.keys().copied().collect();
    let vector_count = vector_legs.iter().filter(|leg| !leg.is_empty()).count();
    for leg in vector_legs.iter().filter(|leg| !leg.is_empty()) {
        let mut available = Vec::new();
        let mut seen = HashSet::new();
        for &id in leg.iter().take(config.top_k_per_leg.min(config.candidate_pool)) {
            if cancel.is_cancelled() { bail!("已停止") }
            if !seen.insert(id) { continue; }
            if resolved.insert(id) {
                // Deleted, stale, unreadable, or out-of-scope hits are skipped
                // when materializing results; preserve the leg's original ranks.
                if let Ok(Some(candidate)) = store.candidate_by_id(id, scope, references) {
                    candidates.insert(id, candidate);
                }
            }
            available.push(id);
        }
        if !available.is_empty() { legs.push(available); weights.push(config.vector_weight / vector_count as f64); }
    }
    // Variant BM25 legs recall vocabulary-mismatched evidence the main legs miss.
    for variant in &options.variant_queries {
        let hits = store.candidate_query(&match_query(variant), None, scope, references, &[],
            config.top_k_per_leg.min(config.candidate_pool))?;
        let mut available = Vec::new();
        let mut seen = HashSet::new();
        for hit in hits {
            if cancel.is_cancelled() { bail!("已停止") }
            let id = hit.id;
            if !seen.insert(id) { continue; }
            if resolved.insert(id) {
                // Keep original ranks even when a hit cannot be materialized.
                if let Ok(Some(candidate)) = store.candidate_by_id(id, scope, references) {
                    candidates.insert(id, candidate);
                }
            }
            available.push(id);
        }
        if !available.is_empty() { legs.push(available); weights.push(config.keyword_weight); }
    }
    if cancel.is_cancelled() { bail!("已停止") }
    // Normalize reciprocal ranks to [0,1] before applying the confidence
    // threshold. Weights are shared across active vector legs; unavailable
    // legs do not suppress the confidence of the remaining legs.
    let total_weight: f64 = weights.iter().sum();
    let mut scores = IndexMap::<i64, f64>::new();
    for (leg, weight) in legs.iter().zip(weights) {
        let mut unique = HashSet::new();
        for (rank, id) in leg.iter().enumerate() {
            if candidates.contains_key(id) && unique.insert(*id) && total_weight > 0. {
                *scores.entry(*id).or_default() += weight / total_weight
                    * (config.rrf_k as f64 + 1.) / (config.rrf_k as f64 + rank as f64 + 1.);
            }
        }
    }
    let mut ranked: Vec<_> = scores.into_iter().filter(|(_, confidence)| *confidence > config.similarity_threshold).collect();
    // Dataset PageRank is multiplicative: 1.0 preserves the hybrid score.
    // Apply it before the pool/quota/top-N cutoff, after confidence filtering.
    for (id, score) in &mut ranked {
        *score *= store.document_weight(candidates[id].document_id)?;
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(config.candidate_pool);
    let mut groups = HashMap::<String, usize>::new();
    output.results = ranked.into_iter().filter_map(|(id, confidence)| {
        let candidate = candidates.get_mut(&id)?;
        let count = groups.entry(crate::models::source_author(&candidate.title, &candidate.heading)).or_default();
        if *count >= config.quota_per_author { return None; }
        *count += 1;
        candidate.score = confidence;
        Some(candidate.clone())
    }).take(config.top_k_final).enumerate().map(|(order, mut candidate)| {
        candidate.order = order;
        candidate
    }).collect();
    Ok(output)
}

/// Fusion retrieval with the optional rerank stage (Stage 3, 11.1).
/// Same as `retrieve_fused`, then `rerank::apply_rerank` when the config
/// enables it. `reranker` is `None` when no rerank model is configured;
/// with reranking off (the default) the scorer is never consulted and the
/// result is identical to `retrieve_fused`.
pub async fn retrieve_fused_reranked<F, Fut>(
    store: &Store,
    query: &str,
    scope: &[String],
    references: &[String],
    options: &Retrieval,
    vector_legs: &[Vec<i64>],
    cancel: &CancellationToken,
    score: F,
    progress: impl Fn(&str, usize),
    reranker: Option<&dyn crate::rerank::Reranker>,
) -> Result<SearchOutput>
where
    F: FnMut(Vec<Candidate>, Vec<Candidate>) -> Fut,
    Fut: Future<Output = Result<Vec<Candidate>>>,
{
    let output = retrieve_fused(store, query, scope, references, options, vector_legs, cancel, score, progress).await?;
    let config = crate::stage_config::StageConfig::from_retrieval(options);
    crate::rerank::apply_rerank(query, output, &config, reranker)
}

async fn retrieve_inner<F, Fut>(
    store: &Store,
    query: &str,
    scope: &[String],
    references: &[String],
    options: &Retrieval,
    cancel: &CancellationToken,
    file_limit: Option<usize>,
    dataset_prior: bool,
    mut score: F,
    progress: impl Fn(&str, usize),
) -> Result<SearchOutput>
where
    F: FnMut(Vec<Candidate>, Vec<Candidate>) -> Fut,
    Fut: Future<Output = Result<Vec<Candidate>>>,
{
    let config = crate::stage_config::StageConfig::from_retrieval(options);
    validate_knobs(&config)?;
    let version = store.version()?;
    if options.max_hops > 2 { bail!("SAG2 V10 最多扩展两轮") }
    if cancel.is_cancelled() { bail!("已停止") }
    progress("正在加载搜索索引", 0);
    if cancel.is_cancelled() { bail!("已停止") }
    let corpus = store.retrieval_statistics(scope, references, cancel)?;
    let repaired = corpus.repair(query, |term| store.has_candidate_match(&quote(term), scope, references, &[]))?;
    let expression = match_query(&repaired);
    if expression.is_empty() && references.is_empty() {
        return Ok(SearchOutput {
            results: vec![],
            trace: vec![],
        });
    };
    let expression = if expression.is_empty() {
        String::from("\"\"")
    } else {
        expression
    };
    let mut retained: Vec<Candidate> = vec![];
    let mut frontier = vec![];
    let mut expanded = HashSet::new();
    let mut seen = HashSet::new();
    let mut carried = IndexMap::new();
    let mut protected = IndexMap::new();
    let mut trace = vec![];
    let mut latest = IndexMap::<i64, Candidate>::new();
    for hop in 0..=options.max_hops {
        if cancel.is_cancelled() {
            bail!("已停止")
        };
        if store.version()? != version {
            bail!("索引已更新，请重新查询")
        };
        if hop > 0 && frontier.is_empty() {
            break;
        }
        let mut terms = if hop > 0 {
            expansion_terms(store, query, &frontier, options.expansion_term_limit)?
        } else {
            vec![]
        };
        if hop > 0 && terms.is_empty() {
            break;
        };
        progress(
            if hop == 0 {
                "正在检索原文"
            } else {
                "正在扩展检索"
            },
            hop,
        );
        let excluded: Vec<i64> = seen.iter().copied().collect();
        corpus.resolve_terms(store, &mut terms, scope, references, &excluded)?;
        let candidates = if hop == 0 {
            let mut hits = store.candidate_query(
                &expression,
                None,
                scope,
                references,
                &excluded,
                config.candidate_pool,
            )?;
            let known: HashSet<_> = hits.iter().map(|c| c.id).collect();
            let literal: Vec<_> = store.literal_candidates(query, scope, references, config.candidate_pool)?
                .into_iter().filter(|c| !known.contains(&c.id)).collect();
            if !literal.is_empty() {
                hits = interleave(vec![literal, hits], config.candidate_pool);
            }
            for c in hits.iter().take(5) {
                protected.insert(c.id, c.clone());
            }
            if references.is_empty() {
                hits
            } else {
                let mut combined = IndexMap::new();
                for c in store.reference_candidates(references, config.candidate_pool)? {
                    combined.insert(c.id, c);
                }
                for c in hits {
                    combined.insert(c.id, c);
                }
                combined
                    .into_values()
                    .take(config.candidate_pool)
                    .enumerate()
                    .map(|(i, mut c)| {
                        c.order = i;
                        c
                    })
                    .collect()
            }
        } else {
            let ranking = format!(
                "{} OR ({expression})",
                terms
                    .iter()
                    .map(|t| quote(&t.term))
                    .collect::<Vec<_>>()
                    .join(" OR ")
            );
            let required = terms.iter().map(|term| quote(&term.term)).collect::<Vec<_>>();
            let lists = store.candidate_queries(
                &ranking,
                &required,
                scope,
                references,
                &excluded,
                config.candidate_pool,
            )?;
            interleave(lists, config.candidate_pool)
        };
        if candidates.is_empty() {
            break;
        };
        for c in &frontier {
            carried.insert(c.id, c.clone());
        }
        let mut pool = IndexMap::new();
        if hop > 0 {
            pool.extend(protected.iter().map(|(id, c)| (*id, c.clone())));
            pool.extend(carried.iter().map(|(id, c)| (*id, c.clone())));
        }
        pool.extend(candidates.iter().map(|c| (c.id, c.clone())));
        let mut evidence = if hop > 0 {
            let mut e = carried.clone();
            e.extend(
                candidates
                    .iter()
                    .take(options.seed_k)
                    .map(|c| (c.id, c.clone())),
            );
            e.into_values().collect()
        } else {
            vec![]
        };
        if hop == 2 {
            let count = pool.len();
            let mut history: IndexMap<i64, Candidate> = pool.iter().filter(|(id,_)| latest.contains_key(*id)).map(|(id,c)| (*id,c.clone())).collect();
            let incoming: Vec<_> = pool.values().filter(|c| !latest.contains_key(&c.id)).take(20).cloned().collect();
            let mut previous: Vec<_> = latest.values().cloned().collect();
            previous.sort_by(|a,b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
            history.extend(previous.into_iter().map(|c| (c.id,c)));
            pool = history.into_iter().take(count-incoming.len()).chain(incoming.into_iter().map(|c| (c.id,c))).collect();
            evidence = retained.iter().take(10).cloned().map(|mut c| {c.previous_score=Some(c.score);c}).collect();
        }
        trace.push(Trace {
            hop,
            candidates: candidates.iter().map(|c| c.id).collect(),
            terms,
            evidence: evidence.iter().map(|c: &Candidate| c.id).collect(),
        });
        progress("Jev 正在判断相关原文", pool.len());
        retained = rank(score(pool.into_values().collect(), evidence).await?);
        latest.extend(retained.iter().map(|c| (c.id,c.clone())));
        if cancel.is_cancelled() {
            bail!("已停止")
        };
        if store.version()? != version {
            bail!("索引已更新，请重新查询")
        };
        seen.extend(retained.iter().map(|c| c.id));
        frontier = retained
            .iter()
            .filter(|c| !expanded.contains(&c.id))
            .take(options.seed_k)
            .cloned()
            .collect();
        expanded.extend(frontier.iter().map(|c| c.id));
    }
    corpus.diversify(&mut retained);
    // Keep the default SAG2 diversity order; a non-default dataset prior
    // deliberately reorders candidates before truncation.
    let best = retained.iter().map(|c| c.score).fold(0., f64::max);
    let confidence: HashMap<_, _> = retained.iter().map(|c| (c.id, if best > 0. { c.score / best } else { 0. })).collect();
    let mut weighted = false;
    for candidate in &mut retained {
        let weight = if dataset_prior { store.document_weight(candidate.document_id)? } else { 1. };
        weighted |= weight != 1.;
        candidate.score *= weight;
    }
    if weighted { retained = rank(retained); }
    if let Some(limit) = file_limit {
        // The deep-search list also retains already scored matches after the V10 result.
        // This keeps file browsing coverage without another model request.
        let current: HashSet<_> = retained.iter().map(|c| c.id).collect();
        let mut additional = rank(latest.into_values().filter(|c| !current.contains(&c.id)).collect());
        if dataset_prior {
            for candidate in &mut additional {
                let weight = store.document_weight(candidate.document_id)?;
                weighted |= weight != 1.;
                candidate.score *= weight;
            }
        }
        retained.extend(additional);
        if weighted { retained = rank(retained); }
        let mut files = HashSet::new();
        retained.retain(|candidate| {
            files.contains(&candidate.path)
                || (files.len() < limit && files.insert(candidate.path.clone()))
        });
    } else {
        // SAG2 relevance has no fixed scale. Use confidence relative to the
        // best result for filtering; retain its score scale for diversification.
        retained.retain(|c| config.keyword_weight > 0.
            && confidence[&c.id] > config.similarity_threshold);
        for c in &mut retained { c.score *= config.keyword_weight; }
        retained.truncate(options.top_k.min(config.candidate_pool));
    }
    Ok(SearchOutput {
        results: retained,
        trace,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn c(id: i64) -> Candidate {
        Candidate {
            previous_score: None,
            id,
            document_id: id,
            path: String::new(),
            title: String::new(),
            heading: String::new(),
            content: String::new(),
            start_line: 1,
            end_line: 1,
            source_ranges: vec![(1, 1)],
            source_spans: vec![],
            bm25: 0.,
            score: 0.,
            order: 0,
            hash: String::new(),
        }
    }
    #[tokio::test]
    async fn dataset_weight_prioritizes_ties_before_final_cutoff() {
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("index.sqlite");
        let store = Store::open(&database).unwrap();
        let mut roots = vec![];
        for folder in ["low", "high"] {
            let path = dir.path().join(folder);
            std::fs::create_dir(&path).unwrap();
            std::fs::write(path.join("same.txt"), "nebula evidence").unwrap();
            let root = store.add_root(&path).unwrap();
            assert_eq!(store.root_weight(&root).unwrap(), 1.);
            store.scan(&root, |_| {}).unwrap();
            roots.push(root);
        }
        let hits = store.candidate_query(&match_query("nebula"), None, &[], &[], &[], 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].bm25, hits[1].bm25);
        let high = hits.iter().find(|c| c.path.starts_with(&roots[1])).unwrap().id;
        store.set_root_weight(&roots[1], 2.).unwrap();
        assert!(store.set_root_weight(&roots[0], f64::NAN).is_err());
        drop(store);
        let store = Store::open(&database).unwrap();
        assert_eq!(store.root_weight(&roots[1]).unwrap(), 2.);
        async fn score(mut rows: Vec<Candidate>, _: Vec<Candidate>) -> Result<Vec<Candidate>> {
            for row in &mut rows { row.score = 0.9; }
            Ok(rows)
        }
        let mut options = Retrieval { max_hops: 0, top_k: 1, top_k_final: 1,
            keyword_weight: 0.5, vector_weight: 0.5, ..Default::default() };
        let cancel = CancellationToken::new();
        let fast = retrieve(&store, "nebula", &[], &[], &options, &cancel, score, |_, _| {}).await.unwrap();
        assert_eq!(fast.results[0].id, high);
        // Opposite equally weighted lists give both chunks equal RRF scores.
        options.mode = crate::types::RetrievalMode::Accurate;
        store.set_root_weight(&roots[1], 1.).unwrap();
        let unweighted = retrieve_fused(&store, "nebula", &[], &[], &options,
            &[vec![hits[1].id, hits[0].id]], &cancel, score, |_, _| {}).await.unwrap();
        store.set_root_weight(&roots[1], 2.).unwrap();
        let weighted = retrieve_fused(&store, "nebula", &[], &[], &options,
            &[vec![hits[1].id, hits[0].id]], &cancel, score, |_, _| {}).await.unwrap();
        assert_ne!(unweighted.results[0].id, high);
        assert_eq!(weighted.results[0].id, high);
    }

    #[tokio::test]
    async fn configured_weights_threshold_and_pool_change_retrieval() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        std::fs::create_dir(&files).unwrap();
        std::fs::write(files.join("lexical.txt"), "nebula evidence").unwrap();
        std::fs::write(files.join("vector.txt"), "quasar evidence").unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let root = store.add_root(&files).unwrap();
        store.scan(&root, |_| {}).unwrap();
        let vector = store.candidate_query(&match_query("quasar"), None, &[], &[], &[], 1).unwrap()[0].id;
        let legs = vec![vec![vector]];
        let cancel = CancellationToken::new();
        async fn score(mut rows: Vec<Candidate>, _: Vec<Candidate>) -> Result<Vec<Candidate>> {
            for row in &mut rows { row.score = 0.9; }
            Ok(rows)
        }
        let mut options = Retrieval { mode: crate::types::RetrievalMode::Accurate, max_hops: 0, similarity_threshold: 0., ..Default::default() };
        let lexical = retrieve_fused(&store, "nebula", &[], &[], &options, &legs, &cancel, score, |_, _| {}).await.unwrap();
        assert_eq!(lexical.results.len(), 2);
        assert_ne!(lexical.results[0].id, vector);
        options.keyword_weight = 0.1;
        options.vector_weight = 0.9;
        let semantic = retrieve_fused(&store, "nebula", &[], &[], &options, &legs, &cancel, score, |_, _| {}).await.unwrap();
        assert_eq!(semantic.results[0].id, vector);
        options.candidate_k = 1;
        let limited = retrieve_fused(&store, "nebula", &[], &[], &options, &legs, &cancel, score, |_, _| {}).await.unwrap();
        assert_eq!(limited.results.len(), 1);
        assert_eq!(limited.results[0].id, vector);
        for mode in [crate::types::RetrievalMode::Fast, crate::types::RetrievalMode::Accurate] {
            options.mode = mode;
            options.similarity_threshold = 1.;
            assert!(retrieve_fused(&store, "nebula", &[], &[], &options, &legs, &cancel, score, |_, _| {}).await.unwrap().results.is_empty());
            options.similarity_threshold = 0.;
            assert!(!retrieve_fused(&store, "nebula", &[], &[], &options, &legs, &cancel, score, |_, _| {}).await.unwrap().results.is_empty());
        }
    }

    #[tokio::test]
    async fn fused_retrieval_quota_is_shared_across_documents_by_author() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        std::fs::create_dir(&files).unwrap();
        for name in ["Author-first.md", "Author-second.md"] {
            let content = (0..6).map(|i| format!("# Section {i}\n\n{}\n\n", "nebula evidence ".repeat(40)))
                .collect::<String>();
            std::fs::write(files.join(name), content).unwrap();
        }
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let root = store.add_root(&files).unwrap();
        store.scan(&root, |_| {}).unwrap();
        let hits = store.candidate_query(&match_query("nebula"), None, &[], &[], &[], 50).unwrap();
        let documents: HashSet<_> = hits.iter().map(|c| c.document_id).collect();
        assert_eq!(documents.len(), 2);
        for document in documents {
            assert!(hits.iter().filter(|c| c.document_id == document).count() >= 5);
        }
        assert!(hits.iter().all(|c| crate::models::source_author(&c.title, &c.heading) == "Author"));
        let vectors = vec![hits.iter().map(|c| c.id).collect()];
        let options = Retrieval { mode: crate::types::RetrievalMode::Accurate, max_hops: 0, ..Default::default() };
        let found = retrieve_fused(&store, "", &[], &[], &options, &vectors, &CancellationToken::new(),
            |rows, _| async { Ok(rows) }, |_, _| {}).await.unwrap();
        assert_eq!(found.results.len(), 5);
    }

    #[tokio::test]
    async fn fused_retrieval_variant_recalls_evidence_missed_by_main_legs() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        std::fs::create_dir(&files).unwrap();
        std::fs::write(files.join("main.txt"), "nebula evidence").unwrap();
        std::fs::write(files.join("variant.txt"), "quasar testimony").unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let root = store.add_root(&files).unwrap();
        store.scan(&root, |_| {}).unwrap();
        let main = store.candidate_query(&match_query("nebula"), None, &[], &[], &[], 1).unwrap()[0].id;
        let variant = store.candidate_query(&match_query("quasar"), None, &[], &[], &[], 1).unwrap()[0].id;
        let vectors = vec![vec![main]];
        let mut options = Retrieval { mode: crate::types::RetrievalMode::Accurate, max_hops: 0, ..Default::default() };
        let cancel = CancellationToken::new();
        async fn score(mut rows: Vec<Candidate>, _: Vec<Candidate>) -> Result<Vec<Candidate>> {
            for row in &mut rows { row.score = 0.9; }
            Ok(rows)
        }
        let baseline = retrieve_fused(&store, "nebula", &[], &[], &options, &vectors, &cancel, score, |_, _| {}).await.unwrap();
        assert!(!baseline.results.iter().any(|c| c.id == variant));
        options.variant_queries = vec!["quasar".into()];
        let expanded = retrieve_fused(&store, "nebula", &[], &[], &options, &vectors, &cancel, score, |_, _| {}).await.unwrap();
        assert!(expanded.results.iter().take(10).any(|c| c.id == variant));
    }

    #[tokio::test]
    async fn literal_leg_recovers_buried_number_outside_bm25_pool() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        std::fs::create_dir(&files).unwrap();
        let buried = files.join("buried.txt");
        std::fs::write(&buried, format!("{} 40天", "background ".repeat(500))).unwrap();
        let short = files.join("short.txt");
        std::fs::write(&short, "40天 40天 40天").unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let root = store.add_root(&files).unwrap();
        store.index_file(&root, &buried, 20000).unwrap();
        store.index_file(&root, &short, 20000).unwrap();
        let bm25 = store.candidate_query(&match_query("40天"), None, &[], &[], &[], 1).unwrap();
        // Stored paths are canonical (see index_file); compare against the
        // canonical path, which differs from the raw tempdir path under symlinks.
        let canonical_short = short.canonicalize().unwrap();
        assert_eq!(bm25[0].path, canonical_short.to_str().unwrap());
        let exact = grep_leg(&store, "40天", &[], &[], 1).unwrap();
        assert_ne!(exact[0], bm25[0].id);
        let output = retrieve(&store, "40天", &[], &[],
            &Retrieval { candidate_k: 1, max_hops: 0, ..Default::default() },
            &CancellationToken::new(), |mut candidates, _| async move {
                for c in &mut candidates { c.score = 0.9; }
                Ok(candidates)
            }, |_, _| {}).await.unwrap();
        assert_eq!(output.results[0].id, exact[0]);
        assert!(grep_leg(&store, "40.*天", &[], &[], 10).unwrap().is_empty());
        assert!(grep_leg(&store, "40天", &[files.join("other").to_string_lossy().into()], &[], 10).unwrap().is_empty());
    }

    #[test]
    fn sentence_boundaries_preserve_english_initials_and_split_chinese() {
        assert_eq!(
            sentences("台风登陆。救援开始！人员安全吗？仍在核查").unwrap(),
            vec!["台风登陆", "救援开始", "人员安全吗", "仍在核查"]
        );
        assert_eq!(
            sentences("J. R. Smith arrived. Help! Why?\nNext").unwrap(),
            vec!["J. R. Smith arrived", "Help", "Why", "Next"]
        );
    }
    #[test]
    fn chinese_sentences_do_not_share_expansion_context() {
        for punctuation in ["。", "！", "？"] {
            let dir = tempfile::tempdir().unwrap();
            let files = dir.path().join("files");
            std::fs::create_dir(&files).unwrap();
            let content = [
                "台风登陆福建，救援人员前往沿海",
                "成都大运会开幕",
                "母婴设施建设启动",
            ].join(punctuation);
            std::fs::write(files.join("news.txt"), &content).unwrap();
            let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
            let root = store.add_root(&files).unwrap();
            store.scan(&root, |_| {}).unwrap();
            let mut seed = c(1);
            seed.content = content;
            seed.score = 1.;
            let terms = expansion_terms(&store, "台风 福建", &[seed], 100).unwrap();
            let rescue = terms.iter().find(|t| t.term == "救援").unwrap();
            for word in ["大运会", "母婴"] {
                let unrelated = terms.iter().find(|t| t.term == word).unwrap();
                assert!(rescue.score > unrelated.score, "{punctuation}: {word}");
            }
        }
    }
    #[test]
    fn interleaves_rank_then_term_and_deduplicates() {
        let result = interleave(vec![vec![c(1), c(3), c(4)], vec![c(2), c(3), c(5)]], 5);
        assert_eq!(
            result.iter().map(|c| c.id).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );
    }
    #[tokio::test]
    async fn fused_retrieval_modes_fallback_and_vector_hydration() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        std::fs::create_dir(&files).unwrap();
        std::fs::write(files.join("match.txt"), "nebula evidence.").unwrap();
        std::fs::write(files.join("vector.txt"), "quasar vector evidence.").unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let root = store.add_root(&files).unwrap();
        store.scan(&root, |_| {}).unwrap();
        let mut options = Retrieval { max_hops: 0, mode: crate::types::RetrievalMode::Accurate, top_k: 1, ..Default::default() };
        let cancel = CancellationToken::new();
        async fn score(mut candidates: Vec<Candidate>, _: Vec<Candidate>) -> Result<Vec<Candidate>> {
            for candidate in &mut candidates { candidate.score = 0.9; }
            Ok(candidates)
        }
        let plain = retrieve(&store, "nebula", &[], &[], &options, &cancel, score, |_, _| {}).await.unwrap();
        assert!(!plain.results.is_empty());
        let degraded = retrieve_fused(&store, "nebula", &[], &[], &options, &[], &cancel, score, |_, _| {}).await.unwrap();
        assert_eq!(serde_json::to_value(&plain).unwrap(), serde_json::to_value(&degraded).unwrap());
        let vector = store.candidate_query(&match_query("quasar"), None, &[], &[], &[], options.candidate_k).unwrap().remove(0);
        let vectors = vec![vec![], vec![vector.id, i64::MAX]];
        let fused = retrieve_fused(&store, "nebula", &[], &[], &options, &vectors, &cancel, score, |_, _| {}).await.unwrap();
        assert_eq!(fused.results.len(), 2);
        assert_eq!(fused.results[1].content, vector.content);
        assert_eq!(fused.results[1].source_ranges, vector.source_ranges);
        assert_eq!(fused.results[1].document_id, vector.document_id);
        assert_eq!(fused.results[1].order, 1);
        let only_vector = retrieve_fused(&store, "", &[], &[], &options, &vectors, &cancel, score, |_, _| {}).await.unwrap();
        assert_eq!(only_vector.results.len(), 1);
        assert_eq!(only_vector.results[0].id, vector.id);
        let empty = retrieve_fused(&store, "", &[], &[], &options, &[vec![]], &cancel, score, |_, _| {}).await.unwrap();
        assert!(empty.results.is_empty());
        let scope = vec![files.join("match.txt").to_str().unwrap().to_owned()];
        let scoped = retrieve_fused(&store, "", &scope, &[], &options, &vectors, &cancel, score, |_, _| {}).await.unwrap();
        assert!(scoped.results.is_empty());
        // Failure of the SAG2 scoring leg leaves the vector leg available.
        let failed = retrieve_fused(&store, "nebula", &[], &[], &options, &vectors, &cancel,
            |_, _| async { bail!("unavailable scoring leg") }, |_, _| {}).await.unwrap();
        assert_eq!(failed.results.len(), 1);
        assert_eq!(failed.results[0].id, vector.id);
        options.mode = crate::types::RetrievalMode::Fast;
        let fast = retrieve_fused(&store, "nebula", &[], &[], &options, &vectors, &cancel, score, |_, _| {}).await.unwrap();
        assert_eq!(serde_json::to_value(&plain).unwrap(), serde_json::to_value(&fast).unwrap());
        options.mode = crate::types::RetrievalMode::Accurate;
        cancel.cancel();
        assert!(retrieve_fused(&store, "nebula", &[], &[], &options, &vectors, &cancel, score, |_, _| {}).await.is_err());
    }

    #[test]
    fn names_keep_initials_and_boundaries() {
        let names =
            name_phrases("J. R. Smith met Mary-Jane Watson. A city called New York").unwrap();
        assert_eq!(names.len(), 3);
        assert!(names[0].2.contains("smith"));
    }
    #[tokio::test]
    async fn deep_search_reuses_preparation_and_refreshes_after_index_changes() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        std::fs::create_dir(&files).unwrap();
        let matched = files.join("match.txt");
        let unrelated = files.join("unrelated.txt");
        std::fs::write(&matched, "nebula evidence.").unwrap();
        std::fs::write(&unrelated, "unrelated material.").unwrap();
        let database = dir.path().join("index.sqlite");
        let store = Store::open(&database).unwrap();
        let root = store.add_root(&files).unwrap();
        store.scan(&root, |_| {}).unwrap();
        let options = Retrieval { max_hops: 0, ..Default::default() };
        let cancel = CancellationToken::new();
        async fn score(mut candidates: Vec<Candidate>, _: Vec<Candidate>) -> Result<Vec<Candidate>> {
            for candidate in &mut candidates { candidate.score = 0.9; }
            Ok(candidates)
        }
        let first = search(&store, "nebula", &[], &options, &cancel, score, |_, _| {}).await.unwrap();
        // Statistics saved by indexing survive an application restart.
        drop(store);
        let store = Store::open(&database).unwrap();
        // An unchanged index must reuse statistics without reopening unrelated files.
        std::fs::remove_file(&unrelated).unwrap();
        let repeated = search(&store, "nebula", &[], &options, &cancel, score, |_, _| {}).await.unwrap();
        assert_eq!(serde_json::to_value(&first.results).unwrap(), serde_json::to_value(&repeated.results).unwrap());
        // A passage whose source changed since indexing is dropped, not reported as an error.
        std::fs::write(&matched, "changed evidence.").unwrap();
        let stale = search(&store, "nebula", &[], &options, &cancel, score, |_, _| {}).await.unwrap();
        assert!(stale.results.iter().all(|c| c.path != matched.to_str().unwrap()));
        // Another process updating the index must invalidate cached statistics.
        std::fs::write(&matched, "comet replacement evidence.").unwrap();
        let writer = Store::open(&database).unwrap();
        writer.scan(&root, |_| {}).unwrap();
        let updated = search(&store, "comet", &[], &options, &cancel, score, |_, _| {}).await.unwrap();
        assert_eq!(updated.results.len(), 1);
        assert_eq!(updated.results[0].content, "comet replacement evidence.");
    }

    #[tokio::test]
    async fn deep_search_reports_preparation_before_reading_and_checks_cancellation() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let cancel = CancellationToken::new();
        let stages = std::cell::RefCell::new(Vec::new());
        let output = search(&store, "nebula", &[], &Retrieval::default(), &cancel,
            |_, _| async { panic!("a cancelled search must not score passages") },
            |stage, _| { stages.borrow_mut().push(stage.to_owned()); cancel.cancel(); }).await;
        assert_eq!(stages.borrow().first().map(String::as_str), Some("正在加载搜索索引"));
        assert_eq!(output.err().unwrap().to_string(), "已停止");
    }

    #[tokio::test]
    async fn deep_search_returns_one_hundred_ranked_files_without_changing_answer_limits() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        std::fs::create_dir(&files).unwrap();
        for id in 0..120 {
            std::fs::write(files.join(format!("file-{id:03}.txt")), format!("nebula bridgeword evidence {id}")).unwrap();
        }
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let root = store.add_root(&files).unwrap();
        store.scan(&root, |_| {}).unwrap();
        let options = Retrieval::default();
        async fn score(mut candidates: Vec<Candidate>, _: Vec<Candidate>) -> Result<Vec<Candidate>> {
            for candidate in &mut candidates {
                let id: usize = candidate.title.trim_start_matches("file-").trim_end_matches(".txt").parse().unwrap();
                candidate.score = (120 - id) as f64 / 120.;
            }
            Ok(candidates)
        }
        let output = search(&store, "nebula", &[], &options, &CancellationToken::new(), score, |_, _| {}).await.unwrap();
        assert_eq!(output.results.len(), 100);
        assert_eq!(output.results.iter().map(|c| c.path.as_str()).collect::<HashSet<_>>().len(), 100);
        assert_eq!(output.results.iter().map(|c| c.id).collect::<HashSet<_>>().len(), 100);
        assert_eq!(output.results.first().unwrap().title, "file-000.txt");
        assert!(output.trace.len() <= 3);
        let answer = retrieve(&store, "nebula", &[], &[], &options, &CancellationToken::new(), score, |_, _| {}).await.unwrap();
        assert_eq!(answer.results.len(), 10);
    }
    #[tokio::test]
    async fn deep_search_keeps_multiple_matches_per_file_beyond_one_hundred_passages() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        std::fs::create_dir(&files).unwrap();
        for id in 0..60 {
            let paragraph = format!("nebula bridgeword chapter{id}. ").repeat(60);
            std::fs::write(files.join(format!("file-{id}.txt")), [&paragraph, &paragraph, &paragraph].into_iter().cloned().collect::<Vec<_>>().join("\n\n")).unwrap();
        }
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let root = store.add_root(&files).unwrap();
        store.scan(&root, |_| {}).unwrap();
        let output = search(&store, "nebula", &[], &Retrieval::default(), &CancellationToken::new(), |mut candidates, _| async move {
            for candidate in &mut candidates { candidate.score = 0.8; }
            Ok(candidates)
        }, |_, _| {}).await.unwrap();
        assert!(output.results.len() > 100);
        assert!(output.results.iter().map(|c| &c.path).collect::<HashSet<_>>().len() < 100);
        assert_eq!(output.results.iter().map(|c| c.id).collect::<HashSet<_>>().len(), output.results.len());
    }
    #[tokio::test]
    async fn all_hops_keep_scope() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        std::fs::create_dir(&a).unwrap();
        std::fs::create_dir(&b).unwrap();
        std::fs::write(a.join("a.md"), "bridge beta alpha").unwrap();
        std::fs::write(b.join("b.md"), "bridge gamma beta").unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let root = store.add_root(dir.path()).unwrap();
        store.scan(&root, |_| {}).unwrap();
        let output = retrieve(
            &store,
            "bridge",
            &[a.to_str().unwrap().into()],
            &[],
            &Retrieval::default(),
            &CancellationToken::new(),
            |mut candidates, evidence| async move {
                assert!(evidence.iter().all(|c| c.path.contains("/a/")));
                for c in &mut candidates {
                    c.score = 0.8;
                }
                Ok(candidates)
            },
            |_, _| {},
        )
        .await
        .unwrap();
        assert!(output
            .results
            .iter()
            .all(|c| Path::new(&c.path).starts_with(&a)));
    }
    use std::path::Path;
}

#[cfg(test)]
mod oracle_tests {
    use super::*;
    use rusqlite::params;
    use serde_json::{json, Value};
    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../tests/sag2_v10_oracle.json")).unwrap()
    }
    #[test]
    fn tokenizer_matches_frozen_v10() {
        for row in fixture()["tokenizer"].as_array().unwrap() {
            assert_eq!(
                json!(tokenize(row["text"].as_str().unwrap())),
                row["tokens"],
                "{}",
                row["text"]
            );
        }
    }
    #[tokio::test]
    async fn frozen_v10_requests_and_rounds() {
        let fixture = fixture();
        for case in fixture["cases"].as_array().unwrap() {
            let dir = tempfile::tempdir().unwrap();
            let store = Store::open(&dir.path().join("oracle.sqlite")).unwrap();
            {
                let db = store.connection();
                for (i, doc) in fixture["corpus"].as_array().unwrap().iter().enumerate() {
                    let id = i as i64 + 1;
                    let title = doc["title"].as_str().unwrap();
                    let content = doc["text"].as_str().unwrap();
                    let path = dir.path().join(format!("{id}.txt"));
                    std::fs::write(&path, content).unwrap();
                    db.execute(
                        "INSERT INTO documents(id,root,path,name,hash,mtime) VALUES(?,?,?,?,?,?)",
                        params![
                            id,
                            "/fixture",
                            path.to_str().unwrap(),
                            title,
                            crate::store::hash(content.as_bytes()),
                            0,
                        ],
                    )
                    .unwrap();
                    db.execute(
                        "INSERT INTO chunks(id,document_id,heading,ranges,start_line,end_line) VALUES(?,?,?,?,?,?)",
                        params![id, id, "", json!([[1, 1]]).to_string(), 1, 1],
                    )
                    .unwrap();
                    db.execute(
                        "INSERT INTO chunk_fts(rowid,title,heading,body) VALUES(?,?,?,?)",
                        params![
                            id,
                            crate::text::index_text(title),
                            "",
                            crate::text::index_text(content)
                        ],
                    )
                    .unwrap();
                }
            }
            let options: Retrieval = serde_json::from_value(case["options"].clone()).unwrap();
            let mut round = 0usize;
            let expected = case["requests"].as_array().unwrap();
            let query = case["query"].as_str().unwrap();
            let output = retrieve(
                &store,
                query,
                &[],
                &[],
                &options,
                &CancellationToken::new(),
                |candidates, evidence| {
                    let actual = crate::models::jev_batches(
                        query,
                        candidates.clone(),
                        &evidence,
                        "jev-latest",
                    )
                    .unwrap();
                    assert_eq!(
                        actual[0].0, expected[round],
                        "request round {round}, {query}"
                    );
                    round += 1;
                    async move {
                        let titles: Vec<_> = evidence.iter().map(|c| c.title.as_str()).collect();
                        Ok(candidates
                            .into_iter()
                            .map(|mut c| {
                                c.score = match c.title.as_str() {
                                    "Anchor" => 0.8,
                                    "Multiple" => {
                                        if titles.contains(&"Anchor") {
                                            0.95
                                        } else {
                                            0.1
                                        }
                                    }
                                    "Bridge" => {
                                        if titles.contains(&"Multiple") {
                                            0.99
                                        } else {
                                            0.05
                                        }
                                    }
                                    "High" => 0.7,
                                    "Series" => 0.08,
                                    _ => 0.01,
                                };
                                c
                            })
                            .collect())
                    }
                },
                |_, _| {},
            )
            .await
            .unwrap();
            assert_eq!(round, expected.len());
            assert_eq!(
                serde_json::json!(output.results.iter().map(|c| c.id).collect::<Vec<_>>()),
                case["result"]["ranked_ids"]
            );
            for (actual, want) in output
                .trace
                .iter()
                .zip(case["result"]["rounds"].as_array().unwrap())
            {
                assert_eq!(json!(actual.candidates), want["ids"]);
                assert_eq!(
                    json!(actual.terms.iter().map(|t| &t.term).collect::<Vec<_>>()),
                    want["terms"]
                );
            }
        }
    }
}
