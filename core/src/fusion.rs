//! Multi-leg retrieval fusion: reciprocal rank fusion (RRF) plus a
//! per-group quota.
//!
//! Ported from the validated Python reference
//! (`hybrid-rag-quota-fusion.py`, variant A1), which reaches 4.0/5 retrieval
//! on the six-question cross-file set (plain global RRF reaches 3.5/5).
//! The fusion is pure ranking math over ranked chunk-id lists, one list per
//! retrieval leg. It does not depend on any embedding model or store, so it
//! can be unit-tested against the reference corpus fixture.
//!
//! RRF: the item at 0-based rank `i` in a leg contributes `1 / (k + i + 1)`.
//! Quota (A1): take the top `quota_per_group` items of each group by global
//! RRF score, merge the groups, order by RRF score, take the top `top_k`.
//! A "group" is the source author in the reference corpus; in SAG-Wiki it
//! generalises to the source document, so no single document can dominate
//! the final list. Groups are taken in order of first appearance.
//!
//! All sorts are stable and items keep their first-seen order, which
//! reproduces the Python reference exactly (verified by the unit tests
//! against the fixture in `fusion_fixture.json`).

use indexmap::IndexMap;
use std::hash::Hash;

/// RRF scores for every item appearing in any leg, in first-seen order.
/// The item at 0-based rank `i` in a leg contributes `1 / (k + i + 1)`.
pub fn rrf_scores<T>(legs: &[Vec<T>], k: f64) -> IndexMap<T, f64>
where
    T: Eq + Hash + Clone,
{
    let mut scores: IndexMap<T, f64> = IndexMap::new();
    for leg in legs {
        for (i, item) in leg.iter().enumerate() {
            let contribution = 1.0 / (k + i as f64 + 1.0);
            *scores.entry(item.clone()).or_insert(0.0) += contribution;
        }
    }
    scores
}

/// Global RRF top-`top_k` (no quota). This is the plain-fusion baseline
/// (3.5/5 on the reference set).
pub fn plain_rrf<T>(legs: &[Vec<T>], k: f64, top_k: usize) -> Vec<T>
where
    T: Eq + Hash + Clone,
{
    let scores = rrf_scores(legs, k);
    let mut items: Vec<(&T, &f64)> = scores.iter().collect();
    items.sort_by(|a, b| b.1.total_cmp(a.1));
    items
        .into_iter()
        .take(top_k)
        .map(|(item, _)| item.clone())
        .collect()
}

/// RRF + per-group quota (variant A1, the validated fusion: 4.0/5).
/// `group_of` maps an item to its group (author in the reference corpus,
/// source document in SAG-Wiki). Groups are processed in order of first
/// appearance; each contributes at most `quota_per_group` items.
pub fn quota_fuse<T, G>(
    legs: &[Vec<T>],
    k: f64,
    quota_per_group: usize,
    top_k: usize,
    group_of: impl Fn(&T) -> G,
) -> Vec<T>
where
    T: Eq + Hash + Clone,
    G: Eq + Hash + Clone,
{
    let scores = rrf_scores(legs, k);
    // (item, score, group) in first-seen order.
    let entries: Vec<(T, f64, G)> = scores
        .iter()
        .map(|(item, score)| (item.clone(), *score, group_of(item)))
        .collect();
    // Groups in order of first appearance.
    let mut groups: Vec<G> = Vec::new();
    for (_, _, group) in &entries {
        if !groups.contains(group) {
            groups.push(group.clone());
        }
    }
    let mut merged: Vec<(T, f64)> = Vec::new();
    for group in &groups {
        let mut items: Vec<(T, f64)> = entries
            .iter()
            .filter(|(_, _, g)| g == group)
            .map(|(item, score, _)| (item.clone(), *score))
            .collect();
        items.sort_by(|a, b| b.1.total_cmp(&a.1));
        merged.extend(items.into_iter().take(quota_per_group));
    }
    merged.sort_by(|a, b| b.1.total_cmp(&a.1));
    merged
        .into_iter()
        .take(top_k)
        .map(|(item, _)| item)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Reference corpus fixture: the three legs' top-50 lists per question
    // (WeMM bq1024, Conan bq, SAG2), taken from the Python validation runs.
    const FIXTURE: &str = include_str!("fusion_fixture.json");

    #[derive(serde::Deserialize)]
    struct Fixture {
        questions: HashMap<String, LegLists>,
    }

    #[derive(serde::Deserialize)]
    struct LegLists {
        wemm: Vec<String>,
        conan: Vec<String>,
        sag: Vec<String>,
    }

    fn fixture() -> Fixture {
        serde_json::from_str(FIXTURE).expect("fusion fixture must parse")
    }

    use crate::retrieval_panel::score_top10;
    fn group_of(id: &String) -> &'static str {
        if id.starts_with("chaofeng") {
            "cjf"
        } else {
            "xw"
        }
    }

    fn legs_for(fixture: &Fixture, qid: &str) -> Vec<Vec<String>> {
        let lists = &fixture.questions[qid];
        vec![lists.wemm.clone(), lists.conan.clone(), lists.sag.clone()]
    }

    #[test]
    fn quota_fuse_matches_python_a1() {
        let fixture = fixture();
        // Exact top-10 per question from the Python A1 run.
        let expected: [(&str, &[&str]); 6] = [
            (
                "Q1",
                &[
                    "xiaowan-full.md#0",
                    "xiaowan.md#15",
                    "chaofeng-full.md#47",
                    "chaofeng-full.md#27",
                    "chaofeng-full.md#15",
                    "chaofeng-full.md#10",
                    "chaofeng-full.md#24",
                    "xiaowan-full.md#7",
                    "xiaowan.md#10",
                    "xiaowan-full.md#21",
                ],
            ),
            (
                "Q2",
                &[
                    "chaofeng-full.md#10",
                    "chaofeng-full.md#35",
                    "chaofeng-full.md#21",
                    "chaofeng-full.md#11",
                    "chaofeng-full.md#20",
                    "xiaowan-full.md#36",
                    "xiaowan.md#4",
                    "xiaowan-full.md#5",
                    "xiaowan.md#24",
                    "xiaowan.md#25",
                ],
            ),
            (
                "Q3",
                &[
                    "xiaowan.md#17",
                    "chaofeng-full.md#15",
                    "chaofeng-full.md#23",
                    "chaofeng-full.md#41",
                    "xiaowan-full.md#20",
                    "xiaowan.md#21",
                    "xiaowan.md#3",
                    "xiaowan.md#4",
                    "chaofeng-full.md#27",
                    "chaofeng-full.md#10",
                ],
            ),
            (
                "Q4",
                &[
                    "chaofeng-full.md#17",
                    "chaofeng-full.md#18",
                    "chaofeng-full.md#44",
                    "chaofeng-full.md#5",
                    "chaofeng-full.md#14",
                    "xiaowan.md#3",
                    "xiaowan-full.md#4",
                    "xiaowan.md#19",
                    "xiaowan-full.md#5",
                    "xiaowan-full.md#28",
                ],
            ),
            (
                "Q5",
                &[
                    "chaofeng-full.md#2",
                    "chaofeng-full.md#3",
                    "chaofeng-full.md#41",
                    "xiaowan.md#10",
                    "chaofeng-full.md#0",
                    "xiaowan.md#22",
                    "chaofeng-full.md#42",
                    "xiaowan-full.md#23",
                    "xiaowan.md#19",
                    "xiaowan-full.md#7",
                ],
            ),
            (
                "Q6",
                &[
                    "xiaowan-full.md#7",
                    "xiaowan.md#15",
                    "xiaowan.md#19",
                    "xiaowan-full.md#0",
                    "xiaowan.md#6",
                    "chaofeng-full.md#0",
                    "chaofeng-full.md#15",
                    "chaofeng-full.md#4",
                    "chaofeng-full.md#14",
                    "chaofeng-full.md#27",
                ],
            ),
        ];
        let mut total = 0.0;
        for (qid, want) in expected {
            let legs = legs_for(&fixture, qid);
            let got = quota_fuse(&legs, 60.0, 5, 10, group_of);
            let want: Vec<String> = want.iter().map(|s| s.to_string()).collect();
            assert_eq!(got, want, "A1 top-10 mismatch for {qid}");
            let score = score_top10(&got, qid);
            if qid != "Q4" {
                total += score;
            }
        }
        assert_eq!(total, 4.0, "A1 cross-file retrieval must be 4.0/5");
    }

    #[test]
    fn plain_rrf_matches_python_baseline() {
        let fixture = fixture();
        let want_scores = [
            ("Q1", 1.0),
            ("Q2", 0.5),
            ("Q3", 1.0),
            ("Q4", 1.0),
            ("Q5", 0.5),
            ("Q6", 0.5),
        ];
        let mut total = 0.0;
        for (qid, want) in want_scores {
            let legs = legs_for(&fixture, qid);
            let got = plain_rrf(&legs, 60.0, 10);
            let score = score_top10(&got, qid);
            assert_eq!(score, want, "plain RRF score mismatch for {qid}");
            if qid != "Q4" {
                total += score;
            }
        }
        assert_eq!(total, 3.5, "plain RRF cross-file retrieval must be 3.5/5");
    }

    #[test]
    fn two_leg_conan_sag_matches_fallback_baseline() {
        // A4: missing WeMM falls back to Conan + SAG2 two legs, retrieval 3.0/5.
        // No model download is involved: this fuses the fixture's stored
        // conan/sag top-50 lists only.
        let fixture = fixture();
        let want_scores = [
            ("Q1", 0.5),
            ("Q2", 1.0),
            ("Q3", 0.5),
            ("Q4", 1.0),
            ("Q5", 0.5),
            ("Q6", 0.5),
        ];
        let mut total = 0.0;
        for (qid, want) in want_scores {
            let lists = &fixture.questions[qid];
            let legs = vec![lists.conan.clone(), lists.sag.clone()];
            let got = quota_fuse(&legs, 60.0, 5, 10, group_of);
            let score = score_top10(&got, qid);
            assert_eq!(score, want, "two-leg (conan+sag) score mismatch for {qid}");
            if qid != "Q4" {
                total += score;
            }
        }
        assert_eq!(total, 3.0, "two-leg fallback retrieval must be 3.0/5");
    }

    #[test]
    fn missing_legs_preserve_remaining_fusion() {
        let config = crate::stage_config::StageConfig::default();
        let fuse = |legs: &[Vec<i64>]| quota_fuse(legs, config.rrf_k as f64, config.quota_per_author, config.top_k_final, |id| *id);
        let remaining = vec![vec![1, 2], vec![2, 3]];
        assert_eq!(fuse(&[vec![], vec![1, 2], vec![2, 3]]), fuse(&remaining));
        assert!(fuse(&[vec![], vec![]]).is_empty());
    }

    #[test]
    fn fusion_edge_cases() {
        let empty: Vec<Vec<String>> = vec![];
        assert!(quota_fuse(&empty, 60.0, 5, 10, group_of).is_empty());
        assert!(plain_rrf(&empty, 60.0, 10).is_empty());
        let single = vec![vec!["a".to_string(), "b".to_string()]];
        assert_eq!(plain_rrf(&single, 60.0, 10), single[0]);
        assert_eq!(
            quota_fuse(&single, 60.0, 0, 10, group_of),
            Vec::<String>::new()
        );
        // A quota of 1 per group still keeps both groups represented.
        let legs = vec![
            vec!["chaofeng-full.md#1".to_string(), "xiaowan.md#1".to_string()],
            vec!["xiaowan.md#1".to_string(), "chaofeng-full.md#1".to_string()],
        ];
        let fused = quota_fuse(&legs, 60.0, 1, 10, group_of);
        assert_eq!(fused.len(), 2);
    }
}
