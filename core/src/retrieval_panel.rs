//! Six-question reference panel. Production runs read the current store only.
use crate::{models, store::Store, types::{Candidate, ModelConfig, Retrieval, RetrievalMode, Settings}};
use anyhow::{bail, Result};
use std::{collections::{HashMap, HashSet}, path::Path};
use tokio_util::sync::CancellationToken;
pub use sag_protocol::retrieval_panel::*;

pub const QUESTIONS: [(&str, &str); 6] = [
    ("Q1", "两位作者的发帖排期分别是什么？能直接照搬对方的吗？"),
    ("Q2", "两位作者的自动化工具栈分别是什么？哪边用了飞书多维表格？"),
    ("Q3", "两位作者怎么做复盘？阶段和做法有何不同？"),
    ("Q4", "超级峰教程里“每篇笔记发布后效果不好，大概率需要知道以下 5 点”列了哪 5 点？"),
    ("Q5", "两位作者各自的起号数据是多少？超级峰那篇的时间口径一致吗？"),
    ("Q6", "两位作者对“照抄爆款、照搬别人的做法”分别是什么态度？"),
];

const VARIANTS: [(&str, &[&str]); 1] = [("Q6", &["两位作者对复制模板、复制别人的做法分别是什么态度？"])];

trait PanelOptionsExt {
    fn apply(&self, base: &Retrieval) -> Result<Retrieval>;
}
// Scoring ported from the Python reference (hybrid-rag-quota-fusion.py).
// Retrieval is scored out of 5 (Q4 excluded); a cross-file question
// scores 1.0 when both sides have evidence in the top-10, 0.5 for one
// side only, 0.0 for none. Q4 scores 1.0 when any support block is hit.
fn prefix_of(fname: &str) -> &'static str {
    match fname {
        "chaofeng-full.md" => "cjf",
        "xiaowan-full.md" => "xwb",
        "xiaowan.md" => "xw",
        _ => "?",
    }
}

fn evidence(qid: &str) -> HashMap<&'static str, HashSet<i64>> {
    let mut map: HashMap<&'static str, HashSet<i64>> = HashMap::new();
    let mut add = |key: &'static str, nums: &[i64]| {
        map.insert(key, nums.iter().copied().collect());
    };
    match qid {
        "Q1" => {
            add("cjf", &[5, 7, 10, 12, 23, 24]);
            add("xwb", &[4, 7]);
            add("xw", &[1, 3, 28]);
        }
        "Q2" => {
            add("cjf", &[10, 16, 17, 34, 35]);
            add(
                "xwb",
                &[0, 1, 13, 28, 29, 30, 31, 32, 33, 36, 39, 42, 43, 48],
            );
            add("xw", &[0, 1, 24, 25, 29]);
        }
        "Q3" => {
            add("cjf", &[14, 15, 22, 23]);
            add("xwb", &[5, 6]);
            add("xw", &[4, 5]);
        }
        "Q5" => {
            add("cjf", &[2, 3]);
            add("xwb", &[1]);
            add("xw", &[1]);
        }
        "Q6" => {
            add("cjf", &[7, 21, 45]);
            add("xwb", &[23]);
            add("xw", &[19]);
        }
        _ => {}
    }
    map
}

fn side_hit(top10: &[String], qid: &str, side: &str) -> bool {
    let ev = evidence(qid);
    let prefixes: &[&str] = match side {
        "cjf" => &["cjf"],
        "xw" => &["xwb", "xw"],
        _ => &[],
    };
    for cid in top10 {
        let (fname, num) = cid.rsplit_once('#').unwrap();
        let prefix = prefix_of(fname);
        if prefixes.contains(&prefix) {
            if let Ok(n) = num.parse::<i64>() {
                if ev.get(prefix).is_some_and(|set| set.contains(&n)) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn score_top10(top10: &[String], qid: &str) -> f64 {
    if qid == "Q4" {
        let support: HashSet<&str> = [
            "chaofeng-full.md#18",
            "chaofeng-full.md#44",
            "chaofeng-full.md#45",
            "xiaowan-full.md#10",
        ]
        .into_iter()
        .collect();
        return if top10.iter().any(|c| support.contains(c.as_str())) {
            1.0
        } else {
            0.0
        };
    }
    let cjf = side_hit(top10, qid, "cjf");
    let xw = side_hit(top10, qid, "xw");
    if cjf && xw {
        1.0
    } else if cjf || xw {
        0.5
    } else {
        0.0
    }
}


impl PanelOptionsExt for PanelOptions {
    fn apply(&self, base: &Retrieval) -> Result<Retrieval> {
        let mut r = base.clone();
        if let Some(v) = self.mode { r.mode = v; }
        if let Some(v) = self.top_k { r.top_k_final = v; }
        r.top_k = r.top_k_final;
        if let Some(v) = self.keyword_weight { r.keyword_weight = v; }
        if let Some(v) = self.vector_weight { r.vector_weight = v; }
        if let Some(v) = self.threshold { r.similarity_threshold = v; }
        if let Some(v) = self.pool { r.candidate_k = v; }
        if !(1..=100).contains(&r.top_k_final) || !(1..=10000).contains(&r.candidate_k) {
            bail!("top-k must be 1..100; pool must be 1..10000");
        }
        crate::retrieval::validate_knobs(&crate::stage_config::StageConfig::from_retrieval(&r))?;
        Ok(r)
    }
}

/// The reference uses zero-based chunk positions within each document, never
/// SQLite global IDs or Candidate::order (which is the retrieval rank).
fn reference_ids(store: &Store, rows: &[Candidate]) -> Result<Vec<String>> {
    let db = store.connection();
    rows.iter().map(|row| {
        let ordinal: i64 = db.query_row(
            "SELECT count(*) FROM chunks WHERE document_id=?1 AND id<?2",
            rusqlite::params![row.document_id, row.id], |r| r.get(0))?;
        let name = Path::new(&row.path).file_name().unwrap_or_default().to_string_lossy();
        Ok(format!("{name}#{ordinal}"))
    }).collect()
}

fn question_result(id: &str, question: &str, mut evidence: Vec<String>) -> QuestionResult {
    evidence.truncate(10);
    let score = score_top10(&evidence, id);
    let mut result = QuestionResult {
        id: id.into(), question: question.into(), retrieval_score: score, evidence,
        generation: "skipped (no key)".into(), generation_score: None,
        valid_citations: 0, invalid_citations: 0, diagnosis: String::new(), advice: String::new(),
    };
    diagnose(&mut result);
    result
}
fn diagnose(r: &mut QuestionResult) {
    let (diagnosis, advice) = if r.retrieval_score < 1.0 {
        ("recall-miss", "check parsing/chunking/parameters")
    } else { match r.generation_score {
        Some(1) => ("recall-ok-answer-ok", "citation/evidence checks passed; factual correctness 未验证"),
        Some(_) => ("recall-ok-answer-bad", "check model/prompt"),
        None => ("recall-ok-answer-unverified", "generation 未验证"),
    }};
    r.diagnosis = diagnosis.into(); r.advice = advice.into();
}

fn evaluate_answer(store: &Store, r: &mut QuestionResult, text: &str, evidence: &[Candidate]) -> Result<()> {
    let checks = models::check_citations(text, evidence);
    let mut cited = Vec::new();
    for check in checks {
        if let models::CitationStatus::Valid { source_number, .. } = check.status {
            r.valid_citations += 1;
            cited.push(evidence[source_number - 1].clone());
        } else { r.invalid_citations += 1; }
    }
    let refused = ["不知道", "无法回答", "无法确定", "不能确定", "不足以", "未提供", "没有足够", "无法列出", "cannot answer", "don't know", "insufficient"]
        .iter().any(|s| text.to_lowercase().contains(s));
    let supported = score_top10(&reference_ids(store, &cited)?, &r.id) == 1.0;
    r.generation_score = Some(u8::from(!text.trim().is_empty() && !refused && supported
        && r.valid_citations > 0 && r.invalid_citations == 0));
    r.generation = "completed (mechanical citation/evidence checks; factual correctness 未验证)".into();
    diagnose(r);
    Ok(())
}

fn output(options: Retrieval, questions: Vec<QuestionResult>, notes: Vec<String>) -> PanelOutput {
    let generation_total = questions.iter().map(|q| q.generation_score).collect::<Option<Vec<_>>>()
        .map(|v| v.into_iter().sum());
    PanelOutput {
        retrieval_total: questions.iter().filter(|q| q.id != "Q4").map(|q| q.retrieval_score).sum(),
        retrieval_max: 5, generation_status: if generation_total.is_some() { "mechanical checks only; factual correctness 未验证" } else { "unverified" }.into(),
        generation_total, options, questions, notes,
    }
}

pub async fn run(store: &Store, client: &reqwest::Client, data: &Path, settings: &Settings,
    scope: &[String], overrides: &PanelOptions, cancel: &CancellationToken) -> Result<PanelOutput> {
    let options = overrides.apply(&settings.retrieval)?;
    let env_key = overrides.minimax_key.clone().or_else(|| std::env::var("MINIMAX_API_KEY").ok()).filter(|k| !k.trim().is_empty());
    let has_env_key = env_key.is_some();
    let saved_minimax = settings.answer.model.starts_with("MiniMax-");
    let key = if let Some(key) = env_key { Some(key) }
        else if saved_minimax && models::has_secret(data, "answer")? { Some(models::secret(data, "answer")?) }
        else { None };
    // Environment credentials go only to the official endpoint. A configured
    // MiniMax provider uses its own saved credential and endpoint instead.
    let model = if has_env_key || !saved_minimax {
        ModelConfig { model: "MiniMax-M3".into(), url: "https://api.minimaxi.com/v1".into(), ..Default::default() }
    } else { settings.answer.clone() };
    let jev_key = models::jev_secret(data, &settings.jev)?;
    let mut notes = vec![
        "Q4 is a hallucination trap, scored separately and excluded from retrieval total /5.".into(),
        "Retrieval checks the first min(top-k,10) results. Evidence identity is filename#zero-based indexed chunk position; reference corpus/chunk alignment 未验证. Missing reference documents score zero.".into(),
        "Generation scores are mechanical citation/evidence checks (0/1), not semantic truth verification. Factual correctness 未验证; Q4 refusal scores 0 under this rubric.".into(),
    ];
    if options.mode == RetrievalMode::Fast { notes.push("fast: SAG2 only; vector weight has no active vector leg.".into()); }
    let mut questions = Vec::new();
    for (id, query) in QUESTIONS {
        if cancel.is_cancelled() { bail!("panel cancelled"); }
        let mut q_options = options.clone();
        q_options.variant_queries = VARIANTS.iter().find(|(qid, _)| *qid == id)
            .map(|(_, variants)| variants.iter().map(|s| (*s).to_owned()).collect())
            .unwrap_or_default();
        let vectors = if options.mode == RetrievalMode::Accurate && options.legs.iter().any(|l| l == "conan") {
            let hits = crate::embedding::conan_leg(store, query, "main", None);
            if hits.is_empty() { notes.push(format!("{id}: Conan unavailable or no indexed vectors; SAG2 fallback; no download.")); }
            vec![hits.into_iter().map(|(id, _)| id).collect()]
        } else { vec![] };
        if options.mode == RetrievalMode::Accurate && options.legs.iter().any(|l| l == "wemm") && questions.is_empty() {
            notes.push("WeMM unavailable in Stage 1; no download.".into());
        }
        let found = crate::retrieval::retrieve_fused(store, query, scope, &[], &q_options, &vectors, cancel,
            |rows, evidence| {
                let key = &jev_key;
                async move { models::score_with_runtime(client, &settings.jev, key, query,
                    rows, &evidence, settings.timeout_secs, &settings.runtime, cancel).await }
            }, |_, _| {}).await.map_err(|_| anyhow::anyhow!("{id}: retrieval failed; check JEV configuration/index/parameters (provider details omitted)"))?;
        let mut result = question_result(id, query, reference_ids(store, &found.results)?);
        if let Some(key) = &key {
            let generated = async {
                let plan = models::prepare_answer(&model, query, &found.results, &[], settings.answer_context_tokens, 8000)?;
                let answer = models::answer(client, &model, key, &plan, settings.timeout_secs, cancel, |_| {}).await?;
                evaluate_answer(store, &mut result, &answer.text, &plan.evidence)
            }.await;
            if generated.is_err() {
                if cancel.is_cancelled() { bail!("panel cancelled"); }
                // Never echo remote errors: provider responses can contain secrets.
                result.generation = "failed (generation request or context error)".into();
                result.generation_score = Some(0);
                diagnose(&mut result);
            }
        }
        questions.push(result);
    }
    Ok(output(options, questions, notes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::quota_fuse;

    fn fixture_store() -> (tempfile::TempDir, Store, HashMap<String, Candidate>) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("fixture.sqlite")).unwrap();
        let fixture: serde_json::Value = serde_json::from_str(include_str!("embedding_fixture.json")).unwrap();
        let mut candidates = HashMap::new();
        let mut db = store.connection();
        let tx = db.transaction().unwrap();
        for name in ["xiaowan.md", "chaofeng-full.md", "xiaowan-full.md"] {
            let path = dir.path().join(name).to_string_lossy().into_owned();
            tx.execute("INSERT INTO documents(root,path,name,hash,mtime) VALUES (?1,?2,?3,'fixture',0)",
                rusqlite::params![dir.path().to_str().unwrap(), path, name]).unwrap();
            let document_id = tx.last_insert_rowid();
            for reference in fixture["chunks"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).filter(|s| s.starts_with(&format!("{name}#"))) {
                tx.execute("INSERT INTO chunks(document_id,heading,ranges,start_line,end_line) VALUES (?1,'','[]',1,1)", [document_id]).unwrap();
                let id = tx.last_insert_rowid();
                candidates.insert(reference.into(), Candidate {
                    id, document_id, path: path.clone(), title: name.into(), heading: String::new(),
                    content: reference.into(), start_line: 1, end_line: 1, source_ranges: vec![], source_spans: vec![],
                    bm25: 0., score: 1., previous_score: None, order: 999, hash: "fixture".into(),
                });
            }
        }
        tx.commit().unwrap();
        drop(db);
        (dir, store, candidates)
    }

    #[test]
    fn fixture_store_panel_matches_harness_four_out_of_five() {
        let (_dir, store, candidates) = fixture_store();
        let fixture: serde_json::Value = serde_json::from_str(include_str!("fusion_fixture.json")).unwrap();
        let expected = [1., 1., 1., 1., 0.5, 0.5];
        let mut questions = vec![];
        for ((qid, query), want) in QUESTIONS.into_iter().zip(expected) {
            let legs: Vec<Vec<String>> = ["wemm", "conan", "sag"].into_iter().map(|leg|
                serde_json::from_value(fixture["questions"][qid][leg].clone()).unwrap()).collect();
            let top = quota_fuse(&legs, 60., 5, 10, |id| if id.starts_with("chaofeng") { "cjf" } else { "xw" });
            let rows: Vec<_> = top.iter().map(|id| candidates[id].clone()).collect();
            let ids = reference_ids(&store, &rows).unwrap();
            assert_eq!(ids, top, "store identity must preserve fixture chunk positions");
            let result = question_result(qid, query, ids);
            assert_eq!(result.retrieval_score, want, "{qid}");
            assert_eq!(result.generation, "skipped (no key)");
            assert_eq!(result.generation_score, None);
            questions.push(result);
        }
        let result = output(Retrieval::default(), questions, vec![]);
        assert_eq!(result.retrieval_total, 4.);
        assert_eq!(result.retrieval_max, 5);
        assert_eq!(result.generation_total, None);
        assert_eq!(result.generation_status, "unverified");
    }

    #[test]
    fn citations_and_diagnosis_separate_recall_from_generation() {
        let (_dir, store, rows) = fixture_store();
        let evidence = vec![rows["chaofeng-full.md#10"].clone(), rows["xiaowan-full.md#7"].clone()];
        let ids = reference_ids(&store, &evidence).unwrap();
        for (answer, score, valid, invalid, diagnosis) in [
            ("排期说明。[1][2]", 1, 2, 0, "recall-ok-answer-ok"),
            ("排期说明。[1][99]", 0, 1, 1, "recall-ok-answer-bad"),
            ("排期说明。", 0, 0, 0, "recall-ok-answer-bad"),
            ("不知道。[1][2]", 0, 2, 0, "recall-ok-answer-bad"),
            ("只引一方。[1]", 0, 1, 0, "recall-ok-answer-bad"),
        ] {
            let mut result = question_result("Q1", QUESTIONS[0].1, ids.clone());
            evaluate_answer(&store, &mut result, answer, &evidence).unwrap();
            assert_eq!((result.generation_score, result.valid_citations, result.invalid_citations), (Some(score), valid, invalid));
            assert_eq!(result.diagnosis, diagnosis);
        }
        assert_eq!(question_result("Q1", "", vec![]).diagnosis, "recall-miss");
        assert_eq!(question_result("Q1", "", ids).diagnosis, "recall-ok-answer-unverified");
    }

    #[test]
    fn overrides_are_temporary_and_validated() {
        let base = Retrieval::default();
        let options = PanelOptions { mode: Some(RetrievalMode::Accurate), top_k: Some(3), keyword_weight: Some(0.6),
            vector_weight: Some(0.4), threshold: Some(0.1), pool: Some(80), minimax_key: None };
        let r = options.apply(&base).unwrap();
        assert_eq!((r.top_k, r.top_k_final, r.candidate_k), (3, 3, 80));
        assert_eq!((r.keyword_weight, r.vector_weight, r.similarity_threshold), (0.6, 0.4, 0.1));
        assert_eq!(r.mode, RetrievalMode::Accurate);
        assert_eq!(base.top_k_final, 10);
        for invalid in [PanelOptions { top_k: Some(0), ..Default::default() },
            PanelOptions { threshold: Some(f64::NAN), ..Default::default() },
            PanelOptions { keyword_weight: Some(-1.), ..Default::default() },
            PanelOptions { pool: Some(0), ..Default::default() }] {
            assert!(invalid.apply(&base).is_err());
        }
        let fixture: serde_json::Value = serde_json::from_str(include_str!("embedding_fixture.json")).unwrap();
        for ((id, question), reference) in QUESTIONS.iter().zip(fixture["questions"].as_array().unwrap()) {
            assert_eq!(*id, reference["id"].as_str().unwrap());
            assert_eq!(*question, reference["query"].as_str().unwrap());
        }
    }
}
