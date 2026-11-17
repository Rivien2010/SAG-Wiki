mod support;
use std::process::Command;

#[test]
fn test_retrieval_json_dispatches_all_overrides_and_prints_valid_json() {
    let data = tempfile::tempdir().unwrap();
    let expected = serde_json::json!({
        "options": sag_protocol::types::Retrieval::default(),
        "notes": ["Q4 is excluded from retrieval total /5"],
        "questions": [{"id":"Q1", "question":"fixture", "retrieval_score":0.5,
            "evidence":["chaofeng-full.md#10"], "generation":"skipped (no key)",
            "generation_score":null,"valid_citations":0,"invalid_citations":0,
            "diagnosis":"recall-miss", "advice":"check parsing/chunking/parameters"}],
        "retrieval_total":0.5,"retrieval_max":5,"generation_total":null,"generation_status":"unverified"
    });
    let reply = expected.clone();
    let server = support::serve(data.path(), 1, move |request| {
        let command = &request["command"];
        assert_eq!(command["command"], "test_retrieval");
        assert_eq!(command["mode"], "accurate");
        assert_eq!(command["top_k"], 7);
        assert_eq!(command["keyword_weight"], 0.6);
        assert_eq!(command["vector_weight"], 0.4);
        assert_eq!(command["threshold"], 0.1);
        assert_eq!(command["pool"], 80);
        serde_json::json!({"type":"retrieval_panel","data":reply})
    });
    let output = Command::new(env!("CARGO_BIN_EXE_sag")).env("SAG_DATA_DIR", data.path())
        .args(["test-retrieval", "--mode", "accurate", "--top-k", "7", "--keyword-weight", "0.6",
            "--vector-weight", "0.4", "--threshold", "0.1", "--pool", "80", "--json"]).output().unwrap();
    server.join().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(), expected);
}

#[test]
fn current_store_panel_runs_without_generation_key_from_unindexed_cwd() {
    use sag_protocol::types::ModelAuthMode;
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("data");
    let files = temp.path().join("indexed");
    let cwd = temp.path().join("unindexed-cwd");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&files).unwrap();
    std::fs::write(files.join("unrelated.txt"), "galaxy nebula telescope astronomy").unwrap();
    let store = sag_core::store::Store::open(&data.join("sag.sqlite")).unwrap();
    let mut settings = store.settings().unwrap();
    settings.jev.auth_mode = ModelAuthMode::None;
    // A request here would fail: unrelated corpus must not need remote scoring.
    settings.jev.url = "http://127.0.0.1:9/v1".into();
    store.save_settings(&settings).unwrap();
    let root = store.add_root(&files).unwrap();
    store.scan(&root, |_| {}).unwrap();
    drop(store);
    let _cleanup = support::ServiceCleanup(data.clone());
    // Service startup queues a readiness scan even for an existing store.
    // Add waits for that fixture scan before the read-only panel request.
    let prepared = Command::new(env!("CARGO_BIN_EXE_sag"))
        .env("SAG_DATA_DIR", &data).env_remove("MINIMAX_API_KEY")
        .current_dir(&cwd).arg("add").arg(&files).arg("--json").output().unwrap();
    assert!(prepared.status.success(), "{}", String::from_utf8_lossy(&prepared.stderr));
    let output = Command::new(env!("CARGO_BIN_EXE_sag"))
        .env("SAG_DATA_DIR", &data).env_remove("MINIMAX_API_KEY")
        .current_dir(&cwd).args(["test-retrieval", "--json"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["questions"].as_array().unwrap().len(), 6);
    assert_eq!(parsed["retrieval_total"], 0.0);
    assert_eq!(parsed["generation_status"], "unverified");
    assert!(parsed["generation_total"].is_null());
    for q in parsed["questions"].as_array().unwrap() {
        assert_eq!(q["generation"], "skipped (no key)");
        assert!(q["generation_score"].is_null());
        assert_eq!(q["diagnosis"], "recall-miss");
    }
}
