//! Hop-0 BM25 recall over a sag.sqlite index, without Jev.
//!
//! Uses the exact SAG2 hop-0 path from sag-core: `text::match_query` for the
//! FTS5 MATCH expression and `Store::candidate_query` (sag_bm25, weights
//! title 2 / heading 2 / body 1, limit 10). No Jev scoring, no expansion,
//! no DPP. Content snippets are sliced from the source file by the chunk's
//! byte ranges (falling back to line numbers).
//!
//! Usage:
//!   bm25_hop0 <path-to-sag.sqlite> <questions.json> <output.json>
//! questions.json: [{"id": "Q1", "query": "..."}, ...]

use anyhow::{Context, Result};
use sag_core::store::Store;
use sag_core::text;
use serde::Serialize;
use std::fs;
use std::path::Path;

#[derive(Serialize)]
struct Hit {
    chunk_id: i64,
    bm25: f64,
    path: String,
    heading: String,
    start_line: usize,
    end_line: usize,
    snippet: String,
}

#[derive(Serialize)]
struct QuestionResult {
    id: String,
    query: String,
    match_expression: String,
    hits: Vec<Hit>,
}

fn snippet_for(path: &str, ranges: &[(usize, usize)], start_line: usize, end_line: usize) -> String {
    let raw = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return String::new(),
    };
    let mut parts: Vec<&str> = vec![];
    for &(s, e) in ranges {
        if let Some(piece) = raw.get(s..e) {
            parts.push(piece);
        }
    }
    if !parts.is_empty() {
        let joined = parts.join("");
        let flat: String = joined.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut chars = flat.chars();
        let short: String = chars.by_ref().take(240).collect();
        return if flat.chars().count() > 240 {
            format!("{short}…")
        } else {
            short
        };
    }
    // Fallback: slice by line numbers.
    let lines: Vec<&str> = raw.lines().collect();
    let lo = start_line.saturating_sub(1);
    let hi = end_line.min(lines.len());
    if lo < hi {
        lines[lo..hi].join(" ")
    } else {
        String::new()
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let db = args.get(1).context("usage: bm25_hop0 <sag.sqlite> <questions.json> <output.json>")?;
    let qfile = args.get(2).context("missing questions.json")?;
    let ofile = args.get(3).context("missing output.json")?;
    let store = Store::open(Path::new(db))?;
    let questions: Vec<serde_json::Value> = serde_json::from_str(&fs::read_to_string(qfile)?)?;
    let mut out: Vec<QuestionResult> = vec![];
    for q in &questions {
        let id = q["id"].as_str().unwrap_or("?").to_string();
        let query = q["query"].as_str().unwrap_or("").to_string();
        let expr = text::match_query(&query);
        let cands = store.candidate_query(&expr, None, &[], &[], &[], 10)?;
        let mut hits: Vec<Hit> = vec![];
        for c in &cands {
            hits.push(Hit {
                chunk_id: c.id,
                bm25: c.bm25,
                path: c.path.clone(),
                heading: c.heading.clone(),
                start_line: c.start_line,
                end_line: c.end_line,
                snippet: snippet_for(&c.path, &c.source_ranges, c.start_line, c.end_line),
            });
        }
        println!("{}: expr={} hits={}", id, expr, hits.len());
        out.push(QuestionResult { id, query, match_expression: expr, hits });
    }
    fs::write(ofile, serde_json::to_string_pretty(&out)?)?;
    println!("wrote {}", ofile);
    Ok(())
}
