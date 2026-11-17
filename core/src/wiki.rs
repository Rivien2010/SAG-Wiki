//! Agent-generated Wiki pages (Stage 2, 9.1-9.2).
//!
//! A Wiki page is a Markdown note the Agent synthesises from retrieved
//! evidence (title, sections, citations), saved under the notes folder and
//! indexed immediately, so it is searchable the moment it lands. Pages link
//! to each other with `[[Title]]`; unknown titles stay plain text.
//! Saving reuses `Store::write_note` (save + immediate index); deleting
//! removes the file, the index entry and the note rows.
use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

use crate::models::{self, AnswerPlan};
use crate::notes::{self, NoteHeader, NoteSource};
use crate::store::{now, Store};
use crate::types::{Candidate, Document, ModelConfig};

/// System prompt for Wiki generation. Citations point at the numbered
/// sources; `[[Title]]` links only to the listed existing pages.
pub const WIKI_SYSTEM: &str = "你是本地知识库的 Wiki 编辑。根据本次提供的来源，为给定主题写一篇结构化 Wiki 页：以 `# 标题` 开头，分小节组织，每个事实后用 [1]、[2] 等引用来源编号，引用必须指向真正支撑该事实的来源。不要把一个来源的内容安到另一个来源头上；不要引入来源中没有出现的人名、地名、数字。相关已有 Wiki 页用 [[标题]] 链接，只链接下面列出的已有页面，不要编造不存在的页面。不知道的就说不知道，不要编答案。证据不足时明确说明。使用用户的语言。只输出页面 Markdown，不要附加解释。";

/// Build the Wiki generation plan for `models::answer`: budget-aware
/// evidence selection (same rule as `prepare_answer`), Wiki system prompt,
/// topic + numbered sources + linkable existing page titles.
pub fn prepare_wiki_page(
    config: &ModelConfig,
    topic: &str,
    evidence: &[Candidate],
    existing_pages: &[String],
    limit: usize,
    output: usize,
) -> Result<AnswerPlan> {
    let budget = models::input_budget(limit, output)?;
    let mut selected: Vec<Candidate> = vec![];
    for candidate in evidence {
        let mut proposed = selected.clone();
        proposed.push(candidate.clone());
        if models::message_tokens(&wiki_messages(topic, &proposed, existing_pages)) <= budget {
            selected = proposed;
        }
    }
    if !evidence.is_empty() && selected.is_empty() {
        anyhow::bail!("单个检索片段超过回答上下文预算，请降低片段最大长度或提高上下文上限");
    }
    let messages = wiki_messages(topic, &selected, existing_pages);
    let estimated = models::message_tokens(&messages);
    let omitted_sources = evidence.len() - selected.len();
    let payload = models::payload(config, messages, true)?;
    Ok(AnswerPlan {
        payload,
        evidence: selected,
        context: crate::types::AnswerContext {
            limit,
            reserved_output: output,
            estimated_input: estimated,
            omitted_sources,
            omitted_history: 0,
            input_tokens: None,
            output_tokens: None,
            elapsed_ms: None,
            generation_ms: None,
            summary: None,
        },
    })
}

fn wiki_messages(topic: &str, evidence: &[Candidate], existing_pages: &[String]) -> Vec<Value> {
    let sources = evidence
        .iter()
        .enumerate()
        .map(|(i, c)| json!({"id": i + 1, "title": c.title, "path": c.path, "heading": c.heading, "content": c.content}))
        .collect::<Vec<_>>();
    vec![
        json!({"role": "system", "content": WIKI_SYSTEM}),
        json!({"role": "user", "content": json!({"topic": topic, "sources": sources, "existing_pages": existing_pages}).to_string()}),
    ]
}

/// Sanitise a topic into a file name: keep letters/digits/CJK, collapse the
/// rest to `-`, cap at 80 chars, fall back to `wiki-page`.
pub fn wiki_filename(topic: &str) -> String {
    let mut out = String::new();
    for ch in topic.chars() {
        if ch.is_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    let mut short: String = trimmed.chars().take(80).collect();
    while short.ends_with('-') { short.pop(); }
    if short.is_empty() { short.push_str("wiki-page"); }
    format!("{short}.md")
}

/// Titles of existing Wiki pages (notes) in the store, for `[[Title]]` links.
pub fn existing_page_titles(store: &Store) -> Result<Vec<String>> {
    Ok(store.notes(None)?.into_iter().map(|n| n.title).filter(|t| !t.is_empty()).collect())
}

/// Resolve `[[Title]]` links: known titles become relative Markdown links
/// to the note file; unknown titles become plain text (no dead links).
pub fn resolve_wiki_links(store: &Store, note_dir: &Path, body: &str) -> Result<String> {
    let notes = store.notes(None)?;
    let mut out = String::new();
    let mut rest = body;
    while let Some(start) = rest.find("[[") {
        let (before, after) = rest.split_at(start);
        out.push_str(before);
        let Some(end) = after.find("]]") else { break };
        let title = &after[2..end];
        match notes.iter().find(|n| n.title == title || n.question == title) {
            Some(note) => {
                let rel = notes::relative_path(note_dir, Path::new(&note.path));
                out.push_str(&format!("[{title}]({rel})"));
            }
            None => out.push_str(title),
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Save a generated Wiki page body as a note (frontmatter + citations +
/// resolved links) and index it immediately. Returns the indexed document.
pub fn save_wiki_page(
    store: &Store,
    note_dir: &Path,
    topic: &str,
    body: &str,
    evidence: &[Candidate],
    model: &str,
    scope: Vec<String>,
) -> Result<Document> {
    let (answer, kept) = notes::renumber_citations(body, evidence);
    let sources: Vec<NoteSource> = kept
        .iter()
        .enumerate()
        .map(|(i, candidate)| notes::source_from_candidate(i + 1, candidate, None))
        .collect();
    let header = NoteHeader { question: topic.to_string(), asked_at: Some(now()), model: model.to_string(), scope, sources };
    let resolved = resolve_wiki_links(store, note_dir, &answer)?;
    let rendered = notes::render(&header, note_dir, &resolved, false);
    let target = note_dir.join(wiki_filename(topic));
    store.write_note(&target, &rendered, None)
}

/// Generate a Wiki page with the model and save it (indexed immediately).
/// The caller supplies retrieved evidence; the model call needs a key.
pub async fn generate_wiki_page(
    client: &Client,
    config: &ModelConfig,
    key: &str,
    store: &Store,
    note_dir: &Path,
    topic: &str,
    evidence: &[Candidate],
    scope: Vec<String>,
    limit: usize,
    output: usize,
    timeout: u64,
    cancel: &CancellationToken,
) -> Result<Document> {
    let existing = existing_page_titles(store)?;
    let plan = prepare_wiki_page(config, topic, evidence, &existing, limit, output)?;
    let result = models::answer(client, config, key, &plan, timeout, cancel, |_| {}).await?;
    save_wiki_page(store, note_dir, topic, &result.text, &plan.evidence, &config.model, scope)
}

/// Delete a Wiki page: remove the file, drop its index entry and note rows.
/// Missing files are tolerated (index cleanup still runs).
pub fn delete_wiki_page(store: &Store, path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| format!("删除 Wiki 页失败：{}", path.display())),
    }
    let path_text = crate::store::path_text(path)?;
    store.remove_indexed(&path_text)?;
    Ok(())
}

/// Note file path for a topic inside the notes folder.
pub fn wiki_path(note_dir: &Path, topic: &str) -> PathBuf {
    note_dir.join(wiki_filename(topic))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;
    use std::fs;

    fn candidate(id: i64, title: &str, content: &str) -> Candidate {
        Candidate {
            id,
            document_id: id,
            path: format!("/kb/{title}.md"),
            title: title.into(),
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

    fn setup() -> (tempfile::TempDir, Store, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("kb");
        fs::create_dir_all(root.join("notes")).unwrap();
        let store = Store::open(&dir.path().join("app/db.sqlite")).unwrap();
        store.add_root(&root).unwrap();
        (dir, store, root.canonicalize().unwrap())
    }

    #[test]
    fn wiki_plan_requests_streaming() {
        let config = ModelConfig::default();
        let evidence = vec![candidate(1, "one", "Alice owns Acme.")];
        let plan = prepare_wiki_page(&config, "Acme", &evidence, &[], 16_384, 8000).unwrap();
        assert_eq!(plan.payload["stream"], json!(true));
    }

    #[test]
    fn wiki_plan_carries_system_prompt_topic_sources_and_links() {
        let config = ModelConfig::default();
        let evidence = vec![candidate(1, "one", "Alice owns Acme."), candidate(2, "two", "Bob owns Beta.")];
        let plan = prepare_wiki_page(&config, "Acme", &evidence, &["Beta".to_string()], 16_384, 8000).unwrap();
        assert_eq!(plan.evidence.len(), 2);
        let payload = plan.payload.to_string();
        assert!(payload.contains("Wiki"), "plan must use the Wiki system prompt");
        assert!(payload.contains("Acme"), "plan must carry the topic");
        assert!(payload.contains("Alice owns Acme"), "plan must carry the evidence");
        assert!(payload.contains("Beta"), "plan must list linkable pages");
        assert_eq!(plan.payload["model"], config.model);
    }

    #[test]
    fn wiki_filename_sanitises_topics() {
        assert_eq!(wiki_filename("Acme 公司介绍"), "Acme-公司介绍.md");
        assert_eq!(wiki_filename("  a/b: c?  "), "a-b-c.md");
        assert_eq!(wiki_filename("!!!"), "wiki-page.md");
        assert!(!wiki_filename(&"x".repeat(200)).contains(' '));
    }

    #[test]
    fn wiki_links_resolve_known_titles_and_drop_unknown_ones() {
        let (_dir, store, root) = setup();
        let note_dir = root.join("notes");
        // Save one page, then link to it (and to a missing one) from new text.
        let page = save_wiki_page(&store, &note_dir, "Beta", "# Beta\n\nBeta is a company [1].", &[candidate(2, "two", "Bob owns Beta.")], "m", vec![]).unwrap();
        assert!(page.path.ends_with("Beta.md"));
        let resolved = resolve_wiki_links(&store, &note_dir, "See [[Beta]] and [[Missing]] for details.").unwrap();
        assert!(resolved.contains("[Beta](Beta.md)"), "known title becomes a relative link: {resolved}");
        assert!(!resolved.contains("[[Missing]]"), "unknown title loses its brackets: {resolved}");
        assert!(resolved.contains("Missing"), "unknown title stays as plain text");
    }

    #[test]
    fn saved_wiki_page_is_searchable_immediately_and_deletable() {
        let (_dir, store, root) = setup();
        let note_dir = root.join("notes");
        let evidence = vec![candidate(1, "one", "Alice owns Acme.")];
        let page = save_wiki_page(&store, &note_dir, "Acme", "# Acme\n\nAlice owns Acme [1].", &evidence, "m", vec![]).unwrap();
        assert!(Path::new(&page.path).exists());
        // 9.2: searchable immediately (no rescan needed). The note file is
        // .md, so Hierarchy chunking (10.5 default) splits it into a
        // frontmatter chunk and a body chunk — both contain the evidence
        // sentence, so the page is found twice; every hit must be the page.
        let found = store.literal_candidates("Alice owns Acme", &[], &[], 10).unwrap();
        assert!(!found.is_empty(), "saved Wiki page must be retrievable at once");
        assert!(found.iter().all(|c| c.path == page.path));
        assert_eq!(store.notes(None).unwrap().len(), 1);
        // 9.1: delete removes file + index + note rows.
        delete_wiki_page(&store, Path::new(&page.path)).unwrap();
        assert!(!Path::new(&page.path).exists());
        assert!(store.literal_candidates("Alice owns Acme", &[], &[], 10).unwrap().is_empty());
        assert!(store.notes(None).unwrap().is_empty());
        // Deleting again is tolerated.
        delete_wiki_page(&store, Path::new(&page.path)).unwrap();
    }
}
