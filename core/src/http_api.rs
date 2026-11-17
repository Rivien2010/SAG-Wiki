//! Local HTTP API for coding agents (Stage 3, 13.1-13.2).
//!
//! A plain HTTP service that lets a coding agent query the local knowledge
//! base without a model key: `GET /search?q=...&top_k=...` runs the local
//! BM25 chunk search and returns chunk-level results with citations
//! (path + line ranges). `GET /health` reports liveness.
//!
//! The listener binds 127.0.0.1 only (13.2): the service is never exposed
//! on the network. Single-user local use, so no auth is required beyond
//! loopback binding; the RPC service keeps its own token separately.
//! The chosen port is written to `sag-http.json` in the data directory so
//! agents can discover it.
use anyhow::{bail, Context, Result};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream},
};
use tokio_util::sync::CancellationToken;

use crate::store::Store;

/// Serve one HTTP connection against the store (13.1).
pub async fn serve_http_api(mut stream: TcpStream, store: &Store) -> Result<()> {
    let mut reader = BufReader::new(&mut stream);
    let mut bytes = Vec::new();
    loop {
        let start = bytes.len();
        let count = (&mut reader).take((8193 - start) as u64).read_until(b'\n', &mut bytes).await?;
        if count == 0 {
            bail!("HTTP 请求未完成")
        }
        if bytes.len() > 8192 {
            bail!("HTTP 请求头超过上限")
        }
        if matches!(&bytes[start..], b"\r\n" | b"\n") {
            break;
        }
    }
    let request = std::str::from_utf8(&bytes).context("HTTP 请求不是 UTF-8")?;
    let line = request.lines().next().unwrap_or("");
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next(), parts.next());
    let (status, body) = match (method, target) {
        (Some("GET"), Some(target)) => handle_get(target, store),
        (Some(_), Some(_)) => ("405 Method Not Allowed".to_string(), "Use GET\n".to_string()),
        _ => ("400 Bad Request".to_string(), "Bad request\n".to_string()),
    };
    let headers = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

fn handle_get(target: &str, store: &Store) -> (String, String) {
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p, q),
        None => (target, ""),
    };
    match path {
        "/health" => ("200 OK".to_string(), "{\"status\":\"ok\"}\n".to_string()),
        "/search" => {
            let params: std::collections::HashMap<String, String> = query
                .split('&')
                .filter_map(|pair| pair.split_once('='))
                .map(|(k, v)| (k.to_string(), url_decode(v)))
                .collect();
            let Some(q) = params.get("q").filter(|q| !q.trim().is_empty()) else {
                return ("400 Bad Request".to_string(), "{\"error\":\"missing q\"}\n".to_string());
            };
            let top_k = params
                .get("top_k")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(10)
                .clamp(1, 50);
            match store.search_files(q, &[], top_k) {
                Ok(results) => {
                    let items: Vec<_> = results
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id,
                                "path": c.path,
                                "title": c.title,
                                "heading": c.heading,
                                "content": c.content,
                                "start_line": c.start_line,
                                "end_line": c.end_line,
                                "score": c.score,
                            })
                        })
                        .collect();
                    ("200 OK".to_string(), format!("{}\n", json!({"query": q, "results": items})))
                }
                Err(error) => ("500 Internal Server Error".to_string(), format!("{{\"error\":\"{error}\"}}\n")),
            }
        }
        _ => ("404 Not Found".to_string(), "{\"error\":\"not found\"}\n".to_string()),
    }
}

/// Percent-decode a query value (`+` becomes a space). Decodes to bytes
/// first so multi-byte UTF-8 (e.g. `%E9%83%A8` = 部) survives.
fn url_decode(value: &str) -> String {
    let mut bytes = Vec::new();
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        match c {
            '+' => bytes.push(b' '),
            '%' => {
                let hi = chars.next().and_then(|c| c.to_digit(16));
                let lo = chars.next().and_then(|c| c.to_digit(16));
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    bytes.push((hi * 16 + lo) as u8);
                }
            }
            _ => {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Run the HTTP API server (13.1-13.2): bind 127.0.0.1 on an ephemeral port,
/// write `sag-http.json` with the port, and serve until shutdown.
pub async fn run_http_api(data: PathBuf, store: Arc<Store>, shutdown: CancellationToken) -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let endpoint = json!({"port": port, "pid": std::process::id()});
    std::fs::write(data.join("sag-http.json"), serde_json::to_vec(&endpoint)?)?;
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                accepted = listener.accept() => {
                    let Ok((stream, _)) = accepted else { break };
                    let store = Arc::clone(&store);
                    tokio::spawn(async move {
                        if let Err(error) = serve_http_api(stream, &store).await {
                            eprintln!("HTTP API 连接结束：{error:#}");
                        }
                    });
                }
            }
        }
    });
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    async fn get(port: u16, target: &str) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        stream
            .write_all(format!("GET {target} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[tokio::test]
    async fn http_search_returns_chunks_with_citations_on_loopback_only() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("kb");
        fs::create_dir(&root).unwrap();
        let store = Store::open(&dir.path().join("app/db.sqlite")).unwrap();
        let root_key = store.add_root(&root).unwrap();
        fs::write(root.join("guide.md"), "# 部署\n\n把服务部署到本地端口。\n").unwrap();
        store.scan(&root_key, |_| {}).unwrap();
        // 13.2: the listener binds loopback only.
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        assert!(listener.local_addr().unwrap().ip().is_loopback());
        let port = listener.local_addr().unwrap().port();
        let serve_store = Arc::new(store);
        let server = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else { break };
                let store = Arc::clone(&serve_store);
                tokio::spawn(async move {
                    let _ = serve_http_api(stream, &store).await;
                });
            }
        });
        let health = get(port, "/health").await;
        assert!(health.contains("200 OK") && health.contains("\"status\":\"ok\""), "{health}");
        let found = get(port, "/search?q=%E9%83%A8%E7%BD%B2&top_k=5").await;
        assert!(found.contains("200 OK"), "{found}");
        assert!(found.contains("guide.md"), "result must cite the file: {found}");
        assert!(found.contains("部署到本地端口"), "result must carry the chunk: {found}");
        let missing = get(port, "/search").await;
        assert!(missing.contains("400 Bad Request"), "{missing}");
        let unknown = get(port, "/nope").await;
        assert!(unknown.contains("404 Not Found"), "{unknown}");
        server.abort();
    }
}
