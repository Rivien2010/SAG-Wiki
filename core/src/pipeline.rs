//! Composable pipelines (Stage 3, 12.1-12.2).
//!
//! A pipeline is Parser -> Chunker -> Compiler -> Indexer, assembled from
//! registered components and published under a version. A named pipeline
//! config picks one published pipeline version plus the rerank choice, and
//! can be saved and reused. Built-in components mirror what the codebase
//! already does; custom components register by name.
//!
//! - Parser: `anydoc` (fast tier), `docling` (accurate tier).
//! - Chunker: `general`, `hierarchy`, `table`, `presentation`,
//!   `parent-child` (see `chunking::ChunkMethod`).
//! - Compiler: `fts` (FTS5 terms only), `fts-vector` (FTS + vector legs).
//! - Indexer: `sqlite` (the local store).
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::chunking::ChunkMethod;

/// Pipeline stage kinds (12.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ComponentKind {
    Parser,
    Chunker,
    Compiler,
    Indexer,
}

impl ComponentKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ComponentKind::Parser => "parser",
            ComponentKind::Chunker => "chunker",
            ComponentKind::Compiler => "compiler",
            ComponentKind::Indexer => "indexer",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "parser" => Some(ComponentKind::Parser),
            "chunker" => Some(ComponentKind::Chunker),
            "compiler" => Some(ComponentKind::Compiler),
            "indexer" => Some(ComponentKind::Indexer),
            _ => None,
        }
    }
}

/// One registered component (12.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineComponent {
    pub kind: ComponentKind,
    pub name: String,
    pub version: String,
    pub description: String,
}

/// Built-in components (12.1). Custom components register on top.
pub fn builtin_components() -> Vec<PipelineComponent> {
    vec![
        PipelineComponent { kind: ComponentKind::Parser, name: "anydoc".into(), version: "1".into(), description: "轻量文本抽取（快档）".into() },
        PipelineComponent { kind: ComponentKind::Parser, name: "docling".into(), version: "1".into(), description: "布局与表格识别（准档）".into() },
        PipelineComponent { kind: ComponentKind::Chunker, name: "general".into(), version: "1".into(), description: "句子切分（默认）".into() },
        PipelineComponent { kind: ComponentKind::Chunker, name: "hierarchy".into(), version: "1".into(), description: "标题层级切分，保留完整父标题路径".into() },
        PipelineComponent { kind: ComponentKind::Chunker, name: "table".into(), version: "1".into(), description: "按行切分，附表头与上下文窗口".into() },
        PipelineComponent { kind: ComponentKind::Chunker, name: "presentation".into(), version: "1".into(), description: "按幻灯片切分".into() },
        PipelineComponent { kind: ComponentKind::Chunker, name: "parent-child".into(), version: "1".into(), description: "小子块检索，父块补全上下文".into() },
        PipelineComponent { kind: ComponentKind::Compiler, name: "fts".into(), version: "1".into(), description: "FTS5 全文词项".into() },
        PipelineComponent { kind: ComponentKind::Compiler, name: "fts-vector".into(), version: "1".into(), description: "FTS5 + 向量腿".into() },
        PipelineComponent { kind: ComponentKind::Indexer, name: "sqlite".into(), version: "1".into(), description: "本地 SQLite 索引".into() },
    ]
}

/// A published pipeline version (12.1): a frozen combination of one
/// component per stage. Publishing the same name again bumps the version.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineVersion {
    pub name: String,
    pub version: i64,
    pub parser: String,
    pub chunker: String,
    pub compiler: String,
    pub indexer: String,
    pub created_at: i64,
}

/// A named, reusable pipeline config (12.2): one published pipeline version
/// plus the rerank choice.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub name: String,
    pub pipeline_name: String,
    pub pipeline_version: i64,
    pub rerank_enabled: bool,
    pub rerank_pool: usize,
}

impl PipelineConfig {
    /// Resolve the chunker component name to a `ChunkMethod`.
    pub fn chunk_method(&self, version: &PipelineVersion) -> Result<ChunkMethod> {
        ChunkMethod::from_str(&version.chunker)
            .ok_or_else(|| anyhow::anyhow!("流水线引用的分块方法不存在：{}", version.chunker))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_cover_all_four_stages() {
        let components = builtin_components();
        for kind in [ComponentKind::Parser, ComponentKind::Chunker, ComponentKind::Compiler, ComponentKind::Indexer] {
            assert!(components.iter().any(|c| c.kind == kind), "missing {kind:?}");
        }
        assert!(components.iter().any(|c| c.name == "hierarchy"));
        assert!(components.iter().any(|c| c.name == "fts-vector"));
    }

    #[test]
    fn chunk_method_resolves_from_pipeline_version() {
        let version = PipelineVersion {
            name: "default".into(),
            version: 1,
            parser: "anydoc".into(),
            chunker: "hierarchy".into(),
            compiler: "fts".into(),
            indexer: "sqlite".into(),
            created_at: 0,
        };
        let config = PipelineConfig { name: "books".into(), pipeline_name: "default".into(), pipeline_version: 1, rerank_enabled: false, rerank_pool: 128 };
        assert_eq!(config.chunk_method(&version).unwrap(), ChunkMethod::Hierarchy);
        let bad = PipelineVersion { chunker: "nope".into(), ..version };
        assert!(config.chunk_method(&bad).is_err());
    }
}
