//! Stage integration defaults (master plan §3).
//!
//! Protocol-owned defaults live in `Retrieval`/`EmbeddingConfig` so old saved
//! settings and this aggregate share one source. These knobs are configuration
//! for integration stages, not a replacement for the existing SAG2 algorithm.
use crate::types::{EmbeddingConfig, Retrieval};

#[derive(Clone, Debug)]
pub struct StageConfig {
    pub model: String,
    pub base_url: String,
    pub temperature: f64,
    pub max_tokens: usize,
    pub thinking: bool,
    pub rrf_k: usize,
    pub quota_per_author: usize,
    pub top_k_per_leg: usize,
    pub top_k_final: usize,
    pub legs: Vec<String>,
    pub embedding: EmbeddingConfig,
    pub rerank_enabled: bool,
    pub rerank_candidate_pool_min: usize,
    pub rerank_candidate_pool_max: usize,
    pub rerank_candidate_pool: usize,
    pub keyword_weight: f64,
    pub vector_weight: f64,
    pub similarity_threshold: f64,
    pub candidate_pool: usize,
    pub chunk_chars: usize,
    pub chunk_overlap: usize,
    pub query_expansion_enabled: bool,
    pub entity_pmi_weight: f64,
    pub entity_relation_weight: f64,
    pub indirect_relation_decay: f64,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self::from_retrieval(&Retrieval::default())
    }
}

impl StageConfig {
    /// Omitted saved settings have already fallen back through serde defaults.
    /// Preserve explicit values (including zero quotas/limits) without silently
    /// changing user configuration. Generation/rerank/graph use stage defaults.
    pub fn from_retrieval(retrieval: &Retrieval) -> Self {
        Self {
            model: "MiniMax-M3".into(),
            base_url: "https://api.minimaxi.com/v1".into(),
            temperature: 0.4,
            max_tokens: 8000,
            thinking: false,
            rrf_k: retrieval.rrf_k,
            quota_per_author: retrieval.quota_per_author,
            top_k_per_leg: retrieval.top_k_per_leg,
            top_k_final: retrieval.top_k_final,
            legs: retrieval.legs.clone(),
            embedding: retrieval.embedding.clone(),
            rerank_enabled: retrieval.rerank_enabled,
            rerank_candidate_pool_min: 64,
            rerank_candidate_pool_max: 256,
            rerank_candidate_pool: retrieval.rerank_candidate_pool.clamp(64, 256),
            keyword_weight: retrieval.keyword_weight,
            vector_weight: retrieval.vector_weight,
            similarity_threshold: retrieval.similarity_threshold,
            candidate_pool: retrieval.candidate_k,
            chunk_chars: retrieval.chunk_chars,
            chunk_overlap: 80,
            query_expansion_enabled: false,
            entity_pmi_weight: 0.6,
            entity_relation_weight: 0.4,
            indirect_relation_decay: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_three_defaults_and_old_configs() {
        let config = StageConfig::default();
        assert_eq!(config.model, "MiniMax-M3");
        assert_eq!(config.base_url, "https://api.minimaxi.com/v1");
        assert_eq!((config.temperature, config.max_tokens, config.thinking), (0.4, 8000, false));
        assert_eq!((config.rrf_k, config.quota_per_author, config.top_k_per_leg, config.top_k_final), (60, 5, 50, 10));
        assert_eq!(config.legs, ["conan", "sag", "wemm"]);
        assert_eq!(config.embedding.conan_model, "TencentBAC/Conan-embedding-v1");
        assert_eq!(config.embedding.wemm_model, "tencent/WeMM-Embedding-2B");
        assert_eq!((config.embedding.conan_dim, config.embedding.wemm_dim, config.embedding.wemm_on_demand), (1792, 1024, true));
        assert!(!config.rerank_enabled);
        assert_eq!((config.rerank_candidate_pool_min, config.rerank_candidate_pool_max, config.rerank_candidate_pool), (64, 256, 128));
        assert_eq!((config.keyword_weight, config.vector_weight, config.similarity_threshold, config.candidate_pool), (0.8, 0.25, 0.2, 50));
        assert_eq!((config.chunk_chars, config.chunk_overlap), (512, 80));
        assert!(!config.query_expansion_enabled);
        assert_eq!((config.entity_pmi_weight, config.entity_relation_weight, config.indirect_relation_decay), (0.6, 0.4, 0.5));

        let old: Retrieval = serde_json::from_str(r#"{"top_k":7}"#).unwrap();
        assert_eq!(old.top_k, 7);
        assert_eq!(old.top_k_final, config.top_k_final);
        assert_eq!(serde_json::to_value(&old.embedding).unwrap(), serde_json::to_value(&config.embedding).unwrap());
        let partial: Retrieval = serde_json::from_str(r#"{"embedding":{"conan_dim":42},"rrf_k":9,"quota_per_author":2,"top_k_per_leg":13,"top_k_final":4,"legs":["sag"],"keyword_weight":0.8,"vector_weight":0.2,"similarity_threshold":0.1,"candidate_k":25}"#).unwrap();
        let custom = StageConfig::from_retrieval(&partial);
        assert_eq!(custom.embedding.conan_dim, 42);
        assert_eq!(custom.embedding.wemm_dim, config.embedding.wemm_dim);
        assert!(custom.embedding.wemm_on_demand);
        assert_eq!((custom.rrf_k, custom.quota_per_author, custom.top_k_per_leg, custom.top_k_final), (9, 2, 13, 4));
        assert_eq!(custom.legs, ["sag"]);
        assert_eq!((custom.keyword_weight, custom.vector_weight, custom.similarity_threshold, custom.candidate_pool), (0.8, 0.2, 0.1, 25));
    }
}
