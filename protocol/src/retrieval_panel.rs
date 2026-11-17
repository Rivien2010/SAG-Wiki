use serde::{Deserialize, Serialize};
use crate::types::{Retrieval, RetrievalMode};

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelOptions {
    pub mode: Option<RetrievalMode>,
    pub top_k: Option<usize>,
    pub keyword_weight: Option<f64>,
    pub vector_weight: Option<f64>,
    pub threshold: Option<f64>,
    pub pool: Option<usize>,
    /// Ephemeral CLI environment credential; never persisted or returned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimax_key: Option<String>,
}
impl std::fmt::Debug for PanelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PanelOptions").finish_non_exhaustive()
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuestionResult {
    pub id: String,
    pub question: String,
    pub retrieval_score: f64,
    pub evidence: Vec<String>,
    pub generation: String,
    pub generation_score: Option<u8>,
    pub valid_citations: usize,
    pub invalid_citations: usize,
    pub diagnosis: String,
    pub advice: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PanelOutput {
    pub options: Retrieval,
    pub notes: Vec<String>,
    pub questions: Vec<QuestionResult>,
    pub retrieval_total: f64,
    pub retrieval_max: u8,
    pub generation_total: Option<u8>,
    pub generation_status: String,
}
