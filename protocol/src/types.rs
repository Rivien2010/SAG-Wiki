use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
    Max,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelProfile {
    pub system_prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    pub sag2_enabled: bool,
    pub mcp_servers: Vec<String>,
    pub mcp_tools: BTreeMap<String, Vec<String>>,
}
impl Default for ModelProfile {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            reasoning_effort: None,
            sag2_enabled: true,
            mcp_servers: vec![],
            mcp_tools: BTreeMap::new(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct McpServer {
    pub id: String,
    pub name: String,
    pub transport: String,
    pub url: String,
    pub command: String,
    pub args: Vec<String>,
    pub require_confirmation: bool,
    pub has_credentials: bool,
    pub tools: Option<Vec<McpToolInfo>>,
}
impl Default for McpServer {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            transport: "http".into(),
            url: String::new(),
            command: String::new(),
            args: vec![],
            require_confirmation: true,
            has_credentials: false,
            tools: None,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct McpCredentials {
    pub env: BTreeMap<String, String>,
    pub headers: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolApproval {
    pub id: String,
    pub query_id: String,
    pub server_name: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Root {
    pub path: String,
    pub name: String,
    pub status: String,
    pub count: usize,
    pub error: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct IndexProgress {
    pub count: usize,
    pub total: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub path: String,
    pub name: String,
    pub directory: bool,
    pub symlink: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: i64,
    pub path: String,
    pub name: String,
    pub body: String,
    pub hash: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileInfo {
    pub id: i64,
    pub path: String,
    pub name: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentTerm {
    pub term: String,
    pub count: usize,
    pub boost: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub start_column: usize,
    pub end_line: usize,
    pub end_column: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub id: i64,
    pub document_id: i64,
    pub path: String,
    pub title: String,
    pub heading: String,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub source_ranges: Vec<(usize, usize)>,
    #[serde(default)]
    pub source_spans: Vec<SourceSpan>,
    pub bm25: f64,
    pub score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_score: Option<f64>,
    pub order: usize,
    pub hash: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchPagination {
    pub total: usize,
    pub page: usize,
    pub page_size: usize,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAuthMode {
    #[default]
    Bearer,
    None,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub url: String,
    pub model: String,
    pub models: Vec<String>,
    pub has_key: bool,
    pub auth_mode: ModelAuthMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
}
impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            url: "https://api.302ai.cn/v1".into(),
            model: "qwen3.6-flash".into(),
            models: vec!["qwen3.6-flash".into()],
            has_key: false,
            auth_mode: ModelAuthMode::Bearer,
            reasoning_effort: None,
        }
    }
}
impl ModelConfig {
    pub fn is_configured(&self) -> bool {
        !self.model.trim().is_empty() && (self.auth_mode == ModelAuthMode::None || self.has_key)
    }
}
/// Retrieval mode: `fast` runs SAG2 only (the pre-existing behaviour);
/// `accurate` fuses SAG2 with the vector legs (Conan / WeMM) via RRF +
/// per-group quota. The mode is a user setting; it defaults to `fast` so
/// existing behaviour is unchanged until the user opts in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalMode {
    #[default]
    Fast,
    Accurate,
}

/// Local embedding models. WeMM is never bundled or automatically downloaded.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    #[serde(default = "default_conan_model")]
    pub conan_model: String,
    #[serde(default = "default_conan_dim")]
    pub conan_dim: usize,
    #[serde(default = "default_wemm_model")]
    pub wemm_model: String,
    #[serde(default = "default_wemm_dim")]
    pub wemm_dim: usize,
    #[serde(default = "default_wemm_on_demand")]
    pub wemm_on_demand: bool,
}
fn default_conan_model() -> String {
    "TencentBAC/Conan-embedding-v1".into()
}
fn default_conan_dim() -> usize {
    1792
}
fn default_wemm_model() -> String {
    "tencent/WeMM-Embedding-2B".into()
}
fn default_wemm_dim() -> usize {
    1024
}
fn default_wemm_on_demand() -> bool {
    true
}
impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            conan_model: default_conan_model(),
            conan_dim: default_conan_dim(),
            wemm_model: default_wemm_model(),
            wemm_dim: default_wemm_dim(),
            wemm_on_demand: default_wemm_on_demand(),
        }
    }
}

/// Allowed extraction vocabulary. Validate before using a saved schema.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct GraphSchema {
    pub version: String,
    pub entity_types: Vec<String>,
    pub relation_types: Vec<String>,
}
impl Default for GraphSchema {
    fn default() -> Self {
        Self {
            version: "1".into(),
            entity_types: ["person", "org", "product", "regulation", "location", "other"].map(String::from).to_vec(),
            relation_types: ["owns", "part_of", "caused_by", "regulates", "located_in", "other"].map(String::from).to_vec(),
        }
    }
}
impl GraphSchema {
    pub fn validate(&self) -> Result<(), String> {
        if self.version.trim().is_empty() { return Err("empty schema version".into()); }
        for names in [&self.entity_types, &self.relation_types] {
            let mut seen = std::collections::HashSet::new();
            if names.is_empty() { return Err("empty schema vocabulary".into()); }
            for name in names {
                if name.trim().is_empty() || name.trim() != name || !seen.insert(name) {
                    return Err(format!("empty, untrimmed or duplicate schema type: {name:?}"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Retrieval {
    pub candidate_k: usize,
    pub seed_k: usize,
    pub expansion_term_limit: usize,
    pub max_hops: usize,
    pub top_k: usize,
    pub chunk_chars: usize,
    /// Fast (SAG2 only) or accurate (multi-leg fusion). Defaults to fast.
    pub mode: RetrievalMode,
    /// RRF damping constant for multi-leg fusion (validated value: 60).
    pub rrf_k: usize,
    /// Per-group quota for fusion (validated value: 5). A "group" is the
    /// source author as determined by `models::source_author` (title/heading).
    pub quota_per_author: usize,
    /// How many ranked chunks each leg contributes to fusion (validated: 50).
    pub top_k_per_leg: usize,
    /// Final result count for accurate-mode fusion; SAG2 retains `top_k`.
    pub top_k_final: usize,
    pub embedding: EmbeddingConfig,
    pub graph_schema: GraphSchema,
    /// Reserved retrieval knobs for vector integration; do not alter SAG2 scoring.
    pub keyword_weight: f64,
    pub vector_weight: f64,
    pub similarity_threshold: f64,
    /// Retrieval legs enabled for accurate mode, in order. The SAG2 leg is
    /// always available; vector legs (`conan`, `wemm`) are skipped when their
    /// embedding model is missing (graceful degradation).
    pub legs: Vec<String>,
    /// Extra BM25 queries for vocabulary-mismatched evidence missed by the main legs.
    #[serde(default)]
    pub variant_queries: Vec<String>,
    /// Optional rerank stage (Stage 3, 11.1-11.2). Off by default; when on,
    /// the fused pool is reordered by a cross-encoder before the final cut.
    /// Measured on the reference corpus, reranking scored below quota fusion
    /// (3.0/2.5 vs 4.0/5), so it stays off unless a model is configured.
    #[serde(default)]
    pub rerank_enabled: bool,
    /// Candidate pool size for reranking, clamped to 64-256.
    #[serde(default = "default_rerank_pool")]
    pub rerank_candidate_pool: usize,
}

fn default_rerank_pool() -> usize {
    128
}
impl Default for Retrieval {
    fn default() -> Self {
        Self {
            candidate_k: 50,
            seed_k: 10,
            expansion_term_limit: 20,
            max_hops: 1,
            top_k: 10,
            chunk_chars: 512,
            mode: RetrievalMode::Fast,
            rrf_k: 60,
            quota_per_author: 5,
            top_k_per_leg: 50,
            top_k_final: 10,
            embedding: EmbeddingConfig::default(),
            graph_schema: GraphSchema::default(),
            keyword_weight: 0.8,
            vector_weight: 0.25,
            similarity_threshold: 0.2,
            legs: vec!["conan".to_owned(), "sag".to_owned(), "wemm".to_owned()],
            variant_queries: vec![],
            rerank_enabled: false,
            rerank_candidate_pool: 128,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FileTypes {
    pub markdown: bool,
    pub text: bool,
    pub code: bool,
    pub config: bool,
}
impl Default for FileTypes {
    fn default() -> Self {
        Self {
            markdown: true,
            text: true,
            code: false,
            config: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FolderIndexOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<String>>,
    pub file_types: FileTypes,
    pub convert_other_formats: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversion_formats: Option<Vec<String>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct UpdateSettings {
    pub source: UpdateSource,
    pub github_url: String,
    pub website_url: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateSource {
    #[default]
    Github,
    Website,
}
impl Default for UpdateSettings {
    fn default() -> Self {
        Self {
            source: UpdateSource::Github,
            github_url: "https://github.com/Zleap-AI/SAG-Wiki".into(),
            website_url: String::new(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub runtime: crate::runtime::RuntimeSettings,
    pub mineru: crate::mineru::MineruConfig,
    pub updates: UpdateSettings,
    pub convert_other_formats: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversion_formats: Option<Vec<String>>,
    #[serde(default)]
    pub retrieval_version: u8,
    pub credential_error: Option<String>,
    pub answer: ModelConfig,
    pub jev: ModelConfig,
    pub retrieval: Retrieval,
    pub theme: String,
    pub language: String,
    pub start_page: String,
    pub editor_size: usize,
    pub show_hidden: bool,
    pub watch: bool,
    pub index_workers: usize,
    pub conversion_workers: usize,
    pub excluded: Vec<String>,
    pub file_types: FileTypes,
    pub timeout_secs: u64,
    pub answer_context_tokens: usize,
    pub answer_output_tokens: usize,
    /// Maximum tool calls per turn; 0 leaves the number unlimited.
    pub max_tool_calls: usize,
    pub search_summary_enabled: bool,
    pub search_summary_sources: usize,
    pub model_profiles: BTreeMap<String, ModelProfile>,
    pub mcp_servers: Vec<McpServer>,
    #[serde(default)]
    pub notes: NoteSettings,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            runtime: crate::runtime::RuntimeSettings::default(),
            mineru: crate::mineru::MineruConfig::default(),
            updates: UpdateSettings::default(),
            convert_other_formats: false,
            conversion_formats: None,
            retrieval_version: crate::RETRIEVAL_VERSION,
            credential_error: None,
            answer: ModelConfig::default(),
            jev: ModelConfig {
                url: "https://api.302ai.cn/typesafeai/v1".into(),
                model: "jev-latest".into(),
                models: vec![],
                has_key: false,
                auth_mode: ModelAuthMode::Bearer,
                reasoning_effort: None,
            },
            retrieval: Retrieval::default(),
            theme: "light".into(),
            language: "zh-CN".into(),
            start_page: "ask".into(),
            editor_size: 15,
            show_hidden: false,
            watch: true,
            index_workers: 8,
            conversion_workers: 1,
            file_types: FileTypes::default(),
            excluded: [
                ".git",
                "node_modules",
                "dist",
                "target",
                ".venv",
                "venv",
                "__pycache__",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            timeout_secs: 120,
            answer_context_tokens: 65536,
            answer_output_tokens: 2048,
            max_tool_calls: 0,
            search_summary_enabled: true,
            search_summary_sources: 10,
            model_profiles: BTreeMap::new(),
            mcp_servers: vec![],
            notes: NoteSettings::default(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextSummary {
    pub text: String,
    pub covered_turns: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnswerContext {
    pub limit: usize,
    pub reserved_output: usize,
    pub estimated_input: usize,
    pub omitted_sources: usize,
    pub omitted_history: usize,
    pub input_tokens: Option<usize>,
    pub output_tokens: Option<usize>,
    #[serde(default)]
    pub elapsed_ms: Option<u64>,
    #[serde(default)]
    pub generation_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<ContextSummary>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub id: String,
    /// UTF-16 position in the answer where this tool was called.
    #[serde(default)]
    pub answer_offset: Option<usize>,
    pub query: String,
    pub source_count: usize,
    #[serde(default)]
    pub elapsed_ms: Option<u64>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub server_name: Option<String>,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub denied: bool,
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReasoningBlock {
    /// UTF-16 position in the answer, after the preceding tool calls.
    pub answer_offset: usize,
    pub tool_count: usize,
    pub content: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<i64>,
    pub question: String,
    pub answer: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_blocks: Vec<ReasoningBlock>,
    pub evidence: Vec<Candidate>,
    pub model: String,
    pub scope: Vec<String>,
    pub references: Vec<String>,
    pub status: String,
    pub error: Option<String>,
    pub retrieval_question: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default)]
    pub context: Option<AnswerContext>,
    /// Absolute path of the note this answer was saved to, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note_path: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub messages: Vec<Message>,
    pub updated: i64,
    pub deleted: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Draft {
    pub path: String,
    pub name: String,
    pub body: String,
    pub hash: String,
    pub is_new: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolProgress {
    pub call: ToolCallRecord,
    pub finished: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Progress {
    pub replace: bool,
    pub id: String,
    pub stage: String,
    pub delta: Option<String>,
    pub reasoning_delta: Option<String>,
    pub reasoning_block: Option<ReasoningBlock>,
    pub count: usize,
    pub tool: Option<ToolProgress>,
}

#[cfg(test)]
mod default_tests {
    use super::*;
    #[test]
    fn message_timestamps_preserve_existing_history_without_inventing_dates() {
        let old = serde_json::json!({
            "question":"old question","answer":"old answer","evidence":[],
            "model":"test","scope":[],"references":[],"status":"complete",
            "error":null,"retrieval_question":"old question"
        });
        let mut message: Message = serde_json::from_value(old).unwrap();
        assert!(message.created_at.is_none());
        assert!(serde_json::to_value(&message).unwrap().get("created_at").is_none());
        message.created_at = Some(1_792_368_300_000);
        let restored: Message = serde_json::from_value(serde_json::to_value(message).unwrap()).unwrap();
        assert_eq!(restored.created_at, Some(1_792_368_300_000));
    }
    #[test]
    fn sag2_stays_enabled_for_new_and_existing_model_profiles() {
        assert!(ModelProfile::default().sag2_enabled);
        assert!(
            serde_json::from_str::<ModelProfile>(
                r#"{"system_prompt":"原有提示词","mcp_servers":["wiki"]}"#
            )
            .unwrap()
            .sag2_enabled
        );
        assert!(
            !serde_json::from_str::<ModelProfile>(r#"{"sag2_enabled":false}"#)
                .unwrap()
                .sag2_enabled
        );
    }
    #[test]
    fn answer_context_defaults_to_64k() {
        assert_eq!(Settings::default().answer_context_tokens, 65_536);
        assert_eq!(
            serde_json::from_str::<Settings>("{}")
                .unwrap()
                .answer_context_tokens,
            65_536
        );
    }
    #[test]
    fn tool_call_limit_defaults_to_unlimited() {
        assert_eq!(Settings::default().max_tool_calls, 0);
        assert_eq!(serde_json::from_str::<Settings>("{}").unwrap().max_tool_calls, 0);
        assert_eq!(serde_json::from_str::<Settings>(r#"{"max_tool_calls":10}"#).unwrap().max_tool_calls, 10);
    }

    #[test]
    fn retrieval_defaults_to_v10() {
        // Snapshot of the full Retrieval default. HEAD's Retrieval had only
        // the six fields below; Stage 1 added mode/weights/legs/quota/rerank
        // without updating this snapshot, so it has failed since Stage 1.
        // This refresh records the complete current default (incl. the B1
        // `variant_queries` field, empty by default).
        let expected = serde_json::json!({"candidate_k":50,"seed_k":10,"expansion_term_limit":20,"max_hops":1,"top_k":10,"chunk_chars":512,"mode":"fast","rrf_k":60,"quota_per_author":5,"top_k_per_leg":50,"top_k_final":10,"keyword_weight":0.8,"vector_weight":0.25,"similarity_threshold":0.2,"legs":["conan","sag","wemm"],"variant_queries":[],"rerank_enabled":false,"rerank_candidate_pool":128,"embedding":{"conan_dim":1792,"conan_model":"TencentBAC/Conan-embedding-v1","wemm_dim":1024,"wemm_model":"tencent/WeMM-Embedding-2B","wemm_on_demand":true},"graph_schema":{"entity_types":["person","org","product","regulation","location","other"],"relation_types":["owns","part_of","caused_by","regulates","located_in","other"],"version":"1"}});
        assert_eq!(
            serde_json::to_value(Settings::default().retrieval).unwrap(),
            expected
        );
        assert_eq!(
            serde_json::to_value(serde_json::from_str::<Settings>("{}").unwrap().retrieval)
                .unwrap(),
            expected
        );
    }
}

/// Notes: answers saved as Markdown with verifiable source anchors.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NoteSettings {
    /// Folder that receives saved notes; `None` uses the default library folder.
    pub folder: Option<String>,
    /// Ask for a location on every save instead of reusing the remembered folder.
    pub ask_location: bool,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteSourceState {
    Intact,
    Moved,
    Changed,
    Missing,
    #[default]
    Unverified,
}
impl NoteSourceState {
    /// Attention order: missing > changed > unverified > moved > intact.
    pub fn severity(self) -> u8 {
        match self {
            Self::Missing => 4,
            Self::Changed => 3,
            Self::Unverified => 2,
            Self::Moved => 1,
            Self::Intact => 0,
        }
    }
    pub fn needs_attention(self) -> bool {
        self.severity() >= 2
    }
}
/// One citation anchor stored in a note's frontmatter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoteSource {
    pub id: usize,
    /// Absolute path after resolution.
    pub path: String,
    /// Original document for converted files, shown instead of the Markdown output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_path: Option<String>,
    pub hash: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoteSourceStatus {
    pub source: NoteSource,
    pub state: NoteSourceState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<SourceSpan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_at: Option<i64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoteSummary {
    pub path: String,
    pub title: String,
    pub question: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asked_at: Option<i64>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub scope: Vec<String>,
    pub state: NoteSourceState,
    pub source_count: usize,
    /// Sources whose state needs attention.
    pub attention: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parse_error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoteDetail {
    pub summary: NoteSummary,
    pub sources: Vec<NoteSourceStatus>,
}
