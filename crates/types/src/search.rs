use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SearchMode {
    Auto,
    Deep,
    Keyword,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchRequest {
    pub query: String,
    pub mode: SearchMode,
    pub space: Option<String>,
    pub collections: Vec<String>,
    pub limit: usize,
    pub min_score: f32,
    pub no_rerank: bool,
    pub debug: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub query: String,
    pub mode: SearchMode,
    pub staleness_hint: Option<String>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    pub docid: String,
    pub path: String,
    pub title: String,
    pub space: String,
    pub collection: String,
    pub heading: Option<String>,
    pub text: String,
    pub score: f32,
    pub signals: Option<SearchSignals>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchSignals {
    pub bm25: Option<f32>,
    pub dense: Option<f32>,
    pub rrf: f32,
    pub reranker: Option<f32>,
}
