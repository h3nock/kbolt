use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatusResponse {
    pub spaces: Vec<SpaceStatus>,
    pub models: ModelStatus,
    pub cache_dir: PathBuf,
    pub config_dir: PathBuf,
    pub total_documents: usize,
    pub total_chunks: usize,
    pub total_embedded: usize,
    pub disk_usage: DiskUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpaceStatus {
    pub name: String,
    pub description: Option<String>,
    pub collections: Vec<CollectionStatus>,
    pub last_updated: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CollectionStatus {
    pub name: String,
    pub path: PathBuf,
    pub documents: usize,
    pub active_documents: usize,
    pub chunks: usize,
    pub embedded_chunks: usize,
    pub last_updated: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelStatus {
    pub embedder: ModelInfo,
    pub reranker: ModelInfo,
    pub expander: ModelInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    pub name: String,
    pub downloaded: bool,
    pub size_bytes: Option<u64>,
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiskUsage {
    pub sqlite_bytes: u64,
    pub tantivy_bytes: u64,
    pub usearch_bytes: u64,
    pub models_bytes: u64,
    pub total_bytes: u64,
}
