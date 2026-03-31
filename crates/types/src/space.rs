use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::indexing::UpdateReport;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActiveSpaceSource {
    Flag,
    EnvVar,
    ConfigDefault,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveSpace {
    pub name: String,
    pub source: ActiveSpaceSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpaceInfo {
    pub name: String,
    pub description: Option<String>,
    pub collection_count: usize,
    pub document_count: usize,
    pub chunk_count: usize,
    pub created: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AddCollectionRequest {
    pub path: PathBuf,
    pub space: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub extensions: Option<Vec<String>>,
    pub no_index: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CollectionInfo {
    pub name: String,
    pub space: String,
    pub path: PathBuf,
    pub description: Option<String>,
    pub extensions: Option<Vec<String>>,
    pub document_count: usize,
    pub active_document_count: usize,
    pub chunk_count: usize,
    pub embedded_chunk_count: usize,
    pub created: String,
    pub updated: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AddCollectionResult {
    pub collection: CollectionInfo,
    pub initial_indexing: InitialIndexingOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InitialIndexingOutcome {
    Skipped,
    Indexed(UpdateReport),
    Blocked(InitialIndexingBlock),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum InitialIndexingBlock {
    SpaceDenseRepairRequired { space: String, reason: String },
    ModelNotAvailable { name: String },
}
