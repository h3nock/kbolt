use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpdateOptions {
    pub space: Option<String>,
    pub collections: Vec<String>,
    pub no_embed: bool,
    pub dry_run: bool,
    pub verbose: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpdateReport {
    pub scanned_docs: usize,
    pub skipped_mtime_docs: usize,
    pub skipped_hash_docs: usize,
    pub added_docs: usize,
    pub updated_docs: usize,
    pub failed_docs: usize,
    pub deactivated_docs: usize,
    pub reactivated_docs: usize,
    pub reaped_docs: usize,
    pub embedded_chunks: usize,
    pub decisions: Vec<UpdateDecision>,
    pub errors: Vec<FileError>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UpdateDecision {
    pub space: String,
    pub collection: String,
    pub path: String,
    pub kind: UpdateDecisionKind,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UpdateDecisionKind {
    New,
    Changed,
    SkippedMtime,
    SkippedHash,
    Ignored,
    Unsupported,
    ReadFailed,
    ExtractFailed,
    Reactivated,
    Deactivated,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileError {
    pub path: String,
    pub error: String,
}
