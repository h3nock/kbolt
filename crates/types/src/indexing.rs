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
    pub scanned: usize,
    pub skipped_mtime: usize,
    pub skipped_hash: usize,
    pub added: usize,
    pub updated: usize,
    pub deactivated: usize,
    pub reactivated: usize,
    pub reaped: usize,
    pub embedded: usize,
    pub errors: Vec<FileError>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileError {
    pub path: String,
    pub error: String,
}
