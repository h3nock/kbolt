use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Locator {
    Path(String),
    DocId(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GetRequest {
    pub locator: Locator,
    pub space: Option<String>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentResponse {
    pub docid: String,
    pub path: String,
    pub title: String,
    pub space: String,
    pub collection: String,
    pub content: String,
    pub stale: bool,
    pub total_lines: usize,
    pub returned_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiGetRequest {
    pub locators: Vec<Locator>,
    pub space: Option<String>,
    pub max_files: usize,
    pub max_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiGetResponse {
    pub documents: Vec<DocumentResponse>,
    pub omitted: Vec<OmittedFile>,
    pub resolved_count: usize,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OmittedFile {
    pub path: String,
    pub docid: String,
    pub size_bytes: usize,
    pub reason: OmitReason,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OmitReason {
    MaxFiles,
    MaxBytes,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileEntry {
    pub path: String,
    pub title: String,
    pub docid: String,
    pub active: bool,
    pub chunk_count: usize,
    pub embedded: bool,
}
