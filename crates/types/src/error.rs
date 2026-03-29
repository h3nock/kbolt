use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum KboltError {
    #[error("database error: {0}")]
    Database(String),

    #[error("tantivy error: {0}")]
    Tantivy(String),

    #[error("usearch error: {0}")]
    USearch(String),

    #[error("space not found: {name}")]
    SpaceNotFound { name: String },

    #[error("collection not found: {name}")]
    CollectionNotFound { name: String },

    #[error("document not found: {path}")]
    DocumentNotFound { path: String },

    #[error("ambiguous space: collection '{collection}' exists in spaces: {spaces:?}")]
    AmbiguousSpace {
        collection: String,
        spaces: Vec<String>,
    },

    #[error("space already exists: {name}")]
    SpaceAlreadyExists { name: String },

    #[error("collection already exists: {name} (in space {space})")]
    CollectionAlreadyExists { name: String, space: String },

    #[error("no active space: use --space, set KBOLT_SPACE, or configure a default")]
    NoActiveSpace,

    #[error("file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("file deleted since indexing: {0}. Run `kbolt update` to refresh.")]
    FileDeleted(PathBuf),

    #[error("model not available: {name}")]
    ModelNotAvailable { name: String },

    #[error("model download failed: {0}")]
    ModelDownload(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("invalid path: {0}")]
    InvalidPath(PathBuf),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, KboltError>;
