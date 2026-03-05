use std::path::Path;
use std::time::UNIX_EPOCH;

use kbolt_types::{FileError, KboltError};
use sha2::{Digest, Sha256};

use crate::Result;

pub(super) fn modified_token(metadata: &std::fs::Metadata) -> Result<String> {
    let modified = metadata.modified()?;
    let duration = modified.duration_since(UNIX_EPOCH).map_err(|_| {
        KboltError::Internal("file modified timestamp predates unix epoch".to_string())
    })?;
    Ok(duration.as_nanos().to_string())
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub(super) fn file_title(path: &Path) -> String {
    path.file_name()
        .and_then(|item| item.to_str())
        .map(ToString::to_string)
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

pub(super) fn file_error(path: Option<std::path::PathBuf>, error: String) -> FileError {
    FileError {
        path: path
            .map(|item| item.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<unknown>".to_string()),
        error,
    }
}
