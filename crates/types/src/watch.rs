use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WatchBackend {
    Launchd,
    SystemdUser,
    Unsupported,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WatchMode {
    Native,
    Polling,
    Foreground,
    Disabled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WatchHealth {
    Ok,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WatchRuntimeState {
    Starting,
    Idle,
    RefreshingKeyword,
    RefreshingSemantic,
    Checking,
    BackingOff,
    Stopping,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WatchSemanticState {
    None,
    Pending {
        pending_chunks: usize,
    },
    Unavailable {
        pending_chunks: usize,
        reason: String,
    },
    Blocked {
        space: String,
        reason: String,
        fix: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WatchServiceStatus {
    pub enabled: bool,
    pub running: bool,
    pub backend: WatchBackend,
    pub pid: Option<u32>,
    pub issue: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WatchStatusResponse {
    pub service: WatchServiceStatus,
    pub runtime: Option<WatchRuntimeStatus>,
    pub log_file: PathBuf,
    pub state_file: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WatchRuntimeStatus {
    pub mode: WatchMode,
    pub health: WatchHealth,
    pub state: WatchRuntimeState,
    pub pid: u32,
    pub started_at: String,
    pub updated_at: String,
    pub watched_collections: usize,
    pub dirty_collections: usize,
    pub semantic_pending_collections: usize,
    pub semantic_unavailable_collections: usize,
    pub semantic_blocked_spaces: Vec<WatchSpaceBlock>,
    pub collections: Vec<WatchCollectionStatus>,
    pub last_keyword_refresh: Option<WatchRefreshSummary>,
    pub last_semantic_refresh: Option<WatchRefreshSummary>,
    pub last_safety_scan: Option<String>,
    pub last_catalog_refresh: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WatchCollectionStatus {
    pub space: String,
    pub collection: String,
    pub path: PathBuf,
    pub dirty: bool,
    pub semantic: WatchSemanticState,
    pub last_event_at: Option<String>,
    pub last_keyword_refresh: Option<String>,
    pub last_semantic_refresh: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WatchSpaceBlock {
    pub space: String,
    pub reason: String,
    pub fix: String,
    pub set_at: String,
    pub backoff_until: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WatchRefreshSummary {
    pub space: String,
    pub collection: String,
    pub started_at: String,
    pub finished_at: String,
    pub elapsed_ms: u64,
    pub scanned_docs: usize,
    pub changed_docs: usize,
    pub embedded_chunks: usize,
}
