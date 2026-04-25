pub mod log;
pub mod scheduler;
pub mod service;
pub mod source;
pub mod state;

use std::path::PathBuf;
use std::time::Duration;

pub(crate) const EVENT_DEBOUNCE: Duration = Duration::from_secs(2);
pub(crate) const KEYWORD_QUIET: Duration = Duration::from_secs(30);
pub(crate) const KEYWORD_CAP: Duration = Duration::from_secs(5 * 60);
pub(crate) const SEMANTIC_QUIET: Duration = Duration::from_secs(5 * 60);
pub(crate) const SEMANTIC_BLOCK_BACKOFF: Duration = Duration::from_secs(60 * 60);
pub(crate) const SEMANTIC_UNAVAILABLE_BACKOFF: Duration = Duration::from_secs(10 * 60);
pub(crate) const UPDATE_ERROR_BACKOFF: Duration = Duration::from_secs(60);
pub(crate) const POLL_INTERVAL: Duration = Duration::from_secs(5 * 60);
pub(crate) const SAFETY_RESCAN_BASE: Duration = Duration::from_secs(60 * 60);
pub(crate) const CATALOG_REFRESH_INTERVAL: Duration = Duration::from_secs(60);
pub(crate) const WATCH_LOG_MAX_BYTES: u64 = 10 * 1024 * 1024;
pub(crate) const WATCH_LOG_ROTATIONS: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct CollectionKey {
    pub space: String,
    pub collection: String,
}

impl CollectionKey {
    pub(crate) fn new(space: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            space: space.into(),
            collection: collection.into(),
        }
    }
}

impl std::fmt::Display for CollectionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.space, self.collection)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WatchCollection {
    pub key: CollectionKey,
    pub path: PathBuf,
}

pub(crate) fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}
