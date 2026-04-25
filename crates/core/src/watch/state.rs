use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use kbolt_types::WatchRuntimeStatus;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::Result;

const WATCH_STATE_DIR: &str = "watch";
const WATCH_STATE_FILE: &str = "state.json";

pub(crate) struct WatchStateStore;

impl WatchStateStore {
    pub(crate) fn load(cache_dir: &Path) -> Result<Option<WatchRuntimeStatus>> {
        let state_file = Self::file_path(cache_dir);
        let bytes = match fs::read(&state_file) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err.into()),
        };

        Ok(Some(serde_json::from_slice(&bytes)?))
    }

    pub(crate) fn save(cache_dir: &Path, state: &WatchRuntimeStatus) -> Result<()> {
        fs::create_dir_all(Self::dir_path(cache_dir))?;
        let state_file = Self::file_path(cache_dir);
        let tmp_file = state_file.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(state)?;
        fs::write(&tmp_file, bytes)?;
        fs::rename(tmp_file, state_file)?;
        Ok(())
    }

    pub(crate) fn remove(cache_dir: &Path) -> Result<()> {
        match fs::remove_file(Self::file_path(cache_dir)) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err.into()),
        }
    }

    pub(crate) fn file_path(cache_dir: &Path) -> PathBuf {
        Self::dir_path(cache_dir).join(WATCH_STATE_FILE)
    }

    fn dir_path(cache_dir: &Path) -> PathBuf {
        cache_dir.join(WATCH_STATE_DIR)
    }
}

pub(crate) fn utc_now_string() -> Result<String> {
    system_time_string(SystemTime::now())
}

pub(crate) fn system_time_string(value: SystemTime) -> Result<String> {
    OffsetDateTime::from(value).format(&Rfc3339).map_err(|err| {
        kbolt_types::KboltError::Internal(format!("failed to format utc timestamp: {err}")).into()
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    use kbolt_types::{WatchHealth, WatchMode, WatchRuntimeState, WatchRuntimeStatus};
    use tempfile::tempdir;

    use super::WatchStateStore;

    fn sample_state(pid: u32, watched_collections: usize) -> WatchRuntimeStatus {
        WatchRuntimeStatus {
            mode: WatchMode::Native,
            health: WatchHealth::Ok,
            state: WatchRuntimeState::Idle,
            pid,
            started_at: "2026-04-25T00:00:00Z".to_string(),
            updated_at: "2026-04-25T00:00:01Z".to_string(),
            watched_collections,
            dirty_collections: 0,
            semantic_pending_collections: 0,
            semantic_unavailable_collections: 0,
            semantic_blocked_spaces: Vec::new(),
            collections: Vec::new(),
            last_keyword_refresh: None,
            last_semantic_refresh: None,
            last_safety_scan: None,
            last_catalog_refresh: None,
            last_error: None,
        }
    }

    #[test]
    fn save_and_load_roundtrip() {
        let tmp = tempdir().expect("tempdir");
        let state = sample_state(42, 2);

        WatchStateStore::save(tmp.path(), &state).expect("save");
        let loaded = WatchStateStore::load(tmp.path())
            .expect("load")
            .expect("state");

        assert_eq!(loaded, state);
    }

    #[test]
    fn remove_deletes_state_file_when_present() {
        let tmp = tempdir().expect("tempdir");
        WatchStateStore::save(tmp.path(), &sample_state(42, 2)).expect("save");

        WatchStateStore::remove(tmp.path()).expect("remove");

        assert!(WatchStateStore::load(tmp.path()).expect("load").is_none());
        WatchStateStore::remove(tmp.path()).expect("remove missing is ok");
    }

    #[test]
    fn readers_never_observe_partial_json_during_atomic_writes() {
        let tmp = tempdir().expect("tempdir");
        let done = Arc::new(AtomicBool::new(false));
        let cache_dir = tmp.path().to_path_buf();
        WatchStateStore::save(&cache_dir, &sample_state(1, 0)).expect("initial save");

        let writer_done = done.clone();
        let writer_cache = cache_dir.clone();
        let writer = thread::spawn(move || {
            for index in 0..200 {
                WatchStateStore::save(&writer_cache, &sample_state(1, index)).expect("save");
            }
            writer_done.store(true, Ordering::Release);
        });

        let readers = (0..4)
            .map(|_| {
                let reader_done = done.clone();
                let reader_cache = cache_dir.clone();
                thread::spawn(move || {
                    while !reader_done.load(Ordering::Acquire) {
                        let _ = WatchStateStore::load(&reader_cache)
                            .expect("state load should never see malformed json");
                    }
                })
            })
            .collect::<Vec<_>>();

        writer.join().expect("writer");
        for reader in readers {
            reader.join().expect("reader");
        }
    }
}
