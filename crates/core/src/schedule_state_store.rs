use std::fs;
use std::path::{Path, PathBuf};

use kbolt_types::ScheduleRunState;

use crate::error::Result;

const SCHEDULE_STATE_DIR: &str = "schedules";

pub(crate) struct ScheduleRunStateStore;

impl ScheduleRunStateStore {
    pub(crate) fn load(cache_dir: &Path, id: &str) -> Result<ScheduleRunState> {
        fs::create_dir_all(Self::dir_path(cache_dir))?;

        let state_file = Self::file_path(cache_dir, id);
        let bytes = match fs::read(&state_file) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ScheduleRunState::default())
            }
            Err(err) => return Err(err.into()),
        };

        Ok(serde_json::from_slice(&bytes)?)
    }

    pub(crate) fn save(cache_dir: &Path, id: &str, state: &ScheduleRunState) -> Result<()> {
        fs::create_dir_all(Self::dir_path(cache_dir))?;

        let state_file = Self::file_path(cache_dir, id);
        let tmp_file = state_file.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(state)?;
        fs::write(&tmp_file, bytes)?;
        fs::rename(tmp_file, state_file)?;
        Ok(())
    }

    pub(crate) fn remove(cache_dir: &Path, id: &str) -> Result<()> {
        let state_file = Self::file_path(cache_dir, id);
        match fs::remove_file(state_file) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err.into()),
        }
    }

    pub(crate) fn file_path(cache_dir: &Path, id: &str) -> PathBuf {
        Self::dir_path(cache_dir).join(format!("{id}.json"))
    }

    fn dir_path(cache_dir: &Path) -> PathBuf {
        cache_dir.join(SCHEDULE_STATE_DIR)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::ScheduleRunStateStore;
    use kbolt_types::{ScheduleRunResult, ScheduleRunState};

    #[test]
    fn load_returns_default_state_when_file_is_missing() {
        let tmp = tempdir().expect("create tempdir");

        let state = ScheduleRunStateStore::load(tmp.path(), "s1").expect("load missing state");

        assert_eq!(state, ScheduleRunState::default());
    }

    #[test]
    fn save_and_load_roundtrip_schedule_run_state() {
        let tmp = tempdir().expect("create tempdir");
        let state = ScheduleRunState {
            last_started: Some("2026-03-07T12:00:00Z".to_string()),
            last_finished: Some("2026-03-07T12:00:09Z".to_string()),
            last_result: Some(ScheduleRunResult::Success),
            last_error: None,
        };

        ScheduleRunStateStore::save(tmp.path(), "s1", &state).expect("save state");
        let loaded = ScheduleRunStateStore::load(tmp.path(), "s1").expect("load state");

        assert_eq!(loaded, state);
    }

    #[test]
    fn remove_deletes_state_file_when_present() {
        let tmp = tempdir().expect("create tempdir");
        ScheduleRunStateStore::save(tmp.path(), "s1", &ScheduleRunState::default())
            .expect("save state");

        ScheduleRunStateStore::remove(tmp.path(), "s1").expect("remove state");

        let loaded = ScheduleRunStateStore::load(tmp.path(), "s1").expect("load removed state");
        assert_eq!(loaded, ScheduleRunState::default());
    }
}
