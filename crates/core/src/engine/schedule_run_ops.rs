use crate::lock::LockMode;
use crate::schedule_state_store::ScheduleRunStateStore;
use crate::Result;
use kbolt_types::{KboltError, ScheduleRunResult, ScheduleRunState, ScheduleScope, UpdateOptions};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use super::{schedule_ops::load_schedule_definition, Engine};

impl Engine {
    pub fn run_schedule(&self, id: &str) -> Result<ScheduleRunState> {
        let schedule = load_schedule_definition(&self.config.config_dir, id)?;
        let started_at = utc_now_string()?;
        let mut state = ScheduleRunState {
            last_started: Some(started_at),
            last_finished: None,
            last_result: None,
            last_error: None,
        };
        ScheduleRunStateStore::save(&self.config.cache_dir, &schedule.id, &state)?;

        let lock = match self.acquire_operation_lock(LockMode::Exclusive) {
            Ok(lock) => lock,
            Err(err) if is_lock_contention_error(&err) => {
                state.last_finished = Some(utc_now_string()?);
                state.last_result = Some(ScheduleRunResult::SkippedLock);
                state.last_error = None;
                ScheduleRunStateStore::save(&self.config.cache_dir, &schedule.id, &state)?;
                return Ok(state);
            }
            Err(err) => {
                persist_failed_run_state(
                    &self.config.cache_dir,
                    &schedule.id,
                    &mut state,
                    err.to_string(),
                )?;
                return Err(err);
            }
        };

        let schedule = match load_schedule_definition(&self.config.config_dir, &schedule.id) {
            Ok(schedule) => schedule,
            Err(err) => {
                persist_failed_run_state(
                    &self.config.cache_dir,
                    &schedule.id,
                    &mut state,
                    err.to_string(),
                )?;
                return Err(err);
            }
        };

        let run_result = self.update_unlocked(update_options_for_schedule(&schedule.scope));
        drop(lock);

        match run_result {
            Ok(_) => {
                state.last_finished = Some(utc_now_string()?);
                state.last_result = Some(ScheduleRunResult::Success);
                state.last_error = None;
                ScheduleRunStateStore::save(&self.config.cache_dir, &schedule.id, &state)?;
                Ok(state)
            }
            Err(err) => {
                persist_failed_run_state(
                    &self.config.cache_dir,
                    &schedule.id,
                    &mut state,
                    err.to_string(),
                )?;
                Err(err)
            }
        }
    }

    pub fn schedule_run_state(&self, id: &str) -> Result<ScheduleRunState> {
        let schedule = load_schedule_definition(&self.config.config_dir, id)?;
        ScheduleRunStateStore::load(&self.config.cache_dir, &schedule.id)
    }
}

fn update_options_for_schedule(scope: &ScheduleScope) -> UpdateOptions {
    match scope {
        ScheduleScope::All => UpdateOptions {
            space: None,
            collections: Vec::new(),
            no_embed: false,
            dry_run: false,
            verbose: false,
        },
        ScheduleScope::Space { space } => UpdateOptions {
            space: Some(space.clone()),
            collections: Vec::new(),
            no_embed: false,
            dry_run: false,
            verbose: false,
        },
        ScheduleScope::Collections { space, collections } => UpdateOptions {
            space: Some(space.clone()),
            collections: collections.clone(),
            no_embed: false,
            dry_run: false,
            verbose: false,
        },
    }
}

fn is_lock_contention_error(err: &crate::error::CoreError) -> bool {
    err.to_string().contains("Another kbolt process is active")
}

fn persist_failed_run_state(
    cache_dir: &std::path::Path,
    schedule_id: &str,
    state: &mut ScheduleRunState,
    error_message: String,
) -> Result<()> {
    state.last_finished = Some(utc_now_string()?);
    state.last_result = Some(ScheduleRunResult::Failed);
    state.last_error = Some(error_message);

    if let Err(save_err) = ScheduleRunStateStore::save(cache_dir, schedule_id, &state) {
        let run_error = state.last_error.as_deref().unwrap_or("schedule run failed");
        return Err(KboltError::Internal(format!(
            "schedule run failed: {run_error}; additionally failed to save run state: {save_err}"
        ))
        .into());
    }

    Ok(())
}

fn utc_now_string() -> Result<String> {
    OffsetDateTime::now_utc().format(&Rfc3339).map_err(|err| {
        KboltError::Internal(format!("failed to format utc timestamp: {err}")).into()
    })
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::persist_failed_run_state;
    use kbolt_types::{KboltError, ScheduleRunResult, ScheduleRunState};

    #[test]
    fn persist_failed_run_state_preserves_original_error_when_save_also_fails() {
        let tmp = tempdir().expect("create tempdir");
        std::fs::write(tmp.path().join("schedules"), "conflict").expect("write conflicting file");

        let err = persist_failed_run_state(
            tmp.path(),
            "s1",
            &mut ScheduleRunState::default(),
            "target update failed".to_string(),
        )
        .expect_err("save failure should be reported");

        match KboltError::from(err) {
            KboltError::Internal(message) => {
                assert!(
                    message.contains("schedule run failed: target update failed"),
                    "unexpected message: {message}"
                );
                assert!(
                    message.contains("failed to save run state"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn persist_failed_run_state_sets_failed_fields_before_saving() {
        let tmp = tempdir().expect("create tempdir");
        let mut state = ScheduleRunState::default();

        persist_failed_run_state(
            tmp.path(),
            "s1",
            &mut state,
            "target update failed".to_string(),
        )
        .expect("save failed state");

        assert!(state.last_started.is_none());
        assert!(state.last_finished.is_some());
        assert_eq!(state.last_result, Some(ScheduleRunResult::Failed));
        assert_eq!(state.last_error.as_deref(), Some("target update failed"));
    }
}
