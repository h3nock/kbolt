use crate::error::CoreError;
use crate::schedule_backend::{current_schedule_backend, inspect_schedule_backend};
use crate::schedule_state_store::ScheduleRunStateStore;
use crate::schedule_store::ScheduleCatalog;
use crate::schedule_support::schedule_id_sort_key;
use crate::Result;
use kbolt_types::{
    KboltError, ScheduleOrphan, ScheduleScope, ScheduleState, ScheduleStatusEntry,
    ScheduleStatusResponse,
};

use super::Engine;

impl Engine {
    pub fn schedule_status(&self) -> Result<ScheduleStatusResponse> {
        let backend = current_schedule_backend()?;
        let mut schedules = ScheduleCatalog::load(&self.config.config_dir)?.schedules;
        schedules.sort_by_key(|schedule| schedule_id_sort_key(&schedule.id));

        let inspection =
            inspect_schedule_backend(&self.config.config_dir, &self.config.cache_dir, &schedules)?;
        let mut entries = Vec::with_capacity(schedules.len());

        for schedule in schedules {
            let state = if self.schedule_targets_missing(&schedule.scope)? {
                ScheduleState::TargetMissing
            } else if inspection.drifted_ids.contains(&schedule.id) {
                ScheduleState::Drifted
            } else {
                ScheduleState::Installed
            };

            let run_state = ScheduleRunStateStore::load(&self.config.cache_dir, &schedule.id)?;
            entries.push(ScheduleStatusEntry {
                schedule,
                backend,
                state,
                run_state,
            });
        }

        let orphans = inspection
            .orphan_ids
            .into_iter()
            .map(|id| ScheduleOrphan { id, backend })
            .collect();

        Ok(ScheduleStatusResponse {
            schedules: entries,
            orphans,
        })
    }

    fn schedule_targets_missing(&self, scope: &ScheduleScope) -> Result<bool> {
        match scope {
            ScheduleScope::All => Ok(false),
            ScheduleScope::Space { space } => match self.storage.get_space(space) {
                Ok(_) => Ok(false),
                Err(CoreError::Domain(KboltError::SpaceNotFound { .. })) => Ok(true),
                Err(err) => Err(err),
            },
            ScheduleScope::Collections { space, collections } => {
                let resolved_space = match self.storage.get_space(space) {
                    Ok(space) => space,
                    Err(CoreError::Domain(KboltError::SpaceNotFound { .. })) => return Ok(true),
                    Err(err) => return Err(err),
                };

                for collection in collections {
                    match self.storage.get_collection(resolved_space.id, collection) {
                        Ok(_) => {}
                        Err(CoreError::Domain(KboltError::CollectionNotFound { .. })) => {
                            return Ok(true)
                        }
                        Err(err) => return Err(err),
                    }
                }

                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use fs2::FileExt;
    use std::fs::OpenOptions;
    use std::mem;

    use tempfile::tempdir;

    use super::Engine;
    use crate::config::{ChunkingConfig, Config, RankingConfig, ReapingConfig};
    use crate::storage::Storage;
    use kbolt_types::{AddScheduleRequest, ScheduleScope, ScheduleTrigger};

    #[test]
    fn schedule_status_succeeds_while_global_lock_is_held() {
        let engine = test_engine();
        engine
            .add_schedule(AddScheduleRequest {
                trigger: ScheduleTrigger::Daily {
                    time: "09:00".to_string(),
                },
                scope: ScheduleScope::All,
            })
            .expect("add schedule");

        let lock_path = engine.config().cache_dir.join("kbolt.lock");
        std::fs::create_dir_all(&engine.config().cache_dir).expect("create cache dir");
        let holder = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .expect("open lock file");
        FileExt::try_lock_exclusive(&holder).expect("acquire global lock");

        let status = engine.schedule_status().expect("load schedule status");
        assert_eq!(status.schedules.len(), 1);
        assert!(status.orphans.is_empty());
    }

    fn test_engine() -> Engine {
        let root = tempdir().expect("create temp root");
        let root_path = root.path().to_path_buf();
        mem::forget(root);
        let config_dir = root_path.join("config");
        let cache_dir = root_path.join("cache");
        let storage = Storage::new(&cache_dir).expect("create storage");
        let config = Config {
            config_dir,
            cache_dir,
            default_space: None,
            providers: std::collections::HashMap::new(),
            roles: crate::config::RoleBindingsConfig::default(),
            reaping: ReapingConfig { days: 7 },
            chunking: ChunkingConfig::default(),
            ranking: RankingConfig::default(),
        };
        Engine::from_parts(storage, config)
    }
}
