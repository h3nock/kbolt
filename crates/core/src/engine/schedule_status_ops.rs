use crate::error::CoreError;
use crate::lock::LockMode;
use crate::schedule_backend::{current_schedule_backend, inspect_schedule_backend};
use crate::schedule_state_store::ScheduleRunStateStore;
use crate::schedule_store::ScheduleCatalog;
use crate::Result;
use kbolt_types::{
    KboltError, ScheduleOrphan, ScheduleScope, ScheduleState, ScheduleStatusEntry,
    ScheduleStatusResponse,
};

use super::Engine;

impl Engine {
    pub fn schedule_status(&self) -> Result<ScheduleStatusResponse> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let backend = current_schedule_backend()?;
        let mut schedules = ScheduleCatalog::load(&self.config.config_dir)?.schedules;
        schedules.sort_by_key(|schedule| schedule_id_number(&schedule.id));

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

fn schedule_id_number(id: &str) -> u32 {
    id.strip_prefix('s')
        .and_then(|raw| raw.parse::<u32>().ok())
        .unwrap_or(u32::MAX)
}
