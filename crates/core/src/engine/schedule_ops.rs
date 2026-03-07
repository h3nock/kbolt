use std::collections::{BTreeSet, HashSet};

use kbolt_types::{
    AddScheduleRequest, KboltError, RemoveScheduleRequest, RemoveScheduleSelector,
    ScheduleAddResponse, ScheduleDefinition, ScheduleInterval, ScheduleIntervalUnit,
    ScheduleRemoveResponse, ScheduleScope, ScheduleTrigger, ScheduleWeekday,
};

use crate::lock::LockMode;
use crate::schedule_backend::reconcile_schedule_backend;
use crate::schedule_state_store::ScheduleRunStateStore;
use crate::schedule_store::ScheduleCatalog;
use crate::Result;

use super::Engine;

const MIN_SCHEDULE_INTERVAL_MINUTES: u32 = 5;

impl Engine {
    pub fn add_schedule(&self, req: AddScheduleRequest) -> Result<ScheduleAddResponse> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let trigger = normalize_schedule_trigger(req.trigger)?;
        let scope = self.normalize_schedule_scope(req.scope, true)?;
        let mut catalog = ScheduleCatalog::load(&self.config.config_dir)?;

        if let Some(existing) = catalog
            .schedules
            .iter()
            .find(|schedule| schedule.trigger == trigger && schedule.scope == scope)
        {
            return Err(KboltError::InvalidInput(format!(
                "schedule already exists: {}",
                existing.id
            ))
            .into());
        }

        let schedule_id = format!("s{}", catalog.next_id);
        catalog.next_id = catalog.next_id.checked_add(1).ok_or_else(|| {
            KboltError::InvalidInput("cannot create schedule: schedule ids exhausted".to_string())
        })?;

        let schedule = ScheduleDefinition {
            id: schedule_id,
            trigger,
            scope,
        };
        catalog.schedules.push(schedule.clone());
        catalog.save(&self.config.config_dir)?;
        let backend = match reconcile_schedule_backend(
            &self.config.config_dir,
            &self.config.cache_dir,
            &catalog.schedules,
        ) {
            Ok(backend) => backend,
            Err(err) => {
                return Err(KboltError::Internal(format!(
                    "schedule {} was saved, but backend reconcile failed: {err}",
                    schedule.id
                ))
                .into())
            }
        };

        Ok(ScheduleAddResponse { schedule, backend })
    }

    pub fn list_schedules(&self) -> Result<Vec<ScheduleDefinition>> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let mut schedules = ScheduleCatalog::load(&self.config.config_dir)?.schedules;
        schedules.sort_by_key(|schedule| schedule_id_number(&schedule.id));
        Ok(schedules)
    }

    pub fn remove_schedule(&self, req: RemoveScheduleRequest) -> Result<ScheduleRemoveResponse> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let mut catalog = ScheduleCatalog::load(&self.config.config_dir)?;
        let removed_ids = self.resolve_removed_schedule_ids(req.selector, &catalog.schedules)?;

        if removed_ids.is_empty() {
            return Ok(ScheduleRemoveResponse { removed_ids });
        }

        let removed = removed_ids.iter().cloned().collect::<HashSet<_>>();
        catalog
            .schedules
            .retain(|schedule| !removed.contains(&schedule.id));
        catalog.save(&self.config.config_dir)?;
        for schedule_id in &removed_ids {
            ScheduleRunStateStore::remove(&self.config.cache_dir, schedule_id)?;
        }
        if let Err(err) = reconcile_schedule_backend(
            &self.config.config_dir,
            &self.config.cache_dir,
            &catalog.schedules,
        ) {
            return Err(KboltError::Internal(format!(
                "removed schedules {}, but backend reconcile failed: {err}",
                removed_ids.join(", ")
            ))
            .into());
        }

        Ok(ScheduleRemoveResponse { removed_ids })
    }

    fn normalize_schedule_scope(
        &self,
        scope: ScheduleScope,
        validate_targets: bool,
    ) -> Result<ScheduleScope> {
        match scope {
            ScheduleScope::All => Ok(ScheduleScope::All),
            ScheduleScope::Space { space } => {
                let normalized_space = normalize_scope_name("space", &space)?;
                if validate_targets {
                    let resolved = self.storage.get_space(&normalized_space)?;
                    return Ok(ScheduleScope::Space {
                        space: resolved.name,
                    });
                }

                Ok(ScheduleScope::Space {
                    space: normalized_space,
                })
            }
            ScheduleScope::Collections { space, collections } => {
                let normalized_space = normalize_scope_name("space", &space)?;
                let normalized_collections = normalize_collection_names(collections)?;

                if validate_targets {
                    let resolved = self.storage.get_space(&normalized_space)?;
                    for collection in &normalized_collections {
                        self.storage.get_collection(resolved.id, collection)?;
                    }
                    return Ok(ScheduleScope::Collections {
                        space: resolved.name,
                        collections: normalized_collections,
                    });
                }

                Ok(ScheduleScope::Collections {
                    space: normalized_space,
                    collections: normalized_collections,
                })
            }
        }
    }

    fn resolve_removed_schedule_ids(
        &self,
        selector: RemoveScheduleSelector,
        schedules: &[ScheduleDefinition],
    ) -> Result<Vec<String>> {
        match selector {
            RemoveScheduleSelector::All => Ok(schedules
                .iter()
                .map(|schedule| schedule.id.clone())
                .collect()),
            RemoveScheduleSelector::Id { id } => {
                let normalized_id = normalize_schedule_id(&id)?;
                if schedules
                    .iter()
                    .any(|schedule| schedule.id == normalized_id)
                {
                    return Ok(vec![normalized_id]);
                }

                Err(KboltError::InvalidInput(format!("schedule not found: {normalized_id}")).into())
            }
            RemoveScheduleSelector::Scope { scope } => {
                let normalized_scope = self.normalize_schedule_scope(scope, false)?;
                let mut matches = schedules
                    .iter()
                    .filter(|schedule| schedule.scope == normalized_scope)
                    .map(|schedule| schedule.id.clone())
                    .collect::<Vec<_>>();
                matches.sort_by_key(|id| schedule_id_number(id));

                match matches.len() {
                    0 => Err(KboltError::InvalidInput(
                        "no schedules matched the requested scope".to_string(),
                    )
                    .into()),
                    1 => Ok(matches),
                    _ => Err(KboltError::InvalidInput(format!(
                        "schedule scope matched multiple schedules: {}",
                        matches.join(", ")
                    ))
                    .into()),
                }
            }
        }
    }
}

pub(super) fn load_schedule_definition(
    config_dir: &std::path::Path,
    id: &str,
) -> Result<ScheduleDefinition> {
    let normalized_id = normalize_schedule_id(id)?;
    let catalog = ScheduleCatalog::load(config_dir)?;
    catalog
        .schedules
        .into_iter()
        .find(|schedule| schedule.id == normalized_id)
        .ok_or_else(|| {
            KboltError::InvalidInput(format!("schedule not found: {normalized_id}")).into()
        })
}

fn normalize_schedule_trigger(trigger: ScheduleTrigger) -> Result<ScheduleTrigger> {
    match trigger {
        ScheduleTrigger::Every { interval } => Ok(ScheduleTrigger::Every {
            interval: normalize_schedule_interval(interval)?,
        }),
        ScheduleTrigger::Daily { time } => Ok(ScheduleTrigger::Daily {
            time: normalize_schedule_time(&time)?,
        }),
        ScheduleTrigger::Weekly { weekdays, time } => Ok(ScheduleTrigger::Weekly {
            weekdays: normalize_schedule_weekdays(weekdays)?,
            time: normalize_schedule_time(&time)?,
        }),
    }
}

fn normalize_schedule_interval(interval: ScheduleInterval) -> Result<ScheduleInterval> {
    if interval.value == 0 {
        return Err(KboltError::InvalidInput(
            "schedule interval must be greater than zero".to_string(),
        )
        .into());
    }

    match interval.unit {
        ScheduleIntervalUnit::Minutes if interval.value < MIN_SCHEDULE_INTERVAL_MINUTES => {
            Err(KboltError::InvalidInput(format!(
                "schedule interval must be at least {MIN_SCHEDULE_INTERVAL_MINUTES} minutes"
            ))
            .into())
        }
        ScheduleIntervalUnit::Minutes | ScheduleIntervalUnit::Hours => Ok(interval),
    }
}

fn normalize_schedule_weekdays(weekdays: Vec<ScheduleWeekday>) -> Result<Vec<ScheduleWeekday>> {
    let normalized = weekdays.into_iter().collect::<BTreeSet<_>>();
    if normalized.is_empty() {
        return Err(KboltError::InvalidInput(
            "weekly schedules require at least one weekday".to_string(),
        )
        .into());
    }

    Ok(normalized.into_iter().collect())
}

fn normalize_schedule_time(input: &str) -> Result<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(KboltError::InvalidInput("schedule time must not be empty".to_string()).into());
    }

    let collapsed = trimmed
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();

    let (time_part, meridiem) = if let Some(time) = collapsed.strip_suffix("am") {
        (time, Some("am"))
    } else if let Some(time) = collapsed.strip_suffix("pm") {
        (time, Some("pm"))
    } else {
        (collapsed.as_str(), None)
    };

    if time_part.is_empty() {
        return Err(invalid_schedule_time(input).into());
    }

    let (mut hour, minute) = if let Some((hour_part, minute_part)) = time_part.split_once(':') {
        if minute_part.contains(':') {
            return Err(invalid_schedule_time(input).into());
        }
        (
            parse_time_component(hour_part, input)?,
            parse_time_component(minute_part, input)?,
        )
    } else {
        if meridiem.is_none() {
            return Err(invalid_schedule_time(input).into());
        }
        (parse_time_component(time_part, input)?, 0)
    };

    if minute > 59 {
        return Err(invalid_schedule_time(input).into());
    }

    match meridiem {
        Some("am") => {
            if hour == 0 || hour > 12 {
                return Err(invalid_schedule_time(input).into());
            }
            if hour == 12 {
                hour = 0;
            }
        }
        Some("pm") => {
            if hour == 0 || hour > 12 {
                return Err(invalid_schedule_time(input).into());
            }
            if hour != 12 {
                hour += 12;
            }
        }
        None => {
            if hour > 23 {
                return Err(invalid_schedule_time(input).into());
            }
        }
        Some(_) => unreachable!("only am/pm meridiems are supported"),
    }

    Ok(format!("{hour:02}:{minute:02}"))
}

fn parse_time_component(component: &str, input: &str) -> Result<u32> {
    if component.is_empty() {
        return Err(invalid_schedule_time(input).into());
    }

    component
        .parse::<u32>()
        .map_err(|_| invalid_schedule_time(input).into())
}

fn invalid_schedule_time(input: &str) -> KboltError {
    KboltError::InvalidInput(format!(
        "invalid schedule time '{input}': use HH:MM, 3pm, or 3:00pm"
    ))
}

fn normalize_scope_name(label: &str, name: &str) -> Result<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(KboltError::InvalidInput(format!("{label} name must not be empty")).into());
    }
    Ok(trimmed.to_string())
}

fn normalize_collection_names(collections: Vec<String>) -> Result<Vec<String>> {
    let mut normalized = BTreeSet::new();
    for collection in collections {
        let trimmed = collection.trim();
        if trimmed.is_empty() {
            return Err(
                KboltError::InvalidInput("collection names must not be empty".to_string()).into(),
            );
        }
        normalized.insert(trimmed.to_string());
    }

    if normalized.is_empty() {
        return Err(KboltError::InvalidInput(
            "collection scope must include at least one collection".to_string(),
        )
        .into());
    }

    Ok(normalized.into_iter().collect())
}

fn normalize_schedule_id(id: &str) -> Result<String> {
    let normalized = id.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(KboltError::InvalidInput("schedule id must not be empty".to_string()).into());
    }

    if schedule_id_number(&normalized) == 0 {
        return Err(KboltError::InvalidInput(format!("invalid schedule id: {id}")).into());
    }

    Ok(normalized)
}

fn schedule_id_number(id: &str) -> u32 {
    id.strip_prefix('s')
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0)
}
