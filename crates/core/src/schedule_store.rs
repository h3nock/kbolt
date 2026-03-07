use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use kbolt_types::{KboltError, ScheduleDefinition};
use serde::{Deserialize, Serialize};

use crate::error::Result;

const SCHEDULES_FILENAME: &str = "schedules.toml";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ScheduleCatalog {
    #[serde(default = "default_next_id")]
    pub(crate) next_id: u32,
    #[serde(default)]
    pub(crate) schedules: Vec<ScheduleDefinition>,
}

impl Default for ScheduleCatalog {
    fn default() -> Self {
        Self {
            next_id: default_next_id(),
            schedules: Vec::new(),
        }
    }
}

impl ScheduleCatalog {
    pub(crate) fn load(config_dir: &Path) -> Result<Self> {
        fs::create_dir_all(config_dir)?;

        let schedule_file = Self::file_path(config_dir);
        let raw = match fs::read_to_string(&schedule_file) {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(err) => return Err(err.into()),
        };

        let mut catalog: Self = toml::from_str(&raw).map_err(|err| {
            KboltError::Config(format!(
                "invalid schedule file {}: {err}",
                schedule_file.display()
            ))
        })?;
        catalog.normalize(&schedule_file)?;
        Ok(catalog)
    }

    pub(crate) fn save(&self, config_dir: &Path) -> Result<()> {
        fs::create_dir_all(config_dir)?;

        let schedule_file = Self::file_path(config_dir);
        let mut normalized = self.clone();
        normalized.normalize(&schedule_file)?;

        let serialized = toml::to_string_pretty(&normalized)?;
        let content = if serialized.ends_with('\n') {
            serialized
        } else {
            format!("{serialized}\n")
        };

        let tmp_file = config_dir.join(format!("{SCHEDULES_FILENAME}.tmp"));
        fs::write(&tmp_file, content)?;
        fs::rename(tmp_file, schedule_file)?;
        Ok(())
    }

    pub(crate) fn file_path(config_dir: &Path) -> PathBuf {
        config_dir.join(SCHEDULES_FILENAME)
    }

    fn normalize(&mut self, schedule_file: &Path) -> Result<()> {
        let mut seen_ids = HashSet::new();
        let mut max_schedule_id = 0u32;

        for schedule in &self.schedules {
            let id_number = parse_schedule_id(&schedule.id, schedule_file)?;
            if !seen_ids.insert(id_number) {
                return Err(KboltError::Config(format!(
                    "invalid schedule file {}: duplicate schedule id {}",
                    schedule_file.display(),
                    schedule.id
                ))
                .into());
            }
            max_schedule_id = max_schedule_id.max(id_number);
        }

        let recovered_next_id = max_schedule_id
            .checked_add(1)
            .ok_or_else(|| {
                KboltError::Config(format!(
                    "invalid schedule file {}: schedule ids exceeded supported range",
                    schedule_file.display()
                ))
            })?
            .max(default_next_id());

        self.next_id = self.next_id.max(recovered_next_id);
        Ok(())
    }
}

fn default_next_id() -> u32 {
    1
}

fn parse_schedule_id(id: &str, schedule_file: &Path) -> Result<u32> {
    let Some(raw_number) = id.strip_prefix('s') else {
        return Err(invalid_schedule_id_error(schedule_file, id).into());
    };

    let parsed = raw_number
        .parse::<u32>()
        .map_err(|_| invalid_schedule_id_error(schedule_file, id))?;

    if parsed == 0 {
        return Err(invalid_schedule_id_error(schedule_file, id).into());
    }

    Ok(parsed)
}

fn invalid_schedule_id_error(schedule_file: &Path, id: &str) -> KboltError {
    KboltError::Config(format!(
        "invalid schedule file {}: schedule ids must use the form s<number>: {id}",
        schedule_file.display()
    ))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use kbolt_types::{
        ScheduleDefinition, ScheduleInterval, ScheduleIntervalUnit, ScheduleScope, ScheduleTrigger,
        ScheduleWeekday,
    };
    use tempfile::tempdir;

    use super::{ScheduleCatalog, SCHEDULES_FILENAME};

    #[test]
    fn load_returns_default_catalog_when_schedule_file_is_missing() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");

        let catalog = ScheduleCatalog::load(&config_dir).expect("load catalog");

        assert_eq!(catalog, ScheduleCatalog::default());
        assert!(config_dir.is_dir());
        assert!(!config_dir.join(SCHEDULES_FILENAME).exists());
    }

    #[test]
    fn save_and_load_roundtrip_schedule_catalog() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        let catalog = ScheduleCatalog {
            next_id: 4,
            schedules: vec![
                schedule_definition(
                    "s1",
                    ScheduleTrigger::Every {
                        interval: ScheduleInterval {
                            value: 30,
                            unit: ScheduleIntervalUnit::Minutes,
                        },
                    },
                    ScheduleScope::All,
                ),
                schedule_definition(
                    "s3",
                    ScheduleTrigger::Weekly {
                        weekdays: vec![ScheduleWeekday::Mon, ScheduleWeekday::Fri],
                        time: "15:00".to_string(),
                    },
                    ScheduleScope::Collections {
                        space: "work".to_string(),
                        collections: vec!["api".to_string(), "docs".to_string()],
                    },
                ),
            ],
        };

        catalog.save(&config_dir).expect("save catalog");
        let loaded = ScheduleCatalog::load(&config_dir).expect("load catalog");

        assert_eq!(loaded, catalog);
    }

    #[test]
    fn load_recovers_next_id_from_existing_schedule_ids() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        let schedule_file = config_dir.join(SCHEDULES_FILENAME);
        fs::create_dir_all(&config_dir).expect("create config dir");
        fs::write(
            &schedule_file,
            r#"
next_id = 1

[[schedules]]
id = "s2"

[schedules.trigger]
kind = "daily"
time = "03:00"

[schedules.scope]
kind = "space"
space = "work"

[[schedules]]
id = "s5"

[schedules.trigger]
kind = "weekly"
weekdays = ["mon", "fri"]
time = "15:00"

[schedules.scope]
kind = "collections"
space = "work"
collections = ["api", "docs"]
"#,
        )
        .expect("write schedule file");

        let catalog = ScheduleCatalog::load(&config_dir).expect("load catalog");

        assert_eq!(catalog.next_id, 6);
        assert_eq!(catalog.schedules.len(), 2);
    }

    #[test]
    fn load_rejects_invalid_schedule_id() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        let schedule_file = config_dir.join(SCHEDULES_FILENAME);
        fs::create_dir_all(&config_dir).expect("create config dir");
        fs::write(
            &schedule_file,
            r#"
next_id = 2

[[schedules]]
id = "schedule-1"

[schedules.trigger]
kind = "every"

[schedules.trigger.interval]
value = 30
unit = "minutes"

[schedules.scope]
kind = "all"
"#,
        )
        .expect("write schedule file");

        let err = ScheduleCatalog::load(&config_dir).expect_err("invalid schedule id should fail");
        let message = err.to_string();

        assert!(
            message.contains("invalid schedule file"),
            "unexpected message: {message}"
        );
        assert!(
            message.contains(&schedule_file.display().to_string()),
            "unexpected message: {message}"
        );
        assert!(
            message.contains("schedule ids must use the form s<number>"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn load_rejects_duplicate_schedule_ids() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        let schedule_file = config_dir.join(SCHEDULES_FILENAME);
        fs::create_dir_all(&config_dir).expect("create config dir");
        fs::write(
            &schedule_file,
            r#"
next_id = 3

[[schedules]]
id = "s1"

[schedules.trigger]
kind = "daily"
time = "03:00"

[schedules.scope]
kind = "all"

[[schedules]]
id = "s1"

[schedules.trigger]
kind = "daily"
time = "15:00"

[schedules.scope]
kind = "space"
space = "work"
"#,
        )
        .expect("write schedule file");

        let err = ScheduleCatalog::load(&config_dir).expect_err("duplicate ids should fail");
        let message = err.to_string();

        assert!(
            message.contains("duplicate schedule id s1"),
            "unexpected message: {message}"
        );
    }

    fn schedule_definition(
        id: &str,
        trigger: ScheduleTrigger,
        scope: ScheduleScope,
    ) -> ScheduleDefinition {
        ScheduleDefinition {
            id: id.to_string(),
            trigger,
            scope,
        }
    }
}
