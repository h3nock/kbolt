use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use kbolt_types::{ScheduleDefinition, ScheduleIntervalUnit, ScheduleTrigger, ScheduleWeekday};

use crate::Result;

const MANAGED_UNIT_PREFIX: &str = "kbolt-schedule-";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemdUserPaths {
    pub unit_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemdServiceUnit {
    pub name: String,
    pub path: PathBuf,
    pub contents: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemdTimerUnit {
    pub name: String,
    pub path: PathBuf,
    pub contents: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemdPlan {
    pub services: Vec<SystemdServiceUnit>,
    pub timers: Vec<SystemdTimerUnit>,
    pub stale_paths: Vec<PathBuf>,
}

pub(crate) fn plan_systemd_user(
    paths: &SystemdUserPaths,
    schedules: &[ScheduleDefinition],
    executable: &Path,
) -> Result<SystemdPlan> {
    fs::create_dir_all(&paths.unit_dir)?;

    let mut services = Vec::with_capacity(schedules.len());
    let mut timers = Vec::with_capacity(schedules.len());
    for schedule in schedules {
        let service_name = systemd_service_name(&schedule.id);
        let timer_name = systemd_timer_name(&schedule.id);
        services.push(SystemdServiceUnit {
            name: service_name.clone(),
            path: paths.unit_dir.join(&service_name),
            contents: render_systemd_service(schedule, executable),
        });
        timers.push(SystemdTimerUnit {
            name: timer_name.clone(),
            path: paths.unit_dir.join(&timer_name),
            contents: render_systemd_timer(schedule)?,
        });
    }
    services.sort_by(|left, right| left.name.cmp(&right.name));
    timers.sort_by(|left, right| left.name.cmp(&right.name));

    let desired_paths = services
        .iter()
        .map(|unit| unit.path.clone())
        .chain(timers.iter().map(|unit| unit.path.clone()))
        .collect::<HashSet<_>>();

    let mut stale_paths = Vec::new();
    for entry in fs::read_dir(&paths.unit_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !is_managed_systemd_unit(&path) {
            continue;
        }
        if !desired_paths.contains(&path) {
            stale_paths.push(path);
        }
    }
    stale_paths.sort();

    Ok(SystemdPlan {
        services,
        timers,
        stale_paths,
    })
}

pub(crate) fn render_systemd_service(schedule: &ScheduleDefinition, executable: &Path) -> String {
    format!(
        "[Unit]\nDescription=Run kbolt schedule {schedule_id}\n\n[Service]\nType=oneshot\nExecStart={} __schedule-run {}\n",
        shell_escape(executable),
        shell_escape_arg(&schedule.id),
        schedule_id = schedule.id,
    )
}

pub(crate) fn render_systemd_timer(schedule: &ScheduleDefinition) -> Result<String> {
    let trigger = match &schedule.trigger {
        ScheduleTrigger::Every { interval } => {
            let value = interval.value;
            let unit = match interval.unit {
                ScheduleIntervalUnit::Minutes => "min",
                ScheduleIntervalUnit::Hours => "h",
            };
            format!("OnBootSec={value}{unit}\nOnUnitActiveSec={value}{unit}\n")
        }
        ScheduleTrigger::Daily { time } => {
            let calendar = systemd_calendar(None, time)?;
            format!("OnCalendar={calendar}\nPersistent=true\n")
        }
        ScheduleTrigger::Weekly { weekdays, time } => {
            let calendar = systemd_calendar(Some(weekdays), time)?;
            format!("OnCalendar={calendar}\nPersistent=true\n")
        }
    };

    Ok(format!(
        "[Unit]\nDescription=Run kbolt schedule {schedule_id}\n\n[Timer]\n{trigger}Unit={service_name}\n\n[Install]\nWantedBy=timers.target\n",
        schedule_id = schedule.id,
        service_name = systemd_service_name(&schedule.id),
    ))
}

fn systemd_service_name(schedule_id: &str) -> String {
    format!("{MANAGED_UNIT_PREFIX}{schedule_id}.service")
}

fn systemd_timer_name(schedule_id: &str) -> String {
    format!("{MANAGED_UNIT_PREFIX}{schedule_id}.timer")
}

fn is_managed_systemd_unit(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.starts_with(MANAGED_UNIT_PREFIX)
                && (name.ends_with(".service") || name.ends_with(".timer"))
        })
}

fn systemd_calendar(weekdays: Option<&[ScheduleWeekday]>, time: &str) -> Result<String> {
    let (hour, minute) = parse_canonical_time(time)?;
    let hhmmss = format!("{hour:02}:{minute:02}:00");
    match weekdays {
        Some(days) => Ok(format!(
            "{} *-*-* {hhmmss}",
            days.iter()
                .map(systemd_weekday)
                .collect::<Vec<_>>()
                .join(",")
        )),
        None => Ok(format!("*-*-* {hhmmss}")),
    }
}

fn parse_canonical_time(time: &str) -> Result<(u32, u32)> {
    let (hour, minute) = time.split_once(':').ok_or_else(|| {
        kbolt_types::KboltError::InvalidInput(format!("invalid schedule time: {time}"))
    })?;
    let hour = hour.parse::<u32>().map_err(|_| {
        kbolt_types::KboltError::InvalidInput(format!("invalid schedule time: {time}"))
    })?;
    let minute = minute.parse::<u32>().map_err(|_| {
        kbolt_types::KboltError::InvalidInput(format!("invalid schedule time: {time}"))
    })?;
    Ok((hour, minute))
}

fn systemd_weekday(weekday: &ScheduleWeekday) -> &'static str {
    match weekday {
        ScheduleWeekday::Mon => "Mon",
        ScheduleWeekday::Tue => "Tue",
        ScheduleWeekday::Wed => "Wed",
        ScheduleWeekday::Thu => "Thu",
        ScheduleWeekday::Fri => "Fri",
        ScheduleWeekday::Sat => "Sat",
        ScheduleWeekday::Sun => "Sun",
    }
}

fn shell_escape(path: &Path) -> String {
    shell_escape_arg(&path.display().to_string())
}

fn shell_escape_arg(value: &str) -> String {
    let escaped = value.replace('\'', "'\\''");
    format!("'{escaped}'")
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::tempdir;

    use super::{
        plan_systemd_user, render_systemd_service, render_systemd_timer, SystemdUserPaths,
    };
    use kbolt_types::{
        ScheduleDefinition, ScheduleInterval, ScheduleIntervalUnit, ScheduleScope, ScheduleTrigger,
        ScheduleWeekday,
    };

    #[test]
    fn render_systemd_service_and_interval_timer() {
        let schedule = ScheduleDefinition {
            id: "s1".to_string(),
            trigger: ScheduleTrigger::Every {
                interval: ScheduleInterval {
                    value: 30,
                    unit: ScheduleIntervalUnit::Minutes,
                },
            },
            scope: ScheduleScope::All,
        };

        let service = render_systemd_service(&schedule, Path::new("/usr/local/bin/kbolt"));
        let timer = render_systemd_timer(&schedule).expect("render timer");

        assert!(service.contains("ExecStart='/usr/local/bin/kbolt' __schedule-run 's1'"));
        assert!(timer.contains("OnBootSec=30min"));
        assert!(timer.contains("OnUnitActiveSec=30min"));
        assert!(timer.contains("Unit=kbolt-schedule-s1.service"));
        assert!(timer.contains("WantedBy=timers.target"));
    }

    #[test]
    fn render_systemd_timer_for_weekly_schedule_uses_on_calendar() {
        let timer = render_systemd_timer(&ScheduleDefinition {
            id: "s2".to_string(),
            trigger: ScheduleTrigger::Weekly {
                weekdays: vec![ScheduleWeekday::Mon, ScheduleWeekday::Fri],
                time: "15:00".to_string(),
            },
            scope: ScheduleScope::Space {
                space: "work".to_string(),
            },
        })
        .expect("render weekly timer");

        assert!(timer.contains("OnCalendar=Mon,Fri *-*-* 15:00:00"));
        assert!(timer.contains("Persistent=true"));
    }

    #[test]
    fn plan_systemd_user_detects_stale_managed_units() {
        let tmp = tempdir().expect("create tempdir");
        let paths = SystemdUserPaths {
            unit_dir: tmp.path().join("systemd-user"),
        };
        std::fs::create_dir_all(&paths.unit_dir).expect("create unit dir");
        std::fs::write(
            paths.unit_dir.join("kbolt-schedule-s9.service"),
            "stale service",
        )
        .expect("write stale service");
        std::fs::write(
            paths.unit_dir.join("kbolt-schedule-s9.timer"),
            "stale timer",
        )
        .expect("write stale timer");
        std::fs::write(paths.unit_dir.join("foreign.timer"), "foreign timer")
            .expect("write foreign timer");

        let plan = plan_systemd_user(
            &paths,
            &[ScheduleDefinition {
                id: "s1".to_string(),
                trigger: ScheduleTrigger::Daily {
                    time: "09:00".to_string(),
                },
                scope: ScheduleScope::All,
            }],
            Path::new("/usr/local/bin/kbolt"),
        )
        .expect("plan systemd units");

        assert_eq!(plan.services.len(), 1);
        assert_eq!(plan.timers.len(), 1);
        assert_eq!(plan.services[0].name, "kbolt-schedule-s1.service");
        assert_eq!(plan.timers[0].name, "kbolt-schedule-s1.timer");
        assert_eq!(
            plan.stale_paths,
            vec![
                paths.unit_dir.join("kbolt-schedule-s9.service"),
                paths.unit_dir.join("kbolt-schedule-s9.timer"),
            ]
        );
    }
}
