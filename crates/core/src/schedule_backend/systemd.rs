use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use kbolt_types::{
    KboltError, ScheduleDefinition, ScheduleIntervalUnit, ScheduleTrigger, ScheduleWeekday,
};

use super::{command_failure, write_if_changed, BackendInspection, CommandRunner};
use crate::Result;

const MANAGED_UNIT_PREFIX: &str = "kbolt-schedule-";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemdUserPaths {
    pub unit_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemdServiceUnit {
    pub schedule_id: String,
    pub name: String,
    pub path: PathBuf,
    pub contents: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemdTimerUnit {
    pub schedule_id: String,
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
    let mut services = Vec::with_capacity(schedules.len());
    let mut timers = Vec::with_capacity(schedules.len());
    for schedule in schedules {
        let service_name = systemd_service_name(&schedule.id);
        let timer_name = systemd_timer_name(&schedule.id);
        services.push(SystemdServiceUnit {
            schedule_id: schedule.id.clone(),
            name: service_name.clone(),
            path: paths.unit_dir.join(&service_name),
            contents: render_systemd_service(schedule, executable),
        });
        timers.push(SystemdTimerUnit {
            schedule_id: schedule.id.clone(),
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
    if let Ok(entries) = fs::read_dir(&paths.unit_dir) {
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !is_managed_systemd_unit(&path) {
                continue;
            }
            if !desired_paths.contains(&path) {
                stale_paths.push(path);
            }
        }
    }
    stale_paths.sort();

    Ok(SystemdPlan {
        services,
        timers,
        stale_paths,
    })
}

pub(crate) fn inspect_systemd_user(
    paths: &SystemdUserPaths,
    schedules: &[ScheduleDefinition],
    executable: &Path,
    runner: &dyn CommandRunner,
) -> Result<BackendInspection> {
    let plan = plan_systemd_user(paths, schedules, executable)?;
    let mut drifted_ids = HashSet::new();
    let planned_services = plan
        .services
        .iter()
        .map(|service| (service.schedule_id.as_str(), service))
        .collect::<HashMap<_, _>>();
    let planned_timers = plan
        .timers
        .iter()
        .map(|timer| (timer.schedule_id.as_str(), timer))
        .collect::<HashMap<_, _>>();

    for schedule in schedules {
        let service = planned_services.get(schedule.id.as_str()).ok_or_else(|| {
            KboltError::Internal(format!(
                "planned systemd service missing for schedule {}",
                schedule.id
            ))
        })?;
        let timer = planned_timers.get(schedule.id.as_str()).ok_or_else(|| {
            KboltError::Internal(format!(
                "planned systemd timer missing for schedule {}",
                schedule.id
            ))
        })?;
        if systemd_unit_is_drifted(&service.path, &service.contents)?
            || systemd_unit_is_drifted(&timer.path, &timer.contents)?
            || !is_systemd_timer_enabled(runner, &timer.name)?
        {
            drifted_ids.insert(schedule.id.clone());
        }
    }

    let orphan_ids = plan
        .stale_paths
        .iter()
        .filter_map(|path| schedule_id_from_systemd_path(path))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut orphan_ids = orphan_ids;
    orphan_ids.sort_by_key(|id| schedule_id_number(id));

    Ok(BackendInspection {
        drifted_ids,
        orphan_ids,
    })
}

pub(crate) fn reconcile_systemd_user(
    paths: &SystemdUserPaths,
    schedules: &[ScheduleDefinition],
    executable: &Path,
    runner: &dyn CommandRunner,
) -> Result<()> {
    fs::create_dir_all(&paths.unit_dir)?;

    let plan = plan_systemd_user(paths, schedules, executable)?;
    let mut changed_backend = false;
    let stale_ids = plan
        .stale_paths
        .iter()
        .filter_map(|path| schedule_id_from_systemd_path(path))
        .collect::<HashSet<_>>();

    for schedule_id in &stale_ids {
        disable_systemd_timer(runner, &systemd_timer_name(schedule_id))?;
    }

    for stale_path in &plan.stale_paths {
        match fs::remove_file(stale_path) {
            Ok(()) => changed_backend = true,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }

    for service in &plan.services {
        changed_backend |= write_if_changed(&service.path, &service.contents)?;
    }

    for timer in &plan.timers {
        changed_backend |= write_if_changed(&timer.path, &timer.contents)?;
    }

    if changed_backend {
        let output = runner.run("systemctl", &["--user", "daemon-reload"])?;
        if !output.success {
            return Err(command_failure("systemctl", &["--user", "daemon-reload"], &output).into());
        }
    }

    for timer in &plan.timers {
        let output = runner.run("systemctl", &["--user", "enable", "--now", &timer.name])?;
        if !output.success {
            return Err(command_failure(
                "systemctl",
                &["--user", "enable", "--now", &timer.name],
                &output,
            )
            .into());
        }
    }

    Ok(())
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

fn systemd_unit_is_drifted(path: &Path, expected: &str) -> Result<bool> {
    match fs::read_to_string(path) {
        Ok(existing) => Ok(existing != expected),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(err) => Err(err.into()),
    }
}

fn is_systemd_timer_enabled(runner: &dyn CommandRunner, timer_name: &str) -> Result<bool> {
    let output = runner.run("systemctl", &["--user", "is-enabled", timer_name])?;
    if output.success {
        return Ok(true);
    }

    if is_missing_systemd_unit(&output.stdout) || is_missing_systemd_unit(&output.stderr) {
        return Ok(false);
    }

    if output.stdout.trim() == "disabled" {
        return Ok(false);
    }

    Err(command_failure("systemctl", &["--user", "is-enabled", timer_name], &output).into())
}

fn disable_systemd_timer(runner: &dyn CommandRunner, timer_name: &str) -> Result<()> {
    let output = runner.run("systemctl", &["--user", "disable", "--now", timer_name])?;
    if output.success
        || is_missing_systemd_unit(&output.stdout)
        || is_missing_systemd_unit(&output.stderr)
    {
        return Ok(());
    }

    Err(command_failure(
        "systemctl",
        &["--user", "disable", "--now", timer_name],
        &output,
    )
    .into())
}

fn is_missing_systemd_unit(output: &str) -> bool {
    let normalized = output.to_ascii_lowercase();
    normalized.contains("not loaded")
        || normalized.contains("not found")
        || normalized.contains("no such file")
}

fn schedule_id_from_systemd_path(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| {
            name.strip_prefix(MANAGED_UNIT_PREFIX).and_then(|rest| {
                rest.strip_suffix(".service")
                    .or_else(|| rest.strip_suffix(".timer"))
            })
        })
        .map(ToString::to_string)
}

fn schedule_id_number(id: &str) -> u32 {
    id.strip_prefix('s')
        .and_then(|raw| raw.parse::<u32>().ok())
        .unwrap_or(u32::MAX)
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
    use std::collections::HashSet;
    use std::path::Path;
    use std::sync::Mutex;

    use tempfile::tempdir;

    use super::{
        inspect_systemd_user, plan_systemd_user, reconcile_systemd_user, render_systemd_service,
        render_systemd_timer, SystemdUserPaths,
    };
    use crate::schedule_backend::{CommandOutput, CommandRunner};
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

    #[test]
    fn inspect_systemd_marks_missing_units_as_drifted_and_extra_units_as_orphans() {
        let tmp = tempdir().expect("create tempdir");
        let schedule = ScheduleDefinition {
            id: "s1".to_string(),
            trigger: ScheduleTrigger::Daily {
                time: "09:00".to_string(),
            },
            scope: ScheduleScope::All,
        };
        let paths = SystemdUserPaths {
            unit_dir: tmp.path().join("systemd-user"),
        };
        std::fs::create_dir_all(&paths.unit_dir).expect("create unit dir");
        std::fs::write(
            paths.unit_dir.join("kbolt-schedule-s9.timer"),
            "orphan timer",
        )
        .expect("write orphan timer");

        let inspection = inspect_systemd_user(
            &paths,
            &[schedule],
            Path::new("/usr/local/bin/kbolt"),
            &NoopRunner,
        )
        .expect("inspect systemd");

        assert_eq!(inspection.drifted_ids, HashSet::from(["s1".to_string()]));
        assert_eq!(inspection.orphan_ids, vec!["s9".to_string()]);
    }

    #[test]
    fn inspect_systemd_matches_units_by_schedule_id_when_ids_reach_double_digits() {
        let tmp = tempdir().expect("create tempdir");
        let schedules = vec![
            ScheduleDefinition {
                id: "s2".to_string(),
                trigger: ScheduleTrigger::Daily {
                    time: "09:00".to_string(),
                },
                scope: ScheduleScope::All,
            },
            ScheduleDefinition {
                id: "s10".to_string(),
                trigger: ScheduleTrigger::Daily {
                    time: "10:00".to_string(),
                },
                scope: ScheduleScope::All,
            },
        ];
        let paths = SystemdUserPaths {
            unit_dir: tmp.path().join("systemd-user"),
        };
        std::fs::create_dir_all(&paths.unit_dir).expect("create unit dir");

        let plan =
            plan_systemd_user(&paths, &schedules, Path::new("/usr/local/bin/kbolt")).expect("plan");
        let s2_service = plan
            .services
            .iter()
            .find(|service| service.schedule_id == "s2")
            .expect("s2 service");
        let s2_timer = plan
            .timers
            .iter()
            .find(|timer| timer.schedule_id == "s2")
            .expect("s2 timer");
        std::fs::write(&s2_service.path, &s2_service.contents).expect("write s2 service");
        std::fs::write(&s2_timer.path, &s2_timer.contents).expect("write s2 timer");

        let inspection = inspect_systemd_user(
            &paths,
            &schedules,
            Path::new("/usr/local/bin/kbolt"),
            &NoopRunner,
        )
        .expect("inspect systemd");

        assert_eq!(inspection.drifted_ids, HashSet::from(["s10".to_string()]));
    }

    #[test]
    fn reconcile_systemd_writes_units_and_enables_timers() {
        let tmp = tempdir().expect("create tempdir");
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
        let paths = SystemdUserPaths {
            unit_dir: tmp.path().join("systemd-user"),
        };
        let runner = RecordingRunner::new();

        reconcile_systemd_user(
            &paths,
            &[schedule],
            Path::new("/usr/local/bin/kbolt"),
            &runner,
        )
        .expect("reconcile systemd");

        assert!(paths.unit_dir.join("kbolt-schedule-s1.service").exists());
        assert!(paths.unit_dir.join("kbolt-schedule-s1.timer").exists());
        assert_eq!(
            runner.commands(),
            vec![
                vec![
                    "systemctl".to_string(),
                    "--user".to_string(),
                    "daemon-reload".to_string(),
                ],
                vec![
                    "systemctl".to_string(),
                    "--user".to_string(),
                    "enable".to_string(),
                    "--now".to_string(),
                    "kbolt-schedule-s1.timer".to_string(),
                ],
            ]
        );
    }

    struct NoopRunner;

    impl CommandRunner for NoopRunner {
        fn run(&self, _program: &str, _args: &[&str]) -> crate::Result<CommandOutput> {
            Ok(CommandOutput {
                success: true,
                stdout: String::new(),
                stderr: String::new(),
            })
        }
    }

    struct RecordingRunner {
        commands: Mutex<Vec<Vec<String>>>,
    }

    impl RecordingRunner {
        fn new() -> Self {
            Self {
                commands: Mutex::new(Vec::new()),
            }
        }

        fn commands(&self) -> Vec<Vec<String>> {
            self.commands.lock().expect("lock runner").clone()
        }
    }

    impl CommandRunner for RecordingRunner {
        fn run(&self, program: &str, args: &[&str]) -> crate::Result<CommandOutput> {
            self.commands.lock().expect("lock runner").push(
                std::iter::once(program.to_string())
                    .chain(args.iter().map(|arg| arg.to_string()))
                    .collect(),
            );

            Ok(CommandOutput {
                success: true,
                stdout: String::new(),
                stderr: String::new(),
            })
        }
    }
}
