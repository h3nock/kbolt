use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use kbolt_types::{KboltError, ScheduleDefinition, ScheduleIntervalUnit, ScheduleTrigger};

use crate::Result;

const MANAGED_LABEL_PREFIX: &str = "com.kbolt.schedule.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LaunchdPaths {
    pub agents_dir: PathBuf,
    pub log_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LaunchdJob {
    pub label: String,
    pub plist_path: PathBuf,
    pub plist_contents: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LaunchdPlan {
    pub jobs: Vec<LaunchdJob>,
    pub stale_paths: Vec<PathBuf>,
}

pub(crate) fn plan_launchd(
    paths: &LaunchdPaths,
    schedules: &[ScheduleDefinition],
    executable: &Path,
) -> Result<LaunchdPlan> {
    fs::create_dir_all(&paths.agents_dir)?;
    fs::create_dir_all(&paths.log_dir)?;

    let mut jobs = schedules
        .iter()
        .map(|schedule| {
            let label = launchd_label(&schedule.id);
            let plist_path = paths.agents_dir.join(format!("{label}.plist"));
            let plist_contents = render_launchd_plist(schedule, executable, &paths.log_dir)?;
            Ok(LaunchdJob {
                label,
                plist_path,
                plist_contents,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    jobs.sort_by(|left, right| left.label.cmp(&right.label));

    let desired_paths = jobs
        .iter()
        .map(|job| job.plist_path.clone())
        .collect::<HashSet<_>>();

    let mut stale_paths = Vec::new();
    for entry in fs::read_dir(&paths.agents_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !is_managed_launchd_plist(&path) {
            continue;
        }
        if !desired_paths.contains(&path) {
            stale_paths.push(path);
        }
    }
    stale_paths.sort();

    Ok(LaunchdPlan { jobs, stale_paths })
}

pub(crate) fn render_launchd_plist(
    schedule: &ScheduleDefinition,
    executable: &Path,
    log_dir: &Path,
) -> Result<String> {
    let label = launchd_label(&schedule.id);
    let stdout_path = log_dir.join(format!("{label}.out.log"));
    let stderr_path = log_dir.join(format!("{label}.err.log"));

    let trigger_xml = match &schedule.trigger {
        ScheduleTrigger::Every { interval } => {
            let seconds = match interval.unit {
                ScheduleIntervalUnit::Minutes => interval.value.checked_mul(60),
                ScheduleIntervalUnit::Hours => interval.value.checked_mul(60 * 60),
            }
            .ok_or_else(|| {
                KboltError::InvalidInput(format!(
                    "schedule interval is too large for launchd: {}",
                    schedule.id
                ))
            })?;

            format!("    <key>StartInterval</key>\n    <integer>{seconds}</integer>\n")
        }
        ScheduleTrigger::Daily { time } => {
            let (hour, minute) = parse_canonical_time(time)?;
            format!(
                "    <key>StartCalendarInterval</key>\n    <dict>\n      <key>Hour</key>\n      <integer>{hour}</integer>\n      <key>Minute</key>\n      <integer>{minute}</integer>\n    </dict>\n"
            )
        }
        ScheduleTrigger::Weekly { weekdays, time } => {
            let (hour, minute) = parse_canonical_time(time)?;
            let entries = weekdays
                .iter()
                .map(|weekday| {
                    format!(
                        "      <dict>\n        <key>Weekday</key>\n        <integer>{}</integer>\n        <key>Hour</key>\n        <integer>{hour}</integer>\n        <key>Minute</key>\n        <integer>{minute}</integer>\n      </dict>",
                        launchd_weekday(*weekday)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            format!("    <key>StartCalendarInterval</key>\n    <array>\n{entries}\n    </array>\n")
        }
    };

    Ok(format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
      <string>{program}</string>
      <string>__schedule-run</string>
      <string>{schedule_id}</string>
    </array>
{trigger_xml}    <key>StandardOutPath</key>
    <string>{stdout_path}</string>
    <key>StandardErrorPath</key>
    <string>{stderr_path}</string>
</dict>
</plist>
"#,
        program = xml_escape(&executable.display().to_string()),
        schedule_id = xml_escape(&schedule.id),
        stdout_path = xml_escape(&stdout_path.display().to_string()),
        stderr_path = xml_escape(&stderr_path.display().to_string()),
    ))
}

fn launchd_label(schedule_id: &str) -> String {
    format!("{MANAGED_LABEL_PREFIX}{schedule_id}")
}

fn is_managed_launchd_plist(path: &Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()) == Some("plist")
        && path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .is_some_and(|stem| stem.starts_with(MANAGED_LABEL_PREFIX))
}

fn parse_canonical_time(time: &str) -> Result<(u32, u32)> {
    let (hour, minute) = time
        .split_once(':')
        .ok_or_else(|| KboltError::InvalidInput(format!("invalid schedule time: {time}")))?;
    let hour = hour
        .parse::<u32>()
        .map_err(|_| KboltError::InvalidInput(format!("invalid schedule time: {time}")))?;
    let minute = minute
        .parse::<u32>()
        .map_err(|_| KboltError::InvalidInput(format!("invalid schedule time: {time}")))?;
    Ok((hour, minute))
}

fn launchd_weekday(weekday: kbolt_types::ScheduleWeekday) -> u32 {
    match weekday {
        kbolt_types::ScheduleWeekday::Sun => 0,
        kbolt_types::ScheduleWeekday::Mon => 1,
        kbolt_types::ScheduleWeekday::Tue => 2,
        kbolt_types::ScheduleWeekday::Wed => 3,
        kbolt_types::ScheduleWeekday::Thu => 4,
        kbolt_types::ScheduleWeekday::Fri => 5,
        kbolt_types::ScheduleWeekday::Sat => 6,
    }
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::tempdir;

    use super::{plan_launchd, render_launchd_plist, LaunchdPaths};
    use kbolt_types::{
        ScheduleDefinition, ScheduleInterval, ScheduleIntervalUnit, ScheduleScope, ScheduleTrigger,
        ScheduleWeekday,
    };

    #[test]
    fn render_launchd_plist_for_interval_schedule_uses_start_interval() {
        let plist = render_launchd_plist(
            &ScheduleDefinition {
                id: "s1".to_string(),
                trigger: ScheduleTrigger::Every {
                    interval: ScheduleInterval {
                        value: 30,
                        unit: ScheduleIntervalUnit::Minutes,
                    },
                },
                scope: ScheduleScope::All,
            },
            Path::new("/usr/local/bin/kbolt"),
            Path::new("/tmp/kbolt/logs"),
        )
        .expect("render plist");

        assert!(plist.contains("<string>com.kbolt.schedule.s1</string>"));
        assert!(plist.contains("<string>/usr/local/bin/kbolt</string>"));
        assert!(plist.contains("<string>__schedule-run</string>"));
        assert!(plist.contains("<string>s1</string>"));
        assert!(plist.contains("<key>StartInterval</key>"));
        assert!(plist.contains("<integer>1800</integer>"));
        assert!(plist.contains("/tmp/kbolt/logs/com.kbolt.schedule.s1.out.log"));
    }

    #[test]
    fn render_launchd_plist_for_weekly_schedule_uses_start_calendar_array() {
        let plist = render_launchd_plist(
            &ScheduleDefinition {
                id: "s2".to_string(),
                trigger: ScheduleTrigger::Weekly {
                    weekdays: vec![ScheduleWeekday::Mon, ScheduleWeekday::Fri],
                    time: "15:00".to_string(),
                },
                scope: ScheduleScope::Space {
                    space: "work".to_string(),
                },
            },
            Path::new("/usr/local/bin/kbolt"),
            Path::new("/tmp/kbolt/logs"),
        )
        .expect("render plist");

        assert!(plist.contains("<key>StartCalendarInterval</key>"));
        assert!(plist.contains("<array>"));
        assert!(plist.contains("<integer>1</integer>"));
        assert!(plist.contains("<integer>5</integer>"));
        assert!(plist.contains("<integer>15</integer>"));
        assert!(plist.contains("<integer>0</integer>"));
    }

    #[test]
    fn plan_launchd_detects_stale_managed_plists() {
        let tmp = tempdir().expect("create tempdir");
        let paths = LaunchdPaths {
            agents_dir: tmp.path().join("LaunchAgents"),
            log_dir: tmp.path().join("logs"),
        };
        std::fs::create_dir_all(&paths.agents_dir).expect("create agents dir");
        std::fs::write(
            paths.agents_dir.join("com.kbolt.schedule.s9.plist"),
            "stale job",
        )
        .expect("write stale plist");
        std::fs::write(
            paths.agents_dir.join("com.example.other.plist"),
            "foreign job",
        )
        .expect("write foreign plist");

        let plan = plan_launchd(
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
        .expect("plan launchd");

        assert_eq!(plan.jobs.len(), 1);
        assert_eq!(plan.jobs[0].label, "com.kbolt.schedule.s1");
        assert_eq!(
            plan.stale_paths,
            vec![paths.agents_dir.join("com.kbolt.schedule.s9.plist")]
        );
    }
}
