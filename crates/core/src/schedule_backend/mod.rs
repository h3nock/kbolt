use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(not(test))]
use std::process::Command;

use kbolt_types::{KboltError, ScheduleBackend, ScheduleDefinition};

use crate::Result;

#[cfg(any(target_os = "macos", test))]
pub(crate) mod launchd;
#[cfg(any(target_os = "linux", test))]
pub(crate) mod systemd;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct BackendInspection {
    pub drifted_ids: HashSet<String>,
    pub orphan_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommandOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

pub(crate) trait CommandRunner {
    fn run(&self, program: &str, args: &[&str]) -> Result<CommandOutput>;
}

#[cfg(not(test))]
struct ProcessCommandRunner;

#[cfg(not(test))]
impl CommandRunner for ProcessCommandRunner {
    fn run(&self, program: &str, args: &[&str]) -> Result<CommandOutput> {
        let output = Command::new(program).args(args).output()?;
        Ok(CommandOutput {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

#[cfg(test)]
struct NoopCommandRunner;

#[cfg(test)]
impl CommandRunner for NoopCommandRunner {
    fn run(&self, _program: &str, _args: &[&str]) -> Result<CommandOutput> {
        Ok(CommandOutput {
            success: true,
            stdout: String::new(),
            stderr: String::new(),
        })
    }
}

pub(crate) fn current_schedule_backend() -> Result<ScheduleBackend> {
    #[cfg(target_os = "macos")]
    {
        Ok(ScheduleBackend::Launchd)
    }

    #[cfg(target_os = "linux")]
    {
        Ok(ScheduleBackend::SystemdUser)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Err(
            KboltError::InvalidInput("schedule is not supported on this platform".to_string())
                .into(),
        )
    }
}

pub(crate) fn inspect_schedule_backend(
    config_dir: &Path,
    cache_dir: &Path,
    schedules: &[ScheduleDefinition],
) -> Result<BackendInspection> {
    let executable = current_executable_path()?;
    let runner = default_command_runner();
    #[cfg(target_os = "macos")]
    {
        let paths = launchd_paths(config_dir, cache_dir)?;
        return launchd::inspect_launchd(&paths, schedules, &executable, runner);
    }

    #[cfg(target_os = "linux")]
    {
        let paths = systemd_user_paths(config_dir)?;
        return systemd::inspect_systemd_user(&paths, schedules, &executable, runner);
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (config_dir, cache_dir, schedules, executable, runner);
        Err(
            KboltError::InvalidInput("schedule is not supported on this platform".to_string())
                .into(),
        )
    }
}

pub(crate) fn reconcile_schedule_backend(
    config_dir: &Path,
    cache_dir: &Path,
    schedules: &[ScheduleDefinition],
) -> Result<ScheduleBackend> {
    let executable = current_executable_path()?;
    let runner = default_command_runner();
    #[cfg(target_os = "macos")]
    {
        let paths = launchd_paths(config_dir, cache_dir)?;
        launchd::reconcile_launchd(&paths, schedules, &executable, runner)?;
        return Ok(ScheduleBackend::Launchd);
    }

    #[cfg(target_os = "linux")]
    {
        let paths = systemd_user_paths(config_dir)?;
        systemd::reconcile_systemd_user(&paths, schedules, &executable, runner)?;
        return Ok(ScheduleBackend::SystemdUser);
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (config_dir, cache_dir, schedules, executable, runner);
        Err(
            KboltError::InvalidInput("schedule is not supported on this platform".to_string())
                .into(),
        )
    }
}

pub(crate) fn write_if_changed(path: &Path, content: &str) -> Result<bool> {
    let needs_write = match fs::read_to_string(path) {
        Ok(existing) => existing != content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => true,
        Err(err) => return Err(err.into()),
    };

    if !needs_write {
        return Ok(false);
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, content)?;
    fs::rename(tmp_path, path)?;
    Ok(true)
}

pub(crate) fn current_executable_path() -> Result<PathBuf> {
    std::env::current_exe().map_err(Into::into)
}

pub(crate) fn command_failure(program: &str, args: &[&str], output: &CommandOutput) -> KboltError {
    let joined_args = args.join(" ");
    let stderr = output.stderr.trim();
    let stdout = output.stdout.trim();

    let detail = if !stderr.is_empty() {
        stderr.to_string()
    } else if !stdout.is_empty() {
        stdout.to_string()
    } else {
        "command returned a non-zero exit status".to_string()
    };

    KboltError::Internal(format!("`{program} {joined_args}` failed: {detail}"))
}

#[cfg(target_os = "macos")]
fn launchd_paths(_config_dir: &Path, cache_dir: &Path) -> Result<launchd::LaunchdPaths> {
    #[cfg(test)]
    {
        Ok(launchd::LaunchdPaths {
            agents_dir: _config_dir.join("launchd/LaunchAgents"),
            log_dir: cache_dir.join("schedules/logs"),
            domain: "gui/test".to_string(),
        })
    }

    #[cfg(not(test))]
    {
        let home_dir = dirs::home_dir()
            .ok_or_else(|| KboltError::Config("unable to determine home directory".to_string()))?;
        Ok(launchd::LaunchdPaths {
            agents_dir: home_dir.join("Library/LaunchAgents"),
            log_dir: cache_dir.join("schedules/logs"),
            domain: format!("gui/{}", current_uid()?),
        })
    }
}

#[cfg(target_os = "linux")]
fn systemd_user_paths(config_dir: &Path) -> Result<systemd::SystemdUserPaths> {
    #[cfg(test)]
    {
        Ok(systemd::SystemdUserPaths {
            unit_dir: config_dir.join("systemd/user"),
        })
    }

    #[cfg(not(test))]
    {
        let base = dirs::config_dir().ok_or_else(|| {
            KboltError::Config("unable to determine user config directory".to_string())
        })?;
        Ok(systemd::SystemdUserPaths {
            unit_dir: base.join("systemd/user"),
        })
    }
}

#[cfg(not(test))]
fn default_command_runner() -> &'static dyn CommandRunner {
    static RUNNER: ProcessCommandRunner = ProcessCommandRunner;
    &RUNNER
}

#[cfg(test)]
fn default_command_runner() -> &'static dyn CommandRunner {
    static RUNNER: NoopCommandRunner = NoopCommandRunner;
    &RUNNER
}

#[cfg(target_os = "macos")]
#[cfg(not(test))]
fn current_uid() -> Result<String> {
    let output = Command::new("id").arg("-u").output()?;
    if !output.status.success() {
        return Err(KboltError::Internal("failed to determine current uid".to_string()).into());
    }

    let uid = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if uid.is_empty() {
        return Err(KboltError::Internal("failed to determine current uid".to_string()).into());
    }

    Ok(uid)
}
