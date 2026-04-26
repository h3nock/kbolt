use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use kbolt_types::{KboltError, WatchBackend, WatchServiceStatus, WatchStatusResponse};

use crate::config;
use crate::schedule_backend::{command_failure, current_executable_path, write_if_changed};
use crate::Result;

use super::log::{log_file_path, read_recent_lines};
use super::state::WatchStateStore;

const LAUNCHD_LABEL: &str = "com.kbolt.watch";
const SYSTEMD_SERVICE: &str = "kbolt-watch.service";
const WATCH_PID_FILE: &str = "watch.pid";
const SERVICE_STOP_TIMEOUT: Duration = Duration::from_secs(10);
const SERVICE_STOP_POLL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone)]
pub(crate) struct WatchPaths {
    pub cache_dir: PathBuf,
    pub pid_file: PathBuf,
    pub log_file: PathBuf,
    pub state_file: PathBuf,
    pub service_file: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommandOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

pub(crate) trait WatchCommandRunner {
    fn run(&self, program: &str, args: &[&str]) -> Result<CommandOutput>;
}

struct ProcessCommandRunner;

impl WatchCommandRunner for ProcessCommandRunner {
    fn run(&self, program: &str, args: &[&str]) -> Result<CommandOutput> {
        let output = Command::new(program).args(args).output()?;
        Ok(CommandOutput {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

pub fn enable(config_path: Option<&Path>) -> Result<WatchStatusResponse> {
    let config = config::load_existing(config_path)?;
    let executable = current_executable_path()?;
    let paths = watch_paths(&config.config_dir, &config.cache_dir)?;
    let runner = ProcessCommandRunner;
    enable_with_runner(&paths, &executable, &runner)?;
    status_from_paths(&paths, &runner)
}

pub fn disable(config_path: Option<&Path>) -> Result<WatchStatusResponse> {
    let config_dir = config::default_config_dir()?;
    let cache_dir = config::default_cache_dir()?;
    let paths = watch_paths_for_config_path(config_path, &config_dir, &cache_dir)?;
    let runner = ProcessCommandRunner;
    disable_with_runner(&paths, &runner)?;
    status_from_paths(&paths, &runner)
}

pub fn status(config_path: Option<&Path>) -> Result<WatchStatusResponse> {
    let config_dir = config::default_config_dir()?;
    let cache_dir = config::default_cache_dir()?;
    let paths = watch_paths_for_config_path(config_path, &config_dir, &cache_dir)?;
    let runner = ProcessCommandRunner;
    status_from_paths(&paths, &runner)
}

pub fn logs(config_path: Option<&Path>, max_lines: usize) -> Result<String> {
    let config_dir = config::default_config_dir()?;
    let cache_dir = config::default_cache_dir()?;
    let paths = watch_paths_for_config_path(config_path, &config_dir, &cache_dir)?;
    read_recent_lines(&paths.log_file, max_lines)
}

pub(crate) fn watch_paths(config_dir: &Path, cache_dir: &Path) -> Result<WatchPaths> {
    watch_paths_for_config_path(None, config_dir, cache_dir)
}

fn watch_paths_for_config_path(
    _config_path: Option<&Path>,
    config_dir: &Path,
    cache_dir: &Path,
) -> Result<WatchPaths> {
    let service_file = match current_backend() {
        WatchBackend::Launchd => Some(launchd_plist_path(config_dir)?),
        WatchBackend::SystemdUser => Some(systemd_service_path(config_dir)?),
        WatchBackend::Unsupported => None,
    };
    Ok(WatchPaths {
        cache_dir: cache_dir.to_path_buf(),
        pid_file: cache_dir.join("run").join(WATCH_PID_FILE),
        log_file: log_file_path(cache_dir),
        state_file: WatchStateStore::file_path(cache_dir),
        service_file,
    })
}

pub(crate) fn current_backend() -> WatchBackend {
    #[cfg(target_os = "macos")]
    {
        WatchBackend::Launchd
    }
    #[cfg(target_os = "linux")]
    {
        WatchBackend::SystemdUser
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        WatchBackend::Unsupported
    }
}

fn enable_with_runner(
    paths: &WatchPaths,
    executable: &Path,
    runner: &dyn WatchCommandRunner,
) -> Result<()> {
    fs::create_dir_all(paths.cache_dir.join("run"))?;
    fs::create_dir_all(paths.cache_dir.join("logs"))?;

    match current_backend() {
        WatchBackend::Launchd => enable_launchd(paths, executable, runner),
        WatchBackend::SystemdUser => enable_systemd(paths, executable, runner),
        WatchBackend::Unsupported => Err(KboltError::InvalidInput(
            "managed watch service is not supported on this platform; use `kbolt watch --foreground`"
                .to_string(),
        )
        .into()),
    }
}

fn disable_with_runner(paths: &WatchPaths, runner: &dyn WatchCommandRunner) -> Result<()> {
    let managed_was_active = manager_active(runner)?;
    let initial_pid = read_pid(&paths.pid_file)?;
    match current_backend() {
        WatchBackend::Launchd => disable_launchd(paths, runner)?,
        WatchBackend::SystemdUser => disable_systemd(paths, runner)?,
        WatchBackend::Unsupported => {}
    }

    if managed_was_active {
        wait_for_managed_stop(paths, runner, initial_pid, SERVICE_STOP_TIMEOUT)?;
    } else if initial_pid.is_some_and(pid_is_alive) {
        return Err(KboltError::InvalidInput(
            "kbolt watch is running outside the managed service; stop that process directly"
                .to_string(),
        )
        .into());
    }

    remove_pid_file(&paths.pid_file)?;
    WatchStateStore::remove(&paths.cache_dir)?;
    Ok(())
}

fn status_from_paths(
    paths: &WatchPaths,
    runner: &dyn WatchCommandRunner,
) -> Result<WatchStatusResponse> {
    let service = service_status(paths, runner)?;
    let runtime = if service.running {
        WatchStateStore::load(&paths.cache_dir)?
    } else {
        None
    };
    Ok(WatchStatusResponse {
        service,
        runtime,
        log_file: paths.log_file.clone(),
        state_file: paths.state_file.clone(),
    })
}

fn service_status(
    paths: &WatchPaths,
    runner: &dyn WatchCommandRunner,
) -> Result<WatchServiceStatus> {
    let backend = current_backend();
    let enabled = paths
        .service_file
        .as_ref()
        .is_some_and(|path| path.is_file());
    let pid = read_pid(&paths.pid_file)?;
    let pid_alive = pid.is_some_and(pid_is_alive);
    let manager_active = match backend {
        WatchBackend::Launchd => is_launchd_active(runner)?,
        WatchBackend::SystemdUser => is_systemd_active(runner)?,
        WatchBackend::Unsupported => false,
    };
    let running = pid_alive || manager_active;
    let issue = if backend == WatchBackend::Unsupported {
        Some("managed watch service is not supported on this platform".to_string())
    } else if pid.is_some() && !pid_alive && !manager_active {
        Some("stale watcher pid file found".to_string())
    } else {
        None
    };

    Ok(WatchServiceStatus {
        enabled,
        running,
        backend,
        pid: pid.filter(|value| pid_is_alive(*value)),
        issue,
    })
}

fn manager_active(runner: &dyn WatchCommandRunner) -> Result<bool> {
    match current_backend() {
        WatchBackend::Launchd => is_launchd_active(runner),
        WatchBackend::SystemdUser => is_systemd_active(runner),
        WatchBackend::Unsupported => Ok(false),
    }
}

fn wait_for_managed_stop(
    paths: &WatchPaths,
    runner: &dyn WatchCommandRunner,
    initial_pid: Option<u32>,
    timeout: Duration,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let pid_file_pid = read_pid(&paths.pid_file)?;
        let pid_alive =
            initial_pid.is_some_and(pid_is_alive) || pid_file_pid.is_some_and(pid_is_alive);
        if !manager_active(runner)? && !pid_alive {
            return Ok(());
        }

        if Instant::now() >= deadline {
            return Err(KboltError::Internal(format!(
                "watch service did not stop within {}s",
                timeout.as_secs()
            ))
            .into());
        }

        thread::sleep(SERVICE_STOP_POLL);
    }
}

fn enable_launchd(
    paths: &WatchPaths,
    executable: &Path,
    runner: &dyn WatchCommandRunner,
) -> Result<()> {
    let plist_path = paths.service_file.as_ref().ok_or_else(|| {
        KboltError::Internal("missing launchd plist path for watch service".to_string())
    })?;
    if let Some(parent) = plist_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let content = render_launchd_plist(executable, &paths.log_file)?;
    let changed = write_if_changed(plist_path, &content)?;
    if changed && is_launchd_active(runner)? {
        bootout_launchd(runner)?;
    }
    if changed || !is_launchd_active(runner)? {
        let domain = launchd_domain()?;
        let plist = path_str(plist_path)?;
        let output = runner.run("launchctl", &["bootstrap", domain.as_str(), plist])?;
        if !output.success {
            return Err(command_failure(
                "launchctl",
                &["bootstrap", domain.as_str(), plist],
                &to_schedule_output(output),
            )
            .into());
        }
    }
    Ok(())
}

fn disable_launchd(paths: &WatchPaths, runner: &dyn WatchCommandRunner) -> Result<()> {
    if is_launchd_active(runner)? {
        bootout_launchd(runner)?;
    }
    if let Some(path) = paths.service_file.as_ref() {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

fn enable_systemd(
    paths: &WatchPaths,
    executable: &Path,
    runner: &dyn WatchCommandRunner,
) -> Result<()> {
    let service_path = paths.service_file.as_ref().ok_or_else(|| {
        KboltError::Internal("missing systemd service path for watch service".to_string())
    })?;
    if let Some(parent) = service_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let changed = write_if_changed(service_path, &render_systemd_service(executable))?;
    if changed {
        let output = runner.run("systemctl", &["--user", "daemon-reload"])?;
        if !output.success {
            return Err(command_failure(
                "systemctl",
                &["--user", "daemon-reload"],
                &to_schedule_output(output),
            )
            .into());
        }
    }
    let output = runner.run("systemctl", &["--user", "enable", "--now", SYSTEMD_SERVICE])?;
    if !output.success {
        return Err(command_failure(
            "systemctl",
            &["--user", "enable", "--now", SYSTEMD_SERVICE],
            &to_schedule_output(output),
        )
        .into());
    }
    Ok(())
}

fn disable_systemd(paths: &WatchPaths, runner: &dyn WatchCommandRunner) -> Result<()> {
    let output = runner.run(
        "systemctl",
        &["--user", "disable", "--now", SYSTEMD_SERVICE],
    )?;
    if !output.success
        && !is_missing_systemd_unit(&output.stderr)
        && !is_missing_systemd_unit(&output.stdout)
    {
        return Err(command_failure(
            "systemctl",
            &["--user", "disable", "--now", SYSTEMD_SERVICE],
            &to_schedule_output(output),
        )
        .into());
    }
    if let Some(path) = paths.service_file.as_ref() {
        match fs::remove_file(path) {
            Ok(()) => {
                let output = runner.run("systemctl", &["--user", "daemon-reload"])?;
                if !output.success {
                    return Err(command_failure(
                        "systemctl",
                        &["--user", "daemon-reload"],
                        &to_schedule_output(output),
                    )
                    .into());
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

fn render_launchd_plist(executable: &Path, log_file: &Path) -> Result<String> {
    let stderr_path = log_file.with_file_name("watch.err.log");
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
      <string>__watch-run</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
      <key>SuccessfulExit</key>
      <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{stdout_path}</string>
    <key>StandardErrorPath</key>
    <string>{stderr_path}</string>
</dict>
</plist>
"#,
        label = LAUNCHD_LABEL,
        program = xml_escape(&executable.display().to_string()),
        stdout_path = xml_escape(&log_file.display().to_string()),
        stderr_path = xml_escape(&stderr_path.display().to_string()),
    ))
}

fn render_systemd_service(executable: &Path) -> String {
    format!(
        "[Unit]\nDescription=Keep kbolt collections fresh\n\n[Service]\nType=simple\nExecStart={} __watch-run\nRestart=on-failure\nRestartSec=5\nStartLimitIntervalSec=300\nStartLimitBurst=5\n\n[Install]\nWantedBy=default.target\n",
        shell_escape(executable),
    )
}

fn launchd_plist_path(_config_dir: &Path) -> Result<PathBuf> {
    #[cfg(test)]
    {
        return Ok(_config_dir
            .join("launchd")
            .join("LaunchAgents")
            .join(format!("{LAUNCHD_LABEL}.plist")));
    }
    #[cfg(not(test))]
    {
        let home_dir = dirs::home_dir()
            .ok_or_else(|| KboltError::Config("unable to determine home directory".to_string()))?;
        Ok(home_dir
            .join("Library")
            .join("LaunchAgents")
            .join(format!("{LAUNCHD_LABEL}.plist")))
    }
}

fn systemd_service_path(_config_dir: &Path) -> Result<PathBuf> {
    #[cfg(test)]
    {
        return Ok(_config_dir
            .join("systemd")
            .join("user")
            .join(SYSTEMD_SERVICE));
    }
    #[cfg(not(test))]
    {
        let base = dirs::config_dir().ok_or_else(|| {
            KboltError::Config("unable to determine user config directory".to_string())
        })?;
        Ok(base.join("systemd").join("user").join(SYSTEMD_SERVICE))
    }
}

fn is_launchd_active(runner: &dyn WatchCommandRunner) -> Result<bool> {
    let service_target = format!("{}/{}", launchd_domain()?, LAUNCHD_LABEL);
    let output = runner.run("launchctl", &["print", &service_target])?;
    Ok(output.success)
}

fn bootout_launchd(runner: &dyn WatchCommandRunner) -> Result<()> {
    let service_target = format!("{}/{}", launchd_domain()?, LAUNCHD_LABEL);
    let output = runner.run("launchctl", &["bootout", &service_target])?;
    if !output.success
        && !launchd_missing_service(&output.stdout)
        && !launchd_missing_service(&output.stderr)
    {
        return Err(command_failure(
            "launchctl",
            &["bootout", &service_target],
            &to_schedule_output(output),
        )
        .into());
    }
    Ok(())
}

fn is_systemd_active(runner: &dyn WatchCommandRunner) -> Result<bool> {
    let output = runner.run("systemctl", &["--user", "is-active", SYSTEMD_SERVICE])?;
    Ok(output.success && output.stdout.trim() == "active")
}

fn launchd_domain() -> Result<String> {
    #[cfg(test)]
    {
        Ok("gui/test".to_string())
    }
    #[cfg(not(test))]
    {
        let output = Command::new("id").arg("-u").output()?;
        if !output.status.success() {
            return Err(KboltError::Internal("failed to determine current uid".to_string()).into());
        }
        let uid = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if uid.is_empty() {
            return Err(KboltError::Internal("failed to determine current uid".to_string()).into());
        }
        Ok(format!("gui/{uid}"))
    }
}

fn launchd_missing_service(output: &str) -> bool {
    output.contains("Could not find service") || output.contains("No such process")
}

fn is_missing_systemd_unit(output: &str) -> bool {
    output.contains("not loaded") || output.contains("No such file") || output.contains("not found")
}

fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| KboltError::InvalidPath(path.to_path_buf()).into())
}

fn xml_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn shell_escape(path: &Path) -> String {
    let raw = path.display().to_string();
    if raw
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-'))
    {
        raw
    } else {
        format!("'{}'", raw.replace('\'', r#"'\''"#))
    }
}

fn to_schedule_output(output: CommandOutput) -> crate::schedule_backend::CommandOutput {
    crate::schedule_backend::CommandOutput {
        success: output.success,
        stdout: output.stdout,
        stderr: output.stderr,
    }
}

pub(crate) fn write_pid(path: &Path, pid: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{pid}\n"))?;
    Ok(())
}

pub(crate) fn read_pid(path: &Path) -> Result<Option<u32>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)?;
    let pid = raw.trim().parse::<u32>().map_err(|err| {
        KboltError::Internal(format!("invalid pid file {}: {err}", path.display()))
    })?;
    Ok(Some(pid))
}

pub(crate) fn remove_pid_file(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

#[cfg(unix)]
pub(crate) fn pid_is_alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(not(unix))]
pub(crate) fn pid_is_alive(_pid: u32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use kbolt_types::{WatchHealth, WatchMode, WatchRuntimeState, WatchRuntimeStatus};
    use tempfile::tempdir;

    use super::{
        render_launchd_plist, render_systemd_service, status_from_paths, CommandOutput,
        WatchCommandRunner, WatchPaths,
    };
    use crate::watch::state::WatchStateStore;

    struct InactiveRunner;

    impl WatchCommandRunner for InactiveRunner {
        fn run(&self, _program: &str, _args: &[&str]) -> crate::Result<CommandOutput> {
            Ok(CommandOutput {
                success: false,
                stdout: String::new(),
                stderr: String::new(),
            })
        }
    }

    fn sample_runtime_state() -> WatchRuntimeStatus {
        WatchRuntimeStatus {
            mode: WatchMode::Native,
            health: WatchHealth::Ok,
            state: WatchRuntimeState::Idle,
            pid: 42,
            started_at: "2026-04-25T00:00:00Z".to_string(),
            updated_at: "2026-04-25T00:00:01Z".to_string(),
            watched_collections: 1,
            dirty_collections: 0,
            semantic_pending_collections: 0,
            semantic_unavailable_collections: 0,
            semantic_blocked_spaces: Vec::new(),
            collections: Vec::new(),
            last_keyword_refresh: None,
            last_semantic_refresh: None,
            last_safety_scan: None,
            last_catalog_refresh: None,
            last_error: None,
        }
    }

    #[test]
    fn launchd_plist_runs_hidden_watch_runner() {
        let plist = render_launchd_plist(
            Path::new("/usr/local/bin/kbolt"),
            Path::new("/tmp/watch.log"),
        )
        .expect("render plist");

        assert!(plist.contains("__watch-run"));
        assert!(plist.contains("RunAtLoad"));
        assert!(plist.contains("SuccessfulExit"));
    }

    #[test]
    fn systemd_service_runs_hidden_watch_runner() {
        let unit = render_systemd_service(Path::new("/usr/local/bin/kbolt"));

        assert!(unit.contains("ExecStart=/usr/local/bin/kbolt __watch-run"));
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("WantedBy=default.target"));
    }

    #[test]
    fn status_suppresses_runtime_state_when_watcher_is_not_running() {
        let tmp = tempdir().expect("tempdir");
        let cache_dir = tmp.path().join("cache");
        let paths = WatchPaths {
            cache_dir: cache_dir.clone(),
            pid_file: cache_dir.join("run").join("watch.pid"),
            log_file: cache_dir.join("logs").join("watch.log"),
            state_file: WatchStateStore::file_path(&cache_dir),
            service_file: None,
        };
        WatchStateStore::save(&cache_dir, &sample_runtime_state()).expect("save stale state");

        let status = status_from_paths(&paths, &InactiveRunner).expect("load status");

        assert!(
            status.runtime.is_none(),
            "stale runtime state should not be reported when no watcher is running"
        );
        assert!(
            WatchStateStore::load(&cache_dir)
                .expect("load state")
                .is_some(),
            "status reads should not delete the stale state file"
        );
    }
}
