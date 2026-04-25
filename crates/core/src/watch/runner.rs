use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use kbolt_types::{
    CollectionInfo, KboltError, UpdateOptions, UpdateReport, WatchCollectionStatus, WatchHealth,
    WatchMode, WatchRefreshSummary, WatchRuntimeState, WatchRuntimeStatus, WatchSemanticState,
    WatchSpaceBlock,
};

use crate::engine::ignore_helpers::{is_hard_ignored_dir_name, is_hard_ignored_file};
use crate::engine::Engine;
use crate::error::CoreError;
use crate::Result;

use super::log::WatchLogger;
use super::scheduler::{ScheduledAction, SemanticScheduleState, Tick, WatchScheduler};
use super::service::{
    self, pid_is_alive, read_pid, remove_pid_file, watch_paths, write_pid, WatchPaths,
};
use super::source::{path_has_component, NativeWatchSource, WatchSourceEvent};
use super::state::{system_time_string, utc_now_string, WatchStateStore};
use super::{
    duration_millis, CollectionKey, WatchCollection, CATALOG_REFRESH_INTERVAL, EVENT_DEBOUNCE,
    POLL_INTERVAL, SAFETY_RESCAN_BASE,
};

const WATCH_LOOP_SLEEP: Duration = Duration::from_millis(500);
const STATE_WRITE_INTERVAL: Duration = Duration::from_secs(5);

pub fn run_service() -> Result<()> {
    run(false)
}

pub fn run_foreground() -> Result<()> {
    let status = service::status(None)?;
    if status.service.running {
        return Err(KboltError::InvalidInput(
            "kbolt watch service is already running; use `kbolt watch status`, `kbolt watch logs`, or `kbolt watch disable` first"
                .to_string(),
        )
        .into());
    }

    run(true)
}

fn run(foreground: bool) -> Result<()> {
    let engine = Engine::new(None)?;
    let paths = watch_paths(&engine.config().config_dir, &engine.config().cache_dir)?;
    let logger = WatchLogger::new(&engine.config().cache_dir);
    let pid = std::process::id();

    if let Some(existing_pid) = read_pid(&paths.pid_file)? {
        if existing_pid != pid && pid_is_alive(existing_pid) {
            return Err(KboltError::InvalidInput(format!(
                "kbolt watch is already running with pid {existing_pid}"
            ))
            .into());
        }
    }

    write_pid(&paths.pid_file, pid)?;
    let shutdown = install_signal_handler()?;
    let mut runner = WatchRunner::new(engine, paths, logger, foreground, shutdown)?;
    let result = runner.run_loop();
    let cleanup_result = remove_pid_file(&runner.paths.pid_file);

    match (result, cleanup_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(err), Ok(())) => Err(err),
        (Ok(()), Err(err)) => Err(err),
        (Err(err), Err(cleanup_err)) => Err(KboltError::Internal(format!(
            "watcher failed: {err}; pid cleanup also failed: {cleanup_err}"
        ))
        .into()),
    }
}

struct WatchRunner {
    engine: Engine,
    paths: WatchPaths,
    logger: WatchLogger,
    scheduler: WatchScheduler,
    source: Option<NativeWatchSource>,
    foreground: bool,
    shutdown: Arc<AtomicBool>,
    started_instant: Instant,
    started_system: SystemTime,
    started_at: String,
    next_catalog_refresh: Instant,
    next_poll: Instant,
    next_safety_scan: Instant,
    next_state_write: Instant,
    collections: BTreeMap<CollectionKey, CollectionInfo>,
    current_state: WatchRuntimeState,
    last_keyword_refresh: Option<WatchRefreshSummary>,
    last_semantic_refresh: Option<WatchRefreshSummary>,
    last_safety_scan: Option<String>,
    last_catalog_refresh: Option<String>,
    last_error: Option<String>,
}

impl WatchRunner {
    fn new(
        engine: Engine,
        paths: WatchPaths,
        logger: WatchLogger,
        foreground: bool,
        shutdown: Arc<AtomicBool>,
    ) -> Result<Self> {
        let now = Instant::now();
        let started_system = SystemTime::now();
        let mut runner = Self {
            engine,
            paths,
            logger,
            scheduler: WatchScheduler::default(),
            source: None,
            foreground,
            shutdown,
            started_instant: now,
            started_system,
            started_at: system_time_string(started_system)?,
            next_catalog_refresh: now,
            next_poll: now + POLL_INTERVAL,
            next_safety_scan: now + jittered_safety_interval(std::process::id()),
            next_state_write: now,
            collections: BTreeMap::new(),
            current_state: WatchRuntimeState::Starting,
            last_keyword_refresh: None,
            last_semantic_refresh: None,
            last_safety_scan: None,
            last_catalog_refresh: None,
            last_error: None,
        };

        runner.log("watcher starting")?;
        runner.refresh_catalog(runner.now_tick(), "startup")?;
        runner.configure_native_source()?;
        runner.write_state(WatchRuntimeState::Starting)?;
        runner.run_detection_scan("startup reconciliation")?;
        runner.write_state(WatchRuntimeState::Idle)?;
        Ok(runner)
    }

    fn run_loop(&mut self) -> Result<()> {
        while !self.shutdown.load(Ordering::Acquire) {
            let now = Instant::now();
            self.drain_native_events()?;

            if now >= self.next_catalog_refresh {
                self.current_state = WatchRuntimeState::Checking;
                self.refresh_catalog(self.now_tick(), "catalog refresh")?;
                self.configure_native_source()?;
                self.next_catalog_refresh = now + CATALOG_REFRESH_INTERVAL;
            }

            if now >= self.next_safety_scan {
                self.run_detection_scan("safety rescan")?;
                self.last_safety_scan = Some(utc_now_string()?);
                self.next_safety_scan = now + jittered_safety_interval(std::process::id());
            }

            if self.source.is_none() && now >= self.next_poll {
                self.run_detection_scan("poll fallback")?;
                self.next_poll = now + POLL_INTERVAL;
            }

            self.run_due_actions()?;

            if now >= self.next_state_write {
                self.write_state(WatchRuntimeState::Idle)?;
                self.next_state_write = now + STATE_WRITE_INTERVAL;
            }

            thread::sleep(WATCH_LOOP_SLEEP);
        }

        self.log("watcher stopping")?;
        self.write_state(WatchRuntimeState::Stopping)?;
        Ok(())
    }

    fn drain_native_events(&mut self) -> Result<()> {
        let Some(source) = self.source.as_ref() else {
            return Ok(());
        };

        let events = source.drain_events();
        if events.is_empty() {
            return Ok(());
        }

        let now = self.now_tick();
        for event in events {
            match event {
                WatchSourceEvent::Paths(paths) => self.handle_changed_paths(paths, now)?,
                WatchSourceEvent::OverflowOrError(message) => {
                    self.last_error = Some(format!(
                        "native watcher overflow or error; switching to polling: {message}"
                    ));
                    self.log(&format!(
                        "native watcher overflow or error; switching to polling: {message}"
                    ))?;
                    self.source = None;
                    self.scheduler.mark_all_dirty(now);
                    break;
                }
            }
        }

        Ok(())
    }

    fn handle_changed_paths(&mut self, paths: Vec<PathBuf>, now: Tick) -> Result<()> {
        let mut dirty = BTreeSet::new();
        let config_dir = self.engine.config().config_dir.clone();
        let mut config_changed = false;

        for path in paths {
            if path_matches_root(&path, &config_dir) {
                config_changed = true;
                continue;
            }

            if should_ignore_event_path(&path) {
                continue;
            }

            for (key, info) in &self.collections {
                if path_matches_root(&path, &info.path) {
                    dirty.insert(key.clone());
                }
            }
        }

        if config_changed {
            self.reload_engine()?;
            self.refresh_catalog(now, "config change")?;
            for key in self.collections.keys() {
                dirty.insert(key.clone());
            }
        }

        if dirty.is_empty() {
            return Ok(());
        }

        for key in &dirty {
            self.scheduler.mark_dirty(key, now);
        }
        self.log(&format!(
            "filesystem activity marked {} collection(s) dirty",
            dirty.len()
        ))?;
        Ok(())
    }

    fn reload_engine(&mut self) -> Result<()> {
        match Engine::new(None) {
            Ok(engine) => {
                self.engine = engine;
                self.scheduler.clear_semantic_unavailable();
                self.last_error = None;
                self.log("configuration reloaded")
            }
            Err(err) => {
                let message =
                    format!("configuration reload failed; keeping previous config: {err}");
                self.last_error = Some(message.clone());
                self.log(&message)
            }
        }
    }

    fn refresh_catalog(&mut self, now: Tick, label: &str) -> Result<()> {
        let infos = self.engine.list_collections(None)?;
        let watch_collections = infos.iter().map(collection_to_watch).collect::<Vec<_>>();
        let diff = self.scheduler.sync_catalog(watch_collections, now);
        self.collections = infos
            .into_iter()
            .map(|info| (CollectionKey::new(&info.space, &info.name), info))
            .collect();

        let keys = self.collections.keys().cloned().collect::<Vec<_>>();
        for key in keys {
            let pending = self.pending_chunks(&key)?;
            self.scheduler.set_pending_chunks(&key, pending);
        }
        self.scheduler.clear_resolved_space_blocks();

        self.last_catalog_refresh = Some(utc_now_string()?);
        if !diff.added.is_empty() || !diff.removed.is_empty() || !diff.path_changed.is_empty() {
            self.log(&format!(
                "{label}: catalog changed (added {}, removed {}, path changed {})",
                diff.added.len(),
                diff.removed.len(),
                diff.path_changed.len()
            ))?;
        }
        Ok(())
    }

    fn configure_native_source(&mut self) -> Result<()> {
        if self.source.is_none() && self.native_roots().is_empty() {
            return Ok(());
        }

        let roots = self.native_roots();
        if let Some(source) = self.source.as_mut() {
            if let Err(err) = source.sync_roots(roots) {
                self.last_error = Some(format!(
                    "native watcher failed; switching to polling: {err}"
                ));
                self.log(&format!(
                    "native watcher failed; switching to polling: {err}"
                ))?;
                self.source = None;
            }
            return Ok(());
        }

        match NativeWatchSource::new(EVENT_DEBOUNCE, roots) {
            Ok(source) => {
                self.source = Some(source);
                self.log("native watcher active")
            }
            Err(err) => {
                self.last_error = Some(format!("native watcher unavailable; using polling: {err}"));
                self.log(&format!("native watcher unavailable; using polling: {err}"))
            }
        }
    }

    fn native_roots(&mut self) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        let config_dir = self.engine.config().config_dir.clone();
        if config_dir.is_dir() {
            roots.push(config_dir);
        }

        let now = self.now_tick();
        let keys = self.collections.keys().cloned().collect::<Vec<_>>();
        for key in keys {
            let Some(info) = self.collections.get(&key) else {
                continue;
            };
            if info.path.is_dir() {
                roots.push(info.path.clone());
            } else {
                self.scheduler.mark_update_error(
                    &key,
                    now,
                    format!("collection root does not exist: {}", info.path.display()),
                );
            }
        }
        roots
    }

    fn run_due_actions(&mut self) -> Result<()> {
        let actions = self.scheduler.due_actions(self.now_tick());
        for action in actions {
            match action {
                ScheduledAction::RunKeywordUpdate { key } => {
                    self.run_keyword_update(&key, KeywordTrigger::NativeEvent)?
                }
                ScheduledAction::RunSemanticUpdate { key } => self.run_semantic_update(&key)?,
            }
        }
        Ok(())
    }

    fn run_detection_scan(&mut self, label: &str) -> Result<()> {
        self.current_state = WatchRuntimeState::Checking;
        self.log(&format!("{label}: starting keyword-only detection"))?;
        let keys = self.collections.keys().cloned().collect::<Vec<_>>();
        for key in keys {
            if self.shutdown.load(Ordering::Acquire) {
                break;
            }
            self.run_keyword_update(&key, KeywordTrigger::Detection)?;
        }
        self.log(&format!("{label}: finished"))?;
        Ok(())
    }

    fn run_keyword_update(&mut self, key: &CollectionKey, trigger: KeywordTrigger) -> Result<()> {
        self.current_state = WatchRuntimeState::RefreshingKeyword;
        let started_instant = Instant::now();
        let started_tick = self.now_tick();
        let started_at = self.tick_to_time(started_tick)?;
        let options = UpdateOptions {
            space: Some(key.space.clone()),
            collections: vec![key.collection.clone()],
            no_embed: true,
            dry_run: false,
            verbose: false,
        };

        let report = match self.engine.update(options) {
            Ok(report) => report,
            Err(err) => {
                let message = format!("keyword refresh failed for {key}: {err}");
                self.scheduler
                    .mark_update_error(key, self.now_tick(), message.clone());
                self.last_error = Some(message.clone());
                self.log(&message)?;
                return Ok(());
            }
        };

        let pending = self.pending_chunks(key).unwrap_or(0);
        let changed = report_changed_docs(&report) > 0 || report.failed_docs > 0;
        match trigger {
            KeywordTrigger::NativeEvent => {
                self.scheduler
                    .mark_keyword_success(key, self.now_tick(), pending);
            }
            KeywordTrigger::Detection => {
                self.scheduler.mark_detection_keyword_success(
                    key,
                    self.now_tick(),
                    pending,
                    changed,
                );
            }
        }

        if !report.errors.is_empty() {
            let message = format!(
                "keyword refresh for {key} completed with {} file error(s)",
                report.errors.len()
            );
            self.scheduler
                .mark_update_error(key, self.now_tick(), message.clone());
            self.last_error = Some(message.clone());
            self.log(&message)?;
        }

        let summary = refresh_summary(
            key,
            &report,
            started_at,
            utc_now_string()?,
            started_instant.elapsed(),
        );
        self.last_keyword_refresh = Some(summary);
        self.log(&format!(
            "keyword refresh finished for {key}: scanned {}, changed {}, pending semantic chunks {}",
            report.scanned_docs,
            report_changed_docs(&report),
            pending
        ))?;
        self.write_state(WatchRuntimeState::Idle)?;
        Ok(())
    }

    fn run_semantic_update(&mut self, key: &CollectionKey) -> Result<()> {
        self.current_state = WatchRuntimeState::RefreshingSemantic;
        let pending_before = self.pending_chunks(key).unwrap_or(0);
        if pending_before == 0 {
            self.scheduler
                .mark_semantic_success(key, self.now_tick(), pending_before);
            return Ok(());
        }

        if self.engine.config().roles.embedder.is_none() {
            self.scheduler.mark_semantic_unavailable(
                key,
                self.now_tick(),
                pending_before,
                "embedder is not configured".to_string(),
            );
            self.log(&format!(
                "semantic refresh skipped for {key}: embedder is not configured"
            ))?;
            return Ok(());
        }

        let started_instant = Instant::now();
        let started_tick = self.now_tick();
        let started_at = self.tick_to_time(started_tick)?;
        let options = UpdateOptions {
            space: Some(key.space.clone()),
            collections: vec![key.collection.clone()],
            no_embed: false,
            dry_run: false,
            verbose: false,
        };

        match self.engine.update(options) {
            Ok(report) => {
                let pending = self.pending_chunks(key).unwrap_or(0);
                if pending > 0 && (report.failed_docs > 0 || !report.errors.is_empty()) {
                    self.scheduler.mark_semantic_unavailable(
                        key,
                        self.now_tick(),
                        pending,
                        format!(
                            "semantic refresh left {pending} chunk(s) unembedded after partial failures"
                        ),
                    );
                } else {
                    self.scheduler
                        .mark_semantic_success(key, self.now_tick(), pending);
                }

                self.last_semantic_refresh = Some(refresh_summary(
                    key,
                    &report,
                    started_at,
                    utc_now_string()?,
                    started_instant.elapsed(),
                ));
                self.log(&format!(
                    "semantic refresh finished for {key}: embedded {}, pending {}",
                    report.embedded_chunks, pending
                ))?;
            }
            Err(err) => {
                self.handle_semantic_error(key, err, pending_before)?;
            }
        }

        self.write_state(WatchRuntimeState::Idle)?;
        Ok(())
    }

    fn handle_semantic_error(
        &mut self,
        key: &CollectionKey,
        err: CoreError,
        pending_chunks: usize,
    ) -> Result<()> {
        match &err {
            CoreError::Domain(KboltError::SpaceDenseRepairRequired { space, reason }) => {
                self.scheduler
                    .mark_space_blocked(space, self.now_tick(), reason.clone());
                let message = format!(
                    "semantic indexing blocked for space {space}: {reason}; run `kbolt --space {space} update`"
                );
                self.last_error = Some(message.clone());
                self.log(&message)
            }
            CoreError::Domain(KboltError::ModelNotAvailable { name }) => {
                let reason = format!("model not available: {name}");
                self.scheduler.mark_semantic_unavailable(
                    key,
                    self.now_tick(),
                    pending_chunks,
                    reason.clone(),
                );
                self.last_error = Some(reason.clone());
                self.log(&format!("semantic refresh unavailable for {key}: {reason}"))
            }
            CoreError::Domain(KboltError::Inference(message))
            | CoreError::Domain(KboltError::ModelDownload(message)) => {
                self.scheduler.mark_semantic_unavailable(
                    key,
                    self.now_tick(),
                    pending_chunks,
                    message.clone(),
                );
                self.last_error = Some(message.clone());
                self.log(&format!(
                    "semantic refresh unavailable for {key}: {message}"
                ))
            }
            _ => {
                let message = format!("semantic refresh failed for {key}: {err}");
                self.scheduler
                    .mark_update_error(key, self.now_tick(), message.clone());
                self.last_error = Some(message.clone());
                self.log(&message)
            }
        }
    }

    fn pending_chunks(&self, key: &CollectionKey) -> Result<usize> {
        let collections = self.engine.list_collections(Some(&key.space))?;
        let Some(info) = collections
            .into_iter()
            .find(|item| item.name == key.collection)
        else {
            return Ok(0);
        };

        Ok(info.chunk_count.saturating_sub(info.embedded_chunk_count))
    }

    fn write_state(&mut self, state: WatchRuntimeState) -> Result<()> {
        self.current_state = state;
        let status = self.runtime_status(state)?;
        WatchStateStore::save(&self.paths.cache_dir, &status)?;
        Ok(())
    }

    fn runtime_status(&self, state: WatchRuntimeState) -> Result<WatchRuntimeStatus> {
        let collection_statuses = self
            .scheduler
            .collections()
            .into_iter()
            .map(|status| {
                let semantic = match status.semantic {
                    SemanticScheduleState::None => WatchSemanticState::None,
                    SemanticScheduleState::Pending { pending_chunks } => {
                        WatchSemanticState::Pending { pending_chunks }
                    }
                    SemanticScheduleState::Unavailable {
                        pending_chunks,
                        reason,
                        ..
                    } => WatchSemanticState::Unavailable {
                        pending_chunks,
                        reason,
                    },
                    SemanticScheduleState::Blocked { space, reason, .. } => {
                        WatchSemanticState::Blocked {
                            fix: format!("kbolt --space {space} update"),
                            space,
                            reason,
                        }
                    }
                };
                Ok(WatchCollectionStatus {
                    space: status.key.space,
                    collection: status.key.collection,
                    path: status.path,
                    dirty: status.dirty,
                    semantic,
                    last_event_at: tick_to_optional_time(
                        self.started_system,
                        status.last_event_at,
                    )?,
                    last_keyword_refresh: tick_to_optional_time(
                        self.started_system,
                        status.last_keyword_refresh_at,
                    )?,
                    last_semantic_refresh: tick_to_optional_time(
                        self.started_system,
                        status.last_semantic_refresh_at,
                    )?,
                    last_error: status.last_error,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let semantic_pending_collections = collection_statuses
            .iter()
            .filter(|item| matches!(item.semantic, WatchSemanticState::Pending { .. }))
            .count();
        let semantic_unavailable_collections = collection_statuses
            .iter()
            .filter(|item| matches!(item.semantic, WatchSemanticState::Unavailable { .. }))
            .count();
        let dirty_collections = collection_statuses.iter().filter(|item| item.dirty).count();
        let semantic_blocked_spaces = self
            .scheduler
            .space_blocks()
            .into_iter()
            .map(|(space, block)| {
                Ok(WatchSpaceBlock {
                    fix: format!("kbolt --space {space} update"),
                    space,
                    reason: block.reason,
                    set_at: self.tick_to_time(block.set_at)?,
                    backoff_until: self.tick_to_time(block.backoff_until)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mode = if self.source.is_some() {
            WatchMode::Native
        } else {
            WatchMode::Polling
        };
        let health = if self.last_error.is_some()
            || !semantic_blocked_spaces.is_empty()
            || semantic_unavailable_collections > 0
            || mode == WatchMode::Polling
        {
            WatchHealth::Warning
        } else {
            WatchHealth::Ok
        };

        Ok(WatchRuntimeStatus {
            mode,
            health,
            state,
            pid: std::process::id(),
            started_at: self.started_at.clone(),
            updated_at: utc_now_string()?,
            watched_collections: collection_statuses.len(),
            dirty_collections,
            semantic_pending_collections,
            semantic_unavailable_collections,
            semantic_blocked_spaces,
            collections: collection_statuses,
            last_keyword_refresh: self.last_keyword_refresh.clone(),
            last_semantic_refresh: self.last_semantic_refresh.clone(),
            last_safety_scan: self.last_safety_scan.clone(),
            last_catalog_refresh: self.last_catalog_refresh.clone(),
            last_error: self.last_error.clone(),
        })
    }

    fn tick_to_time(&self, tick: Tick) -> Result<String> {
        system_time_string(self.started_system + Duration::from_millis(tick))
    }

    fn now_tick(&self) -> Tick {
        self.started_instant.elapsed().as_millis() as Tick
    }

    fn log(&self, message: &str) -> Result<()> {
        let line = format!("{} {message}", utc_now_string()?);
        self.logger.write_line(&line)?;
        if self.foreground {
            println!("{line}");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeywordTrigger {
    NativeEvent,
    Detection,
}

fn collection_to_watch(info: &CollectionInfo) -> WatchCollection {
    WatchCollection {
        key: CollectionKey::new(&info.space, &info.name),
        path: info.path.clone(),
    }
}

fn should_ignore_event_path(path: &Path) -> bool {
    path_has_component(path, is_hard_ignored_dir_name) || is_hard_ignored_file(path)
}

fn path_matches_root(path: &Path, root: &Path) -> bool {
    path.starts_with(root) || canonicalish_path(path).starts_with(canonicalish_path(root))
}

fn canonicalish_path(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }

    let Some(parent) = path.parent() else {
        return path.to_path_buf();
    };
    let Ok(canonical_parent) = parent.canonicalize() else {
        return path.to_path_buf();
    };
    match path.file_name() {
        Some(name) => canonical_parent.join(name),
        None => canonical_parent,
    }
}

fn report_changed_docs(report: &UpdateReport) -> usize {
    report.added_docs
        + report.updated_docs
        + report.deactivated_docs
        + report.reactivated_docs
        + report.reaped_docs
}

fn refresh_summary(
    key: &CollectionKey,
    report: &UpdateReport,
    started_at: String,
    finished_at: String,
    elapsed: Duration,
) -> WatchRefreshSummary {
    WatchRefreshSummary {
        space: key.space.clone(),
        collection: key.collection.clone(),
        started_at,
        finished_at,
        elapsed_ms: elapsed.as_millis() as u64,
        scanned_docs: report.scanned_docs,
        changed_docs: report_changed_docs(report),
        embedded_chunks: report.embedded_chunks,
    }
}

fn tick_to_optional_time(started_system: SystemTime, tick: Option<Tick>) -> Result<Option<String>> {
    tick.map(|value| system_time_string(started_system + Duration::from_millis(value)))
        .transpose()
}

fn jittered_safety_interval(seed: u32) -> Duration {
    let base_ms = duration_millis(SAFETY_RESCAN_BASE);
    let jitter_window = base_ms / 4;
    let offset = (u64::from(seed) % (jitter_window.saturating_mul(2).saturating_add(1))) as i64
        - jitter_window as i64;
    Duration::from_millis(base_ms.saturating_add_signed(offset))
}

fn install_signal_handler() -> Result<Arc<AtomicBool>> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let signals_seen = Arc::new(AtomicUsize::new(0));
    let shutdown_handler = Arc::clone(&shutdown);
    let signals_handler = Arc::clone(&signals_seen);
    ctrlc::set_handler(move || {
        let previous = signals_handler.fetch_add(1, Ordering::AcqRel);
        if previous == 0 {
            shutdown_handler.store(true, Ordering::Release);
        } else {
            std::process::exit(130);
        }
    })
    .map_err(|err| KboltError::Internal(format!("failed to install signal handler: {err}")))?;
    Ok(shutdown)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Duration;

    use tempfile::tempdir;

    use super::{jittered_safety_interval, path_matches_root, SAFETY_RESCAN_BASE};

    #[test]
    fn safety_interval_is_jittered_within_twenty_five_percent() {
        let base = SAFETY_RESCAN_BASE.as_millis() as i128;
        for seed in [1, 42, 9999] {
            let interval = jittered_safety_interval(seed).as_millis() as i128;
            let delta = (interval - base).abs();
            assert!(
                delta <= base / 4,
                "interval {interval} exceeded jitter window for base {base}"
            );
        }
    }

    #[test]
    fn safety_interval_stays_near_hour_default() {
        let interval = jittered_safety_interval(7);

        assert!(interval >= Duration::from_secs(45 * 60));
        assert!(interval <= Duration::from_secs(75 * 60));
    }

    #[cfg(unix)]
    #[test]
    fn path_matching_handles_symlinked_collection_roots() {
        let tmp = tempdir().expect("tempdir");
        let real_root = tmp.path().join("real").join("docs");
        fs::create_dir_all(&real_root).expect("create real root");
        let link_root = tmp.path().join("link-docs");
        std::os::unix::fs::symlink(&real_root, &link_root).expect("symlink root");
        let real_file = real_root.join("note.md");
        fs::write(&real_file, "hello").expect("write file");

        assert!(path_matches_root(&real_file, &link_root));
    }

    #[cfg(unix)]
    #[test]
    fn path_matching_handles_deleted_files_under_symlinked_roots() {
        let tmp = tempdir().expect("tempdir");
        let real_root = tmp.path().join("real").join("docs");
        fs::create_dir_all(&real_root).expect("create real root");
        let link_root = tmp.path().join("link-docs");
        std::os::unix::fs::symlink(&real_root, &link_root).expect("symlink root");
        let deleted_file = real_root.join("deleted.md");

        assert!(path_matches_root(&deleted_file, &link_root));
    }
}
