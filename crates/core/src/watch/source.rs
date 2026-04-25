use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use kbolt_types::KboltError;
use notify_debouncer_full::notify::RecursiveMode;
use notify_debouncer_full::{new_debouncer, DebounceEventResult};

use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WatchSourceEvent {
    Paths(Vec<PathBuf>),
    OverflowOrError(String),
}

pub(crate) struct NativeWatchSource {
    debouncer: notify_debouncer_full::Debouncer<
        notify_debouncer_full::notify::RecommendedWatcher,
        notify_debouncer_full::RecommendedCache,
    >,
    receiver: Receiver<DebounceEventResult>,
    roots: BTreeSet<PathBuf>,
}

impl NativeWatchSource {
    pub(crate) fn new(timeout: Duration, roots: impl IntoIterator<Item = PathBuf>) -> Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let debouncer = new_debouncer(timeout, None, sender).map_err(|err| {
            KboltError::Internal(format!("failed to create native watcher: {err}"))
        })?;
        let mut source = Self {
            debouncer,
            receiver,
            roots: BTreeSet::new(),
        };
        source.sync_roots(roots)?;
        Ok(source)
    }

    pub(crate) fn sync_roots(&mut self, roots: impl IntoIterator<Item = PathBuf>) -> Result<()> {
        let next = roots
            .into_iter()
            .map(normalize_root)
            .collect::<BTreeSet<_>>();

        for removed in self.roots.difference(&next).cloned().collect::<Vec<_>>() {
            if let Err(err) = self.debouncer.unwatch(&removed) {
                return Err(KboltError::Internal(format!(
                    "failed to unwatch {}: {err}",
                    removed.display()
                ))
                .into());
            }
            self.roots.remove(&removed);
        }

        for added in next.difference(&self.roots).cloned().collect::<Vec<_>>() {
            if let Err(err) = self.debouncer.watch(&added, RecursiveMode::Recursive) {
                return Err(KboltError::Internal(format!(
                    "failed to watch {}: {err}",
                    added.display()
                ))
                .into());
            }
            self.roots.insert(added);
        }

        Ok(())
    }

    pub(crate) fn drain_events(&self) -> Vec<WatchSourceEvent> {
        let mut events = Vec::new();
        while let Ok(result) = self.receiver.try_recv() {
            match result {
                Ok(debounced) => {
                    let paths = debounced
                        .into_iter()
                        .flat_map(|event| event.paths.iter().cloned().collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    if !paths.is_empty() {
                        events.push(WatchSourceEvent::Paths(paths));
                    }
                }
                Err(errors) => {
                    let message = errors
                        .into_iter()
                        .map(|err| err.to_string())
                        .collect::<Vec<_>>()
                        .join("; ");
                    events.push(WatchSourceEvent::OverflowOrError(message));
                }
            }
        }
        events
    }
}

fn normalize_root(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

pub(crate) fn path_has_component(
    path: &Path,
    predicate: impl Fn(&std::ffi::OsStr) -> bool,
) -> bool {
    path.components().any(|component| match component {
        std::path::Component::Normal(name) => predicate(name),
        _ => false,
    })
}
