use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use super::{
    duration_millis, CollectionKey, WatchCollection, KEYWORD_CAP, KEYWORD_QUIET,
    SEMANTIC_BLOCK_BACKOFF, SEMANTIC_QUIET, SEMANTIC_UNAVAILABLE_BACKOFF, UPDATE_ERROR_BACKOFF,
};

pub(crate) type Tick = u64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ScheduledAction {
    RunKeywordUpdate { key: CollectionKey },
    RunSemanticUpdate { key: CollectionKey },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SemanticScheduleState {
    None,
    Pending {
        pending_chunks: usize,
    },
    Unavailable {
        pending_chunks: usize,
        reason: String,
        retry_at: Tick,
    },
    Blocked {
        space: String,
        reason: String,
        retry_at: Tick,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CollectionScheduleStatus {
    pub key: CollectionKey,
    pub path: PathBuf,
    pub dirty: bool,
    pub semantic: SemanticScheduleState,
    pub last_event_at: Option<Tick>,
    pub last_keyword_refresh_at: Option<Tick>,
    pub last_semantic_refresh_at: Option<Tick>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpaceBlock {
    pub reason: String,
    pub set_at: Tick,
    pub backoff_until: Tick,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CatalogDiff {
    pub added: Vec<CollectionKey>,
    pub removed: Vec<CollectionKey>,
    pub path_changed: Vec<CollectionKey>,
}

#[derive(Debug, Default)]
pub(crate) struct WatchScheduler {
    collections: BTreeMap<CollectionKey, CollectionSchedule>,
    space_blocks: BTreeMap<String, SpaceBlock>,
}

#[derive(Debug, Clone)]
struct CollectionSchedule {
    path: PathBuf,
    dirty_since: Option<Tick>,
    last_event_at: Option<Tick>,
    last_keyword_refresh_at: Option<Tick>,
    last_semantic_refresh_at: Option<Tick>,
    keyword_pending: bool,
    pending_chunks: usize,
    semantic_unavailable: Option<SemanticUnavailable>,
    update_backoff_until: Option<Tick>,
    last_error: Option<String>,
}

#[derive(Debug, Clone)]
struct SemanticUnavailable {
    reason: String,
    retry_at: Tick,
}

impl WatchScheduler {
    pub(crate) fn sync_catalog(
        &mut self,
        collections: impl IntoIterator<Item = WatchCollection>,
        now: Tick,
    ) -> CatalogDiff {
        let incoming = collections
            .into_iter()
            .map(|collection| (collection.key, collection.path))
            .collect::<BTreeMap<_, _>>();
        let existing_keys = self.collections.keys().cloned().collect::<BTreeSet<_>>();
        let incoming_keys = incoming.keys().cloned().collect::<BTreeSet<_>>();

        let mut diff = CatalogDiff::default();

        for removed in existing_keys.difference(&incoming_keys) {
            self.collections.remove(removed);
            diff.removed.push(removed.clone());
        }

        for (key, path) in incoming {
            match self.collections.get_mut(&key) {
                Some(existing) if existing.path != path => {
                    existing.path = path;
                    existing.dirty_since = Some(now);
                    existing.last_event_at = Some(now);
                    existing.keyword_pending = true;
                    existing.last_error = None;
                    diff.path_changed.push(key);
                }
                Some(_) => {}
                None => {
                    self.collections.insert(
                        key.clone(),
                        CollectionSchedule {
                            path,
                            dirty_since: Some(now),
                            last_event_at: Some(now),
                            last_keyword_refresh_at: None,
                            last_semantic_refresh_at: None,
                            keyword_pending: true,
                            pending_chunks: 0,
                            semantic_unavailable: None,
                            update_backoff_until: None,
                            last_error: None,
                        },
                    );
                    diff.added.push(key);
                }
            }
        }

        diff
    }

    pub(crate) fn mark_dirty(&mut self, key: &CollectionKey, now: Tick) {
        if let Some(collection) = self.collections.get_mut(key) {
            if collection.dirty_since.is_none() {
                collection.dirty_since = Some(now);
            }
            collection.last_event_at = Some(now);
            collection.keyword_pending = true;
            collection.last_error = None;
        }
    }

    pub(crate) fn mark_all_dirty(&mut self, now: Tick) {
        let keys = self.collections.keys().cloned().collect::<Vec<_>>();
        for key in keys {
            self.mark_dirty(&key, now);
        }
    }

    pub(crate) fn set_pending_chunks(&mut self, key: &CollectionKey, pending_chunks: usize) {
        if let Some(collection) = self.collections.get_mut(key) {
            collection.pending_chunks = pending_chunks;
            if pending_chunks == 0 {
                collection.semantic_unavailable = None;
            }
        }
    }

    pub(crate) fn clear_resolved_space_blocks(&mut self) {
        let blocked_spaces = self.space_blocks.keys().cloned().collect::<Vec<_>>();
        for space in blocked_spaces {
            let has_pending = self
                .collections
                .iter()
                .any(|(key, collection)| key.space == space && collection.pending_chunks > 0);
            if !has_pending {
                self.space_blocks.remove(&space);
            }
        }
    }

    pub(crate) fn mark_keyword_success(
        &mut self,
        key: &CollectionKey,
        now: Tick,
        pending_chunks: usize,
    ) {
        if let Some(collection) = self.collections.get_mut(key) {
            collection.keyword_pending = false;
            collection.dirty_since = None;
            collection.last_keyword_refresh_at = Some(now);
            collection.pending_chunks = pending_chunks;
            collection.update_backoff_until = None;
            collection.last_error = None;
            if pending_chunks == 0 {
                collection.semantic_unavailable = None;
            }
        }
    }

    pub(crate) fn mark_detection_keyword_success(
        &mut self,
        key: &CollectionKey,
        now: Tick,
        pending_chunks: usize,
        changed: bool,
    ) {
        self.mark_keyword_success(key, now, pending_chunks);
        if changed {
            if let Some(collection) = self.collections.get_mut(key) {
                collection.last_event_at = Some(now);
            }
        }
    }

    pub(crate) fn mark_semantic_success(
        &mut self,
        key: &CollectionKey,
        now: Tick,
        pending_chunks: usize,
    ) {
        if let Some(collection) = self.collections.get_mut(key) {
            collection.last_semantic_refresh_at = Some(now);
            collection.pending_chunks = pending_chunks;
            collection.semantic_unavailable = None;
            collection.update_backoff_until = None;
            collection.last_error = None;
            self.space_blocks.remove(&key.space);
        }
    }

    pub(crate) fn clear_semantic_unavailable(&mut self) {
        for collection in self.collections.values_mut() {
            if collection.semantic_unavailable.is_some() {
                collection.semantic_unavailable = None;
                collection.last_error = None;
            }
        }
    }

    pub(crate) fn mark_update_error(&mut self, key: &CollectionKey, now: Tick, error: String) {
        if let Some(collection) = self.collections.get_mut(key) {
            collection.update_backoff_until = Some(now + duration_millis(UPDATE_ERROR_BACKOFF));
            collection.last_error = Some(error);
        }
    }

    pub(crate) fn mark_semantic_unavailable(
        &mut self,
        key: &CollectionKey,
        now: Tick,
        pending_chunks: usize,
        reason: String,
    ) {
        if let Some(collection) = self.collections.get_mut(key) {
            collection.pending_chunks = pending_chunks;
            collection.semantic_unavailable = Some(SemanticUnavailable {
                reason: reason.clone(),
                retry_at: now + duration_millis(SEMANTIC_UNAVAILABLE_BACKOFF),
            });
            collection.last_error = Some(reason);
        }
    }

    pub(crate) fn mark_space_blocked(&mut self, space: &str, now: Tick, reason: String) {
        self.space_blocks.insert(
            space.to_string(),
            SpaceBlock {
                reason,
                set_at: now,
                backoff_until: now + duration_millis(SEMANTIC_BLOCK_BACKOFF),
            },
        );
    }

    pub(crate) fn due_actions(&self, now: Tick) -> Vec<ScheduledAction> {
        let mut actions = Vec::new();
        let mut rechecking_blocked_spaces = BTreeSet::new();

        for (key, collection) in &self.collections {
            if collection
                .update_backoff_until
                .is_some_and(|backoff| now < backoff)
            {
                continue;
            }

            if collection.keyword_pending && keyword_due(collection, now) {
                actions.push(ScheduledAction::RunKeywordUpdate { key: key.clone() });
                continue;
            }

            if collection.pending_chunks == 0 {
                continue;
            }

            if let Some(block) = self.space_blocks.get(&key.space) {
                if now >= block.backoff_until && rechecking_blocked_spaces.insert(key.space.clone())
                {
                    actions.push(ScheduledAction::RunSemanticUpdate { key: key.clone() });
                }
                continue;
            }

            if let Some(unavailable) = collection.semantic_unavailable.as_ref() {
                if now < unavailable.retry_at {
                    continue;
                }
            }

            if semantic_due(collection, now) {
                actions.push(ScheduledAction::RunSemanticUpdate { key: key.clone() });
            }
        }

        actions
    }

    pub(crate) fn collections(&self) -> Vec<CollectionScheduleStatus> {
        self.collections
            .iter()
            .map(|(key, collection)| {
                let semantic = if let Some(block) = self.space_blocks.get(&key.space) {
                    SemanticScheduleState::Blocked {
                        space: key.space.clone(),
                        reason: block.reason.clone(),
                        retry_at: block.backoff_until,
                    }
                } else if collection.pending_chunks == 0 {
                    SemanticScheduleState::None
                } else if let Some(unavailable) = collection.semantic_unavailable.as_ref() {
                    SemanticScheduleState::Unavailable {
                        pending_chunks: collection.pending_chunks,
                        reason: unavailable.reason.clone(),
                        retry_at: unavailable.retry_at,
                    }
                } else {
                    SemanticScheduleState::Pending {
                        pending_chunks: collection.pending_chunks,
                    }
                };

                CollectionScheduleStatus {
                    key: key.clone(),
                    path: collection.path.clone(),
                    dirty: collection.keyword_pending,
                    semantic,
                    last_event_at: collection.last_event_at,
                    last_keyword_refresh_at: collection.last_keyword_refresh_at,
                    last_semantic_refresh_at: collection.last_semantic_refresh_at,
                    last_error: collection.last_error.clone(),
                }
            })
            .collect()
    }

    pub(crate) fn space_blocks(&self) -> Vec<(String, SpaceBlock)> {
        self.space_blocks
            .iter()
            .map(|(space, block)| (space.clone(), block.clone()))
            .collect()
    }
}

fn keyword_due(collection: &CollectionSchedule, now: Tick) -> bool {
    let Some(last_event_at) = collection.last_event_at else {
        return true;
    };
    let quiet_elapsed = now.saturating_sub(last_event_at) >= duration_millis(KEYWORD_QUIET);
    let cap_elapsed = collection
        .dirty_since
        .is_some_and(|dirty_since| now.saturating_sub(dirty_since) >= duration_millis(KEYWORD_CAP));
    quiet_elapsed || cap_elapsed
}

fn semantic_due(collection: &CollectionSchedule, now: Tick) -> bool {
    let Some(last_event_at) = collection.last_event_at else {
        return true;
    };
    now.saturating_sub(last_event_at) >= duration_millis(SEMANTIC_QUIET)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{ScheduledAction, SemanticScheduleState, WatchScheduler};
    use crate::watch::{
        duration_millis, CollectionKey, WatchCollection, KEYWORD_CAP, KEYWORD_QUIET,
        SEMANTIC_BLOCK_BACKOFF, SEMANTIC_QUIET, UPDATE_ERROR_BACKOFF,
    };

    fn collection() -> WatchCollection {
        WatchCollection {
            key: CollectionKey::new("work", "api"),
            path: PathBuf::from("/tmp/api"),
        }
    }

    fn named_collection(name: &str, path: &str) -> WatchCollection {
        WatchCollection {
            key: CollectionKey::new("work", name),
            path: PathBuf::from(path),
        }
    }

    #[test]
    fn sync_catalog_reports_added_removed_and_path_changed() {
        let mut scheduler = WatchScheduler::default();
        let api = named_collection("api", "/tmp/api");
        let docs = named_collection("docs", "/tmp/docs");
        scheduler.sync_catalog([api.clone(), docs.clone()], 0);

        let diff = scheduler.sync_catalog(
            [
                named_collection("api", "/tmp/api-renamed"),
                named_collection("wiki", "/tmp/wiki"),
            ],
            10,
        );

        assert_eq!(diff.added, vec![CollectionKey::new("work", "wiki")]);
        assert_eq!(diff.removed, vec![docs.key]);
        assert_eq!(diff.path_changed, vec![api.key.clone()]);

        let api_status = scheduler
            .collections()
            .into_iter()
            .find(|status| status.key == api.key)
            .expect("api status");
        assert_eq!(api_status.path, PathBuf::from("/tmp/api-renamed"));
        assert!(api_status.dirty);
    }

    #[test]
    fn burst_events_wait_for_keyword_quiet() {
        let mut scheduler = WatchScheduler::default();
        let key = collection().key.clone();
        scheduler.sync_catalog([collection()], 0);
        scheduler.mark_dirty(&key, 10_000);
        scheduler.mark_dirty(&key, 20_000);

        assert_eq!(scheduler.due_actions(20_000), Vec::new());

        assert_eq!(
            scheduler.due_actions(20_000 + duration_millis(KEYWORD_QUIET)),
            vec![ScheduledAction::RunKeywordUpdate { key }]
        );
    }

    #[test]
    fn continuous_events_trigger_keyword_cap() {
        let mut scheduler = WatchScheduler::default();
        let key = collection().key.clone();
        scheduler.sync_catalog([collection()], 0);
        scheduler.mark_dirty(&key, 0);
        scheduler.mark_dirty(&key, duration_millis(KEYWORD_CAP) - 1_000);

        assert_eq!(
            scheduler.due_actions(duration_millis(KEYWORD_CAP)),
            vec![ScheduledAction::RunKeywordUpdate { key }]
        );
    }

    #[test]
    fn continuous_events_do_not_trigger_semantic() {
        let mut scheduler = WatchScheduler::default();
        let key = collection().key.clone();
        scheduler.sync_catalog([collection()], 0);
        scheduler.mark_keyword_success(&key, duration_millis(KEYWORD_QUIET), 3);
        scheduler.mark_dirty(&key, duration_millis(SEMANTIC_QUIET) - 1_000);

        assert!(!scheduler
            .due_actions(duration_millis(SEMANTIC_QUIET))
            .iter()
            .any(|action| matches!(action, ScheduledAction::RunSemanticUpdate { .. })));
    }

    #[test]
    fn semantic_fires_after_quiet_when_chunks_are_pending() {
        let mut scheduler = WatchScheduler::default();
        let key = collection().key.clone();
        scheduler.sync_catalog([collection()], 0);
        scheduler.mark_keyword_success(&key, 1_000, 2);

        assert_eq!(
            scheduler.due_actions(duration_millis(SEMANTIC_QUIET)),
            vec![ScheduledAction::RunSemanticUpdate { key }]
        );
    }

    #[test]
    fn unavailable_semantic_waits_for_retry() {
        let mut scheduler = WatchScheduler::default();
        let key = collection().key.clone();
        scheduler.sync_catalog([collection()], 0);
        scheduler.mark_keyword_success(&key, 1_000, 4);
        scheduler.mark_semantic_unavailable(&key, 1_000, 4, "embedder down".to_string());

        assert_eq!(
            scheduler.collections()[0].semantic,
            SemanticScheduleState::Unavailable {
                pending_chunks: 4,
                reason: "embedder down".to_string(),
                retry_at: 601_000,
            }
        );
        assert_eq!(
            scheduler.due_actions(duration_millis(SEMANTIC_QUIET)),
            Vec::new()
        );
    }

    #[test]
    fn clearing_semantic_unavailable_allows_retry_without_waiting_for_old_timer() {
        let mut scheduler = WatchScheduler::default();
        let key = collection().key.clone();
        scheduler.sync_catalog([collection()], 0);
        scheduler.mark_keyword_success(&key, 1_000, 4);
        scheduler.mark_semantic_unavailable(&key, 1_000, 4, "embedder down".to_string());

        scheduler.clear_semantic_unavailable();

        assert_eq!(
            scheduler.due_actions(duration_millis(SEMANTIC_QUIET)),
            vec![ScheduledAction::RunSemanticUpdate { key }]
        );
    }

    #[test]
    fn space_block_overrides_collection_pending() {
        let mut scheduler = WatchScheduler::default();
        let key = collection().key.clone();
        scheduler.sync_catalog([collection()], 0);
        scheduler.mark_keyword_success(&key, 1_000, 9);
        scheduler.mark_space_blocked("work", 2_000, "repair required".to_string());

        assert!(matches!(
            scheduler.collections()[0].semantic,
            SemanticScheduleState::Blocked { .. }
        ));
    }

    #[test]
    fn blocked_space_rechecks_once_after_backoff() {
        let mut scheduler = WatchScheduler::default();
        let api = named_collection("api", "/tmp/api");
        let docs = named_collection("docs", "/tmp/docs");
        let api_key = api.key.clone();
        scheduler.sync_catalog([api, docs], 0);
        scheduler.mark_keyword_success(&api_key, 1_000, 4);
        scheduler.mark_keyword_success(&CollectionKey::new("work", "docs"), 1_000, 6);
        scheduler.mark_space_blocked("work", 2_000, "repair required".to_string());

        assert_eq!(
            scheduler.due_actions(2_000 + duration_millis(SEMANTIC_BLOCK_BACKOFF) - 1),
            Vec::new()
        );

        let actions = scheduler.due_actions(2_000 + duration_millis(SEMANTIC_BLOCK_BACKOFF));
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            ScheduledAction::RunSemanticUpdate { .. }
        ));
    }

    #[test]
    fn update_error_suppresses_actions_until_backoff_expires() {
        let mut scheduler = WatchScheduler::default();
        let key = collection().key.clone();
        scheduler.sync_catalog([collection()], 0);
        scheduler.mark_dirty(&key, duration_millis(KEYWORD_QUIET));
        scheduler.mark_update_error(&key, 1_000, "lock busy".to_string());

        assert_eq!(
            scheduler.due_actions(1_000 + duration_millis(UPDATE_ERROR_BACKOFF) - 1),
            Vec::new()
        );

        assert_eq!(
            scheduler.due_actions(1_000 + duration_millis(UPDATE_ERROR_BACKOFF)),
            vec![ScheduledAction::RunKeywordUpdate { key }]
        );
    }

    #[test]
    fn semantic_success_clears_space_block_even_when_chunks_remain_pending() {
        let mut scheduler = WatchScheduler::default();
        let key = collection().key.clone();
        scheduler.sync_catalog([collection()], 0);
        scheduler.mark_keyword_success(&key, 1_000, 4);
        scheduler.mark_space_blocked("work", 2_000, "repair required".to_string());

        scheduler.mark_semantic_success(&key, 3_000, 2);

        assert!(scheduler.space_blocks().is_empty());
        assert_eq!(
            scheduler.collections()[0].semantic,
            SemanticScheduleState::Pending { pending_chunks: 2 }
        );
    }

    #[test]
    fn resolved_space_block_clears_when_no_collections_have_pending_chunks() {
        let mut scheduler = WatchScheduler::default();
        let key = collection().key.clone();
        scheduler.sync_catalog([collection()], 0);
        scheduler.mark_keyword_success(&key, 1_000, 4);
        scheduler.mark_space_blocked("work", 2_000, "repair required".to_string());

        scheduler.set_pending_chunks(&key, 0);
        scheduler.clear_resolved_space_blocks();

        assert!(scheduler.space_blocks().is_empty());
        assert_eq!(
            scheduler.collections()[0].semantic,
            SemanticScheduleState::None
        );
    }
}
