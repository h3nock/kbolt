use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::thread::LocalKey;
use std::time::{Duration, Instant};

use serde_json::json;

const UPDATE_PROFILE_ENV: &str = "KBOLT_UPDATE_PROFILE";
const SEARCH_PROFILE_ENV: &str = "KBOLT_SEARCH_PROFILE";

type ProfileSlot = RefCell<Option<Arc<Mutex<ProfileData>>>>;

thread_local! {
    static UPDATE_PROFILE: ProfileSlot = const { RefCell::new(None) };
    static SEARCH_PROFILE: ProfileSlot = const { RefCell::new(None) };
}

#[derive(Default)]
struct ProfileData {
    stages: BTreeMap<&'static str, StageTiming>,
    counts: BTreeMap<&'static str, u64>,
}

#[derive(Default)]
struct StageTiming {
    calls: u64,
    total: Duration,
}

pub(crate) struct UpdateProfileGuard {
    profile: Option<Arc<Mutex<ProfileData>>>,
    started: Instant,
}

impl UpdateProfileGuard {
    pub(crate) fn start() -> Self {
        let profile = if profile_enabled(UPDATE_PROFILE_ENV) {
            let profile = Arc::new(Mutex::new(ProfileData::default()));
            UPDATE_PROFILE.with(|slot| {
                *slot.borrow_mut() = Some(Arc::clone(&profile));
            });
            Some(profile)
        } else {
            None
        };

        Self {
            profile,
            started: Instant::now(),
        }
    }
}

impl Drop for UpdateProfileGuard {
    fn drop(&mut self) {
        let Some(profile) = self.profile.take() else {
            return;
        };

        let total = self.started.elapsed();
        UPDATE_PROFILE.with(|slot| {
            slot.borrow_mut().take();
        });
        emit_profile("kbolt_update_profile", total, profile);
    }
}

pub(crate) struct SearchProfileGuard {
    profile: Option<Arc<Mutex<ProfileData>>>,
    started: Instant,
}

impl SearchProfileGuard {
    pub(crate) fn start() -> Self {
        let profile = if profile_enabled(SEARCH_PROFILE_ENV) {
            let profile = Arc::new(Mutex::new(ProfileData::default()));
            SEARCH_PROFILE.with(|slot| {
                *slot.borrow_mut() = Some(Arc::clone(&profile));
            });
            Some(profile)
        } else {
            None
        };

        Self {
            profile,
            started: Instant::now(),
        }
    }
}

impl Drop for SearchProfileGuard {
    fn drop(&mut self) {
        let Some(profile) = self.profile.take() else {
            return;
        };

        let total = self.started.elapsed();
        SEARCH_PROFILE.with(|slot| {
            slot.borrow_mut().take();
        });
        emit_profile("kbolt_search_profile", total, profile);
    }
}

pub(crate) fn record_update_stage(name: &'static str, duration: Duration) {
    record_stage(&UPDATE_PROFILE, name, duration);
}

pub(crate) fn record_search_stage(name: &'static str, duration: Duration) {
    record_stage(&SEARCH_PROFILE, name, duration);
}

pub(crate) fn increment_update_count(name: &'static str, by: u64) {
    increment_count(&UPDATE_PROFILE, name, by);
}

pub(crate) fn increment_search_count(name: &'static str, by: u64) {
    increment_count(&SEARCH_PROFILE, name, by);
}

pub(crate) fn timed_update_stage<T>(name: &'static str, run: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = run();
    record_update_stage(name, started.elapsed());
    result
}

pub(crate) fn timed_search_stage<T>(name: &'static str, run: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = run();
    record_search_stage(name, started.elapsed());
    result
}

#[derive(Clone)]
pub(crate) struct SearchProfileContext {
    profile: Arc<Mutex<ProfileData>>,
}

pub(crate) fn current_search_profile() -> Option<SearchProfileContext> {
    SEARCH_PROFILE.with(|slot| {
        slot.borrow().as_ref().map(|profile| SearchProfileContext {
            profile: Arc::clone(profile),
        })
    })
}

pub(crate) fn with_search_profile<T>(
    profile: Option<SearchProfileContext>,
    run: impl FnOnce() -> T,
) -> T {
    let previous = SEARCH_PROFILE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let previous = slot.take();
        if let Some(profile) = profile {
            *slot = Some(profile.profile);
        }
        previous
    });
    let result = run();
    SEARCH_PROFILE.with(|slot| {
        *slot.borrow_mut() = previous;
    });
    result
}

fn record_stage(slot: &'static LocalKey<ProfileSlot>, name: &'static str, duration: Duration) {
    slot.with(|slot| {
        let Some(profile) = slot.borrow().as_ref().map(Arc::clone) else {
            return;
        };
        let Ok(mut data) = profile.lock() else {
            return;
        };
        let stage = data.stages.entry(name).or_default();
        stage.calls = stage.calls.saturating_add(1);
        stage.total = stage.total.saturating_add(duration);
    });
}

fn increment_count(slot: &'static LocalKey<ProfileSlot>, name: &'static str, by: u64) {
    slot.with(|slot| {
        let Some(profile) = slot.borrow().as_ref().map(Arc::clone) else {
            return;
        };
        let Ok(mut data) = profile.lock() else {
            return;
        };
        data.counts
            .entry(name)
            .and_modify(|value| *value = value.saturating_add(by))
            .or_insert(by);
    });
}

fn emit_profile(label: &'static str, total: Duration, profile: Arc<Mutex<ProfileData>>) {
    let Ok(mut data) = profile.lock() else {
        return;
    };
    let data = std::mem::take(&mut *data);

    let mut stages = serde_json::Map::new();
    for (name, stage) in data.stages {
        let total_ms = duration_ms(stage.total);
        let avg_ms = if stage.calls == 0 {
            0.0
        } else {
            total_ms / stage.calls as f64
        };
        stages.insert(
            name.to_string(),
            json!({
                "calls": stage.calls,
                "total_ms": total_ms,
                "avg_ms": avg_ms,
            }),
        );
    }

    let counts = data
        .counts
        .into_iter()
        .map(|(name, value)| (name.to_string(), json!(value)))
        .collect::<serde_json::Map<_, _>>();

    let payload = json!({
        "total_ms": duration_ms(total),
        "stages": stages,
        "counts": counts,
    });
    eprintln!("{label} {payload}");
}

fn profile_enabled(env: &str) -> bool {
    std::env::var(env)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !normalized.is_empty() && normalized != "0" && normalized != "false"
        })
        .unwrap_or(false)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
