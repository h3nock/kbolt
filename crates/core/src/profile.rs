use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde_json::json;

const UPDATE_PROFILE_ENV: &str = "KBOLT_UPDATE_PROFILE";

thread_local! {
    static UPDATE_PROFILE: RefCell<Option<UpdateProfileData>> = const { RefCell::new(None) };
}

#[derive(Default)]
struct UpdateProfileData {
    stages: BTreeMap<&'static str, StageTiming>,
    counts: BTreeMap<&'static str, u64>,
}

#[derive(Default)]
struct StageTiming {
    calls: u64,
    total: Duration,
}

pub(crate) struct UpdateProfileGuard {
    active: bool,
    started: Instant,
}

impl UpdateProfileGuard {
    pub(crate) fn start() -> Self {
        let active = update_profile_enabled();
        if active {
            UPDATE_PROFILE.with(|slot| {
                *slot.borrow_mut() = Some(UpdateProfileData::default());
            });
        }

        Self {
            active,
            started: Instant::now(),
        }
    }
}

impl Drop for UpdateProfileGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let total = self.started.elapsed();
        let data = UPDATE_PROFILE.with(|slot| slot.borrow_mut().take());
        let Some(data) = data else {
            return;
        };

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
        eprintln!("kbolt_update_profile {payload}");
    }
}

pub(crate) fn record_update_stage(name: &'static str, duration: Duration) {
    UPDATE_PROFILE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(data) = slot.as_mut() else {
            return;
        };
        let stage = data.stages.entry(name).or_default();
        stage.calls = stage.calls.saturating_add(1);
        stage.total = stage.total.saturating_add(duration);
    });
}

pub(crate) fn increment_update_count(name: &'static str, by: u64) {
    UPDATE_PROFILE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(data) = slot.as_mut() else {
            return;
        };
        data.counts
            .entry(name)
            .and_modify(|value| *value = value.saturating_add(by))
            .or_insert(by);
    });
}

pub(crate) fn timed_update_stage<T>(name: &'static str, run: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = run();
    record_update_stage(name, started.elapsed());
    result
}

fn update_profile_enabled() -> bool {
    std::env::var(UPDATE_PROFILE_ENV)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !normalized.is_empty() && normalized != "0" && normalized != "false"
        })
        .unwrap_or(false)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
