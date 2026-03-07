use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AddScheduleRequest {
    pub trigger: ScheduleTrigger,
    pub scope: ScheduleScope,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduleAddResponse {
    pub schedule: ScheduleDefinition,
    pub backend: ScheduleBackend,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduleDefinition {
    pub id: String,
    pub trigger: ScheduleTrigger,
    pub scope: ScheduleScope,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScheduleTrigger {
    Every {
        interval: ScheduleInterval,
    },
    Daily {
        time: String,
    },
    Weekly {
        weekdays: Vec<ScheduleWeekday>,
        time: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduleInterval {
    pub value: u32,
    pub unit: ScheduleIntervalUnit,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleIntervalUnit {
    Minutes,
    Hours,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleWeekday {
    Mon,
    Tue,
    Wed,
    Thu,
    Fri,
    Sat,
    Sun,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScheduleScope {
    All,
    Space {
        space: String,
    },
    Collections {
        space: String,
        collections: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduleStatusResponse {
    pub schedules: Vec<ScheduleStatusEntry>,
    pub orphans: Vec<ScheduleOrphan>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduleStatusEntry {
    pub schedule: ScheduleDefinition,
    pub backend: ScheduleBackend,
    pub state: ScheduleState,
    pub run_state: ScheduleRunState,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleBackend {
    Launchd,
    SystemdUser,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleState {
    Installed,
    Drifted,
    TargetMissing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ScheduleRunState {
    pub last_started: Option<String>,
    pub last_finished: Option<String>,
    pub last_result: Option<ScheduleRunResult>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleRunResult {
    Success,
    SkippedLock,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduleOrphan {
    pub id: String,
    pub backend: ScheduleBackend,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RemoveScheduleSelector {
    Id { id: String },
    All,
    Scope { scope: ScheduleScope },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoveScheduleRequest {
    pub selector: RemoveScheduleSelector,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduleRemoveResponse {
    pub removed_ids: Vec<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        ScheduleBackend, ScheduleDefinition, ScheduleInterval, ScheduleIntervalUnit,
        ScheduleOrphan, ScheduleRunResult, ScheduleRunState, ScheduleScope, ScheduleState,
        ScheduleStatusEntry, ScheduleStatusResponse, ScheduleTrigger, ScheduleWeekday,
    };

    #[test]
    fn schedule_definition_serializes_weekly_collection_scope_as_snake_case() {
        let value = serde_json::to_value(ScheduleDefinition {
            id: "s2".to_string(),
            trigger: ScheduleTrigger::Weekly {
                weekdays: vec![ScheduleWeekday::Mon, ScheduleWeekday::Fri],
                time: "15:00".to_string(),
            },
            scope: ScheduleScope::Collections {
                space: "work".to_string(),
                collections: vec!["api".to_string(), "docs".to_string()],
            },
        })
        .expect("serialize schedule definition");

        assert_eq!(
            value,
            json!({
                "id": "s2",
                "trigger": {
                    "kind": "weekly",
                    "weekdays": ["mon", "fri"],
                    "time": "15:00"
                },
                "scope": {
                    "kind": "collections",
                    "space": "work",
                    "collections": ["api", "docs"]
                }
            })
        );
    }

    #[test]
    fn schedule_status_response_serializes_run_state_and_orphans() {
        let value = serde_json::to_value(ScheduleStatusResponse {
            schedules: vec![ScheduleStatusEntry {
                schedule: ScheduleDefinition {
                    id: "s1".to_string(),
                    trigger: ScheduleTrigger::Every {
                        interval: ScheduleInterval {
                            value: 30,
                            unit: ScheduleIntervalUnit::Minutes,
                        },
                    },
                    scope: ScheduleScope::All,
                },
                backend: ScheduleBackend::Launchd,
                state: ScheduleState::Installed,
                run_state: ScheduleRunState {
                    last_started: Some("2026-03-07T15:00:00Z".to_string()),
                    last_finished: Some("2026-03-07T15:00:09Z".to_string()),
                    last_result: Some(ScheduleRunResult::Success),
                    last_error: None,
                },
            }],
            orphans: vec![ScheduleOrphan {
                id: "s9".to_string(),
                backend: ScheduleBackend::SystemdUser,
            }],
        })
        .expect("serialize schedule status response");

        assert_eq!(
            value,
            json!({
                "schedules": [
                    {
                        "schedule": {
                            "id": "s1",
                            "trigger": {
                                "kind": "every",
                                "interval": {
                                    "value": 30,
                                    "unit": "minutes"
                                }
                            },
                            "scope": {
                                "kind": "all"
                            }
                        },
                        "backend": "launchd",
                        "state": "installed",
                        "run_state": {
                            "last_started": "2026-03-07T15:00:00Z",
                            "last_finished": "2026-03-07T15:00:09Z",
                            "last_result": "success",
                            "last_error": null
                        }
                    }
                ],
                "orphans": [
                    {
                        "id": "s9",
                        "backend": "systemd_user"
                    }
                ]
            })
        );
    }
}
