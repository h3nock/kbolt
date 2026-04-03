use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DoctorReport {
    pub setup_status: DoctorSetupStatus,
    pub config_file: Option<PathBuf>,
    pub config_dir: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub ready: bool,
    pub checks: Vec<DoctorCheck>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DoctorSetupStatus {
    ConfigMissing,
    ConfigInvalid,
    NotConfigured,
    Configured,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DoctorCheck {
    pub id: String,
    pub scope: String,
    pub status: DoctorCheckStatus,
    pub elapsed_ms: u64,
    pub message: String,
    pub fix: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DoctorCheckStatus {
    Pass,
    Warn,
    Fail,
}
