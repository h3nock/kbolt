use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LocalAction {
    Setup,
    Start,
    Stop,
    Status,
    EnableDeep,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocalReport {
    pub action: LocalAction,
    pub config_file: PathBuf,
    pub cache_dir: PathBuf,
    pub llama_server_path: Option<PathBuf>,
    pub ready: bool,
    pub notes: Vec<String>,
    pub services: Vec<LocalServiceReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocalServiceReport {
    pub name: String,
    pub provider: String,
    pub enabled: bool,
    pub configured: bool,
    pub managed: bool,
    pub running: bool,
    pub ready: bool,
    pub model: String,
    pub model_path: PathBuf,
    pub endpoint: String,
    pub port: u16,
    pub pid: Option<u32>,
    pub pid_file: PathBuf,
    pub log_file: PathBuf,
    pub issue: Option<String>,
}
