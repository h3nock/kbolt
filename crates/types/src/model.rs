use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PullReport {
    pub downloaded: Vec<String>,
    pub already_present: Vec<String>,
    pub total_bytes: u64,
}
