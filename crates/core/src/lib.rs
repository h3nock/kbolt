use std::sync::Arc;

pub mod config;
pub mod doctor;
pub mod engine;
pub mod error;
pub mod eval_import;
mod eval_store;
pub mod ingest;
pub mod local;
mod lock;
mod models;
mod schedule_backend;
mod schedule_state_store;
mod schedule_store;
mod schedule_support;
pub mod storage;

pub use error::{CoreError, Result};
pub type RecoveryNoticeSink = Arc<dyn Fn(&str) + Send + Sync>;
