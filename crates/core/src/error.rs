use kbolt_types::KboltError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error(transparent)]
    Domain(#[from] KboltError),

    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("toml decode error: {0}")]
    TomlDe(#[from] toml::de::Error),

    #[error("toml encode error: {0}")]
    TomlSer(#[from] toml::ser::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("internal error: {0}")]
    Internal(String),
}

impl CoreError {
    pub fn poisoned(resource: &str) -> Self {
        Self::Internal(format!("{resource} mutex poisoned"))
    }
}

impl From<CoreError> for KboltError {
    fn from(value: CoreError) -> Self {
        match value {
            CoreError::Domain(err) => err,
            CoreError::Sqlite(err) => KboltError::Database(err.to_string()),
            CoreError::TomlDe(err) => KboltError::Config(format!("failed to parse config: {err}")),
            CoreError::TomlSer(err) => {
                KboltError::Config(format!("failed to serialize config: {err}"))
            }
            CoreError::Json(err) => KboltError::Internal(format!("json error: {err}")),
            CoreError::Io(err) => KboltError::Io(err),
            CoreError::Internal(msg) => KboltError::Internal(msg),
        }
    }
}

pub type Result<T> = std::result::Result<T, CoreError>;
