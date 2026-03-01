use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::Result;
use kbolt_types::KboltError;
use serde::{Deserialize, Serialize};

const APP_NAME: &str = "kbolt";
const CONFIG_FILENAME: &str = "index.toml";
const DEFAULT_EMBED_MODEL: &str = "google/EmbeddingGemma-256";
const DEFAULT_RERANKER_MODEL: &str = "ExpedientFalcon/qwen3-reranker-0.6b-q8";
const DEFAULT_EXPANDER_MODEL: &str = "Qwen/Qwen3-1.7B-q4";
const DEFAULT_REAP_DAYS: u32 = 7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub config_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub default_space: Option<String>,
    pub models: ModelConfig,
    pub reaping: ReapingConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    pub embed: String,
    pub reranker: String,
    pub expander: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReapingConfig {
    pub days: u32,
}

pub fn load(config_path: Option<&Path>) -> Result<Config> {
    let config_dir = resolve_config_dir(config_path)?;
    let cache_dir = default_cache_dir()?;
    let config_file = config_dir.join(CONFIG_FILENAME);

    load_from_file(&config_file, &config_dir, &cache_dir)
}

pub fn save(config: &Config) -> Result<()> {
    fs::create_dir_all(&config.config_dir)?;
    fs::create_dir_all(&config.cache_dir)?;

    let file_config = FileConfig::from(config);
    let serialized = toml::to_string_pretty(&file_config)?;
    let content = if serialized.ends_with('\n') {
        serialized
    } else {
        format!("{serialized}\n")
    };

    let config_file = config.config_dir.join(CONFIG_FILENAME);
    fs::write(config_file, content)?;
    Ok(())
}

fn resolve_config_dir(config_path: Option<&Path>) -> Result<PathBuf> {
    match config_path {
        None => default_config_dir(),
        Some(path) => {
            if path.file_name() == Some(OsStr::new(CONFIG_FILENAME)) {
                let parent = path.parent().ok_or_else(|| {
                    KboltError::Config(format!("invalid config path: {}", path.display()))
                })?;
                return Ok(parent.to_path_buf());
            }

            if path.extension() == Some(OsStr::new("toml")) {
                return Err(
                    KboltError::Config(format!(
                        "config file override must be named {CONFIG_FILENAME}: {}",
                        path.display()
                    ))
                    .into(),
                );
            }

            Ok(path.to_path_buf())
        }
    }
}

fn default_config_dir() -> Result<PathBuf> {
    let base = dirs::config_dir()
        .ok_or_else(|| KboltError::Config("unable to determine user config directory".into()))?;
    Ok(base.join(APP_NAME))
}

fn default_cache_dir() -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .ok_or_else(|| KboltError::Config("unable to determine user cache directory".into()))?;
    Ok(base.join(APP_NAME))
}

fn load_from_file(config_file: &Path, config_dir: &Path, cache_dir: &Path) -> Result<Config> {
    fs::create_dir_all(config_dir)?;
    fs::create_dir_all(cache_dir)?;

    if !config_file.exists() {
        let default_config = Config {
            config_dir: config_dir.to_path_buf(),
            cache_dir: cache_dir.to_path_buf(),
            default_space: None,
            models: ModelConfig {
                embed: DEFAULT_EMBED_MODEL.to_string(),
                reranker: DEFAULT_RERANKER_MODEL.to_string(),
                expander: DEFAULT_EXPANDER_MODEL.to_string(),
            },
            reaping: ReapingConfig {
                days: DEFAULT_REAP_DAYS,
            },
        };
        save(&default_config)?;
    }

    let raw = fs::read_to_string(config_file)?;
    let file_config: FileConfig = toml::from_str(&raw)?;

    Ok(Config {
        config_dir: config_dir.to_path_buf(),
        cache_dir: cache_dir.to_path_buf(),
        default_space: file_config.default_space,
        models: ModelConfig {
            embed: file_config.models.embed,
            reranker: file_config.models.reranker,
            expander: file_config.models.expander,
        },
        reaping: ReapingConfig {
            days: file_config.reaping.days,
        },
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct FileConfig {
    #[serde(default)]
    default_space: Option<String>,
    #[serde(default)]
    models: FileModelConfig,
    #[serde(default)]
    reaping: FileReapingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FileModelConfig {
    #[serde(default = "default_embed_model")]
    embed: String,
    #[serde(default = "default_reranker_model")]
    reranker: String,
    #[serde(default = "default_expander_model")]
    expander: String,
}

impl Default for FileModelConfig {
    fn default() -> Self {
        Self {
            embed: default_embed_model(),
            reranker: default_reranker_model(),
            expander: default_expander_model(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FileReapingConfig {
    #[serde(default = "default_reap_days")]
    days: u32,
}

impl Default for FileReapingConfig {
    fn default() -> Self {
        Self {
            days: default_reap_days(),
        }
    }
}

impl From<&Config> for FileConfig {
    fn from(value: &Config) -> Self {
        Self {
            default_space: value.default_space.clone(),
            models: FileModelConfig {
                embed: value.models.embed.clone(),
                reranker: value.models.reranker.clone(),
                expander: value.models.expander.clone(),
            },
            reaping: FileReapingConfig {
                days: value.reaping.days,
            },
        }
    }
}

fn default_embed_model() -> String {
    DEFAULT_EMBED_MODEL.to_string()
}

fn default_reranker_model() -> String {
    DEFAULT_RERANKER_MODEL.to_string()
}

fn default_expander_model() -> String {
    DEFAULT_EXPANDER_MODEL.to_string()
}

fn default_reap_days() -> u32 {
    DEFAULT_REAP_DAYS
}

#[cfg(test)]
mod tests;
