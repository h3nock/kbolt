use std::collections::HashMap;
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
const DEFAULT_CHUNK_TARGET_TOKENS: usize = 450;
const DEFAULT_CHUNK_SOFT_MAX_TOKENS: usize = 550;
const DEFAULT_CHUNK_HARD_MAX_TOKENS: usize = 750;
const DEFAULT_CHUNK_BOUNDARY_OVERLAP_TOKENS: usize = 48;
const DEFAULT_CHUNK_NEIGHBOR_WINDOW: usize = 1;
const DEFAULT_CHUNK_CONTEXTUAL_PREFIX: bool = true;
const DEFAULT_CODE_CHUNK_TARGET_TOKENS: usize = 320;
const DEFAULT_CODE_CHUNK_SOFT_MAX_TOKENS: usize = 420;
const DEFAULT_CODE_CHUNK_HARD_MAX_TOKENS: usize = 560;
const DEFAULT_CODE_CHUNK_BOUNDARY_OVERLAP_TOKENS: usize = 24;
const DEFAULT_EMBEDDING_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 32;
const DEFAULT_EMBEDDING_MAX_RETRIES: u32 = 2;
const DEFAULT_INFERENCE_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_INFERENCE_MAX_RETRIES: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub config_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub default_space: Option<String>,
    pub models: ModelConfig,
    pub embeddings: Option<EmbeddingConfig>,
    pub inference: InferenceConfig,
    pub reaping: ReapingConfig,
    pub chunking: ChunkingConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: EmbeddingProvider,
    pub model: String,
    pub base_url: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_embedding_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_embedding_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_embedding_max_retries")]
    pub max_retries: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
    #[serde(rename = "voyage")]
    Voyage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InferenceConfig {
    #[serde(default)]
    pub reranker: Option<TextInferenceConfig>,
    #[serde(default)]
    pub expander: Option<TextInferenceConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextInferenceConfig {
    pub provider: TextInferenceProvider,
    pub output_mode: TextInferenceOutputMode,
    pub model: String,
    pub base_url: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_inference_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_inference_max_retries")]
    pub max_retries: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextInferenceProvider {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextInferenceOutputMode {
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "text")]
    Text,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_embedder_source")]
    pub embedder: ModelSourceConfig,
    #[serde(default = "default_reranker_source")]
    pub reranker: ModelSourceConfig,
    #[serde(default = "default_expander_source")]
    pub expander: ModelSourceConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelProvider {
    #[serde(rename = "huggingface")]
    HuggingFace,
}

impl Default for ModelProvider {
    fn default() -> Self {
        Self::HuggingFace
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSourceConfig {
    #[serde(default)]
    pub provider: ModelProvider,
    pub id: String,
    #[serde(default)]
    pub revision: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            embedder: default_embedder_source(),
            reranker: default_reranker_source(),
            expander: default_expander_source(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReapingConfig {
    pub days: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkingConfig {
    pub defaults: ChunkPolicy,
    pub profiles: HashMap<String, ChunkPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkPolicy {
    #[serde(default = "default_chunk_target_tokens")]
    pub target_tokens: usize,
    #[serde(default = "default_chunk_soft_max_tokens")]
    pub soft_max_tokens: usize,
    #[serde(default = "default_chunk_hard_max_tokens")]
    pub hard_max_tokens: usize,
    #[serde(default = "default_chunk_boundary_overlap_tokens")]
    pub boundary_overlap_tokens: usize,
    #[serde(default = "default_chunk_neighbor_window")]
    pub neighbor_window: usize,
    #[serde(default = "default_chunk_contextual_prefix")]
    pub contextual_prefix: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            defaults: ChunkPolicy::default(),
            profiles: default_chunk_profiles(),
        }
    }
}

impl Default for ChunkPolicy {
    fn default() -> Self {
        Self {
            target_tokens: default_chunk_target_tokens(),
            soft_max_tokens: default_chunk_soft_max_tokens(),
            hard_max_tokens: default_chunk_hard_max_tokens(),
            boundary_overlap_tokens: default_chunk_boundary_overlap_tokens(),
            neighbor_window: default_chunk_neighbor_window(),
            contextual_prefix: default_chunk_contextual_prefix(),
        }
    }
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
    validate_chunking(&config.chunking)?;
    validate_embeddings(config.embeddings.as_ref())?;
    validate_inference(&config.inference)?;

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
                return Err(KboltError::Config(format!(
                    "config file override must be named {CONFIG_FILENAME}: {}",
                    path.display()
                ))
                .into());
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
            models: ModelConfig::default(),
            embeddings: None,
            inference: InferenceConfig::default(),
            reaping: ReapingConfig {
                days: DEFAULT_REAP_DAYS,
            },
            chunking: ChunkingConfig::default(),
        };
        save(&default_config)?;
    }

    let raw = fs::read_to_string(config_file)?;
    let file_config: FileConfig = toml::from_str(&raw).map_err(|err| {
        KboltError::Config(format!(
            "invalid config file {}: {err}",
            config_file.display()
        ))
    })?;
    validate_chunking(&file_config.chunking)?;
    validate_embeddings(file_config.embeddings.as_ref())?;
    validate_inference(&file_config.inference)?;

    Ok(Config {
        config_dir: config_dir.to_path_buf(),
        cache_dir: cache_dir.to_path_buf(),
        default_space: file_config.default_space,
        models: file_config.models,
        embeddings: file_config.embeddings,
        inference: file_config.inference,
        reaping: ReapingConfig {
            days: file_config.reaping.days,
        },
        chunking: file_config.chunking,
    })
}

fn validate_chunking(chunking: &ChunkingConfig) -> Result<()> {
    validate_chunk_policy("chunking.defaults", &chunking.defaults)?;
    for (profile, policy) in &chunking.profiles {
        validate_chunk_policy(format!("chunking.profiles.{profile}").as_str(), policy)?;
    }
    Ok(())
}

fn validate_embeddings(embeddings: Option<&EmbeddingConfig>) -> Result<()> {
    let Some(embeddings) = embeddings else {
        return Ok(());
    };

    if embeddings.model.trim().is_empty() {
        return Err(KboltError::Config("embeddings.model must not be empty".to_string()).into());
    }

    if embeddings.base_url.trim().is_empty() {
        return Err(KboltError::Config("embeddings.base_url must not be empty".to_string()).into());
    }

    if !embeddings.base_url.starts_with("http://") && !embeddings.base_url.starts_with("https://") {
        return Err(KboltError::Config(
            "embeddings.base_url must start with http:// or https://".to_string(),
        )
        .into());
    }

    if embeddings.timeout_ms == 0 {
        return Err(KboltError::Config(
            "embeddings.timeout_ms must be greater than zero".to_string(),
        )
        .into());
    }

    if embeddings.batch_size == 0 {
        return Err(KboltError::Config(
            "embeddings.batch_size must be greater than zero".to_string(),
        )
        .into());
    }

    if let Some(api_key_env) = embeddings.api_key_env.as_deref() {
        if api_key_env.trim().is_empty() {
            return Err(KboltError::Config(
                "embeddings.api_key_env must not be empty when set".to_string(),
            )
            .into());
        }
    }

    Ok(())
}

fn validate_inference(inference: &InferenceConfig) -> Result<()> {
    validate_text_inference("inference.reranker", inference.reranker.as_ref())?;
    validate_text_inference("inference.expander", inference.expander.as_ref())?;
    Ok(())
}

fn validate_text_inference(scope: &str, config: Option<&TextInferenceConfig>) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };

    if config.model.trim().is_empty() {
        return Err(KboltError::Config(format!("{scope}.model must not be empty")).into());
    }

    if config.base_url.trim().is_empty() {
        return Err(KboltError::Config(format!("{scope}.base_url must not be empty")).into());
    }

    if !config.base_url.starts_with("http://") && !config.base_url.starts_with("https://") {
        return Err(KboltError::Config(format!(
            "{scope}.base_url must start with http:// or https://"
        ))
        .into());
    }

    if config.timeout_ms == 0 {
        return Err(
            KboltError::Config(format!("{scope}.timeout_ms must be greater than zero")).into(),
        );
    }

    if let Some(api_key_env) = config.api_key_env.as_deref() {
        if api_key_env.trim().is_empty() {
            return Err(KboltError::Config(format!(
                "{scope}.api_key_env must not be empty when set"
            ))
            .into());
        }
    }

    Ok(())
}

fn validate_chunk_policy(scope: &str, policy: &ChunkPolicy) -> Result<()> {
    if policy.target_tokens == 0 || policy.soft_max_tokens == 0 || policy.hard_max_tokens == 0 {
        return Err(KboltError::Config(format!(
            "{scope} token caps must be greater than zero (target={}, soft_max={}, hard_max={})",
            policy.target_tokens, policy.soft_max_tokens, policy.hard_max_tokens
        ))
        .into());
    }

    if policy.target_tokens > policy.soft_max_tokens {
        return Err(KboltError::Config(format!(
            "{scope} is invalid: target_tokens ({}) cannot exceed soft_max_tokens ({})",
            policy.target_tokens, policy.soft_max_tokens
        ))
        .into());
    }

    if policy.soft_max_tokens > policy.hard_max_tokens {
        return Err(KboltError::Config(format!(
            "{scope} is invalid: soft_max_tokens ({}) cannot exceed hard_max_tokens ({})",
            policy.soft_max_tokens, policy.hard_max_tokens
        ))
        .into());
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct FileConfig {
    #[serde(default)]
    default_space: Option<String>,
    #[serde(default)]
    models: ModelConfig,
    #[serde(default)]
    embeddings: Option<EmbeddingConfig>,
    #[serde(default)]
    inference: InferenceConfig,
    #[serde(default)]
    reaping: FileReapingConfig,
    #[serde(default)]
    chunking: ChunkingConfig,
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
            models: value.models.clone(),
            embeddings: value.embeddings.clone(),
            inference: value.inference.clone(),
            reaping: FileReapingConfig {
                days: value.reaping.days,
            },
            chunking: value.chunking.clone(),
        }
    }
}

fn default_embedder_source() -> ModelSourceConfig {
    ModelSourceConfig {
        provider: ModelProvider::HuggingFace,
        id: DEFAULT_EMBED_MODEL.to_string(),
        revision: None,
    }
}

fn default_reranker_source() -> ModelSourceConfig {
    ModelSourceConfig {
        provider: ModelProvider::HuggingFace,
        id: DEFAULT_RERANKER_MODEL.to_string(),
        revision: None,
    }
}

fn default_expander_source() -> ModelSourceConfig {
    ModelSourceConfig {
        provider: ModelProvider::HuggingFace,
        id: DEFAULT_EXPANDER_MODEL.to_string(),
        revision: None,
    }
}

fn default_reap_days() -> u32 {
    DEFAULT_REAP_DAYS
}

fn default_chunk_target_tokens() -> usize {
    DEFAULT_CHUNK_TARGET_TOKENS
}

fn default_chunk_soft_max_tokens() -> usize {
    DEFAULT_CHUNK_SOFT_MAX_TOKENS
}

fn default_chunk_hard_max_tokens() -> usize {
    DEFAULT_CHUNK_HARD_MAX_TOKENS
}

fn default_chunk_boundary_overlap_tokens() -> usize {
    DEFAULT_CHUNK_BOUNDARY_OVERLAP_TOKENS
}

fn default_chunk_neighbor_window() -> usize {
    DEFAULT_CHUNK_NEIGHBOR_WINDOW
}

fn default_chunk_contextual_prefix() -> bool {
    DEFAULT_CHUNK_CONTEXTUAL_PREFIX
}

fn default_chunk_profiles() -> HashMap<String, ChunkPolicy> {
    HashMap::from([(
        "code".to_string(),
        ChunkPolicy {
            target_tokens: DEFAULT_CODE_CHUNK_TARGET_TOKENS,
            soft_max_tokens: DEFAULT_CODE_CHUNK_SOFT_MAX_TOKENS,
            hard_max_tokens: DEFAULT_CODE_CHUNK_HARD_MAX_TOKENS,
            boundary_overlap_tokens: DEFAULT_CODE_CHUNK_BOUNDARY_OVERLAP_TOKENS,
            neighbor_window: default_chunk_neighbor_window(),
            contextual_prefix: default_chunk_contextual_prefix(),
        },
    )])
}

fn default_embedding_timeout_ms() -> u64 {
    DEFAULT_EMBEDDING_TIMEOUT_MS
}

fn default_embedding_batch_size() -> usize {
    DEFAULT_EMBEDDING_BATCH_SIZE
}

fn default_embedding_max_retries() -> u32 {
    DEFAULT_EMBEDDING_MAX_RETRIES
}

fn default_inference_timeout_ms() -> u64 {
    DEFAULT_INFERENCE_TIMEOUT_MS
}

fn default_inference_max_retries() -> u32 {
    DEFAULT_INFERENCE_MAX_RETRIES
}

#[cfg(test)]
mod tests;
