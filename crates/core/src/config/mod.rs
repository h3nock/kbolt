use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::Result;
use kbolt_types::KboltError;
use serde::{Deserialize, Serialize};

const APP_NAME: &str = "kbolt";
const CONFIG_FILENAME: &str = "index.toml";
const DEFAULT_EMBED_MODEL: &str = "ggml-org/embeddinggemma-300M-GGUF";
const DEFAULT_RERANKER_MODEL: &str = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF";
const DEFAULT_EXPANDER_MODEL: &str = "tobil/qmd-query-expansion-1.7B-gguf";
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
const DEFAULT_LOCAL_EMBEDDING_MAX_LENGTH: usize = 512;
const DEFAULT_LOCAL_GGUF_EMBEDDING_BATCH_SIZE: usize = 8;
const DEFAULT_INFERENCE_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_INFERENCE_MAX_RETRIES: u32 = 2;
const DEFAULT_LOCAL_INFERENCE_MAX_TOKENS: usize = 256;
const DEFAULT_LOCAL_EXPANDER_MAX_TOKENS: usize = 600;
const DEFAULT_LOCAL_INFERENCE_N_CTX: u32 = 2048;
const DEFAULT_RANKING_DEEP_VARIANT_RRF_K: usize = 60;
const DEFAULT_RANKING_DEEP_VARIANTS_MAX: usize = 4;
const DEFAULT_RANKING_INITIAL_CANDIDATE_LIMIT_MIN: usize = 40;
const DEFAULT_RANKING_RERANK_CANDIDATES_MIN: usize = 20;
const DEFAULT_RANKING_RERANK_CANDIDATES_MAX: usize = 30;
const DEFAULT_RANKING_HYBRID_LINEAR_DENSE_WEIGHT: f32 = 0.7;
const DEFAULT_RANKING_HYBRID_LINEAR_BM25_WEIGHT: f32 = 0.3;
const DEFAULT_RANKING_HYBRID_DBSF_DENSE_WEIGHT: f32 = 1.0;
const DEFAULT_RANKING_HYBRID_DBSF_BM25_WEIGHT: f32 = 1.0;
const DEFAULT_RANKING_HYBRID_DBSF_STDDEVS: f32 = 3.0;
const DEFAULT_RANKING_HYBRID_RRF_K: usize = 60;
const DEFAULT_RANKING_BM25_TITLE_BOOST: f32 = 2.0;
const DEFAULT_RANKING_BM25_HEADING_BOOST: f32 = 1.5;
const DEFAULT_RANKING_BM25_BODY_BOOST: f32 = 1.0;
const DEFAULT_RANKING_BM25_FILEPATH_BOOST: f32 = 0.5;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub config_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub default_space: Option<String>,
    pub models: ModelConfig,
    pub embeddings: Option<EmbeddingConfig>,
    pub inference: InferenceConfig,
    pub reaping: ReapingConfig,
    pub chunking: ChunkingConfig,
    pub ranking: RankingConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum EmbeddingConfig {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible {
        model: String,
        base_url: String,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default = "default_embedding_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_embedding_batch_size")]
        batch_size: usize,
        #[serde(default = "default_embedding_max_retries")]
        max_retries: u32,
    },
    Voyage {
        model: String,
        base_url: String,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default = "default_embedding_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_embedding_batch_size")]
        batch_size: usize,
        #[serde(default = "default_embedding_max_retries")]
        max_retries: u32,
    },
    LocalOnnx {
        #[serde(default)]
        onnx_file: Option<String>,
        #[serde(default)]
        tokenizer_file: Option<String>,
        #[serde(default = "default_local_embedding_max_length")]
        max_length: usize,
    },
    LocalGguf {
        #[serde(default)]
        model_file: Option<String>,
        #[serde(default = "default_local_gguf_embedding_batch_size")]
        batch_size: usize,
        #[serde(default)]
        n_threads: Option<u32>,
        #[serde(default)]
        n_threads_batch: Option<u32>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InferenceConfig {
    #[serde(default)]
    pub reranker: Option<TextInferenceConfig>,
    #[serde(default)]
    pub expander: Option<ExpanderInferenceConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextInferenceConfig {
    #[serde(flatten)]
    pub provider: TextInferenceProvider,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpanderInferenceConfig {
    pub adapter: ExpanderAdapter,
    #[serde(flatten)]
    pub provider: TextInferenceProvider,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpanderAdapter {
    JsonVariants,
    Qmd,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum TextInferenceProvider {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible {
        output_mode: TextInferenceOutputMode,
        model: String,
        base_url: String,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default = "default_inference_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_inference_max_retries")]
        max_retries: u32,
    },
    LocalLlama {
        #[serde(default)]
        model_file: Option<String>,
        #[serde(default = "default_local_inference_max_tokens")]
        max_tokens: usize,
        #[serde(default = "default_local_inference_n_ctx")]
        n_ctx: u32,
        #[serde(default = "default_local_inference_n_gpu_layers")]
        n_gpu_layers: Option<u32>,
    },
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelProvider {
    #[serde(rename = "huggingface")]
    #[default]
    HuggingFace,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingConfig {
    #[serde(default = "default_ranking_deep_variant_rrf_k")]
    pub deep_variant_rrf_k: usize,
    #[serde(default = "default_ranking_deep_variants_max")]
    pub deep_variants_max: usize,
    #[serde(default = "default_ranking_initial_candidate_limit_min")]
    pub initial_candidate_limit_min: usize,
    #[serde(default = "default_ranking_rerank_candidates_min")]
    pub rerank_candidates_min: usize,
    #[serde(default = "default_ranking_rerank_candidates_max")]
    pub rerank_candidates_max: usize,
    #[serde(default)]
    pub hybrid_fusion: HybridFusionConfig,
    #[serde(default)]
    pub bm25_boosts: Bm25BoostsConfig,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            deep_variant_rrf_k: default_ranking_deep_variant_rrf_k(),
            deep_variants_max: default_ranking_deep_variants_max(),
            initial_candidate_limit_min: default_ranking_initial_candidate_limit_min(),
            rerank_candidates_min: default_ranking_rerank_candidates_min(),
            rerank_candidates_max: default_ranking_rerank_candidates_max(),
            hybrid_fusion: HybridFusionConfig::default(),
            bm25_boosts: Bm25BoostsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum HybridFusionMode {
    Rrf,
    #[default]
    Linear,
    Dbsf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridFusionConfig {
    #[serde(default)]
    pub mode: HybridFusionMode,
    #[serde(default)]
    pub linear: LinearHybridFusionConfig,
    #[serde(default)]
    pub dbsf: DbsfHybridFusionConfig,
    #[serde(default)]
    pub rrf: RrfHybridFusionConfig,
}

impl Default for HybridFusionConfig {
    fn default() -> Self {
        Self {
            mode: HybridFusionMode::default(),
            linear: LinearHybridFusionConfig::default(),
            dbsf: DbsfHybridFusionConfig::default(),
            rrf: RrfHybridFusionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearHybridFusionConfig {
    #[serde(default = "default_ranking_hybrid_linear_dense_weight")]
    pub dense_weight: f32,
    #[serde(default = "default_ranking_hybrid_linear_bm25_weight")]
    pub bm25_weight: f32,
}

impl Default for LinearHybridFusionConfig {
    fn default() -> Self {
        Self {
            dense_weight: default_ranking_hybrid_linear_dense_weight(),
            bm25_weight: default_ranking_hybrid_linear_bm25_weight(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DbsfHybridFusionConfig {
    #[serde(default = "default_ranking_hybrid_dbsf_dense_weight")]
    pub dense_weight: f32,
    #[serde(default = "default_ranking_hybrid_dbsf_bm25_weight")]
    pub bm25_weight: f32,
    #[serde(default = "default_ranking_hybrid_dbsf_stddevs")]
    pub stddevs: f32,
}

impl Default for DbsfHybridFusionConfig {
    fn default() -> Self {
        Self {
            dense_weight: default_ranking_hybrid_dbsf_dense_weight(),
            bm25_weight: default_ranking_hybrid_dbsf_bm25_weight(),
            stddevs: default_ranking_hybrid_dbsf_stddevs(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RrfHybridFusionConfig {
    #[serde(default = "default_ranking_hybrid_rrf_k")]
    pub k: usize,
}

impl Default for RrfHybridFusionConfig {
    fn default() -> Self {
        Self {
            k: default_ranking_hybrid_rrf_k(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bm25BoostsConfig {
    #[serde(default = "default_ranking_bm25_title_boost")]
    pub title: f32,
    #[serde(default = "default_ranking_bm25_heading_boost")]
    pub heading: f32,
    #[serde(default = "default_ranking_bm25_body_boost")]
    pub body: f32,
    #[serde(default = "default_ranking_bm25_filepath_boost")]
    pub filepath: f32,
}

impl Default for Bm25BoostsConfig {
    fn default() -> Self {
        Self {
            title: default_ranking_bm25_title_boost(),
            heading: default_ranking_bm25_heading_boost(),
            body: default_ranking_bm25_body_boost(),
            filepath: default_ranking_bm25_filepath_boost(),
        }
    }
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

pub fn default_config_file_path() -> Result<PathBuf> {
    Ok(default_config_dir()?.join(CONFIG_FILENAME))
}

pub fn save(config: &Config) -> Result<()> {
    fs::create_dir_all(&config.config_dir)?;
    fs::create_dir_all(&config.cache_dir)?;
    validate_chunking(&config.chunking)?;
    validate_embeddings(config.embeddings.as_ref())?;
    validate_inference(&config.inference)?;
    validate_ranking(&config.ranking)?;

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
            ranking: RankingConfig::default(),
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
    let inference = InferenceConfig::from(file_config.inference.clone());
    validate_inference(&inference)?;
    validate_ranking(&file_config.ranking)?;

    Ok(Config {
        config_dir: config_dir.to_path_buf(),
        cache_dir: cache_dir.to_path_buf(),
        default_space: file_config.default_space,
        models: file_config.models,
        embeddings: file_config.embeddings,
        inference,
        reaping: ReapingConfig {
            days: file_config.reaping.days,
        },
        chunking: file_config.chunking,
        ranking: file_config.ranking,
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

    match embeddings {
        EmbeddingConfig::OpenAiCompatible {
            model,
            base_url,
            api_key_env,
            timeout_ms,
            batch_size,
            ..
        }
        | EmbeddingConfig::Voyage {
            model,
            base_url,
            api_key_env,
            timeout_ms,
            batch_size,
            ..
        } => {
            if model.trim().is_empty() {
                return Err(
                    KboltError::Config("embeddings.model must not be empty".to_string()).into(),
                );
            }

            if base_url.trim().is_empty() {
                return Err(KboltError::Config(
                    "embeddings.base_url must not be empty".to_string(),
                )
                .into());
            }

            if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
                return Err(KboltError::Config(
                    "embeddings.base_url must start with http:// or https://".to_string(),
                )
                .into());
            }

            if *timeout_ms == 0 {
                return Err(KboltError::Config(
                    "embeddings.timeout_ms must be greater than zero".to_string(),
                )
                .into());
            }

            if *batch_size == 0 {
                return Err(KboltError::Config(
                    "embeddings.batch_size must be greater than zero".to_string(),
                )
                .into());
            }

            if let Some(api_key_env) = api_key_env.as_deref() {
                if api_key_env.trim().is_empty() {
                    return Err(KboltError::Config(
                        "embeddings.api_key_env must not be empty when set".to_string(),
                    )
                    .into());
                }
            }

            Ok(())
        }
        EmbeddingConfig::LocalOnnx {
            onnx_file,
            tokenizer_file,
            max_length,
        } => {
            if *max_length == 0 {
                return Err(KboltError::Config(
                    "embeddings.max_length must be greater than zero".to_string(),
                )
                .into());
            }

            if let Some(onnx_file) = onnx_file {
                if onnx_file.trim().is_empty() {
                    return Err(KboltError::Config(
                        "embeddings.onnx_file must not be empty when set".to_string(),
                    )
                    .into());
                }
            }

            if let Some(tokenizer_file) = tokenizer_file {
                if tokenizer_file.trim().is_empty() {
                    return Err(KboltError::Config(
                        "embeddings.tokenizer_file must not be empty when set".to_string(),
                    )
                    .into());
                }
            }

            Ok(())
        }
        EmbeddingConfig::LocalGguf {
            model_file,
            batch_size,
            n_threads,
            n_threads_batch,
        } => {
            if *batch_size == 0 {
                return Err(KboltError::Config(
                    "embeddings.batch_size must be greater than zero".to_string(),
                )
                .into());
            }

            if let Some(model_file) = model_file {
                if model_file.trim().is_empty() {
                    return Err(KboltError::Config(
                        "embeddings.model_file must not be empty when set".to_string(),
                    )
                    .into());
                }
            }

            if matches!(n_threads, Some(0)) {
                return Err(KboltError::Config(
                    "embeddings.n_threads must be greater than zero when set".to_string(),
                )
                .into());
            }

            if matches!(n_threads_batch, Some(0)) {
                return Err(KboltError::Config(
                    "embeddings.n_threads_batch must be greater than zero when set".to_string(),
                )
                .into());
            }

            Ok(())
        }
    }
}

fn validate_inference(inference: &InferenceConfig) -> Result<()> {
    validate_text_inference_config("inference.reranker", inference.reranker.as_ref())?;
    validate_expander_inference("inference.expander", inference.expander.as_ref())?;
    Ok(())
}

fn validate_text_inference_config(scope: &str, config: Option<&TextInferenceConfig>) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };

    validate_text_inference_provider(scope, &config.provider)
}

fn validate_expander_inference(
    scope: &str,
    config: Option<&ExpanderInferenceConfig>,
) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };

    validate_text_inference_provider(scope, &config.provider)?;

    if matches!(config.adapter, ExpanderAdapter::Qmd)
        && !matches!(config.provider, TextInferenceProvider::LocalLlama { .. })
    {
        return Err(KboltError::Config(format!(
            "{scope}.adapter=qmd requires provider=local_llama"
        ))
        .into());
    }

    Ok(())
}

fn validate_text_inference_provider(scope: &str, provider: &TextInferenceProvider) -> Result<()> {
    match provider {
        TextInferenceProvider::OpenAiCompatible {
            model,
            base_url,
            api_key_env,
            timeout_ms,
            ..
        } => {
            if model.trim().is_empty() {
                return Err(KboltError::Config(format!("{scope}.model must not be empty")).into());
            }

            if base_url.trim().is_empty() {
                return Err(
                    KboltError::Config(format!("{scope}.base_url must not be empty")).into(),
                );
            }

            if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
                return Err(KboltError::Config(format!(
                    "{scope}.base_url must start with http:// or https://"
                ))
                .into());
            }

            if *timeout_ms == 0 {
                return Err(KboltError::Config(format!(
                    "{scope}.timeout_ms must be greater than zero"
                ))
                .into());
            }

            if let Some(api_key_env) = api_key_env.as_deref() {
                if api_key_env.trim().is_empty() {
                    return Err(KboltError::Config(format!(
                        "{scope}.api_key_env must not be empty when set"
                    ))
                    .into());
                }
            }

            Ok(())
        }
        TextInferenceProvider::LocalLlama {
            model_file,
            max_tokens,
            n_ctx,
            ..
        } => {
            if *max_tokens == 0 {
                return Err(KboltError::Config(format!(
                    "{scope}.max_tokens must be greater than zero"
                ))
                .into());
            }

            if *n_ctx == 0 {
                return Err(
                    KboltError::Config(format!("{scope}.n_ctx must be greater than zero")).into(),
                );
            }

            if let Some(model_file) = model_file {
                if model_file.trim().is_empty() {
                    return Err(KboltError::Config(format!(
                        "{scope}.model_file must not be empty when set"
                    ))
                    .into());
                }
            }

            Ok(())
        }
    }
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

fn validate_ranking(ranking: &RankingConfig) -> Result<()> {
    if ranking.deep_variant_rrf_k == 0 {
        return Err(KboltError::Config(
            "ranking.deep_variant_rrf_k must be greater than zero".to_string(),
        )
        .into());
    }

    if ranking.deep_variants_max == 0 {
        return Err(KboltError::Config(
            "ranking.deep_variants_max must be greater than zero".to_string(),
        )
        .into());
    }

    if ranking.initial_candidate_limit_min == 0 {
        return Err(KboltError::Config(
            "ranking.initial_candidate_limit_min must be greater than zero".to_string(),
        )
        .into());
    }

    if ranking.rerank_candidates_min == 0 {
        return Err(KboltError::Config(
            "ranking.rerank_candidates_min must be greater than zero".to_string(),
        )
        .into());
    }

    if ranking.rerank_candidates_max < ranking.rerank_candidates_min {
        return Err(KboltError::Config(format!(
            "ranking.rerank_candidates_max ({}) must be greater than or equal to ranking.rerank_candidates_min ({})",
            ranking.rerank_candidates_max, ranking.rerank_candidates_min
        ))
        .into());
    }

    validate_hybrid_fusion_weights(
        "ranking.hybrid_fusion.linear",
        ranking.hybrid_fusion.linear.dense_weight,
        ranking.hybrid_fusion.linear.bm25_weight,
    )?;
    validate_hybrid_fusion_weights(
        "ranking.hybrid_fusion.dbsf",
        ranking.hybrid_fusion.dbsf.dense_weight,
        ranking.hybrid_fusion.dbsf.bm25_weight,
    )?;
    if !ranking.hybrid_fusion.dbsf.stddevs.is_finite() || ranking.hybrid_fusion.dbsf.stddevs <= 0.0
    {
        return Err(KboltError::Config(
            "ranking.hybrid_fusion.dbsf.stddevs must be finite and greater than zero".to_string(),
        )
        .into());
    }
    if ranking.hybrid_fusion.rrf.k == 0 {
        return Err(KboltError::Config(
            "ranking.hybrid_fusion.rrf.k must be greater than zero".to_string(),
        )
        .into());
    }

    validate_positive_finite_boost("ranking.bm25_boosts.title", ranking.bm25_boosts.title)?;
    validate_positive_finite_boost("ranking.bm25_boosts.heading", ranking.bm25_boosts.heading)?;
    validate_positive_finite_boost("ranking.bm25_boosts.body", ranking.bm25_boosts.body)?;
    validate_positive_finite_boost("ranking.bm25_boosts.filepath", ranking.bm25_boosts.filepath)?;

    Ok(())
}

fn validate_positive_finite_boost(scope: &str, value: f32) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(
            KboltError::Config(format!("{scope} must be finite and greater than zero")).into(),
        );
    }

    Ok(())
}

fn validate_nonnegative_finite_weight(scope: &str, value: f32) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(KboltError::Config(format!(
            "{scope} must be finite and greater than or equal to zero"
        ))
        .into());
    }

    Ok(())
}

fn validate_hybrid_fusion_weights(scope: &str, dense_weight: f32, bm25_weight: f32) -> Result<()> {
    validate_nonnegative_finite_weight(format!("{scope}.dense_weight").as_str(), dense_weight)?;
    validate_nonnegative_finite_weight(format!("{scope}.bm25_weight").as_str(), bm25_weight)?;
    if dense_weight + bm25_weight <= 0.0 {
        return Err(
            KboltError::Config(format!("{scope} weights must sum to greater than zero")).into(),
        );
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
struct FileConfig {
    #[serde(default)]
    default_space: Option<String>,
    #[serde(default)]
    models: ModelConfig,
    #[serde(default)]
    embeddings: Option<EmbeddingConfig>,
    #[serde(default)]
    inference: FileInferenceConfig,
    #[serde(default)]
    reaping: FileReapingConfig,
    #[serde(default)]
    chunking: ChunkingConfig,
    #[serde(default)]
    ranking: RankingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct FileInferenceConfig {
    #[serde(default)]
    reranker: Option<FileTextInferenceConfig>,
    #[serde(default)]
    expander: Option<FileExpanderInferenceConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FileTextInferenceConfig {
    #[serde(flatten)]
    provider: FileTextInferenceProvider,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FileExpanderInferenceConfig {
    #[serde(default)]
    adapter: Option<ExpanderAdapter>,
    #[serde(flatten)]
    provider: FileTextInferenceProvider,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "provider", rename_all = "snake_case")]
enum FileTextInferenceProvider {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible {
        output_mode: TextInferenceOutputMode,
        model: String,
        base_url: String,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default = "default_inference_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_inference_max_retries")]
        max_retries: u32,
    },
    LocalLlama {
        #[serde(default)]
        model_file: Option<String>,
        #[serde(default)]
        max_tokens: Option<usize>,
        #[serde(default = "default_local_inference_n_ctx")]
        n_ctx: u32,
        #[serde(default = "default_local_inference_n_gpu_layers")]
        n_gpu_layers: Option<u32>,
    },
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
            inference: FileInferenceConfig::from(&value.inference),
            reaping: FileReapingConfig {
                days: value.reaping.days,
            },
            chunking: value.chunking.clone(),
            ranking: value.ranking.clone(),
        }
    }
}

impl From<FileInferenceConfig> for InferenceConfig {
    fn from(value: FileInferenceConfig) -> Self {
        Self {
            reranker: file_text_inference_to_runtime(
                value.reranker,
                default_local_inference_max_tokens(),
            ),
            expander: file_expander_inference_to_runtime(
                value.expander,
                default_local_expander_max_tokens(),
            ),
        }
    }
}

impl From<&InferenceConfig> for FileInferenceConfig {
    fn from(value: &InferenceConfig) -> Self {
        Self {
            reranker: value.reranker.as_ref().map(FileTextInferenceConfig::from),
            expander: value
                .expander
                .as_ref()
                .map(FileExpanderInferenceConfig::from),
        }
    }
}

impl From<&TextInferenceConfig> for FileTextInferenceConfig {
    fn from(value: &TextInferenceConfig) -> Self {
        let provider = match &value.provider {
            TextInferenceProvider::OpenAiCompatible {
                output_mode,
                model,
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
            } => FileTextInferenceProvider::OpenAiCompatible {
                output_mode: output_mode.clone(),
                model: model.clone(),
                base_url: base_url.clone(),
                api_key_env: api_key_env.clone(),
                timeout_ms: *timeout_ms,
                max_retries: *max_retries,
            },
            TextInferenceProvider::LocalLlama {
                model_file,
                max_tokens,
                n_ctx,
                n_gpu_layers,
            } => FileTextInferenceProvider::LocalLlama {
                model_file: model_file.clone(),
                max_tokens: Some(*max_tokens),
                n_ctx: *n_ctx,
                n_gpu_layers: *n_gpu_layers,
            },
        };

        Self { provider }
    }
}

impl From<&ExpanderInferenceConfig> for FileExpanderInferenceConfig {
    fn from(value: &ExpanderInferenceConfig) -> Self {
        let provider = FileTextInferenceConfig::from(&TextInferenceConfig {
            provider: value.provider.clone(),
        })
        .provider;

        Self {
            adapter: Some(value.adapter.clone()),
            provider,
        }
    }
}

fn file_text_inference_to_runtime(
    config: Option<FileTextInferenceConfig>,
    default_max_tokens: usize,
) -> Option<TextInferenceConfig> {
    config.map(|config| TextInferenceConfig {
        provider: file_text_inference_provider_to_runtime(config.provider, default_max_tokens),
    })
}

fn file_expander_inference_to_runtime(
    config: Option<FileExpanderInferenceConfig>,
    default_max_tokens: usize,
) -> Option<ExpanderInferenceConfig> {
    config.map(|config| {
        let adapter = config.adapter.unwrap_or_else(|| match &config.provider {
            FileTextInferenceProvider::OpenAiCompatible { .. } => ExpanderAdapter::JsonVariants,
            FileTextInferenceProvider::LocalLlama { .. } => ExpanderAdapter::Qmd,
        });

        ExpanderInferenceConfig {
            adapter,
            provider: file_text_inference_provider_to_runtime(config.provider, default_max_tokens),
        }
    })
}

fn file_text_inference_provider_to_runtime(
    provider: FileTextInferenceProvider,
    default_max_tokens: usize,
) -> TextInferenceProvider {
    match provider {
        FileTextInferenceProvider::OpenAiCompatible {
            output_mode,
            model,
            base_url,
            api_key_env,
            timeout_ms,
            max_retries,
        } => TextInferenceProvider::OpenAiCompatible {
            output_mode,
            model,
            base_url,
            api_key_env,
            timeout_ms,
            max_retries,
        },
        FileTextInferenceProvider::LocalLlama {
            model_file,
            max_tokens,
            n_ctx,
            n_gpu_layers,
        } => TextInferenceProvider::LocalLlama {
            model_file,
            max_tokens: max_tokens.unwrap_or(default_max_tokens),
            n_ctx,
            n_gpu_layers,
        },
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

fn default_local_embedding_max_length() -> usize {
    DEFAULT_LOCAL_EMBEDDING_MAX_LENGTH
}

fn default_local_gguf_embedding_batch_size() -> usize {
    DEFAULT_LOCAL_GGUF_EMBEDDING_BATCH_SIZE
}

fn default_inference_timeout_ms() -> u64 {
    DEFAULT_INFERENCE_TIMEOUT_MS
}

fn default_inference_max_retries() -> u32 {
    DEFAULT_INFERENCE_MAX_RETRIES
}

fn default_local_inference_max_tokens() -> usize {
    DEFAULT_LOCAL_INFERENCE_MAX_TOKENS
}

fn default_local_expander_max_tokens() -> usize {
    DEFAULT_LOCAL_EXPANDER_MAX_TOKENS
}

fn default_local_inference_n_ctx() -> u32 {
    DEFAULT_LOCAL_INFERENCE_N_CTX
}

fn default_local_inference_n_gpu_layers() -> Option<u32> {
    None
}

fn default_ranking_deep_variant_rrf_k() -> usize {
    DEFAULT_RANKING_DEEP_VARIANT_RRF_K
}

fn default_ranking_deep_variants_max() -> usize {
    DEFAULT_RANKING_DEEP_VARIANTS_MAX
}

fn default_ranking_initial_candidate_limit_min() -> usize {
    DEFAULT_RANKING_INITIAL_CANDIDATE_LIMIT_MIN
}

fn default_ranking_rerank_candidates_min() -> usize {
    DEFAULT_RANKING_RERANK_CANDIDATES_MIN
}

fn default_ranking_rerank_candidates_max() -> usize {
    DEFAULT_RANKING_RERANK_CANDIDATES_MAX
}

fn default_ranking_hybrid_linear_dense_weight() -> f32 {
    DEFAULT_RANKING_HYBRID_LINEAR_DENSE_WEIGHT
}

fn default_ranking_hybrid_linear_bm25_weight() -> f32 {
    DEFAULT_RANKING_HYBRID_LINEAR_BM25_WEIGHT
}

fn default_ranking_hybrid_dbsf_dense_weight() -> f32 {
    DEFAULT_RANKING_HYBRID_DBSF_DENSE_WEIGHT
}

fn default_ranking_hybrid_dbsf_bm25_weight() -> f32 {
    DEFAULT_RANKING_HYBRID_DBSF_BM25_WEIGHT
}

fn default_ranking_hybrid_dbsf_stddevs() -> f32 {
    DEFAULT_RANKING_HYBRID_DBSF_STDDEVS
}

fn default_ranking_hybrid_rrf_k() -> usize {
    DEFAULT_RANKING_HYBRID_RRF_K
}

fn default_ranking_bm25_title_boost() -> f32 {
    DEFAULT_RANKING_BM25_TITLE_BOOST
}

fn default_ranking_bm25_heading_boost() -> f32 {
    DEFAULT_RANKING_BM25_HEADING_BOOST
}

fn default_ranking_bm25_body_boost() -> f32 {
    DEFAULT_RANKING_BM25_BODY_BOOST
}

fn default_ranking_bm25_filepath_boost() -> f32 {
    DEFAULT_RANKING_BM25_FILEPATH_BOOST
}

#[cfg(test)]
mod tests;
