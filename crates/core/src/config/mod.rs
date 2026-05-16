use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::Result;
use kbolt_types::KboltError;
use serde::{Deserialize, Serialize};

const APP_NAME: &str = "kbolt";
const CONFIG_FILENAME: &str = "index.toml";
const DEFAULT_REAP_DAYS: u32 = 7;
const DEFAULT_CHUNK_TARGET_TOKENS: usize = 800;
const DEFAULT_CHUNK_SOFT_MAX_TOKENS: usize = 950;
const DEFAULT_CHUNK_HARD_MAX_TOKENS: usize = 1200;
const DEFAULT_CHUNK_BOUNDARY_OVERLAP_TOKENS: usize = 48;
const DEFAULT_CHUNK_NEIGHBOR_WINDOW: usize = 1;
const DEFAULT_CHUNK_CONTEXTUAL_PREFIX: bool = true;
const DEFAULT_CODE_CHUNK_TARGET_TOKENS: usize = 320;
const DEFAULT_CODE_CHUNK_SOFT_MAX_TOKENS: usize = 420;
const DEFAULT_CODE_CHUNK_HARD_MAX_TOKENS: usize = 560;
const DEFAULT_CODE_CHUNK_BOUNDARY_OVERLAP_TOKENS: usize = 24;
const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 32;
const DEFAULT_INFERENCE_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_INFERENCE_MAX_RETRIES: u32 = 2;
const MAX_PROVIDER_PARALLEL_REQUESTS: usize = 64;
const DEFAULT_LOCAL_EXPANDER_MAX_TOKENS: usize = 600;
const DEFAULT_EXPANDER_SEED: u32 = 0;
const DEFAULT_EXPANDER_TEMPERATURE: f32 = 0.7;
const DEFAULT_EXPANDER_TOP_K: i32 = 20;
const DEFAULT_EXPANDER_TOP_P: f32 = 0.8;
const DEFAULT_EXPANDER_MIN_P: f32 = 0.0;
const DEFAULT_EXPANDER_REPEAT_LAST_N: i32 = 64;
const DEFAULT_EXPANDER_REPEAT_PENALTY: f32 = 1.0;
const DEFAULT_EXPANDER_FREQUENCY_PENALTY: f32 = 0.0;
const DEFAULT_EXPANDER_PRESENCE_PENALTY: f32 = 0.5;
const DEFAULT_RANKING_DEEP_VARIANT_RRF_K: usize = 60;
const DEFAULT_RANKING_DEEP_VARIANTS_MAX: usize = 4;
const DEFAULT_RANKING_INITIAL_CANDIDATE_LIMIT_MIN: usize = 40;
const DEFAULT_RANKING_RERANK_CANDIDATES_MIN: usize = 20;
const DEFAULT_RANKING_RERANK_CANDIDATES_MAX: usize = 30;
const DEFAULT_RANKING_HYBRID_LINEAR_DENSE_WEIGHT: f32 = 0.7;
const DEFAULT_RANKING_HYBRID_LINEAR_BM25_WEIGHT: f32 = 0.3;
const DEFAULT_RANKING_HYBRID_DBSF_DENSE_WEIGHT: f32 = 1.0;
const DEFAULT_RANKING_HYBRID_DBSF_BM25_WEIGHT: f32 = 0.4;
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
    pub providers: HashMap<String, ProviderProfileConfig>,
    pub roles: RoleBindingsConfig,
    pub reaping: ReapingConfig,
    pub chunking: ChunkingConfig,
    pub ranking: RankingConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderOperation {
    Embedding,
    Reranking,
    ChatCompletion,
}

impl ProviderOperation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Reranking => "reranking",
            Self::ChatCompletion => "chat_completion",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderProfileConfig {
    LlamaCppServer {
        operation: ProviderOperation,
        base_url: String,
        model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parallel_requests: Option<usize>,
        #[serde(default = "default_inference_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_inference_max_retries")]
        max_retries: u32,
    },
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible {
        operation: ProviderOperation,
        base_url: String,
        model: String,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default = "default_inference_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_inference_max_retries")]
        max_retries: u32,
    },
}

impl ProviderProfileConfig {
    pub fn operation(&self) -> ProviderOperation {
        match self {
            Self::LlamaCppServer { operation, .. } | Self::OpenAiCompatible { operation, .. } => {
                *operation
            }
        }
    }

    pub fn base_url(&self) -> &str {
        match self {
            Self::LlamaCppServer { base_url, .. } | Self::OpenAiCompatible { base_url, .. } => {
                base_url
            }
        }
    }

    pub fn model(&self) -> &str {
        match self {
            Self::LlamaCppServer { model, .. } | Self::OpenAiCompatible { model, .. } => model,
        }
    }

    pub fn api_key_env(&self) -> Option<&str> {
        match self {
            Self::LlamaCppServer { .. } => None,
            Self::OpenAiCompatible { api_key_env, .. } => api_key_env.as_deref(),
        }
    }

    pub fn timeout_ms(&self) -> u64 {
        match self {
            Self::LlamaCppServer { timeout_ms, .. } | Self::OpenAiCompatible { timeout_ms, .. } => {
                *timeout_ms
            }
        }
    }

    pub fn max_retries(&self) -> u32 {
        match self {
            Self::LlamaCppServer { max_retries, .. }
            | Self::OpenAiCompatible { max_retries, .. } => *max_retries,
        }
    }

    pub fn parallel_requests(&self) -> Option<usize> {
        match self {
            Self::LlamaCppServer {
                parallel_requests, ..
            } => *parallel_requests,
            Self::OpenAiCompatible { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RoleBindingsConfig {
    #[serde(default)]
    pub embedder: Option<EmbedderRoleConfig>,
    #[serde(default)]
    pub reranker: Option<RerankerRoleConfig>,
    #[serde(default)]
    pub expander: Option<ExpanderRoleConfig>,
}

impl RoleBindingsConfig {
    fn is_empty(&self) -> bool {
        self.embedder.is_none() && self.reranker.is_none() && self.expander.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderRoleConfig {
    pub provider: String,
    #[serde(default = "default_embedding_batch_size")]
    pub batch_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RerankerRoleConfig {
    pub provider: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpanderRoleConfig {
    pub provider: String,
    #[serde(default = "default_local_expander_max_tokens")]
    pub max_tokens: usize,
    #[serde(flatten)]
    pub sampling: ExpanderSamplingConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpanderSamplingConfig {
    #[serde(default = "default_expander_seed")]
    pub seed: u32,
    #[serde(default = "default_expander_temperature")]
    pub temperature: f32,
    #[serde(default = "default_expander_top_k")]
    pub top_k: i32,
    #[serde(default = "default_expander_top_p")]
    pub top_p: f32,
    #[serde(default = "default_expander_min_p")]
    pub min_p: f32,
    #[serde(default = "default_expander_repeat_last_n")]
    pub repeat_last_n: i32,
    #[serde(default = "default_expander_repeat_penalty")]
    pub repeat_penalty: f32,
    #[serde(default = "default_expander_frequency_penalty")]
    pub frequency_penalty: f32,
    #[serde(default = "default_expander_presence_penalty")]
    pub presence_penalty: f32,
}

pub type ExpanderRoleSamplingConfig = ExpanderSamplingConfig;
pub type ExpanderLocalLlamaSamplingConfig = ExpanderSamplingConfig;

impl Default for ExpanderSamplingConfig {
    fn default() -> Self {
        Self {
            seed: default_expander_seed(),
            temperature: default_expander_temperature(),
            top_k: default_expander_top_k(),
            top_p: default_expander_top_p(),
            min_p: default_expander_min_p(),
            repeat_last_n: default_expander_repeat_last_n(),
            repeat_penalty: default_expander_repeat_penalty(),
            frequency_penalty: default_expander_frequency_penalty(),
            presence_penalty: default_expander_presence_penalty(),
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
    Linear,
    #[default]
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
    #[serde(default)]
    pub defaults: ChunkPolicy,
    #[serde(default = "default_chunk_profiles")]
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

    load_from_file(
        &config_file,
        &config_dir,
        &cache_dir,
        ConfigLoadMode::CreateDefault,
    )
}

pub fn load_existing(config_path: Option<&Path>) -> Result<Config> {
    let config_dir = resolve_config_dir(config_path)?;
    let cache_dir = default_cache_dir()?;
    let config_file = config_dir.join(CONFIG_FILENAME);

    load_from_file(
        &config_file,
        &config_dir,
        &cache_dir,
        ConfigLoadMode::ExistingOnly,
    )
}

pub fn default_config_file_path() -> Result<PathBuf> {
    resolve_config_file_path(None)
}

pub fn resolve_config_file_path(config_path: Option<&Path>) -> Result<PathBuf> {
    Ok(resolve_config_dir(config_path)?.join(CONFIG_FILENAME))
}

pub fn save(config: &Config) -> Result<()> {
    fs::create_dir_all(&config.config_dir)?;
    fs::create_dir_all(&config.cache_dir)?;
    validate_chunking(&config.chunking)?;
    validate_provider_profiles(&config.providers)?;
    validate_role_bindings(&config.roles, &config.providers)?;
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

pub fn default_config_dir() -> Result<PathBuf> {
    let base = dirs::config_dir()
        .ok_or_else(|| KboltError::Config("unable to determine user config directory".into()))?;
    Ok(base.join(APP_NAME))
}

pub fn default_cache_dir() -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .ok_or_else(|| KboltError::Config("unable to determine user cache directory".into()))?;
    Ok(base.join(APP_NAME))
}

fn load_from_file(
    config_file: &Path,
    config_dir: &Path,
    cache_dir: &Path,
    mode: ConfigLoadMode,
) -> Result<Config> {
    match mode {
        ConfigLoadMode::CreateDefault => {
            fs::create_dir_all(config_dir)?;
            fs::create_dir_all(cache_dir)?;

            if !config_file.exists() {
                let default_config = Config {
                    config_dir: config_dir.to_path_buf(),
                    cache_dir: cache_dir.to_path_buf(),
                    default_space: None,
                    providers: HashMap::new(),
                    roles: RoleBindingsConfig::default(),
                    reaping: ReapingConfig {
                        days: DEFAULT_REAP_DAYS,
                    },
                    chunking: ChunkingConfig::default(),
                    ranking: RankingConfig::default(),
                };
                save(&default_config)?;
            }
        }
        ConfigLoadMode::ExistingOnly => {
            if !config_file.exists() {
                return Err(KboltError::Config(format!(
                    "config file not found: {}",
                    config_file.display()
                ))
                .into());
            }
        }
    }

    let raw = fs::read_to_string(config_file)?;
    let file_config: FileConfig = toml::from_str(&raw).map_err(|err| {
        KboltError::Config(format!(
            "invalid config file {}: {err}",
            config_file.display()
        ))
    })?;
    validate_chunking(&file_config.chunking)?;
    validate_provider_profiles(&file_config.providers)?;
    validate_role_bindings(&file_config.roles, &file_config.providers)?;
    validate_ranking(&file_config.ranking)?;

    Ok(Config {
        config_dir: config_dir.to_path_buf(),
        cache_dir: cache_dir.to_path_buf(),
        default_space: file_config.default_space,
        providers: file_config.providers,
        roles: file_config.roles,
        reaping: ReapingConfig {
            days: file_config.reaping.days,
        },
        chunking: file_config.chunking,
        ranking: file_config.ranking,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigLoadMode {
    CreateDefault,
    ExistingOnly,
}

fn validate_chunking(chunking: &ChunkingConfig) -> Result<()> {
    validate_chunk_policy("chunking.defaults", &chunking.defaults)?;
    for (profile, policy) in &chunking.profiles {
        validate_chunk_policy(format!("chunking.profiles.{profile}").as_str(), policy)?;
    }
    Ok(())
}

fn validate_provider_profiles(providers: &HashMap<String, ProviderProfileConfig>) -> Result<()> {
    for (name, profile) in providers {
        if name.trim().is_empty() {
            return Err(
                KboltError::Config("providers table names must not be empty".to_string()).into(),
            );
        }

        validate_provider_profile(format!("providers.{name}").as_str(), profile)?;
    }

    Ok(())
}

fn validate_provider_profile(scope: &str, profile: &ProviderProfileConfig) -> Result<()> {
    validate_provider_profile_common(
        scope,
        profile.base_url(),
        profile.model(),
        profile.api_key_env(),
        profile.timeout_ms(),
    )?;
    if let Some(parallel_requests) = profile.parallel_requests() {
        validate_provider_parallel_requests(scope, parallel_requests)?;
        if profile.operation() != ProviderOperation::Reranking {
            return Err(KboltError::Config(format!(
                "{scope}.parallel_requests is only supported for reranking providers"
            ))
            .into());
        }
    }
    Ok(())
}

fn validate_provider_parallel_requests(scope: &str, parallel_requests: usize) -> Result<()> {
    if parallel_requests == 0 {
        return Err(KboltError::Config(format!(
            "{scope}.parallel_requests must be greater than zero"
        ))
        .into());
    }
    if parallel_requests > MAX_PROVIDER_PARALLEL_REQUESTS {
        return Err(KboltError::Config(format!(
            "{scope}.parallel_requests ({parallel_requests}) must be less than or equal to {MAX_PROVIDER_PARALLEL_REQUESTS}"
        ))
        .into());
    }
    Ok(())
}

fn validate_provider_profile_common(
    scope: &str,
    base_url: &str,
    model: &str,
    api_key_env: Option<&str>,
    timeout_ms: u64,
) -> Result<()> {
    if model.trim().is_empty() {
        return Err(KboltError::Config(format!("{scope}.model must not be empty")).into());
    }

    if base_url.trim().is_empty() {
        return Err(KboltError::Config(format!("{scope}.base_url must not be empty")).into());
    }

    if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
        return Err(KboltError::Config(format!(
            "{scope}.base_url must start with http:// or https://"
        ))
        .into());
    }

    if timeout_ms == 0 {
        return Err(
            KboltError::Config(format!("{scope}.timeout_ms must be greater than zero")).into(),
        );
    }

    if let Some(api_key_env) = api_key_env {
        if api_key_env.trim().is_empty() {
            return Err(KboltError::Config(format!(
                "{scope}.api_key_env must not be empty when set"
            ))
            .into());
        }
    }

    Ok(())
}

fn validate_role_bindings(
    roles: &RoleBindingsConfig,
    providers: &HashMap<String, ProviderProfileConfig>,
) -> Result<()> {
    if let Some(embedder) = roles.embedder.as_ref() {
        validate_role_provider_reference(
            "roles.embedder",
            &embedder.provider,
            &[ProviderOperation::Embedding],
            providers,
        )?;
        if embedder.batch_size == 0 {
            return Err(KboltError::Config(
                "roles.embedder.batch_size must be greater than zero".to_string(),
            )
            .into());
        }
    }

    if let Some(reranker) = roles.reranker.as_ref() {
        validate_role_provider_reference(
            "roles.reranker",
            &reranker.provider,
            &[
                ProviderOperation::Reranking,
                ProviderOperation::ChatCompletion,
            ],
            providers,
        )?;
    }

    if let Some(expander) = roles.expander.as_ref() {
        validate_role_provider_reference(
            "roles.expander",
            &expander.provider,
            &[ProviderOperation::ChatCompletion],
            providers,
        )?;
        if expander.max_tokens == 0 {
            return Err(KboltError::Config(
                "roles.expander.max_tokens must be greater than zero".to_string(),
            )
            .into());
        }
        validate_expander_sampling(
            "roles.expander",
            expander.sampling.temperature,
            expander.sampling.top_k,
            expander.sampling.top_p,
            expander.sampling.min_p,
            expander.sampling.repeat_last_n,
            expander.sampling.repeat_penalty,
            expander.sampling.frequency_penalty,
            expander.sampling.presence_penalty,
        )?;
    }

    Ok(())
}

fn validate_role_provider_reference(
    scope: &str,
    provider_name: &str,
    allowed_operations: &[ProviderOperation],
    providers: &HashMap<String, ProviderProfileConfig>,
) -> Result<()> {
    if provider_name.trim().is_empty() {
        return Err(KboltError::Config(format!("{scope}.provider must not be empty")).into());
    }

    let Some(profile) = providers.get(provider_name) else {
        return Err(KboltError::Config(format!(
            "{scope}.provider references undefined provider profile '{provider_name}'"
        ))
        .into());
    };

    let operation = profile.operation();

    if !allowed_operations.contains(&operation) {
        return Err(KboltError::Config(format!(
            "{scope}.provider '{provider_name}' uses incompatible operation '{}'",
            operation.as_str()
        ))
        .into());
    }

    Ok(())
}

fn validate_expander_sampling(
    scope: &str,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repeat_last_n: i32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
) -> Result<()> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(KboltError::Config(format!(
            "{scope}.temperature must be finite and greater than zero"
        ))
        .into());
    }

    if top_k <= 0 {
        return Err(KboltError::Config(format!("{scope}.top_k must be greater than zero")).into());
    }

    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(KboltError::Config(format!(
            "{scope}.top_p must be finite and in the range (0, 1]"
        ))
        .into());
    }

    if !min_p.is_finite() || min_p < 0.0 || min_p > 1.0 {
        return Err(KboltError::Config(format!(
            "{scope}.min_p must be finite and in the range [0, 1]"
        ))
        .into());
    }

    if repeat_last_n < -1 {
        return Err(KboltError::Config(format!(
            "{scope}.repeat_last_n must be greater than or equal to -1"
        ))
        .into());
    }

    if !repeat_penalty.is_finite() || repeat_penalty <= 0.0 {
        return Err(KboltError::Config(format!(
            "{scope}.repeat_penalty must be finite and greater than zero"
        ))
        .into());
    }

    if !frequency_penalty.is_finite() {
        return Err(KboltError::Config(format!("{scope}.frequency_penalty must be finite")).into());
    }

    if !presence_penalty.is_finite() {
        return Err(KboltError::Config(format!("{scope}.presence_penalty must be finite")).into());
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
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    default_space: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    providers: HashMap<String, ProviderProfileConfig>,
    #[serde(default, skip_serializing_if = "RoleBindingsConfig::is_empty")]
    roles: RoleBindingsConfig,
    #[serde(default)]
    reaping: FileReapingConfig,
    #[serde(default)]
    chunking: ChunkingConfig,
    #[serde(default)]
    ranking: RankingConfig,
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
            providers: value.providers.clone(),
            roles: value.roles.clone(),
            reaping: FileReapingConfig {
                days: value.reaping.days,
            },
            chunking: value.chunking.clone(),
            ranking: value.ranking.clone(),
        }
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

fn default_embedding_batch_size() -> usize {
    DEFAULT_EMBEDDING_BATCH_SIZE
}

fn default_inference_timeout_ms() -> u64 {
    DEFAULT_INFERENCE_TIMEOUT_MS
}

fn default_inference_max_retries() -> u32 {
    DEFAULT_INFERENCE_MAX_RETRIES
}

fn default_local_expander_max_tokens() -> usize {
    DEFAULT_LOCAL_EXPANDER_MAX_TOKENS
}

fn default_expander_seed() -> u32 {
    DEFAULT_EXPANDER_SEED
}

fn default_expander_temperature() -> f32 {
    DEFAULT_EXPANDER_TEMPERATURE
}

fn default_expander_top_k() -> i32 {
    DEFAULT_EXPANDER_TOP_K
}

fn default_expander_top_p() -> f32 {
    DEFAULT_EXPANDER_TOP_P
}

fn default_expander_min_p() -> f32 {
    DEFAULT_EXPANDER_MIN_P
}

fn default_expander_repeat_last_n() -> i32 {
    DEFAULT_EXPANDER_REPEAT_LAST_N
}

fn default_expander_repeat_penalty() -> f32 {
    DEFAULT_EXPANDER_REPEAT_PENALTY
}

fn default_expander_frequency_penalty() -> f32 {
    DEFAULT_EXPANDER_FREQUENCY_PENALTY
}

fn default_expander_presence_penalty() -> f32 {
    DEFAULT_EXPANDER_PRESENCE_PENALTY
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
