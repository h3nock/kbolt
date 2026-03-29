use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use tempfile::tempdir;

use super::*;

fn load_test_config(contents: &str) -> Config {
    let tmp = tempdir().expect("create tempdir");
    let config_dir = tmp.path().join("config");
    let cache_dir = tmp.path().join("cache");
    let config_file = config_dir.join(CONFIG_FILENAME);
    fs::create_dir_all(&config_dir).expect("create config dir");
    fs::write(&config_file, contents).expect("write config file");
    load_from_file(&config_file, &config_dir, &cache_dir).expect("load config")
}

fn load_test_config_error(contents: &str) -> String {
    let tmp = tempdir().expect("create tempdir");
    let config_dir = tmp.path().join("config");
    let cache_dir = tmp.path().join("cache");
    let config_file = config_dir.join(CONFIG_FILENAME);
    fs::create_dir_all(&config_dir).expect("create config dir");
    fs::write(&config_file, contents).expect("write config file");
    load_from_file(&config_file, &config_dir, &cache_dir)
        .expect_err("config should fail to load")
        .to_string()
}

#[test]
fn load_creates_default_config_and_directories() {
    let tmp = tempdir().expect("create tempdir");
    let config_dir = tmp.path().join("config");
    let cache_dir = tmp.path().join("cache");
    let config_file = config_dir.join(CONFIG_FILENAME);

    let config = load_from_file(&config_file, &config_dir, &cache_dir).expect("load config");

    assert!(config_file.exists());
    assert!(config_dir.is_dir());
    assert!(cache_dir.is_dir());
    assert_eq!(config.default_space, None);
    assert!(config.providers.is_empty());
    assert_eq!(config.roles, RoleBindingsConfig::default());
    assert_eq!(config.reaping.days, DEFAULT_REAP_DAYS);
    assert_eq!(config.chunking.defaults.target_tokens, 800);
    assert_eq!(config.chunking.defaults.soft_max_tokens, 950);
    assert_eq!(config.chunking.defaults.hard_max_tokens, 1200);
    assert_eq!(config.chunking.defaults.boundary_overlap_tokens, 48);
    assert_eq!(config.chunking.defaults.neighbor_window, 1);
    assert!(config.chunking.defaults.contextual_prefix);
    assert_eq!(config.ranking, RankingConfig::default());
}

#[test]
fn load_reads_existing_provider_profiles_and_role_bindings() {
    let config = load_test_config(
        r#"
default_space = "work"

[providers.local_embed]
kind = "llama_cpp_server"
operation = "embedding"
base_url = "http://127.0.0.1:8101"
model = "embeddinggemma"

[providers.openai_expand]
kind = "openai_compatible"
operation = "chat_completion"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5-mini"
timeout_ms = 22000
max_retries = 2

[roles.embedder]
provider = "local_embed"
batch_size = 16

[roles.expander]
provider = "openai_expand"
max_tokens = 480
temperature = 0.6
top_k = 32
top_p = 0.9
min_p = 0.05
repeat_last_n = 32
repeat_penalty = 1.1
frequency_penalty = 0.1
presence_penalty = 0.2

[reaping]
days = 14

[chunking.defaults]
target_tokens = 500
soft_max_tokens = 600
hard_max_tokens = 800
boundary_overlap_tokens = 32
neighbor_window = 2
contextual_prefix = false
"#,
    );

    assert_eq!(config.default_space.as_deref(), Some("work"));
    assert_eq!(
        config.providers.get("local_embed"),
        Some(&ProviderProfileConfig::LlamaCppServer {
            operation: ProviderOperation::Embedding,
            base_url: "http://127.0.0.1:8101".to_string(),
            model: "embeddinggemma".to_string(),
            timeout_ms: DEFAULT_INFERENCE_TIMEOUT_MS,
            max_retries: DEFAULT_INFERENCE_MAX_RETRIES,
        })
    );
    assert_eq!(
        config.providers.get("openai_expand"),
        Some(&ProviderProfileConfig::OpenAiCompatible {
            operation: ProviderOperation::ChatCompletion,
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-5-mini".to_string(),
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            timeout_ms: 22_000,
            max_retries: 2,
        })
    );
    assert_eq!(
        config.roles,
        RoleBindingsConfig {
            embedder: Some(EmbedderRoleConfig {
                provider: "local_embed".to_string(),
                batch_size: 16,
            }),
            reranker: None,
            expander: Some(ExpanderRoleConfig {
                provider: "openai_expand".to_string(),
                max_tokens: 480,
                sampling: ExpanderRoleSamplingConfig {
                    seed: DEFAULT_EXPANDER_SEED,
                    temperature: 0.6,
                    top_k: 32,
                    top_p: 0.9,
                    min_p: 0.05,
                    repeat_last_n: 32,
                    repeat_penalty: 1.1,
                    frequency_penalty: 0.1,
                    presence_penalty: 0.2,
                },
            }),
        }
    );
    assert_eq!(config.reaping.days, 14);
    assert_eq!(config.chunking.defaults.target_tokens, 500);
    assert_eq!(config.chunking.defaults.soft_max_tokens, 600);
    assert_eq!(config.chunking.defaults.hard_max_tokens, 800);
    assert_eq!(config.chunking.defaults.boundary_overlap_tokens, 32);
    assert_eq!(config.chunking.defaults.neighbor_window, 2);
    assert!(!config.chunking.defaults.contextual_prefix);
}

#[test]
fn load_rejects_removed_pre_refactor_schema() {
    let tmp = tempdir().expect("create tempdir");
    let config_dir = tmp.path().join("config");
    let cache_dir = tmp.path().join("cache");
    let config_file = config_dir.join(CONFIG_FILENAME);
    fs::create_dir_all(&config_dir).expect("create config dir");
    fs::write(
        &config_file,
        r#"
[models]
embedder = "google/EmbeddingGemma-256"
"#,
    )
    .expect("write invalid config");

    let err = load_from_file(&config_file, &config_dir, &cache_dir)
        .expect_err("removed schema should fail");
    let message = err.to_string();
    assert!(
        message.contains("invalid config file"),
        "unexpected message: {message}"
    );
    assert!(
        message.contains(&config_file.display().to_string()),
        "unexpected message: {message}"
    );
}

#[test]
fn save_writes_index_toml() {
    let tmp = tempdir().expect("create tempdir");
    let config_dir = tmp.path().join("config");
    let cache_dir = tmp.path().join("cache");
    let config = Config {
        config_dir: config_dir.clone(),
        cache_dir: cache_dir.clone(),
        default_space: Some("notes".to_string()),
        providers: HashMap::from([(
            "local_rerank".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::Reranking,
                base_url: "http://127.0.0.1:8102".to_string(),
                model: "qwen3-reranker".to_string(),
                timeout_ms: DEFAULT_INFERENCE_TIMEOUT_MS,
                max_retries: DEFAULT_INFERENCE_MAX_RETRIES,
            },
        )]),
        roles: RoleBindingsConfig {
            embedder: None,
            reranker: Some(RerankerRoleConfig {
                provider: "local_rerank".to_string(),
            }),
            expander: None,
        },
        reaping: ReapingConfig { days: 30 },
        chunking: ChunkingConfig {
            defaults: ChunkPolicy {
                target_tokens: 480,
                soft_max_tokens: 580,
                hard_max_tokens: 760,
                boundary_overlap_tokens: 40,
                neighbor_window: 2,
                contextual_prefix: false,
            },
            profiles: [(
                "md".to_string(),
                ChunkPolicy {
                    target_tokens: 450,
                    soft_max_tokens: 550,
                    hard_max_tokens: 750,
                    boundary_overlap_tokens: 48,
                    neighbor_window: 1,
                    contextual_prefix: true,
                },
            )]
            .into_iter()
            .collect(),
        },
        ranking: RankingConfig::default(),
    };

    save(&config).expect("save config");
    let written = fs::read_to_string(config_dir.join(CONFIG_FILENAME)).expect("read config");
    let parsed: FileConfig = toml::from_str(&written).expect("parse config");

    assert_eq!(parsed.default_space.as_deref(), Some("notes"));
    assert_eq!(parsed.providers, config.providers);
    assert_eq!(parsed.roles, config.roles);
    assert_eq!(parsed.reaping.days, 30);
    assert_eq!(parsed.chunking.defaults.target_tokens, 480);
    assert_eq!(
        parsed.chunking.profiles.get("md").unwrap().target_tokens,
        450
    );
    assert_eq!(parsed.ranking, config.ranking);
    assert!(cache_dir.is_dir());
}

#[test]
fn save_rejects_invalid_provider_profiles_and_role_bindings() {
    let tmp = tempdir().expect("create tempdir");
    let config_dir = tmp.path().join("config");
    let cache_dir = tmp.path().join("cache");
    let mut config = Config {
        config_dir,
        cache_dir,
        default_space: None,
        providers: HashMap::from([(
            "local_embed".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::Embedding,
                base_url: "127.0.0.1:8101".to_string(),
                model: "embeddinggemma".to_string(),
                timeout_ms: DEFAULT_INFERENCE_TIMEOUT_MS,
                max_retries: DEFAULT_INFERENCE_MAX_RETRIES,
            },
        )]),
        roles: RoleBindingsConfig::default(),
        reaping: ReapingConfig {
            days: DEFAULT_REAP_DAYS,
        },
        chunking: ChunkingConfig::default(),
        ranking: RankingConfig::default(),
    };

    let err = save(&config).expect_err("invalid provider profile should fail on save");
    assert!(err
        .to_string()
        .contains("providers.local_embed.base_url must start with http:// or https://"));

    config.providers.insert(
        "local_embed".to_string(),
        ProviderProfileConfig::LlamaCppServer {
            operation: ProviderOperation::Embedding,
            base_url: "http://127.0.0.1:8101".to_string(),
            model: "embeddinggemma".to_string(),
            timeout_ms: DEFAULT_INFERENCE_TIMEOUT_MS,
            max_retries: DEFAULT_INFERENCE_MAX_RETRIES,
        },
    );
    config.roles.embedder = Some(EmbedderRoleConfig {
        provider: "missing".to_string(),
        batch_size: DEFAULT_EMBEDDING_BATCH_SIZE,
    });

    let err = save(&config).expect_err("invalid role binding should fail on save");
    assert!(err
        .to_string()
        .contains("roles.embedder.provider references undefined provider profile 'missing'"));
}

#[test]
fn resolve_config_dir_accepts_directory_and_index_file_path() {
    let dir = PathBuf::from("/tmp/kbolt-test");
    let file = dir.join(CONFIG_FILENAME);

    assert_eq!(resolve_config_dir(Some(&dir)).expect("resolve dir"), dir);
    assert_eq!(
        resolve_config_dir(Some(&file)).expect("resolve file"),
        PathBuf::from("/tmp/kbolt-test")
    );
}

#[test]
fn resolve_config_dir_rejects_nonstandard_toml_filename() {
    let path = PathBuf::from("/tmp/custom.toml");
    let err = resolve_config_dir(Some(&path)).expect_err("reject custom file name");

    assert!(
        err.to_string().contains(CONFIG_FILENAME),
        "error should mention expected filename"
    );
}

#[test]
fn chunk_policy_default_uses_markdown_tuned_budget() {
    let policy = ChunkPolicy::default();

    assert_eq!(policy.target_tokens, 800);
    assert_eq!(policy.soft_max_tokens, 950);
    assert_eq!(policy.hard_max_tokens, 1200);
    assert_eq!(policy.boundary_overlap_tokens, 48);
    assert_eq!(policy.neighbor_window, 1);
    assert!(policy.contextual_prefix);
}

#[test]
fn chunking_config_default_includes_code_profile() {
    let chunking = ChunkingConfig::default();

    assert_eq!(chunking.defaults, ChunkPolicy::default());
    assert_eq!(
        chunking.profiles.get("code"),
        Some(&ChunkPolicy {
            target_tokens: 320,
            soft_max_tokens: 420,
            hard_max_tokens: 560,
            boundary_overlap_tokens: 24,
            neighbor_window: 1,
            contextual_prefix: true
        })
    );
}

#[test]
fn ranking_config_default_uses_tuned_dbsf_hybrid_fusion() {
    let ranking = RankingConfig::default();

    assert_eq!(ranking.hybrid_fusion.mode, HybridFusionMode::Dbsf);
    assert_eq!(
        ranking.hybrid_fusion.linear,
        LinearHybridFusionConfig {
            dense_weight: DEFAULT_RANKING_HYBRID_LINEAR_DENSE_WEIGHT,
            bm25_weight: DEFAULT_RANKING_HYBRID_LINEAR_BM25_WEIGHT,
        }
    );
}

#[test]
fn load_rejects_invalid_chunking_budget_order() {
    let err = load_test_config_error(
        r#"
[chunking.defaults]
target_tokens = 600
soft_max_tokens = 550
hard_max_tokens = 750
"#,
    );
    assert!(err.contains("target_tokens"));
}

#[test]
fn load_rejects_zero_chunking_caps() {
    let err = load_test_config_error(
        r#"
[chunking.defaults]
target_tokens = 450
soft_max_tokens = 550
hard_max_tokens = 750

[chunking.profiles.md]
target_tokens = 0
soft_max_tokens = 550
hard_max_tokens = 750
"#,
    );
    assert!(err.contains("must be greater than zero"));
}

#[test]
fn load_rejects_invalid_provider_profile_fields() {
    let err = load_test_config_error(
        r#"
[providers.local_embed]
kind = "llama_cpp_server"
operation = "embedding"
base_url = "http://127.0.0.1:8101"
model = "   "
"#,
    );
    assert!(err.contains("providers.local_embed.model must not be empty"));

    let err = load_test_config_error(
        r#"
[providers.local_embed]
kind = "llama_cpp_server"
operation = "embedding"
base_url = "127.0.0.1:8101"
model = "embeddinggemma"
"#,
    );
    assert!(err.contains("providers.local_embed.base_url must start with http:// or https://"));

    let err = load_test_config_error(
        r#"
[providers.local_embed]
kind = "llama_cpp_server"
operation = "embedding"
base_url = "http://127.0.0.1:8101"
model = "embeddinggemma"
timeout_ms = 0
"#,
    );
    assert!(err.contains("providers.local_embed.timeout_ms must be greater than zero"));
}

#[test]
fn load_validates_role_provider_bindings() {
    let err = load_test_config_error(
        r#"
[roles.embedder]
provider = "missing"
"#,
    );
    assert!(err.contains("roles.embedder.provider references undefined provider profile 'missing'"));

    let err = load_test_config_error(
        r#"
[providers.local_expand]
kind = "llama_cpp_server"
operation = "chat_completion"
base_url = "http://127.0.0.1:8103"
model = "qwen3-1.7b"

[roles.embedder]
provider = "local_expand"
"#,
    );
    assert!(err.contains(
        "roles.embedder.provider 'local_expand' uses incompatible operation 'chat_completion'"
    ));

    let config = load_test_config(
        r#"
[providers.remote_rerank]
kind = "openai_compatible"
operation = "chat_completion"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5-mini"

[roles.reranker]
provider = "remote_rerank"
"#,
    );
    assert_eq!(
        config.roles.reranker,
        Some(RerankerRoleConfig {
            provider: "remote_rerank".to_string(),
        })
    );
}

#[test]
fn load_rejects_invalid_role_settings() {
    let err = load_test_config_error(
        r#"
[providers.local_embed]
kind = "llama_cpp_server"
operation = "embedding"
base_url = "http://127.0.0.1:8101"
model = "embeddinggemma"

[roles.embedder]
provider = "local_embed"
batch_size = 0
"#,
    );
    assert!(err.contains("roles.embedder.batch_size must be greater than zero"));

    let err = load_test_config_error(
        r#"
[providers.remote_expand]
kind = "openai_compatible"
operation = "chat_completion"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5-mini"

[roles.expander]
provider = "remote_expand"
max_tokens = 0
"#,
    );
    assert!(err.contains("roles.expander.max_tokens must be greater than zero"));

    let err = load_test_config_error(
        r#"
[providers.remote_expand]
kind = "openai_compatible"
operation = "chat_completion"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5-mini"

[roles.expander]
provider = "remote_expand"
temperature = 0.0
"#,
    );
    assert!(err.contains("roles.expander.temperature must be finite and greater than zero"));
}

#[test]
fn load_rejects_invalid_ranking_config() {
    let err = load_test_config_error(
        r#"
[ranking]
rerank_candidates_min = 12
rerank_candidates_max = 8
"#,
    );
    assert!(err.contains("ranking.rerank_candidates_max (8) must be greater than or equal to ranking.rerank_candidates_min (12)"));
}

#[test]
fn load_reads_explicit_hybrid_fusion_config() {
    let config = load_test_config(
        r#"
[ranking.hybrid_fusion]
mode = "linear"

[ranking.hybrid_fusion.linear]
dense_weight = 0.55
bm25_weight = 0.45

[ranking.hybrid_fusion.dbsf]
dense_weight = 0.8
bm25_weight = 0.2
stddevs = 2.75
"#,
    );

    assert_eq!(
        config.ranking.hybrid_fusion,
        HybridFusionConfig {
            mode: HybridFusionMode::Linear,
            linear: LinearHybridFusionConfig {
                dense_weight: 0.55,
                bm25_weight: 0.45,
            },
            dbsf: DbsfHybridFusionConfig {
                dense_weight: 0.8,
                bm25_weight: 0.2,
                stddevs: 2.75,
            },
            rrf: RrfHybridFusionConfig {
                k: DEFAULT_RANKING_HYBRID_RRF_K,
            },
        }
    );
}
