use std::fs;
use std::path::PathBuf;

use tempfile::tempdir;

use super::*;

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
    assert_eq!(config.models.embedder.provider, ModelProvider::HuggingFace);
    assert_eq!(config.models.embedder.id, DEFAULT_EMBED_MODEL);
    assert_eq!(config.models.embedder.revision, None);
    assert_eq!(config.models.reranker.provider, ModelProvider::HuggingFace);
    assert_eq!(config.models.reranker.id, DEFAULT_RERANKER_MODEL);
    assert_eq!(config.models.reranker.revision, None);
    assert_eq!(config.models.expander.provider, ModelProvider::HuggingFace);
    assert_eq!(config.models.expander.id, DEFAULT_EXPANDER_MODEL);
    assert_eq!(config.models.expander.revision, None);
    assert_eq!(config.reaping.days, DEFAULT_REAP_DAYS);
    assert_eq!(config.chunking.defaults.target_tokens, 450);
    assert_eq!(config.chunking.defaults.soft_max_tokens, 550);
    assert_eq!(config.chunking.defaults.hard_max_tokens, 750);
    assert_eq!(config.chunking.defaults.boundary_overlap_tokens, 48);
    assert_eq!(config.chunking.defaults.neighbor_window, 1);
    assert!(config.chunking.defaults.contextual_prefix);
    assert_eq!(
        config.chunking.profiles.get("code"),
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
fn load_reads_existing_values() {
    let tmp = tempdir().expect("create tempdir");
    let config_dir = tmp.path().join("config");
    let cache_dir = tmp.path().join("cache");
    let config_file = config_dir.join(CONFIG_FILENAME);
    fs::create_dir_all(&config_dir).expect("create config dir");
    fs::write(
        &config_file,
        r#"
default_space = "work"

[models]

[models.embedder]
provider = "huggingface"
id = "embed-model"
revision = "main"

[models.reranker]
provider = "huggingface"
id = "reranker-model"

[models.expander]
provider = "huggingface"
id = "expander-model"

[reaping]
days = 14

[chunking.defaults]
target_tokens = 500
soft_max_tokens = 600
hard_max_tokens = 800
boundary_overlap_tokens = 32
neighbor_window = 2
contextual_prefix = false

[chunking.profiles.code]
target_tokens = 320
soft_max_tokens = 420
hard_max_tokens = 560
boundary_overlap_tokens = 24
neighbor_window = 1
contextual_prefix = true
"#,
    )
    .expect("write config file");

    let config = load_from_file(&config_file, &config_dir, &cache_dir).expect("load config");

    assert_eq!(config.default_space.as_deref(), Some("work"));
    assert_eq!(config.models.embedder.provider, ModelProvider::HuggingFace);
    assert_eq!(config.models.embedder.id, "embed-model");
    assert_eq!(config.models.embedder.revision.as_deref(), Some("main"));
    assert_eq!(config.models.reranker.provider, ModelProvider::HuggingFace);
    assert_eq!(config.models.reranker.id, "reranker-model");
    assert_eq!(config.models.reranker.revision, None);
    assert_eq!(config.models.expander.provider, ModelProvider::HuggingFace);
    assert_eq!(config.models.expander.id, "expander-model");
    assert_eq!(config.models.expander.revision, None);
    assert_eq!(config.reaping.days, 14);
    assert_eq!(config.chunking.defaults.target_tokens, 500);
    assert_eq!(config.chunking.defaults.soft_max_tokens, 600);
    assert_eq!(config.chunking.defaults.hard_max_tokens, 800);
    assert_eq!(config.chunking.defaults.boundary_overlap_tokens, 32);
    assert_eq!(config.chunking.defaults.neighbor_window, 2);
    assert!(!config.chunking.defaults.contextual_prefix);
    assert_eq!(
        config.chunking.profiles.get("code"),
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
fn save_writes_index_toml() {
    let tmp = tempdir().expect("create tempdir");
    let config_dir = tmp.path().join("config");
    let cache_dir = tmp.path().join("cache");
    let config = Config {
        config_dir: config_dir.clone(),
        cache_dir: cache_dir.clone(),
        default_space: Some("notes".to_string()),
        models: ModelConfig {
            embedder: ModelSourceConfig {
                provider: ModelProvider::HuggingFace,
                id: "embed-model".to_string(),
                revision: Some("main".to_string()),
            },
            reranker: ModelSourceConfig {
                provider: ModelProvider::HuggingFace,
                id: "reranker-model".to_string(),
                revision: None,
            },
            expander: ModelSourceConfig {
                provider: ModelProvider::HuggingFace,
                id: "expander-model".to_string(),
                revision: None,
            },
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
    };

    save(&config).expect("save config");
    let written = fs::read_to_string(config_dir.join(CONFIG_FILENAME)).expect("read config");
    let parsed: FileConfig = toml::from_str(&written).expect("parse config");

    assert_eq!(parsed.default_space.as_deref(), Some("notes"));
    assert_eq!(parsed.models.embedder.provider, ModelProvider::HuggingFace);
    assert_eq!(parsed.models.embedder.id, "embed-model");
    assert_eq!(parsed.models.embedder.revision.as_deref(), Some("main"));
    assert_eq!(parsed.models.reranker.provider, ModelProvider::HuggingFace);
    assert_eq!(parsed.models.reranker.id, "reranker-model");
    assert_eq!(parsed.models.reranker.revision, None);
    assert_eq!(parsed.models.expander.provider, ModelProvider::HuggingFace);
    assert_eq!(parsed.models.expander.id, "expander-model");
    assert_eq!(parsed.models.expander.revision, None);
    assert_eq!(parsed.reaping.days, 30);
    assert_eq!(parsed.chunking.defaults.target_tokens, 480);
    assert_eq!(parsed.chunking.defaults.soft_max_tokens, 580);
    assert_eq!(parsed.chunking.defaults.hard_max_tokens, 760);
    assert_eq!(parsed.chunking.defaults.boundary_overlap_tokens, 40);
    assert_eq!(parsed.chunking.defaults.neighbor_window, 2);
    assert!(!parsed.chunking.defaults.contextual_prefix);
    assert_eq!(
        parsed.chunking.profiles.get("md"),
        Some(&ChunkPolicy {
            target_tokens: 450,
            soft_max_tokens: 550,
            hard_max_tokens: 750,
            boundary_overlap_tokens: 48,
            neighbor_window: 1,
            contextual_prefix: true
        })
    );
    assert!(cache_dir.is_dir());
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

    assert_eq!(policy.target_tokens, 450);
    assert_eq!(policy.soft_max_tokens, 550);
    assert_eq!(policy.hard_max_tokens, 750);
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
