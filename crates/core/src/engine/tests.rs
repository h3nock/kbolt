use fs2::FileExt;
use tempfile::tempdir;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::sync::{Mutex, OnceLock};

use crate::config::{
    ChunkingConfig, Config, ModelConfig, ModelProvider, ModelSourceConfig, ReapingConfig,
};
use crate::engine::{retrieval_text_with_prefix, Engine};
use crate::storage::Storage;
use crate::ModelPullEvent;
use kbolt_types::{
    ActiveSpaceSource, AddCollectionRequest, GetRequest, KboltError, Locator, MultiGetRequest,
    OmitReason, SearchMode, SearchRequest, UpdateOptions,
};

fn test_engine() -> Engine {
    let root = tempdir().expect("create temp root");
    let root_path = root.path().to_path_buf();
    std::mem::forget(root);
    let config_dir = root_path.join("config");
    let cache_dir = root_path.join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let config = Config {
        config_dir,
        cache_dir,
        default_space: None,
        models: ModelConfig {
            embedder: ModelSourceConfig {
                provider: ModelProvider::HuggingFace,
                id: "embed-model".to_string(),
                revision: None,
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
        reaping: ReapingConfig { days: 7 },
        chunking: ChunkingConfig::default(),
    };
    Engine::from_parts(storage, config)
}

fn test_engine_with_default_space(default_space: Option<&str>) -> Engine {
    let root = tempdir().expect("create temp root");
    let root_path = root.path().to_path_buf();
    std::mem::forget(root);
    let config_dir = root_path.join("config");
    let cache_dir = root_path.join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let config = Config {
        config_dir,
        cache_dir,
        default_space: default_space.map(ToString::to_string),
        models: ModelConfig {
            embedder: ModelSourceConfig {
                provider: ModelProvider::HuggingFace,
                id: "embed-model".to_string(),
                revision: None,
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
        reaping: ReapingConfig { days: 7 },
        chunking: ChunkingConfig::default(),
    };
    Engine::from_parts(storage, config)
}

fn with_kbolt_space_env<T>(value: Option<&str>, run: impl FnOnce() -> T) -> T {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("lock env mutex");

    let old_value: Option<OsString> = std::env::var_os("KBOLT_SPACE");
    match value {
        Some(v) => std::env::set_var("KBOLT_SPACE", v),
        None => std::env::remove_var("KBOLT_SPACE"),
    }

    let result = run();
    match old_value {
        Some(v) => std::env::set_var("KBOLT_SPACE", v),
        None => std::env::remove_var("KBOLT_SPACE"),
    }
    result
}

fn update_options(space: Option<&str>, collections: &[&str]) -> UpdateOptions {
    UpdateOptions {
        space: space.map(ToString::to_string),
        collections: collections.iter().map(|item| item.to_string()).collect(),
        no_embed: false,
        dry_run: false,
        verbose: false,
    }
}

fn add_collection_fixture(engine: &Engine, space: &str, name: &str, path: std::path::PathBuf) {
    engine
        .add_collection(AddCollectionRequest {
            path,
            space: Some(space.to_string()),
            name: Some(name.to_string()),
            description: None,
            extensions: None,
            no_index: true,
        })
        .expect("add collection fixture");
}

fn write_text_file(path: &std::path::Path, text: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent directories");
    }
    std::fs::write(path, text).expect("write file");
}

#[test]
fn retrieval_text_with_prefix_adds_title_and_heading_context() {
    let text = retrieval_text_with_prefix("body text", "Guide", Some("Setup > Install"), true);
    assert_eq!(text, "title: Guide\nheading: Setup > Install\n\nbody text");
}

#[test]
fn retrieval_text_with_prefix_respects_disabled_flag() {
    let text = retrieval_text_with_prefix("body text", "Guide", Some("Setup"), false);
    assert_eq!(text, "body text");
}

const MODEL_MANIFEST_FILENAME: &str = ".kbolt-model-manifest.json";

fn model_provider_key(provider: &ModelProvider) -> &'static str {
    match provider {
        ModelProvider::HuggingFace => "huggingface",
    }
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn seed_model_artifact(
    model_root: &std::path::Path,
    role: &str,
    source: &ModelSourceConfig,
    payload: &[u8],
) {
    let role_dir = model_root.join(role);
    std::fs::create_dir_all(&role_dir).expect("create model role dir");
    std::fs::write(role_dir.join("model.bin"), payload).expect("write model payload");

    let provider = model_provider_key(&source.provider);
    let id = json_escape(&source.id);
    let manifest = match source.revision.as_deref() {
        Some(revision) => format!(
            "{{\n  \"provider\": \"{provider}\",\n  \"id\": \"{id}\",\n  \"revision\": \"{}\"\n}}\n",
            json_escape(revision)
        ),
        None => format!(
            "{{\n  \"provider\": \"{provider}\",\n  \"id\": \"{id}\",\n  \"revision\": null\n}}\n"
        ),
    };

    std::fs::write(role_dir.join(MODEL_MANIFEST_FILENAME), manifest).expect("write model manifest");
}

#[test]
fn add_space_and_space_info_include_description_and_zero_counts() {
    let engine = test_engine();

    let added = engine
        .add_space("work", Some("work docs"))
        .expect("add space");
    assert_eq!(added.name, "work");
    assert_eq!(added.description.as_deref(), Some("work docs"));
    assert_eq!(added.collection_count, 0);
    assert_eq!(added.document_count, 0);
    assert_eq!(added.chunk_count, 0);

    let fetched = engine.space_info("work").expect("fetch space info");
    assert_eq!(fetched.name, "work");
    assert_eq!(fetched.description.as_deref(), Some("work docs"));
}

#[test]
fn list_spaces_returns_default_and_added_spaces() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    engine.add_space("notes", None).expect("add notes");

    let spaces = engine.list_spaces().expect("list spaces");
    let names: Vec<String> = spaces.into_iter().map(|space| space.name).collect();
    assert_eq!(
        names,
        vec![
            "default".to_string(),
            "notes".to_string(),
            "work".to_string()
        ]
    );
}

#[test]
fn describe_rename_and_remove_space_delegate_to_storage() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    engine
        .describe_space("work", "new description")
        .expect("describe space");
    let described = engine.space_info("work").expect("space info");
    assert_eq!(described.description.as_deref(), Some("new description"));

    engine
        .rename_space("work", "team")
        .expect("rename work to team");
    let renamed = engine.space_info("team").expect("team should exist");
    assert_eq!(renamed.name, "team");
    let missing_old = engine.space_info("work").expect_err("work should be missing");
    match KboltError::from(missing_old) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "work"),
        other => panic!("unexpected error: {other}"),
    }

    engine.remove_space("team").expect("remove team");
    let missing_team = engine.space_info("team").expect_err("team should be missing");
    match KboltError::from(missing_team) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "team"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn config_and_storage_accessors_expose_engine_components() {
    let engine = test_engine();
    assert_eq!(
        engine.config().models.embedder.id,
        "embed-model",
        "config accessor should expose loaded config"
    );

    let default_space = engine
        .storage()
        .get_space("default")
        .expect("default space should exist");
    assert_eq!(default_space.name, "default");
    assert_eq!(engine.config().default_space, None::<String>);
    assert!(!engine.config().config_dir.as_os_str().is_empty());
}

#[test]
fn resolve_space_returns_explicit_space_when_provided() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let resolved = engine
        .resolve_space(Some("work"))
        .expect("resolve explicit space");
    assert_eq!(resolved, "work");
}

#[test]
fn resolve_space_uses_configured_default_when_no_explicit_space() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");

        let resolved = engine.resolve_space(None).expect("resolve default space");
        assert_eq!(resolved, "work");
    });
}

#[test]
fn set_default_space_persists_config_and_can_clear_it() {
    let mut engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let set = engine
        .set_default_space(Some("work"))
        .expect("set default space");
    assert_eq!(set.as_deref(), Some("work"));
    assert_eq!(engine.config().default_space.as_deref(), Some("work"));

    let loaded = crate::config::load(Some(engine.config().config_dir.as_path()))
        .expect("reload config from disk");
    assert_eq!(loaded.default_space.as_deref(), Some("work"));

    let cleared = engine.set_default_space(None).expect("clear default space");
    assert_eq!(cleared, None);
    assert_eq!(engine.config().default_space, None);
}

#[test]
fn set_default_space_requires_existing_space() {
    let mut engine = test_engine();

    let err = engine
        .set_default_space(Some("missing"))
        .expect_err("missing space should fail");
    match KboltError::from(err) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn resolve_space_prefers_env_over_config_default() {
    with_kbolt_space_env(Some("notes"), || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let resolved = engine.resolve_space(None).expect("resolve space");
        assert_eq!(resolved, "notes");
    });
}

#[test]
fn resolve_space_returns_no_active_space_when_no_sources_exist() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        let err = engine
            .resolve_space(None)
            .expect_err("expected no active space");
        match KboltError::from(err) {
            KboltError::NoActiveSpace => {}
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn current_space_prefers_flag_over_env_and_default() {
    with_kbolt_space_env(Some("notes"), || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");
        engine.add_space("ops", None).expect("add ops");

        let current = engine
            .current_space(Some("ops"))
            .expect("resolve current space")
            .expect("expected active space");
        assert_eq!(current.name, "ops");
        assert_eq!(current.source, ActiveSpaceSource::Flag);
    });
}

#[test]
fn current_space_reports_env_source() {
    with_kbolt_space_env(Some("notes"), || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let current = engine
            .current_space(None)
            .expect("resolve current space")
            .expect("expected active space");
        assert_eq!(current.name, "notes");
        assert_eq!(current.source, ActiveSpaceSource::EnvVar);
    });
}

#[test]
fn current_space_reports_default_source() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");

        let current = engine
            .current_space(None)
            .expect("resolve current space")
            .expect("expected active space");
        assert_eq!(current.name, "work");
        assert_eq!(current.source, ActiveSpaceSource::ConfigDefault);
    });
}

#[test]
fn current_space_returns_none_when_no_space_is_active() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        let current = engine.current_space(None).expect("resolve current space");
        assert_eq!(current, None);
    });
}

#[test]
fn collection_info_without_space_uses_unique_collection_lookup() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        engine
            .add_collection(AddCollectionRequest {
                path: collection_path,
                space: Some("work".to_string()),
                name: Some("api".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add collection");

        let info = engine
            .collection_info(None, "api")
            .expect("resolve unique collection");
        assert_eq!(info.space, "work");
        assert_eq!(info.name, "api");
    });
}

#[test]
fn collection_info_without_space_reports_ambiguous_collection_lookup() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");
        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        engine
            .add_collection(AddCollectionRequest {
                path: work_path,
                space: Some("work".to_string()),
                name: Some("api".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add work collection");
        engine
            .add_collection(AddCollectionRequest {
                path: notes_path,
                space: Some("notes".to_string()),
                name: Some("api".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add notes collection");

        let err = engine
            .collection_info(None, "api")
            .expect_err("expected ambiguous collection");
        match KboltError::from(err) {
            KboltError::AmbiguousSpace { collection, spaces } => {
                assert_eq!(collection, "api");
                assert_eq!(spaces, vec!["notes".to_string(), "work".to_string()]);
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn add_collection_and_collection_info_with_explicit_space() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    let root = tempdir().expect("create temp root");
    let collection_path = root.path().join("api");
    std::fs::create_dir_all(&collection_path).expect("create collection dir");

    let added = engine
        .add_collection(AddCollectionRequest {
            path: collection_path.clone(),
            space: Some("work".to_string()),
            name: Some("api".to_string()),
            description: Some("API docs".to_string()),
            extensions: Some(vec!["rs".to_string(), "md".to_string()]),
            no_index: true,
        })
        .expect("add collection");
    assert_eq!(added.name, "api");
    assert_eq!(added.space, "work");
    assert_eq!(added.path, collection_path);
    assert_eq!(added.description.as_deref(), Some("API docs"));
    assert_eq!(
        added.extensions,
        Some(vec!["rs".to_string(), "md".to_string()])
    );
    assert_eq!(added.document_count, 0);
    assert_eq!(added.active_document_count, 0);
    assert_eq!(added.chunk_count, 0);
    assert_eq!(added.embedded_chunk_count, 0);

    let info = engine
        .collection_info(Some("work"), "api")
        .expect("fetch collection info");
    assert_eq!(info.name, "api");
    assert_eq!(info.space, "work");
}

#[test]
fn add_collection_without_no_index_triggers_initial_index_update() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    let root = tempdir().expect("create temp root");
    let collection_path = root.path().join("api");
    std::fs::create_dir_all(&collection_path).expect("create collection dir");
    write_text_file(&collection_path.join("src/lib.rs"), "fn alpha() {}\n");

    let added = engine
        .add_collection(AddCollectionRequest {
            path: collection_path.clone(),
            space: Some("work".to_string()),
            name: Some("api".to_string()),
            description: None,
            extensions: None,
            no_index: false,
        })
        .expect("collection add should index by default");
    assert_eq!(added.space, "work");
    assert_eq!(added.name, "api");
    assert_eq!(added.path, collection_path);
    assert_eq!(added.document_count, 1);
    assert_eq!(added.active_document_count, 1);
    assert_eq!(added.chunk_count, 1);
}

#[test]
fn collection_mutation_wrappers_delegate_to_storage_with_explicit_space() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    let root = tempdir().expect("create temp root");
    let collection_path = root.path().join("api");
    std::fs::create_dir_all(&collection_path).expect("create collection dir");
    engine
        .add_collection(AddCollectionRequest {
            path: collection_path,
            space: Some("work".to_string()),
            name: Some("api".to_string()),
            description: None,
            extensions: None,
            no_index: true,
        })
        .expect("add collection");
    let ignore_dir = engine.config().config_dir.join("ignores").join("work");
    std::fs::create_dir_all(&ignore_dir).expect("create ignore dir");
    let old_ignore_path = ignore_dir.join("api.ignore");
    write_text_file(&old_ignore_path, "dist/\n");
    assert!(old_ignore_path.exists(), "ignore file should exist");

    engine
        .describe_collection(Some("work"), "api", "updated desc")
        .expect("describe collection");
    let described = engine
        .collection_info(Some("work"), "api")
        .expect("collection info");
    assert_eq!(described.description.as_deref(), Some("updated desc"));

    engine
        .rename_collection(Some("work"), "api", "backend")
        .expect("rename collection");
    let renamed = engine
        .collection_info(Some("work"), "backend")
        .expect("backend info");
    assert_eq!(renamed.name, "backend");
    let renamed_ignore_path = ignore_dir.join("backend.ignore");
    assert!(!old_ignore_path.exists(), "old ignore file should be renamed");
    assert!(renamed_ignore_path.exists(), "renamed ignore file should exist");

    engine
        .remove_collection(Some("work"), "backend")
        .expect("remove collection");
    assert!(
        !renamed_ignore_path.exists(),
        "ignore file should be deleted with collection"
    );
    let missing = engine
        .collection_info(Some("work"), "backend")
        .expect_err("backend should be removed");
    match KboltError::from(missing) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "backend"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn list_collections_returns_all_or_space_scoped_collections() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    engine.add_space("notes", None).expect("add notes");

    let root = tempdir().expect("create temp root");
    let work_path = root.path().join("work-api");
    let notes_path = root.path().join("notes-wiki");
    std::fs::create_dir_all(&work_path).expect("create work dir");
    std::fs::create_dir_all(&notes_path).expect("create notes dir");

    engine
        .add_collection(AddCollectionRequest {
            path: work_path,
            space: Some("work".to_string()),
            name: Some("api".to_string()),
            description: None,
            extensions: None,
            no_index: true,
        })
        .expect("add work collection");
    engine
        .add_collection(AddCollectionRequest {
            path: notes_path,
            space: Some("notes".to_string()),
            name: Some("wiki".to_string()),
            description: None,
            extensions: None,
            no_index: true,
        })
        .expect("add notes collection");

    let all = engine.list_collections(None).expect("list all");
    assert_eq!(all.len(), 2);
    assert!(
        all.iter()
            .any(|collection| collection.space == "work" && collection.name == "api")
    );
    assert!(
        all.iter()
            .any(|collection| collection.space == "notes" && collection.name == "wiki")
    );

    let work_only = engine
        .list_collections(Some("work"))
        .expect("list work only");
    assert_eq!(work_only.len(), 1);
    assert_eq!(work_only[0].space, "work");
    assert_eq!(work_only[0].name, "api");
}

#[test]
fn read_collection_ignore_returns_none_when_file_missing() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let (space, content) = engine
            .read_collection_ignore(Some("work"), "api")
            .expect("read ignore file");
        assert_eq!(space, "work");
        assert_eq!(content, None);
    });
}

#[test]
fn read_collection_ignore_returns_file_contents() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        write_text_file(&ignore_path, "dist/\n*.tmp\n");

        let (space, content) = engine
            .read_collection_ignore(None, "api")
            .expect("read ignore file");
        assert_eq!(space, "work");
        assert_eq!(content.as_deref(), Some("dist/\n*.tmp"));
    });
}

#[test]
fn add_collection_ignore_pattern_creates_file_and_appends_patterns() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let (space, first) = engine
            .add_collection_ignore_pattern(Some("work"), "api", "dist/")
            .expect("add first pattern");
        assert_eq!(space, "work");
        assert_eq!(first, "dist/");

        let (space, second) = engine
            .add_collection_ignore_pattern(None, "api", "*.tmp")
            .expect("add second pattern");
        assert_eq!(space, "work");
        assert_eq!(second, "*.tmp");

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        let saved = std::fs::read_to_string(ignore_path).expect("read ignore file");
        assert_eq!(saved, "dist/\n*.tmp\n");
    });
}

#[test]
fn add_collection_ignore_pattern_rejects_empty_or_multiline_input() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let empty = engine
            .add_collection_ignore_pattern(Some("work"), "api", "   ")
            .expect_err("empty pattern should fail");
        match KboltError::from(empty) {
            KboltError::InvalidInput(message) => {
                assert!(
                    message.contains("cannot be empty"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }

        let multiline = engine
            .add_collection_ignore_pattern(Some("work"), "api", "dist/\n*.tmp")
            .expect_err("multiline pattern should fail");
        match KboltError::from(multiline) {
            KboltError::InvalidInput(message) => {
                assert!(
                    message.contains("single line"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn remove_collection_ignore_pattern_removes_matches_and_deletes_empty_file() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        write_text_file(&ignore_path, "dist/\n*.tmp\ndist/\n");

        let (space, removed) = engine
            .remove_collection_ignore_pattern(Some("work"), "api", "dist/")
            .expect("remove dist pattern");
        assert_eq!(space, "work");
        assert_eq!(removed, 2);
        let saved = std::fs::read_to_string(&ignore_path).expect("read updated ignore file");
        assert_eq!(saved, "*.tmp\n");

        let (space, removed) = engine
            .remove_collection_ignore_pattern(None, "api", "*.tmp")
            .expect("remove tmp pattern");
        assert_eq!(space, "work");
        assert_eq!(removed, 1);
        assert!(!ignore_path.exists(), "ignore file should be deleted");
    });
}

#[test]
fn remove_collection_ignore_pattern_returns_zero_when_pattern_or_file_is_missing() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let (space, removed) = engine
            .remove_collection_ignore_pattern(Some("work"), "api", "dist/")
            .expect("remove from missing file");
        assert_eq!(space, "work");
        assert_eq!(removed, 0);

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        write_text_file(&ignore_path, "*.tmp\n");

        let (space, removed) = engine
            .remove_collection_ignore_pattern(None, "api", "dist/")
            .expect("remove missing pattern");
        assert_eq!(space, "work");
        assert_eq!(removed, 0);
        let saved = std::fs::read_to_string(ignore_path).expect("read untouched ignore file");
        assert_eq!(saved, "*.tmp\n");
    });
}

#[test]
fn list_collection_ignores_returns_entries_with_pattern_counts_and_space_scope() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "wiki", notes_path);

        write_text_file(
            &engine
                .config()
                .config_dir
                .join("ignores")
                .join("work")
                .join("api.ignore"),
            "dist/\n*.tmp\n",
        );
        write_text_file(
            &engine
                .config()
                .config_dir
                .join("ignores")
                .join("notes")
                .join("wiki.ignore"),
            "# comment\n\nbuild/\n",
        );

        let all = engine.list_collection_ignores(None).expect("list all ignores");
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].space, "notes");
        assert_eq!(all[0].collection, "wiki");
        assert_eq!(all[0].pattern_count, 1);
        assert_eq!(all[1].space, "work");
        assert_eq!(all[1].collection, "api");
        assert_eq!(all[1].pattern_count, 2);

        let scoped = engine
            .list_collection_ignores(Some("work"))
            .expect("list scoped ignores");
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].space, "work");
        assert_eq!(scoped[0].collection, "api");
        assert_eq!(scoped[0].pattern_count, 2);
    });
}

#[test]
fn prepare_collection_ignore_edit_creates_missing_ignore_file() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let (space, path) = engine
            .prepare_collection_ignore_edit(None, "api")
            .expect("prepare ignore file");
        assert_eq!(space, "work");
        assert!(
            path.ends_with(std::path::Path::new("ignores/work/api.ignore")),
            "unexpected ignore path: {}",
            path.display()
        );
        assert!(path.exists(), "ignore file should be created");
        let content = std::fs::read_to_string(path).expect("read ignore file");
        assert_eq!(content, "");
    });
}

#[test]
fn list_files_returns_entries_and_applies_prefix_filter() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&work_path.join("docs/guide.md"), "guide text\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let work_space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        let to_deactivate = engine
            .storage()
            .get_document_by_path(collection.id, "docs/guide.md")
            .expect("get docs/guide.md")
            .expect("docs/guide.md should exist");
        engine
            .storage()
            .deactivate_document(to_deactivate.id)
            .expect("deactivate docs file");

        let all = engine
            .list_files(Some("work"), "api", None)
            .expect("list all files");
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].path, "docs/guide.md");
        assert_eq!(all[1].path, "src/lib.rs");
        assert!(!all[0].active);
        assert!(all[1].active);
        assert!(all.iter().all(|file| file.docid.starts_with('#')));
        assert!(all.iter().all(|file| file.chunk_count > 0));
        assert!(all.iter().all(|file| !file.embedded));

        let src_only = engine
            .list_files(Some("work"), "api", Some("src"))
            .expect("list src files");
        assert_eq!(src_only.len(), 1);
        assert_eq!(src_only[0].path, "src/lib.rs");
    });
}

#[test]
fn list_files_without_space_uses_unique_collection_lookup() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");
        engine
            .update(update_options(None, &["api"]))
            .expect("initial update");

        let files = engine
            .list_files(None, "api", None)
            .expect("list files with unique lookup");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/lib.rs");
    });
}

#[test]
fn list_files_errors_for_ambiguous_collection_and_invalid_prefix() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "api", notes_path);

        let err = engine
            .list_files(None, "api", None)
            .expect_err("expected ambiguous collection");
        match KboltError::from(err) {
            KboltError::AmbiguousSpace { collection, spaces } => {
                assert_eq!(collection, "api");
                assert_eq!(spaces, vec!["notes".to_string(), "work".to_string()]);
            }
            other => panic!("unexpected error: {other}"),
        }

        let err = engine
            .list_files(Some("work"), "api", Some("../src"))
            .expect_err("expected invalid prefix");
        match KboltError::from(err) {
            KboltError::InvalidInput(message) => {
                assert!(message.contains("prefix"), "unexpected message: {message}");
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn get_document_by_path_supports_offsets_and_stale_detection() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let file_path = work_path.join("src/lib.rs");
        write_text_file(&file_path, "line-a\nline-b\nline-c\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let sliced = engine
            .get_document(GetRequest {
                locator: Locator::Path("api/src/lib.rs".to_string()),
                space: Some("work".to_string()),
                offset: Some(1),
                limit: Some(1),
            })
            .expect("get sliced document");
        assert_eq!(sliced.path, "api/src/lib.rs");
        assert_eq!(sliced.space, "work");
        assert_eq!(sliced.collection, "api");
        assert_eq!(sliced.content, "line-b");
        assert_eq!(sliced.total_lines, 3);
        assert_eq!(sliced.returned_lines, 1);
        assert!(!sliced.stale);

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "line-a\nline-b\nline-c\nline-d\n");
        let stale = engine
            .get_document(GetRequest {
                locator: Locator::Path("api/src/lib.rs".to_string()),
                space: Some("work".to_string()),
                offset: None,
                limit: None,
            })
            .expect("get stale document");
        assert!(stale.stale);
        assert_eq!(stale.returned_lines, stale.total_lines);
    });
}

#[test]
fn get_document_by_docid_resolves_uniquely_and_honors_optional_space_scope() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "wiki", notes_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&notes_path.join("guide.md"), "notes guide\n");
        engine
            .update(update_options(None, &[]))
            .expect("initial update");

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list files");
        let docid = files[0].docid.clone();

        let doc = engine
            .get_document(GetRequest {
                locator: Locator::DocId(docid.clone()),
                space: None,
                offset: None,
                limit: None,
            })
            .expect("get document by docid");
        assert_eq!(doc.space, "work");
        assert_eq!(doc.collection, "api");
        assert_eq!(doc.path, "api/src/lib.rs");

        let wrong_scope = engine
            .get_document(GetRequest {
                locator: Locator::DocId(docid),
                space: Some("notes".to_string()),
                offset: None,
                limit: None,
            })
            .expect_err("wrong space scope should not resolve docid");
        match KboltError::from(wrong_scope) {
            KboltError::DocumentNotFound { path } => assert!(path.starts_with('#')),
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn get_document_errors_for_deleted_file_and_ambiguous_docid() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let file_path = work_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        std::fs::remove_file(&file_path).expect("remove file");

        let deleted_err = engine
            .get_document(GetRequest {
                locator: Locator::Path("api/src/lib.rs".to_string()),
                space: Some("work".to_string()),
                offset: None,
                limit: None,
            })
            .expect_err("deleted file should error");
        match KboltError::from(deleted_err) {
            KboltError::FileDeleted(path) => {
                assert!(path.ends_with("src/lib.rs"), "unexpected path: {}", path.display());
            }
            other => panic!("unexpected error: {other}"),
        }

        let work_space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        engine
            .storage()
            .upsert_document(
                collection.id,
                "a.rs",
                "a.rs",
                "abc123000000",
                "2026-03-01T10:00:00Z",
            )
            .expect("insert first synthetic hash");
        engine
            .storage()
            .upsert_document(
                collection.id,
                "b.rs",
                "b.rs",
                "abc123999999",
                "2026-03-01T10:01:00Z",
            )
            .expect("insert second synthetic hash");

        let ambiguous = engine
            .get_document(GetRequest {
                locator: Locator::DocId("#abc123".to_string()),
                space: Some("work".to_string()),
                offset: None,
                limit: None,
            })
            .expect_err("ambiguous docid should fail");
        match KboltError::from(ambiguous) {
            KboltError::InvalidInput(message) => {
                assert!(message.contains("ambiguous"), "unexpected message: {message}");
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn multi_get_respects_max_files_and_preserves_locator_order() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("a.md"), "a\n");
        write_text_file(&work_path.join("b.md"), "bb\n");
        write_text_file(&work_path.join("c.md"), "ccc\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let result = engine
            .multi_get(MultiGetRequest {
                locators: vec![
                    Locator::Path("api/a.md".to_string()),
                    Locator::Path("api/b.md".to_string()),
                    Locator::Path("api/c.md".to_string()),
                ],
                space: Some("work".to_string()),
                max_files: 2,
                max_bytes: 1024,
            })
            .expect("run multi_get");

        assert_eq!(result.resolved_count, 3);
        assert_eq!(result.documents.len(), 2);
        assert_eq!(result.documents[0].path, "api/a.md");
        assert_eq!(result.documents[1].path, "api/b.md");
        assert_eq!(result.omitted.len(), 1);
        assert_eq!(result.omitted[0].path, "api/c.md");
        assert_eq!(result.omitted[0].reason, OmitReason::MaxFiles);
    });
}

#[test]
fn multi_get_respects_max_bytes_and_supports_mixed_locators() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("a.md"), "alpha\n");
        write_text_file(&work_path.join("b.md"), "beta\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list files");
        let docid = files
            .iter()
            .find(|entry| entry.path == "a.md")
            .expect("a.md entry should exist")
            .docid
            .clone();

        let result = engine
            .multi_get(MultiGetRequest {
                locators: vec![
                    Locator::DocId(docid),
                    Locator::Path("api/b.md".to_string()),
                ],
                space: Some("work".to_string()),
                max_files: 10,
                max_bytes: 7,
            })
            .expect("run multi_get");

        assert_eq!(result.resolved_count, 2);
        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.documents[0].path, "api/a.md");
        assert_eq!(result.omitted.len(), 1);
        assert_eq!(result.omitted[0].path, "api/b.md");
        assert_eq!(result.omitted[0].reason, OmitReason::MaxBytes);
        assert!(result.warnings.is_empty());
    });
}

#[test]
fn multi_get_reports_deleted_files_as_warnings() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let existing = work_path.join("a.md");
        let deleted = work_path.join("b.md");
        write_text_file(&existing, "alpha\n");
        write_text_file(&deleted, "beta\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        std::fs::remove_file(&deleted).expect("remove b.md");

        let result = engine
            .multi_get(MultiGetRequest {
                locators: vec![
                    Locator::Path("api/a.md".to_string()),
                    Locator::Path("api/b.md".to_string()),
                ],
                space: Some("work".to_string()),
                max_files: 10,
                max_bytes: 51_200,
            })
            .expect("run multi_get");

        assert_eq!(result.resolved_count, 1);
        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.documents[0].path, "api/a.md");
        assert!(result.omitted.is_empty());
        assert_eq!(result.warnings.len(), 1);
        assert!(result.warnings[0].contains("file deleted since indexing:"));
        assert!(result.warnings[0].contains("b.md"));
    });
}

#[test]
fn search_keyword_returns_ranked_results_for_targeted_collection() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "fn alpha_search_term() {}\n");
        write_text_file(&work_path.join("src/other.rs"), "fn beta() {}\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "alpha_search_term".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 5,
                min_score: 0.0,
                no_rerank: false,
                debug: true,
            })
            .expect("run keyword search");

        assert_eq!(response.mode, SearchMode::Keyword);
        assert_eq!(response.query, "alpha_search_term");
        assert!(!response.results.is_empty(), "expected at least one result");
        let first = &response.results[0];
        assert_eq!(first.space, "work");
        assert_eq!(first.collection, "api");
        assert!(first.path.starts_with("api/"));
        assert!(first.docid.starts_with('#'));
        assert!(first.text.contains("alpha_search_term"));
        assert!(first.score >= 0.0 && first.score <= 1.0);
        assert!(response.staleness_hint.is_some());
        let signals = first.signals.as_ref().expect("debug signals");
        assert!(signals.bm25.is_some());
        assert!(signals.dense.is_none());
        assert!(signals.reranker.is_none());
    });
}

#[test]
fn search_keyword_includes_neighbor_chunks_for_context() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let left = std::iter::repeat_n("leftctx", 300).collect::<Vec<_>>().join(" ");
        let middle = std::iter::repeat_n("targetonly", 300).collect::<Vec<_>>().join(" ");
        let right = std::iter::repeat_n("rightctx", 300).collect::<Vec<_>>().join(" ");
        let markdown = format!("# Title\n\n{left}\n\n{middle}\n\n{right}\n");
        write_text_file(&work_path.join("docs/guide.md"), &markdown);
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "targetonly".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 5,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect("run keyword search");

        assert!(!response.results.is_empty(), "expected at least one result");
        let first = &response.results[0];
        assert!(first.text.contains("targetonly"));
        assert!(
            first.text.contains("leftctx"),
            "neighbor window should include previous chunk"
        );
        assert!(
            first.text.contains("rightctx"),
            "neighbor window should include next chunk"
        );
    });
}

#[test]
fn search_auto_mode_uses_keyword_path_and_scopes_space() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "api", notes_path.clone());

        write_text_file(&work_path.join("a.md"), "space scoped token\n");
        write_text_file(&notes_path.join("a.md"), "space scoped token\n");
        engine
            .update(update_options(None, &[]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "scoped".to_string(),
                mode: SearchMode::Auto,
                space: Some("work".to_string()),
                collections: vec![],
                limit: 10,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect("run auto search");

        assert_eq!(response.mode, SearchMode::Keyword);
        assert!(response.results.iter().all(|item| item.space == "work"));
    });
}

#[test]
fn search_rejects_unimplemented_modes_and_ambiguous_collection_scope() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "api", notes_path);

        let semantic_err = engine
            .search(SearchRequest {
                query: "test".to_string(),
                mode: SearchMode::Semantic,
                space: None,
                collections: vec![],
                limit: 10,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect_err("semantic mode should error");
        match KboltError::from(semantic_err) {
            KboltError::InvalidInput(message) => {
                assert!(message.contains("semantic mode"), "unexpected message: {message}");
            }
            other => panic!("unexpected error: {other}"),
        }

        let deep_err = engine
            .search(SearchRequest {
                query: "test".to_string(),
                mode: SearchMode::Deep,
                space: None,
                collections: vec![],
                limit: 10,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect_err("deep mode should error");
        match KboltError::from(deep_err) {
            KboltError::InvalidInput(message) => {
                assert!(message.contains("deep mode"), "unexpected message: {message}");
            }
            other => panic!("unexpected error: {other}"),
        }

        let ambiguous_err = engine
            .search(SearchRequest {
                query: "test".to_string(),
                mode: SearchMode::Keyword,
                space: None,
                collections: vec!["api".to_string()],
                limit: 10,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect_err("ambiguous collection should error");
        match KboltError::from(ambiguous_err) {
            KboltError::AmbiguousSpace { collection, spaces } => {
                assert_eq!(collection, "api");
                assert_eq!(spaces, vec!["notes".to_string(), "work".to_string()]);
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn resolve_update_targets_returns_all_collections_when_unscoped() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");

        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "wiki", notes_path);

        let targets = engine
            .resolve_update_targets(&update_options(None, &[]))
            .expect("resolve update targets");
        assert_eq!(targets.len(), 2);
        assert!(
            targets
                .iter()
                .any(|target| target.space == "work" && target.collection.name == "api")
        );
        assert!(
            targets
                .iter()
                .any(|target| target.space == "notes" && target.collection.name == "wiki")
        );
    });
}

#[test]
fn resolve_update_targets_scopes_to_requested_space() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");

        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "wiki", notes_path);

        let targets = engine
            .resolve_update_targets(&update_options(Some("work"), &[]))
            .expect("resolve update targets");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].space, "work");
        assert_eq!(targets[0].collection.name, "api");
    });
}

#[test]
fn resolve_update_targets_named_collection_uses_unique_lookup() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let targets = engine
            .resolve_update_targets(&update_options(None, &["api"]))
            .expect("resolve update targets");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].space, "work");
        assert_eq!(targets[0].collection.name, "api");
    });
}

#[test]
fn resolve_update_targets_named_collection_errors_on_ambiguity() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");

        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "api", notes_path);

        let err = engine
            .resolve_update_targets(&update_options(None, &["api"]))
            .expect_err("expected ambiguous collection");
        match KboltError::from(err) {
            KboltError::AmbiguousSpace { collection, spaces } => {
                assert_eq!(collection, "api");
                assert_eq!(spaces, vec!["notes".to_string(), "work".to_string()]);
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn resolve_update_targets_named_collection_honors_default_space_precedence() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "notes", "api", notes_path);

        let err = engine
            .resolve_update_targets(&update_options(None, &["api"]))
            .expect_err("default precedence should look in work first");
        match KboltError::from(err) {
            KboltError::CollectionNotFound { name } => assert_eq!(name, "api"),
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn resolve_update_targets_deduplicates_repeated_collection_names() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let targets = engine
            .resolve_update_targets(&update_options(Some("work"), &["api", "api"]))
            .expect("resolve update targets");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].space, "work");
        assert_eq!(targets[0].collection.name, "api");
    });
}

#[test]
fn resolve_update_targets_rejects_empty_collection_names() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        let err = engine
            .resolve_update_targets(&update_options(None, &[""]))
            .expect_err("empty collection names should be rejected");
        match KboltError::from(err) {
            KboltError::InvalidInput(message) => {
                assert!(message.contains("cannot be empty"), "unexpected message: {message}");
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn update_indexes_new_document_and_skips_unchanged_mtime() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("first update");
        assert_eq!(first.scanned, 1);
        assert_eq!(first.added, 1);
        assert_eq!(first.updated, 0);
        assert_eq!(first.deactivated, 0);
        assert!(first.errors.is_empty(), "unexpected errors: {:?}", first.errors);

        let hits = engine
            .storage()
            .query_bm25("work", "alpha", &[("body", 1.0)], 10)
            .expect("query bm25");
        assert_eq!(hits.len(), 1);

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("second update");
        assert_eq!(second.scanned, 1);
        assert_eq!(second.skipped_mtime, 1);
        assert_eq!(second.added, 0);
        assert_eq!(second.updated, 0);
        assert_eq!(second.deactivated, 0);
    });
}

#[test]
fn update_replays_fts_dirty_documents_before_mtime_fast_path() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("first update");
        assert_eq!(first.scanned, 1);
        assert_eq!(first.added, 1);
        assert!(first.errors.is_empty(), "unexpected errors: {:?}", first.errors);

        let work_space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        let stored_doc = engine
            .storage()
            .get_document_by_path(collection.id, "src/lib.rs")
            .expect("query document")
            .expect("document should exist");
        let chunks = engine
            .storage()
            .get_chunks_for_document(stored_doc.id)
            .expect("load chunks");
        let chunk_ids = chunks.iter().map(|chunk| chunk.id).collect::<Vec<_>>();
        assert!(!chunk_ids.is_empty(), "expected indexed chunks");

        engine
            .storage()
            .delete_tantivy("work", &chunk_ids)
            .expect("delete tantivy entries");
        engine
            .storage()
            .commit_tantivy("work")
            .expect("commit tantivy deletes");
        let removed_hits = engine
            .storage()
            .query_bm25("work", "alpha", &[("body", 1.0)], 10)
            .expect("query bm25 after delete");
        assert!(
            removed_hits.is_empty(),
            "search should be empty before replay, got {} hits",
            removed_hits.len()
        );

        engine
            .storage()
            .upsert_document(
                collection.id,
                &stored_doc.path,
                &stored_doc.title,
                &stored_doc.hash,
                &stored_doc.modified,
            )
            .expect("mark document fts dirty");
        let dirty_before = engine
            .storage()
            .get_fts_dirty_documents()
            .expect("load dirty docs");
        assert_eq!(dirty_before.len(), 1);
        assert_eq!(dirty_before[0].doc_id, stored_doc.id);

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("second update");
        assert!(second.errors.is_empty(), "unexpected errors: {:?}", second.errors);

        let replayed_hits = engine
            .storage()
            .query_bm25("work", "alpha", &[("body", 1.0)], 10)
            .expect("query bm25 after replay");
        assert_eq!(replayed_hits.len(), 1);
        assert_eq!(replayed_hits[0].chunk_id, chunk_ids[0]);

        let dirty_after = engine
            .storage()
            .get_fts_dirty_documents()
            .expect("load dirty docs after replay");
        assert!(
            dirty_after.is_empty(),
            "expected replay to clear fts_dirty flags"
        );
    });
}

#[test]
fn update_replay_skips_hash_mismatch_outside_scoped_targets() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "wiki", notes_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "worktoken\n");
        write_text_file(&notes_path.join("docs/guide.md"), "oldtoken\n");
        let first = engine
            .update(update_options(None, &[]))
            .expect("index initial fixtures");
        assert_eq!(first.added, 2);
        assert!(first.errors.is_empty(), "unexpected errors: {:?}", first.errors);

        let notes_space = engine.storage().get_space("notes").expect("get notes space");
        let notes_collection = engine
            .storage()
            .get_collection(notes_space.id, "wiki")
            .expect("get notes collection");
        let notes_doc = engine
            .storage()
            .get_document_by_path(notes_collection.id, "docs/guide.md")
            .expect("query notes document")
            .expect("notes document should exist");
        let note_chunks = engine
            .storage()
            .get_chunks_for_document(notes_doc.id)
            .expect("load note chunks");
        let note_chunk_ids = note_chunks.iter().map(|chunk| chunk.id).collect::<Vec<_>>();
        assert!(!note_chunk_ids.is_empty(), "expected note chunks");

        engine
            .storage()
            .delete_tantivy("notes", &note_chunk_ids)
            .expect("delete notes tantivy entries");
        engine
            .storage()
            .commit_tantivy("notes")
            .expect("commit notes tantivy deletes");
        engine
            .storage()
            .upsert_document(
                notes_collection.id,
                &notes_doc.path,
                &notes_doc.title,
                &notes_doc.hash,
                &notes_doc.modified,
            )
            .expect("mark notes document fts dirty");

        write_text_file(&notes_path.join("docs/guide.md"), "newtoken\n");

        let scoped = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("run scoped update");
        assert!(scoped.errors.is_empty(), "unexpected errors: {:?}", scoped.errors);

        let notes_new_hits = engine
            .storage()
            .query_bm25("notes", "newtoken", &[("body", 1.0)], 10)
            .expect("query notes bm25 for newtoken");
        assert!(
            notes_new_hits.is_empty(),
            "hash-mismatched replay should be skipped for out-of-scope collection"
        );
    });
}

#[test]
fn update_markdown_uses_structural_chunking_and_heading_metadata() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let repeated_words = std::iter::repeat_n("chunktoken", 900).collect::<Vec<_>>().join(" ");
        let markdown = format!("# Title\n\n{repeated_words}\n");
        let file_path = collection_path.join("docs/guide.md");
        write_text_file(&file_path, &markdown);

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update markdown");
        assert_eq!(report.scanned, 1);
        assert_eq!(report.added, 1);
        assert!(report.errors.is_empty(), "unexpected errors: {:?}", report.errors);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "docs/guide.md")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");

        assert!(chunks.len() >= 2, "expected markdown hard-split chunks");
        assert!(
            chunks.iter().any(|chunk| chunk.kind == "paragraph"),
            "expected paragraph chunk kind"
        );
        assert!(
            chunks.iter().any(|chunk| chunk.heading.as_deref() == Some("Title")),
            "expected heading breadcrumb on narrative chunks"
        );
    });
}

#[test]
fn update_code_files_use_code_chunking_profile() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let repeated_tokens = std::iter::repeat_n("ident", 700).collect::<Vec<_>>().join(" ");
        let source = format!("fn alpha() {{\n    {repeated_tokens}\n}}\n");
        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, &source);

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update code");
        assert_eq!(report.scanned, 1);
        assert_eq!(report.added, 1);
        assert!(report.errors.is_empty(), "unexpected errors: {:?}", report.errors);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "src/lib.rs")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");

        assert!(
            chunks.len() >= 2,
            "expected hard split from code profile (560 hard max)"
        );
        assert!(
            chunks.iter().all(|chunk| chunk.kind == "code"),
            "expected code chunk kind for code extractor output"
        );
    });
}

#[test]
fn update_code_uses_blank_line_grouping_before_token_fallback() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let g1 = std::iter::repeat_n("g1token", 240).collect::<Vec<_>>().join(" ");
        let g2 = std::iter::repeat_n("g2token", 240).collect::<Vec<_>>().join(" ");
        let g3 = std::iter::repeat_n("g3token", 240).collect::<Vec<_>>().join(" ");
        let source = format!("{g1}\n\n{g2}\n\n{g3}\n");
        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, &source);

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update code");
        assert_eq!(report.scanned, 1);
        assert_eq!(report.added, 1);
        assert!(report.errors.is_empty(), "unexpected errors: {:?}", report.errors);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "src/lib.rs")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");

        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().all(|chunk| chunk.kind == "code"));

        let bytes = std::fs::read(&file_path).expect("read source bytes");
        let first = {
            let start = chunks[0].offset.min(bytes.len());
            let end = chunks[0]
                .offset
                .saturating_add(chunks[0].length)
                .min(bytes.len());
            String::from_utf8_lossy(&bytes[start..end]).into_owned()
        };
        let second = {
            let start = chunks[1].offset.min(bytes.len());
            let end = chunks[1]
                .offset
                .saturating_add(chunks[1].length)
                .min(bytes.len());
            String::from_utf8_lossy(&bytes[start..end]).into_owned()
        };
        assert!(first.contains("g1token"));
        assert!(first.contains("g2token"));
        assert!(!first.contains("g3token"));
        assert!(second.contains("g3token"));
    });
}

#[test]
fn update_preserves_structural_boundaries_across_chunk_kinds() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let markdown = r#"# Intro

alpha beta

```rust
fn main() {}
```

gamma delta
"#;
        let file_path = collection_path.join("docs/guide.md");
        write_text_file(&file_path, markdown);

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update markdown");
        assert_eq!(report.scanned, 1);
        assert_eq!(report.added, 1);
        assert!(report.errors.is_empty(), "unexpected errors: {:?}", report.errors);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "docs/guide.md")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");
        let kinds = chunks.iter().map(|chunk| chunk.kind.as_str()).collect::<Vec<_>>();

        assert_eq!(kinds, vec!["section", "code", "paragraph"]);
        assert!(!kinds.contains(&"mixed"));
    });
}

#[test]
fn update_skips_hardcoded_ignored_paths() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&collection_path.join(".git/config"), "[core]\n");
        write_text_file(
            &collection_path.join("node_modules/pkg/index.js"),
            "module.exports = {};\n",
        );
        write_text_file(&collection_path.join(".DS_Store"), "ignored\n");
        write_text_file(&collection_path.join("Cargo.lock"), "ignored\n");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("run update");
        assert_eq!(report.scanned, 1);
        assert_eq!(report.added, 1);

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list indexed files");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/lib.rs");
    });
}

#[test]
fn update_applies_collection_ignore_patterns() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&collection_path.join("docs/guide.md"), "guide\n");

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        write_text_file(&ignore_path, "docs/**\n");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("run update");
        assert_eq!(report.scanned, 1);
        assert_eq!(report.added, 1);

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list indexed files");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/lib.rs");
    });
}

#[test]
fn update_tracks_modified_and_deactivated_documents() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "fn beta() {}\n");
        let changed = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("changed update");
        assert_eq!(changed.updated, 1);
        assert_eq!(changed.added, 0);
        assert_eq!(changed.deactivated, 0);

        std::fs::remove_file(&file_path).expect("remove file");
        let removed = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("deactivate removed file");
        assert_eq!(removed.deactivated, 1);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let docs = engine
            .storage()
            .list_documents(collection.id, false)
            .expect("list all documents");
        assert_eq!(docs.len(), 1);
        assert!(!docs[0].active, "removed document should be inactive");
    });
}

#[test]
fn update_dry_run_reports_changes_without_writing() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let mut options = update_options(Some("work"), &["api"]);
        options.dry_run = true;
        let report = engine.update(options).expect("dry run update");
        assert_eq!(report.scanned, 1);
        assert_eq!(report.added, 1);
        assert_eq!(report.updated, 0);
        assert_eq!(report.deactivated, 0);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let docs = engine
            .storage()
            .list_documents(collection.id, false)
            .expect("list all documents");
        assert!(docs.is_empty(), "dry run should not persist documents");
    });
}

#[test]
fn update_creates_global_lock_file() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");

        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("run update");

        let lock_path = engine.config().cache_dir.join("kbolt.lock");
        assert!(lock_path.exists(), "expected lock file at {}", lock_path.display());
    });
}

#[test]
fn update_fails_fast_when_global_lock_is_unavailable() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");

        let lock_path = engine.config().cache_dir.join("kbolt.lock");
        std::fs::create_dir_all(&engine.config().cache_dir).expect("create cache dir");
        let holder = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .expect("open lock file");
        FileExt::try_lock_exclusive(&holder).expect("acquire lock in test");

        let err = engine
            .update(update_options(Some("work"), &["api"]))
            .expect_err("update should fail while lock is held");
        match KboltError::from(err) {
            KboltError::Internal(message) => {
                assert!(
                    message.contains("Another kbolt process is active. Try again shortly."),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn add_space_fails_fast_when_global_lock_is_unavailable() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);

        let lock_path = engine.config().cache_dir.join("kbolt.lock");
        std::fs::create_dir_all(&engine.config().cache_dir).expect("create cache dir");
        let holder = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .expect("open lock file");
        FileExt::try_lock_exclusive(&holder).expect("acquire lock in test");

        let err = engine
            .add_space("work", None)
            .expect_err("add_space should fail while lock is held");
        match KboltError::from(err) {
            KboltError::Internal(message) => {
                assert!(
                    message.contains("Another kbolt process is active. Try again shortly."),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn status_reports_space_collection_and_model_counts() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine
            .add_space("work", Some("work docs"))
            .expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work collection dir");
        std::fs::create_dir_all(&notes_path).expect("create notes collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "wiki", notes_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&work_path.join("README.md"), "# docs\n");
        write_text_file(&notes_path.join("notes.md"), "meeting notes\n");

        engine
            .update(update_options(None, &[]))
            .expect("initial update");

        let work_space = engine.storage().get_space("work").expect("get work space");
        let work_collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get work collection");
        let work_active_docs = engine
            .storage()
            .list_documents(work_collection.id, true)
            .expect("list active work docs");
        engine
            .storage()
            .deactivate_document(work_active_docs[0].id)
            .expect("deactivate one work doc");

        let status = engine.status(None).expect("get global status");

        assert_eq!(status.cache_dir, engine.config().cache_dir);
        assert_eq!(status.config_dir, engine.config().config_dir);
        assert_eq!(status.total_documents, 3);
        assert_eq!(status.total_documents, engine.storage().count_documents(None).unwrap());
        assert_eq!(status.total_chunks, engine.storage().count_chunks(None).unwrap());
        assert_eq!(
            status.total_embedded,
            engine.storage().count_embedded_chunks(None).unwrap()
        );

        assert_eq!(status.models.embedder.name, engine.config().models.embedder.id);
        assert_eq!(status.models.reranker.name, engine.config().models.reranker.id);
        assert_eq!(status.models.expander.name, engine.config().models.expander.id);
        assert!(!status.models.embedder.downloaded);
        assert!(!status.models.reranker.downloaded);
        assert!(!status.models.expander.downloaded);
        assert_eq!(status.models.embedder.size_bytes, None);
        assert_eq!(status.models.reranker.size_bytes, None);
        assert_eq!(status.models.expander.size_bytes, None);
        let expected_embedder = engine.config().cache_dir.join("models/embedder");
        let expected_reranker = engine.config().cache_dir.join("models/reranker");
        let expected_expander = engine.config().cache_dir.join("models/expander");
        assert_eq!(
            status.models.embedder.path.as_deref(),
            Some(expected_embedder.as_path())
        );
        assert_eq!(
            status.models.reranker.path.as_deref(),
            Some(expected_reranker.as_path())
        );
        assert_eq!(
            status.models.expander.path.as_deref(),
            Some(expected_expander.as_path())
        );

        let default_status = status
            .spaces
            .iter()
            .find(|space| space.name == "default")
            .expect("default space status should exist");
        assert!(default_status.collections.is_empty());
        assert_eq!(default_status.last_updated, None);

        let work_status = status
            .spaces
            .iter()
            .find(|space| space.name == "work")
            .expect("work status should exist");
        assert_eq!(work_status.description.as_deref(), Some("work docs"));
        assert_eq!(work_status.collections.len(), 1);
        assert!(work_status.last_updated.is_some());

        let work_collection_status = &work_status.collections[0];
        assert_eq!(work_collection_status.name, "api");
        assert_eq!(work_collection_status.path, work_path);
        assert_eq!(
            work_collection_status.documents,
            engine
                .storage()
                .count_documents_in_collection(work_collection.id, false)
                .unwrap()
        );
        assert_eq!(
            work_collection_status.active_documents,
            engine
                .storage()
                .count_documents_in_collection(work_collection.id, true)
                .unwrap()
        );
        assert_eq!(
            work_collection_status.chunks,
            engine
                .storage()
                .count_chunks_in_collection(work_collection.id)
                .unwrap()
        );
        assert_eq!(
            work_collection_status.embedded_chunks,
            engine
                .storage()
                .count_embedded_chunks_in_collection(work_collection.id)
                .unwrap()
        );
        assert_eq!(
            work_status.last_updated.as_deref(),
            Some(work_collection_status.last_updated.as_str())
        );

        let notes_space = engine.storage().get_space("notes").expect("get notes space");
        let notes_collection = engine
            .storage()
            .get_collection(notes_space.id, "wiki")
            .expect("get notes collection");
        let notes_status = status
            .spaces
            .iter()
            .find(|space| space.name == "notes")
            .expect("notes status should exist");
        assert_eq!(notes_status.collections.len(), 1);
        assert!(notes_status.last_updated.is_some());

        let notes_collection_status = &notes_status.collections[0];
        assert_eq!(notes_collection_status.name, "wiki");
        assert_eq!(notes_collection_status.path, notes_path);
        assert_eq!(
            notes_collection_status.documents,
            engine
                .storage()
                .count_documents_in_collection(notes_collection.id, false)
                .unwrap()
        );
        assert_eq!(
            notes_collection_status.active_documents,
            engine
                .storage()
                .count_documents_in_collection(notes_collection.id, true)
                .unwrap()
        );
        assert_eq!(
            notes_collection_status.chunks,
            engine
                .storage()
                .count_chunks_in_collection(notes_collection.id)
                .unwrap()
        );
        assert_eq!(
            notes_collection_status.embedded_chunks,
            engine
                .storage()
                .count_embedded_chunks_in_collection(notes_collection.id)
                .unwrap()
        );
        assert_eq!(
            notes_status.last_updated.as_deref(),
            Some(notes_collection_status.last_updated.as_str())
        );
    });
}

#[test]
fn status_scopes_to_requested_space() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work collection dir");
        std::fs::create_dir_all(&notes_path).expect("create notes collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "wiki", notes_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&notes_path.join("notes.md"), "meeting notes\n");
        engine
            .update(update_options(None, &[]))
            .expect("initial update");

        let work_space = engine.storage().get_space("work").expect("get work space");
        let scoped = engine.status(Some("work")).expect("get scoped status");

        assert_eq!(scoped.spaces.len(), 1);
        assert_eq!(scoped.spaces[0].name, "work");
        assert_eq!(
            scoped.total_documents,
            engine.storage().count_documents(Some(work_space.id)).unwrap()
        );
        assert_eq!(
            scoped.total_chunks,
            engine.storage().count_chunks(Some(work_space.id)).unwrap()
        );
        assert_eq!(
            scoped.total_embedded,
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .unwrap()
        );
    });
}

#[test]
fn status_errors_for_missing_space_scope() {
    let engine = test_engine_with_default_space(None);
    let err = engine
        .status(Some("missing"))
        .expect_err("missing status scope should error");
    match KboltError::from(err) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn model_status_reflects_configured_model_names() {
    let engine = test_engine_with_default_space(None);
    let status = engine.model_status().expect("read model status");

    assert_eq!(status.embedder.name, "embed-model");
    assert_eq!(status.reranker.name, "reranker-model");
    assert_eq!(status.expander.name, "expander-model");
    assert!(!status.embedder.downloaded);
    assert!(!status.reranker.downloaded);
    assert!(!status.expander.downloaded);
    assert_eq!(status.embedder.size_bytes, None);
    assert_eq!(status.reranker.size_bytes, None);
    assert_eq!(status.expander.size_bytes, None);
    let expected_embedder = engine.config().cache_dir.join("models/embedder");
    let expected_reranker = engine.config().cache_dir.join("models/reranker");
    let expected_expander = engine.config().cache_dir.join("models/expander");
    assert_eq!(
        status.embedder.path.as_deref(),
        Some(expected_embedder.as_path())
    );
    assert_eq!(
        status.reranker.path.as_deref(),
        Some(expected_reranker.as_path())
    );
    assert_eq!(
        status.expander.path.as_deref(),
        Some(expected_expander.as_path())
    );
}

#[test]
fn pull_models_skips_already_present_model_directories() {
    let engine = test_engine();
    let model_dir = engine.config().cache_dir.join("models");
    let models = &engine.config().models;
    seed_model_artifact(&model_dir, "embedder", &models.embedder, b"e");
    seed_model_artifact(&model_dir, "reranker", &models.reranker, b"r");
    seed_model_artifact(&model_dir, "expander", &models.expander, b"x");

    let report = engine.pull_models().expect("pull models");
    assert_eq!(report.downloaded.len(), 0);
    assert_eq!(report.already_present.len(), 3);
    assert_eq!(report.total_bytes, 0);
}

#[test]
fn pull_models_with_progress_emits_already_present_events() {
    let engine = test_engine();
    let model_dir = engine.config().cache_dir.join("models");
    let models = &engine.config().models;
    seed_model_artifact(&model_dir, "embedder", &models.embedder, b"e");
    seed_model_artifact(&model_dir, "reranker", &models.reranker, b"r");
    seed_model_artifact(&model_dir, "expander", &models.expander, b"x");

    let mut events = Vec::new();
    let report = engine
        .pull_models_with_progress(|event| events.push(event))
        .expect("pull models with progress");
    assert_eq!(report.downloaded.len(), 0);
    assert_eq!(report.already_present.len(), 3);

    assert_eq!(
        events,
        vec![
            ModelPullEvent::AlreadyPresent {
                role: "embedder".to_string(),
                model: "embed-model".to_string(),
                bytes: 1,
            },
            ModelPullEvent::AlreadyPresent {
                role: "reranker".to_string(),
                model: "reranker-model".to_string(),
                bytes: 1,
            },
            ModelPullEvent::AlreadyPresent {
                role: "expander".to_string(),
                model: "expander-model".to_string(),
                bytes: 1,
            },
        ]
    );
}
