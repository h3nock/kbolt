use tempfile::tempdir;
use std::ffi::OsString;
use std::sync::{Mutex, OnceLock};

use crate::config::{Config, ModelConfig, ReapingConfig};
use crate::engine::Engine;
use crate::storage::Storage;
use kbolt_types::{ActiveSpaceSource, AddCollectionRequest, KboltError, UpdateOptions};

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
            embed: "embed-model".to_string(),
            reranker: "reranker-model".to_string(),
            expander: "expander-model".to_string(),
        },
        reaping: ReapingConfig { days: 7 },
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
            embed: "embed-model".to_string(),
            reranker: "reranker-model".to_string(),
            expander: "expander-model".to_string(),
        },
        reaping: ReapingConfig { days: 7 },
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
        engine.config().models.embed,
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
fn add_collection_without_no_index_returns_explicit_error_and_does_not_create_row() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    let root = tempdir().expect("create temp root");
    let collection_path = root.path().join("api");
    std::fs::create_dir_all(&collection_path).expect("create collection dir");

    let err = engine
        .add_collection(AddCollectionRequest {
            path: collection_path,
            space: Some("work".to_string()),
            name: Some("api".to_string()),
            description: None,
            extensions: None,
            no_index: false,
        })
        .expect_err("collection add should fail without --no-index until update is wired");
    assert!(
        err.to_string()
            .contains("automatic indexing on collection add is not wired yet"),
        "unexpected error: {err}"
    );

    let collections = engine
        .list_collections(Some("work"))
        .expect("list collections");
    assert!(collections.is_empty(), "collection should not have been created");
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

    engine
        .remove_collection(Some("work"), "backend")
        .expect("remove collection");
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

        assert_eq!(status.models.embedder.name, engine.config().models.embed);
        assert_eq!(status.models.reranker.name, engine.config().models.reranker);
        assert_eq!(status.models.expander.name, engine.config().models.expander);
        assert!(!status.models.embedder.downloaded);
        assert!(!status.models.reranker.downloaded);
        assert!(!status.models.expander.downloaded);
        assert_eq!(status.models.embedder.size_bytes, None);
        assert_eq!(status.models.reranker.size_bytes, None);
        assert_eq!(status.models.expander.size_bytes, None);
        assert_eq!(status.models.embedder.path, None);
        assert_eq!(status.models.reranker.path, None);
        assert_eq!(status.models.expander.path, None);

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
