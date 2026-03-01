use tempfile::tempdir;

use crate::config::{Config, ModelConfig, ReapingConfig};
use crate::engine::Engine;
use crate::storage::Storage;
use kbolt_types::{AddCollectionRequest, KboltError};

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
    let engine = test_engine_with_default_space(Some("work"));
    engine.add_space("work", None).expect("add work");

    let resolved = engine.resolve_space(None).expect("resolve default space");
    assert_eq!(resolved, "work");
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
