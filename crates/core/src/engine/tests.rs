use tempfile::tempdir;

use crate::config::{Config, ModelConfig, ReapingConfig};
use crate::engine::Engine;
use crate::storage::Storage;
use kbolt_types::KboltError;

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
