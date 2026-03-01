use tempfile::tempdir;

use super::Storage;
use kbolt_types::KboltError;

#[test]
fn new_creates_db_and_default_space() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");

    let storage = Storage::new(&cache_dir).expect("create storage");

    assert!(cache_dir.join("meta.sqlite").exists());

    let default_space = storage
        .get_space("default")
        .expect("default space should exist");
    assert_eq!(default_space.name, "default");
}

#[test]
fn create_and_list_spaces() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let space_id = storage
        .create_space("work", Some("Work documents"))
        .expect("create space");
    assert!(space_id > 0);

    let names: Vec<String> = storage
        .list_spaces()
        .expect("list spaces")
        .into_iter()
        .map(|space| space.name)
        .collect();

    assert_eq!(names, vec!["default".to_string(), "work".to_string()]);
}

#[test]
fn create_space_duplicate_returns_space_already_exists() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    storage
        .create_space("work", None)
        .expect("first create succeeds");
    let err = storage
        .create_space("work", None)
        .expect_err("duplicate create should fail");

    match err {
        KboltError::SpaceAlreadyExists { name } => assert_eq!(name, "work"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn get_space_returns_not_found_for_missing_name() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .get_space("missing")
        .expect_err("missing space should fail");

    match err {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}
