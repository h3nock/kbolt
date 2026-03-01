use tempfile::tempdir;

use super::Storage;
use kbolt_types::KboltError;
use rusqlite::Connection;

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

#[test]
fn new_creates_expected_schema_objects() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    Storage::new(&cache_dir).expect("create storage");

    let db_path = cache_dir.join("meta.sqlite");
    let conn = Connection::open(db_path).expect("open sqlite");

    let tables = [
        "spaces",
        "collections",
        "documents",
        "chunks",
        "embeddings",
        "llm_cache",
    ];
    for table in tables {
        let exists = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = ?1",
                [table],
                |row| row.get::<_, i64>(0),
            )
            .expect("query sqlite_master");
        assert_eq!(exists, 1, "missing expected table: {table}");
    }

    let indexes = [
        "idx_documents_collection",
        "idx_documents_hash",
        "idx_chunks_doc",
    ];
    for index in indexes {
        let exists = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name = ?1",
                [index],
                |row| row.get::<_, i64>(0),
            )
            .expect("query sqlite_master for index");
        assert_eq!(exists, 1, "missing expected index: {index}");
    }
}
