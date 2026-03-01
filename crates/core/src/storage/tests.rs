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

#[test]
fn delete_space_removes_non_default_space() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage
        .create_space("work", Some("work docs"))
        .expect("create space");

    storage.delete_space("work").expect("delete space");
    let err = storage
        .get_space("work")
        .expect_err("deleted space should not exist");

    match err {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "work"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn delete_default_space_clears_contents_but_keeps_space() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let conn = storage.db.lock().expect("lock db");

    let default_id: i64 = conn
        .query_row("SELECT id FROM spaces WHERE name = 'default'", [], |row| {
            row.get(0)
        })
        .expect("query default space id");
    conn.execute(
        "INSERT INTO collections (space_id, name, path, description, extensions, created, updated)
         VALUES (?1, ?2, ?3, ?4, NULL, strftime('%Y-%m-%dT%H:%M:%SZ','now'), strftime('%Y-%m-%dT%H:%M:%SZ','now'))",
        rusqlite::params![default_id, "col1", "/tmp/col1", "test"],
    )
    .expect("insert collection");
    drop(conn);

    storage
        .delete_space("default")
        .expect("clear default space contents");

    let default_space = storage
        .get_space("default")
        .expect("default space should still exist");
    assert_eq!(default_space.name, "default");

    let conn = storage.db.lock().expect("lock db");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM collections WHERE space_id = ?1",
            [default_space.id],
            |row| row.get(0),
        )
        .expect("count collections");
    assert_eq!(count, 0, "default space collections should be cleared");
}

#[test]
fn rename_space_rejects_default_space() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .rename_space("default", "renamed")
        .expect_err("renaming default should fail");
    assert!(
        err.to_string().contains("cannot rename reserved space"),
        "unexpected error: {err}"
    );
}

#[test]
fn rename_space_updates_name_for_non_default_space() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage
        .rename_space("work", "team")
        .expect("rename work to team");

    let renamed = storage.get_space("team").expect("get renamed space");
    assert_eq!(renamed.name, "team");

    let err = storage
        .get_space("work")
        .expect_err("old name should not resolve");
    match err {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "work"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn rename_space_to_existing_name_returns_space_already_exists() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");
    storage.create_space("notes", None).expect("create notes");

    let err = storage
        .rename_space("work", "notes")
        .expect_err("duplicate name should fail");
    match err {
        KboltError::SpaceAlreadyExists { name } => assert_eq!(name, "notes"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn update_space_description_persists_new_value() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage
        .update_space_description("work", "updated description")
        .expect("update description");

    let space = storage.get_space("work").expect("get work");
    assert_eq!(space.description.as_deref(), Some("updated description"));
}

#[test]
fn update_space_description_missing_space_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .update_space_description("missing", "desc")
        .expect_err("missing space should fail");
    match err {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}
