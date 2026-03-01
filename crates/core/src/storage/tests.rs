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

    match KboltError::from(err) {
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

    match KboltError::from(err) {
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

    match KboltError::from(err) {
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
    match KboltError::from(err) {
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
    match KboltError::from(err) {
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
    match KboltError::from(err) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn create_get_and_list_collections_in_space() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let work_space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let extensions = vec!["rs".to_string(), "md".to_string()];

    let collection_id = storage
        .create_collection(
            work_space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            Some("API docs"),
            Some(&extensions),
        )
        .expect("create collection");
    assert!(collection_id > 0);

    let collection = storage
        .get_collection(work_space_id, "api")
        .expect("get collection");
    assert_eq!(collection.name, "api");
    assert_eq!(collection.space_id, work_space_id);
    assert_eq!(collection.path, std::path::PathBuf::from("/tmp/api"));
    assert_eq!(collection.description.as_deref(), Some("API docs"));
    assert_eq!(collection.extensions, Some(extensions.clone()));

    let in_space = storage
        .list_collections(Some(work_space_id))
        .expect("list collections in space");
    assert_eq!(in_space.len(), 1);
    assert_eq!(in_space[0].name, "api");

    let across_all = storage
        .list_collections(None)
        .expect("list collections across all spaces");
    assert_eq!(across_all.len(), 1);
    assert_eq!(across_all[0].name, "api");
}

#[test]
fn create_collection_duplicate_name_in_space_returns_collection_already_exists() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let work_space_id = storage
        .create_space("work", None)
        .expect("create work space");

    storage
        .create_collection(
            work_space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create first collection");

    let err = storage
        .create_collection(
            work_space_id,
            "api",
            std::path::Path::new("/tmp/api-v2"),
            None,
            None,
        )
        .expect_err("duplicate collection should fail");
    match KboltError::from(err) {
        KboltError::CollectionAlreadyExists { name, space } => {
            assert_eq!(name, "api");
            assert_eq!(space, "work");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn create_collection_in_missing_space_returns_space_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .create_collection(99999, "api", std::path::Path::new("/tmp/api"), None, None)
        .expect_err("missing space should fail");

    match KboltError::from(err) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "id=99999"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn get_collection_missing_name_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let work_space_id = storage
        .create_space("work", None)
        .expect("create work space");

    let err = storage
        .get_collection(work_space_id, "missing")
        .expect_err("missing collection should fail");
    match KboltError::from(err) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn delete_collection_removes_entry() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    storage
        .delete_collection(space_id, "api")
        .expect("delete collection");

    let err = storage
        .get_collection(space_id, "api")
        .expect_err("collection should not exist");
    match KboltError::from(err) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "api"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn delete_collection_missing_name_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");

    let err = storage
        .delete_collection(space_id, "missing")
        .expect_err("missing collection should fail");
    match KboltError::from(err) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn rename_collection_updates_name() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    storage
        .rename_collection(space_id, "api", "backend")
        .expect("rename collection");

    let renamed = storage
        .get_collection(space_id, "backend")
        .expect("get renamed collection");
    assert_eq!(renamed.name, "backend");
    let err = storage
        .get_collection(space_id, "api")
        .expect_err("old name should not exist");
    match KboltError::from(err) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "api"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn rename_collection_to_existing_name_returns_already_exists() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create api");
    storage
        .create_collection(
            space_id,
            "backend",
            std::path::Path::new("/tmp/backend"),
            None,
            None,
        )
        .expect("create backend");

    let err = storage
        .rename_collection(space_id, "api", "backend")
        .expect_err("rename duplicate should fail");
    match KboltError::from(err) {
        KboltError::CollectionAlreadyExists { name, space } => {
            assert_eq!(name, "backend");
            assert_eq!(space, "work");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn update_collection_description_persists_value() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    storage
        .update_collection_description(space_id, "api", "API docs")
        .expect("update description");

    let updated = storage
        .get_collection(space_id, "api")
        .expect("get updated collection");
    assert_eq!(updated.description.as_deref(), Some("API docs"));
}

#[test]
fn update_collection_description_missing_name_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");

    let err = storage
        .update_collection_description(space_id, "missing", "desc")
        .expect_err("missing collection should fail");
    match KboltError::from(err) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn update_collection_timestamp_refreshes_updated_column() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let collection_id = storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    {
        let conn = storage.db.lock().expect("lock db");
        conn.execute(
            "UPDATE collections SET updated = '2000-01-01T00:00:00Z' WHERE id = ?1",
            [collection_id],
        )
        .expect("set old timestamp");
    }

    storage
        .update_collection_timestamp(collection_id)
        .expect("update timestamp");

    let refreshed = storage
        .get_collection(space_id, "api")
        .expect("get collection");
    assert_ne!(refreshed.updated, "2000-01-01T00:00:00Z");
}

#[test]
fn update_collection_timestamp_missing_collection_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .update_collection_timestamp(99999)
        .expect_err("missing collection should fail");
    match KboltError::from(err) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "id=99999"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn upsert_document_inserts_active_dirty_document() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let collection_id = storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    let doc_id = storage
        .upsert_document(
            collection_id,
            "src/lib.rs",
            "lib",
            "hash-1",
            "2026-03-01T10:00:00Z",
        )
        .expect("upsert document");
    assert!(doc_id > 0);

    let stored = storage
        .get_document_by_path(collection_id, "src/lib.rs")
        .expect("get document")
        .expect("document should exist");
    assert_eq!(stored.id, doc_id);
    assert_eq!(stored.title, "lib");
    assert_eq!(stored.hash, "hash-1");
    assert_eq!(stored.modified, "2026-03-01T10:00:00Z");
    assert!(stored.active);
    assert!(stored.fts_dirty);
    assert_eq!(stored.deactivated_at, None);
}

#[test]
fn upsert_document_updates_existing_row_and_reactivates() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let collection_id = storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    let doc_id_1 = storage
        .upsert_document(
            collection_id,
            "src/lib.rs",
            "lib",
            "hash-1",
            "2026-03-01T10:00:00Z",
        )
        .expect("first upsert");

    {
        let conn = storage.db.lock().expect("lock db");
        conn.execute(
            "UPDATE documents
             SET active = 0, deactivated_at = '2026-03-01T11:00:00Z', fts_dirty = 0
             WHERE id = ?1",
            [doc_id_1],
        )
        .expect("mark inactive");
    }

    let doc_id_2 = storage
        .upsert_document(
            collection_id,
            "src/lib.rs",
            "lib-updated",
            "hash-2",
            "2026-03-01T12:00:00Z",
        )
        .expect("second upsert");
    assert_eq!(doc_id_1, doc_id_2, "upsert should preserve row identity");

    let stored = storage
        .get_document_by_path(collection_id, "src/lib.rs")
        .expect("get document")
        .expect("document should exist");
    assert_eq!(stored.title, "lib-updated");
    assert_eq!(stored.hash, "hash-2");
    assert_eq!(stored.modified, "2026-03-01T12:00:00Z");
    assert!(stored.active);
    assert_eq!(stored.deactivated_at, None);
    assert!(stored.fts_dirty);
}

#[test]
fn upsert_document_missing_collection_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .upsert_document(99999, "src/lib.rs", "lib", "hash", "2026-03-01T10:00:00Z")
        .expect_err("missing collection should fail");
    match KboltError::from(err) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "id=99999"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn get_document_by_path_missing_path_returns_none() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let collection_id = storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    let missing = storage
        .get_document_by_path(collection_id, "missing.rs")
        .expect("query missing path");
    assert!(missing.is_none());
}

#[test]
fn get_document_by_path_missing_collection_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .get_document_by_path(99999, "src/lib.rs")
        .expect_err("missing collection should fail");
    match KboltError::from(err) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "id=99999"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn list_documents_respects_active_only_filter() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let collection_id = storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    let active_doc_id = storage
        .upsert_document(collection_id, "a.rs", "a", "hash-a", "2026-03-01T10:00:00Z")
        .expect("insert active doc");
    let inactive_doc_id = storage
        .upsert_document(collection_id, "b.rs", "b", "hash-b", "2026-03-01T11:00:00Z")
        .expect("insert inactive doc");
    assert!(active_doc_id > 0);

    {
        let conn = storage.db.lock().expect("lock db");
        conn.execute(
            "UPDATE documents
             SET active = 0, deactivated_at = '2026-03-01T12:00:00Z'
             WHERE id = ?1",
            [inactive_doc_id],
        )
        .expect("mark inactive");
    }

    let all_docs = storage
        .list_documents(collection_id, false)
        .expect("list all documents");
    let active_docs = storage
        .list_documents(collection_id, true)
        .expect("list active documents");

    assert_eq!(all_docs.len(), 2);
    assert_eq!(active_docs.len(), 1);
    assert_eq!(active_docs[0].path, "a.rs");
}

#[test]
fn list_documents_missing_collection_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .list_documents(99999, true)
        .expect_err("missing collection should fail");
    match KboltError::from(err) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "id=99999"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn get_document_by_hash_prefix_returns_matching_rows() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let collection_id = storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    let _doc1 = storage
        .upsert_document(
            collection_id,
            "src/a.rs",
            "a",
            "abcd0001",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc1");
    let _doc2 = storage
        .upsert_document(
            collection_id,
            "src/b.rs",
            "b",
            "abcd0002",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert doc2");
    let _doc3 = storage
        .upsert_document(
            collection_id,
            "src/c.rs",
            "c",
            "ffff0003",
            "2026-03-01T10:02:00Z",
        )
        .expect("insert doc3");

    let matches = storage
        .get_document_by_hash_prefix("abcd")
        .expect("query prefix");
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].hash, "abcd0001");
    assert_eq!(matches[1].hash, "abcd0002");
}

#[test]
fn deactivate_document_marks_document_inactive() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let collection_id = storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    let doc_id = storage
        .upsert_document(
            collection_id,
            "src/lib.rs",
            "lib",
            "hash-1",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc");

    storage
        .deactivate_document(doc_id)
        .expect("deactivate document");

    let stored = storage
        .get_document_by_path(collection_id, "src/lib.rs")
        .expect("get document")
        .expect("document exists");
    assert!(!stored.active);
    assert!(stored.deactivated_at.is_some());
}

#[test]
fn deactivate_document_missing_id_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .deactivate_document(99999)
        .expect_err("missing document should fail");
    match KboltError::from(err) {
        KboltError::DocumentNotFound { path } => assert_eq!(path, "id=99999"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn reactivate_document_marks_document_active_and_clears_deactivated_at() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let collection_id = storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/api"),
            None,
            None,
        )
        .expect("create collection");

    let doc_id = storage
        .upsert_document(
            collection_id,
            "src/lib.rs",
            "lib",
            "hash-1",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc");
    storage
        .deactivate_document(doc_id)
        .expect("deactivate document");

    storage
        .reactivate_document(doc_id)
        .expect("reactivate document");

    let stored = storage
        .get_document_by_path(collection_id, "src/lib.rs")
        .expect("get document")
        .expect("document exists");
    assert!(stored.active);
    assert_eq!(stored.deactivated_at, None);
}

#[test]
fn reactivate_document_missing_id_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .reactivate_document(99999)
        .expect_err("missing document should fail");
    match KboltError::from(err) {
        KboltError::DocumentNotFound { path } => assert_eq!(path, "id=99999"),
        other => panic!("unexpected error: {other}"),
    }
}
