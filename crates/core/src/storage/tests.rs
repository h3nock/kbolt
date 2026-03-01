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
fn new_preloads_default_space_index_paths() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");

    assert!(cache_dir.join("spaces/default/tantivy").is_dir());
    assert!(cache_dir.join("spaces/default/tantivy/meta.json").is_file());
    assert!(cache_dir.join("spaces/default/vectors.usearch").is_file());

    let spaces = storage.spaces.read().expect("lock spaces map");
    assert!(spaces.contains_key("default"));
}

#[test]
fn open_space_registers_paths_for_existing_space() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage.open_space("work").expect("open work space");
    assert!(cache_dir.join("spaces/work/tantivy").is_dir());
    assert!(cache_dir.join("spaces/work/tantivy/meta.json").is_file());
    assert!(cache_dir.join("spaces/work/vectors.usearch").is_file());

    let spaces = storage.spaces.read().expect("lock spaces map");
    assert!(spaces.contains_key("work"));
}

#[test]
fn open_space_missing_space_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .open_space("missing")
        .expect_err("missing space should fail");
    match KboltError::from(err) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn close_space_removes_loaded_space_entry() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");
    storage.open_space("work").expect("open work");

    storage.close_space("work").expect("close work");
    let spaces = storage.spaces.read().expect("lock spaces map");
    assert!(!spaces.contains_key("work"));
}

#[test]
fn close_space_missing_space_returns_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let err = storage
        .close_space("missing")
        .expect_err("missing space should fail");
    match KboltError::from(err) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
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
fn create_space_provisions_index_paths() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");

    storage.create_space("work", None).expect("create work space");
    assert!(cache_dir.join("spaces/work/tantivy").is_dir());
    assert!(cache_dir.join("spaces/work/tantivy/meta.json").is_file());
    assert!(cache_dir.join("spaces/work/vectors.usearch").is_file());

    let spaces = storage.spaces.read().expect("lock spaces map");
    assert!(spaces.contains_key("work"));
}

#[test]
fn open_space_initializes_expected_tantivy_schema_fields() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    storage.create_space("work", None).expect("create work");
    storage.open_space("work").expect("open work");

    let index = tantivy::Index::open_in_dir(cache_dir.join("spaces/work/tantivy"))
        .expect("open tantivy index");
    let schema = index.schema();
    assert!(schema.get_field("chunk_id").is_ok());
    assert!(schema.get_field("doc_id").is_ok());
    assert!(schema.get_field("filepath").is_ok());
    assert!(schema.get_field("title").is_ok());
    assert!(schema.get_field("heading").is_ok());
    assert!(schema.get_field("body").is_ok());
}

#[test]
fn find_space_for_collection_returns_not_found_when_absent() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let resolution = storage
        .find_space_for_collection("missing")
        .expect("resolve collection");
    assert_eq!(resolution, super::SpaceResolution::NotFound);
}

#[test]
fn find_space_for_collection_returns_found_for_unique_match() {
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
        .expect("create collection");

    let resolution = storage
        .find_space_for_collection("api")
        .expect("resolve collection");
    match resolution {
        super::SpaceResolution::Found(space) => assert_eq!(space.name, "work"),
        other => panic!("unexpected resolution: {other:?}"),
    }
}

#[test]
fn find_space_for_collection_returns_ambiguous_with_sorted_space_names() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let zebra_space_id = storage
        .create_space("zebra", None)
        .expect("create zebra space");
    let alpha_space_id = storage
        .create_space("alpha", None)
        .expect("create alpha space");
    storage
        .create_collection(
            zebra_space_id,
            "api",
            std::path::Path::new("/tmp/zebra-api"),
            None,
            None,
        )
        .expect("create zebra collection");
    storage
        .create_collection(
            alpha_space_id,
            "api",
            std::path::Path::new("/tmp/alpha-api"),
            None,
            None,
        )
        .expect("create alpha collection");

    let resolution = storage
        .find_space_for_collection("api")
        .expect("resolve collection");
    match resolution {
        super::SpaceResolution::Ambiguous(spaces) => {
            assert_eq!(spaces, vec!["alpha".to_string(), "zebra".to_string()]);
        }
        other => panic!("unexpected resolution: {other:?}"),
    }
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
    let cache_dir = tmp.path().join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    storage
        .create_space("work", Some("work docs"))
        .expect("create space");
    assert!(cache_dir.join("spaces/work/tantivy").is_dir());

    storage.delete_space("work").expect("delete space");
    let err = storage
        .get_space("work")
        .expect_err("deleted space should not exist");

    match KboltError::from(err) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "work"),
        other => panic!("unexpected error: {other}"),
    }

    assert!(!cache_dir.join("spaces/work").exists());
    let spaces = storage.spaces.read().expect("lock spaces map");
    assert!(!spaces.contains_key("work"));
}

#[test]
fn delete_default_space_clears_contents_but_keeps_space() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    std::fs::write(
        cache_dir.join("spaces/default/tantivy/marker.bin"),
        vec![b'm'; 8],
    )
    .expect("write marker file");
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

    assert!(cache_dir.join("spaces/default/tantivy").is_dir());
    assert!(cache_dir.join("spaces/default/vectors.usearch").is_file());
    assert!(!cache_dir.join("spaces/default/tantivy/marker.bin").exists());
    let spaces = storage.spaces.read().expect("lock spaces map");
    assert!(spaces.contains_key("default"));
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
    let cache_dir = tmp.path().join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    storage.create_space("work", None).expect("create work");
    std::fs::write(
        cache_dir.join("spaces/work/tantivy/marker.bin"),
        vec![b'm'; 8],
    )
    .expect("write marker");

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

    assert!(!cache_dir.join("spaces/work").exists());
    assert!(cache_dir.join("spaces/team/tantivy").is_dir());
    assert!(cache_dir.join("spaces/team/vectors.usearch").is_file());
    assert!(cache_dir.join("spaces/team/tantivy/marker.bin").is_file());

    let spaces = storage.spaces.read().expect("lock spaces map");
    assert!(!spaces.contains_key("work"));
    assert!(spaces.contains_key("team"));
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
fn rename_space_rolls_back_when_destination_artifacts_exist() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    storage.create_space("work", None).expect("create work");
    std::fs::create_dir_all(cache_dir.join("spaces/team/tantivy"))
        .expect("create stale destination dir");

    let err = storage
        .rename_space("work", "team")
        .expect_err("rename should fail when destination artifacts already exist");
    assert!(
        err.to_string().contains("destination already exists"),
        "unexpected error: {err}"
    );

    let work = storage.get_space("work").expect("work should still exist");
    assert_eq!(work.name, "work");
    let missing_team = storage.get_space("team");
    assert!(missing_team.is_err(), "team should not exist in sqlite");

    let spaces = storage.spaces.read().expect("lock spaces map");
    assert!(spaces.contains_key("work"));
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

#[test]
fn reap_documents_deletes_only_old_deactivated_rows() {
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

    let old_doc_id = storage
        .upsert_document(
            collection_id,
            "old.rs",
            "old",
            "hash-old",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert old doc");
    let recent_doc_id = storage
        .upsert_document(
            collection_id,
            "recent.rs",
            "recent",
            "hash-recent",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert recent doc");
    let active_doc_id = storage
        .upsert_document(
            collection_id,
            "active.rs",
            "active",
            "hash-active",
            "2026-03-01T10:02:00Z",
        )
        .expect("insert active doc");

    {
        let conn = storage.db.lock().expect("lock db");
        conn.execute(
            "UPDATE documents
             SET active = 0, deactivated_at = '2000-01-01T00:00:00Z'
             WHERE id = ?1",
            [old_doc_id],
        )
        .expect("mark old deactivated");
        conn.execute(
            "UPDATE documents
             SET active = 0, deactivated_at = '2999-01-01T00:00:00Z'
             WHERE id = ?1",
            [recent_doc_id],
        )
        .expect("mark recent deactivated");
    }

    let reaped = storage.reap_documents(7).expect("reap docs");
    assert_eq!(reaped, vec![old_doc_id]);

    let old_doc = storage
        .get_document_by_path(collection_id, "old.rs")
        .expect("query old path");
    assert!(old_doc.is_none(), "old document should be deleted");

    let recent_doc = storage
        .get_document_by_path(collection_id, "recent.rs")
        .expect("query recent path")
        .expect("recent document should remain");
    assert!(!recent_doc.active);

    let active_doc = storage
        .get_document_by_path(collection_id, "active.rs")
        .expect("query active path")
        .expect("active document should remain");
    assert!(active_doc.active);
    assert!(active_doc_id > 0);
}

#[test]
fn reap_documents_returns_empty_when_no_documents_qualify() {
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

    storage
        .upsert_document(collection_id, "a.rs", "a", "hash-a", "2026-03-01T10:00:00Z")
        .expect("insert doc");

    let reaped = storage.reap_documents(7).expect("reap docs");
    assert!(reaped.is_empty());
}

#[test]
fn insert_and_get_chunks_for_document() {
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

    let inserts = vec![
        super::ChunkInsert {
            seq: 0,
            offset: 0,
            length: 100,
            heading: Some("# Intro".to_string()),
            kind: "section".to_string(),
        },
        super::ChunkInsert {
            seq: 1,
            offset: 100,
            length: 80,
            heading: Some("# Usage".to_string()),
            kind: "section".to_string(),
        },
    ];

    let ids = storage
        .insert_chunks(doc_id, &inserts)
        .expect("insert chunks");
    assert_eq!(ids.len(), 2);

    let chunks = storage
        .get_chunks_for_document(doc_id)
        .expect("get chunks for doc");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].seq, 0);
    assert_eq!(chunks[0].offset, 0);
    assert_eq!(chunks[0].length, 100);
    assert_eq!(chunks[0].heading.as_deref(), Some("# Intro"));
    assert_eq!(chunks[1].seq, 1);
}

#[test]
fn delete_chunks_for_document_returns_deleted_ids() {
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

    let inserts = vec![
        super::ChunkInsert {
            seq: 0,
            offset: 0,
            length: 100,
            heading: None,
            kind: "section".to_string(),
        },
        super::ChunkInsert {
            seq: 1,
            offset: 100,
            length: 50,
            heading: None,
            kind: "section".to_string(),
        },
    ];
    let inserted_ids = storage
        .insert_chunks(doc_id, &inserts)
        .expect("insert chunks");

    let deleted_ids = storage
        .delete_chunks_for_document(doc_id)
        .expect("delete chunks");
    assert_eq!(deleted_ids, inserted_ids);

    let remaining = storage
        .get_chunks_for_document(doc_id)
        .expect("get chunks for doc");
    assert!(remaining.is_empty());
}

#[test]
fn get_chunks_by_id_returns_requested_chunks() {
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

    let inserts = vec![
        super::ChunkInsert {
            seq: 0,
            offset: 0,
            length: 100,
            heading: None,
            kind: "section".to_string(),
        },
        super::ChunkInsert {
            seq: 1,
            offset: 100,
            length: 50,
            heading: None,
            kind: "section".to_string(),
        },
    ];
    let inserted_ids = storage
        .insert_chunks(doc_id, &inserts)
        .expect("insert chunks");

    let fetched = storage
        .get_chunks(&[inserted_ids[1], inserted_ids[0]])
        .expect("get chunks by id");
    assert_eq!(fetched.len(), 2);
    assert_eq!(fetched[0].id, inserted_ids[0]);
    assert_eq!(fetched[1].id, inserted_ids[1]);
}

#[test]
fn chunk_methods_missing_document_return_not_found() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let inserts = vec![super::ChunkInsert {
        seq: 0,
        offset: 0,
        length: 10,
        heading: None,
        kind: "section".to_string(),
    }];

    let insert_err = storage
        .insert_chunks(99999, &inserts)
        .expect_err("insert missing doc should fail");
    match KboltError::from(insert_err) {
        KboltError::DocumentNotFound { path } => assert_eq!(path, "id=99999"),
        other => panic!("unexpected insert error: {other}"),
    }

    let list_err = storage
        .get_chunks_for_document(99999)
        .expect_err("list missing doc should fail");
    match KboltError::from(list_err) {
        KboltError::DocumentNotFound { path } => assert_eq!(path, "id=99999"),
        other => panic!("unexpected list error: {other}"),
    }

    let delete_err = storage
        .delete_chunks_for_document(99999)
        .expect_err("delete missing doc should fail");
    match KboltError::from(delete_err) {
        KboltError::DocumentNotFound { path } => assert_eq!(path, "id=99999"),
        other => panic!("unexpected delete error: {other}"),
    }
}

#[test]
fn insert_count_and_delete_embeddings_by_model() {
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
    let chunk_ids = storage
        .insert_chunks(
            doc_id,
            &[
                super::ChunkInsert {
                    seq: 0,
                    offset: 0,
                    length: 100,
                    heading: None,
                    kind: "section".to_string(),
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 100,
                    length: 50,
                    heading: None,
                    kind: "section".to_string(),
                },
            ],
        )
        .expect("insert chunks");

    assert_eq!(storage.count_embeddings().expect("count embeddings"), 0);

    storage
        .insert_embeddings(&[
            (chunk_ids[0], "model-a"),
            (chunk_ids[1], "model-a"),
            (chunk_ids[1], "model-b"),
        ])
        .expect("insert embeddings");

    storage
        .insert_embeddings(&[(chunk_ids[0], "model-a")])
        .expect("idempotent upsert");

    assert_eq!(storage.count_embeddings().expect("count embeddings"), 3);

    let deleted_a = storage
        .delete_embeddings_for_model("model-a")
        .expect("delete model a");
    assert_eq!(deleted_a, 2);
    assert_eq!(storage.count_embeddings().expect("count embeddings"), 1);

    let deleted_missing = storage
        .delete_embeddings_for_model("missing-model")
        .expect("delete missing model");
    assert_eq!(deleted_missing, 0);
}

#[test]
fn get_unembedded_chunks_filters_active_and_model_specific_backlog() {
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
        .upsert_document(
            collection_id,
            "src/active.rs",
            "active",
            "hash-active",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert active doc");
    let inactive_doc_id = storage
        .upsert_document(
            collection_id,
            "src/inactive.rs",
            "inactive",
            "hash-inactive",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert inactive doc");

    let active_chunk_ids = storage
        .insert_chunks(
            active_doc_id,
            &[
                super::ChunkInsert {
                    seq: 0,
                    offset: 0,
                    length: 100,
                    heading: None,
                    kind: "section".to_string(),
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 100,
                    length: 50,
                    heading: None,
                    kind: "section".to_string(),
                },
            ],
        )
        .expect("insert active chunks");
    let inactive_chunk_ids = storage
        .insert_chunks(
            inactive_doc_id,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 42,
                heading: None,
                kind: "section".to_string(),
            }],
        )
        .expect("insert inactive chunk");

    storage
        .insert_embeddings(&[
            (active_chunk_ids[0], "model-a"),
            (active_chunk_ids[1], "model-b"),
        ])
        .expect("insert embeddings");
    storage
        .deactivate_document(inactive_doc_id)
        .expect("deactivate second doc");

    let backlog = storage
        .get_unembedded_chunks("model-a", 10)
        .expect("query backlog");
    assert_eq!(backlog.len(), 1);
    assert_eq!(backlog[0].chunk_id, active_chunk_ids[1]);
    assert_eq!(backlog[0].doc_path, "src/active.rs");
    assert_eq!(
        backlog[0].collection_path,
        std::path::PathBuf::from("/tmp/api")
    );
    assert_eq!(backlog[0].space_name, "work");
    assert_eq!(backlog[0].offset, 100);
    assert_eq!(backlog[0].length, 50);
    assert_ne!(backlog[0].chunk_id, inactive_chunk_ids[0]);

    let limited = storage
        .get_unembedded_chunks("model-a", 1)
        .expect("query limited backlog");
    assert_eq!(limited.len(), 1);
}

#[test]
fn get_fts_dirty_documents_returns_context_and_chunks() {
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
            "lib title",
            "hash-abc123",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc");
    storage
        .insert_chunks(
            doc_id,
            &[
                super::ChunkInsert {
                    seq: 0,
                    offset: 0,
                    length: 100,
                    heading: Some("# Intro".to_string()),
                    kind: "section".to_string(),
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 100,
                    length: 80,
                    heading: Some("# Usage".to_string()),
                    kind: "section".to_string(),
                },
            ],
        )
        .expect("insert chunks");

    let dirty = storage
        .get_fts_dirty_documents()
        .expect("get fts dirty docs");
    assert_eq!(dirty.len(), 1);
    assert_eq!(dirty[0].doc_id, doc_id);
    assert_eq!(dirty[0].doc_path, "src/lib.rs");
    assert_eq!(dirty[0].doc_title, "lib title");
    assert_eq!(dirty[0].doc_hash, "hash-abc123");
    assert_eq!(
        dirty[0].collection_path,
        std::path::PathBuf::from("/tmp/api")
    );
    assert_eq!(dirty[0].space_name, "work");
    assert_eq!(dirty[0].chunks.len(), 2);
    assert_eq!(dirty[0].chunks[0].seq, 0);
    assert_eq!(dirty[0].chunks[1].seq, 1);
}

#[test]
fn batch_clear_fts_dirty_clears_selected_documents_only() {
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

    let doc_a = storage
        .upsert_document(
            collection_id,
            "src/a.rs",
            "a",
            "hash-a",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc a");
    let doc_b = storage
        .upsert_document(
            collection_id,
            "src/b.rs",
            "b",
            "hash-b",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert doc b");

    storage
        .batch_clear_fts_dirty(&[doc_a])
        .expect("clear doc_a dirty flag");

    let dirty = storage
        .get_fts_dirty_documents()
        .expect("get remaining dirty docs");
    assert_eq!(dirty.len(), 1);
    assert_eq!(dirty[0].doc_id, doc_b);

    storage
        .batch_clear_fts_dirty(&[])
        .expect("empty batch should be no-op");
}

#[test]
fn cache_get_returns_none_for_missing_key() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    let value = storage.cache_get("missing").expect("query missing key");
    assert_eq!(value, None);
}

#[test]
fn cache_set_upserts_value_by_key() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");

    storage
        .cache_set("query:abc", "first-value")
        .expect("insert cache value");
    let first = storage
        .cache_get("query:abc")
        .expect("read inserted value")
        .expect("value should exist");
    assert_eq!(first, "first-value");

    storage
        .cache_set("query:abc", "updated-value")
        .expect("update cache value");
    let updated = storage
        .cache_get("query:abc")
        .expect("read updated value")
        .expect("updated value should exist");
    assert_eq!(updated, "updated-value");

    let conn = storage.db.lock().expect("lock db");
    let rows: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM llm_cache WHERE key = 'query:abc'",
            [],
            |row| row.get(0),
        )
        .expect("count cache rows");
    assert_eq!(rows, 1);
}

#[test]
fn count_documents_scopes_by_space_and_counts_inactive_rows() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let work_space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let notes_space_id = storage
        .create_space("notes", None)
        .expect("create notes space");

    let work_collection_id = storage
        .create_collection(
            work_space_id,
            "api",
            std::path::Path::new("/tmp/work-api"),
            None,
            None,
        )
        .expect("create work collection");
    let notes_collection_id = storage
        .create_collection(
            notes_space_id,
            "wiki",
            std::path::Path::new("/tmp/notes-wiki"),
            None,
            None,
        )
        .expect("create notes collection");

    let work_doc_a = storage
        .upsert_document(
            work_collection_id,
            "src/a.rs",
            "a",
            "hash-a",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert work doc a");
    storage
        .upsert_document(
            work_collection_id,
            "src/b.rs",
            "b",
            "hash-b",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert work doc b");
    storage
        .upsert_document(
            notes_collection_id,
            "README.md",
            "readme",
            "hash-readme",
            "2026-03-01T10:02:00Z",
        )
        .expect("insert notes doc");

    storage
        .deactivate_document(work_doc_a)
        .expect("deactivate one work doc");

    assert_eq!(storage.count_documents(None).expect("count all docs"), 3);
    assert_eq!(
        storage
            .count_documents(Some(work_space_id))
            .expect("count work docs"),
        2
    );
    assert_eq!(
        storage
            .count_documents(Some(notes_space_id))
            .expect("count notes docs"),
        1
    );

    let err = storage
        .count_documents(Some(99999))
        .expect_err("missing space should fail");
    match KboltError::from(err) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "id=99999"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn count_chunks_scopes_by_space() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let work_space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let notes_space_id = storage
        .create_space("notes", None)
        .expect("create notes space");

    let work_collection_id = storage
        .create_collection(
            work_space_id,
            "api",
            std::path::Path::new("/tmp/work-api"),
            None,
            None,
        )
        .expect("create work collection");
    let notes_collection_id = storage
        .create_collection(
            notes_space_id,
            "wiki",
            std::path::Path::new("/tmp/notes-wiki"),
            None,
            None,
        )
        .expect("create notes collection");

    let work_doc = storage
        .upsert_document(
            work_collection_id,
            "src/lib.rs",
            "lib",
            "hash-lib",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert work doc");
    let notes_doc = storage
        .upsert_document(
            notes_collection_id,
            "README.md",
            "readme",
            "hash-readme",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert notes doc");

    storage
        .insert_chunks(
            work_doc,
            &[
                super::ChunkInsert {
                    seq: 0,
                    offset: 0,
                    length: 10,
                    heading: None,
                    kind: "section".to_string(),
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 10,
                    length: 12,
                    heading: None,
                    kind: "section".to_string(),
                },
            ],
        )
        .expect("insert work chunks");
    storage
        .insert_chunks(
            notes_doc,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 7,
                heading: None,
                kind: "section".to_string(),
            }],
        )
        .expect("insert notes chunk");

    assert_eq!(storage.count_chunks(None).expect("count all chunks"), 3);
    assert_eq!(
        storage
            .count_chunks(Some(work_space_id))
            .expect("count work chunks"),
        2
    );
    assert_eq!(
        storage
            .count_chunks(Some(notes_space_id))
            .expect("count notes chunks"),
        1
    );
}

#[test]
fn count_embedded_chunks_scopes_by_space_and_deduplicates_models() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let work_space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let notes_space_id = storage
        .create_space("notes", None)
        .expect("create notes space");

    let work_collection_id = storage
        .create_collection(
            work_space_id,
            "api",
            std::path::Path::new("/tmp/work-api"),
            None,
            None,
        )
        .expect("create work collection");
    let notes_collection_id = storage
        .create_collection(
            notes_space_id,
            "wiki",
            std::path::Path::new("/tmp/notes-wiki"),
            None,
            None,
        )
        .expect("create notes collection");

    let work_doc = storage
        .upsert_document(
            work_collection_id,
            "src/lib.rs",
            "lib",
            "hash-lib",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert work doc");
    let notes_doc = storage
        .upsert_document(
            notes_collection_id,
            "README.md",
            "readme",
            "hash-readme",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert notes doc");

    let work_chunk_ids = storage
        .insert_chunks(
            work_doc,
            &[
                super::ChunkInsert {
                    seq: 0,
                    offset: 0,
                    length: 10,
                    heading: None,
                    kind: "section".to_string(),
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 10,
                    length: 12,
                    heading: None,
                    kind: "section".to_string(),
                },
            ],
        )
        .expect("insert work chunks");
    let notes_chunk_ids = storage
        .insert_chunks(
            notes_doc,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 7,
                heading: None,
                kind: "section".to_string(),
            }],
        )
        .expect("insert notes chunk");

    storage
        .insert_embeddings(&[
            (work_chunk_ids[0], "model-a"),
            (work_chunk_ids[0], "model-b"),
            (work_chunk_ids[1], "model-a"),
            (notes_chunk_ids[0], "model-a"),
        ])
        .expect("insert embeddings");

    assert_eq!(
        storage
            .count_embedded_chunks(None)
            .expect("count embedded chunks"),
        3
    );
    assert_eq!(
        storage
            .count_embedded_chunks(Some(work_space_id))
            .expect("count work embedded chunks"),
        2
    );
    assert_eq!(
        storage
            .count_embedded_chunks(Some(notes_space_id))
            .expect("count notes embedded chunks"),
        1
    );
}

#[test]
fn disk_usage_sums_sqlite_indexes_and_models() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let baseline = storage.disk_usage().expect("calculate baseline disk usage");

    std::fs::create_dir_all(cache_dir.join("spaces/work/tantivy/segments"))
        .expect("create work tantivy dir");
    std::fs::create_dir_all(cache_dir.join("spaces/notes")).expect("create notes dir");
    std::fs::create_dir_all(cache_dir.join("models")).expect("create models dir");

    std::fs::write(cache_dir.join("spaces/work/tantivy/seg1.bin"), vec![b'x'; 10])
        .expect("write tantivy segment 1");
    std::fs::write(
        cache_dir.join("spaces/work/tantivy/segments/seg2.bin"),
        vec![b'y'; 4],
    )
    .expect("write tantivy segment 2");
    std::fs::write(cache_dir.join("spaces/work/vectors.usearch"), vec![b'v'; 7])
        .expect("write work vectors");
    std::fs::write(cache_dir.join("spaces/notes/vectors.usearch"), vec![b'w'; 3])
        .expect("write notes vectors");
    std::fs::write(cache_dir.join("models/embed.onnx"), vec![b'm'; 11]).expect("write model file");
    std::fs::write(cache_dir.join("spaces/work/ignored.bin"), vec![b'i'; 9])
        .expect("write ignored file");

    let sqlite_bytes = std::fs::metadata(cache_dir.join("meta.sqlite"))
        .expect("stat sqlite file")
        .len();

    let usage = storage.disk_usage().expect("calculate disk usage");
    assert_eq!(usage.sqlite_bytes, sqlite_bytes);
    assert_eq!(usage.tantivy_bytes, baseline.tantivy_bytes + 14);
    assert_eq!(usage.usearch_bytes, baseline.usearch_bytes + 10);
    assert_eq!(usage.models_bytes, baseline.models_bytes + 11);
    assert_eq!(
        usage.total_bytes,
        usage.sqlite_bytes + usage.tantivy_bytes + usage.usearch_bytes + usage.models_bytes
    );
}

#[test]
fn disk_usage_handles_missing_optional_directories() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");

    let usage = storage.disk_usage().expect("calculate disk usage");
    assert!(usage.sqlite_bytes > 0);
    assert!(usage.tantivy_bytes > 0);
    assert_eq!(usage.usearch_bytes, 0);
    assert_eq!(usage.models_bytes, 0);
    assert_eq!(
        usage.total_bytes,
        usage.sqlite_bytes + usage.tantivy_bytes + usage.usearch_bytes + usage.models_bytes
    );
}
