use tempfile::tempdir;

use super::{DocumentTitleSource, Storage};
use crate::ingest::chunk::FinalChunkKind;
use kbolt_types::KboltError;
use rusqlite::Connection;

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
fn storage_instances_can_share_cache_before_any_tantivy_write() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    let storage_a = Storage::new(&cache_dir).expect("create first storage");
    storage_a.create_space("work", None).expect("create work");
    storage_a.open_space("work").expect("open work");

    let storage_b = Storage::new(&cache_dir).expect("create second storage");
    let spaces = storage_b
        .list_spaces()
        .expect("list spaces from second storage");
    assert_eq!(
        spaces
            .into_iter()
            .map(|space| space.name)
            .collect::<Vec<_>>(),
        vec!["default".to_string(), "work".to_string()]
    );
}

#[test]
fn commit_tantivy_releases_writer_for_other_processes() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    let storage_a = Storage::new(&cache_dir).expect("create first storage");
    storage_a.create_space("work", None).expect("create work");

    storage_a
        .index_tantivy(
            "work",
            &[super::TantivyEntry {
                chunk_id: 11,
                doc_id: 1,
                filepath: "docs/a.md".to_string(),
                semantic_title: Some("alpha".to_string()),
                heading: None,
                body: "alpha from first storage".to_string(),
            }],
        )
        .expect("index from first storage");
    storage_a
        .commit_tantivy("work")
        .expect("commit first writer");

    let storage_b = Storage::new(&cache_dir).expect("create second storage");
    storage_b
        .index_tantivy(
            "work",
            &[super::TantivyEntry {
                chunk_id: 22,
                doc_id: 2,
                filepath: "docs/b.md".to_string(),
                semantic_title: Some("beta".to_string()),
                heading: None,
                body: "beta from second storage".to_string(),
            }],
        )
        .expect("second storage should acquire tantivy writer");
    storage_b
        .commit_tantivy("work")
        .expect("commit second writer");

    let hits = storage_a
        .query_bm25("work", "beta", &[("body", 1.0)], 10)
        .expect("query second entry from first storage");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk_id, 22);
}

#[test]
fn tantivy_index_and_query_returns_ranked_hits() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage
        .index_tantivy(
            "work",
            &[
                super::TantivyEntry {
                    chunk_id: 11,
                    doc_id: 1,
                    filepath: "api/lib.rs".to_string(),
                    semantic_title: Some("alpha guide".to_string()),
                    heading: Some("intro".to_string()),
                    body: "alpha token in body".to_string(),
                },
                super::TantivyEntry {
                    chunk_id: 22,
                    doc_id: 2,
                    filepath: "api/main.rs".to_string(),
                    semantic_title: Some("beta guide".to_string()),
                    heading: Some("usage".to_string()),
                    body: "beta token only".to_string(),
                },
            ],
        )
        .expect("index entries");
    storage.commit_tantivy("work").expect("commit tantivy");

    let hits = storage
        .query_bm25("work", "alpha", &[("title", 3.0), ("body", 1.0)], 10)
        .expect("query bm25");
    assert!(!hits.is_empty(), "expected at least one hit");
    assert_eq!(hits[0].chunk_id, 11);
}

#[test]
fn bm25_document_filter_does_not_change_scores() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage
        .index_tantivy(
            "work",
            &[
                super::TantivyEntry {
                    chunk_id: 11,
                    doc_id: 1,
                    filepath: "api/strong.md".to_string(),
                    semantic_title: None,
                    heading: None,
                    body: "alpha alpha alpha".to_string(),
                },
                super::TantivyEntry {
                    chunk_id: 22,
                    doc_id: 2,
                    filepath: "api/weak.md".to_string(),
                    semantic_title: None,
                    heading: None,
                    body: "alpha".to_string(),
                },
                super::TantivyEntry {
                    chunk_id: 33,
                    doc_id: 3,
                    filepath: "other/excluded.md".to_string(),
                    semantic_title: None,
                    heading: None,
                    body: "alpha alpha".to_string(),
                },
            ],
        )
        .expect("index entries");
    storage.commit_tantivy("work").expect("commit tantivy");

    let unfiltered = storage
        .query_bm25("work", "alpha", &[("body", 1.0)], 10)
        .expect("query unfiltered bm25");
    let filtered = storage
        .query_bm25_in_documents("work", "alpha", &[("body", 1.0)], &[1, 2], 10)
        .expect("query filtered bm25");
    let expected = unfiltered
        .iter()
        .filter(|hit| [11, 22].contains(&hit.chunk_id))
        .collect::<Vec<_>>();

    assert_eq!(filtered.len(), expected.len());
    for (filtered, expected) in filtered.iter().zip(expected) {
        assert_eq!(filtered.chunk_id, expected.chunk_id);
        assert!(
            (filtered.score - expected.score).abs() < f32::EPSILON,
            "filter should not perturb BM25 score for chunk {}: filtered={}, unfiltered={}",
            filtered.chunk_id,
            filtered.score,
            expected.score
        );
    }
}

#[test]
fn bm25_literal_queries_accept_punctuation_heavy_user_text() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage
        .index_tantivy(
            "work",
            &[super::TantivyEntry {
                chunk_id: 11,
                doc_id: 1,
                filepath: "docs/scifact.md".to_string(),
                semantic_title: Some("thalassemia note".to_string()),
                heading: Some("alpha trait".to_string()),
                body: "high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha thalassemia trait subjects".to_string(),
            }],
        )
        .expect("index entries");
    storage.commit_tantivy("work").expect("commit tantivy");

    let scifact_hits = storage
        .query_bm25(
            "work",
            "A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects.",
            &[("body", 1.0)],
            10,
        )
        .expect("query sci fact punctuation");
    assert!(!scifact_hits.is_empty(), "expected scifact hit");
    assert_eq!(scifact_hits[0].chunk_id, 11);
}

#[test]
fn bm25_literal_queries_return_empty_hits_for_punctuation_only_input() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage
        .index_tantivy(
            "work",
            &[super::TantivyEntry {
                chunk_id: 11,
                doc_id: 1,
                filepath: "docs/alpha.md".to_string(),
                semantic_title: Some("alpha guide".to_string()),
                heading: None,
                body: "alpha token in body".to_string(),
            }],
        )
        .expect("index entries");
    storage.commit_tantivy("work").expect("commit tantivy");

    let hits = storage
        .query_bm25("work", "(+)-::///", &[("title", 3.0), ("body", 1.0)], 10)
        .expect("query punctuation only");
    assert!(hits.is_empty(), "punctuation-only query should not match");
}

#[test]
fn delete_tantivy_removes_chunk_after_commit() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage
        .index_tantivy(
            "work",
            &[
                super::TantivyEntry {
                    chunk_id: 11,
                    doc_id: 1,
                    filepath: "api/lib.rs".to_string(),
                    semantic_title: Some("alpha guide".to_string()),
                    heading: None,
                    body: "alphaunique".to_string(),
                },
                super::TantivyEntry {
                    chunk_id: 22,
                    doc_id: 2,
                    filepath: "api/main.rs".to_string(),
                    semantic_title: Some("beta guide".to_string()),
                    heading: None,
                    body: "betaunique".to_string(),
                },
            ],
        )
        .expect("index entries");
    storage.commit_tantivy("work").expect("commit tantivy");

    storage
        .delete_tantivy("work", &[11])
        .expect("delete first chunk");
    storage.commit_tantivy("work").expect("commit delete");

    let alpha_hits = storage
        .query_bm25("work", "alphaunique", &[("body", 1.0)], 10)
        .expect("query alpha");
    assert!(
        alpha_hits.is_empty(),
        "deleted chunk should not be returned"
    );

    let beta_hits = storage
        .query_bm25("work", "betaunique", &[("body", 1.0)], 10)
        .expect("query beta");
    assert_eq!(beta_hits.len(), 1);
    assert_eq!(beta_hits[0].chunk_id, 22);
}

#[test]
fn delete_tantivy_by_doc_removes_all_doc_chunks_after_commit() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage
        .index_tantivy(
            "work",
            &[
                super::TantivyEntry {
                    chunk_id: 11,
                    doc_id: 1,
                    filepath: "api/lib.rs".to_string(),
                    semantic_title: Some("alpha one".to_string()),
                    heading: None,
                    body: "doconea".to_string(),
                },
                super::TantivyEntry {
                    chunk_id: 12,
                    doc_id: 1,
                    filepath: "api/lib.rs".to_string(),
                    semantic_title: Some("alpha two".to_string()),
                    heading: None,
                    body: "doconeb".to_string(),
                },
                super::TantivyEntry {
                    chunk_id: 21,
                    doc_id: 2,
                    filepath: "api/main.rs".to_string(),
                    semantic_title: Some("beta".to_string()),
                    heading: None,
                    body: "doctwo".to_string(),
                },
            ],
        )
        .expect("index entries");
    storage.commit_tantivy("work").expect("commit tantivy");

    storage
        .delete_tantivy_by_doc("work", 1)
        .expect("delete doc 1");
    storage.commit_tantivy("work").expect("commit delete");

    let removed_a = storage
        .query_bm25("work", "doconea", &[("body", 1.0)], 10)
        .expect("query removed chunk a");
    let removed_b = storage
        .query_bm25("work", "doconeb", &[("body", 1.0)], 10)
        .expect("query removed chunk b");
    let remaining = storage
        .query_bm25("work", "doctwo", &[("body", 1.0)], 10)
        .expect("query remaining chunk");
    assert!(removed_a.is_empty());
    assert!(removed_b.is_empty());
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].chunk_id, 21);
}

#[test]
fn usearch_insert_query_and_count_round_trip() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage
        .insert_usearch("work", 11, &[1.0, 0.0])
        .expect("insert first vector");
    storage
        .insert_usearch("work", 22, &[0.0, 1.0])
        .expect("insert second vector");

    let count = storage.count_usearch("work").expect("count vectors");
    assert_eq!(count, 2);

    let hits = storage
        .query_dense("work", &[1.0, 0.0], 2)
        .expect("query dense");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].chunk_id, 11);
}

#[test]
fn batch_insert_usearch_rejects_mixed_dimensions() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    let err = storage
        .batch_insert_usearch("work", &[(1, &[1.0, 0.0]), (2, &[1.0, 0.0, 0.0])])
        .expect_err("mixed dimensions should fail");
    assert!(
        err.to_string().contains("vector dimension mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn delete_usearch_removes_keys() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");

    storage
        .batch_insert_usearch("work", &[(11, &[1.0, 0.0]), (22, &[0.0, 1.0])])
        .expect("insert vectors");
    assert_eq!(
        storage.count_usearch("work").expect("count before delete"),
        2
    );

    storage
        .delete_usearch("work", &[11])
        .expect("delete key 11 from usearch");
    assert_eq!(
        storage.count_usearch("work").expect("count after delete"),
        1
    );

    let hits = storage
        .query_dense("work", &[0.0, 1.0], 2)
        .expect("query after delete");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk_id, 22);
}

#[test]
fn clear_usearch_resets_index() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");
    storage
        .batch_insert_usearch("work", &[(11, &[1.0, 0.0]), (22, &[0.0, 1.0])])
        .expect("insert vectors");

    storage.clear_usearch("work").expect("clear usearch");
    assert_eq!(storage.count_usearch("work").expect("count after clear"), 0);
    let hits = storage
        .query_dense("work", &[1.0, 0.0], 2)
        .expect("query after clear");
    assert!(hits.is_empty());
}

#[test]
fn usearch_persists_across_close_and_open_space() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    storage.create_space("work", None).expect("create work");
    storage
        .insert_usearch("work", 11, &[1.0, 0.0])
        .expect("insert vector");
    assert_eq!(
        storage.count_usearch("work").expect("count before close"),
        1
    );

    storage.close_space("work").expect("close work");
    storage.open_space("work").expect("reopen work");

    assert_eq!(
        storage.count_usearch("work").expect("count after reopen"),
        1
    );
    let hits = storage
        .query_dense("work", &[1.0, 0.0], 1)
        .expect("query after reopen");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk_id, 11);
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
        "document_texts",
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

    let title_source_exists = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('documents') WHERE name = 'title_source'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("query documents title_source column");
    assert_eq!(title_source_exists, 1, "missing documents.title_source");

    let schema_version = conn
        .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
        .expect("query schema version");
    assert_eq!(schema_version, 1);
}

#[test]
fn new_rejects_non_empty_legacy_schema_without_canonical_text() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    std::fs::create_dir_all(&cache_dir).expect("create cache dir");
    let db_path = cache_dir.join("meta.sqlite");
    let conn = Connection::open(&db_path).expect("open sqlite");
    conn.execute_batch(
        r#"
PRAGMA foreign_keys = ON;

CREATE TABLE spaces (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    created     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE collections (
    id          INTEGER PRIMARY KEY,
    space_id    INTEGER NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    description TEXT,
    extensions  TEXT,
    created     TEXT NOT NULL,
    updated     TEXT NOT NULL,
    UNIQUE(space_id, name)
);

CREATE TABLE documents (
    id              INTEGER PRIMARY KEY,
    collection_id   INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    path            TEXT NOT NULL,
    title           TEXT NOT NULL,
    hash            TEXT NOT NULL,
    modified        TEXT NOT NULL,
    active          INTEGER NOT NULL DEFAULT 1,
    deactivated_at  TEXT,
    fts_dirty       INTEGER NOT NULL DEFAULT 0,
    UNIQUE(collection_id, path)
);

CREATE TABLE chunks (
    id       INTEGER PRIMARY KEY,
    doc_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq      INTEGER NOT NULL,
    offset   INTEGER NOT NULL,
    length   INTEGER NOT NULL,
    heading  TEXT,
    kind     TEXT NOT NULL DEFAULT 'section',
    UNIQUE(doc_id, seq)
);

CREATE TABLE embeddings (
    chunk_id    INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
"#,
    )
    .expect("create legacy schema");
    conn.execute(
        "INSERT INTO spaces (id, name, description, created) VALUES (1, 'default', NULL, '2026-03-01T00:00:00Z')",
        [],
    )
    .expect("insert space");
    conn.execute(
        "INSERT INTO collections (id, space_id, name, path, description, extensions, created, updated)
         VALUES (1, 1, 'docs', '/tmp/docs', NULL, NULL, '2026-03-01T00:00:00Z', '2026-03-01T00:00:00Z')",
        [],
    )
    .expect("insert collection");
    conn.execute(
        "INSERT INTO documents (id, collection_id, path, title, hash, modified, active, deactivated_at, fts_dirty)
         VALUES (1, 1, 'guide.md', 'Guide', 'hash-1', '2026-03-01T00:00:00Z', 1, NULL, 0)",
        [],
    )
    .expect("insert legacy document");
    drop(conn);

    let err = match Storage::new(&cache_dir) {
        Ok(_) => panic!("non-empty legacy schema should fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("older text-storage format"),
        "unexpected error: {err}"
    );
}

#[test]
fn new_initializes_empty_legacy_schema_with_canonical_text_table() {
    let tmp = tempdir().expect("create tempdir");
    let cache_dir = tmp.path().join("cache");
    std::fs::create_dir_all(&cache_dir).expect("create cache dir");
    let db_path = cache_dir.join("meta.sqlite");
    let conn = Connection::open(&db_path).expect("open sqlite");
    conn.execute_batch(
        r#"
PRAGMA foreign_keys = ON;

CREATE TABLE spaces (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    created     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE collections (
    id          INTEGER PRIMARY KEY,
    space_id    INTEGER NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    description TEXT,
    extensions  TEXT,
    created     TEXT NOT NULL,
    updated     TEXT NOT NULL,
    UNIQUE(space_id, name)
);

CREATE TABLE documents (
    id              INTEGER PRIMARY KEY,
    collection_id   INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    path            TEXT NOT NULL,
    title           TEXT NOT NULL,
    hash            TEXT NOT NULL,
    modified        TEXT NOT NULL,
    active          INTEGER NOT NULL DEFAULT 1,
    deactivated_at  TEXT,
    fts_dirty       INTEGER NOT NULL DEFAULT 0,
    UNIQUE(collection_id, path)
);

CREATE TABLE chunks (
    id       INTEGER PRIMARY KEY,
    doc_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq      INTEGER NOT NULL,
    offset   INTEGER NOT NULL,
    length   INTEGER NOT NULL,
    heading  TEXT,
    kind     TEXT NOT NULL DEFAULT 'section',
    UNIQUE(doc_id, seq)
);

CREATE TABLE embeddings (
    chunk_id    INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
"#,
    )
    .expect("create empty legacy schema");
    drop(conn);

    Storage::new(&cache_dir).expect("initialize empty legacy schema");
    let conn = Connection::open(db_path).expect("open sqlite after init");
    let document_texts_exists = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = 'document_texts'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("query document_texts table");
    assert_eq!(document_texts_exists, 1);
    let title_source_exists = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('documents') WHERE name = 'title_source'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("query documents title_source column");
    assert_eq!(title_source_exists, 1);
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
            DocumentTitleSource::Extracted,
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
    assert_eq!(stored.title_source, DocumentTitleSource::Extracted);
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
            DocumentTitleSource::Extracted,
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
            DocumentTitleSource::Extracted,
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
    assert_eq!(stored.title_source, DocumentTitleSource::Extracted);
    assert_eq!(stored.hash, "hash-2");
    assert_eq!(stored.modified, "2026-03-01T12:00:00Z");
    assert!(stored.active);
    assert_eq!(stored.deactivated_at, None);
    assert!(stored.fts_dirty);
}

#[test]
fn refresh_document_activity_updates_modified_and_reactivates_without_dirtying() {
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
            DocumentTitleSource::Extracted,
            "hash-1",
            "2026-03-01T10:00:00Z",
        )
        .expect("upsert document");

    {
        let conn = storage.db.lock().expect("lock db");
        conn.execute(
            "UPDATE documents
             SET active = 0, deactivated_at = '2026-03-01T11:00:00Z', fts_dirty = 0
             WHERE id = ?1",
            [doc_id],
        )
        .expect("mark inactive and clean");
    }

    storage
        .refresh_document_activity(doc_id, "2026-03-01T12:00:00Z")
        .expect("refresh document activity");

    let stored = storage
        .get_document_by_path(collection_id, "src/lib.rs")
        .expect("get document")
        .expect("document should exist");
    assert_eq!(stored.title, "lib");
    assert_eq!(stored.title_source, DocumentTitleSource::Extracted);
    assert_eq!(stored.modified, "2026-03-01T12:00:00Z");
    assert!(stored.active);
    assert_eq!(stored.deactivated_at, None);
    assert!(
        !stored.fts_dirty,
        "metadata-only update should not mark dirty"
    );
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
        .upsert_document(
            collection_id,
            "a.rs",
            "a",
            DocumentTitleSource::Extracted,
            "hash-a",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert active doc");
    let inactive_doc_id = storage
        .upsert_document(
            collection_id,
            "b.rs",
            "b",
            DocumentTitleSource::Extracted,
            "hash-b",
            "2026-03-01T11:00:00Z",
        )
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
fn list_collection_file_rows_reports_counts_and_active_filter() {
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
            "a.rs",
            DocumentTitleSource::Extracted,
            "hash-a",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc a");
    let doc_b = storage
        .upsert_document(
            collection_id,
            "src/b.rs",
            "b.rs",
            DocumentTitleSource::Extracted,
            "hash-b",
            "2026-03-01T11:00:00Z",
        )
        .expect("insert doc b");

    let chunk_ids = storage
        .insert_chunks(
            doc_a,
            &[
                super::ChunkInsert {
                    seq: 0,
                    offset: 0,
                    length: 10,
                    heading: None,
                    kind: FinalChunkKind::Section,
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 10,
                    length: 8,
                    heading: None,
                    kind: FinalChunkKind::Section,
                },
            ],
        )
        .expect("insert chunks for doc a");
    storage
        .insert_chunks(
            doc_b,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 12,
                heading: None,
                kind: FinalChunkKind::Section,
            }],
        )
        .expect("insert chunk for doc b");
    storage
        .insert_embeddings(&[(chunk_ids[0], "embed-model")])
        .expect("insert embeddings");
    storage
        .deactivate_document(doc_b)
        .expect("deactivate doc b");

    let all = storage
        .list_collection_file_rows(collection_id, false)
        .expect("list all files");
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].path, "src/a.rs");
    assert_eq!(all[0].chunk_count, 2);
    assert_eq!(all[0].embedded_chunk_count, 1);
    assert!(all[0].active);
    assert_eq!(all[1].path, "src/b.rs");
    assert_eq!(all[1].chunk_count, 1);
    assert_eq!(all[1].embedded_chunk_count, 0);
    assert!(!all[1].active);

    let active_only = storage
        .list_collection_file_rows(collection_id, true)
        .expect("list active files");
    assert_eq!(active_only.len(), 1);
    assert_eq!(active_only[0].path, "src/a.rs");
    assert!(active_only[0].active);
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
            DocumentTitleSource::Extracted,
            "abcd0001",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc1");
    let _doc2 = storage
        .upsert_document(
            collection_id,
            "src/b.rs",
            "b",
            DocumentTitleSource::Extracted,
            "abcd0002",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert doc2");
    let _doc3 = storage
        .upsert_document(
            collection_id,
            "src/c.rs",
            "c",
            DocumentTitleSource::Extracted,
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
            DocumentTitleSource::Extracted,
            "hash-old",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert old doc");
    let recent_doc_id = storage
        .upsert_document(
            collection_id,
            "recent.rs",
            "recent",
            DocumentTitleSource::Extracted,
            "hash-recent",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert recent doc");
    let active_doc_id = storage
        .upsert_document(
            collection_id,
            "active.rs",
            "active",
            DocumentTitleSource::Extracted,
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
            DocumentTitleSource::Extracted,
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
            kind: FinalChunkKind::Section,
        },
        super::ChunkInsert {
            seq: 1,
            offset: 100,
            length: 80,
            heading: Some("# Usage".to_string()),
            kind: FinalChunkKind::Section,
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
    assert_eq!(chunks[0].kind, FinalChunkKind::Section);
    assert_eq!(chunks[1].seq, 1);
    assert_eq!(chunks[1].kind, FinalChunkKind::Section);
}

#[test]
fn get_chunks_for_document_rejects_invalid_stored_chunk_kind() {
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
            DocumentTitleSource::Extracted,
            "hash-1",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc");

    {
        let conn = storage.db.lock().expect("lock db");
        conn.execute(
            "INSERT INTO chunks (doc_id, seq, offset, length, heading, kind)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![doc_id, 0, 0, 10, Option::<String>::None, "broken"],
        )
        .expect("insert invalid chunk kind");
    }

    let err = storage
        .get_chunks_for_document(doc_id)
        .expect_err("invalid chunk kind should fail");
    assert!(err.to_string().contains("invalid stored chunk kind"));
}

#[test]
fn get_chunks_for_document_rejects_negative_stored_span() {
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
            DocumentTitleSource::Extracted,
            "hash-1",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc");

    {
        let conn = storage.db.lock().expect("lock db");
        conn.execute(
            "INSERT INTO chunks (doc_id, seq, offset, length, heading, kind)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![doc_id, 0, -1, 10, Option::<String>::None, "section"],
        )
        .expect("insert invalid chunk span");
    }

    let err = storage
        .get_chunks_for_document(doc_id)
        .expect_err("negative chunk span should fail");
    assert!(err
        .to_string()
        .contains("chunks.offset must not be negative"));
}

#[test]
fn put_get_and_hydrate_document_text() {
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
            DocumentTitleSource::Extracted,
            "hash-1",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc");

    let canonical_text = "alpha café\n\nbeta";
    assert!(!storage
        .has_document_text(doc_id)
        .expect("check missing document text"));
    storage
        .put_document_text(doc_id, "txt", "hash-1", "text-hash-1", canonical_text)
        .expect("put document text");
    assert!(storage
        .has_document_text(doc_id)
        .expect("check stored document text"));
    let stored = storage
        .get_document_text(doc_id)
        .expect("get document text");
    assert_eq!(stored.extractor_key, "txt");
    assert_eq!(stored.source_hash, "hash-1");
    assert_eq!(stored.text_hash, "text-hash-1");
    assert_eq!(stored.text, canonical_text);

    let chunk_ids = storage
        .insert_chunks(
            doc_id,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: "alpha café".len(),
                heading: None,
                kind: FinalChunkKind::Paragraph,
            }],
        )
        .expect("insert chunk");

    let chunk_text = storage
        .get_chunk_text(chunk_ids[0])
        .expect("hydrate chunk text");
    assert_eq!(chunk_text.extractor_key, "txt");
    assert_eq!(chunk_text.text, "alpha café");
    assert_eq!(chunk_text.chunk.id, chunk_ids[0]);
}

#[test]
fn replace_document_generation_writes_text_and_chunks_atomically() {
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

    let replacement = super::DocumentGenerationReplace {
        collection_id,
        path: "src/lib.rs",
        title: "lib",
        title_source: DocumentTitleSource::Extracted,
        hash: "hash-1",
        modified: "2026-03-01T10:00:00Z",
        extractor_key: "code",
        source_hash: "hash-1",
        text_hash: "text-hash-1",
        text: "alpha\n\nbeta",
        chunks: &[
            super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 5,
                heading: None,
                kind: FinalChunkKind::Paragraph,
            },
            super::ChunkInsert {
                seq: 1,
                offset: 7,
                length: 4,
                heading: None,
                kind: FinalChunkKind::Paragraph,
            },
        ],
    };

    let result = storage
        .replace_document_generation(replacement)
        .expect("replace generation");
    assert!(result.old_chunk_ids.is_empty());
    assert_eq!(result.chunk_ids.len(), 2);

    let doc = storage
        .get_document_by_path(collection_id, "src/lib.rs")
        .expect("get document")
        .expect("document exists");
    assert_eq!(doc.id, result.doc_id);
    assert!(doc.fts_dirty);
    assert_eq!(
        storage
            .get_document_text(result.doc_id)
            .expect("get document text")
            .text,
        "alpha\n\nbeta"
    );
    assert_eq!(
        storage
            .get_chunk_text(result.chunk_ids[1])
            .expect("hydrate chunk")
            .text,
        "beta"
    );
}

#[test]
fn replace_document_generation_returns_old_chunks_and_cascades_embeddings() {
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

    let first = storage
        .replace_document_generation(super::DocumentGenerationReplace {
            collection_id,
            path: "src/lib.rs",
            title: "lib",
            title_source: DocumentTitleSource::Extracted,
            hash: "hash-1",
            modified: "2026-03-01T10:00:00Z",
            extractor_key: "code",
            source_hash: "hash-1",
            text_hash: "text-hash-1",
            text: "alpha",
            chunks: &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 5,
                heading: None,
                kind: FinalChunkKind::Paragraph,
            }],
        })
        .expect("initial replace");
    storage
        .insert_embeddings(&[(first.chunk_ids[0], "embed-model")])
        .expect("insert old embedding");
    assert_eq!(storage.count_embeddings().expect("count embeddings"), 1);

    let second = storage
        .replace_document_generation(super::DocumentGenerationReplace {
            collection_id,
            path: "src/lib.rs",
            title: "lib v2",
            title_source: DocumentTitleSource::Extracted,
            hash: "hash-2",
            modified: "2026-03-01T11:00:00Z",
            extractor_key: "code",
            source_hash: "hash-2",
            text_hash: "text-hash-2",
            text: "gamma",
            chunks: &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 5,
                heading: None,
                kind: FinalChunkKind::Paragraph,
            }],
        })
        .expect("second replace");

    assert_eq!(second.doc_id, first.doc_id);
    assert_eq!(second.old_chunk_ids, first.chunk_ids);
    assert_eq!(storage.count_embeddings().expect("count embeddings"), 0);
    assert_eq!(
        storage
            .get_document_text(second.doc_id)
            .expect("get updated text")
            .text,
        "gamma"
    );
}

#[test]
fn replace_document_generation_rejects_invalid_span_without_mutating_existing() {
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

    let first = storage
        .replace_document_generation(super::DocumentGenerationReplace {
            collection_id,
            path: "src/lib.rs",
            title: "lib",
            title_source: DocumentTitleSource::Extracted,
            hash: "hash-1",
            modified: "2026-03-01T10:00:00Z",
            extractor_key: "code",
            source_hash: "hash-1",
            text_hash: "text-hash-1",
            text: "alpha",
            chunks: &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 5,
                heading: None,
                kind: FinalChunkKind::Paragraph,
            }],
        })
        .expect("initial replace");

    let err = storage
        .replace_document_generation(super::DocumentGenerationReplace {
            collection_id,
            path: "src/lib.rs",
            title: "lib broken",
            title_source: DocumentTitleSource::Extracted,
            hash: "hash-2",
            modified: "2026-03-01T11:00:00Z",
            extractor_key: "code",
            source_hash: "hash-2",
            text_hash: "text-hash-2",
            text: "éclair",
            chunks: &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 1,
                heading: None,
                kind: FinalChunkKind::Paragraph,
            }],
        })
        .expect_err("invalid span should fail before mutation");
    assert!(err.to_string().contains("not on UTF-8 boundaries"));

    let doc = storage
        .get_document_by_path(collection_id, "src/lib.rs")
        .expect("get document")
        .expect("document exists");
    assert_eq!(doc.hash, "hash-1");
    assert_eq!(
        storage
            .get_document_text(first.doc_id)
            .expect("get existing text")
            .text,
        "alpha"
    );
    assert_eq!(
        storage
            .get_chunks_for_document(first.doc_id)
            .expect("get existing chunks")
            .len(),
        1
    );
}

#[test]
fn get_chunk_text_rejects_invalid_canonical_span() {
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
            DocumentTitleSource::Extracted,
            "hash-1",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc");
    storage
        .put_document_text(doc_id, "txt", "hash-1", "text-hash-1", "éclair")
        .expect("put document text");
    let chunk_ids = storage
        .insert_chunks(
            doc_id,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 1,
                heading: None,
                kind: FinalChunkKind::Paragraph,
            }],
        )
        .expect("insert chunk");

    let err = storage
        .get_chunk_text(chunk_ids[0])
        .expect_err("invalid utf8 boundary should fail");
    assert!(err.to_string().contains("not on UTF-8 boundaries"));
}

#[test]
fn get_chunk_text_reports_missing_document_text() {
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
            DocumentTitleSource::Extracted,
            "hash-1",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc");
    let chunk_ids = storage
        .insert_chunks(
            doc_id,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 5,
                heading: None,
                kind: FinalChunkKind::Paragraph,
            }],
        )
        .expect("insert chunk");

    let err = storage
        .get_chunk_text(chunk_ids[0])
        .expect_err("missing document text should fail");
    assert!(err.to_string().contains("missing persisted canonical text"));
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
            DocumentTitleSource::Extracted,
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
            kind: FinalChunkKind::Section,
        },
        super::ChunkInsert {
            seq: 1,
            offset: 100,
            length: 50,
            heading: None,
            kind: FinalChunkKind::Section,
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
            DocumentTitleSource::Extracted,
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
                    kind: FinalChunkKind::Section,
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 100,
                    length: 50,
                    heading: None,
                    kind: FinalChunkKind::Section,
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
fn list_embedding_models_in_space_returns_distinct_sorted_models() {
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

    let work_doc_id = storage
        .upsert_document(
            work_collection_id,
            "src/lib.rs",
            "lib",
            DocumentTitleSource::Extracted,
            "hash-work",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert work doc");
    let notes_doc_id = storage
        .upsert_document(
            notes_collection_id,
            "README.md",
            "readme",
            DocumentTitleSource::Extracted,
            "hash-notes",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert notes doc");

    let work_chunk_ids = storage
        .insert_chunks(
            work_doc_id,
            &[
                super::ChunkInsert {
                    seq: 0,
                    offset: 0,
                    length: 10,
                    heading: None,
                    kind: FinalChunkKind::Section,
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 10,
                    length: 12,
                    heading: None,
                    kind: FinalChunkKind::Section,
                },
            ],
        )
        .expect("insert work chunks");
    let notes_chunk_ids = storage
        .insert_chunks(
            notes_doc_id,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 7,
                heading: None,
                kind: FinalChunkKind::Section,
            }],
        )
        .expect("insert notes chunks");

    storage
        .insert_embeddings(&[
            (work_chunk_ids[0], "model-b"),
            (work_chunk_ids[0], "model-a"),
            (work_chunk_ids[1], "model-a"),
            (notes_chunk_ids[0], "model-z"),
        ])
        .expect("insert embeddings");

    let work_models = storage
        .list_embedding_models_in_space(work_space_id)
        .expect("list work models");
    assert_eq!(
        work_models,
        vec!["model-a".to_string(), "model-b".to_string()]
    );

    let notes_models = storage
        .list_embedding_models_in_space(notes_space_id)
        .expect("list notes models");
    assert_eq!(notes_models, vec!["model-z".to_string()]);
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
            DocumentTitleSource::Extracted,
            "hash-active",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert active doc");
    let inactive_doc_id = storage
        .upsert_document(
            collection_id,
            "src/inactive.rs",
            "inactive",
            DocumentTitleSource::Extracted,
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
                    kind: FinalChunkKind::Section,
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 100,
                    length: 50,
                    heading: None,
                    kind: FinalChunkKind::Section,
                },
                super::ChunkInsert {
                    seq: 2,
                    offset: 150,
                    length: 25,
                    heading: None,
                    kind: FinalChunkKind::Section,
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
                kind: FinalChunkKind::Section,
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
        .get_unembedded_chunks("model-a", 0, 10)
        .expect("query backlog");
    assert_eq!(backlog.len(), 2);
    assert_eq!(backlog[0].chunk_id, active_chunk_ids[1]);
    assert_eq!(backlog[0].doc_path, "src/active.rs");
    assert_eq!(
        backlog[0].collection_path,
        std::path::PathBuf::from("/tmp/api")
    );
    assert_eq!(backlog[0].space_name, "work");
    assert_eq!(backlog[0].offset, 100);
    assert_eq!(backlog[0].length, 50);
    assert_eq!(backlog[1].chunk_id, active_chunk_ids[2]);
    assert_ne!(backlog[0].chunk_id, inactive_chunk_ids[0]);

    let limited = storage
        .get_unembedded_chunks("model-a", 0, 1)
        .expect("query limited backlog");
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].chunk_id, active_chunk_ids[1]);

    let next_page = storage
        .get_unembedded_chunks("model-a", limited[0].chunk_id, 10)
        .expect("query paged backlog");
    assert_eq!(next_page.len(), 1);
    assert_eq!(next_page[0].chunk_id, active_chunk_ids[2]);
}

#[test]
fn get_unembedded_chunks_can_scope_backlog_to_selected_collections() {
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

    let work_doc_id = storage
        .upsert_document(
            work_collection_id,
            "src/lib.rs",
            "work doc",
            DocumentTitleSource::Extracted,
            "hash-work",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert work doc");
    let notes_doc_id = storage
        .upsert_document(
            notes_collection_id,
            "docs/guide.md",
            "notes doc",
            DocumentTitleSource::Extracted,
            "hash-notes",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert notes doc");

    let work_chunk_ids = storage
        .insert_chunks(
            work_doc_id,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 32,
                heading: None,
                kind: FinalChunkKind::Section,
            }],
        )
        .expect("insert work chunk");
    let notes_chunk_ids = storage
        .insert_chunks(
            notes_doc_id,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 48,
                heading: None,
                kind: FinalChunkKind::Section,
            }],
        )
        .expect("insert notes chunk");

    let work_only = storage
        .get_unembedded_chunks_in_collections("model-a", &[work_collection_id], 0, 10)
        .expect("query work-only backlog");
    assert_eq!(work_only.len(), 1);
    assert_eq!(work_only[0].chunk_id, work_chunk_ids[0]);
    assert_eq!(work_only[0].space_name, "work");

    let notes_only = storage
        .get_unembedded_chunks_in_space("model-a", notes_space_id, 0, 10)
        .expect("query notes-only backlog");
    assert_eq!(notes_only.len(), 1);
    assert_eq!(notes_only[0].chunk_id, notes_chunk_ids[0]);
    assert_eq!(notes_only[0].space_name, "notes");
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
            DocumentTitleSource::Extracted,
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
                    kind: FinalChunkKind::Section,
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 100,
                    length: 80,
                    heading: Some("# Usage".to_string()),
                    kind: FinalChunkKind::Section,
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
    assert_eq!(dirty[0].doc_title_source, DocumentTitleSource::Extracted);
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
            DocumentTitleSource::Extracted,
            "hash-a",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc a");
    let doc_b = storage
        .upsert_document(
            collection_id,
            "src/b.rs",
            "b",
            DocumentTitleSource::Extracted,
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
            DocumentTitleSource::Extracted,
            "hash-a",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert work doc a");
    storage
        .upsert_document(
            work_collection_id,
            "src/b.rs",
            "b",
            DocumentTitleSource::Extracted,
            "hash-b",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert work doc b");
    storage
        .upsert_document(
            notes_collection_id,
            "README.md",
            "readme",
            DocumentTitleSource::Extracted,
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
            DocumentTitleSource::Extracted,
            "hash-lib",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert work doc");
    let notes_doc = storage
        .upsert_document(
            notes_collection_id,
            "README.md",
            "readme",
            DocumentTitleSource::Extracted,
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
                    kind: FinalChunkKind::Section,
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 10,
                    length: 12,
                    heading: None,
                    kind: FinalChunkKind::Section,
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
                kind: FinalChunkKind::Section,
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
            DocumentTitleSource::Extracted,
            "hash-lib",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert work doc");
    let notes_doc = storage
        .upsert_document(
            notes_collection_id,
            "README.md",
            "readme",
            DocumentTitleSource::Extracted,
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
                    kind: FinalChunkKind::Section,
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 10,
                    length: 12,
                    heading: None,
                    kind: FinalChunkKind::Section,
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
                kind: FinalChunkKind::Section,
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
fn per_collection_count_methods_return_expected_values() {
    let tmp = tempdir().expect("create tempdir");
    let storage = Storage::new(&tmp.path().join("cache")).expect("create storage");
    let space_id = storage
        .create_space("work", None)
        .expect("create work space");
    let collection_id = storage
        .create_collection(
            space_id,
            "api",
            std::path::Path::new("/tmp/work-api"),
            None,
            None,
        )
        .expect("create collection");
    let doc_a = storage
        .upsert_document(
            collection_id,
            "a.rs",
            "a",
            DocumentTitleSource::Extracted,
            "hash-a",
            "2026-03-01T10:00:00Z",
        )
        .expect("insert doc a");
    let doc_b = storage
        .upsert_document(
            collection_id,
            "b.rs",
            "b",
            DocumentTitleSource::Extracted,
            "hash-b",
            "2026-03-01T10:01:00Z",
        )
        .expect("insert doc b");

    let chunk_ids = storage
        .insert_chunks(
            doc_a,
            &[
                super::ChunkInsert {
                    seq: 0,
                    offset: 0,
                    length: 10,
                    heading: None,
                    kind: FinalChunkKind::Section,
                },
                super::ChunkInsert {
                    seq: 1,
                    offset: 10,
                    length: 12,
                    heading: None,
                    kind: FinalChunkKind::Section,
                },
            ],
        )
        .expect("insert chunks for doc a");
    storage
        .insert_chunks(
            doc_b,
            &[super::ChunkInsert {
                seq: 0,
                offset: 0,
                length: 9,
                heading: None,
                kind: FinalChunkKind::Section,
            }],
        )
        .expect("insert chunks for doc b");
    storage
        .insert_embeddings(&[(chunk_ids[0], "model-a"), (chunk_ids[0], "model-b")])
        .expect("insert embeddings");
    storage
        .deactivate_document(doc_b)
        .expect("deactivate second doc");

    assert_eq!(
        storage
            .count_documents_in_collection(collection_id, false)
            .expect("count all docs"),
        2
    );
    assert_eq!(
        storage
            .count_documents_in_collection(collection_id, true)
            .expect("count active docs"),
        1
    );
    assert_eq!(
        storage
            .count_chunks_in_collection(collection_id)
            .expect("count chunks"),
        3
    );
    assert_eq!(
        storage
            .count_embedded_chunks_in_collection(collection_id)
            .expect("count embedded chunks"),
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

    std::fs::write(
        cache_dir.join("spaces/work/tantivy/seg1.bin"),
        vec![b'x'; 10],
    )
    .expect("write tantivy segment 1");
    std::fs::write(
        cache_dir.join("spaces/work/tantivy/segments/seg2.bin"),
        vec![b'y'; 4],
    )
    .expect("write tantivy segment 2");
    std::fs::write(cache_dir.join("spaces/work/vectors.usearch"), vec![b'v'; 7])
        .expect("write work vectors");
    std::fs::write(
        cache_dir.join("spaces/notes/vectors.usearch"),
        vec![b'w'; 3],
    )
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
