use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

use crate::error::{CoreError, Result};
use kbolt_types::{DiskUsage, KboltError};
use rusqlite::{params, params_from_iter, Connection, Error, ErrorCode};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Value, FAST, INDEXED, STORED, TEXT};
use tantivy::{Index, IndexWriter, TantivyDocument, Term};
use usearch::{IndexOptions, MetricKind, ScalarKind};

const DB_FILE: &str = "meta.sqlite";
const DEFAULT_SPACE_NAME: &str = "default";
const SPACES_DIR: &str = "spaces";
const TANTIVY_DIR_NAME: &str = "tantivy";
const USEARCH_FILENAME: &str = "vectors.usearch";

pub struct Storage {
    db: Mutex<Connection>,
    cache_dir: PathBuf,
    spaces: RwLock<HashMap<String, SpaceIndexes>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceRow {
    pub id: i64,
    pub name: String,
    pub description: Option<String>,
    pub created: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionRow {
    pub id: i64,
    pub space_id: i64,
    pub name: String,
    pub path: PathBuf,
    pub description: Option<String>,
    pub extensions: Option<Vec<String>>,
    pub created: String,
    pub updated: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentRow {
    pub id: i64,
    pub collection_id: i64,
    pub path: String,
    pub title: String,
    pub hash: String,
    pub modified: String,
    pub active: bool,
    pub deactivated_at: Option<String>,
    pub fts_dirty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRow {
    pub id: i64,
    pub doc_id: i64,
    pub seq: i32,
    pub offset: usize,
    pub length: usize,
    pub heading: Option<String>,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkInsert {
    pub seq: i32,
    pub offset: usize,
    pub length: usize,
    pub heading: Option<String>,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedRecord {
    pub chunk_id: i64,
    pub doc_path: String,
    pub collection_path: PathBuf,
    pub space_name: String,
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtsDirtyRecord {
    pub doc_id: i64,
    pub doc_path: String,
    pub doc_title: String,
    pub doc_hash: String,
    pub collection_path: PathBuf,
    pub space_name: String,
    pub chunks: Vec<ChunkRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TantivyEntry {
    pub chunk_id: i64,
    pub doc_id: i64,
    pub filepath: String,
    pub title: String,
    pub heading: Option<String>,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BM25Hit {
    pub chunk_id: i64,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseHit {
    pub chunk_id: i64,
    pub distance: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpaceResolution {
    Found(SpaceRow),
    Ambiguous(Vec<String>),
    NotFound,
}

struct SpaceIndexes {
    _tantivy_dir: PathBuf,
    usearch_path: PathBuf,
    tantivy_index: Index,
    tantivy_writer: Mutex<IndexWriter>,
    usearch_index: RwLock<usearch::Index>,
    fields: TantivyFields,
}

#[derive(Debug, Clone, Copy)]
struct TantivyFields {
    chunk_id: Field,
    doc_id: Field,
    filepath: Field,
    title: Field,
    heading: Field,
    body: Field,
}

impl Storage {
    pub fn new(cache_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)?;
        let db_path = cache_dir.join(DB_FILE);
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            r#"
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS spaces (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    created     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS collections (
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

CREATE TABLE IF NOT EXISTS documents (
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
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id, active);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_documents_fts_dirty ON documents(fts_dirty) WHERE fts_dirty = 1;

CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY,
    doc_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq      INTEGER NOT NULL,
    offset   INTEGER NOT NULL,
    length   INTEGER NOT NULL,
    heading  TEXT,
    kind     TEXT NOT NULL DEFAULT 'section',
    UNIQUE(doc_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id    INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

CREATE TABLE IF NOT EXISTS llm_cache (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL,
    created TEXT NOT NULL
);
"#,
        )?;

        conn.execute(
            "INSERT OR IGNORE INTO spaces (name, description) VALUES (?1, NULL)",
            params![DEFAULT_SPACE_NAME],
        )?;

        let mut stmt = conn.prepare("SELECT name FROM spaces ORDER BY name ASC")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let space_names = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);

        let storage = Self {
            db: Mutex::new(conn),
            cache_dir: cache_dir.to_path_buf(),
            spaces: RwLock::new(HashMap::new()),
        };

        for space_name in space_names {
            storage.open_space(&space_name)?;
        }

        Ok(storage)
    }

    pub fn open_space(&self, name: &str) -> Result<()> {
        let _space = self.get_space(name)?;

        {
            let spaces = self
                .spaces
                .read()
                .map_err(|_| CoreError::poisoned("spaces"))?;
            if spaces.contains_key(name) {
                return Ok(());
            }
        }

        let (tantivy_dir, usearch_path) = self.space_paths(name);
        std::fs::create_dir_all(&tantivy_dir)?;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&usearch_path)?;
        let tantivy_index = open_or_create_tantivy_index(&tantivy_dir)?;
        let usearch_index = open_or_create_usearch_index(&usearch_path)?;
        let fields = tantivy_fields_from_schema(&tantivy_index.schema())?;
        let tantivy_writer = tantivy_index.writer(50_000_000)?;

        let mut spaces = self
            .spaces
            .write()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        spaces.entry(name.to_string()).or_insert(SpaceIndexes {
            _tantivy_dir: tantivy_dir,
            usearch_path,
            tantivy_index,
            tantivy_writer: Mutex::new(tantivy_writer),
            usearch_index: RwLock::new(usearch_index),
            fields,
        });

        Ok(())
    }

    pub fn close_space(&self, name: &str) -> Result<()> {
        let _space = self.get_space(name)?;
        let mut spaces = self
            .spaces
            .write()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let _removed = spaces.remove(name);
        Ok(())
    }

    pub fn create_space(&self, name: &str, description: Option<&str>) -> Result<i64> {
        let space_id = {
            let conn = self
                .db
                .lock()
                .map_err(|_| CoreError::poisoned("database"))?;

            let result = conn.execute(
                "INSERT INTO spaces (name, description) VALUES (?1, ?2)",
                params![name, description],
            );

            match result {
                Ok(_) => conn.last_insert_rowid(),
                Err(err) => {
                    return match err {
                        Error::SqliteFailure(sqlite_err, _)
                            if sqlite_err.code == ErrorCode::ConstraintViolation =>
                        {
                            Err(KboltError::SpaceAlreadyExists {
                                name: name.to_string(),
                            }
                            .into())
                        }
                        other => Err(other.into()),
                    };
                }
            }
        };

        if let Err(open_err) = self.open_space(name) {
            let rollback_result = self
                .db
                .lock()
                .map_err(|_| CoreError::poisoned("database"))?
                .execute("DELETE FROM spaces WHERE id = ?1", params![space_id]);

            if let Err(rollback_err) = rollback_result {
                return Err(CoreError::Internal(format!(
                    "failed to provision indexes for space '{name}': {open_err}; rollback failed: {rollback_err}"
                )));
            }

            return Err(open_err);
        }

        Ok(space_id)
    }

    pub fn get_space(&self, name: &str) -> Result<SpaceRow> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut stmt =
            conn.prepare("SELECT id, name, description, created FROM spaces WHERE name = ?1")?;

        let row = stmt.query_row(params![name], |row| {
            Ok(SpaceRow {
                id: row.get(0)?,
                name: row.get(1)?,
                description: row.get(2)?,
                created: row.get(3)?,
            })
        });

        match row {
            Ok(space) => Ok(space),
            Err(Error::QueryReturnedNoRows) => Err(KboltError::SpaceNotFound {
                name: name.to_string(),
            }
            .into()),
            Err(err) => Err(err.into()),
        }
    }

    pub fn list_spaces(&self) -> Result<Vec<SpaceRow>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut stmt = conn.prepare(
            "SELECT id, name, description, created
             FROM spaces
             ORDER BY name ASC",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(SpaceRow {
                id: row.get(0)?,
                name: row.get(1)?,
                description: row.get(2)?,
                created: row.get(3)?,
            })
        })?;

        let spaces = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(spaces)
    }

    pub fn find_space_for_collection(&self, collection: &str) -> Result<SpaceResolution> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut stmt = conn.prepare(
            "SELECT s.id, s.name, s.description, s.created
             FROM spaces s
             JOIN collections c ON c.space_id = s.id
             WHERE c.name = ?1
             ORDER BY s.name ASC",
        )?;
        let rows = stmt.query_map(params![collection], |row| {
            Ok(SpaceRow {
                id: row.get(0)?,
                name: row.get(1)?,
                description: row.get(2)?,
                created: row.get(3)?,
            })
        })?;
        let matches = rows.collect::<std::result::Result<Vec<_>, _>>()?;

        if matches.is_empty() {
            return Ok(SpaceResolution::NotFound);
        }

        if matches.len() == 1 {
            return Ok(SpaceResolution::Found(matches[0].clone()));
        }

        let spaces = matches.into_iter().map(|space| space.name).collect();
        Ok(SpaceResolution::Ambiguous(spaces))
    }

    pub fn delete_space(&self, name: &str) -> Result<()> {
        if name == DEFAULT_SPACE_NAME {
            let conn = self
                .db
                .lock()
                .map_err(|_| CoreError::poisoned("database"))?;
            conn.execute(
                "DELETE FROM collections
                 WHERE space_id = (SELECT id FROM spaces WHERE name = ?1)",
                params![name],
            )?;
            drop(conn);

            self.unload_space(name)?;
            self.remove_space_artifacts(name)?;
            self.open_space(name)?;
            return Ok(());
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let deleted = conn.execute("DELETE FROM spaces WHERE name = ?1", params![name])?;
        drop(conn);

        if deleted == 0 {
            return Err(KboltError::SpaceNotFound {
                name: name.to_string(),
            }
            .into());
        }

        self.unload_space(name)?;
        self.remove_space_artifacts(name)?;

        Ok(())
    }

    pub fn rename_space(&self, old: &str, new: &str) -> Result<()> {
        if old == DEFAULT_SPACE_NAME {
            return Err(
                KboltError::Config("cannot rename reserved space: default".to_string()).into(),
            );
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let result = conn.execute(
            "UPDATE spaces SET name = ?1 WHERE name = ?2",
            params![new, old],
        );
        drop(conn);

        match result {
            Ok(0) => Err(KboltError::SpaceNotFound {
                name: old.to_string(),
            }
            .into()),
            Ok(_) => {
                if let Err(rename_err) = self.rename_space_artifacts(old, new) {
                    let rollback = self
                        .db
                        .lock()
                        .map_err(|_| CoreError::poisoned("database"))?
                        .execute(
                            "UPDATE spaces SET name = ?1 WHERE name = ?2",
                            params![old, new],
                        );

                    if let Err(rollback_err) = rollback {
                        return Err(CoreError::Internal(format!(
                            "failed to rename space artifacts from '{old}' to '{new}': {rename_err}; rollback failed: {rollback_err}"
                        )));
                    }

                    return Err(rename_err);
                }

                self.unload_space(old)?;
                if let Err(open_err) = self.open_space(new) {
                    let _ = self.rename_space_artifacts(new, old);
                    let _ = self
                        .db
                        .lock()
                        .map_err(|_| CoreError::poisoned("database"))?
                        .execute(
                            "UPDATE spaces SET name = ?1 WHERE name = ?2",
                            params![old, new],
                        );
                    let _ = self.open_space(old);
                    return Err(open_err);
                }

                Ok(())
            }
            Err(Error::SqliteFailure(sqlite_err, _))
                if sqlite_err.code == ErrorCode::ConstraintViolation =>
            {
                Err(KboltError::SpaceAlreadyExists {
                    name: new.to_string(),
                }
                .into())
            }
            Err(err) => Err(err.into()),
        }
    }

    pub fn update_space_description(&self, name: &str, description: &str) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let updated = conn.execute(
            "UPDATE spaces SET description = ?1 WHERE name = ?2",
            params![description, name],
        )?;

        if updated == 0 {
            return Err(KboltError::SpaceNotFound {
                name: name.to_string(),
            }
            .into());
        }

        Ok(())
    }

    pub fn create_collection(
        &self,
        space_id: i64,
        name: &str,
        path: &Path,
        description: Option<&str>,
        extensions: Option<&[String]>,
    ) -> Result<i64> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let space_name = lookup_space_name(&conn, space_id)?;
        let extensions_json = serialize_extensions(extensions)?;
        let result = conn.execute(
            "INSERT INTO collections (space_id, name, path, description, extensions, created, updated)
             VALUES (?1, ?2, ?3, ?4, ?5, strftime('%Y-%m-%dT%H:%M:%SZ','now'), strftime('%Y-%m-%dT%H:%M:%SZ','now'))",
            params![space_id, name, path.to_string_lossy(), description, extensions_json],
        );

        match result {
            Ok(_) => Ok(conn.last_insert_rowid()),
            Err(Error::SqliteFailure(sqlite_err, _))
                if sqlite_err.code == ErrorCode::ConstraintViolation =>
            {
                Err(KboltError::CollectionAlreadyExists {
                    name: name.to_string(),
                    space: space_name,
                }
                .into())
            }
            Err(err) => Err(err.into()),
        }
    }

    pub fn get_collection(&self, space_id: i64, name: &str) -> Result<CollectionRow> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut stmt = conn.prepare(
            "SELECT id, space_id, name, path, description, extensions, created, updated
             FROM collections
             WHERE space_id = ?1 AND name = ?2",
        )?;

        let result = stmt.query_row(params![space_id, name], decode_collection_row);
        match result {
            Ok(row) => Ok(row),
            Err(Error::QueryReturnedNoRows) => Err(KboltError::CollectionNotFound {
                name: name.to_string(),
            }
            .into()),
            Err(err) => Err(err.into()),
        }
    }

    pub fn list_collections(&self, space_id: Option<i64>) -> Result<Vec<CollectionRow>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let (sql, params): (&str, Vec<i64>) = match space_id {
            Some(id) => (
                "SELECT id, space_id, name, path, description, extensions, created, updated
                 FROM collections
                 WHERE space_id = ?1
                 ORDER BY name ASC",
                vec![id],
            ),
            None => (
                "SELECT id, space_id, name, path, description, extensions, created, updated
                 FROM collections
                 ORDER BY space_id ASC, name ASC",
                Vec::new(),
            ),
        };

        let mut stmt = conn.prepare(sql)?;
        let rows = if params.is_empty() {
            stmt.query_map([], decode_collection_row)?
        } else {
            stmt.query_map(params![params[0]], decode_collection_row)?
        };
        let collections = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(collections)
    }

    pub fn delete_collection(&self, space_id: i64, name: &str) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _space_name = lookup_space_name(&conn, space_id)?;

        let deleted = conn.execute(
            "DELETE FROM collections WHERE space_id = ?1 AND name = ?2",
            params![space_id, name],
        )?;

        if deleted == 0 {
            return Err(KboltError::CollectionNotFound {
                name: name.to_string(),
            }
            .into());
        }

        Ok(())
    }

    pub fn rename_collection(&self, space_id: i64, old: &str, new: &str) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let space_name = lookup_space_name(&conn, space_id)?;
        let result = conn.execute(
            "UPDATE collections
             SET name = ?1, updated = strftime('%Y-%m-%dT%H:%M:%SZ','now')
             WHERE space_id = ?2 AND name = ?3",
            params![new, space_id, old],
        );

        match result {
            Ok(0) => Err(KboltError::CollectionNotFound {
                name: old.to_string(),
            }
            .into()),
            Ok(_) => Ok(()),
            Err(Error::SqliteFailure(sqlite_err, _))
                if sqlite_err.code == ErrorCode::ConstraintViolation =>
            {
                Err(KboltError::CollectionAlreadyExists {
                    name: new.to_string(),
                    space: space_name,
                }
                .into())
            }
            Err(err) => Err(err.into()),
        }
    }

    pub fn update_collection_description(
        &self,
        space_id: i64,
        name: &str,
        desc: &str,
    ) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _space_name = lookup_space_name(&conn, space_id)?;

        let updated = conn.execute(
            "UPDATE collections
             SET description = ?1, updated = strftime('%Y-%m-%dT%H:%M:%SZ','now')
             WHERE space_id = ?2 AND name = ?3",
            params![desc, space_id, name],
        )?;

        if updated == 0 {
            return Err(KboltError::CollectionNotFound {
                name: name.to_string(),
            }
            .into());
        }

        Ok(())
    }

    pub fn update_collection_timestamp(&self, collection_id: i64) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let updated = conn.execute(
            "UPDATE collections
             SET updated = strftime('%Y-%m-%dT%H:%M:%SZ','now')
             WHERE id = ?1",
            params![collection_id],
        )?;

        if updated == 0 {
            return Err(KboltError::CollectionNotFound {
                name: format!("id={collection_id}"),
            }
            .into());
        }

        Ok(())
    }

    pub fn upsert_document(
        &self,
        collection_id: i64,
        path: &str,
        title: &str,
        hash: &str,
        modified: &str,
    ) -> Result<i64> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _collection_name = lookup_collection_name(&conn, collection_id)?;

        conn.execute(
            "INSERT INTO documents (collection_id, path, title, hash, modified, active, deactivated_at, fts_dirty)
             VALUES (?1, ?2, ?3, ?4, ?5, 1, NULL, 1)
             ON CONFLICT(collection_id, path) DO UPDATE SET
                 title = excluded.title,
                 hash = excluded.hash,
                 modified = excluded.modified,
                 active = 1,
                 deactivated_at = NULL,
                 fts_dirty = 1",
            params![collection_id, path, title, hash, modified],
        )?;

        let id: i64 = conn.query_row(
            "SELECT id FROM documents WHERE collection_id = ?1 AND path = ?2",
            params![collection_id, path],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    pub fn get_document_by_path(
        &self,
        collection_id: i64,
        path: &str,
    ) -> Result<Option<DocumentRow>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _collection_name = lookup_collection_name(&conn, collection_id)?;
        let mut stmt = conn.prepare(
            "SELECT id, collection_id, path, title, hash, modified, active, deactivated_at, fts_dirty
             FROM documents
             WHERE collection_id = ?1 AND path = ?2",
        )?;

        let result = stmt.query_row(params![collection_id, path], decode_document_row);
        match result {
            Ok(row) => Ok(Some(row)),
            Err(Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    pub fn list_documents(
        &self,
        collection_id: i64,
        active_only: bool,
    ) -> Result<Vec<DocumentRow>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _collection_name = lookup_collection_name(&conn, collection_id)?;

        let sql = if active_only {
            "SELECT id, collection_id, path, title, hash, modified, active, deactivated_at, fts_dirty
             FROM documents
             WHERE collection_id = ?1 AND active = 1
             ORDER BY path ASC"
        } else {
            "SELECT id, collection_id, path, title, hash, modified, active, deactivated_at, fts_dirty
             FROM documents
             WHERE collection_id = ?1
             ORDER BY path ASC"
        };

        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(params![collection_id], decode_document_row)?;
        let docs = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(docs)
    }

    pub fn get_document_by_hash_prefix(&self, prefix: &str) -> Result<Vec<DocumentRow>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let pattern = format!("{prefix}%");
        let mut stmt = conn.prepare(
            "SELECT id, collection_id, path, title, hash, modified, active, deactivated_at, fts_dirty
             FROM documents
             WHERE hash LIKE ?1
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![pattern], decode_document_row)?;
        let docs = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(docs)
    }

    pub fn deactivate_document(&self, doc_id: i64) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let updated = conn.execute(
            "UPDATE documents
             SET active = 0,
                 deactivated_at = CASE
                    WHEN active = 1 THEN strftime('%Y-%m-%dT%H:%M:%SZ','now')
                    ELSE deactivated_at
                 END
             WHERE id = ?1",
            params![doc_id],
        )?;

        if updated == 0 {
            return Err(KboltError::DocumentNotFound {
                path: format!("id={doc_id}"),
            }
            .into());
        }

        Ok(())
    }

    pub fn reactivate_document(&self, doc_id: i64) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let updated = conn.execute(
            "UPDATE documents
             SET active = 1, deactivated_at = NULL
             WHERE id = ?1",
            params![doc_id],
        )?;

        if updated == 0 {
            return Err(KboltError::DocumentNotFound {
                path: format!("id={doc_id}"),
            }
            .into());
        }

        Ok(())
    }

    pub fn reap_documents(&self, older_than_days: u32) -> Result<Vec<i64>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let modifier = format!("-{} days", older_than_days);
        let mut stmt = conn.prepare(
            "SELECT id
             FROM documents
             WHERE active = 0
               AND deactivated_at IS NOT NULL
               AND deactivated_at <= strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?1)
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![modifier], |row| row.get::<_, i64>(0))?;
        let doc_ids = rows.collect::<std::result::Result<Vec<_>, _>>()?;

        if doc_ids.is_empty() {
            return Ok(doc_ids);
        }

        let placeholders = vec!["?"; doc_ids.len()].join(", ");
        let sql = format!("DELETE FROM documents WHERE id IN ({placeholders})");
        conn.execute(&sql, params_from_iter(doc_ids.iter()))?;

        Ok(doc_ids)
    }

    pub fn insert_chunks(&self, doc_id: i64, chunks: &[ChunkInsert]) -> Result<Vec<i64>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _doc = lookup_document_id(&conn, doc_id)?;

        let tx = conn.unchecked_transaction()?;
        let mut stmt = tx.prepare(
            "INSERT INTO chunks (doc_id, seq, offset, length, heading, kind)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;

        let mut ids = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            stmt.execute(params![
                doc_id,
                chunk.seq,
                chunk.offset as i64,
                chunk.length as i64,
                chunk.heading,
                chunk.kind,
            ])?;
            ids.push(tx.last_insert_rowid());
        }
        drop(stmt);
        tx.commit()?;
        Ok(ids)
    }

    pub fn delete_chunks_for_document(&self, doc_id: i64) -> Result<Vec<i64>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _doc = lookup_document_id(&conn, doc_id)?;

        let mut stmt = conn.prepare("SELECT id FROM chunks WHERE doc_id = ?1 ORDER BY seq ASC")?;
        let rows = stmt.query_map(params![doc_id], |row| row.get::<_, i64>(0))?;
        let chunk_ids = rows.collect::<std::result::Result<Vec<_>, _>>()?;

        conn.execute("DELETE FROM chunks WHERE doc_id = ?1", params![doc_id])?;
        Ok(chunk_ids)
    }

    pub fn get_chunks_for_document(&self, doc_id: i64) -> Result<Vec<ChunkRow>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _doc = lookup_document_id(&conn, doc_id)?;

        let mut stmt = conn.prepare(
            "SELECT id, doc_id, seq, offset, length, heading, kind
             FROM chunks
             WHERE doc_id = ?1
             ORDER BY seq ASC",
        )?;
        let rows = stmt.query_map(params![doc_id], decode_chunk_row)?;
        let chunks = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(chunks)
    }

    pub fn get_chunks(&self, chunk_ids: &[i64]) -> Result<Vec<ChunkRow>> {
        if chunk_ids.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let placeholders = vec!["?"; chunk_ids.len()].join(", ");
        let sql = format!(
            "SELECT id, doc_id, seq, offset, length, heading, kind
             FROM chunks
             WHERE id IN ({placeholders})
             ORDER BY id ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(chunk_ids.iter()), decode_chunk_row)?;
        let chunks = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(chunks)
    }

    pub fn insert_embeddings(&self, entries: &[(i64, &str)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let tx = conn.unchecked_transaction()?;
        let mut stmt = tx.prepare(
            "INSERT INTO embeddings (chunk_id, model, embedded_at)
             VALUES (?1, ?2, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
             ON CONFLICT(chunk_id, model) DO UPDATE SET
               embedded_at = excluded.embedded_at",
        )?;

        for (chunk_id, model) in entries {
            stmt.execute(params![chunk_id, model])?;
        }

        drop(stmt);
        tx.commit()?;
        Ok(())
    }

    pub fn get_unembedded_chunks(&self, model: &str, limit: usize) -> Result<Vec<EmbedRecord>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let sql_limit = i64::try_from(limit)
            .map_err(|_| CoreError::Internal("limit too large for sqlite".to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT c.id, d.path, col.path, s.name, c.offset, c.length
             FROM chunks c
             JOIN documents d ON d.id = c.doc_id
             JOIN collections col ON col.id = d.collection_id
             JOIN spaces s ON s.id = col.space_id
             LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model = ?1
             WHERE d.active = 1 AND e.chunk_id IS NULL
             ORDER BY c.id ASC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![model, sql_limit], |row| {
            let offset_value: i64 = row.get(4)?;
            let length_value: i64 = row.get(5)?;
            Ok(EmbedRecord {
                chunk_id: row.get(0)?,
                doc_path: row.get(1)?,
                collection_path: PathBuf::from(row.get::<_, String>(2)?),
                space_name: row.get(3)?,
                offset: offset_value as usize,
                length: length_value as usize,
            })
        })?;

        let records = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(records)
    }

    pub fn delete_embeddings_for_model(&self, model: &str) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let deleted = conn.execute("DELETE FROM embeddings WHERE model = ?1", params![model])?;
        Ok(deleted)
    }

    pub fn count_embeddings(&self) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let count: i64 = conn.query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    pub fn index_tantivy(&self, space: &str, entries: &[TantivyEntry]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        self.open_space(space)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let space_indexes = spaces.get(space).ok_or_else(|| KboltError::SpaceNotFound {
            name: space.to_string(),
        })?;
        let writer = space_indexes
            .tantivy_writer
            .lock()
            .map_err(|_| CoreError::poisoned("tantivy writer"))?;

        for entry in entries {
            let chunk_id = u64::try_from(entry.chunk_id).map_err(|_| {
                CoreError::Internal(format!(
                    "chunk_id must be non-negative for tantivy indexing: {}",
                    entry.chunk_id
                ))
            })?;
            let doc_id = u64::try_from(entry.doc_id).map_err(|_| {
                CoreError::Internal(format!(
                    "doc_id must be non-negative for tantivy indexing: {}",
                    entry.doc_id
                ))
            })?;

            let mut doc = TantivyDocument::default();
            doc.add_u64(space_indexes.fields.chunk_id, chunk_id);
            doc.add_u64(space_indexes.fields.doc_id, doc_id);
            doc.add_text(space_indexes.fields.filepath, &entry.filepath);
            doc.add_text(space_indexes.fields.title, &entry.title);
            if let Some(heading) = &entry.heading {
                doc.add_text(space_indexes.fields.heading, heading);
            }
            doc.add_text(space_indexes.fields.body, &entry.body);
            writer.add_document(doc)?;
        }

        Ok(())
    }

    pub fn delete_tantivy(&self, space: &str, chunk_ids: &[i64]) -> Result<()> {
        if chunk_ids.is_empty() {
            return Ok(());
        }

        self.open_space(space)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let space_indexes = spaces.get(space).ok_or_else(|| KboltError::SpaceNotFound {
            name: space.to_string(),
        })?;
        let writer = space_indexes
            .tantivy_writer
            .lock()
            .map_err(|_| CoreError::poisoned("tantivy writer"))?;

        for chunk_id in chunk_ids {
            let chunk_key = u64::try_from(*chunk_id).map_err(|_| {
                CoreError::Internal(format!(
                    "chunk_id must be non-negative for tantivy delete: {chunk_id}"
                ))
            })?;
            writer.delete_term(Term::from_field_u64(space_indexes.fields.chunk_id, chunk_key));
        }

        Ok(())
    }

    pub fn delete_tantivy_by_doc(&self, space: &str, doc_id: i64) -> Result<()> {
        self.open_space(space)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let space_indexes = spaces.get(space).ok_or_else(|| KboltError::SpaceNotFound {
            name: space.to_string(),
        })?;
        let writer = space_indexes
            .tantivy_writer
            .lock()
            .map_err(|_| CoreError::poisoned("tantivy writer"))?;
        let doc_key = u64::try_from(doc_id).map_err(|_| {
            CoreError::Internal(format!(
                "doc_id must be non-negative for tantivy delete-by-doc: {doc_id}"
            ))
        })?;
        writer.delete_term(Term::from_field_u64(space_indexes.fields.doc_id, doc_key));
        Ok(())
    }

    pub fn query_bm25(
        &self,
        space: &str,
        query: &str,
        fields: &[(&str, f32)],
        limit: usize,
    ) -> Result<Vec<BM25Hit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        if fields.is_empty() {
            return Err(CoreError::Internal(
                "bm25 query requires at least one field".to_string(),
            ));
        }

        self.open_space(space)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let space_indexes = spaces.get(space).ok_or_else(|| KboltError::SpaceNotFound {
            name: space.to_string(),
        })?;

        let query_fields = fields
            .iter()
            .map(|(name, _)| resolve_tantivy_field(space_indexes.fields, name))
            .collect::<Result<Vec<_>>>()?;
        let mut parser = QueryParser::for_index(&space_indexes.tantivy_index, query_fields);
        for (field_name, boost) in fields {
            let field = resolve_tantivy_field(space_indexes.fields, field_name)?;
            parser.set_field_boost(field, *boost);
        }

        let parsed_query = parser.parse_query(query).map_err(|err| {
            CoreError::Internal(format!("failed to parse bm25 query '{query}': {err}"))
        })?;
        let reader = space_indexes.tantivy_index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let docs = searcher.search(&parsed_query, &TopDocs::with_limit(limit))?;

        let mut hits = Vec::with_capacity(docs.len());
        for (score, address) in docs {
            let doc = searcher.doc::<TantivyDocument>(address)?;
            let chunk_id = doc
                .get_first(space_indexes.fields.chunk_id)
                .and_then(|value| value.as_u64())
                .ok_or_else(|| {
                    CoreError::Internal("tantivy hit missing chunk_id field".to_string())
                })?;
            hits.push(BM25Hit {
                chunk_id: chunk_id as i64,
                score,
            });
        }

        Ok(hits)
    }

    pub fn commit_tantivy(&self, space: &str) -> Result<()> {
        self.open_space(space)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let space_indexes = spaces.get(space).ok_or_else(|| KboltError::SpaceNotFound {
            name: space.to_string(),
        })?;
        let mut writer = space_indexes
            .tantivy_writer
            .lock()
            .map_err(|_| CoreError::poisoned("tantivy writer"))?;
        writer.commit()?;
        Ok(())
    }

    pub fn insert_usearch(&self, space: &str, key: i64, vector: &[f32]) -> Result<()> {
        self.batch_insert_usearch(space, &[(key, vector)])
    }

    pub fn batch_insert_usearch(&self, space: &str, entries: &[(i64, &[f32])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        self.open_space(space)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let space_indexes = spaces.get(space).ok_or_else(|| KboltError::SpaceNotFound {
            name: space.to_string(),
        })?;

        let expected_dimensions = entries[0].1.len();
        if expected_dimensions == 0 {
            return Err(CoreError::Internal(
                "cannot insert empty vector into usearch index".to_string(),
            ));
        }
        for (_, vector) in entries {
            if vector.len() != expected_dimensions {
                return Err(CoreError::Internal(format!(
                    "vector dimension mismatch in batch insert: expected {expected_dimensions}, got {}",
                    vector.len()
                )));
            }
        }

        let mut index = space_indexes
            .usearch_index
            .write()
            .map_err(|_| CoreError::poisoned("usearch index"))?;
        ensure_usearch_dimensions(&mut index, expected_dimensions)?;
        let target_capacity = index.size().saturating_add(entries.len());
        index.reserve(target_capacity).map_err(|err| {
            CoreError::Internal(format!("usearch reserve failed for {target_capacity} items: {err}"))
        })?;

        for (key, vector) in entries {
            let key = u64::try_from(*key).map_err(|_| {
                CoreError::Internal(format!(
                    "usearch key must be non-negative: {}",
                    *key
                ))
            })?;
            index
                .add::<f32>(key, vector)
                .map_err(|err| CoreError::Internal(format!("usearch add failed: {err}")))?;
        }

        save_usearch_index(&index, &space_indexes.usearch_path)?;
        Ok(())
    }

    pub fn delete_usearch(&self, space: &str, keys: &[i64]) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }

        self.open_space(space)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let space_indexes = spaces.get(space).ok_or_else(|| KboltError::SpaceNotFound {
            name: space.to_string(),
        })?;
        let index = space_indexes
            .usearch_index
            .write()
            .map_err(|_| CoreError::poisoned("usearch index"))?;

        for key in keys {
            let key = u64::try_from(*key).map_err(|_| {
                CoreError::Internal(format!(
                    "usearch key must be non-negative: {}",
                    *key
                ))
            })?;
            index
                .remove(key)
                .map_err(|err| CoreError::Internal(format!("usearch remove failed: {err}")))?;
        }

        save_usearch_index(&index, &space_indexes.usearch_path)?;
        Ok(())
    }

    pub fn query_dense(&self, space: &str, vector: &[f32], limit: usize) -> Result<Vec<DenseHit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        if vector.is_empty() {
            return Err(CoreError::Internal(
                "cannot query usearch with empty vector".to_string(),
            ));
        }

        self.open_space(space)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let space_indexes = spaces.get(space).ok_or_else(|| KboltError::SpaceNotFound {
            name: space.to_string(),
        })?;
        let index = space_indexes
            .usearch_index
            .read()
            .map_err(|_| CoreError::poisoned("usearch index"))?;

        if index.size() == 0 {
            return Ok(Vec::new());
        }
        if vector.len() != index.dimensions() {
            return Err(CoreError::Internal(format!(
                "query vector dimension mismatch: expected {}, got {}",
                index.dimensions(),
                vector.len()
            )));
        }

        let matches = index
            .search::<f32>(vector, limit)
            .map_err(|err| CoreError::Internal(format!("usearch query failed: {err}")))?;
        let hits = matches
            .keys
            .into_iter()
            .zip(matches.distances)
            .map(|(key, distance)| DenseHit {
                chunk_id: key as i64,
                distance,
            })
            .collect();
        Ok(hits)
    }

    pub fn count_usearch(&self, space: &str) -> Result<usize> {
        self.open_space(space)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let space_indexes = spaces.get(space).ok_or_else(|| KboltError::SpaceNotFound {
            name: space.to_string(),
        })?;
        let index = space_indexes
            .usearch_index
            .read()
            .map_err(|_| CoreError::poisoned("usearch index"))?;
        Ok(index.size())
    }

    pub fn clear_usearch(&self, space: &str) -> Result<()> {
        self.open_space(space)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        let space_indexes = spaces.get(space).ok_or_else(|| KboltError::SpaceNotFound {
            name: space.to_string(),
        })?;
        let index = space_indexes
            .usearch_index
            .write()
            .map_err(|_| CoreError::poisoned("usearch index"))?;
        index
            .reset()
            .map_err(|err| CoreError::Internal(format!("usearch clear failed: {err}")))?;
        std::fs::File::create(&space_indexes.usearch_path)?;
        Ok(())
    }

    pub fn get_fts_dirty_documents(&self) -> Result<Vec<FtsDirtyRecord>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let mut stmt = conn.prepare(
            "SELECT d.id, d.path, d.title, d.hash, c.path, s.name
             FROM documents d
             JOIN collections c ON c.id = d.collection_id
             JOIN spaces s ON s.id = c.space_id
             WHERE d.fts_dirty = 1
             ORDER BY d.id ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?;
        let headers = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);

        let mut records = Vec::with_capacity(headers.len());
        for (doc_id, doc_path, doc_title, doc_hash, collection_path, space_name) in headers {
            let chunks = load_chunks_for_doc(&conn, doc_id)?;
            records.push(FtsDirtyRecord {
                doc_id,
                doc_path,
                doc_title,
                doc_hash,
                collection_path: PathBuf::from(collection_path),
                space_name,
                chunks,
            });
        }

        Ok(records)
    }

    pub fn batch_clear_fts_dirty(&self, doc_ids: &[i64]) -> Result<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let placeholders = vec!["?"; doc_ids.len()].join(", ");
        let sql = format!("UPDATE documents SET fts_dirty = 0 WHERE id IN ({placeholders})");
        conn.execute(&sql, params_from_iter(doc_ids.iter()))?;
        Ok(())
    }

    pub fn cache_get(&self, key: &str) -> Result<Option<String>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let value = conn.query_row(
            "SELECT value FROM llm_cache WHERE key = ?1",
            params![key],
            |row| row.get::<_, String>(0),
        );
        match value {
            Ok(value) => Ok(Some(value)),
            Err(Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    pub fn cache_set(&self, key: &str, value: &str) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        conn.execute(
            "INSERT INTO llm_cache (key, value, created)
             VALUES (?1, ?2, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
             ON CONFLICT(key) DO UPDATE SET
               value = excluded.value,
               created = excluded.created",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn count_documents(&self, space_id: Option<i64>) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let count: i64 = match space_id {
            Some(space_id) => {
                let _space_name = lookup_space_name(&conn, space_id)?;
                conn.query_row(
                    "SELECT COUNT(*)
                     FROM documents d
                     JOIN collections c ON c.id = d.collection_id
                     WHERE c.space_id = ?1",
                    params![space_id],
                    |row| row.get(0),
                )?
            }
            None => conn.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?,
        };

        Ok(count as usize)
    }

    pub fn count_chunks(&self, space_id: Option<i64>) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let count: i64 = match space_id {
            Some(space_id) => {
                let _space_name = lookup_space_name(&conn, space_id)?;
                conn.query_row(
                    "SELECT COUNT(*)
                     FROM chunks c
                     JOIN documents d ON d.id = c.doc_id
                     JOIN collections col ON col.id = d.collection_id
                     WHERE col.space_id = ?1",
                    params![space_id],
                    |row| row.get(0),
                )?
            }
            None => conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?,
        };

        Ok(count as usize)
    }

    pub fn count_embedded_chunks(&self, space_id: Option<i64>) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let count: i64 = match space_id {
            Some(space_id) => {
                let _space_name = lookup_space_name(&conn, space_id)?;
                conn.query_row(
                    "SELECT COUNT(DISTINCT e.chunk_id)
                     FROM embeddings e
                     JOIN chunks c ON c.id = e.chunk_id
                     JOIN documents d ON d.id = c.doc_id
                     JOIN collections col ON col.id = d.collection_id
                     WHERE col.space_id = ?1",
                    params![space_id],
                    |row| row.get(0),
                )?
            }
            None => conn.query_row("SELECT COUNT(DISTINCT chunk_id) FROM embeddings", [], |row| {
                row.get(0)
            })?,
        };

        Ok(count as usize)
    }

    pub fn disk_usage(&self) -> Result<DiskUsage> {
        let sqlite_bytes = file_size_or_zero(&self.cache_dir.join(DB_FILE))?;

        let mut tantivy_bytes = 0_u64;
        let mut usearch_bytes = 0_u64;
        let spaces_dir = self.cache_dir.join(SPACES_DIR);
        if spaces_dir.exists() {
            for entry in std::fs::read_dir(&spaces_dir)? {
                let space_dir = entry?.path();
                if !space_dir.is_dir() {
                    continue;
                }

                tantivy_bytes += dir_size_or_zero(&space_dir.join(TANTIVY_DIR_NAME))?;
                usearch_bytes += file_size_or_zero(&space_dir.join(USEARCH_FILENAME))?;
            }
        }

        let models_bytes = dir_size_or_zero(&self.cache_dir.join("models"))?;
        let total_bytes = sqlite_bytes + tantivy_bytes + usearch_bytes + models_bytes;

        Ok(DiskUsage {
            sqlite_bytes,
            tantivy_bytes,
            usearch_bytes,
            models_bytes,
            total_bytes,
        })
    }

    fn unload_space(&self, name: &str) -> Result<()> {
        let mut spaces = self
            .spaces
            .write()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        spaces.remove(name);
        Ok(())
    }

    fn remove_space_artifacts(&self, name: &str) -> Result<()> {
        let space_root = self.space_root_path(name);
        if space_root.exists() {
            std::fs::remove_dir_all(space_root)?;
        }
        Ok(())
    }

    fn rename_space_artifacts(&self, old: &str, new: &str) -> Result<()> {
        let old_root = self.space_root_path(old);
        let new_root = self.space_root_path(new);
        if !old_root.exists() {
            return Ok(());
        }

        if new_root.exists() {
            return Err(CoreError::Internal(format!(
                "cannot rename space artifacts: destination already exists: {}",
                new_root.display()
            )));
        }

        if let Some(parent) = new_root.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(old_root, new_root)?;
        Ok(())
    }

    fn space_root_path(&self, name: &str) -> PathBuf {
        self.cache_dir.join(SPACES_DIR).join(name)
    }

    fn space_paths(&self, name: &str) -> (PathBuf, PathBuf) {
        let space_root = self.space_root_path(name);
        let tantivy_dir = space_root.join(TANTIVY_DIR_NAME);
        let usearch_path = space_root.join(USEARCH_FILENAME);
        (tantivy_dir, usearch_path)
    }
}

fn lookup_space_name(conn: &Connection, space_id: i64) -> Result<String> {
    let result = conn.query_row(
        "SELECT name FROM spaces WHERE id = ?1",
        params![space_id],
        |row| row.get::<_, String>(0),
    );
    match result {
        Ok(name) => Ok(name),
        Err(Error::QueryReturnedNoRows) => Err(KboltError::SpaceNotFound {
            name: format!("id={space_id}"),
        }
        .into()),
        Err(err) => Err(err.into()),
    }
}

fn lookup_collection_name(conn: &Connection, collection_id: i64) -> Result<String> {
    let result = conn.query_row(
        "SELECT name FROM collections WHERE id = ?1",
        params![collection_id],
        |row| row.get::<_, String>(0),
    );
    match result {
        Ok(name) => Ok(name),
        Err(Error::QueryReturnedNoRows) => Err(KboltError::CollectionNotFound {
            name: format!("id={collection_id}"),
        }
        .into()),
        Err(err) => Err(err.into()),
    }
}

fn lookup_document_id(conn: &Connection, doc_id: i64) -> Result<i64> {
    let result = conn.query_row(
        "SELECT id FROM documents WHERE id = ?1",
        params![doc_id],
        |row| row.get::<_, i64>(0),
    );
    match result {
        Ok(id) => Ok(id),
        Err(Error::QueryReturnedNoRows) => Err(KboltError::DocumentNotFound {
            path: format!("id={doc_id}"),
        }
        .into()),
        Err(err) => Err(err.into()),
    }
}

fn load_chunks_for_doc(conn: &Connection, doc_id: i64) -> Result<Vec<ChunkRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, doc_id, seq, offset, length, heading, kind
         FROM chunks
         WHERE doc_id = ?1
         ORDER BY seq ASC",
    )?;
    let rows = stmt.query_map(params![doc_id], decode_chunk_row)?;
    let chunks = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(chunks)
}

fn file_size_or_zero(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }

    let metadata = std::fs::metadata(path)?;
    if metadata.is_file() {
        Ok(metadata.len())
    } else {
        Ok(0)
    }
}

fn dir_size_or_zero(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }

    let mut total = 0_u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let child_path = entry.path();
        let metadata = std::fs::symlink_metadata(&child_path)?;
        if metadata.is_file() {
            total += metadata.len();
        } else if metadata.is_dir() {
            total += dir_size_or_zero(&child_path)?;
        }
    }

    Ok(total)
}

fn open_or_create_tantivy_index(path: &Path) -> Result<Index> {
    let meta_path = path.join("meta.json");
    if meta_path.exists() {
        return Ok(Index::open_in_dir(path)?);
    }

    Ok(Index::create_in_dir(path, tantivy_schema())?)
}

fn new_usearch_index(dimensions: usize) -> Result<usearch::Index> {
    let mut options = IndexOptions::default();
    options.dimensions = dimensions;
    options.metric = MetricKind::Cos;
    options.quantization = ScalarKind::F32;
    options.connectivity = 16;
    options.expansion_add = 200;
    options.expansion_search = 100;
    usearch::Index::new(&options)
        .map_err(|err| CoreError::Internal(format!("usearch init failed: {err}")))
}

fn open_or_create_usearch_index(path: &Path) -> Result<usearch::Index> {
    let index = new_usearch_index(256)?;
    let file_size = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    if file_size > 0 {
        let path = path
            .to_str()
            .ok_or_else(|| CoreError::Internal("invalid usearch path encoding".to_string()))?;
        index
            .load(path)
            .map_err(|err| CoreError::Internal(format!("usearch load failed: {err}")))?;
    }
    Ok(index)
}

fn ensure_usearch_dimensions(index: &mut usearch::Index, expected_dimensions: usize) -> Result<()> {
    if index.size() == 0 && index.dimensions() != expected_dimensions {
        *index = new_usearch_index(expected_dimensions)?;
        return Ok(());
    }

    if index.dimensions() != expected_dimensions {
        return Err(CoreError::Internal(format!(
            "usearch vector dimension mismatch: index expects {}, got {}",
            index.dimensions(),
            expected_dimensions
        )));
    }
    Ok(())
}

fn save_usearch_index(index: &usearch::Index, path: &Path) -> Result<()> {
    if index.size() == 0 {
        std::fs::File::create(path)?;
        return Ok(());
    }

    let path = path
        .to_str()
        .ok_or_else(|| CoreError::Internal("invalid usearch path encoding".to_string()))?;
    index
        .save(path)
        .map_err(|err| CoreError::Internal(format!("usearch save failed: {err}")))?;
    Ok(())
}

fn tantivy_schema() -> tantivy::schema::Schema {
    let mut builder = tantivy::schema::Schema::builder();
    builder.add_u64_field("chunk_id", INDEXED | STORED | FAST);
    builder.add_u64_field("doc_id", INDEXED | STORED | FAST);
    builder.add_text_field("filepath", TEXT | STORED);
    builder.add_text_field("title", TEXT | STORED);
    builder.add_text_field("heading", TEXT | STORED);
    builder.add_text_field("body", TEXT);
    builder.build()
}

fn tantivy_fields_from_schema(schema: &tantivy::schema::Schema) -> Result<TantivyFields> {
    Ok(TantivyFields {
        chunk_id: schema.get_field("chunk_id").map_err(|_| {
            CoreError::Internal("tantivy schema missing field: chunk_id".to_string())
        })?,
        doc_id: schema.get_field("doc_id").map_err(|_| {
            CoreError::Internal("tantivy schema missing field: doc_id".to_string())
        })?,
        filepath: schema.get_field("filepath").map_err(|_| {
            CoreError::Internal("tantivy schema missing field: filepath".to_string())
        })?,
        title: schema
            .get_field("title")
            .map_err(|_| CoreError::Internal("tantivy schema missing field: title".to_string()))?,
        heading: schema.get_field("heading").map_err(|_| {
            CoreError::Internal("tantivy schema missing field: heading".to_string())
        })?,
        body: schema
            .get_field("body")
            .map_err(|_| CoreError::Internal("tantivy schema missing field: body".to_string()))?,
    })
}

fn resolve_tantivy_field(fields: TantivyFields, name: &str) -> Result<Field> {
    match name {
        "chunk_id" => Ok(fields.chunk_id),
        "doc_id" => Ok(fields.doc_id),
        "filepath" => Ok(fields.filepath),
        "title" => Ok(fields.title),
        "heading" => Ok(fields.heading),
        "body" => Ok(fields.body),
        other => Err(CoreError::Internal(format!(
            "unsupported tantivy field: {other}"
        ))),
    }
}

fn serialize_extensions(extensions: Option<&[String]>) -> Result<Option<String>> {
    match extensions {
        None => Ok(None),
        Some(values) => serde_json::to_string(values).map(Some).map_err(Into::into),
    }
}

fn deserialize_extensions(raw: Option<String>) -> Result<Option<Vec<String>>> {
    match raw {
        None => Ok(None),
        Some(json) => serde_json::from_str::<Vec<String>>(&json)
            .map(Some)
            .map_err(Into::into),
    }
}

fn decode_collection_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CollectionRow> {
    let raw_extensions: Option<String> = row.get(5)?;
    let extensions = deserialize_extensions(raw_extensions).map_err(|err| {
        Error::FromSqlConversionFailure(5, rusqlite::types::Type::Text, Box::new(err))
    })?;
    Ok(CollectionRow {
        id: row.get(0)?,
        space_id: row.get(1)?,
        name: row.get(2)?,
        path: PathBuf::from(row.get::<_, String>(3)?),
        description: row.get(4)?,
        extensions,
        created: row.get(6)?,
        updated: row.get(7)?,
    })
}

fn decode_document_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DocumentRow> {
    let active_value: i64 = row.get(6)?;
    let fts_dirty_value: i64 = row.get(8)?;
    Ok(DocumentRow {
        id: row.get(0)?,
        collection_id: row.get(1)?,
        path: row.get(2)?,
        title: row.get(3)?,
        hash: row.get(4)?,
        modified: row.get(5)?,
        active: active_value != 0,
        deactivated_at: row.get(7)?,
        fts_dirty: fts_dirty_value != 0,
    })
}

fn decode_chunk_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChunkRow> {
    let offset_value: i64 = row.get(3)?;
    let length_value: i64 = row.get(4)?;
    Ok(ChunkRow {
        id: row.get(0)?,
        doc_id: row.get(1)?,
        seq: row.get(2)?,
        offset: offset_value as usize,
        length: length_value as usize,
        heading: row.get(5)?,
        kind: row.get(6)?,
    })
}

#[cfg(test)]
mod tests;
