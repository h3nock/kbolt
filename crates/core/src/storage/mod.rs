use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::error::{CoreError, Result};
use kbolt_types::KboltError;
use rusqlite::{params, params_from_iter, Connection, Error, ErrorCode};

const DB_FILE: &str = "meta.sqlite";
const DEFAULT_SPACE_NAME: &str = "default";

pub struct Storage {
    db: Mutex<Connection>,
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

        Ok(Self {
            db: Mutex::new(conn),
        })
    }

    pub fn create_space(&self, name: &str, description: Option<&str>) -> Result<i64> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let result = conn.execute(
            "INSERT INTO spaces (name, description) VALUES (?1, ?2)",
            params![name, description],
        );

        match result {
            Ok(_) => Ok(conn.last_insert_rowid()),
            Err(err) => match err {
                Error::SqliteFailure(sqlite_err, _)
                    if sqlite_err.code == ErrorCode::ConstraintViolation =>
                {
                    Err(KboltError::SpaceAlreadyExists {
                        name: name.to_string(),
                    }
                    .into())
                }
                other => Err(other.into()),
            },
        }
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

    pub fn delete_space(&self, name: &str) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        if name == DEFAULT_SPACE_NAME {
            let affected = conn.execute(
                "DELETE FROM collections
                 WHERE space_id = (SELECT id FROM spaces WHERE name = ?1)",
                params![name],
            )?;
            let _ = affected;
            return Ok(());
        }

        let deleted = conn.execute("DELETE FROM spaces WHERE name = ?1", params![name])?;
        if deleted == 0 {
            return Err(KboltError::SpaceNotFound {
                name: name.to_string(),
            }
            .into());
        }

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

        match result {
            Ok(0) => Err(KboltError::SpaceNotFound {
                name: old.to_string(),
            }
            .into()),
            Ok(_) => Ok(()),
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
