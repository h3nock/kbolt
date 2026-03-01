use std::path::Path;
use std::sync::Mutex;

use kbolt_types::{KboltError, Result};
use rusqlite::{params, Connection, Error, ErrorCode};

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
            .map_err(|_| KboltError::Config("database mutex poisoned".to_string()))?;

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
                    })
                }
                other => Err(other.into()),
            },
        }
    }

    pub fn get_space(&self, name: &str) -> Result<SpaceRow> {
        let conn = self
            .db
            .lock()
            .map_err(|_| KboltError::Config("database mutex poisoned".to_string()))?;
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
            }),
            Err(err) => Err(err.into()),
        }
    }

    pub fn list_spaces(&self) -> Result<Vec<SpaceRow>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| KboltError::Config("database mutex poisoned".to_string()))?;
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
}

#[cfg(test)]
mod tests;
