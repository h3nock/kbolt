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
