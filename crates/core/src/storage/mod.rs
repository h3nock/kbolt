use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use crate::error::{CoreError, Result};
use crate::ingest::chunk::FinalChunkKind;
use kbolt_types::{DiskUsage, KboltError};
use rusqlite::types::{Type as SqlType, Value as SqlValue};
use rusqlite::{params, params_from_iter, Connection, Error, ErrorCode};
use tantivy::collector::TopDocs;
use tantivy::query::{
    BooleanQuery, BoostQuery, ConstScoreQuery, Occur, Query, TermQuery, TermSetQuery,
};
use tantivy::schema::{Field, IndexRecordOption, Value, FAST, INDEXED, STORED, TEXT};
use tantivy::tokenizer::TokenStream;
use tantivy::{Index, IndexReader, IndexWriter, TantivyDocument, Term};
use usearch::{IndexOptions, MetricKind, ScalarKind};

const DB_FILE: &str = "meta.sqlite";
const DEFAULT_SPACE_NAME: &str = "default";
const SPACES_DIR: &str = "spaces";
const TANTIVY_DIR_NAME: &str = "tantivy";
const USEARCH_FILENAME: &str = "vectors.usearch";
const SCHEMA_VERSION: i64 = 3;
const SQLITE_IN_CLAUSE_BATCH_SIZE: usize = 500;

#[derive(Clone, Copy)]
enum UsearchSaveMode {
    Immediate,
    Deferred,
}

pub struct Storage {
    db: Mutex<Connection>,
    cache_dir: PathBuf,
    spaces: RwLock<HashMap<String, Arc<SpaceIndexes>>>,
    dirty_usearch_spaces: Mutex<HashSet<String>>,
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
    pub title_source: DocumentTitleSource,
    pub hash: String,
    pub modified: String,
    pub active: bool,
    pub deactivated_at: Option<String>,
    pub fts_dirty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileListRow {
    pub doc_id: i64,
    pub path: String,
    pub title: String,
    pub hash: String,
    pub active: bool,
    pub chunk_count: usize,
    pub embedded_chunk_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRow {
    pub id: i64,
    pub doc_id: i64,
    pub seq: i32,
    pub offset: usize,
    pub length: usize,
    pub heading: Option<String>,
    pub kind: FinalChunkKind,
    pub retrieval_prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkInsert {
    pub seq: i32,
    pub offset: usize,
    pub length: usize,
    pub heading: Option<String>,
    pub kind: FinalChunkKind,
    pub retrieval_prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentTextRow {
    pub doc_id: i64,
    pub extractor_key: String,
    pub source_hash: String,
    pub text_hash: String,
    pub generation_key: String,
    pub text: String,
    pub created: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkTextRow {
    pub chunk: ChunkRow,
    pub extractor_key: String,
    pub text: String,
}

pub struct DocumentGenerationReplace<'a> {
    pub collection_id: i64,
    pub path: &'a str,
    pub title: &'a str,
    pub title_source: DocumentTitleSource,
    pub hash: &'a str,
    pub modified: &'a str,
    pub extractor_key: &'a str,
    pub source_hash: &'a str,
    pub text_hash: &'a str,
    pub generation_key: &'a str,
    pub text: &'a str,
    pub chunks: &'a [ChunkInsert],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentGenerationReplaceResult {
    pub doc_id: i64,
    pub old_chunk_ids: Vec<i64>,
    pub chunk_ids: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedRecord {
    pub chunk: ChunkRow,
    pub doc_path: String,
    pub collection_path: PathBuf,
    pub space_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtsDirtyRecord {
    pub doc_id: i64,
    pub doc_path: String,
    pub doc_title: String,
    pub doc_title_source: DocumentTitleSource,
    pub doc_hash: String,
    pub collection_path: PathBuf,
    pub space_name: String,
    pub chunks: Vec<ChunkRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReapableDocument {
    pub doc_id: i64,
    pub space_name: String,
    pub chunk_ids: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TantivyEntry {
    pub chunk_id: i64,
    pub doc_id: i64,
    pub filepath: String,
    pub semantic_title: Option<String>,
    pub heading: Option<String>,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentTitleSource {
    Extracted,
    FilenameFallback,
}

impl DocumentTitleSource {
    pub fn as_sql(self) -> &'static str {
        match self {
            Self::Extracted => "extracted",
            Self::FilenameFallback => "filename_fallback",
        }
    }

    fn from_sql(raw: &str) -> std::result::Result<Self, KboltError> {
        match raw {
            "extracted" => Ok(Self::Extracted),
            "filename_fallback" => Ok(Self::FilenameFallback),
            other => Err(KboltError::InvalidInput(format!(
                "invalid stored document title source: {other}"
            ))),
        }
    }

    pub fn semantic_title(self, title: &str) -> Option<&str> {
        matches!(self, Self::Extracted)
            .then_some(title.trim())
            .filter(|title| !title.is_empty())
    }
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
pub struct SearchScopeSummary {
    pub document_ids: Vec<i64>,
    pub chunk_count: usize,
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
    tantivy_reader: IndexReader,
    tantivy_writer: Mutex<Option<IndexWriter>>,
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

#[derive(Debug, Clone, Copy)]
struct Bm25FieldSpec {
    field: Field,
    boost: f32,
    index_record_option: IndexRecordOption,
}

impl Storage {
    pub fn new(cache_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)?;
        let db_path = cache_dir.join(DB_FILE);
        let conn = Connection::open(db_path)?;
        reject_incompatible_legacy_index(&conn)?;
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
    title_source    TEXT NOT NULL DEFAULT 'extracted',
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
    retrieval_prefix TEXT,
    UNIQUE(doc_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id    INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

CREATE TABLE IF NOT EXISTS document_texts (
    doc_id            INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    extractor_key     TEXT NOT NULL,
    source_hash       TEXT NOT NULL,
    text_hash         TEXT NOT NULL,
    generation_key    TEXT NOT NULL DEFAULT '',
    text              TEXT NOT NULL,
    created           TEXT NOT NULL
);
"#,
        )?;
        ensure_documents_title_source_column(&conn)?;
        ensure_document_texts_generation_key_column(&conn)?;
        ensure_chunks_retrieval_prefix_column(&conn)?;
        ensure_schema_version(&conn)?;

        conn.execute(
            "INSERT OR IGNORE INTO spaces (name, description) VALUES (?1, NULL)",
            params![DEFAULT_SPACE_NAME],
        )?;

        let storage = Self {
            db: Mutex::new(conn),
            cache_dir: cache_dir.to_path_buf(),
            spaces: RwLock::new(HashMap::new()),
            dirty_usearch_spaces: Mutex::new(HashSet::new()),
        };
        storage.open_space(DEFAULT_SPACE_NAME)?;

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
        let tantivy_reader = tantivy_index.reader()?;
        let usearch_index = open_or_create_usearch_index(&usearch_path)?;
        let fields = tantivy_fields_from_schema(&tantivy_index.schema())?;

        let mut spaces = self
            .spaces
            .write()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        spaces
            .entry(name.to_string())
            .or_insert(Arc::new(SpaceIndexes {
                _tantivy_dir: tantivy_dir,
                usearch_path,
                tantivy_index,
                tantivy_reader,
                tantivy_writer: Mutex::new(None),
                usearch_index: RwLock::new(usearch_index),
                fields,
            }));

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

        let row = stmt.query_row(params![name], decode_space_row);

        match row {
            Ok(space) => Ok(space),
            Err(Error::QueryReturnedNoRows) => Err(KboltError::SpaceNotFound {
                name: name.to_string(),
            }
            .into()),
            Err(err) => Err(err.into()),
        }
    }

    pub fn get_space_by_id(&self, id: i64) -> Result<SpaceRow> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut stmt =
            conn.prepare("SELECT id, name, description, created FROM spaces WHERE id = ?1")?;

        let row = stmt.query_row(params![id], decode_space_row);

        match row {
            Ok(space) => Ok(space),
            Err(Error::QueryReturnedNoRows) => Err(KboltError::SpaceNotFound {
                name: format!("id={id}"),
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

        let rows = stmt.query_map([], decode_space_row)?;

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
        let rows = stmt.query_map(params![collection], decode_space_row)?;
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

    pub fn get_collection_by_id(&self, id: i64) -> Result<CollectionRow> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut stmt = conn.prepare(
            "SELECT id, space_id, name, path, description, extensions, created, updated
             FROM collections
             WHERE id = ?1",
        )?;

        let result = stmt.query_row(params![id], decode_collection_row);
        match result {
            Ok(row) => Ok(row),
            Err(Error::QueryReturnedNoRows) => Err(KboltError::CollectionNotFound {
                name: format!("id={id}"),
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
        title_source: DocumentTitleSource,
        hash: &str,
        modified: &str,
    ) -> Result<i64> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _collection_name = lookup_collection_name(&conn, collection_id)?;

        let id = conn.query_row(
            "INSERT INTO documents (collection_id, path, title, title_source, hash, modified, active, deactivated_at, fts_dirty)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, NULL, 1)
             ON CONFLICT(collection_id, path) DO UPDATE SET
                 title = excluded.title,
                 title_source = excluded.title_source,
                 hash = excluded.hash,
                 modified = excluded.modified,
                 active = 1,
                 deactivated_at = NULL,
                 fts_dirty = 1
             RETURNING id",
            params![
                collection_id,
                path,
                title,
                title_source.as_sql(),
                hash,
                modified
            ],
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
            "SELECT id, collection_id, path, title, title_source, hash, modified, active, deactivated_at, fts_dirty
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

    pub fn get_document_by_id(&self, id: i64) -> Result<DocumentRow> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut stmt = conn.prepare(
            "SELECT id, collection_id, path, title, title_source, hash, modified, active, deactivated_at, fts_dirty
             FROM documents
             WHERE id = ?1",
        )?;

        let result = stmt.query_row(params![id], decode_document_row);
        match result {
            Ok(row) => Ok(row),
            Err(Error::QueryReturnedNoRows) => Err(KboltError::DocumentNotFound {
                path: format!("id={id}"),
            }
            .into()),
            Err(err) => Err(err.into()),
        }
    }

    pub fn get_documents_by_ids(&self, ids: &[i64]) -> Result<Vec<DocumentRow>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let placeholders = vec!["?"; ids.len()].join(", ");
        let sql = format!(
            "SELECT id, collection_id, path, title, title_source, hash, modified, active, deactivated_at, fts_dirty
             FROM documents
             WHERE id IN ({placeholders})"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(ids.iter()), decode_document_row)?;
        let docs = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(docs)
    }

    pub fn refresh_document_activity(&self, doc_id: i64, modified: &str) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let updated = conn.execute(
            "UPDATE documents
             SET modified = ?1,
                 active = 1,
                 deactivated_at = NULL
             WHERE id = ?2",
            params![modified, doc_id],
        )?;

        if updated == 0 {
            return Err(KboltError::DocumentNotFound {
                path: format!("id={doc_id}"),
            }
            .into());
        }

        Ok(())
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
        let active_only = i64::from(active_only);
        let mut stmt = conn.prepare(
            "SELECT id, collection_id, path, title, title_source, hash, modified, active, deactivated_at, fts_dirty
             FROM documents
             WHERE collection_id = ?1
               AND (?2 = 0 OR active = 1)
             ORDER BY path ASC",
        )?;
        let rows = stmt.query_map(params![collection_id, active_only], decode_document_row)?;
        let docs = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(docs)
    }

    pub fn list_collection_file_rows(
        &self,
        collection_id: i64,
        active_only: bool,
    ) -> Result<Vec<FileListRow>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _collection_name = lookup_collection_name(&conn, collection_id)?;
        let active_only = i64::from(active_only);
        let mut stmt = conn.prepare(
            "SELECT d.id, d.path, d.title, d.hash, d.active,
                    COUNT(DISTINCT c.id) AS chunk_count,
                    COUNT(DISTINCT e.chunk_id) AS embedded_chunk_count
             FROM documents d
             LEFT JOIN chunks c ON c.doc_id = d.id
             LEFT JOIN embeddings e ON e.chunk_id = c.id
             WHERE d.collection_id = ?1
               AND (?2 = 0 OR d.active = 1)
             GROUP BY d.id, d.path, d.title, d.hash, d.active
             ORDER BY d.path ASC",
        )?;
        let rows = stmt.query_map(params![collection_id, active_only], |row| {
            let chunk_count: i64 = row.get(5)?;
            let embedded_chunk_count: i64 = row.get(6)?;
            Ok(FileListRow {
                doc_id: row.get(0)?,
                path: row.get(1)?,
                title: row.get(2)?,
                hash: row.get(3)?,
                active: row.get::<_, i64>(4)? != 0,
                chunk_count: chunk_count as usize,
                embedded_chunk_count: embedded_chunk_count as usize,
            })
        })?;
        let files = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(files)
    }

    pub fn get_document_by_hash_prefix(&self, prefix: &str) -> Result<Vec<DocumentRow>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let pattern = format!("{prefix}%");
        let mut stmt = conn.prepare(
            "SELECT id, collection_id, path, title, title_source, hash, modified, active, deactivated_at, fts_dirty
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
        let reaped = self.list_reapable_documents(older_than_days)?;
        let doc_ids = reaped.iter().map(|item| item.doc_id).collect::<Vec<_>>();
        self.delete_documents(&doc_ids)?;
        Ok(doc_ids)
    }

    pub fn list_reapable_documents(&self, older_than_days: u32) -> Result<Vec<ReapableDocument>> {
        self.list_reapable_documents_filtered(older_than_days, "", Vec::new())
    }

    pub fn list_reapable_documents_in_space(
        &self,
        older_than_days: u32,
        space_id: i64,
    ) -> Result<Vec<ReapableDocument>> {
        self.list_reapable_documents_filtered(
            older_than_days,
            " AND c.space_id = ?",
            vec![SqlValue::Integer(space_id)],
        )
    }

    pub fn list_reapable_documents_in_collections(
        &self,
        older_than_days: u32,
        collection_ids: &[i64],
    ) -> Result<Vec<ReapableDocument>> {
        if collection_ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = vec!["?"; collection_ids.len()].join(", ");
        let clause = format!(" AND d.collection_id IN ({placeholders})");
        let params = collection_ids
            .iter()
            .map(|id| SqlValue::Integer(*id))
            .collect::<Vec<_>>();
        self.list_reapable_documents_filtered(older_than_days, &clause, params)
    }

    fn list_reapable_documents_filtered(
        &self,
        older_than_days: u32,
        scope_clause: &str,
        scope_params: Vec<SqlValue>,
    ) -> Result<Vec<ReapableDocument>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let modifier = format!("-{} days", older_than_days);
        let sql = format!(
            "SELECT d.id, s.name
             FROM documents d
             JOIN collections c ON c.id = d.collection_id
             JOIN spaces s ON s.id = c.space_id
             WHERE d.active = 0
               AND d.deactivated_at IS NOT NULL
               AND d.deactivated_at <= strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?){scope_clause}
             ORDER BY d.id ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut params = Vec::with_capacity(scope_params.len() + 1);
        params.push(SqlValue::Text(modifier));
        params.extend(scope_params);
        let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        let headers = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);

        let mut documents = Vec::with_capacity(headers.len());
        for (doc_id, space_name) in headers {
            documents.push(ReapableDocument {
                doc_id,
                space_name,
                chunk_ids: load_chunk_ids_for_doc(&conn, doc_id)?,
            });
        }

        Ok(documents)
    }

    pub fn delete_documents(&self, doc_ids: &[i64]) -> Result<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let placeholders = vec!["?"; doc_ids.len()].join(", ");
        let sql = format!("DELETE FROM documents WHERE id IN ({placeholders})");
        conn.execute(&sql, params_from_iter(doc_ids.iter()))?;
        Ok(())
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
            "INSERT INTO chunks (doc_id, seq, offset, length, heading, kind, retrieval_prefix)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;

        let mut ids = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            stmt.execute(params![
                doc_id,
                chunk.seq,
                chunk.offset as i64,
                chunk.length as i64,
                chunk.heading,
                chunk.kind.as_storage_kind(),
                chunk.retrieval_prefix.as_deref(),
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
            "SELECT id, doc_id, seq, offset, length, heading, kind, retrieval_prefix
             FROM chunks
             WHERE doc_id = ?1
             ORDER BY seq ASC",
        )?;
        let rows = stmt.query_map(params![doc_id], decode_chunk_row)?;
        let chunks = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(chunks)
    }

    pub fn get_chunks_for_documents(&self, doc_ids: &[i64]) -> Result<HashMap<i64, Vec<ChunkRow>>> {
        if doc_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut requested = doc_ids.to_vec();
        requested.sort_unstable();
        requested.dedup();

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let placeholders = vec!["?"; requested.len()].join(", ");
        let sql = format!(
            "SELECT id, doc_id, seq, offset, length, heading, kind, retrieval_prefix
             FROM chunks
             WHERE doc_id IN ({placeholders})
             ORDER BY doc_id ASC, seq ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(requested.iter()), decode_chunk_row)?;
        let mut chunks_by_doc: HashMap<i64, Vec<ChunkRow>> = HashMap::new();
        for doc_id in requested {
            chunks_by_doc.insert(doc_id, Vec::new());
        }
        for row in rows {
            let chunk = row?;
            chunks_by_doc.entry(chunk.doc_id).or_default().push(chunk);
        }

        Ok(chunks_by_doc)
    }

    pub fn get_chunks_for_document_seq_ranges(
        &self,
        ranges: &[(i64, i32, i32)],
    ) -> Result<HashMap<i64, Vec<ChunkRow>>> {
        if ranges.is_empty() {
            return Ok(HashMap::new());
        }

        let mut ranges_by_doc: HashMap<i64, Vec<(i32, i32)>> = HashMap::new();
        for (doc_id, min_seq, max_seq) in ranges {
            if min_seq > max_seq {
                return Err(CoreError::Internal(format!(
                    "chunk seq range min must be <= max for doc {doc_id}: {min_seq} > {max_seq}"
                )));
            }

            ranges_by_doc
                .entry(*doc_id)
                .or_default()
                .push((*min_seq, *max_seq));
        }

        for doc_ranges in ranges_by_doc.values_mut() {
            doc_ranges.sort_unstable();
            let mut merged: Vec<(i32, i32)> = Vec::with_capacity(doc_ranges.len());
            for (min_seq, max_seq) in doc_ranges.drain(..) {
                if let Some((_, merged_max)) = merged.last_mut() {
                    if min_seq <= merged_max.saturating_add(1) {
                        *merged_max = (*merged_max).max(max_seq);
                        continue;
                    }
                }
                merged.push((min_seq, max_seq));
            }
            *doc_ranges = merged;
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut stmt = conn.prepare(
            "SELECT id, doc_id, seq, offset, length, heading, kind, retrieval_prefix
             FROM chunks
             WHERE doc_id = ?1 AND seq BETWEEN ?2 AND ?3
             ORDER BY seq ASC",
        )?;

        let mut chunks_by_doc: HashMap<i64, Vec<ChunkRow>> = HashMap::new();
        for (doc_id, doc_ranges) in ranges_by_doc {
            chunks_by_doc.entry(doc_id).or_default();
            for (min_seq, max_seq) in doc_ranges {
                let rows = stmt.query_map(params![doc_id, min_seq, max_seq], decode_chunk_row)?;
                for row in rows {
                    chunks_by_doc.entry(doc_id).or_default().push(row?);
                }
            }
        }

        Ok(chunks_by_doc)
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
            "SELECT id, doc_id, seq, offset, length, heading, kind, retrieval_prefix
             FROM chunks
             WHERE id IN ({placeholders})
             ORDER BY id ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(chunk_ids.iter()), decode_chunk_row)?;
        let chunks = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(chunks)
    }

    pub fn get_active_search_scope_summary_in_collections(
        &self,
        collection_ids: &[i64],
    ) -> Result<SearchScopeSummary> {
        if collection_ids.is_empty() {
            return Ok(SearchScopeSummary {
                document_ids: Vec::new(),
                chunk_count: 0,
            });
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let placeholders = vec!["?"; collection_ids.len()].join(", ");
        let sql = format!(
            "SELECT d.id, COUNT(c.id)
             FROM chunks c
             JOIN documents d ON d.id = c.doc_id
             WHERE d.active = 1
               AND d.collection_id IN ({placeholders})
             GROUP BY d.id
             ORDER BY d.id ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(collection_ids.iter()), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })?;

        let mut document_ids = Vec::new();
        let mut chunk_count = 0usize;
        for row in rows {
            let (doc_id, doc_chunk_count) = row?;
            let doc_chunk_count = usize::try_from(doc_chunk_count).map_err(|_| {
                CoreError::Internal(format!(
                    "active chunk count must be non-negative for document {doc_id}"
                ))
            })?;
            document_ids.push(doc_id);
            chunk_count = chunk_count.saturating_add(doc_chunk_count);
        }

        Ok(SearchScopeSummary {
            document_ids,
            chunk_count,
        })
    }

    pub fn count_active_chunks_in_collections(&self, collection_ids: &[i64]) -> Result<usize> {
        if collection_ids.is_empty() {
            return Ok(0);
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let placeholders = vec!["?"; collection_ids.len()].join(", ");
        let sql = format!(
            "SELECT COUNT(*)
             FROM chunks c
             JOIN documents d ON d.id = c.doc_id
             WHERE d.active = 1
               AND d.collection_id IN ({placeholders})"
        );
        query_count(&conn, &sql, params_from_iter(collection_ids.iter()))
    }

    pub fn has_inactive_documents_in_collections(&self, collection_ids: &[i64]) -> Result<bool> {
        if collection_ids.is_empty() {
            return Ok(false);
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let placeholders = vec!["?"; collection_ids.len()].join(", ");
        let sql = format!(
            "SELECT EXISTS(
                 SELECT 1
                 FROM documents
                 WHERE active = 0
                   AND collection_id IN ({placeholders})
             )"
        );
        let exists: i64 = conn.query_row(&sql, params_from_iter(collection_ids.iter()), |row| {
            row.get(0)
        })?;
        Ok(exists != 0)
    }

    pub fn get_active_chunk_ids_in_collections(&self, collection_ids: &[i64]) -> Result<Vec<i64>> {
        if collection_ids.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let placeholders = vec!["?"; collection_ids.len()].join(", ");
        let sql = format!(
            "SELECT c.id
             FROM chunks c
             JOIN documents d ON d.id = c.doc_id
             WHERE d.active = 1
               AND d.collection_id IN ({placeholders})
             ORDER BY d.id ASC, c.seq ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(collection_ids.iter()), |row| row.get(0))?;
        let chunk_ids = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(chunk_ids)
    }

    pub fn replace_document_generation(
        &self,
        replacement: DocumentGenerationReplace<'_>,
    ) -> Result<DocumentGenerationReplaceResult> {
        for chunk in replacement.chunks {
            validate_text_span(
                replacement.text,
                chunk.offset,
                chunk.length,
                &format!("chunk seq {}", chunk.seq),
            )?;
        }

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _collection_name = lookup_collection_name(&conn, replacement.collection_id)?;

        let tx = conn.unchecked_transaction()?;
        let doc_id: i64 = tx.query_row(
            "INSERT INTO documents (collection_id, path, title, title_source, hash, modified, active, deactivated_at, fts_dirty)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, NULL, 1)
             ON CONFLICT(collection_id, path) DO UPDATE SET
                 title = excluded.title,
                 title_source = excluded.title_source,
                 hash = excluded.hash,
                 modified = excluded.modified,
                 active = 1,
                 deactivated_at = NULL,
                 fts_dirty = 1
             RETURNING id",
            params![
                replacement.collection_id,
                replacement.path,
                replacement.title,
                replacement.title_source.as_sql(),
                replacement.hash,
                replacement.modified
            ],
            |row| row.get(0),
        )?;

        let old_chunk_ids = {
            let mut stmt =
                tx.prepare("SELECT id FROM chunks WHERE doc_id = ?1 ORDER BY seq ASC")?;
            let rows = stmt.query_map(params![doc_id], |row| row.get::<_, i64>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };

        tx.execute(
            "INSERT INTO document_texts (doc_id, extractor_key, source_hash, text_hash, generation_key, text, created)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
             ON CONFLICT(doc_id) DO UPDATE SET
                 extractor_key = excluded.extractor_key,
                 source_hash = excluded.source_hash,
                 text_hash = excluded.text_hash,
                 generation_key = excluded.generation_key,
                 text = excluded.text,
                 created = excluded.created",
            params![
                doc_id,
                replacement.extractor_key,
                replacement.source_hash,
                replacement.text_hash,
                replacement.generation_key,
                replacement.text
            ],
        )?;

        tx.execute("DELETE FROM chunks WHERE doc_id = ?1", params![doc_id])?;

        let mut stmt = tx.prepare(
            "INSERT INTO chunks (doc_id, seq, offset, length, heading, kind, retrieval_prefix)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        let mut chunk_ids = Vec::with_capacity(replacement.chunks.len());
        for chunk in replacement.chunks {
            stmt.execute(params![
                doc_id,
                chunk.seq,
                chunk.offset as i64,
                chunk.length as i64,
                chunk.heading,
                chunk.kind.as_storage_kind(),
                chunk.retrieval_prefix.as_deref(),
            ])?;
            chunk_ids.push(tx.last_insert_rowid());
        }
        drop(stmt);

        tx.commit()?;
        Ok(DocumentGenerationReplaceResult {
            doc_id,
            old_chunk_ids,
            chunk_ids,
        })
    }

    pub fn put_document_text(
        &self,
        doc_id: i64,
        extractor_key: &str,
        source_hash: &str,
        text_hash: &str,
        generation_key: &str,
        text: &str,
    ) -> Result<()> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _doc = lookup_document_id(&conn, doc_id)?;

        conn.execute(
            "INSERT INTO document_texts (doc_id, extractor_key, source_hash, text_hash, generation_key, text, created)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
             ON CONFLICT(doc_id) DO UPDATE SET
                 extractor_key = excluded.extractor_key,
                 source_hash = excluded.source_hash,
                 text_hash = excluded.text_hash,
                 generation_key = excluded.generation_key,
                 text = excluded.text,
                 created = excluded.created",
            params![
                doc_id,
                extractor_key,
                source_hash,
                text_hash,
                generation_key,
                text
            ],
        )?;
        Ok(())
    }

    pub fn get_document_text(&self, doc_id: i64) -> Result<DocumentTextRow> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _doc = lookup_document_id(&conn, doc_id)?;

        let result = conn.query_row(
            "SELECT doc_id, extractor_key, source_hash, text_hash, generation_key, text, created
             FROM document_texts
             WHERE doc_id = ?1",
            params![doc_id],
            decode_document_text_row,
        );
        match result {
            Ok(row) => Ok(row),
            Err(Error::QueryReturnedNoRows) => Err(missing_document_text_error(doc_id)),
            Err(err) => Err(err.into()),
        }
    }

    pub fn get_document_texts(&self, doc_ids: &[i64]) -> Result<HashMap<i64, DocumentTextRow>> {
        if doc_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut requested = doc_ids.to_vec();
        requested.sort_unstable();
        requested.dedup();

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let placeholders = vec!["?"; requested.len()].join(", ");
        let sql = format!(
            "SELECT doc_id, extractor_key, source_hash, text_hash, generation_key, text, created
             FROM document_texts
             WHERE doc_id IN ({placeholders})
             ORDER BY doc_id ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(requested.iter()), decode_document_text_row)?;
        let mut text_by_doc = HashMap::new();
        for row in rows {
            let row = row?;
            text_by_doc.insert(row.doc_id, row);
        }

        for doc_id in requested {
            if !text_by_doc.contains_key(&doc_id) {
                return Err(missing_document_text_error(doc_id));
            }
        }

        Ok(text_by_doc)
    }

    pub fn has_document_text(&self, doc_id: i64) -> Result<bool> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let exists: i64 = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM document_texts WHERE doc_id = ?1)",
            params![doc_id],
            |row| row.get(0),
        )?;
        Ok(exists != 0)
    }

    pub fn has_current_document_text(&self, doc_id: i64, generation_key: &str) -> Result<bool> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let exists: i64 = conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM document_texts
                WHERE doc_id = ?1 AND generation_key = ?2
            )",
            params![doc_id, generation_key],
            |row| row.get(0),
        )?;
        Ok(exists != 0)
    }

    pub fn get_document_text_generation_keys(
        &self,
        doc_ids: &[i64],
    ) -> Result<HashMap<i64, String>> {
        if doc_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut requested = doc_ids.to_vec();
        requested.sort_unstable();
        requested.dedup();

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut generation_by_doc = HashMap::with_capacity(requested.len());
        for batch in requested.chunks(SQLITE_IN_CLAUSE_BATCH_SIZE) {
            let placeholders = vec!["?"; batch.len()].join(", ");
            let sql = format!(
                "SELECT doc_id, generation_key
                 FROM document_texts
                 WHERE doc_id IN ({placeholders})"
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params_from_iter(batch.iter()), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (doc_id, generation_key) = row?;
                generation_by_doc.insert(doc_id, generation_key);
            }
        }

        Ok(generation_by_doc)
    }

    pub fn get_document_text_extractors(&self, doc_ids: &[i64]) -> Result<HashMap<i64, String>> {
        if doc_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut requested = doc_ids.to_vec();
        requested.sort_unstable();
        requested.dedup();

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut extractor_by_doc = HashMap::with_capacity(requested.len());
        for batch in requested.chunks(SQLITE_IN_CLAUSE_BATCH_SIZE) {
            let placeholders = vec!["?"; batch.len()].join(", ");
            let sql = format!(
                "SELECT doc_id, extractor_key
                 FROM document_texts
                 WHERE doc_id IN ({placeholders})"
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params_from_iter(batch.iter()), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (doc_id, extractor_key) = row?;
                extractor_by_doc.insert(doc_id, extractor_key);
            }
        }

        for doc_id in requested {
            if !extractor_by_doc.contains_key(&doc_id) {
                return Err(missing_document_text_error(doc_id));
            }
        }

        Ok(extractor_by_doc)
    }

    pub fn get_canonical_chunk_texts(&self, chunk_ids: &[i64]) -> Result<HashMap<i64, String>> {
        if chunk_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut requested = chunk_ids.to_vec();
        requested.sort_unstable();
        requested.dedup();

        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let mut text_by_chunk = HashMap::with_capacity(requested.len());
        for batch in requested.chunks(SQLITE_IN_CLAUSE_BATCH_SIZE) {
            let placeholders = vec!["?"; batch.len()].join(", ");
            let sql = format!(
                "SELECT c.id,
                        substr(CAST(dt.text AS BLOB), c.offset + 1, c.length)
                 FROM chunks c
                 JOIN document_texts dt ON dt.doc_id = c.doc_id
                 WHERE c.id IN ({placeholders})"
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params_from_iter(batch.iter()), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?;
            for row in rows {
                let (chunk_id, bytes) = row?;
                let text = String::from_utf8(bytes).map_err(|err| {
                    CoreError::Internal(format!(
                        "stored canonical text slice for chunk {chunk_id} is invalid UTF-8: {err}"
                    ))
                })?;
                text_by_chunk.insert(chunk_id, text);
            }
        }

        for chunk_id in requested {
            if !text_by_chunk.contains_key(&chunk_id) {
                return Err(CoreError::Internal(format!(
                    "canonical text missing for chunk {chunk_id}"
                )));
            }
        }

        Ok(text_by_chunk)
    }

    pub fn get_chunk_text(&self, chunk_id: i64) -> Result<ChunkTextRow> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let result = conn.query_row(
            "SELECT c.id, c.doc_id, c.seq, c.offset, c.length, c.heading, c.kind, c.retrieval_prefix, dt.extractor_key, dt.text
             FROM chunks c
             JOIN document_texts dt ON dt.doc_id = c.doc_id
             WHERE c.id = ?1",
            params![chunk_id],
            |row| {
                let chunk = decode_chunk_row(row)?;
                let extractor_key: String = row.get(8)?;
                let document_text: String = row.get(9)?;
                Ok((chunk, extractor_key, document_text))
            },
        );
        let (chunk, extractor_key, document_text) = match result {
            Ok(row) => row,
            Err(Error::QueryReturnedNoRows) => {
                let doc_id = lookup_chunk_doc_id(&conn, chunk_id)?;
                return Err(missing_document_text_error(doc_id));
            }
            Err(err) => return Err(err.into()),
        };
        let text = chunk_text_from_canonical(&document_text, &chunk)?;
        Ok(ChunkTextRow {
            chunk,
            extractor_key,
            text,
        })
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

    pub fn get_unembedded_chunks(
        &self,
        model: &str,
        after_chunk_id: i64,
        limit: usize,
    ) -> Result<Vec<EmbedRecord>> {
        self.get_unembedded_chunks_filtered(model, after_chunk_id, limit, "", Vec::new())
    }

    pub fn get_unembedded_chunks_in_space(
        &self,
        model: &str,
        space_id: i64,
        after_chunk_id: i64,
        limit: usize,
    ) -> Result<Vec<EmbedRecord>> {
        self.get_unembedded_chunks_filtered(
            model,
            after_chunk_id,
            limit,
            " AND col.space_id = ?",
            vec![SqlValue::Integer(space_id)],
        )
    }

    pub fn get_unembedded_chunks_in_collections(
        &self,
        model: &str,
        collection_ids: &[i64],
        after_chunk_id: i64,
        limit: usize,
    ) -> Result<Vec<EmbedRecord>> {
        if collection_ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = vec!["?"; collection_ids.len()].join(", ");
        let clause = format!(" AND d.collection_id IN ({placeholders})");
        let params = collection_ids
            .iter()
            .map(|id| SqlValue::Integer(*id))
            .collect::<Vec<_>>();
        self.get_unembedded_chunks_filtered(model, after_chunk_id, limit, &clause, params)
    }

    fn get_unembedded_chunks_filtered(
        &self,
        model: &str,
        after_chunk_id: i64,
        limit: usize,
        scope_clause: &str,
        scope_params: Vec<SqlValue>,
    ) -> Result<Vec<EmbedRecord>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let sql_limit = i64::try_from(limit)
            .map_err(|_| CoreError::Internal("limit too large for sqlite".to_string()))?;

        let sql = format!(
            "SELECT c.id, c.doc_id, c.seq, c.offset, c.length, c.heading, c.kind, c.retrieval_prefix,
                    d.path, col.path, s.name
             FROM chunks c
             JOIN documents d ON d.id = c.doc_id
             JOIN collections col ON col.id = d.collection_id
             JOIN spaces s ON s.id = col.space_id
             LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model = ?
             WHERE d.active = 1 AND e.chunk_id IS NULL AND c.id > ?{scope_clause}
             ORDER BY c.id ASC
             LIMIT ?"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut params = Vec::with_capacity(scope_params.len() + 3);
        params.push(SqlValue::Text(model.to_string()));
        params.push(SqlValue::Integer(after_chunk_id));
        params.extend(scope_params);
        params.push(SqlValue::Integer(sql_limit));
        let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
            Ok(EmbedRecord {
                chunk: decode_chunk_row(row)?,
                doc_path: row.get(8)?,
                collection_path: PathBuf::from(row.get::<_, String>(9)?),
                space_name: row.get(10)?,
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

    pub fn delete_embeddings_for_space(&self, space_id: i64) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _space_name = lookup_space_name(&conn, space_id)?;

        let deleted = conn.execute(
            "DELETE FROM embeddings
             WHERE chunk_id IN (
                 SELECT c.id
                 FROM chunks c
                 JOIN documents d ON d.id = c.doc_id
                 JOIN collections col ON col.id = d.collection_id
                 WHERE col.space_id = ?1
             )",
            params![space_id],
        )?;
        Ok(deleted)
    }

    pub fn list_embedding_models_in_space(&self, space_id: i64) -> Result<Vec<String>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _space_name = lookup_space_name(&conn, space_id)?;

        let mut stmt = conn.prepare(
            "SELECT DISTINCT e.model
             FROM embeddings e
             JOIN chunks c ON c.id = e.chunk_id
             JOIN documents d ON d.id = c.doc_id
             JOIN collections col ON col.id = d.collection_id
             WHERE col.space_id = ?1
             ORDER BY e.model ASC",
        )?;
        let rows = stmt.query_map(params![space_id], |row| row.get::<_, String>(0))?;
        let models = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(models)
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

        let space_indexes = self.get_space_indexes(space)?;
        with_tantivy_writer(&space_indexes, |writer| {
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
                if let Some(title) = &entry.semantic_title {
                    doc.add_text(space_indexes.fields.title, title);
                }
                if let Some(heading) = &entry.heading {
                    doc.add_text(space_indexes.fields.heading, heading);
                }
                doc.add_text(space_indexes.fields.body, &entry.body);
                writer.add_document(doc)?;
            }
            Ok(())
        })
    }

    pub fn delete_tantivy(&self, space: &str, chunk_ids: &[i64]) -> Result<()> {
        if chunk_ids.is_empty() {
            return Ok(());
        }

        let space_indexes = self.get_space_indexes(space)?;
        with_tantivy_writer(&space_indexes, |writer| {
            for chunk_id in chunk_ids {
                let chunk_key = u64::try_from(*chunk_id).map_err(|_| {
                    CoreError::Internal(format!(
                        "chunk_id must be non-negative for tantivy delete: {chunk_id}"
                    ))
                })?;
                writer.delete_term(Term::from_field_u64(
                    space_indexes.fields.chunk_id,
                    chunk_key,
                ));
            }

            Ok(())
        })
    }

    pub fn delete_tantivy_by_doc(&self, space: &str, doc_id: i64) -> Result<()> {
        let space_indexes = self.get_space_indexes(space)?;
        with_tantivy_writer(&space_indexes, |writer| {
            let doc_key = u64::try_from(doc_id).map_err(|_| {
                CoreError::Internal(format!(
                    "doc_id must be non-negative for tantivy delete-by-doc: {doc_id}"
                ))
            })?;
            writer.delete_term(Term::from_field_u64(space_indexes.fields.doc_id, doc_key));
            Ok(())
        })
    }

    pub fn query_bm25(
        &self,
        space: &str,
        query: &str,
        fields: &[(&str, f32)],
        limit: usize,
    ) -> Result<Vec<BM25Hit>> {
        self.query_bm25_filtered(space, query, fields, None, limit, true)
    }

    pub(crate) fn query_bm25_cached(
        &self,
        space: &str,
        query: &str,
        fields: &[(&str, f32)],
        limit: usize,
    ) -> Result<Vec<BM25Hit>> {
        self.query_bm25_filtered(space, query, fields, None, limit, false)
    }

    pub fn query_bm25_in_documents(
        &self,
        space: &str,
        query: &str,
        fields: &[(&str, f32)],
        document_ids: &[i64],
        limit: usize,
    ) -> Result<Vec<BM25Hit>> {
        if document_ids.is_empty() {
            return Ok(Vec::new());
        }

        self.query_bm25_filtered(space, query, fields, Some(document_ids), limit, true)
    }

    pub(crate) fn query_bm25_in_documents_cached(
        &self,
        space: &str,
        query: &str,
        fields: &[(&str, f32)],
        document_ids: &[i64],
        limit: usize,
    ) -> Result<Vec<BM25Hit>> {
        if document_ids.is_empty() {
            return Ok(Vec::new());
        }

        self.query_bm25_filtered(space, query, fields, Some(document_ids), limit, false)
    }

    fn query_bm25_filtered(
        &self,
        space: &str,
        query: &str,
        fields: &[(&str, f32)],
        document_ids: Option<&[i64]>,
        limit: usize,
        reload_reader: bool,
    ) -> Result<Vec<BM25Hit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        if fields.is_empty() {
            return Err(CoreError::Internal(
                "bm25 query requires at least one field".to_string(),
            ));
        }

        let space_indexes = self.get_space_indexes(space)?;

        let schema = space_indexes.tantivy_index.schema();
        let query_fields = fields
            .iter()
            .map(|(name, boost)| {
                let field = resolve_tantivy_field(space_indexes.fields, name)?;
                let field_entry = schema.get_field_entry(field);
                let index_record_option = field_entry
                    .field_type()
                    .get_index_record_option()
                    .ok_or_else(|| {
                        CoreError::Internal(format!(
                            "bm25 field '{}' is not indexed",
                            field_entry.name()
                        ))
                    })?;
                Ok(Bm25FieldSpec {
                    field,
                    boost: *boost,
                    index_record_option,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let Some(parsed_query) =
            build_literal_bm25_query(&space_indexes.tantivy_index, &query_fields, query)?
        else {
            return Ok(Vec::new());
        };
        let parsed_query = if let Some(document_ids) = document_ids {
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, parsed_query),
                (
                    Occur::Must,
                    build_doc_id_filter_query(space_indexes.fields.doc_id, document_ids)?,
                ),
            ])) as Box<dyn Query>
        } else {
            parsed_query
        };
        if reload_reader {
            space_indexes.tantivy_reader.reload()?;
        }
        let searcher = space_indexes.tantivy_reader.searcher();
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

    pub(crate) fn reload_tantivy_reader(&self, space: &str) -> Result<()> {
        let space_indexes = self.get_space_indexes(space)?;
        space_indexes.tantivy_reader.reload()?;
        Ok(())
    }

    pub fn commit_tantivy(&self, space: &str) -> Result<()> {
        let space_indexes = self.get_space_indexes(space)?;
        let mut writer = space_indexes
            .tantivy_writer
            .lock()
            .map_err(|_| CoreError::poisoned("tantivy writer"))?;
        let Some(mut writer) = writer.take() else {
            return Ok(());
        };
        writer.commit()?;
        space_indexes.tantivy_reader.reload()?;
        Ok(())
    }

    pub fn insert_usearch(&self, space: &str, key: i64, vector: &[f32]) -> Result<()> {
        self.batch_insert_usearch(space, &[(key, vector)])
    }

    pub fn batch_insert_usearch(&self, space: &str, entries: &[(i64, &[f32])]) -> Result<()> {
        self.batch_insert_usearch_with_save_mode(space, entries, UsearchSaveMode::Immediate)
    }

    pub(crate) fn batch_insert_usearch_deferred(
        &self,
        space: &str,
        entries: &[(i64, &[f32])],
    ) -> Result<()> {
        self.batch_insert_usearch_with_save_mode(space, entries, UsearchSaveMode::Deferred)
    }

    fn batch_insert_usearch_with_save_mode(
        &self,
        space: &str,
        entries: &[(i64, &[f32])],
        save_mode: UsearchSaveMode,
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let space_indexes = self.get_space_indexes(space)?;

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
        let ensure_started = std::time::Instant::now();
        ensure_usearch_dimensions(&mut index, expected_dimensions)?;
        crate::profile::record_update_stage("usearch_ensure_dimensions", ensure_started.elapsed());

        let target_capacity = index.size().saturating_add(entries.len());
        let reserve_started = std::time::Instant::now();
        index.reserve(target_capacity).map_err(|err| {
            CoreError::Internal(format!(
                "usearch reserve failed for {target_capacity} items: {err}"
            ))
        })?;
        crate::profile::record_update_stage("usearch_reserve", reserve_started.elapsed());

        if matches!(save_mode, UsearchSaveMode::Deferred) {
            self.mark_usearch_dirty(space)?;
        }

        let add_started = std::time::Instant::now();
        for (key, vector) in entries {
            let key = u64::try_from(*key).map_err(|_| {
                CoreError::Internal(format!("usearch key must be non-negative: {}", *key))
            })?;
            index
                .add::<f32>(key, vector)
                .map_err(|err| CoreError::Internal(format!("usearch add failed: {err}")))?;
        }
        crate::profile::record_update_stage("usearch_add", add_started.elapsed());

        if matches!(save_mode, UsearchSaveMode::Immediate) {
            let save_started = std::time::Instant::now();
            save_usearch_index(&index, &space_indexes.usearch_path)?;
            crate::profile::record_update_stage("usearch_save", save_started.elapsed());
        }
        Ok(())
    }

    pub fn delete_usearch(&self, space: &str, keys: &[i64]) -> Result<()> {
        self.delete_usearch_with_save_mode(space, keys, UsearchSaveMode::Immediate)
    }

    pub(crate) fn delete_usearch_deferred(&self, space: &str, keys: &[i64]) -> Result<()> {
        self.delete_usearch_with_save_mode(space, keys, UsearchSaveMode::Deferred)
    }

    fn delete_usearch_with_save_mode(
        &self,
        space: &str,
        keys: &[i64],
        save_mode: UsearchSaveMode,
    ) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }

        let space_indexes = self.get_space_indexes(space)?;
        let index = space_indexes
            .usearch_index
            .write()
            .map_err(|_| CoreError::poisoned("usearch index"))?;

        if matches!(save_mode, UsearchSaveMode::Deferred) {
            self.mark_usearch_dirty(space)?;
        }

        let delete_started = std::time::Instant::now();
        for key in keys {
            let key = u64::try_from(*key).map_err(|_| {
                CoreError::Internal(format!("usearch key must be non-negative: {}", *key))
            })?;
            index
                .remove(key)
                .map_err(|err| CoreError::Internal(format!("usearch remove failed: {err}")))?;
        }
        crate::profile::record_update_stage("usearch_delete", delete_started.elapsed());

        if matches!(save_mode, UsearchSaveMode::Immediate) {
            let save_started = std::time::Instant::now();
            save_usearch_index(&index, &space_indexes.usearch_path)?;
            crate::profile::record_update_stage("usearch_save", save_started.elapsed());
        }
        Ok(())
    }

    pub(crate) fn save_dirty_usearch_indexes(&self) -> Result<()> {
        let spaces = {
            let mut dirty = self
                .dirty_usearch_spaces
                .lock()
                .map_err(|_| CoreError::poisoned("dirty usearch spaces"))?;
            if dirty.is_empty() {
                return Ok(());
            }

            let mut spaces = dirty.drain().collect::<Vec<_>>();
            spaces.sort();
            spaces
        };

        for (index, space) in spaces.iter().enumerate() {
            if let Err(err) = self.save_usearch_for_space(space) {
                let mut dirty = self
                    .dirty_usearch_spaces
                    .lock()
                    .map_err(|_| CoreError::poisoned("dirty usearch spaces"))?;
                for unsaved in &spaces[index..] {
                    dirty.insert(unsaved.clone());
                }
                return Err(err);
            }
            crate::profile::increment_update_count("usearch_saved_spaces", 1);
        }

        Ok(())
    }

    fn mark_usearch_dirty(&self, space: &str) -> Result<()> {
        self.dirty_usearch_spaces
            .lock()
            .map_err(|_| CoreError::poisoned("dirty usearch spaces"))?
            .insert(space.to_string());
        Ok(())
    }

    fn save_usearch_for_space(&self, space: &str) -> Result<()> {
        let space_indexes = self.get_space_indexes(space)?;
        let index = space_indexes
            .usearch_index
            .read()
            .map_err(|_| CoreError::poisoned("usearch index"))?;

        let save_started = std::time::Instant::now();
        save_usearch_index(&index, &space_indexes.usearch_path)?;
        crate::profile::record_update_stage("usearch_save", save_started.elapsed());
        Ok(())
    }

    pub fn query_dense(&self, space: &str, vector: &[f32], limit: usize) -> Result<Vec<DenseHit>> {
        self.query_dense_filtered(space, vector, None, limit)
    }

    pub fn query_dense_in_chunks(
        &self,
        space: &str,
        vector: &[f32],
        chunk_ids: &[i64],
        limit: usize,
    ) -> Result<Vec<DenseHit>> {
        if chunk_ids.is_empty() {
            return Ok(Vec::new());
        }

        let allowed_keys = chunk_ids
            .iter()
            .map(|chunk_id| {
                u64::try_from(*chunk_id).map_err(|_| {
                    CoreError::Internal(format!(
                        "chunk_id must be non-negative for usearch query: {chunk_id}"
                    ))
                })
            })
            .collect::<Result<HashSet<_>>>()?;
        self.query_dense_in_key_set(space, vector, &allowed_keys, limit)
    }

    pub(crate) fn query_dense_in_key_set(
        &self,
        space: &str,
        vector: &[f32],
        allowed_keys: &HashSet<u64>,
        limit: usize,
    ) -> Result<Vec<DenseHit>> {
        if allowed_keys.is_empty() {
            return Ok(Vec::new());
        }

        self.query_dense_filtered(space, vector, Some(allowed_keys), limit)
    }

    fn query_dense_filtered(
        &self,
        space: &str,
        vector: &[f32],
        allowed_keys: Option<&HashSet<u64>>,
        limit: usize,
    ) -> Result<Vec<DenseHit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        if vector.is_empty() {
            return Err(CoreError::Internal(
                "cannot query usearch with empty vector".to_string(),
            ));
        }

        let space_indexes = self.get_space_indexes(space)?;
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

        let matches = if let Some(allowed_keys) = allowed_keys {
            index
                .filtered_search::<f32, _>(vector, limit, |key| allowed_keys.contains(&key))
                .map_err(|err| {
                    CoreError::Internal(format!("usearch filtered query failed: {err}"))
                })?
        } else {
            index
                .search::<f32>(vector, limit)
                .map_err(|err| CoreError::Internal(format!("usearch query failed: {err}")))?
        };
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
        let space_indexes = self.get_space_indexes(space)?;
        let index = space_indexes
            .usearch_index
            .read()
            .map_err(|_| CoreError::poisoned("usearch index"))?;
        Ok(index.size())
    }

    pub fn clear_usearch(&self, space: &str) -> Result<()> {
        let space_indexes = self.get_space_indexes(space)?;
        let index = space_indexes
            .usearch_index
            .write()
            .map_err(|_| CoreError::poisoned("usearch index"))?;
        let clear_started = std::time::Instant::now();
        index
            .reset()
            .map_err(|err| CoreError::Internal(format!("usearch clear failed: {err}")))?;
        std::fs::File::create(&space_indexes.usearch_path)?;
        crate::profile::record_update_stage("usearch_clear", clear_started.elapsed());
        Ok(())
    }

    pub fn get_fts_dirty_documents(&self) -> Result<Vec<FtsDirtyRecord>> {
        self.get_fts_dirty_documents_filtered("", Vec::new())
    }

    pub fn get_fts_dirty_documents_in_space(&self, space_id: i64) -> Result<Vec<FtsDirtyRecord>> {
        self.get_fts_dirty_documents_filtered(
            " AND c.space_id = ?",
            vec![SqlValue::Integer(space_id)],
        )
    }

    pub fn get_fts_dirty_documents_in_collections(
        &self,
        collection_ids: &[i64],
    ) -> Result<Vec<FtsDirtyRecord>> {
        if collection_ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = vec!["?"; collection_ids.len()].join(", ");
        let clause = format!(" AND d.collection_id IN ({placeholders})");
        let params = collection_ids
            .iter()
            .map(|id| SqlValue::Integer(*id))
            .collect::<Vec<_>>();
        self.get_fts_dirty_documents_filtered(&clause, params)
    }

    fn get_fts_dirty_documents_filtered(
        &self,
        scope_clause: &str,
        scope_params: Vec<SqlValue>,
    ) -> Result<Vec<FtsDirtyRecord>> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        let sql = format!(
            "SELECT d.id, d.path, d.title, d.title_source, d.hash, c.path, s.name
             FROM documents d
             JOIN collections c ON c.id = d.collection_id
             JOIN spaces s ON s.id = c.space_id
             WHERE d.fts_dirty = 1{scope_clause}
             ORDER BY d.id ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(scope_params.iter()), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
            ))
        })?;
        let headers = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);

        let mut records = Vec::with_capacity(headers.len());
        for (
            doc_id,
            doc_path,
            doc_title,
            doc_title_source,
            doc_hash,
            collection_path,
            space_name,
        ) in headers
        {
            let chunks = load_chunks_for_doc(&conn, doc_id)?;
            records.push(FtsDirtyRecord {
                doc_id,
                doc_path,
                doc_title,
                doc_title_source: DocumentTitleSource::from_sql(&doc_title_source)?,
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

    pub fn count_documents_in_collection(
        &self,
        collection_id: i64,
        active_only: bool,
    ) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _collection_name = lookup_collection_name(&conn, collection_id)?;
        let active_only = i64::from(active_only);
        query_count(
            &conn,
            "SELECT COUNT(*)
             FROM documents
             WHERE collection_id = ?1
               AND (?2 = 0 OR active = 1)",
            params![collection_id, active_only],
        )
    }

    pub fn count_chunks_in_collection(&self, collection_id: i64) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _collection_name = lookup_collection_name(&conn, collection_id)?;

        query_count(
            &conn,
            "SELECT COUNT(*)
             FROM chunks c
             JOIN documents d ON d.id = c.doc_id
             WHERE d.collection_id = ?1",
            params![collection_id],
        )
    }

    pub fn count_embedded_chunks_in_collection(&self, collection_id: i64) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;
        let _collection_name = lookup_collection_name(&conn, collection_id)?;

        query_count(
            &conn,
            "SELECT COUNT(DISTINCT e.chunk_id)
             FROM embeddings e
             JOIN chunks c ON c.id = e.chunk_id
             JOIN documents d ON d.id = c.doc_id
             WHERE d.collection_id = ?1",
            params![collection_id],
        )
    }

    pub fn count_documents(&self, space_id: Option<i64>) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        match space_id {
            Some(space_id) => {
                let _space_name = lookup_space_name(&conn, space_id)?;
                query_count(
                    &conn,
                    "SELECT COUNT(*)
                     FROM documents d
                     JOIN collections c ON c.id = d.collection_id
                     WHERE c.space_id = ?1",
                    params![space_id],
                )
            }
            None => query_count(&conn, "SELECT COUNT(*) FROM documents", []),
        }
    }

    pub fn count_chunks(&self, space_id: Option<i64>) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        match space_id {
            Some(space_id) => {
                let _space_name = lookup_space_name(&conn, space_id)?;
                query_count(
                    &conn,
                    "SELECT COUNT(*)
                     FROM chunks c
                     JOIN documents d ON d.id = c.doc_id
                     JOIN collections col ON col.id = d.collection_id
                     WHERE col.space_id = ?1",
                    params![space_id],
                )
            }
            None => query_count(&conn, "SELECT COUNT(*) FROM chunks", []),
        }
    }

    pub fn count_embedded_chunks(&self, space_id: Option<i64>) -> Result<usize> {
        let conn = self
            .db
            .lock()
            .map_err(|_| CoreError::poisoned("database"))?;

        match space_id {
            Some(space_id) => {
                let _space_name = lookup_space_name(&conn, space_id)?;
                query_count(
                    &conn,
                    "SELECT COUNT(DISTINCT e.chunk_id)
                     FROM embeddings e
                     JOIN chunks c ON c.id = e.chunk_id
                     JOIN documents d ON d.id = c.doc_id
                     JOIN collections col ON col.id = d.collection_id
                     WHERE col.space_id = ?1",
                    params![space_id],
                )
            }
            None => query_count(&conn, "SELECT COUNT(DISTINCT chunk_id) FROM embeddings", []),
        }
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

    fn get_space_indexes(&self, name: &str) -> Result<Arc<SpaceIndexes>> {
        self.open_space(name)?;
        let spaces = self
            .spaces
            .read()
            .map_err(|_| CoreError::poisoned("spaces"))?;
        spaces.get(name).cloned().ok_or_else(|| {
            KboltError::SpaceNotFound {
                name: name.to_string(),
            }
            .into()
        })
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
        "SELECT id, doc_id, seq, offset, length, heading, kind, retrieval_prefix
         FROM chunks
         WHERE doc_id = ?1
         ORDER BY seq ASC",
    )?;
    let rows = stmt.query_map(params![doc_id], decode_chunk_row)?;
    let chunks = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(chunks)
}

fn load_chunk_ids_for_doc(conn: &Connection, doc_id: i64) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare("SELECT id FROM chunks WHERE doc_id = ?1 ORDER BY seq ASC")?;
    let rows = stmt.query_map(params![doc_id], |row| row.get::<_, i64>(0))?;
    let chunk_ids = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(chunk_ids)
}

fn lookup_chunk_doc_id(conn: &Connection, chunk_id: i64) -> Result<i64> {
    let result = conn.query_row(
        "SELECT doc_id FROM chunks WHERE id = ?1",
        params![chunk_id],
        |row| row.get::<_, i64>(0),
    );
    match result {
        Ok(doc_id) => Ok(doc_id),
        Err(Error::QueryReturnedNoRows) => Err(KboltError::DocumentNotFound {
            path: format!("chunk_id={chunk_id}"),
        }
        .into()),
        Err(err) => Err(err.into()),
    }
}

fn query_count<P: rusqlite::Params>(conn: &Connection, sql: &str, params: P) -> Result<usize> {
    let count: i64 = conn.query_row(sql, params, |row| row.get(0))?;
    Ok(count as usize)
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

fn with_tantivy_writer<T>(
    space_indexes: &SpaceIndexes,
    f: impl FnOnce(&mut IndexWriter) -> Result<T>,
) -> Result<T> {
    let mut writer = space_indexes
        .tantivy_writer
        .lock()
        .map_err(|_| CoreError::poisoned("tantivy writer"))?;

    if writer.is_none() {
        *writer = Some(space_indexes.tantivy_index.writer(50_000_000)?);
    }

    let writer = writer
        .as_mut()
        .ok_or_else(|| CoreError::Internal("failed to initialize tantivy writer".to_string()))?;
    f(writer)
}

fn reject_incompatible_legacy_index(conn: &Connection) -> Result<()> {
    if !table_exists(conn, "documents")? || table_exists(conn, "document_texts")? {
        return Ok(());
    }

    let document_count = query_count(conn, "SELECT COUNT(*) FROM documents", [])?;
    if document_count == 0 {
        return Ok(());
    }

    Err(KboltError::Config(
        "cache index uses an older text-storage format; rebuild the kbolt cache before using this version".to_string(),
    )
    .into())
}

fn ensure_schema_version(conn: &Connection) -> Result<()> {
    let current: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if current > SCHEMA_VERSION {
        return Err(KboltError::Config(format!(
            "cache index schema version {current} is newer than supported version {SCHEMA_VERSION}"
        ))
        .into());
    }

    if current < SCHEMA_VERSION {
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    }

    Ok(())
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool> {
    let exists = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = ?1",
        [table],
        |row| row.get::<_, i64>(0),
    )?;
    Ok(exists != 0)
}

fn ensure_documents_title_source_column(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(documents)")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    let columns = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    drop(stmt);

    if columns.iter().any(|column| column == "title_source") {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE documents ADD COLUMN title_source TEXT NOT NULL DEFAULT 'extracted'",
        [],
    )?;
    Ok(())
}

fn ensure_document_texts_generation_key_column(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(document_texts)")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    let columns = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    drop(stmt);

    if columns.iter().any(|column| column == "generation_key") {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE document_texts ADD COLUMN generation_key TEXT NOT NULL DEFAULT ''",
        [],
    )?;
    Ok(())
}

fn ensure_chunks_retrieval_prefix_column(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(chunks)")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    let columns = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    drop(stmt);

    if columns.iter().any(|column| column == "retrieval_prefix") {
        return Ok(());
    }

    conn.execute("ALTER TABLE chunks ADD COLUMN retrieval_prefix TEXT", [])?;
    Ok(())
}

fn build_literal_bm25_query(
    index: &Index,
    fields: &[Bm25FieldSpec],
    query: &str,
) -> Result<Option<Box<dyn Query>>> {
    let mut clauses = Vec::new();
    for field in fields {
        for token in analyzed_terms_for_field(index, field.field, query)? {
            let term_query: Box<dyn Query> = Box::new(TermQuery::new(
                Term::from_field_text(field.field, &token),
                field.index_record_option,
            ));
            let query = if (field.boost - 1.0).abs() > f32::EPSILON {
                Box::new(BoostQuery::new(term_query, field.boost)) as Box<dyn Query>
            } else {
                term_query
            };
            clauses.push((Occur::Should, query));
        }
    }

    if clauses.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Box::new(BooleanQuery::new(clauses))))
    }
}

fn build_doc_id_filter_query(field: Field, document_ids: &[i64]) -> Result<Box<dyn Query>> {
    let mut terms = Vec::new();
    let mut seen = HashSet::new();
    for doc_id in document_ids {
        let doc_id = u64::try_from(*doc_id).map_err(|_| {
            CoreError::Internal(format!(
                "doc_id must be non-negative for tantivy query: {doc_id}"
            ))
        })?;
        if seen.insert(doc_id) {
            terms.push(Term::from_field_u64(field, doc_id));
        }
    }

    Ok(Box::new(ConstScoreQuery::new(
        Box::new(TermSetQuery::new(terms)),
        0.0,
    )))
}

fn analyzed_terms_for_field(index: &Index, field: Field, query: &str) -> Result<Vec<String>> {
    let mut analyzer = index.tokenizer_for_field(field)?;
    let mut stream = analyzer.token_stream(query);
    let mut terms = Vec::new();
    let mut seen = HashSet::new();
    while let Some(token) = stream.next() {
        if token.text.is_empty() {
            continue;
        }
        let text = token.text.clone();
        if seen.insert(text.clone()) {
            terms.push(text);
        }
    }
    Ok(terms)
}

fn new_usearch_index(dimensions: usize) -> Result<usearch::Index> {
    let options = IndexOptions {
        dimensions,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: 16,
        expansion_add: 200,
        expansion_search: 100,
        ..IndexOptions::default()
    };
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
        doc_id: schema
            .get_field("doc_id")
            .map_err(|_| CoreError::Internal("tantivy schema missing field: doc_id".to_string()))?,
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

fn decode_space_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SpaceRow> {
    Ok(SpaceRow {
        id: row.get(0)?,
        name: row.get(1)?,
        description: row.get(2)?,
        created: row.get(3)?,
    })
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
    let raw_title_source: String = row.get(4)?;
    let title_source = DocumentTitleSource::from_sql(&raw_title_source).map_err(|err| {
        Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(err))
    })?;
    let active_value: i64 = row.get(7)?;
    let fts_dirty_value: i64 = row.get(9)?;
    Ok(DocumentRow {
        id: row.get(0)?,
        collection_id: row.get(1)?,
        path: row.get(2)?,
        title: row.get(3)?,
        title_source,
        hash: row.get(5)?,
        modified: row.get(6)?,
        active: active_value != 0,
        deactivated_at: row.get(8)?,
        fts_dirty: fts_dirty_value != 0,
    })
}

fn decode_document_text_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DocumentTextRow> {
    Ok(DocumentTextRow {
        doc_id: row.get(0)?,
        extractor_key: row.get(1)?,
        source_hash: row.get(2)?,
        text_hash: row.get(3)?,
        generation_key: row.get(4)?,
        text: row.get(5)?,
        created: row.get(6)?,
    })
}

fn decode_chunk_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChunkRow> {
    let offset_value: i64 = row.get(3)?;
    let length_value: i64 = row.get(4)?;
    let kind_raw: String = row.get(6)?;
    let kind = FinalChunkKind::try_from(kind_raw.as_str()).map_err(|err| {
        Error::FromSqlConversionFailure(6, rusqlite::types::Type::Text, Box::new(err))
    })?;
    Ok(ChunkRow {
        id: row.get(0)?,
        doc_id: row.get(1)?,
        seq: row.get(2)?,
        offset: decode_non_negative_usize(offset_value, 3, "chunks.offset")?,
        length: decode_non_negative_usize(length_value, 4, "chunks.length")?,
        heading: row.get(5)?,
        kind,
        retrieval_prefix: row.get(7)?,
    })
}

fn decode_non_negative_usize(
    value: i64,
    column: usize,
    name: &'static str,
) -> rusqlite::Result<usize> {
    if value < 0 {
        return Err(Error::FromSqlConversionFailure(
            column,
            SqlType::Integer,
            Box::new(KboltError::Internal(format!("{name} must not be negative"))),
        ));
    }

    Ok(value as usize)
}

pub(crate) fn chunk_text_from_canonical(document_text: &str, chunk: &ChunkRow) -> Result<String> {
    let label = format!("chunk {}", chunk.id);
    let end = validate_text_span(document_text, chunk.offset, chunk.length, &label)?;

    Ok(document_text[chunk.offset..end].to_string())
}

fn validate_text_span(
    document_text: &str,
    offset: usize,
    length: usize,
    label: &str,
) -> Result<usize> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| CoreError::Internal(format!("{label} text span overflows usize")))?;
    if end > document_text.len() {
        return Err(CoreError::Internal(format!(
            "{label} text span {}..{} exceeds document text length {}",
            offset,
            end,
            document_text.len()
        )));
    }
    if !document_text.is_char_boundary(offset) || !document_text.is_char_boundary(end) {
        return Err(CoreError::Internal(format!(
            "{label} text span {offset}..{end} is not on UTF-8 boundaries"
        )));
    }

    Ok(end)
}

fn missing_document_text_error(doc_id: i64) -> CoreError {
    KboltError::Internal(format!(
        "document {doc_id} is missing persisted canonical text; rebuild the kbolt cache"
    ))
    .into()
}

#[cfg(test)]
mod tests;
