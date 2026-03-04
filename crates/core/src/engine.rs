use std::collections::{HashMap, HashSet};
use std::path::{Component, Path};
use std::sync::Arc;
use std::time::{Instant, UNIX_EPOCH};

use crate::config;
use crate::config::Config;
use crate::ingest::chunk::{chunk_document, resolve_policy};
use crate::ingest::extract::default_registry;
use crate::lock::{LockMode, OperationLock};
use crate::models;
use crate::storage::{ChunkInsert, ChunkRow, CollectionRow, DocumentRow, SpaceResolution, TantivyEntry};
use crate::storage::Storage;
use crate::Result;
use crate::ModelPullEvent;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;
use kbolt_types::{
    ActiveSpace, ActiveSpaceSource, AddCollectionRequest, CollectionInfo, CollectionStatus,
    DocumentResponse, FileEntry, FileError, GetRequest, KboltError, Locator, ModelStatus,
    MultiGetRequest, MultiGetResponse, OmitReason, OmittedFile, PullReport, SearchMode,
    SearchRequest, SearchResponse, SearchResult, SearchSignals, SpaceInfo, SpaceStatus,
    StatusResponse, UpdateOptions, UpdateReport,
};

pub struct Engine {
    storage: Storage,
    config: Config,
    embedder: Option<Arc<dyn models::Embedder>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgnoreListEntry {
    pub space: String,
    pub collection: String,
    pub pattern_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateTarget {
    pub space: String,
    pub collection: CollectionRow,
}

#[derive(Debug, Clone)]
struct SearchCollectionMeta {
    space: String,
    collection: String,
    path: std::path::PathBuf,
}

#[derive(Debug, Clone)]
struct SearchHitCandidate {
    chunk_id: i64,
    bm25_score: f32,
}

#[derive(Debug, Clone)]
struct RankedChunk {
    chunk_id: i64,
    score: f32,
    bm25: Option<f32>,
    dense: Option<f32>,
}

impl Engine {
    pub fn new(config_path: Option<&Path>) -> Result<Self> {
        let config = config::load(config_path)?;
        let storage = Storage::new(&config.cache_dir)?;
        Ok(Self {
            storage,
            config,
            embedder: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_parts(storage: Storage, config: Config) -> Self {
        Self::from_parts_with_embedder(storage, config, None)
    }

    #[cfg(test)]
    pub(crate) fn from_parts_with_embedder(
        storage: Storage,
        config: Config,
        embedder: Option<Arc<dyn models::Embedder>>,
    ) -> Self {
        Self {
            storage,
            config,
            embedder,
        }
    }

    pub fn add_space(&self, name: &str, description: Option<&str>) -> Result<SpaceInfo> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        self.storage.create_space(name, description)?;
        let space = self.storage.get_space(name)?;
        self.build_space_info(&space)
    }

    pub fn remove_space(&self, name: &str) -> Result<()> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        self.storage.delete_space(name)
    }

    pub fn rename_space(&self, old: &str, new: &str) -> Result<()> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        self.storage.rename_space(old, new)
    }

    pub fn describe_space(&self, name: &str, description: &str) -> Result<()> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        self.storage.update_space_description(name, description)
    }

    pub fn list_spaces(&self) -> Result<Vec<SpaceInfo>> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let spaces = self.storage.list_spaces()?;
        let mut infos = Vec::with_capacity(spaces.len());
        for space in spaces {
            infos.push(self.build_space_info(&space)?);
        }
        Ok(infos)
    }

    pub fn space_info(&self, name: &str) -> Result<SpaceInfo> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let space = self.storage.get_space(name)?;
        self.build_space_info(&space)
    }

    pub fn set_default_space(&mut self, name: Option<&str>) -> Result<Option<String>> {
        if let Some(space_name) = name {
            self.storage.get_space(space_name)?;
        }

        self.config.default_space = name.map(ToString::to_string);
        config::save(&self.config)?;
        Ok(self.config.default_space.clone())
    }

    pub fn add_collection(&self, req: AddCollectionRequest) -> Result<CollectionInfo> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let space = self.resolve_space_row(req.space.as_deref(), None)?;
        if !req.path.is_absolute() || !req.path.is_dir() {
            return Err(KboltError::InvalidPath(req.path).into());
        }

        let name = match req.name {
            Some(name) => name,
            None => req
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .map(ToString::to_string)
                .ok_or_else(|| KboltError::InvalidPath(req.path.clone()))?,
        };

        self.storage.create_collection(
            space.id,
            &name,
            &req.path,
            req.description.as_deref(),
            req.extensions.as_deref(),
        )?;

        if !req.no_index {
            self.update_unlocked(UpdateOptions {
                space: Some(space.name.clone()),
                collections: vec![name.clone()],
                no_embed: false,
                dry_run: false,
                verbose: false,
            })?;
        }

        let collection = self.storage.get_collection(space.id, &name)?;
        self.build_collection_info(&space.name, &collection)
    }

    pub fn remove_collection(&self, space: Option<&str>, name: &str) -> Result<()> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let resolved = self.resolve_space_row(space, Some(name))?;
        self.storage.delete_collection(resolved.id, name)?;

        let ignore_path = collection_ignore_file_path(&self.config.config_dir, &resolved.name, name);
        if ignore_path.is_file() {
            std::fs::remove_file(ignore_path)?;
        }

        Ok(())
    }

    pub fn rename_collection(&self, space: Option<&str>, old: &str, new: &str) -> Result<()> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let resolved = self.resolve_space_row(space, Some(old))?;
        let old_ignore_path = collection_ignore_file_path(&self.config.config_dir, &resolved.name, old);
        let new_ignore_path = collection_ignore_file_path(&self.config.config_dir, &resolved.name, new);
        if old_ignore_path.is_file() && new_ignore_path.exists() {
            return Err(KboltError::Internal(format!(
                "cannot rename ignore file: destination already exists: {}",
                new_ignore_path.display()
            ))
            .into());
        }

        self.storage.rename_collection(resolved.id, old, new)?;

        if old_ignore_path.is_file() {
            if let Some(parent) = new_ignore_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if let Err(rename_err) = std::fs::rename(&old_ignore_path, &new_ignore_path) {
                match self.storage.rename_collection(resolved.id, new, old) {
                    Ok(()) => {
                        return Err(KboltError::Internal(format!(
                            "renamed collection was rolled back after ignore rename failure: {rename_err}"
                        ))
                        .into())
                    }
                    Err(rollback_err) => {
                        return Err(KboltError::Internal(format!(
                            "ignore rename failed: {rename_err}; rollback failed: {rollback_err}"
                        ))
                        .into())
                    }
                }
            }
        }

        Ok(())
    }

    pub fn describe_collection(&self, space: Option<&str>, name: &str, desc: &str) -> Result<()> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let resolved = self.resolve_space_row(space, Some(name))?;
        self.storage.update_collection_description(resolved.id, name, desc)
    }

    pub fn list_collections(&self, space: Option<&str>) -> Result<Vec<CollectionInfo>> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let (space_id_filter, spaces_by_id) = if let Some(space_name) = space {
            let resolved = self.resolve_space_row(Some(space_name), None)?;
            let mut map = std::collections::HashMap::new();
            map.insert(resolved.id, resolved.name.clone());
            (Some(resolved.id), map)
        } else {
            let spaces = self.storage.list_spaces()?;
            let map = spaces
                .into_iter()
                .map(|space| (space.id, space.name))
                .collect::<std::collections::HashMap<_, _>>();
            (None, map)
        };

        let collections = self.storage.list_collections(space_id_filter)?;
        let mut infos = Vec::with_capacity(collections.len());
        for collection in collections {
            let space_name = spaces_by_id
                .get(&collection.space_id)
                .ok_or_else(|| {
                    KboltError::Internal(format!(
                        "missing space mapping for collection '{}'",
                        collection.name
                    ))
                })?
                .clone();
            infos.push(self.build_collection_info(&space_name, &collection)?);
        }
        Ok(infos)
    }

    pub fn collection_info(&self, space: Option<&str>, name: &str) -> Result<CollectionInfo> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let resolved = self.resolve_space_row(space, Some(name))?;
        let collection = self.storage.get_collection(resolved.id, name)?;
        self.build_collection_info(&resolved.name, &collection)
    }

    pub fn list_files(
        &self,
        space: Option<&str>,
        collection: &str,
        prefix: Option<&str>,
    ) -> Result<Vec<FileEntry>> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let resolved_space = self.resolve_space_row(space, Some(collection))?;
        let collection_row = self
            .storage
            .get_collection(resolved_space.id, collection)?;
        let normalized_prefix = normalize_list_prefix(prefix)?;
        let file_rows = self
            .storage
            .list_collection_file_rows(collection_row.id, false)?;

        let mut files = Vec::with_capacity(file_rows.len());
        for file_row in file_rows {
            if let Some(prefix) = normalized_prefix.as_deref() {
                if !path_matches_prefix(&file_row.path, prefix) {
                    continue;
                }
            }

            files.push(FileEntry {
                path: file_row.path,
                title: file_row.title,
                docid: short_docid(&file_row.hash),
                active: file_row.active,
                chunk_count: file_row.chunk_count,
                embedded: file_row.chunk_count > 0
                    && file_row.embedded_chunk_count >= file_row.chunk_count,
            });
        }

        Ok(files)
    }

    pub fn get_document(&self, req: GetRequest) -> Result<DocumentResponse> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        self.get_document_unlocked(req)
    }

    fn get_document_unlocked(&self, req: GetRequest) -> Result<DocumentResponse> {
        let GetRequest {
            locator,
            space,
            offset,
            limit,
        } = req;

        let (document, collection_row, space_name) = match locator {
            Locator::Path(locator_path) => {
                let (collection_name, relative_path) = split_collection_path(&locator_path)?;
                let resolved_space =
                    self.resolve_space_row(space.as_deref(), Some(&collection_name))?;
                let collection = self
                    .storage
                    .get_collection(resolved_space.id, &collection_name)?;
                let document = self
                    .storage
                    .get_document_by_path(collection.id, &relative_path)?
                    .ok_or_else(|| KboltError::DocumentNotFound {
                        path: locator_path.clone(),
                    })?;
                (document, collection, resolved_space.name)
            }
            Locator::DocId(docid) => {
                let prefix = normalize_docid(&docid)?;
                let mut candidates = self
                    .storage
                    .get_document_by_hash_prefix(&prefix)?
                    .into_iter()
                    .map(|document| {
                        let collection = self.storage.get_collection_by_id(document.collection_id)?;
                        Ok((document, collection))
                    })
                    .collect::<Result<Vec<_>>>()?;

                if let Some(space_name) = space.as_deref() {
                    let resolved_space = self.resolve_space_row(Some(space_name), None)?;
                    candidates.retain(|(_, collection)| collection.space_id == resolved_space.id);
                }

                if candidates.is_empty() {
                    return Err(KboltError::DocumentNotFound {
                        path: format!("#{prefix}"),
                    }
                    .into());
                }

                if candidates.len() > 1 {
                    return Err(KboltError::InvalidInput(
                        "docid is ambiguous; provide more characters".to_string(),
                    )
                    .into());
                }

                let (document, collection) = candidates.pop().expect("candidate exists");
                let space = self.storage.get_space_by_id(collection.space_id)?;
                (document, collection, space.name)
            }
        };

        let full_path = collection_row.path.join(&document.path);
        let bytes = match std::fs::read(&full_path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(KboltError::FileDeleted(full_path).into())
            }
            Err(err) => return Err(err.into()),
        };

        let raw_content = String::from_utf8_lossy(&bytes).into_owned();
        let stale = sha256_hex(&bytes) != document.hash;

        let lines = raw_content.lines().collect::<Vec<_>>();
        let total_lines = lines.len();
        let start = offset.unwrap_or(0).min(total_lines);
        let requested = limit.unwrap_or(total_lines.saturating_sub(start));
        let end = start.saturating_add(requested).min(total_lines);
        let returned_lines = end.saturating_sub(start);
        let content = if returned_lines == 0 {
            String::new()
        } else {
            lines[start..end].join("\n")
        };

        Ok(DocumentResponse {
            docid: short_docid(&document.hash),
            path: format!("{}/{}", collection_row.name, document.path),
            title: document.title,
            space: space_name,
            collection: collection_row.name,
            content,
            stale,
            total_lines,
            returned_lines,
        })
    }

    pub fn multi_get(&self, req: MultiGetRequest) -> Result<MultiGetResponse> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        if req.max_files == 0 {
            return Err(KboltError::InvalidInput("max_files must be greater than 0".to_string()).into());
        }
        if req.max_bytes == 0 {
            return Err(KboltError::InvalidInput("max_bytes must be greater than 0".to_string()).into());
        }

        let mut documents = Vec::new();
        let mut omitted = Vec::new();
        let mut resolved_count = 0usize;
        let mut warnings = Vec::new();
        let mut consumed_bytes = 0usize;

        for locator in req.locators {
            let document = match self.get_document_unlocked(GetRequest {
                locator,
                space: req.space.clone(),
                offset: None,
                limit: None,
            }) {
                Ok(document) => document,
                Err(err) => match KboltError::from(err) {
                    KboltError::FileDeleted(path) => {
                        warnings.push(format!(
                            "file deleted since indexing: {}. run `kbolt update` to refresh.",
                            path.display()
                        ));
                        continue;
                    }
                    other => return Err(other.into()),
                },
            };
            resolved_count = resolved_count.saturating_add(1);

            let size_bytes = document.content.as_bytes().len();
            if documents.len() >= req.max_files {
                omitted.push(OmittedFile {
                    path: document.path,
                    docid: document.docid,
                    size_bytes,
                    reason: OmitReason::MaxFiles,
                });
                continue;
            }

            if consumed_bytes.saturating_add(size_bytes) > req.max_bytes {
                omitted.push(OmittedFile {
                    path: document.path,
                    docid: document.docid,
                    size_bytes,
                    reason: OmitReason::MaxBytes,
                });
                continue;
            }

            consumed_bytes = consumed_bytes.saturating_add(size_bytes);
            documents.push(document);
        }

        Ok(MultiGetResponse {
            resolved_count,
            documents,
            omitted,
            warnings,
        })
    }

    pub fn search(&self, req: SearchRequest) -> Result<SearchResponse> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let started = Instant::now();
        let query = req.query.trim();
        if query.is_empty() {
            return Err(KboltError::InvalidInput("query cannot be empty".to_string()).into());
        }

        let mode = match req.mode {
            SearchMode::Auto => {
                if self.embedder.is_some() {
                    SearchMode::Auto
                } else {
                    SearchMode::Keyword
                }
            }
            SearchMode::Keyword => SearchMode::Keyword,
            SearchMode::Semantic => SearchMode::Semantic,
            SearchMode::Deep => {
                return Err(KboltError::InvalidInput("deep mode is not implemented yet".to_string()).into())
            }
        };

        let targets = self.resolve_update_targets(&UpdateOptions {
            space: req.space.clone(),
            collections: req.collections.clone(),
            no_embed: true,
            dry_run: false,
            verbose: false,
        })?;

        let staleness_hint = targets
            .iter()
            .map(|target| target.collection.updated.clone())
            .max()
            .map(|updated| format!("Index last updated: {updated}"));

        if req.limit == 0 || targets.is_empty() {
            return Ok(SearchResponse {
                results: Vec::new(),
                query: req.query,
                mode,
                staleness_hint,
                elapsed_ms: started.elapsed().as_millis() as u64,
            });
        }

        let mut collections_by_id: HashMap<i64, SearchCollectionMeta> = HashMap::new();
        for target in &targets {
            collections_by_id.insert(
                target.collection.id,
                SearchCollectionMeta {
                    space: target.space.clone(),
                    collection: target.collection.name.clone(),
                    path: target.collection.path.clone(),
                },
            );
        }

        let ranked_chunks = match mode {
            SearchMode::Keyword => self.rank_keyword_chunks(&targets, query, req.limit, req.min_score)?,
            SearchMode::Auto => self.rank_auto_chunks(&targets, query, req.limit, req.min_score)?,
            SearchMode::Semantic => {
                self.rank_semantic_chunks(&targets, query, req.limit, req.min_score)?
            }
            SearchMode::Deep => unreachable!(),
        };

        if ranked_chunks.is_empty() {
            return Ok(SearchResponse {
                results: Vec::new(),
                query: req.query,
                mode,
                staleness_hint,
                elapsed_ms: started.elapsed().as_millis() as u64,
            });
        }

        let results = self.assemble_search_results(ranked_chunks, &collections_by_id, req.debug)?;

        Ok(SearchResponse {
            results,
            query: req.query,
            mode,
            staleness_hint,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    fn rank_keyword_chunks(
        &self,
        targets: &[UpdateTarget],
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<RankedChunk>> {
        let mut candidates = Vec::new();
        for target in targets {
            let hits = self.storage.query_bm25(
                &target.space,
                query,
                &[("title", 2.0), ("heading", 1.5), ("body", 1.0), ("filepath", 0.5)],
                limit,
            )?;
            for hit in hits {
                candidates.push(SearchHitCandidate {
                    chunk_id: hit.chunk_id,
                    bm25_score: hit.score,
                });
            }
        }

        candidates.sort_by(|left, right| right.bm25_score.total_cmp(&left.bm25_score));
        let max_bm25 = candidates
            .iter()
            .map(|candidate| candidate.bm25_score)
            .fold(0.0_f32, f32::max);

        let mut ranked = Vec::new();
        let mut seen_chunks = HashSet::new();
        for candidate in candidates {
            if !seen_chunks.insert(candidate.chunk_id) {
                continue;
            }

            let normalized_score = if max_bm25 > 0.0 {
                candidate.bm25_score / max_bm25
            } else {
                0.0
            };
            if normalized_score < min_score {
                continue;
            }

            ranked.push(RankedChunk {
                chunk_id: candidate.chunk_id,
                score: normalized_score,
                bm25: Some(normalized_score),
                dense: None,
            });
            if ranked.len() >= limit {
                break;
            }
        }

        Ok(ranked)
    }

    fn rank_semantic_chunks(
        &self,
        targets: &[UpdateTarget],
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<RankedChunk>> {
        let Some(embedder) = self.embedder.as_ref() else {
            return Err(KboltError::ModelNotAvailable {
                name: self.config.models.embedder.id.clone(),
            }
            .into());
        };

        let vectors = embedder.embed_batch(&[query.to_string()])?;
        if vectors.len() != 1 || vectors[0].is_empty() {
            return Err(KboltError::Inference(
                "embedder must return one non-empty query vector".to_string(),
            )
            .into());
        }
        let query_vector = &vectors[0];

        let mut candidates = Vec::new();
        for target in targets {
            let hits = self.storage.query_dense(&target.space, query_vector, limit)?;
            for hit in hits {
                candidates.push(hit);
            }
        }
        candidates.sort_by(|left, right| left.distance.total_cmp(&right.distance));

        let mut ranked = Vec::new();
        let mut seen_chunks = HashSet::new();
        for candidate in candidates {
            if !seen_chunks.insert(candidate.chunk_id) {
                continue;
            }

            let dense_score = dense_distance_to_score(candidate.distance);
            if dense_score < min_score {
                continue;
            }

            ranked.push(RankedChunk {
                chunk_id: candidate.chunk_id,
                score: dense_score,
                bm25: None,
                dense: Some(dense_score),
            });
            if ranked.len() >= limit {
                break;
            }
        }

        Ok(ranked)
    }

    fn rank_auto_chunks(
        &self,
        targets: &[UpdateTarget],
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<RankedChunk>> {
        let candidate_limit = limit.saturating_mul(4).max(limit);
        let keyword = self.rank_keyword_chunks(targets, query, candidate_limit, 0.0)?;
        let semantic = if self.embedder.is_some() {
            self.rank_semantic_chunks(targets, query, candidate_limit, 0.0)?
        } else {
            Vec::new()
        };

        if semantic.is_empty() {
            return Ok(keyword
                .into_iter()
                .filter(|item| item.score >= min_score)
                .take(limit)
                .collect());
        }

        let mut bm25_rank = HashMap::new();
        let mut bm25_score = HashMap::new();
        for (index, item) in keyword.iter().enumerate() {
            bm25_rank.insert(item.chunk_id, index + 1);
            bm25_score.insert(item.chunk_id, item.score);
        }

        let mut dense_rank = HashMap::new();
        let mut dense_score = HashMap::new();
        for (index, item) in semantic.iter().enumerate() {
            dense_rank.insert(item.chunk_id, index + 1);
            dense_score.insert(item.chunk_id, item.score);
        }

        let mut all_chunk_ids = HashSet::new();
        for item in &keyword {
            all_chunk_ids.insert(item.chunk_id);
        }
        for item in &semantic {
            all_chunk_ids.insert(item.chunk_id);
        }

        let mut fused = Vec::new();
        for chunk_id in all_chunk_ids {
            let mut rrf = 0.0_f32;
            let has_bm25 = if let Some(rank) = bm25_rank.get(&chunk_id) {
                rrf += 1.0 / (60.0 + *rank as f32);
                true
            } else {
                false
            };
            let has_dense = if let Some(rank) = dense_rank.get(&chunk_id) {
                rrf += 1.0 / (60.0 + *rank as f32);
                true
            } else {
                false
            };
            if has_bm25 && has_dense {
                rrf *= 1.2;
            }

            fused.push(RankedChunk {
                chunk_id,
                score: rrf,
                bm25: bm25_score.get(&chunk_id).copied(),
                dense: dense_score.get(&chunk_id).copied(),
            });
        }
        fused.sort_by(|left, right| right.score.total_cmp(&left.score));

        let max_score = fused
            .iter()
            .map(|item| item.score)
            .fold(0.0_f32, f32::max);
        if max_score > 0.0 {
            for item in &mut fused {
                item.score /= max_score;
            }
        }

        Ok(fused
            .into_iter()
            .filter(|item| item.score >= min_score)
            .take(limit)
            .collect())
    }

    fn assemble_search_results(
        &self,
        ranked_chunks: Vec<RankedChunk>,
        collections_by_id: &HashMap<i64, SearchCollectionMeta>,
        debug: bool,
    ) -> Result<Vec<SearchResult>> {
        let chunk_ids = ranked_chunks
            .iter()
            .map(|candidate| candidate.chunk_id)
            .collect::<Vec<_>>();
        let chunk_rows = self.storage.get_chunks(&chunk_ids)?;
        let chunk_by_id = chunk_rows
            .into_iter()
            .map(|chunk| (chunk.id, chunk))
            .collect::<HashMap<_, _>>();

        let mut docs_by_id: HashMap<i64, DocumentRow> = HashMap::new();
        let mut chunks_by_doc: HashMap<i64, Vec<ChunkRow>> = HashMap::new();
        let neighbor_window = self.config.chunking.defaults.neighbor_window;
        let mut results = Vec::new();
        for ranked in ranked_chunks {
            let Some(chunk) = chunk_by_id.get(&ranked.chunk_id) else {
                continue;
            };

            let document = if let Some(existing) = docs_by_id.get(&chunk.doc_id) {
                existing.clone()
            } else {
                let loaded = self.storage.get_document_by_id(chunk.doc_id)?;
                docs_by_id.insert(chunk.doc_id, loaded.clone());
                loaded
            };
            if !document.active {
                continue;
            }

            let Some(collection) = collections_by_id.get(&document.collection_id) else {
                continue;
            };

            let full_path = collection.path.join(&document.path);
            let bytes = match std::fs::read(&full_path) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            if neighbor_window > 0 && !chunks_by_doc.contains_key(&chunk.doc_id) {
                chunks_by_doc.insert(
                    chunk.doc_id,
                    self.storage.get_chunks_for_document(chunk.doc_id)?,
                );
            }
            let text = search_text_with_neighbors(
                &bytes,
                chunk,
                chunks_by_doc.get(&chunk.doc_id),
                neighbor_window,
            );

            results.push(SearchResult {
                docid: short_docid(&document.hash),
                path: format!("{}/{}", collection.collection, document.path),
                title: document.title,
                space: collection.space.clone(),
                collection: collection.collection.clone(),
                heading: chunk.heading.clone(),
                text,
                score: ranked.score,
                signals: if debug {
                    Some(SearchSignals {
                        bm25: ranked.bm25,
                        dense: ranked.dense,
                        rrf: ranked.score,
                        reranker: None,
                    })
                } else {
                    None
                },
            });
        }

        Ok(results)
    }

    pub fn resolve_space(&self, explicit: Option<&str>) -> Result<String> {
        let resolved = self.resolve_space_row(explicit, None)?;
        Ok(resolved.name)
    }

    pub fn current_space(&self, explicit: Option<&str>) -> Result<Option<ActiveSpace>> {
        let resolved = self.resolve_preferred_space(explicit)?;
        Ok(resolved.map(|(space, source)| ActiveSpace {
            name: space.name,
            source,
        }))
    }

    pub fn update(&self, options: UpdateOptions) -> Result<UpdateReport> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        self.update_unlocked(options)
    }

    fn update_unlocked(&self, options: UpdateOptions) -> Result<UpdateReport> {
        let started = Instant::now();
        let mut report = UpdateReport {
            scanned: 0,
            skipped_mtime: 0,
            skipped_hash: 0,
            added: 0,
            updated: 0,
            deactivated: 0,
            reactivated: 0,
            reaped: 0,
            embedded: 0,
            errors: Vec::new(),
            elapsed_ms: 0,
        };

        if !options.dry_run {
            self.replay_fts_dirty_documents(&mut report)?;
        }

        let targets = self.resolve_update_targets(&options)?;
        if targets.is_empty() {
            report.elapsed_ms = started.elapsed().as_millis() as u64;
            return Ok(report);
        }

        self.reconcile_dense_integrity(&targets, &options)?;

        let mut fts_dirty_by_space: HashMap<String, HashSet<i64>> = HashMap::new();
        for target in &targets {
            self.update_collection_target(target, &options, &mut report, &mut fts_dirty_by_space)?;
        }

        if !options.dry_run {
            for (space, doc_ids) in fts_dirty_by_space {
                if doc_ids.is_empty() {
                    continue;
                }

                self.storage.commit_tantivy(&space)?;
                let mut ids = doc_ids.into_iter().collect::<Vec<_>>();
                ids.sort_unstable();
                self.storage.batch_clear_fts_dirty(&ids)?;
            }

            self.embed_pending_chunks(&options, &mut report)?;

            let reaped = self.storage.reap_documents(self.config.reaping.days)?;
            report.reaped = reaped.len();
        }

        report.elapsed_ms = started.elapsed().as_millis() as u64;
        Ok(report)
    }

    fn reconcile_dense_integrity(&self, targets: &[UpdateTarget], options: &UpdateOptions) -> Result<()> {
        if options.no_embed || options.dry_run {
            return Ok(());
        }

        let mut visited_spaces = HashSet::new();
        for target in targets {
            if !visited_spaces.insert(target.collection.space_id) {
                continue;
            }

            let sqlite_count = self
                .storage
                .count_embedded_chunks(Some(target.collection.space_id))?;
            let usearch_count = self.storage.count_usearch(&target.space)?;

            if sqlite_count == usearch_count {
                continue;
            }

            self.storage
                .delete_embeddings_for_space(target.collection.space_id)?;
            self.storage.clear_usearch(&target.space)?;
        }

        Ok(())
    }

    fn embed_pending_chunks(&self, options: &UpdateOptions, report: &mut UpdateReport) -> Result<()> {
        if options.no_embed || options.dry_run {
            return Ok(());
        }

        let Some(embedder) = self.embedder.as_ref() else {
            return Ok(());
        };

        let model = self.config.models.embedder.id.as_str();
        loop {
            let backlog = self.storage.get_unembedded_chunks(model, 64)?;
            if backlog.is_empty() {
                break;
            }

            let mut chunk_ids = Vec::new();
            let mut spaces = Vec::new();
            let mut texts = Vec::new();
            for record in backlog {
                let full_path = record.collection_path.join(&record.doc_path);
                let bytes = match std::fs::read(&full_path) {
                    Ok(bytes) => bytes,
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(err) => {
                        report.errors.push(file_error(
                            Some(full_path),
                            format!("embed read failed: {err}"),
                        ));
                        continue;
                    }
                };

                let mut text = chunk_text_from_bytes(&bytes, record.offset, record.length);
                if text.trim().is_empty() {
                    text = " ".to_string();
                }

                chunk_ids.push(record.chunk_id);
                spaces.push(record.space_name);
                texts.push(text);
            }

            if chunk_ids.is_empty() {
                break;
            }

            let vectors = embedder.embed_batch(&texts)?;
            if vectors.len() != chunk_ids.len() {
                return Err(KboltError::Inference(format!(
                    "embedder returned {} vectors for {} chunks",
                    vectors.len(),
                    chunk_ids.len()
                ))
                .into());
            }

            let mut grouped_vectors: HashMap<String, Vec<(i64, Vec<f32>)>> = HashMap::new();
            let mut embedding_rows = Vec::with_capacity(chunk_ids.len());
            for ((chunk_id, space), vector) in chunk_ids
                .iter()
                .zip(spaces.iter())
                .zip(vectors.into_iter())
            {
                if vector.is_empty() {
                    return Err(KboltError::Inference(format!(
                        "embedder returned an empty vector for chunk {chunk_id}"
                    ))
                    .into());
                }

                grouped_vectors
                    .entry(space.clone())
                    .or_default()
                    .push((*chunk_id, vector));
                embedding_rows.push((*chunk_id, model));
            }

            for (space, vectors) in grouped_vectors {
                let refs = vectors
                    .iter()
                    .map(|(chunk_id, vector)| (*chunk_id, vector.as_slice()))
                    .collect::<Vec<_>>();
                self.storage.batch_insert_usearch(&space, &refs)?;
            }

            self.storage.insert_embeddings(&embedding_rows)?;
            report.embedded = report.embedded.saturating_add(chunk_ids.len());
        }

        Ok(())
    }

    fn replay_fts_dirty_documents(&self, report: &mut UpdateReport) -> Result<()> {
        let records = self.storage.get_fts_dirty_documents()?;
        if records.is_empty() {
            return Ok(());
        }

        let mut cleared_by_space: HashMap<String, Vec<i64>> = HashMap::new();
        for record in records {
            let space_name = record.space_name;
            let doc_id = record.doc_id;

            if record.chunks.is_empty() {
                cleared_by_space.entry(space_name).or_default().push(doc_id);
                continue;
            }

            let full_path = record.collection_path.join(&record.doc_path);
            let bytes = match std::fs::read(&full_path) {
                Ok(bytes) => bytes,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                Err(err) => {
                    report.errors.push(file_error(
                        Some(full_path),
                        format!("fts replay read failed: {err}"),
                    ));
                    continue;
                }
            };
            if sha256_hex(&bytes) != record.doc_hash {
                continue;
            }

            self.storage.delete_tantivy_by_doc(&space_name, doc_id)?;

            let file_body = String::from_utf8_lossy(&bytes).into_owned();
            let entries = record
                .chunks
                .iter()
                .map(|chunk| {
                    let chunk_body = chunk_text_from_bytes(&bytes, chunk.offset, chunk.length);
                    let source_body = if chunk_body.is_empty() {
                        file_body.as_str()
                    } else {
                        chunk_body.as_str()
                    };
                    TantivyEntry {
                        chunk_id: chunk.id,
                        doc_id,
                        filepath: record.doc_path.clone(),
                        title: record.doc_title.clone(),
                        heading: chunk.heading.clone(),
                        body: retrieval_text_with_prefix(
                            source_body,
                            record.doc_title.as_str(),
                            chunk.heading.as_deref(),
                            self.config.chunking.defaults.contextual_prefix,
                        ),
                    }
                })
                .collect::<Vec<_>>();

            self.storage.index_tantivy(&space_name, &entries)?;
            cleared_by_space.entry(space_name).or_default().push(doc_id);
        }

        for (space_name, mut doc_ids) in cleared_by_space {
            if doc_ids.is_empty() {
                continue;
            }

            doc_ids.sort_unstable();
            doc_ids.dedup();
            self.storage.commit_tantivy(&space_name)?;
            self.storage.batch_clear_fts_dirty(&doc_ids)?;
        }

        Ok(())
    }

    pub fn status(&self, space: Option<&str>) -> Result<StatusResponse> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let (spaces, totals_scope) = if let Some(space_name) = space {
            let resolved = self.resolve_space_row(Some(space_name), None)?;
            (vec![resolved.clone()], Some(resolved.id))
        } else {
            (self.storage.list_spaces()?, None)
        };

        let mut space_statuses = Vec::with_capacity(spaces.len());
        for space_row in spaces {
            let collections = self.storage.list_collections(Some(space_row.id))?;
            let mut collection_statuses = Vec::with_capacity(collections.len());
            let mut last_updated: Option<String> = None;

            for collection in collections {
                if last_updated
                    .as_ref()
                    .map(|existing| collection.updated > *existing)
                    .unwrap_or(true)
                {
                    last_updated = Some(collection.updated.clone());
                }

                let documents = self
                    .storage
                    .count_documents_in_collection(collection.id, false)?;
                let active_documents = self.storage.count_documents_in_collection(collection.id, true)?;
                let chunks = self.storage.count_chunks_in_collection(collection.id)?;
                let embedded_chunks = self
                    .storage
                    .count_embedded_chunks_in_collection(collection.id)?;

                collection_statuses.push(CollectionStatus {
                    name: collection.name,
                    path: collection.path,
                    documents,
                    active_documents,
                    chunks,
                    embedded_chunks,
                    last_updated: collection.updated,
                });
            }

            space_statuses.push(SpaceStatus {
                name: space_row.name,
                description: space_row.description,
                collections: collection_statuses,
                last_updated,
            });
        }

        let models = self.model_status_unlocked()?;

        Ok(StatusResponse {
            spaces: space_statuses,
            models,
            cache_dir: self.config.cache_dir.clone(),
            config_dir: self.config.config_dir.clone(),
            total_documents: self.storage.count_documents(totals_scope)?,
            total_chunks: self.storage.count_chunks(totals_scope)?,
            total_embedded: self.storage.count_embedded_chunks(totals_scope)?,
            disk_usage: self.storage.disk_usage()?,
        })
    }

    pub fn model_status(&self) -> Result<ModelStatus> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        self.model_status_unlocked()
    }

    pub fn read_collection_ignore(
        &self,
        space: Option<&str>,
        collection: &str,
    ) -> Result<(String, Option<String>)> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let resolved_space = self.resolve_space_row(space, Some(collection))?;
        self.storage
            .get_collection(resolved_space.id, collection)?;

        let path = collection_ignore_file_path(&self.config.config_dir, &resolved_space.name, collection);
        let raw = match std::fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok((resolved_space.name, None))
            }
            Err(err) => return Err(err.into()),
        };

        let trimmed = raw.trim_end_matches('\n').to_string();
        if trimmed.trim().is_empty() {
            return Ok((resolved_space.name, None));
        }

        Ok((resolved_space.name, Some(trimmed)))
    }

    pub fn add_collection_ignore_pattern(
        &self,
        space: Option<&str>,
        collection: &str,
        pattern: &str,
    ) -> Result<(String, String)> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let resolved_space = self.resolve_space_row(space, Some(collection))?;
        self.storage
            .get_collection(resolved_space.id, collection)?;

        let normalized_pattern = validate_ignore_pattern(pattern)?;
        let path = collection_ignore_file_path(&self.config.config_dir, &resolved_space.name, collection);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        use std::io::Write;
        writeln!(file, "{normalized_pattern}")?;

        Ok((resolved_space.name, normalized_pattern))
    }

    pub fn remove_collection_ignore_pattern(
        &self,
        space: Option<&str>,
        collection: &str,
        pattern: &str,
    ) -> Result<(String, usize)> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let resolved_space = self.resolve_space_row(space, Some(collection))?;
        self.storage
            .get_collection(resolved_space.id, collection)?;

        let normalized_pattern = validate_ignore_pattern(pattern)?;
        let path = collection_ignore_file_path(&self.config.config_dir, &resolved_space.name, collection);
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok((resolved_space.name, 0))
            }
            Err(err) => return Err(err.into()),
        };

        let mut removed_count = 0usize;
        let mut remaining = Vec::new();
        for line in raw.lines() {
            if line == normalized_pattern {
                removed_count = removed_count.saturating_add(1);
            } else {
                remaining.push(line.to_string());
            }
        }

        if removed_count == 0 {
            return Ok((resolved_space.name, 0));
        }

        if remaining.is_empty() {
            std::fs::remove_file(path)?;
            return Ok((resolved_space.name, removed_count));
        }

        let mut content = remaining.join("\n");
        content.push('\n');
        std::fs::write(path, content)?;
        Ok((resolved_space.name, removed_count))
    }

    pub fn list_collection_ignores(&self, space: Option<&str>) -> Result<Vec<IgnoreListEntry>> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let (space_id_filter, spaces_by_id) = if let Some(space_name) = space {
            let resolved = self.resolve_space_row(Some(space_name), None)?;
            let mut map = std::collections::HashMap::new();
            map.insert(resolved.id, resolved.name.clone());
            (Some(resolved.id), map)
        } else {
            let spaces = self.storage.list_spaces()?;
            let map = spaces
                .into_iter()
                .map(|space| (space.id, space.name))
                .collect::<std::collections::HashMap<_, _>>();
            (None, map)
        };

        let collections = self.storage.list_collections(space_id_filter)?;
        let mut entries = Vec::new();
        for collection in collections {
            let space_name = spaces_by_id
                .get(&collection.space_id)
                .ok_or_else(|| {
                    KboltError::Internal(format!(
                        "missing space mapping for collection '{}'",
                        collection.name
                    ))
                })?
                .clone();
            let path = collection_ignore_file_path(&self.config.config_dir, &space_name, &collection.name);
            if !path.is_file() {
                continue;
            }

            let raw = std::fs::read_to_string(path)?;
            entries.push(IgnoreListEntry {
                space: space_name,
                collection: collection.name,
                pattern_count: count_ignore_patterns(&raw),
            });
        }
        entries.sort_by(|left, right| {
            left.space
                .cmp(&right.space)
                .then(left.collection.cmp(&right.collection))
        });
        Ok(entries)
    }

    pub fn prepare_collection_ignore_edit(
        &self,
        space: Option<&str>,
        collection: &str,
    ) -> Result<(String, std::path::PathBuf)> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let resolved_space = self.resolve_space_row(space, Some(collection))?;
        self.storage.get_collection(resolved_space.id, collection)?;

        let path = collection_ignore_file_path(&self.config.config_dir, &resolved_space.name, collection);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let _file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&path)?;

        Ok((resolved_space.name, path))
    }

    fn model_status_unlocked(&self) -> Result<ModelStatus> {
        models::status(&self.config.models, &self.model_dir())
    }

    pub fn pull_models(&self) -> Result<PullReport> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        models::pull(&self.config.models, &self.model_dir())
    }

    pub fn pull_models_with_progress<F>(&self, on_event: F) -> Result<PullReport>
    where
        F: FnMut(ModelPullEvent),
    {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        models::pull_with_progress(&self.config.models, &self.model_dir(), on_event)
    }

    pub fn resolve_update_targets(&self, options: &UpdateOptions) -> Result<Vec<UpdateTarget>> {
        let mut targets = Vec::new();

        if options.collections.is_empty() {
            return self.resolve_update_targets_for_all_collections(options.space.as_deref());
        }

        let mut seen = std::collections::HashSet::new();
        for raw_collection_name in &options.collections {
            let collection_name = raw_collection_name.trim();
            if collection_name.is_empty() {
                return Err(
                    KboltError::InvalidInput("collection names cannot be empty".to_string()).into(),
                );
            }

            let resolved_space =
                self.resolve_space_row(options.space.as_deref(), Some(collection_name))?;
            let collection = self
                .storage
                .get_collection(resolved_space.id, collection_name)?;

            if seen.insert((collection.space_id, collection.name.clone())) {
                targets.push(UpdateTarget {
                    space: resolved_space.name,
                    collection,
                });
            }
        }

        Ok(targets)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    fn model_dir(&self) -> std::path::PathBuf {
        self.config.cache_dir.join("models")
    }

    fn acquire_operation_lock(&self, mode: LockMode) -> Result<OperationLock> {
        OperationLock::acquire(&self.config.cache_dir, mode)
    }

    fn build_space_info(&self, space: &crate::storage::SpaceRow) -> Result<SpaceInfo> {
        let collection_count = self.storage.list_collections(Some(space.id))?.len();
        let document_count = self.storage.count_documents(Some(space.id))?;
        let chunk_count = self.storage.count_chunks(Some(space.id))?;

        Ok(SpaceInfo {
            name: space.name.clone(),
            description: space.description.clone(),
            collection_count,
            document_count,
            chunk_count,
            created: space.created.clone(),
        })
    }

    fn build_collection_info(&self, space_name: &str, collection: &CollectionRow) -> Result<CollectionInfo> {
        let document_count = self
            .storage
            .count_documents_in_collection(collection.id, false)?;
        let active_document_count = self.storage.count_documents_in_collection(collection.id, true)?;
        let chunk_count = self.storage.count_chunks_in_collection(collection.id)?;
        let embedded_chunk_count = self
            .storage
            .count_embedded_chunks_in_collection(collection.id)?;

        Ok(CollectionInfo {
            name: collection.name.clone(),
            space: space_name.to_string(),
            path: collection.path.clone(),
            description: collection.description.clone(),
            extensions: collection.extensions.clone(),
            document_count,
            active_document_count,
            chunk_count,
            embedded_chunk_count,
            created: collection.created.clone(),
            updated: collection.updated.clone(),
        })
    }

    fn resolve_space_row(
        &self,
        explicit: Option<&str>,
        collection_for_lookup: Option<&str>,
    ) -> Result<crate::storage::SpaceRow> {
        if let Some((space, _source)) = self.resolve_preferred_space(explicit)? {
            return Ok(space);
        }

        if let Some(collection) = collection_for_lookup {
            return match self.storage.find_space_for_collection(collection)? {
                SpaceResolution::Found(space) => Ok(space),
                SpaceResolution::Ambiguous(spaces) => Err(KboltError::AmbiguousSpace {
                    collection: collection.to_string(),
                    spaces,
                }
                .into()),
                SpaceResolution::NotFound => Err(KboltError::CollectionNotFound {
                    name: collection.to_string(),
                }
                .into()),
            };
        }

        Err(KboltError::NoActiveSpace.into())
    }

    fn resolve_preferred_space(
        &self,
        explicit: Option<&str>,
    ) -> Result<Option<(crate::storage::SpaceRow, ActiveSpaceSource)>> {
        if let Some(space_name) = explicit {
            let space = self.storage.get_space(space_name)?;
            return Ok(Some((space, ActiveSpaceSource::Flag)));
        }

        if let Ok(space_name) = std::env::var("KBOLT_SPACE") {
            let trimmed = space_name.trim();
            if !trimmed.is_empty() {
                let space = self.storage.get_space(trimmed)?;
                return Ok(Some((space, ActiveSpaceSource::EnvVar)));
            }
        }

        if let Some(space_name) = self.config.default_space.as_deref() {
            let space = self.storage.get_space(space_name)?;
            return Ok(Some((space, ActiveSpaceSource::ConfigDefault)));
        }

        Ok(None)
    }

    fn resolve_update_targets_for_all_collections(
        &self,
        space: Option<&str>,
    ) -> Result<Vec<UpdateTarget>> {
        let (space_id_filter, spaces_by_id) = if let Some(space_name) = space {
            let resolved = self.resolve_space_row(Some(space_name), None)?;
            let mut map = std::collections::HashMap::new();
            map.insert(resolved.id, resolved.name.clone());
            (Some(resolved.id), map)
        } else {
            let spaces = self.storage.list_spaces()?;
            let map = spaces
                .into_iter()
                .map(|space| (space.id, space.name))
                .collect::<std::collections::HashMap<_, _>>();
            (None, map)
        };

        let collections = self.storage.list_collections(space_id_filter)?;
        let mut targets = Vec::with_capacity(collections.len());
        for collection in collections {
            let space_name = spaces_by_id
                .get(&collection.space_id)
                .ok_or_else(|| {
                    KboltError::Internal(format!(
                        "missing space mapping for collection '{}'",
                        collection.name
                    ))
                })?
                .clone();
            targets.push(UpdateTarget {
                space: space_name,
                collection,
            });
        }

        Ok(targets)
    }

    fn update_collection_target(
        &self,
        target: &UpdateTarget,
        options: &UpdateOptions,
        report: &mut UpdateReport,
        fts_dirty_by_space: &mut HashMap<String, HashSet<i64>>,
    ) -> Result<()> {
        let all_documents = self.storage.list_documents(target.collection.id, false)?;
        let mut docs_by_path: HashMap<String, DocumentRow> = all_documents
            .into_iter()
            .map(|doc| (doc.path.clone(), doc))
            .collect();
        let mut seen_paths = HashSet::new();
        let extension_filter = normalized_extension_filter(target.collection.extensions.as_deref());
        let ignore_matcher = load_collection_ignore_matcher(
            &self.config.config_dir,
            &target.collection.path,
            &target.space,
            &target.collection.name,
        )?;
        let extractor_registry = default_registry();
        let mut touched_collection = false;

        for entry in WalkDir::new(&target.collection.path)
            .follow_links(false)
            .into_iter()
            .filter_entry(|entry| {
                !entry.file_type().is_dir() || !is_hard_ignored_dir_name(entry.file_name())
            })
        {
            let entry = match entry {
                Ok(item) => item,
                Err(err) => {
                    report.errors.push(file_error(
                        err.path().map(Path::to_path_buf),
                        format!("walkdir error: {err}"),
                    ));
                    continue;
                }
            };

            if !entry.file_type().is_file() {
                continue;
            }

            if is_hard_ignored_file(entry.path()) {
                continue;
            }

            if !extension_allowed(entry.path(), extension_filter.as_ref()) {
                continue;
            }

            let relative_path = match collection_relative_path(&target.collection.path, entry.path()) {
                Ok(path) => path,
                Err(err) => {
                    report.errors.push(file_error(Some(entry.path().to_path_buf()), err.to_string()));
                    continue;
                }
            };

            if let Some(matcher) = ignore_matcher.as_ref() {
                if matcher.matched(Path::new(&relative_path), false).is_ignore() {
                    continue;
                }
            }

            let Some(extractor) = extractor_registry.resolve_for_path(entry.path()) else {
                continue;
            };

            report.scanned += 1;
            seen_paths.insert(relative_path.clone());

            let metadata = match entry.metadata() {
                Ok(data) => data,
                Err(err) => {
                    report
                        .errors
                        .push(file_error(Some(entry.path().to_path_buf()), err.to_string()));
                    continue;
                }
            };

            let modified = match modified_token(&metadata) {
                Ok(value) => value,
                Err(err) => {
                    report.errors.push(file_error(
                        Some(entry.path().to_path_buf()),
                        format!("modified timestamp error: {err}"),
                    ));
                    continue;
                }
            };

            if let Some(existing) = docs_by_path.get(&relative_path) {
                if existing.active && existing.modified == modified {
                    report.skipped_mtime += 1;
                    continue;
                }
            }

            let bytes = match std::fs::read(entry.path()) {
                Ok(data) => data,
                Err(err) => {
                    report
                        .errors
                        .push(file_error(Some(entry.path().to_path_buf()), err.to_string()));
                    continue;
                }
            };
            let hash = sha256_hex(&bytes);
            let mut title = file_title(entry.path());

            let existing = docs_by_path.get(&relative_path).cloned();
            if let Some(doc) = existing.as_ref() {
                if doc.hash == hash {
                    if doc.active {
                        report.skipped_hash += 1;
                    } else {
                        report.reactivated += 1;
                    }

                    if !options.dry_run {
                        self.storage.update_document_metadata(doc.id, &title, &modified)?;
                    }
                    continue;
                }

                report.updated += 1;
                if !doc.active {
                    report.reactivated += 1;
                }
            } else {
                report.added += 1;
            }

            if options.dry_run {
                continue;
            }

            let extracted = match extractor.extract(entry.path(), &bytes) {
                Ok(document) => document,
                Err(err) => {
                    report.errors.push(file_error(
                        Some(entry.path().to_path_buf()),
                        format!("extract failed: {err}"),
                    ));
                    continue;
                }
            };
            if let Some(extracted_title) = extracted.title.as_deref() {
                title = extracted_title.to_string();
            }

            let doc_id = self.storage.upsert_document(
                target.collection.id,
                &relative_path,
                &title,
                &hash,
                &modified,
            )?;

            if let Some(doc) = existing.as_ref() {
                let old_chunk_ids = self.storage.delete_chunks_for_document(doc.id)?;
                if !old_chunk_ids.is_empty() {
                    self.storage.delete_tantivy(&target.space, &old_chunk_ids)?;
                    self.storage.delete_usearch(&target.space, &old_chunk_ids)?;
                }
            }

            let policy = resolve_policy(
                &self.config.chunking,
                Some(extractor.profile_key()),
                None,
            );
            let final_chunks = chunk_document(&extracted, &policy);

            let chunk_inserts = final_chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| ChunkInsert {
                    seq: index as i32,
                    offset: chunk.offset,
                    length: chunk.length,
                    heading: chunk.heading.clone(),
                    kind: chunk.kind.as_storage_kind().to_string(),
                })
                .collect::<Vec<_>>();
            let body = String::from_utf8_lossy(&bytes).into_owned();
            let chunk_ids = self.storage.insert_chunks(doc_id, &chunk_inserts)?;

            if !chunk_ids.is_empty() {
                let entries = chunk_ids
                    .iter()
                    .zip(final_chunks.iter())
                    .map(|(chunk_id, chunk)| TantivyEntry {
                        chunk_id: *chunk_id,
                        doc_id,
                        filepath: relative_path.clone(),
                        title: title.clone(),
                        heading: chunk.heading.clone(),
                        body: retrieval_text_with_prefix(
                            if chunk.text.is_empty() {
                                body.as_str()
                            } else {
                                chunk.text.as_str()
                            },
                            title.as_str(),
                            chunk.heading.as_deref(),
                            policy.contextual_prefix,
                        ),
                    })
                    .collect::<Vec<_>>();
                self.storage.index_tantivy(
                    &target.space,
                    &entries,
                )?;
                fts_dirty_by_space
                    .entry(target.space.clone())
                    .or_default()
                    .insert(doc_id);
            }

            docs_by_path.insert(
                relative_path.clone(),
                self.storage
                    .get_document_by_path(target.collection.id, &relative_path)?
                    .ok_or_else(|| {
                        KboltError::Internal(format!(
                            "document missing after upsert: collection={}, path={relative_path}",
                            target.collection.id
                        ))
                    })?,
            );
            touched_collection = true;
        }

        for doc in docs_by_path.values() {
            if doc.active && !seen_paths.contains(&doc.path) {
                report.deactivated += 1;
                if !options.dry_run {
                    self.storage.deactivate_document(doc.id)?;
                    touched_collection = true;
                }
            }
        }

        if touched_collection && !options.dry_run {
            self.storage.update_collection_timestamp(target.collection.id)?;
        }

        Ok(())
    }
}

fn normalized_extension_filter(raw: Option<&[String]>) -> Option<HashSet<String>> {
    raw.map(|items| {
        items
            .iter()
            .filter_map(|item| {
                let normalized = item.trim().trim_start_matches('.').to_ascii_lowercase();
                if normalized.is_empty() {
                    None
                } else {
                    Some(normalized)
                }
            })
            .collect::<HashSet<_>>()
    })
    .filter(|items| !items.is_empty())
}

fn normalize_list_prefix(prefix: Option<&str>) -> Result<Option<String>> {
    let Some(raw_prefix) = prefix else {
        return Ok(None);
    };

    let trimmed = raw_prefix.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let parsed = Path::new(trimmed);
    if parsed.is_absolute() {
        return Err(KboltError::InvalidInput("prefix must be relative".to_string()).into());
    }

    let mut parts = Vec::new();
    for component in parsed.components() {
        match component {
            Component::Normal(item) => parts.push(item.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(
                    KboltError::InvalidInput("prefix must not traverse directories".to_string())
                        .into(),
                )
            }
        }
    }

    if parts.is_empty() {
        return Ok(None);
    }

    Ok(Some(parts.join("/")))
}

fn split_collection_path(locator: &str) -> Result<(String, String)> {
    let trimmed = locator.trim();
    if trimmed.is_empty() {
        return Err(
            KboltError::InvalidInput("path locator must be '<collection>/<path>'".to_string())
                .into(),
        );
    }

    let parsed = Path::new(trimmed);
    if parsed.is_absolute() {
        return Err(KboltError::InvalidInput("path locator must be relative".to_string()).into());
    }

    let mut parts = Vec::new();
    for component in parsed.components() {
        match component {
            Component::Normal(item) => parts.push(item.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(
                    KboltError::InvalidInput(
                        "path locator must not traverse directories".to_string(),
                    )
                    .into(),
                )
            }
        }
    }

    if parts.len() < 2 {
        return Err(
            KboltError::InvalidInput("path locator must be '<collection>/<path>'".to_string())
                .into(),
        );
    }

    Ok((parts[0].clone(), parts[1..].join("/")))
}

fn normalize_docid(raw: &str) -> Result<String> {
    let normalized = raw.trim().trim_start_matches('#').to_string();
    if normalized.is_empty() {
        return Err(KboltError::InvalidInput("docid cannot be empty".to_string()).into());
    }
    Ok(normalized)
}

fn path_matches_prefix(path: &str, prefix: &str) -> bool {
    path == prefix || path.strip_prefix(prefix).is_some_and(|rest| rest.starts_with('/'))
}

fn short_docid(hash: &str) -> String {
    let short = hash.get(..6).unwrap_or(hash);
    format!("#{short}")
}

fn extension_allowed(path: &Path, filter: Option<&HashSet<String>>) -> bool {
    match filter {
        None => true,
        Some(allowed) => path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| allowed.contains(&ext.to_ascii_lowercase()))
            .unwrap_or(false),
    }
}

fn is_hard_ignored_dir_name(name: &std::ffi::OsStr) -> bool {
    matches!(name.to_str(), Some(".git") | Some("node_modules"))
}

fn is_hard_ignored_file(path: &Path) -> bool {
    if path.file_name().and_then(|name| name.to_str()) == Some(".DS_Store") {
        return true;
    }

    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("lock"))
}

fn load_collection_ignore_matcher(
    config_dir: &Path,
    collection_root: &Path,
    space: &str,
    collection: &str,
) -> Result<Option<Gitignore>> {
    let ignore_file = collection_ignore_file_path(config_dir, space, collection);
    if !ignore_file.is_file() {
        return Ok(None);
    }

    let mut builder = GitignoreBuilder::new(collection_root);
    if let Some(err) = builder.add(&ignore_file) {
        return Err(KboltError::InvalidInput(format!(
            "invalid ignore file '{}': {err}",
            ignore_file.display()
        ))
        .into());
    }

    let matcher = builder.build().map_err(|err| {
        KboltError::InvalidInput(format!(
            "invalid ignore file '{}': {err}",
            ignore_file.display()
        ))
    })?;
    Ok(Some(matcher))
}

fn collection_ignore_file_path(config_dir: &Path, space: &str, collection: &str) -> std::path::PathBuf {
    config_dir
        .join("ignores")
        .join(space)
        .join(format!("{collection}.ignore"))
}

fn validate_ignore_pattern(pattern: &str) -> Result<String> {
    if pattern.trim().is_empty() {
        return Err(
            KboltError::InvalidInput("ignore pattern cannot be empty".to_string()).into(),
        );
    }

    if pattern.contains('\n') || pattern.contains('\r') {
        return Err(
            KboltError::InvalidInput("ignore pattern must be a single line".to_string()).into(),
        );
    }

    Ok(pattern.to_string())
}

fn count_ignore_patterns(content: &str) -> usize {
    content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .count()
}

fn collection_relative_path(root: &Path, full_path: &Path) -> std::result::Result<String, KboltError> {
    let relative = full_path
        .strip_prefix(root)
        .map_err(|_| KboltError::InvalidPath(full_path.to_path_buf()))?;

    let parts = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(item) => Some(item.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();

    if parts.is_empty() {
        return Err(KboltError::InvalidPath(full_path.to_path_buf()));
    }

    Ok(parts.join("/"))
}

fn modified_token(metadata: &std::fs::Metadata) -> Result<String> {
    let modified = metadata.modified()?;
    let duration = modified.duration_since(UNIX_EPOCH).map_err(|_| {
        KboltError::Internal("file modified timestamp predates unix epoch".to_string())
    })?;
    Ok(duration.as_nanos().to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn file_title(path: &Path) -> String {
    path.file_name()
        .and_then(|item| item.to_str())
        .map(ToString::to_string)
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

fn file_error(path: Option<std::path::PathBuf>, error: String) -> FileError {
    FileError {
        path: path
            .map(|item| item.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<unknown>".to_string()),
        error,
    }
}

fn chunk_text_from_bytes(bytes: &[u8], offset: usize, length: usize) -> String {
    let start = offset.min(bytes.len());
    let end = offset.saturating_add(length).min(bytes.len());
    String::from_utf8_lossy(&bytes[start..end]).into_owned()
}

fn search_text_with_neighbors(
    bytes: &[u8],
    primary: &ChunkRow,
    doc_chunks: Option<&Vec<ChunkRow>>,
    neighbor_window: usize,
) -> String {
    if neighbor_window == 0 {
        return chunk_text_from_bytes(bytes, primary.offset, primary.length);
    }

    let Some(chunks) = doc_chunks else {
        return chunk_text_from_bytes(bytes, primary.offset, primary.length);
    };

    let window = neighbor_window.min(i32::MAX as usize) as i32;
    let min_seq = primary.seq.saturating_sub(window);
    let max_seq = primary.seq.saturating_add(window);
    let mut snippets = Vec::new();
    for chunk in chunks {
        if chunk.seq < min_seq || chunk.seq > max_seq {
            continue;
        }

        let snippet = chunk_text_from_bytes(bytes, chunk.offset, chunk.length);
        if !snippet.is_empty() {
            snippets.push(snippet);
        }
    }

    if snippets.is_empty() {
        chunk_text_from_bytes(bytes, primary.offset, primary.length)
    } else {
        snippets.join("\n\n")
    }
}

fn dense_distance_to_score(distance: f32) -> f32 {
    1.0 / (1.0 + distance.max(0.0))
}

pub(crate) fn retrieval_text_with_prefix(
    source_text: &str,
    title: &str,
    heading: Option<&str>,
    contextual_prefix: bool,
) -> String {
    if !contextual_prefix {
        return source_text.to_string();
    }

    let mut lines = Vec::new();
    let normalized_title = title.trim();
    if !normalized_title.is_empty() {
        lines.push(format!("title: {normalized_title}"));
    }

    if let Some(raw_heading) = heading {
        let normalized_heading = raw_heading.trim();
        if !normalized_heading.is_empty() {
            lines.push(format!("heading: {normalized_heading}"));
        }
    }

    if lines.is_empty() {
        source_text.to_string()
    } else {
        format!("{}\n\n{}", lines.join("\n"), source_text)
    }
}

#[cfg(test)]
mod tests;
