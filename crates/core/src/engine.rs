use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

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
use walkdir::WalkDir;
use kbolt_types::{
    ActiveSpace, ActiveSpaceSource, AddCollectionRequest, CollectionInfo, CollectionStatus,
    DocumentResponse, FileEntry, GetRequest, KboltError, Locator, ModelStatus,
    MultiGetRequest, MultiGetResponse, OmitReason, OmittedFile, PullReport, SearchMode,
    SearchRequest, SearchResponse, SearchResult, SearchSignals, SpaceInfo, SpaceStatus,
    StatusResponse, UpdateOptions, UpdateReport,
};

mod scoring;
mod text_helpers;
mod ignore_helpers;
mod ignore_ops;
mod path_utils;
mod file_utils;
use file_utils::{file_error, file_title, modified_token, sha256_hex};
use ignore_helpers::{
    collection_ignore_file_path, count_ignore_patterns, is_hard_ignored_dir_name, is_hard_ignored_file,
    load_collection_ignore_matcher, validate_ignore_pattern,
};
use path_utils::{
    collection_relative_path, extension_allowed, normalize_docid, normalize_list_prefix,
    normalized_extension_filter, path_matches_prefix, short_docid, split_collection_path,
};
use scoring::{dense_distance_to_score, max_option, normalize_scores};
pub(crate) use text_helpers::retrieval_text_with_prefix;
use text_helpers::{chunk_text_from_bytes, search_text_with_neighbors};

pub struct Engine {
    storage: Storage,
    config: Config,
    embedder: Option<Arc<dyn models::Embedder>>,
    reranker: Arc<dyn models::Reranker>,
    expander: Arc<dyn models::Expander>,
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
    rrf: f32,
    reranker: Option<f32>,
    bm25: Option<f32>,
    dense: Option<f32>,
}

#[derive(Debug, Clone)]
struct AssembledCandidate {
    docid: String,
    path: String,
    title: String,
    space: String,
    collection: String,
    heading: Option<String>,
    text: String,
    bm25: Option<f32>,
    dense: Option<f32>,
    rrf: f32,
    reranker: Option<f32>,
    final_score: f32,
    rerank_input: String,
}

impl Engine {
    pub fn new(config_path: Option<&Path>) -> Result<Self> {
        let config = config::load(config_path)?;
        let storage = Storage::new(&config.cache_dir)?;
        let model_dir = config.cache_dir.join("models");
        let embedder = models::build_embedder_with_local_runtime(
            config.embeddings.as_ref(),
            &config.models,
            &model_dir,
        )?;
        let reranker = models::build_reranker_with_local_runtime(
            config.inference.reranker.as_ref(),
            &config.models,
            &model_dir,
        )?;
        let expander = models::build_expander_with_local_runtime(
            config.inference.expander.as_ref(),
            &config.models,
            &model_dir,
        )?;
        Ok(Self {
            storage,
            config,
            embedder,
            reranker,
            expander,
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
        let model_dir = config.cache_dir.join("models");
        let reranker = models::build_reranker_with_local_runtime(
            config.inference.reranker.as_ref(),
            &config.models,
            &model_dir,
        )
        .expect("build reranker for test engine");
        let expander = models::build_expander_with_local_runtime(
            config.inference.expander.as_ref(),
            &config.models,
            &model_dir,
        )
        .expect("build expander for test engine");
        Self {
            storage,
            config,
            embedder,
            reranker,
            expander,
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
                    KboltError::DocumentNotFound { path } => {
                        warnings.push(format!("document not found: {path}"));
                        continue;
                    }
                    KboltError::InvalidInput(message) => {
                        warnings.push(format!("invalid locator: {message}"));
                        continue;
                    }
                    KboltError::InvalidPath(path) => {
                        warnings.push(format!("invalid locator path: {}", path.display()));
                        continue;
                    }
                    KboltError::AmbiguousSpace { collection, spaces } => {
                        warnings.push(format!(
                            "ambiguous locator space for collection '{collection}': {:?}. use --space to disambiguate.",
                            spaces
                        ));
                        continue;
                    }
                    KboltError::SpaceNotFound { name } => {
                        warnings.push(format!("space not found for locator: {name}"));
                        continue;
                    }
                    KboltError::CollectionNotFound { name } => {
                        warnings.push(format!("collection not found for locator: {name}"));
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
            SearchMode::Deep => SearchMode::Deep,
        };
        let rerank_enabled = matches!(mode, SearchMode::Auto | SearchMode::Deep) && !req.no_rerank;

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
            SearchMode::Auto => {
                let retrieval_limit = if rerank_enabled {
                    req.limit.max(20).saturating_mul(4)
                } else {
                    req.limit
                };
                self.rank_auto_chunks(&targets, query, retrieval_limit, req.min_score)?
            }
            SearchMode::Semantic => {
                self.rank_semantic_chunks(&targets, query, req.limit, req.min_score)?
            }
            SearchMode::Deep => {
                let retrieval_limit = if rerank_enabled {
                    req.limit.max(20).saturating_mul(4)
                } else {
                    req.limit.max(20)
                };
                self.rank_deep_chunks(&targets, query, retrieval_limit, req.min_score)?
            }
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

        let results = self.assemble_search_results(
            query,
            ranked_chunks,
            &collections_by_id,
            req.debug,
            rerank_enabled,
            req.limit,
        )?;

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
                rrf: normalized_score,
                reranker: None,
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
            return Err(KboltError::InvalidInput(
                "semantic search requires embeddings configuration. add [embeddings] to index.toml with provider = \"openai_compatible\", \"voyage\", or \"local_onnx\"".to_string(),
            )
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
                rrf: dense_score,
                reranker: None,
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
                rrf,
                reranker: None,
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
                item.rrf = item.score;
            }
        }

        Ok(fused
            .into_iter()
            .filter(|item| item.score >= min_score)
            .take(limit)
            .collect())
    }

    fn rank_deep_chunks(
        &self,
        targets: &[UpdateTarget],
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<RankedChunk>> {
        let variants = self.expander.expand(query)?;
        let mut aggregates: HashMap<i64, RankedChunk> = HashMap::new();

        for variant in variants {
            let ranked = self.rank_auto_chunks(targets, &variant, limit, 0.0)?;
            for (index, item) in ranked.into_iter().enumerate() {
                let variant_rrf = 1.0 / (40.0 + (index + 1) as f32);
                let entry = aggregates.entry(item.chunk_id).or_insert_with(|| RankedChunk {
                    chunk_id: item.chunk_id,
                    score: 0.0,
                    rrf: 0.0,
                    reranker: None,
                    bm25: None,
                    dense: None,
                });
                entry.score += variant_rrf;
                entry.bm25 = max_option(entry.bm25, item.bm25);
                entry.dense = max_option(entry.dense, item.dense);
            }
        }

        let mut fused = aggregates.into_values().collect::<Vec<_>>();
        fused.sort_by(|left, right| right.score.total_cmp(&left.score));

        let max_score = fused
            .iter()
            .map(|item| item.score)
            .fold(0.0_f32, f32::max);
        if max_score > 0.0 {
            for item in &mut fused {
                item.score /= max_score;
                item.rrf = item.score;
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
        query: &str,
        ranked_chunks: Vec<RankedChunk>,
        collections_by_id: &HashMap<i64, SearchCollectionMeta>,
        debug: bool,
        apply_rerank: bool,
        limit: usize,
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
        let mut candidates = Vec::new();
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
            let primary_text = chunk_text_from_bytes(&bytes, chunk.offset, chunk.length);
            let rerank_input = retrieval_text_with_prefix(
                if primary_text.trim().is_empty() {
                    text.as_str()
                } else {
                    primary_text.as_str()
                },
                document.title.as_str(),
                chunk.heading.as_deref(),
                self.config.chunking.defaults.contextual_prefix,
            );

            candidates.push(AssembledCandidate {
                docid: short_docid(&document.hash),
                path: format!("{}/{}", collection.collection, document.path),
                title: document.title,
                space: collection.space.clone(),
                collection: collection.collection.clone(),
                heading: chunk.heading.clone(),
                text,
                bm25: ranked.bm25,
                dense: ranked.dense,
                rrf: ranked.rrf,
                reranker: ranked.reranker,
                final_score: ranked.score,
                rerank_input,
            });
        }

        if apply_rerank && !candidates.is_empty() {
            let rerank_count = candidates.len().min(20);
            let rerank_inputs = candidates
                .iter()
                .take(rerank_count)
                .map(|candidate| candidate.rerank_input.clone())
                .collect::<Vec<_>>();
            let raw_scores = self.reranker.rerank(query, &rerank_inputs)?;
            if raw_scores.len() != rerank_inputs.len() {
                return Err(KboltError::Inference(format!(
                    "reranker returned {} scores for {} candidates",
                    raw_scores.len(),
                    rerank_inputs.len()
                ))
                .into());
            }
            let normalized_scores = normalize_scores(&raw_scores);
            for (candidate, reranker_score) in candidates
                .iter_mut()
                .take(rerank_count)
                .zip(normalized_scores.into_iter())
            {
                candidate.reranker = Some(reranker_score);
                candidate.final_score = 0.7 * reranker_score + 0.3 * candidate.rrf;
            }
        }

        candidates.sort_by(|left, right| right.final_score.total_cmp(&left.final_score));
        candidates.truncate(limit);

        let results = candidates
            .into_iter()
            .map(|candidate| SearchResult {
                docid: candidate.docid,
                path: candidate.path,
                title: candidate.title,
                space: candidate.space,
                collection: candidate.collection,
                heading: candidate.heading,
                text: candidate.text,
                score: candidate.final_score,
                signals: if debug {
                    Some(SearchSignals {
                        bm25: candidate.bm25,
                        dense: candidate.dense,
                        rrf: candidate.rrf,
                        reranker: candidate.reranker,
                    })
                } else {
                    None
                },
            })
            .collect::<Vec<_>>();

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

        let expected_model = self.embedding_model_key();
        let mut visited_spaces = HashSet::new();
        for target in targets {
            if !visited_spaces.insert(target.collection.space_id) {
                continue;
            }

            let models = self
                .storage
                .list_embedding_models_in_space(target.collection.space_id)?;
            if models.iter().any(|model| model != expected_model) {
                self.storage
                    .delete_embeddings_for_space(target.collection.space_id)?;
                self.storage.clear_usearch(&target.space)?;
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

        let model = self.embedding_model_key();
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

    fn embedding_model_key(&self) -> &str {
        self.config
            .embeddings
            .as_ref()
            .and_then(|config| match config {
                config::EmbeddingConfig::OpenAiCompatible { model, .. }
                | config::EmbeddingConfig::Voyage { model, .. } => Some(model.as_str()),
                config::EmbeddingConfig::LocalOnnx { .. } => None,
            })
            .unwrap_or(self.config.models.embedder.id.as_str())
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

#[cfg(test)]
mod tests;
