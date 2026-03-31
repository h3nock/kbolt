use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::config;
use crate::config::Config;
use crate::error::CoreError;
use crate::ingest::chunk::{chunk_document, resolve_policy};
use crate::ingest::extract::default_registry;
use crate::lock::{LockMode, OperationLock};
use crate::models;
use crate::storage::Storage;
use crate::storage::{
    ChunkInsert, ChunkRow, CollectionRow, DocumentRow, DocumentTitleSource, SpaceResolution,
    TantivyEntry,
};
use crate::ModelPullEvent;
use crate::Result;
use kbolt_types::{
    ActiveSpace, ActiveSpaceSource, AddCollectionRequest, AddCollectionResult, CollectionInfo,
    CollectionStatus, DocumentResponse, FileEntry, GetRequest, InitialIndexingBlock,
    InitialIndexingOutcome, KboltError, Locator, ModelStatus, MultiGetRequest, MultiGetResponse,
    OmitReason, OmittedFile, PullReport, SearchMode, SearchPipeline, SearchPipelineNotice,
    SearchPipelineStep, SearchPipelineUnavailableReason, SearchRequest, SearchResponse,
    SearchResult, SearchSignals, SpaceInfo, SpaceStatus, StatusResponse, UpdateOptions,
    UpdateReport,
};
use walkdir::WalkDir;

mod eval_ops;
mod file_utils;
mod ignore_helpers;
mod ignore_ops;
mod path_utils;
mod schedule_ops;
mod schedule_run_ops;
mod schedule_status_ops;
mod scoring;
mod search_ops;
mod text_helpers;
mod update_ops;
use file_utils::{file_error, file_title, modified_token, sha256_hex};
use ignore_helpers::{
    collection_ignore_file_path, count_ignore_patterns, is_hard_ignored_dir_name,
    is_hard_ignored_file, load_collection_ignore_matcher, validate_ignore_pattern,
};
use path_utils::{
    collection_relative_path, extension_allowed, normalize_docid, normalize_list_prefix,
    normalized_extension_filter, path_matches_prefix, short_docid, split_collection_path,
};
use scoring::{dense_distance_to_score, max_option};
pub(crate) use text_helpers::retrieval_text_with_prefix;
use text_helpers::{chunk_text_from_bytes, search_text_with_neighbors};

pub struct Engine {
    storage: Storage,
    config: Config,
    embedder: Option<Arc<dyn models::Embedder>>,
    reranker: Option<Arc<dyn models::Reranker>>,
    expander: Option<Arc<dyn models::Expander>>,
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

#[derive(Debug, Clone, Copy)]
struct TargetScope<'a> {
    space: Option<&'a str>,
    collections: &'a [String],
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
    fusion: f32,
    reranker: Option<f32>,
    bm25: Option<f32>,
    dense: Option<f32>,
    original_rank: Option<usize>,
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
        Self::from_parts_with_models(storage, config, None, None, None)
    }

    #[cfg(test)]
    pub(crate) fn from_parts_with_embedder(
        storage: Storage,
        config: Config,
        embedder: Option<Arc<dyn models::Embedder>>,
    ) -> Self {
        Self::from_parts_with_models(storage, config, embedder, None, None)
    }

    #[cfg(test)]
    pub(crate) fn from_parts_with_models(
        storage: Storage,
        config: Config,
        embedder: Option<Arc<dyn models::Embedder>>,
        reranker: Option<Arc<dyn models::Reranker>>,
        expander: Option<Arc<dyn models::Expander>>,
    ) -> Self {
        let model_dir = config.cache_dir.join("models");
        let reranker = reranker.or_else(|| {
            models::build_reranker_with_local_runtime(
                config.inference.reranker.as_ref(),
                &config.models,
                &model_dir,
            )
            .expect("build reranker for test engine")
        });
        let expander = expander.or_else(|| {
            models::build_expander_with_local_runtime(
                config.inference.expander.as_ref(),
                &config.models,
                &model_dir,
            )
            .expect("build expander for test engine")
        });
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

    pub fn add_collection(&self, req: AddCollectionRequest) -> Result<AddCollectionResult> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let AddCollectionRequest {
            path,
            space: requested_space,
            name: requested_name,
            description,
            extensions,
            no_index,
        } = req;

        let space = match requested_space.as_deref() {
            Some(space_name) => match self.storage.get_space(space_name) {
                Ok(space) => space,
                Err(CoreError::Domain(KboltError::SpaceNotFound { .. })) => {
                    self.storage.create_space(space_name, None)?;
                    self.storage.get_space(space_name)?
                }
                Err(err) => return Err(err),
            },
            None => self.resolve_space_row(None, None)?,
        };
        if !path.is_absolute() || !path.is_dir() {
            return Err(KboltError::InvalidPath(path).into());
        }

        let name = match requested_name {
            Some(name) => name,
            None => path
                .file_name()
                .and_then(|name| name.to_str())
                .map(ToString::to_string)
                .ok_or_else(|| KboltError::InvalidPath(path.clone()))?,
        };

        self.storage.create_collection(
            space.id,
            &name,
            &path,
            description.as_deref(),
            extensions.as_deref(),
        )?;

        let initial_indexing = if no_index {
            InitialIndexingOutcome::Skipped
        } else {
            match self.update_unlocked(UpdateOptions {
                space: Some(space.name.clone()),
                collections: vec![name.clone()],
                no_embed: false,
                dry_run: false,
                verbose: false,
            }) {
                Ok(report) => InitialIndexingOutcome::Indexed(report),
                Err(CoreError::Domain(KboltError::SpaceDenseRepairRequired { space, reason })) => {
                    InitialIndexingOutcome::Blocked(
                        InitialIndexingBlock::SpaceDenseRepairRequired { space, reason },
                    )
                }
                Err(CoreError::Domain(KboltError::ModelNotAvailable { name })) => {
                    InitialIndexingOutcome::Blocked(InitialIndexingBlock::ModelNotAvailable {
                        name,
                    })
                }
                Err(err) => return Err(err),
            }
        };

        let collection = self.storage.get_collection(space.id, &name)?;
        Ok(AddCollectionResult {
            collection: self.build_collection_info(&space.name, &collection)?,
            initial_indexing,
        })
    }

    pub fn remove_collection(&self, space: Option<&str>, name: &str) -> Result<()> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let resolved = self.resolve_space_row(space, Some(name))?;
        let collection = self.storage.get_collection(resolved.id, name)?;
        let documents = self.storage.list_documents(collection.id, false)?;
        let doc_ids = documents.into_iter().map(|doc| doc.id).collect::<Vec<_>>();
        let chunk_ids = self.collect_document_chunk_ids(&doc_ids)?;
        self.purge_space_chunks(&resolved.name, &chunk_ids)?;
        self.storage.delete_collection(resolved.id, name)?;

        let ignore_path =
            collection_ignore_file_path(&self.config.config_dir, &resolved.name, name);
        if ignore_path.is_file() {
            std::fs::remove_file(ignore_path)?;
        }

        Ok(())
    }

    pub fn rename_collection(&self, space: Option<&str>, old: &str, new: &str) -> Result<()> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        let resolved = self.resolve_space_row(space, Some(old))?;
        let old_ignore_path =
            collection_ignore_file_path(&self.config.config_dir, &resolved.name, old);
        let new_ignore_path =
            collection_ignore_file_path(&self.config.config_dir, &resolved.name, new);
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
        self.storage
            .update_collection_description(resolved.id, name, desc)
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
        let collection_row = self.storage.get_collection(resolved_space.id, collection)?;
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
                        let collection =
                            self.storage.get_collection_by_id(document.collection_id)?;
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
            return Err(
                KboltError::InvalidInput("max_files must be greater than 0".to_string()).into(),
            );
        }
        if req.max_bytes == 0 {
            return Err(
                KboltError::InvalidInput("max_bytes must be greater than 0".to_string()).into(),
            );
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

            let size_bytes = document.content.len();
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

        let requested_mode = req.mode.clone();
        let rerank_enabled =
            matches!(requested_mode, SearchMode::Auto | SearchMode::Deep) && !req.no_rerank;
        let mut pipeline = self.initial_search_pipeline(&requested_mode);

        let targets = self.resolve_targets(TargetScope {
            space: req.space.as_deref(),
            collections: &req.collections,
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
                requested_mode: requested_mode.clone(),
                effective_mode: self.effective_search_mode(&requested_mode, &pipeline),
                pipeline,
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

        let max_candidates = self.max_search_candidates(&targets)?;
        let mut retrieval_limit =
            self.initial_search_candidate_limit(&requested_mode, req.limit, rerank_enabled);
        let results = loop {
            let ranked_chunks = self.rank_chunks_for_mode(
                &requested_mode,
                &targets,
                query,
                retrieval_limit,
                req.min_score,
                &mut pipeline,
            )?;

            if ranked_chunks.is_empty() {
                return Ok(SearchResponse {
                    results: Vec::new(),
                    query: req.query,
                    requested_mode: requested_mode.clone(),
                    effective_mode: self.effective_search_mode(&requested_mode, &pipeline),
                    pipeline,
                    staleness_hint,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                });
            }

            let ranked_len = ranked_chunks.len();
            let results = self.assemble_search_results(
                query,
                &requested_mode,
                ranked_chunks,
                &collections_by_id,
                req.debug,
                rerank_enabled,
                &mut pipeline,
                req.limit,
            )?;

            if results.len() >= req.limit
                || ranked_len < retrieval_limit
                || retrieval_limit >= max_candidates
            {
                break results;
            }

            let next_limit = retrieval_limit.saturating_mul(2).min(max_candidates);
            if next_limit <= retrieval_limit {
                break results;
            }
            retrieval_limit = next_limit;
        };

        Ok(SearchResponse {
            results,
            query: req.query,
            requested_mode: requested_mode.clone(),
            effective_mode: self.effective_search_mode(&requested_mode, &pipeline),
            pipeline,
            staleness_hint,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
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
                let active_documents = self
                    .storage
                    .count_documents_in_collection(collection.id, true)?;
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
        models::status(
            &self.config.models,
            self.config.embeddings.as_ref(),
            &self.config.inference,
            &self.model_dir(),
        )
    }

    pub fn pull_models(&self) -> Result<PullReport> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        models::pull(
            &self.config.models,
            self.config.embeddings.as_ref(),
            &self.config.inference,
            &self.model_dir(),
        )
    }

    pub fn pull_models_with_progress<F>(&self, on_event: F) -> Result<PullReport>
    where
        F: FnMut(ModelPullEvent),
    {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        models::pull_with_progress(
            &self.config.models,
            self.config.embeddings.as_ref(),
            &self.config.inference,
            &self.model_dir(),
            on_event,
        )
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
                config::EmbeddingConfig::LocalOnnx { .. }
                | config::EmbeddingConfig::LocalGguf { .. } => None,
            })
            .unwrap_or(self.config.models.embedder.id.as_str())
    }

    fn model_dir(&self) -> std::path::PathBuf {
        self.config.cache_dir.join("models")
    }

    fn acquire_operation_lock(&self, mode: LockMode) -> Result<OperationLock> {
        OperationLock::acquire(&self.config.cache_dir, mode)
    }

    fn collect_document_chunk_ids(&self, doc_ids: &[i64]) -> Result<Vec<i64>> {
        let mut chunk_ids = Vec::new();
        for doc_id in doc_ids {
            let chunks = self.storage.get_chunks_for_document(*doc_id)?;
            chunk_ids.extend(chunks.into_iter().map(|chunk| chunk.id));
        }
        Ok(chunk_ids)
    }

    fn purge_space_chunks(&self, space: &str, chunk_ids: &[i64]) -> Result<()> {
        if chunk_ids.is_empty() {
            return Ok(());
        }

        self.storage.delete_tantivy(space, chunk_ids)?;
        self.storage.commit_tantivy(space)?;
        self.storage.delete_usearch(space, chunk_ids)?;
        Ok(())
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

    fn build_collection_info(
        &self,
        space_name: &str,
        collection: &CollectionRow,
    ) -> Result<CollectionInfo> {
        let document_count = self
            .storage
            .count_documents_in_collection(collection.id, false)?;
        let active_document_count = self
            .storage
            .count_documents_in_collection(collection.id, true)?;
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
}

#[cfg(test)]
mod tests;
