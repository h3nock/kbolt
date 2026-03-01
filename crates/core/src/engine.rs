use std::collections::{HashMap, HashSet};
use std::path::{Component, Path};
use std::time::{Instant, UNIX_EPOCH};

use crate::config;
use crate::config::Config;
use crate::storage::{ChunkInsert, CollectionRow, DocumentRow, SpaceResolution, TantivyEntry};
use crate::storage::Storage;
use crate::Result;
use sha2::{Digest, Sha256};
use walkdir::WalkDir;
use kbolt_types::{
    ActiveSpace, ActiveSpaceSource, AddCollectionRequest, CollectionInfo, KboltError, SpaceInfo,
    FileError, UpdateOptions, UpdateReport,
};

pub struct Engine {
    storage: Storage,
    config: Config,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateTarget {
    pub space: String,
    pub collection: CollectionRow,
}

impl Engine {
    pub fn new(config_path: Option<&Path>) -> Result<Self> {
        let config = config::load(config_path)?;
        let storage = Storage::new(&config.cache_dir)?;
        Ok(Self { storage, config })
    }

    #[cfg(test)]
    pub(crate) fn from_parts(storage: Storage, config: Config) -> Self {
        Self { storage, config }
    }

    pub fn add_space(&self, name: &str, description: Option<&str>) -> Result<SpaceInfo> {
        self.storage.create_space(name, description)?;
        self.space_info(name)
    }

    pub fn remove_space(&self, name: &str) -> Result<()> {
        self.storage.delete_space(name)
    }

    pub fn rename_space(&self, old: &str, new: &str) -> Result<()> {
        self.storage.rename_space(old, new)
    }

    pub fn describe_space(&self, name: &str, description: &str) -> Result<()> {
        self.storage.update_space_description(name, description)
    }

    pub fn list_spaces(&self) -> Result<Vec<SpaceInfo>> {
        let spaces = self.storage.list_spaces()?;
        let mut infos = Vec::with_capacity(spaces.len());
        for space in spaces {
            infos.push(self.build_space_info(&space)?);
        }
        Ok(infos)
    }

    pub fn space_info(&self, name: &str) -> Result<SpaceInfo> {
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
        if !req.no_index {
            return Err(KboltError::Internal(
                "automatic indexing on collection add is not wired yet; use --no-index and run `kbolt update` manually for now".to_string(),
            )
            .into());
        }

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

        self.collection_info(Some(&space.name), &name)
    }

    pub fn remove_collection(&self, space: Option<&str>, name: &str) -> Result<()> {
        let resolved = self.resolve_space_row(space, Some(name))?;
        self.storage.delete_collection(resolved.id, name)
    }

    pub fn rename_collection(&self, space: Option<&str>, old: &str, new: &str) -> Result<()> {
        let resolved = self.resolve_space_row(space, Some(old))?;
        self.storage.rename_collection(resolved.id, old, new)
    }

    pub fn describe_collection(&self, space: Option<&str>, name: &str, desc: &str) -> Result<()> {
        let resolved = self.resolve_space_row(space, Some(name))?;
        self.storage.update_collection_description(resolved.id, name, desc)
    }

    pub fn list_collections(&self, space: Option<&str>) -> Result<Vec<CollectionInfo>> {
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
        let resolved = self.resolve_space_row(space, Some(name))?;
        let collection = self.storage.get_collection(resolved.id, name)?;
        self.build_collection_info(&resolved.name, &collection)
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

        let targets = self.resolve_update_targets(&options)?;
        if targets.is_empty() {
            report.elapsed_ms = started.elapsed().as_millis() as u64;
            return Ok(report);
        }

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

            let reaped = self.storage.reap_documents(self.config.reaping.days)?;
            report.reaped = reaped.len();
        }

        report.elapsed_ms = started.elapsed().as_millis() as u64;
        Ok(report)
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
        let mut touched_collection = false;

        for entry in WalkDir::new(&target.collection.path)
            .follow_links(false)
            .into_iter()
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
            let title = file_title(entry.path());

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

            let body = String::from_utf8_lossy(&bytes).into_owned();
            let chunk_ids = self.storage.insert_chunks(
                doc_id,
                &[ChunkInsert {
                    seq: 0,
                    offset: 0,
                    length: body.len(),
                    heading: None,
                    kind: "section".to_string(),
                }],
            )?;

            if let Some(chunk_id) = chunk_ids.first() {
                self.storage.index_tantivy(
                    &target.space,
                    &[TantivyEntry {
                        chunk_id: *chunk_id,
                        doc_id,
                        filepath: relative_path.clone(),
                        title,
                        heading: None,
                        body,
                    }],
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

#[cfg(test)]
mod tests;
