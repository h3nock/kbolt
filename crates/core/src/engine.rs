use std::path::Path;

use crate::config;
use crate::config::Config;
use crate::storage::{CollectionRow, SpaceResolution};
use crate::storage::Storage;
use crate::Result;
use kbolt_types::{
    ActiveSpace, ActiveSpaceSource, AddCollectionRequest, CollectionInfo, KboltError, SpaceInfo,
    UpdateOptions,
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
}

#[cfg(test)]
mod tests;
