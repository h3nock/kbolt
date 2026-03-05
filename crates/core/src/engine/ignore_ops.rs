use super::{
    collection_ignore_file_path, count_ignore_patterns, validate_ignore_pattern, Engine,
    IgnoreListEntry,
};
use crate::lock::LockMode;
use crate::Result;
use kbolt_types::KboltError;

impl Engine {
    pub fn read_collection_ignore(
        &self,
        space: Option<&str>,
        collection: &str,
    ) -> Result<(String, Option<String>)> {
        let _lock = self.acquire_operation_lock(LockMode::Shared)?;
        let resolved_space = self.resolve_space_row(space, Some(collection))?;
        self.storage.get_collection(resolved_space.id, collection)?;

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
        self.storage.get_collection(resolved_space.id, collection)?;

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
        self.storage.get_collection(resolved_space.id, collection)?;

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
}
