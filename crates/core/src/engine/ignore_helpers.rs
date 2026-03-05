use std::path::Path;

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use kbolt_types::KboltError;

use crate::Result;

pub(super) fn is_hard_ignored_dir_name(name: &std::ffi::OsStr) -> bool {
    matches!(name.to_str(), Some(".git") | Some("node_modules"))
}

pub(super) fn is_hard_ignored_file(path: &Path) -> bool {
    if path.file_name().and_then(|name| name.to_str()) == Some(".DS_Store") {
        return true;
    }

    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("lock"))
}

pub(super) fn load_collection_ignore_matcher(
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

pub(super) fn collection_ignore_file_path(config_dir: &Path, space: &str, collection: &str) -> std::path::PathBuf {
    config_dir
        .join("ignores")
        .join(space)
        .join(format!("{collection}.ignore"))
}

pub(super) fn validate_ignore_pattern(pattern: &str) -> Result<String> {
    if pattern.trim().is_empty() {
        return Err(KboltError::InvalidInput("ignore pattern cannot be empty".to_string()).into());
    }

    if pattern.contains('\n') || pattern.contains('\r') {
        return Err(
            KboltError::InvalidInput("ignore pattern must be a single line".to_string()).into(),
        );
    }

    Ok(pattern.to_string())
}

pub(super) fn count_ignore_patterns(content: &str) -> usize {
    content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .count()
}
