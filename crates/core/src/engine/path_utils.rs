use std::collections::HashSet;
use std::path::{Component, Path};

use kbolt_types::KboltError;

use crate::Result;

pub(super) fn normalized_extension_filter(raw: Option<&[String]>) -> Option<HashSet<String>> {
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

pub(super) fn normalize_list_prefix(prefix: Option<&str>) -> Result<Option<String>> {
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
                return Err(KboltError::InvalidInput(
                    "prefix must not traverse directories".to_string(),
                )
                .into())
            }
        }
    }

    if parts.is_empty() {
        return Ok(None);
    }

    Ok(Some(parts.join("/")))
}

pub(super) fn split_collection_path(locator: &str) -> Result<(String, String)> {
    let trimmed = locator.trim();
    if trimmed.is_empty() {
        return Err(KboltError::InvalidInput(
            "path locator must be '<collection>/<path>'".to_string(),
        )
        .into());
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
                return Err(KboltError::InvalidInput(
                    "path locator must not traverse directories".to_string(),
                )
                .into())
            }
        }
    }

    if parts.len() < 2 {
        return Err(KboltError::InvalidInput(
            "path locator must be '<collection>/<path>'".to_string(),
        )
        .into());
    }

    Ok((parts[0].clone(), parts[1..].join("/")))
}

pub(super) fn normalize_docid(raw: &str) -> Result<String> {
    let normalized = raw.trim().trim_start_matches('#').to_string();
    if normalized.is_empty() {
        return Err(KboltError::InvalidInput("docid cannot be empty".to_string()).into());
    }
    Ok(normalized)
}

pub(super) fn path_matches_prefix(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('/'))
}

pub(super) fn short_docid(hash: &str) -> String {
    let short = hash.get(..6).unwrap_or(hash);
    format!("#{short}")
}

pub(super) fn extension_allowed(path: &Path, filter: Option<&HashSet<String>>) -> bool {
    match filter {
        None => true,
        Some(allowed) => path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| allowed.contains(&ext.to_ascii_lowercase()))
            .unwrap_or(false),
    }
}

pub(super) fn collection_relative_path(
    root: &Path,
    full_path: &Path,
) -> std::result::Result<String, KboltError> {
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
