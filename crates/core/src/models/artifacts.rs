use std::path::{Path, PathBuf};

use kbolt_types::KboltError;
use walkdir::WalkDir;

use crate::Result;

pub(super) fn resolve_tokenizer_file(
    artifact_dir: &Path,
    configured: Option<&str>,
    config_field: &str,
) -> Result<PathBuf> {
    if let Some(configured) = configured {
        return resolve_configured_file(artifact_dir, configured, config_field);
    }

    let matches = discover_files(artifact_dir, |path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.eq_ignore_ascii_case("tokenizer.json"))
            .unwrap_or(false)
    })?;
    match matches.len() {
        0 => Err(KboltError::Inference(format!(
            "missing tokenizer.json under {}",
            artifact_dir.display()
        ))
        .into()),
        1 => Ok(matches[0].clone()),
        _ => Err(KboltError::Inference(format!(
            "multiple tokenizer.json files found under {}. set {config_field} to choose one",
            artifact_dir.display()
        ))
        .into()),
    }
}

pub(super) fn resolve_file_with_extension(
    artifact_dir: &Path,
    configured: Option<&str>,
    extension: &str,
    config_field: &str,
) -> Result<PathBuf> {
    if let Some(configured) = configured {
        return resolve_configured_file(artifact_dir, configured, config_field);
    }

    let matches = discover_files(artifact_dir, |path| {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case(extension))
            .unwrap_or(false)
    })?;
    match matches.len() {
        0 => Err(KboltError::Inference(format!(
            "missing .{extension} artifact under {}",
            artifact_dir.display()
        ))
        .into()),
        1 => Ok(matches[0].clone()),
        _ => Err(KboltError::Inference(format!(
            "multiple .{extension} artifacts found under {}. set {config_field} to choose one",
            artifact_dir.display()
        ))
        .into()),
    }
}

fn resolve_configured_file(artifact_dir: &Path, configured: &str, config_field: &str) -> Result<PathBuf> {
    let configured_path = Path::new(configured);
    let path = if configured_path.is_absolute() {
        configured_path.to_path_buf()
    } else {
        artifact_dir.join(configured_path)
    };
    if !path.is_file() {
        return Err(KboltError::Inference(format!(
            "{config_field} does not point to an existing file: {}",
            path.display()
        ))
        .into());
    }
    Ok(path)
}

fn discover_files<F>(root: &Path, mut predicate: F) -> Result<Vec<PathBuf>>
where
    F: FnMut(&Path) -> bool,
{
    let mut results = Vec::new();
    for entry in WalkDir::new(root).follow_links(true) {
        let entry = entry.map_err(|err| {
            KboltError::Inference(format!(
                "failed to walk model artifact directory {}: {err}",
                root.display()
            ))
        })?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if predicate(path) {
            results.push(path.to_path_buf());
        }
    }
    results.sort();
    Ok(results)
}
