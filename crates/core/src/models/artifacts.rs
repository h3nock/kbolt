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

fn resolve_configured_file(
    artifact_dir: &Path,
    configured: &str,
    config_field: &str,
) -> Result<PathBuf> {
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

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{resolve_file_with_extension, resolve_tokenizer_file};

    fn write_file(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent directories");
        }
        fs::write(path, b"fixture").expect("write fixture file");
    }

    use std::path::Path;

    #[test]
    fn resolve_file_with_extension_returns_single_match() {
        let root = tempdir().expect("create tempdir");
        let model = root.path().join("model.onnx");
        write_file(&model);

        let resolved =
            resolve_file_with_extension(root.path(), None, "onnx", "embeddings.onnx_file")
                .expect("resolve .onnx file");
        assert_eq!(resolved, model);
    }

    #[test]
    fn resolve_file_with_extension_errors_on_ambiguous_match() {
        let root = tempdir().expect("create tempdir");
        write_file(&root.path().join("a/model.onnx"));
        write_file(&root.path().join("b/model.onnx"));

        let err = resolve_file_with_extension(root.path(), None, "onnx", "embeddings.onnx_file")
            .expect_err("ambiguous .onnx files should error");
        assert!(err.to_string().contains("multiple .onnx artifacts found"));
    }

    #[test]
    fn resolve_file_with_extension_uses_configured_override() {
        let root = tempdir().expect("create tempdir");
        write_file(&root.path().join("a/model.onnx"));
        let preferred = root.path().join("b/preferred.onnx");
        write_file(&preferred);

        let resolved = resolve_file_with_extension(
            root.path(),
            Some("b/preferred.onnx"),
            "onnx",
            "embeddings.onnx_file",
        )
        .expect("resolve configured onnx file");
        assert_eq!(resolved, preferred);
    }

    #[test]
    fn resolve_tokenizer_file_errors_when_missing() {
        let root = tempdir().expect("create tempdir");
        let err = resolve_tokenizer_file(root.path(), None, "embeddings.tokenizer_file")
            .expect_err("missing tokenizer should error");
        assert!(err.to_string().contains("missing tokenizer.json"));
    }
}
