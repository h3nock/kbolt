use std::collections::HashSet;
use std::fs;
use std::path::Path;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use kbolt_types::KboltError;

use super::super::provider::{ModelArtifactProvider, ModelDownloadRequest, ModelFileRequirement};
use crate::Result;

pub(crate) struct HfHubDownloader;

impl ModelArtifactProvider for HfHubDownloader {
    fn download_model(&self, request: &ModelDownloadRequest, target_dir: &Path) -> Result<u64> {
        fs::create_dir_all(target_dir)?;
        let api = ApiBuilder::new()
            .with_cache_dir(target_dir.to_path_buf())
            .build()
            .map_err(|err| KboltError::ModelDownload(format!("{}: {err}", request.model_id)))?;
        let repo = api.repo(Repo::new(request.model_id.clone(), RepoType::Model));
        let info = repo
            .info()
            .map_err(|err| KboltError::ModelDownload(format!("{}: {err}", request.model_id)))?;
        let siblings = info
            .siblings
            .into_iter()
            .map(|sibling| sibling.rfilename)
            .collect::<Vec<_>>();
        let required_files =
            resolve_required_files(&request.model_id, &siblings, &request.requirements)?;
        if required_files.is_empty() {
            return Err(KboltError::ModelDownload(format!(
                "{}: no files selected for download",
                request.model_id
            ))
            .into());
        }

        let mut total_bytes = 0u64;
        for sibling in required_files {
            let file_path = repo
                .get(&sibling)
                .map_err(|err| KboltError::ModelDownload(format!("{}: {err}", request.model_id)))?;
            total_bytes = total_bytes.saturating_add(file_size_bytes(&file_path)?);
        }

        if total_bytes == 0 {
            return Err(KboltError::ModelDownload(format!(
                "{}: no files were downloaded",
                request.model_id
            ))
            .into());
        }

        Ok(total_bytes)
    }
}

fn resolve_required_files(
    model_id: &str,
    siblings: &[String],
    requirements: &[ModelFileRequirement],
) -> Result<Vec<String>> {
    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    for requirement in requirements {
        let resolved = resolve_requirement(model_id, siblings, requirement)?;
        if seen.insert(resolved.clone()) {
            selected.push(resolved);
        }
    }
    Ok(selected)
}

fn resolve_requirement(
    model_id: &str,
    siblings: &[String],
    requirement: &ModelFileRequirement,
) -> Result<String> {
    match requirement {
        ModelFileRequirement::ExactPath { path, config_field } => {
            let normalized = normalize_repo_path(path);
            if siblings.iter().any(|sibling| sibling == &normalized) {
                Ok(normalized)
            } else {
                Err(KboltError::ModelDownload(format!(
                    "{model_id}: configured file for {config_field} not found in repo: {path}"
                ))
                .into())
            }
        }
        ModelFileRequirement::SingleExtension {
            extension,
            config_field,
        } => {
            let matches = siblings
                .iter()
                .filter(|path| has_extension(path, extension))
                .cloned()
                .collect::<Vec<_>>();
            match matches.len() {
                0 => Err(KboltError::ModelDownload(format!(
                    "{model_id}: missing .{extension} artifact. set {config_field} to the desired file"
                ))
                .into()),
                1 => Ok(matches[0].clone()),
                _ => Err(KboltError::ModelDownload(format!(
                    "{model_id}: multiple .{extension} artifacts found. set {config_field} to choose one"
                ))
                .into()),
            }
        }
        ModelFileRequirement::SingleTokenizerJson { config_field } => {
            let matches = siblings
                .iter()
                .filter(|path| is_tokenizer_json(path))
                .cloned()
                .collect::<Vec<_>>();
            match matches.len() {
                0 => Err(KboltError::ModelDownload(format!(
                    "{model_id}: missing tokenizer.json. set {config_field} to the tokenizer path"
                ))
                .into()),
                1 => Ok(matches[0].clone()),
                _ => Err(KboltError::ModelDownload(format!(
                    "{model_id}: multiple tokenizer.json files found. set {config_field} to choose one"
                ))
                .into()),
            }
        }
    }
}

fn normalize_repo_path(path: &str) -> String {
    path.trim_start_matches("./").to_string()
}

fn has_extension(path: &str, extension: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(extension))
        .unwrap_or(false)
}

fn is_tokenizer_json(path: &str) -> bool {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("tokenizer.json"))
        .unwrap_or(false)
}

fn file_size_bytes(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)?.len())
}
