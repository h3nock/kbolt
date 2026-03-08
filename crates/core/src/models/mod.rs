use std::fs;
use std::path::Path;

use kbolt_types::KboltError;
use kbolt_types::{ModelInfo, ModelStatus, PullReport};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::config::{
    EmbeddingConfig, InferenceConfig, ModelConfig, ModelProvider, ModelSourceConfig,
    TextInferenceConfig, TextInferenceProvider,
};
use crate::Result;

mod artifacts;
mod chat;
mod completion;
mod embedder;
mod expander;
mod http;
mod inference;
mod local_gguf;
mod local_llama;
mod local_onnx;
mod provider;
mod providers;
mod reranker;
mod text;

const MODEL_DIRNAME_EMBEDDER: &str = "embedder";
const MODEL_DIRNAME_RERANKER: &str = "reranker";
const MODEL_DIRNAME_EXPANDER: &str = "expander";
const MODEL_MANIFEST_FILENAME: &str = ".kbolt-model-manifest.json";

pub(crate) use embedder::Embedder;
pub(crate) use expander::Expander;
#[cfg(test)]
pub(crate) use inference::build_embedder;
pub(crate) use inference::{
    build_embedder_with_local_runtime, build_expander_with_local_runtime,
    build_reranker_with_local_runtime,
};
pub(crate) use provider::{ModelArtifactProvider, ModelDownloadRequest, ModelFileRequirement};
use providers::hf::HfHubDownloader;
pub(crate) use reranker::Reranker;

#[derive(Debug, Clone)]
struct ModelTarget {
    role: ModelRole,
    role_dir: &'static str,
    source: ModelSourceConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelRole {
    Embedder,
    Reranker,
    Expander,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedModelArtifact {
    pub role: ModelRole,
    pub source: ModelSourceConfig,
    pub path: std::path::PathBuf,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ModelManifest {
    provider: ModelProvider,
    id: String,
    #[serde(default)]
    revision: Option<String>,
}

impl ModelManifest {
    fn from_source(source: &ModelSourceConfig) -> Self {
        Self {
            provider: source.provider.clone(),
            id: source.id.clone(),
            revision: source.revision.clone(),
        }
    }

    fn matches_source(&self, source: &ModelSourceConfig) -> bool {
        self.provider == source.provider && self.id == source.id && self.revision == source.revision
    }
}

impl ModelRole {
    fn role_dir(self) -> &'static str {
        match self {
            Self::Embedder => MODEL_DIRNAME_EMBEDDER,
            Self::Reranker => MODEL_DIRNAME_RERANKER,
            Self::Expander => MODEL_DIRNAME_EXPANDER,
        }
    }

    fn source(self, config: &ModelConfig) -> &ModelSourceConfig {
        match self {
            Self::Embedder => &config.embedder,
            Self::Reranker => &config.reranker,
            Self::Expander => &config.expander,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelPullEvent {
    DownloadStarted {
        role: String,
        model: String,
    },
    DownloadCompleted {
        role: String,
        model: String,
        bytes: u64,
    },
    AlreadyPresent {
        role: String,
        model: String,
        bytes: u64,
    },
}

pub fn pull(
    model_config: &ModelConfig,
    embeddings: Option<&EmbeddingConfig>,
    inference: &InferenceConfig,
    model_dir: &Path,
) -> Result<PullReport> {
    let downloader = HfHubDownloader;
    pull_with_downloader_and_progress(
        model_config,
        embeddings,
        inference,
        model_dir,
        &downloader,
        |_| {},
    )
}

#[cfg(test)]
pub(crate) fn pull_with_downloader(
    model_config: &ModelConfig,
    embeddings: Option<&EmbeddingConfig>,
    inference: &InferenceConfig,
    model_dir: &Path,
    downloader: &dyn ModelArtifactProvider,
) -> Result<PullReport> {
    pull_with_downloader_and_progress(
        model_config,
        embeddings,
        inference,
        model_dir,
        downloader,
        |_| {},
    )
}

pub fn pull_with_progress<F>(
    model_config: &ModelConfig,
    embeddings: Option<&EmbeddingConfig>,
    inference: &InferenceConfig,
    model_dir: &Path,
    on_event: F,
) -> Result<PullReport>
where
    F: FnMut(ModelPullEvent),
{
    let downloader = HfHubDownloader;
    pull_with_downloader_and_progress(
        model_config,
        embeddings,
        inference,
        model_dir,
        &downloader,
        on_event,
    )
}

pub(crate) fn pull_with_downloader_and_progress<F>(
    model_config: &ModelConfig,
    embeddings: Option<&EmbeddingConfig>,
    inference: &InferenceConfig,
    model_dir: &Path,
    downloader: &dyn ModelArtifactProvider,
    mut on_event: F,
) -> Result<PullReport>
where
    F: FnMut(ModelPullEvent),
{
    fs::create_dir_all(model_dir)?;
    let mut report = PullReport {
        downloaded: Vec::new(),
        already_present: Vec::new(),
        total_bytes: 0,
    };

    for target in model_targets(model_config) {
        let role = target.role_dir.to_string();
        let model = target.source.id.clone();
        let target_dir = model_dir.join(target.role_dir);
        let request = download_request_for_target(&target, embeddings, inference);
        if let Some(existing_bytes) = model_payload_size_bytes(&target_dir, &target.source)? {
            ensure_runtime_paths(&target_dir, &request.requirements)?;
            on_event(ModelPullEvent::AlreadyPresent {
                role,
                model: model.clone(),
                bytes: existing_bytes,
            });
            report.already_present.push(model);
            continue;
        }

        on_event(ModelPullEvent::DownloadStarted {
            role: role.clone(),
            model: model.clone(),
        });
        let downloaded_bytes = downloader.download_model(&request, &target_dir)?;
        ensure_runtime_paths(&target_dir, &request.requirements)?;
        on_event(ModelPullEvent::DownloadCompleted {
            role,
            model: model.clone(),
            bytes: downloaded_bytes,
        });
        write_model_manifest(&target_dir, &target.source)?;
        report.downloaded.push(model);
        report.total_bytes = report.total_bytes.saturating_add(downloaded_bytes);
    }

    Ok(report)
}

pub fn status(config: &ModelConfig, model_dir: &Path) -> Result<ModelStatus> {
    let targets = model_targets(config);
    let embedder = info_for_target(model_dir, &targets[0])?;
    let reranker = info_for_target(model_dir, &targets[1])?;
    let expander = info_for_target(model_dir, &targets[2])?;

    Ok(ModelStatus {
        embedder,
        reranker,
        expander,
    })
}

pub(crate) fn resolve_model_artifact(
    config: &ModelConfig,
    model_dir: &Path,
    role: ModelRole,
) -> Result<ResolvedModelArtifact> {
    let source = role.source(config);
    let path = model_dir.join(role.role_dir());
    let Some(size_bytes) = model_payload_size_bytes(&path, source)? else {
        return Err(KboltError::ModelNotAvailable {
            name: source.id.clone(),
        }
        .into());
    };

    Ok(ResolvedModelArtifact {
        role,
        source: source.clone(),
        path,
        size_bytes,
    })
}

fn model_targets(config: &ModelConfig) -> [ModelTarget; 3] {
    [
        ModelTarget {
            role: ModelRole::Embedder,
            role_dir: MODEL_DIRNAME_EMBEDDER,
            source: config.embedder.clone(),
        },
        ModelTarget {
            role: ModelRole::Reranker,
            role_dir: MODEL_DIRNAME_RERANKER,
            source: config.reranker.clone(),
        },
        ModelTarget {
            role: ModelRole::Expander,
            role_dir: MODEL_DIRNAME_EXPANDER,
            source: config.expander.clone(),
        },
    ]
}

fn download_request_for_target(
    target: &ModelTarget,
    embeddings: Option<&EmbeddingConfig>,
    inference: &InferenceConfig,
) -> ModelDownloadRequest {
    let requirements = match target.role {
        ModelRole::Embedder => embedder_download_requirements(embeddings),
        ModelRole::Reranker => text_role_download_requirements(
            inference.reranker.as_ref(),
            "inference.reranker.model_file",
        ),
        ModelRole::Expander => text_role_download_requirements(
            inference.expander.as_ref(),
            "inference.expander.model_file",
        ),
    };

    ModelDownloadRequest {
        model_id: target.source.id.clone(),
        requirements,
    }
}

fn embedder_download_requirements(
    embeddings: Option<&EmbeddingConfig>,
) -> Vec<ModelFileRequirement> {
    match embeddings {
        Some(EmbeddingConfig::LocalOnnx {
            onnx_file,
            tokenizer_file,
            ..
        }) => {
            let onnx_override = configured_repo_path(onnx_file.as_deref());
            let tokenizer_override = configured_repo_path(tokenizer_file.as_deref());
            vec![
                onnx_override
                    .map(|path| ModelFileRequirement::ExactPath {
                        path,
                        config_field: "embeddings.onnx_file",
                    })
                    .unwrap_or(ModelFileRequirement::SingleExtension {
                        extension: "onnx",
                        config_field: "embeddings.onnx_file",
                    }),
                tokenizer_override
                    .map(|path| ModelFileRequirement::ExactPath {
                        path,
                        config_field: "embeddings.tokenizer_file",
                    })
                    .unwrap_or(ModelFileRequirement::SingleTokenizerJson {
                        config_field: "embeddings.tokenizer_file",
                    }),
            ]
        }
        Some(EmbeddingConfig::LocalGguf { model_file, .. }) => {
            let configured = configured_repo_path(model_file.as_deref()).map(|path| {
                ModelFileRequirement::ExactPath {
                    path,
                    config_field: "embeddings.model_file",
                }
            });
            vec![configured.unwrap_or(ModelFileRequirement::SingleExtension {
                extension: "gguf",
                config_field: "embeddings.model_file",
            })]
        }
        Some(EmbeddingConfig::OpenAiCompatible { .. })
        | Some(EmbeddingConfig::Voyage { .. })
        | None => {
            vec![ModelFileRequirement::SingleExtension {
                extension: "gguf",
                config_field: "embeddings.model_file",
            }]
        }
    }
}

fn text_role_download_requirements(
    config: Option<&TextInferenceConfig>,
    config_field: &'static str,
) -> Vec<ModelFileRequirement> {
    let configured = config
        .and_then(|config| match &config.provider {
            TextInferenceProvider::LocalLlama { model_file, .. } => {
                configured_repo_path(model_file.as_deref())
            }
            TextInferenceProvider::OpenAiCompatible { .. } => None,
        })
        .map(|path| ModelFileRequirement::ExactPath { path, config_field });

    vec![configured.unwrap_or(ModelFileRequirement::SingleExtension {
        extension: "gguf",
        config_field,
    })]
}

fn configured_repo_path(configured: Option<&str>) -> Option<String> {
    let configured = configured?.trim();
    if configured.is_empty() || Path::new(configured).is_absolute() {
        return None;
    }
    Some(configured.trim_start_matches("./").to_string())
}

fn info_for_target(model_dir: &Path, target: &ModelTarget) -> Result<ModelInfo> {
    let target_dir = model_dir.join(target.role_dir);
    let size = model_payload_size_bytes(&target_dir, &target.source)?.unwrap_or(0);
    Ok(ModelInfo {
        name: target.source.id.clone(),
        downloaded: size > 0,
        size_bytes: if size > 0 { Some(size) } else { None },
        path: Some(target_dir),
    })
}

fn model_payload_size_bytes(path: &Path, source: &ModelSourceConfig) -> Result<Option<u64>> {
    let manifest = read_model_manifest(path)?;
    let Some(manifest) = manifest else {
        return Ok(None);
    };

    if !manifest.matches_source(source) {
        return Ok(None);
    }

    let size = dir_size_bytes(path, true)?;
    if size == 0 {
        return Ok(None);
    }

    Ok(Some(size))
}

fn write_model_manifest(path: &Path, source: &ModelSourceConfig) -> Result<()> {
    fs::create_dir_all(path)?;
    let manifest = ModelManifest::from_source(source);
    let serialized = serde_json::to_vec_pretty(&manifest)?;
    let manifest_path = path.join(MODEL_MANIFEST_FILENAME);
    let tmp_path = path.join(format!("{MODEL_MANIFEST_FILENAME}.tmp"));
    fs::write(&tmp_path, serialized)?;
    fs::rename(tmp_path, manifest_path)?;
    Ok(())
}

fn read_model_manifest(path: &Path) -> Result<Option<ModelManifest>> {
    let manifest_path = path.join(MODEL_MANIFEST_FILENAME);
    let bytes = match fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };

    let parsed = match serde_json::from_slice::<ModelManifest>(&bytes) {
        Ok(parsed) => parsed,
        Err(_) => return Ok(None),
    };

    Ok(Some(parsed))
}

fn ensure_runtime_paths(target_dir: &Path, requirements: &[ModelFileRequirement]) -> Result<()> {
    for requirement in requirements {
        match requirement {
            ModelFileRequirement::ExactPath { path, config_field } => {
                ensure_runtime_path(target_dir, path, config_field)?;
            }
            ModelFileRequirement::SingleExtension {
                extension,
                config_field,
            } => ensure_single_extension_runtime_path(target_dir, extension, config_field)?,
            ModelFileRequirement::SingleTokenizerJson { config_field } => {
                ensure_tokenizer_runtime_path(target_dir, config_field)?
            }
        }
    }
    cleanup_provider_cache_dirs(target_dir)?;
    Ok(())
}

fn ensure_runtime_path(target_dir: &Path, configured_path: &str, config_field: &str) -> Result<()> {
    let runtime_path = target_dir.join(configured_path);
    match fs::symlink_metadata(&runtime_path) {
        Ok(metadata) => {
            if metadata.file_type().is_file() {
                return Ok(());
            }
            if metadata.file_type().is_dir() {
                return Err(KboltError::Inference(format!(
                    "{config_field} path is a directory, expected a file: {}",
                    runtime_path.display()
                ))
                .into());
            }
            fs::remove_file(&runtime_path)?;
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }

    let source = locate_cached_file(target_dir, configured_path, config_field)?;
    if let Some(parent) = runtime_path.parent() {
        fs::create_dir_all(parent)?;
    }
    materialize_runtime_file(&source, &runtime_path).map_err(|err| {
        KboltError::Inference(format!(
            "failed to materialize {config_field} at {} from {}: {err}",
            runtime_path.display(),
            source.display()
        ))
    })?;
    Ok(())
}

fn ensure_single_extension_runtime_path(
    target_dir: &Path,
    extension: &str,
    config_field: &str,
) -> Result<()> {
    let materialized =
        discover_materialized_files(target_dir, |path| has_extension(path, extension))?;
    match materialized.len() {
        1 => return Ok(()),
        0 => {}
        _ => {
            return Err(KboltError::Inference(format!(
                "multiple .{extension} artifacts found under {}. set {config_field} to choose one",
                target_dir.display()
            ))
            .into())
        }
    }

    let source = locate_single_cached_file(
        target_dir,
        |path| has_extension(path, extension),
        &format!(".{extension} artifact"),
        config_field,
    )?;
    let file_name = source
        .file_name()
        .ok_or_else(|| {
            KboltError::Inference(format!(
                "downloaded .{extension} artifact is missing a file name: {}",
                source.display()
            ))
        })?
        .to_owned();
    let runtime_path = target_dir.join(file_name);
    materialize_runtime_file(&source, &runtime_path).map_err(|err| {
        KboltError::Inference(format!(
            "failed to materialize {config_field} at {} from {}: {err}",
            runtime_path.display(),
            source.display()
        ))
    })?;
    Ok(())
}

fn ensure_tokenizer_runtime_path(target_dir: &Path, config_field: &str) -> Result<()> {
    let materialized = discover_materialized_files(target_dir, is_tokenizer_json)?;
    match materialized.len() {
        1 => return Ok(()),
        0 => {}
        _ => {
            return Err(KboltError::Inference(format!(
                "multiple tokenizer.json files found under {}. set {config_field} to choose one",
                target_dir.display()
            ))
            .into())
        }
    }

    let source = locate_single_cached_file(
        target_dir,
        is_tokenizer_json,
        "tokenizer.json",
        config_field,
    )?;
    let runtime_path = target_dir.join("tokenizer.json");
    materialize_runtime_file(&source, &runtime_path).map_err(|err| {
        KboltError::Inference(format!(
            "failed to materialize {config_field} at {} from {}: {err}",
            runtime_path.display(),
            source.display()
        ))
    })?;
    Ok(())
}

fn locate_cached_file(
    target_dir: &Path,
    configured_path: &str,
    config_field: &str,
) -> Result<std::path::PathBuf> {
    let repo_path = Path::new(configured_path);
    if let Some(current) = locate_current_snapshot_file(target_dir, repo_path)? {
        return Ok(current);
    }

    let matches = discover_cached_files(target_dir, repo_path)?;
    match matches.len() {
        0 => Err(KboltError::Inference(format!(
            "{config_field} does not point to a downloaded file under {}: {}",
            target_dir.display(),
            configured_path
        ))
        .into()),
        1 => Ok(matches[0].clone()),
        _ => Err(KboltError::Inference(format!(
            "{config_field} matches multiple downloaded files under {}: {}",
            target_dir.display(),
            configured_path
        ))
        .into()),
    }
}

fn locate_current_snapshot_file(
    target_dir: &Path,
    repo_path: &Path,
) -> Result<Option<std::path::PathBuf>> {
    let mut matches = Vec::new();
    for entry in fs::read_dir(target_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let repo_dir = entry.path();
        if !is_provider_cache_dir(&repo_dir) {
            continue;
        }
        let refs_main = repo_dir.join("refs").join("main");
        let snapshot_id = match fs::read_to_string(&refs_main) {
            Ok(contents) => contents.trim().to_string(),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err.into()),
        };
        if snapshot_id.is_empty() {
            continue;
        }
        let candidate = repo_dir.join("snapshots").join(snapshot_id).join(repo_path);
        if candidate.is_file() {
            matches.push(candidate);
        }
    }

    match matches.len() {
        0 => Ok(None),
        1 => Ok(matches.into_iter().next()),
        _ => Err(KboltError::Inference(format!(
            "multiple snapshot files found for {} under {}",
            repo_path.display(),
            target_dir.display()
        ))
        .into()),
    }
}

fn discover_cached_files(target_dir: &Path, repo_path: &Path) -> Result<Vec<std::path::PathBuf>> {
    discover_provider_cache_files(target_dir, |relative| path_ends_with(relative, repo_path))
}

fn locate_single_cached_file<F>(
    target_dir: &Path,
    predicate: F,
    label: &str,
    config_field: &str,
) -> Result<std::path::PathBuf>
where
    F: FnMut(&Path) -> bool,
{
    let matches = discover_provider_cache_files(target_dir, predicate)?;
    match matches.len() {
        0 => Err(KboltError::Inference(format!(
            "missing {label} under {}. set {config_field} to choose one",
            target_dir.display()
        ))
        .into()),
        1 => Ok(matches[0].clone()),
        _ => Err(KboltError::Inference(format!(
            "multiple {label} files found under {}. set {config_field} to choose one",
            target_dir.display()
        ))
        .into()),
    }
}

fn discover_provider_cache_files<F>(
    target_dir: &Path,
    mut predicate: F,
) -> Result<Vec<std::path::PathBuf>>
where
    F: FnMut(&Path) -> bool,
{
    let mut matches = Vec::new();
    for entry in fs::read_dir(target_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let repo_dir = entry.path();
        if !is_provider_cache_dir(&repo_dir) {
            continue;
        }
        for entry in WalkDir::new(&repo_dir).follow_links(false) {
            let entry = entry.map_err(|err| {
                KboltError::Inference(format!(
                    "failed to walk downloaded model artifacts under {}: {err}",
                    repo_dir.display()
                ))
            })?;
            if !entry.file_type().is_file() && !entry.file_type().is_symlink() {
                continue;
            }
            let path = entry.path();
            let Ok(relative) = path.strip_prefix(&repo_dir) else {
                continue;
            };
            if predicate(relative) {
                matches.push(path.to_path_buf());
            }
        }
    }
    matches.sort();
    Ok(matches)
}

fn discover_materialized_files<F>(
    target_dir: &Path,
    mut predicate: F,
) -> Result<Vec<std::path::PathBuf>>
where
    F: FnMut(&Path) -> bool,
{
    let mut matches = Vec::new();
    for entry in WalkDir::new(target_dir)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| !is_provider_cache_dir(entry.path()))
    {
        let entry = entry.map_err(|err| {
            KboltError::Inference(format!(
                "failed to walk model runtime directory {}: {err}",
                target_dir.display()
            ))
        })?;
        if !entry.file_type().is_file() {
            continue;
        }
        if predicate(entry.path()) {
            matches.push(entry.path().to_path_buf());
        }
    }
    matches.sort();
    Ok(matches)
}

fn path_ends_with(path: &Path, suffix: &Path) -> bool {
    let path_components = path.components().collect::<Vec<_>>();
    let suffix_components = suffix.components().collect::<Vec<_>>();
    if suffix_components.len() > path_components.len() {
        return false;
    }
    path_components[path_components.len() - suffix_components.len()..] == suffix_components[..]
}

fn cleanup_provider_cache_dirs(target_dir: &Path) -> Result<()> {
    for entry in fs::read_dir(target_dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_dir() {
            continue;
        }
        if is_provider_cache_dir(&entry.path()) {
            fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

fn is_provider_cache_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("models--"))
        .unwrap_or(false)
}

fn dir_size_bytes(path: &Path, skip_manifest: bool) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }

    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        let metadata = fs::symlink_metadata(&entry_path)?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_file() {
            if skip_manifest
                && entry_path.file_name().and_then(|name| name.to_str())
                    == Some(MODEL_MANIFEST_FILENAME)
            {
                continue;
            }
            total = total.saturating_add(metadata.len());
            continue;
        }
        if file_type.is_dir() {
            total = total.saturating_add(dir_size_bytes(&entry_path, skip_manifest)?);
        }
    }
    Ok(total)
}

fn materialize_runtime_file(source: &Path, target: &Path) -> std::io::Result<()> {
    let source = fs::canonicalize(source)?;
    match fs::hard_link(&source, target) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(&source, target)?;
            Ok(())
        }
    }
}

fn has_extension(path: &Path, extension: &str) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(extension))
        .unwrap_or(false)
}

fn is_tokenizer_json(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("tokenizer.json"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests;
