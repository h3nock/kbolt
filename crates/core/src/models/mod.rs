use std::fs;
use std::path::Path;

use kbolt_types::KboltError;
use kbolt_types::{ModelInfo, ModelStatus, PullReport};
use serde::{Deserialize, Serialize};

use crate::config::{ModelConfig, ModelProvider, ModelSourceConfig};
use crate::Result;

mod artifacts;
mod chat;
mod completion;
mod embedder;
mod expander;
mod http;
mod inference;
mod local_llama;
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
pub(crate) use provider::ModelArtifactProvider;
use providers::hf::HfHubDownloader;
pub(crate) use reranker::Reranker;

#[derive(Debug, Clone)]
struct ModelTarget {
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

pub fn pull(config: &ModelConfig, model_dir: &Path) -> Result<PullReport> {
    let downloader = HfHubDownloader;
    pull_with_downloader_and_progress(config, model_dir, &downloader, |_| {})
}

#[cfg(test)]
pub(crate) fn pull_with_downloader(
    config: &ModelConfig,
    model_dir: &Path,
    downloader: &dyn ModelArtifactProvider,
) -> Result<PullReport> {
    pull_with_downloader_and_progress(config, model_dir, downloader, |_| {})
}

pub fn pull_with_progress<F>(
    config: &ModelConfig,
    model_dir: &Path,
    on_event: F,
) -> Result<PullReport>
where
    F: FnMut(ModelPullEvent),
{
    let downloader = HfHubDownloader;
    pull_with_downloader_and_progress(config, model_dir, &downloader, on_event)
}

pub(crate) fn pull_with_downloader_and_progress<F>(
    config: &ModelConfig,
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

    for target in model_targets(config) {
        let role = target.role_dir.to_string();
        let model = target.source.id.clone();
        let target_dir = model_dir.join(target.role_dir);
        if let Some(existing_bytes) = model_payload_size_bytes(&target_dir, &target.source)? {
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
        let downloaded_bytes = downloader.download_model(&target.source.id, &target_dir)?;
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
            role_dir: MODEL_DIRNAME_EMBEDDER,
            source: config.embedder.clone(),
        },
        ModelTarget {
            role_dir: MODEL_DIRNAME_RERANKER,
            source: config.reranker.clone(),
        },
        ModelTarget {
            role_dir: MODEL_DIRNAME_EXPANDER,
            source: config.expander.clone(),
        },
    ]
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

fn dir_size_bytes(path: &Path, skip_manifest: bool) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }

    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            if skip_manifest
                && entry_path.file_name().and_then(|name| name.to_str())
                    == Some(MODEL_MANIFEST_FILENAME)
            {
                continue;
            }
            total = total.saturating_add(metadata.len());
            continue;
        }
        if metadata.is_dir() {
            total = total.saturating_add(dir_size_bytes(&entry_path, skip_manifest)?);
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests;
