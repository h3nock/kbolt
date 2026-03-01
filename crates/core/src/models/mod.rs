use std::fs;
use std::path::Path;

use kbolt_types::{ModelInfo, ModelStatus, PullReport};

use crate::config::{ModelConfig, ModelSourceConfig};
use crate::Result;

mod provider;
mod providers;

const MODEL_DIRNAME_EMBEDDER: &str = "embedder";
const MODEL_DIRNAME_RERANKER: &str = "reranker";
const MODEL_DIRNAME_EXPANDER: &str = "expander";

pub(crate) use provider::ModelArtifactProvider;
use providers::hf::HfHubDownloader;

#[derive(Debug, Clone)]
struct ModelTarget {
    role_dir: &'static str,
    source: ModelSourceConfig,
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

pub fn pull_with_progress<F>(config: &ModelConfig, model_dir: &Path, on_event: F) -> Result<PullReport>
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
        let existing_bytes = dir_size_bytes(&target_dir).unwrap_or(0);
        if existing_bytes > 0 {
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
    let size = dir_size_bytes(&target_dir)?;
    Ok(ModelInfo {
        name: target.source.id.clone(),
        downloaded: size > 0,
        size_bytes: if size > 0 { Some(size) } else { None },
        path: Some(target_dir),
    })
}

fn dir_size_bytes(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }

    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            total = total.saturating_add(metadata.len());
            continue;
        }
        if metadata.is_dir() {
            total = total.saturating_add(dir_size_bytes(&entry_path)?);
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests;
