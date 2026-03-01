use std::fs;
use std::path::Path;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use kbolt_types::{KboltError, ModelInfo, ModelStatus, PullReport};

use crate::config::ModelConfig;
use crate::Result;

const MODEL_DIRNAME_EMBEDDER: &str = "embedder";
const MODEL_DIRNAME_RERANKER: &str = "reranker";
const MODEL_DIRNAME_EXPANDER: &str = "expander";

#[derive(Debug, Clone)]
struct ModelTarget {
    role_dir: &'static str,
    model_id: String,
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

pub trait ModelDownloader {
    fn download_model(&self, model_id: &str, target_dir: &Path) -> Result<u64>;
}

pub struct HfHubDownloader;

impl ModelDownloader for HfHubDownloader {
    fn download_model(&self, model_id: &str, target_dir: &Path) -> Result<u64> {
        fs::create_dir_all(target_dir)?;
        let api = ApiBuilder::new()
            .with_cache_dir(target_dir.to_path_buf())
            .build()
            .map_err(|err| KboltError::ModelDownload(format!("{model_id}: {err}")))?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
        let info = repo
            .info()
            .map_err(|err| KboltError::ModelDownload(format!("{model_id}: {err}")))?;

        let mut total_bytes = 0u64;
        for sibling in info.siblings {
            let file_path = repo
                .get(&sibling.rfilename)
                .map_err(|err| KboltError::ModelDownload(format!("{model_id}: {err}")))?;
            total_bytes = total_bytes.saturating_add(file_size_bytes(&file_path)?);
        }

        if total_bytes == 0 {
            return Err(
                KboltError::ModelDownload(format!("{model_id}: no files were downloaded")).into(),
            );
        }

        Ok(total_bytes)
    }
}

pub fn pull(config: &ModelConfig, model_dir: &Path) -> Result<PullReport> {
    let downloader = HfHubDownloader;
    pull_with_downloader_and_progress(config, model_dir, &downloader, |_| {})
}

#[cfg(test)]
pub(crate) fn pull_with_downloader(
    config: &ModelConfig,
    model_dir: &Path,
    downloader: &dyn ModelDownloader,
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
    downloader: &dyn ModelDownloader,
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
        let model = target.model_id.clone();
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
        let downloaded_bytes = downloader.download_model(&target.model_id, &target_dir)?;
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
            model_id: config.embed.clone(),
        },
        ModelTarget {
            role_dir: MODEL_DIRNAME_RERANKER,
            model_id: config.reranker.clone(),
        },
        ModelTarget {
            role_dir: MODEL_DIRNAME_EXPANDER,
            model_id: config.expander.clone(),
        },
    ]
}

fn info_for_target(model_dir: &Path, target: &ModelTarget) -> Result<ModelInfo> {
    let target_dir = model_dir.join(target.role_dir);
    let size = dir_size_bytes(&target_dir)?;
    Ok(ModelInfo {
        name: target.model_id.clone(),
        downloaded: size > 0,
        size_bytes: if size > 0 { Some(size) } else { None },
        path: Some(target_dir),
    })
}

fn file_size_bytes(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)?.len())
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
