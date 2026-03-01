use std::fs;
use std::path::Path;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use kbolt_types::KboltError;

use super::super::provider::ModelArtifactProvider;
use crate::Result;

pub(crate) struct HfHubDownloader;

impl ModelArtifactProvider for HfHubDownloader {
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

fn file_size_bytes(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)?.len())
}
