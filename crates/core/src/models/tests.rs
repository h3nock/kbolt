use std::path::Path;

use tempfile::tempdir;

use crate::config::ModelConfig;
use crate::models::{
    pull_with_downloader, pull_with_downloader_and_progress, status, ModelArtifactProvider,
    ModelPullEvent,
};
use crate::Result;

#[derive(Default)]
struct FakeDownloader {
    bytes_per_model: u64,
}

impl ModelArtifactProvider for FakeDownloader {
    fn download_model(&self, model_id: &str, target_dir: &Path) -> Result<u64> {
        std::fs::create_dir_all(target_dir)?;
        std::fs::write(target_dir.join("model.bin"), model_id.as_bytes())?;
        Ok(self.bytes_per_model)
    }
}

fn test_config() -> ModelConfig {
    ModelConfig {
        embed: "embed-model".to_string(),
        reranker: "rerank-model".to_string(),
        expander: "expand-model".to_string(),
    }
}

#[test]
fn status_reports_missing_models_when_directories_are_empty() {
    let root = tempdir().expect("create temp root");
    let config = test_config();

    let model_status = status(&config, root.path()).expect("read model status");
    assert!(!model_status.embedder.downloaded);
    assert!(!model_status.reranker.downloaded);
    assert!(!model_status.expander.downloaded);
    assert_eq!(model_status.embedder.size_bytes, None);
    assert_eq!(model_status.reranker.size_bytes, None);
    assert_eq!(model_status.expander.size_bytes, None);
    let expected_embedder = root.path().join("embedder");
    let expected_reranker = root.path().join("reranker");
    let expected_expander = root.path().join("expander");
    assert_eq!(
        model_status.embedder.path.as_deref(),
        Some(expected_embedder.as_path())
    );
    assert_eq!(
        model_status.reranker.path.as_deref(),
        Some(expected_reranker.as_path())
    );
    assert_eq!(
        model_status.expander.path.as_deref(),
        Some(expected_expander.as_path())
    );
}

#[test]
fn pull_downloads_all_missing_models_and_reports_bytes() {
    let root = tempdir().expect("create temp root");
    let config = test_config();
    let downloader = FakeDownloader { bytes_per_model: 11 };

    let report = pull_with_downloader(&config, root.path(), &downloader).expect("pull models");
    assert_eq!(report.downloaded.len(), 3);
    assert_eq!(report.already_present.len(), 0);
    assert_eq!(report.total_bytes, 33);

    let model_status = status(&config, root.path()).expect("read model status");
    assert!(model_status.embedder.downloaded);
    assert!(model_status.reranker.downloaded);
    assert!(model_status.expander.downloaded);
    assert!(model_status.embedder.size_bytes.unwrap_or(0) > 0);
}

#[test]
fn pull_skips_models_that_are_already_present() {
    let root = tempdir().expect("create temp root");
    let config = test_config();
    let downloader = FakeDownloader { bytes_per_model: 5 };

    std::fs::create_dir_all(root.path().join("embedder")).expect("create embedder dir");
    std::fs::write(root.path().join("embedder/model.bin"), b"existing").expect("seed existing");

    let report = pull_with_downloader(&config, root.path(), &downloader).expect("pull models");
    assert_eq!(report.downloaded.len(), 2);
    assert_eq!(report.already_present, vec!["embed-model".to_string()]);
    assert_eq!(report.total_bytes, 10);
}

#[test]
fn pull_emits_progress_events_for_downloaded_and_present_models() {
    let root = tempdir().expect("create temp root");
    let config = test_config();
    let downloader = FakeDownloader { bytes_per_model: 7 };

    std::fs::create_dir_all(root.path().join("embedder")).expect("create embedder dir");
    std::fs::write(root.path().join("embedder/model.bin"), b"existing").expect("seed existing");

    let mut events = Vec::new();
    let report = pull_with_downloader_and_progress(&config, root.path(), &downloader, |event| {
        events.push(event);
    })
    .expect("pull models");

    assert_eq!(report.downloaded, vec!["rerank-model", "expand-model"]);
    assert_eq!(report.already_present, vec!["embed-model"]);
    assert_eq!(report.total_bytes, 14);

    assert_eq!(
        events,
        vec![
            ModelPullEvent::AlreadyPresent {
                role: "embedder".to_string(),
                model: "embed-model".to_string(),
                bytes: 8,
            },
            ModelPullEvent::DownloadStarted {
                role: "reranker".to_string(),
                model: "rerank-model".to_string(),
            },
            ModelPullEvent::DownloadCompleted {
                role: "reranker".to_string(),
                model: "rerank-model".to_string(),
                bytes: 7,
            },
            ModelPullEvent::DownloadStarted {
                role: "expander".to_string(),
                model: "expand-model".to_string(),
            },
            ModelPullEvent::DownloadCompleted {
                role: "expander".to_string(),
                model: "expand-model".to_string(),
                bytes: 7,
            },
        ]
    );
}
