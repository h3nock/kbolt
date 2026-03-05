use std::path::Path;
use std::sync::Mutex;

use serde_json::json;
use tempfile::tempdir;

use crate::config::{
    EmbeddingConfig, InferenceConfig, ModelConfig, ModelProvider, ModelSourceConfig,
    TextInferenceConfig, TextInferenceOutputMode, TextInferenceProvider,
};
use crate::models::{
    build_embedder, pull_with_downloader, pull_with_downloader_and_progress,
    resolve_model_artifact, status, ModelArtifactProvider, ModelDownloadRequest,
    ModelFileRequirement, ModelPullEvent, ModelRole,
};
use crate::Result;

const MODEL_MANIFEST_FILENAME: &str = ".kbolt-model-manifest.json";

#[derive(Default)]
struct FakeDownloader {
    bytes_per_model: u64,
}

impl ModelArtifactProvider for FakeDownloader {
    fn download_model(&self, request: &ModelDownloadRequest, target_dir: &Path) -> Result<u64> {
        std::fs::create_dir_all(target_dir)?;
        std::fs::write(target_dir.join("model.bin"), request.model_id.as_bytes())?;
        Ok(self.bytes_per_model)
    }
}

#[derive(Default)]
struct CapturingDownloader {
    requests: Mutex<Vec<ModelDownloadRequest>>,
}

impl CapturingDownloader {
    fn take_requests(&self) -> Vec<ModelDownloadRequest> {
        let mut guard = self.requests.lock().expect("lock requests");
        std::mem::take(&mut *guard)
    }
}

impl ModelArtifactProvider for CapturingDownloader {
    fn download_model(&self, request: &ModelDownloadRequest, target_dir: &Path) -> Result<u64> {
        self.requests
            .lock()
            .expect("lock requests")
            .push(request.clone());
        std::fs::create_dir_all(target_dir)?;
        std::fs::write(target_dir.join("model.bin"), request.model_id.as_bytes())?;
        Ok(1)
    }
}

fn provider_key(provider: &ModelProvider) -> &'static str {
    match provider {
        ModelProvider::HuggingFace => "huggingface",
    }
}

fn seed_model(root: &Path, role: &str, source: &ModelSourceConfig, payload: &[u8]) {
    let role_dir = root.join(role);
    std::fs::create_dir_all(&role_dir).expect("create role dir");
    std::fs::write(role_dir.join("model.bin"), payload).expect("write model payload");

    let manifest = json!({
        "provider": provider_key(&source.provider),
        "id": source.id,
        "revision": source.revision,
    });
    let bytes = serde_json::to_vec_pretty(&manifest).expect("serialize manifest");
    std::fs::write(role_dir.join(MODEL_MANIFEST_FILENAME), bytes).expect("write model manifest");
}

fn test_config() -> ModelConfig {
    ModelConfig {
        embedder: ModelSourceConfig {
            provider: ModelProvider::HuggingFace,
            id: "embed-model".to_string(),
            revision: None,
        },
        reranker: ModelSourceConfig {
            provider: ModelProvider::HuggingFace,
            id: "rerank-model".to_string(),
            revision: None,
        },
        expander: ModelSourceConfig {
            provider: ModelProvider::HuggingFace,
            id: "expand-model".to_string(),
            revision: None,
        },
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
    let downloader = FakeDownloader {
        bytes_per_model: 11,
    };

    let report = pull_with_downloader(
        &config,
        None,
        &crate::config::InferenceConfig::default(),
        root.path(),
        &downloader,
    )
    .expect("pull models");
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

    seed_model(root.path(), "embedder", &config.embedder, b"existing");

    let report = pull_with_downloader(
        &config,
        None,
        &crate::config::InferenceConfig::default(),
        root.path(),
        &downloader,
    )
    .expect("pull models");
    assert_eq!(report.downloaded.len(), 2);
    assert_eq!(report.already_present, vec!["embed-model".to_string()]);
    assert_eq!(report.total_bytes, 10);
}

#[test]
fn pull_emits_progress_events_for_downloaded_and_present_models() {
    let root = tempdir().expect("create temp root");
    let config = test_config();
    let downloader = FakeDownloader { bytes_per_model: 7 };

    seed_model(root.path(), "embedder", &config.embedder, b"existing");

    let mut events = Vec::new();
    let report = pull_with_downloader_and_progress(
        &config,
        None,
        &crate::config::InferenceConfig::default(),
        root.path(),
        &downloader,
        |event| {
            events.push(event);
        },
    )
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

#[test]
fn status_treats_payload_without_manifest_as_missing() {
    let root = tempdir().expect("create temp root");
    let config = test_config();

    std::fs::create_dir_all(root.path().join("embedder")).expect("create embedder dir");
    std::fs::write(root.path().join("embedder/model.bin"), b"payload").expect("seed embedder");

    let model_status = status(&config, root.path()).expect("read model status");
    assert!(!model_status.embedder.downloaded);
    assert_eq!(model_status.embedder.size_bytes, None);
}

#[test]
fn pull_redownloads_model_when_manifest_does_not_match_source() {
    let root = tempdir().expect("create temp root");
    let config = test_config();
    let downloader = FakeDownloader { bytes_per_model: 5 };

    let mut mismatched_embedder = config.embedder.clone();
    mismatched_embedder.id = "embed-model-old".to_string();
    seed_model(root.path(), "embedder", &mismatched_embedder, b"existing");
    seed_model(root.path(), "reranker", &config.reranker, b"existing");
    seed_model(root.path(), "expander", &config.expander, b"existing");

    let report = pull_with_downloader(
        &config,
        None,
        &crate::config::InferenceConfig::default(),
        root.path(),
        &downloader,
    )
    .expect("pull models");
    assert_eq!(report.downloaded, vec!["embed-model"]);
    assert_eq!(
        report.already_present,
        vec!["rerank-model".to_string(), "expand-model".to_string()]
    );
    assert_eq!(report.total_bytes, 5);
}

#[test]
fn resolve_model_artifact_returns_metadata_for_matching_manifest() {
    let root = tempdir().expect("create temp root");
    let config = test_config();
    seed_model(root.path(), "embedder", &config.embedder, b"payload");
    seed_model(root.path(), "reranker", &config.reranker, b"rerank");
    seed_model(root.path(), "expander", &config.expander, b"expand");

    let resolved = resolve_model_artifact(&config, root.path(), ModelRole::Embedder)
        .expect("resolve embedder artifact");
    assert_eq!(resolved.role, ModelRole::Embedder);
    assert_eq!(resolved.source, config.embedder);
    assert_eq!(resolved.path, root.path().join("embedder"));
    assert_eq!(resolved.size_bytes, 7);

    let reranker = resolve_model_artifact(&config, root.path(), ModelRole::Reranker)
        .expect("resolve reranker artifact");
    assert_eq!(reranker.role, ModelRole::Reranker);
    assert_eq!(reranker.source, config.reranker);
    assert_eq!(reranker.path, root.path().join("reranker"));
    assert_eq!(reranker.size_bytes, 6);

    let expander = resolve_model_artifact(&config, root.path(), ModelRole::Expander)
        .expect("resolve expander artifact");
    assert_eq!(expander.role, ModelRole::Expander);
    assert_eq!(expander.source, config.expander);
    assert_eq!(expander.path, root.path().join("expander"));
    assert_eq!(expander.size_bytes, 6);
}

#[test]
fn resolve_model_artifact_errors_when_manifest_is_missing_or_mismatched() {
    let root = tempdir().expect("create temp root");
    let config = test_config();

    let missing = resolve_model_artifact(&config, root.path(), ModelRole::Embedder)
        .expect_err("missing model should fail");
    assert!(
        missing.to_string().contains("model not available"),
        "unexpected error: {missing}"
    );

    let mut mismatched = config.embedder.clone();
    mismatched.id = "other-model".to_string();
    seed_model(root.path(), "embedder", &mismatched, b"payload");

    let mismatched_err = resolve_model_artifact(&config, root.path(), ModelRole::Embedder)
        .expect_err("mismatched manifest should fail");
    assert!(
        mismatched_err.to_string().contains("model not available"),
        "unexpected error: {mismatched_err}"
    );
}

#[test]
fn build_embedder_returns_none_without_embedding_config() {
    let embedder = build_embedder(None).expect("build embedder");
    assert!(embedder.is_none());
}

#[test]
fn pull_builds_role_specific_default_download_requirements() {
    let root = tempdir().expect("create temp root");
    let config = test_config();
    let downloader = CapturingDownloader::default();
    let inference = InferenceConfig::default();

    let report = pull_with_downloader(&config, None, &inference, root.path(), &downloader)
        .expect("pull models");
    assert_eq!(report.downloaded.len(), 3);

    let requests = downloader.take_requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].model_id, "embed-model");
    assert_eq!(
        requests[0].requirements,
        vec![
            ModelFileRequirement::SingleExtension {
                extension: "onnx",
                config_field: "embeddings.onnx_file",
            },
            ModelFileRequirement::SingleTokenizerJson {
                config_field: "embeddings.tokenizer_file",
            },
        ]
    );
    assert_eq!(requests[1].model_id, "rerank-model");
    assert_eq!(
        requests[1].requirements,
        vec![ModelFileRequirement::SingleExtension {
            extension: "gguf",
            config_field: "inference.reranker.model_file",
        }]
    );
    assert_eq!(requests[2].model_id, "expand-model");
    assert_eq!(
        requests[2].requirements,
        vec![ModelFileRequirement::SingleExtension {
            extension: "gguf",
            config_field: "inference.expander.model_file",
        }]
    );
}

#[test]
fn pull_uses_configured_relative_file_overrides() {
    let root = tempdir().expect("create temp root");
    let config = test_config();
    let downloader = CapturingDownloader::default();
    let embeddings = EmbeddingConfig::LocalOnnx {
        onnx_file: Some("onnx/model.onnx".to_string()),
        tokenizer_file: Some("tokenizer.json".to_string()),
        max_length: 512,
    };
    let inference = InferenceConfig {
        reranker: Some(TextInferenceConfig {
            provider: TextInferenceProvider::LocalLlama {
                model_file: Some("weights/rerank.gguf".to_string()),
                max_tokens: 128,
                n_ctx: 2048,
                n_gpu_layers: 0,
            },
        }),
        expander: Some(TextInferenceConfig {
            provider: TextInferenceProvider::OpenAiCompatible {
                output_mode: TextInferenceOutputMode::JsonObject,
                model: "remote".to_string(),
                base_url: "https://example.com/v1".to_string(),
                api_key_env: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        }),
    };

    let report = pull_with_downloader(
        &config,
        Some(&embeddings),
        &inference,
        root.path(),
        &downloader,
    )
    .expect("pull models");
    assert_eq!(report.downloaded.len(), 3);

    let requests = downloader.take_requests();
    assert_eq!(
        requests[0].requirements,
        vec![
            ModelFileRequirement::ExactPath {
                path: "onnx/model.onnx".to_string(),
                config_field: "embeddings.onnx_file",
            },
            ModelFileRequirement::ExactPath {
                path: "tokenizer.json".to_string(),
                config_field: "embeddings.tokenizer_file",
            },
        ]
    );
    assert_eq!(
        requests[1].requirements,
        vec![ModelFileRequirement::ExactPath {
            path: "weights/rerank.gguf".to_string(),
            config_field: "inference.reranker.model_file",
        }]
    );
    assert_eq!(
        requests[2].requirements,
        vec![ModelFileRequirement::SingleExtension {
            extension: "gguf",
            config_field: "inference.expander.model_file",
        }]
    );
}

#[test]
fn pull_uses_gguf_requirements_when_embedder_is_local_gguf() {
    let root = tempdir().expect("create temp root");
    let config = test_config();
    let downloader = CapturingDownloader::default();
    let embeddings = EmbeddingConfig::LocalGguf {
        model_file: Some("embeddinggemma-300M-Q8_0.gguf".to_string()),
        batch_size: 4,
        n_threads: Some(6),
        n_threads_batch: Some(6),
    };

    let report = pull_with_downloader(
        &config,
        Some(&embeddings),
        &InferenceConfig::default(),
        root.path(),
        &downloader,
    )
    .expect("pull models");
    assert_eq!(report.downloaded.len(), 3);

    let requests = downloader.take_requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[0].requirements,
        vec![ModelFileRequirement::ExactPath {
            path: "embeddinggemma-300M-Q8_0.gguf".to_string(),
            config_field: "embeddings.model_file",
        }]
    );
}
