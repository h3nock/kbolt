use std::path::Path;
use std::sync::Arc;

use kbolt_types::KboltError;
use serde::Deserialize;
use serde_json::json;

use crate::config::{EmbeddingConfig, ModelConfig, TextInferenceConfig, TextInferenceProvider};
use crate::models::artifacts::resolve_file_with_extension;
use crate::models::chat::HttpChatClient;
use crate::models::completion::CompletionClient;
use crate::models::expander::HeuristicExpander;
use crate::models::http::{HttpJsonClient, HttpOperation};
use crate::models::local_llama::build_local_llama_completion_client;
use crate::models::local_onnx::build_local_onnx_embedder;
use crate::models::reranker::HeuristicReranker;
use crate::models::{resolve_model_artifact, Embedder, Expander, ModelRole, Reranker};
use crate::Result;

#[cfg(test)]
use crate::config::TextInferenceOutputMode;
#[cfg(test)]
use crate::models::chat::build_chat_payload;
#[cfg(test)]
use crate::models::http::parse_retry_after_seconds;

#[derive(Debug, Clone)]
struct HttpApiEmbedder {
    client: HttpJsonClient,
    model: String,
    batch_size: usize,
}

#[derive(Clone)]
struct ChatBackedReranker {
    chat: Arc<dyn CompletionClient>,
}

#[derive(Clone)]
struct ChatBackedExpander {
    chat: Arc<dyn CompletionClient>,
}

#[cfg(test)]
pub(crate) fn build_embedder(
    config: Option<&EmbeddingConfig>,
) -> Result<Option<Arc<dyn Embedder>>> {
    build_embedder_inner(config, None)
}

pub(crate) fn build_embedder_with_local_runtime(
    config: Option<&EmbeddingConfig>,
    model_config: &ModelConfig,
    model_dir: &Path,
) -> Result<Option<Arc<dyn Embedder>>> {
    build_embedder_inner(
        config,
        Some(LocalRuntimeContext {
            model_config,
            model_dir,
        }),
    )
}

#[cfg(test)]
pub(crate) fn build_reranker(config: Option<&TextInferenceConfig>) -> Result<Arc<dyn Reranker>> {
    build_reranker_inner(config, None)
}

pub(crate) fn build_reranker_with_local_runtime(
    config: Option<&TextInferenceConfig>,
    model_config: &ModelConfig,
    model_dir: &Path,
) -> Result<Arc<dyn Reranker>> {
    build_reranker_inner(
        config,
        Some(LocalRuntimeContext {
            model_config,
            model_dir,
        }),
    )
}

#[cfg(test)]
pub(crate) fn build_expander(config: Option<&TextInferenceConfig>) -> Result<Arc<dyn Expander>> {
    build_expander_inner(config, None)
}

pub(crate) fn build_expander_with_local_runtime(
    config: Option<&TextInferenceConfig>,
    model_config: &ModelConfig,
    model_dir: &Path,
) -> Result<Arc<dyn Expander>> {
    build_expander_inner(
        config,
        Some(LocalRuntimeContext {
            model_config,
            model_dir,
        }),
    )
}

#[derive(Clone, Copy)]
struct LocalRuntimeContext<'a> {
    model_config: &'a ModelConfig,
    model_dir: &'a Path,
}

fn build_embedder_inner(
    config: Option<&EmbeddingConfig>,
    local_runtime: Option<LocalRuntimeContext<'_>>,
) -> Result<Option<Arc<dyn Embedder>>> {
    let Some(config) = config else {
        return Ok(None);
    };

    let (provider_name, model, base_url, api_key_env, timeout_ms, max_retries, batch_size) =
        match config {
            EmbeddingConfig::OpenAiCompatible {
                model,
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
                batch_size,
            } => (
                "openai_compatible",
                model.clone(),
                base_url.as_str(),
                api_key_env.as_deref(),
                *timeout_ms,
                *max_retries,
                *batch_size,
            ),
            EmbeddingConfig::Voyage {
                model,
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
                batch_size,
            } => (
                "voyage",
                model.clone(),
                base_url.as_str(),
                api_key_env.as_deref(),
                *timeout_ms,
                *max_retries,
                *batch_size,
            ),
            EmbeddingConfig::LocalOnnx {
                onnx_file,
                tokenizer_file,
                max_length,
            } => {
                let runtime = local_runtime.ok_or_else(|| {
                    KboltError::Inference(
                        "local_onnx embedder requires local runtime context".to_string(),
                    )
                })?;
                let artifact =
                    resolve_model_artifact(runtime.model_config, runtime.model_dir, ModelRole::Embedder)?;
                let embedder = build_local_onnx_embedder(
                    &artifact.path,
                    onnx_file.as_deref(),
                    tokenizer_file.as_deref(),
                    *max_length,
                )?;
                return Ok(Some(Arc::new(embedder)));
            }
        };
    let client = HttpJsonClient::new(
        base_url,
        api_key_env,
        timeout_ms,
        max_retries,
        "embedding",
        provider_name,
    );
    let embedder: Arc<dyn Embedder> = Arc::new(HttpApiEmbedder {
        client,
        model,
        batch_size,
    });
    Ok(Some(embedder))
}

fn build_reranker_inner(
    config: Option<&TextInferenceConfig>,
    local_runtime: Option<LocalRuntimeContext<'_>>,
) -> Result<Arc<dyn Reranker>> {
    let reranker: Arc<dyn Reranker> = match config {
        Some(config) => Arc::new(ChatBackedReranker {
            chat: build_completion_client_for_role(
                &config.provider,
                local_runtime,
                ModelRole::Reranker,
                "reranker",
            )?,
        }),
        None => Arc::new(HeuristicReranker),
    };
    Ok(reranker)
}

fn build_expander_inner(
    config: Option<&TextInferenceConfig>,
    local_runtime: Option<LocalRuntimeContext<'_>>,
) -> Result<Arc<dyn Expander>> {
    let expander: Arc<dyn Expander> = match config {
        Some(config) => Arc::new(ChatBackedExpander {
            chat: build_completion_client_for_role(
                &config.provider,
                local_runtime,
                ModelRole::Expander,
                "expander",
            )?,
        }),
        None => Arc::new(HeuristicExpander),
    };
    Ok(expander)
}

fn build_completion_client_for_role(
    provider: &TextInferenceProvider,
    local_runtime: Option<LocalRuntimeContext<'_>>,
    role: ModelRole,
    role_label: &str,
) -> Result<Arc<dyn CompletionClient>> {
    match provider {
        TextInferenceProvider::OpenAiCompatible {
            output_mode,
            model,
            base_url,
            api_key_env,
            timeout_ms,
            max_retries,
        } => Ok(Arc::new(HttpChatClient::new(
            base_url,
            api_key_env.as_deref(),
            *timeout_ms,
            *max_retries,
            model,
            output_mode.clone(),
            "openai_compatible",
        ))),
        TextInferenceProvider::LocalLlama {
            model_file,
            max_tokens,
            n_ctx,
            n_gpu_layers,
        } => {
            let runtime = local_runtime.ok_or_else(|| {
                KboltError::Inference(format!(
                    "local_llama {role_label} requires local runtime context"
                ))
            })?;
            build_local_llama_client(
                runtime,
                role,
                model_file.as_deref(),
                *max_tokens,
                *n_ctx,
                *n_gpu_layers,
            )
        }
    }
}

impl Embedder for HttpApiEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        embed_with_http_api(&self.client, &self.model, self.batch_size, texts)
    }
}

impl Reranker for ChatBackedReranker {
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        let system = "You are a retrieval reranker. Return JSON only as {\"scores\":[number,...]} with one score per document, each score between 0 and 1.";
        let mut user = format!("Query:\n{query}\n\nDocuments:\n");
        for (index, doc) in docs.iter().enumerate() {
            user.push_str(&format!("[{index}] {doc}\n"));
        }
        user.push_str("\nRespond with exactly one score per document, in order.");

        let content = self.chat.complete(system, &user)?;
        let parsed: RerankerResponse = parse_json_payload("reranker response", &content)?;
        let mut scores = parsed.into_scores();
        if scores.len() != docs.len() {
            return Err(KboltError::Inference(format!(
                "reranker response size mismatch: expected {}, got {}",
                docs.len(),
                scores.len()
            ))
            .into());
        }

        for score in &mut scores {
            *score = score.clamp(0.0, 1.0);
        }
        Ok(scores)
    }
}

impl Expander for ChatBackedExpander {
    fn expand(&self, query: &str) -> Result<Vec<String>> {
        let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        let system = "You generate search query rewrites. Return JSON only as {\"variants\":[string,...]} with 2 to 4 concise variants.";
        let user = format!(
            "Original query:\n{normalized}\n\nGenerate variants that improve lexical recall, semantic recall, and one HyDE-style descriptive query."
        );
        let content = self.chat.complete(system, &user)?;
        let parsed: ExpanderResponse = parse_json_payload("expander response", &content)?;

        let mut seen = std::collections::HashSet::new();
        let mut variants = Vec::new();
        let normalized_key = normalized.to_ascii_lowercase();
        seen.insert(normalized_key);
        variants.push(normalized);

        for variant in parsed.into_variants() {
            let trimmed = variant.trim();
            if trimmed.is_empty() {
                continue;
            }

            let key = trimmed.to_ascii_lowercase();
            if seen.insert(key) {
                variants.push(trimmed.to_string());
            }
        }

        Ok(variants)
    }
}

fn embed_with_http_api(
    client: &HttpJsonClient,
    model: &str,
    batch_size: usize,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let mut vectors = Vec::new();
    for batch in texts.chunks(batch_size) {
        let payload = json!({
            "model": model,
            "input": batch,
        });
        let parsed: EmbeddingResponseEnvelope =
            client.post_json("embeddings", &payload, HttpOperation::Embedding)?;
        let response_vectors = parsed.into_vectors(batch.len())?;
        vectors.extend(response_vectors);
    }
    Ok(vectors)
}

fn build_local_llama_client(
    runtime: LocalRuntimeContext<'_>,
    role: ModelRole,
    model_file: Option<&str>,
    max_tokens: usize,
    n_ctx: u32,
    n_gpu_layers: u32,
) -> Result<Arc<dyn CompletionClient>> {
    let artifact = resolve_model_artifact(runtime.model_config, runtime.model_dir, role)?;
    let gguf_path = resolve_file_with_extension(&artifact.path, model_file, "gguf", "model_file")?;
    build_local_llama_completion_client(&gguf_path, max_tokens, n_ctx, n_gpu_layers)
}

fn parse_json_payload<T>(label: &str, content: &str) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let trimmed = content.trim();
    serde_json::from_str(trimmed).map_err(|err| {
        KboltError::Inference(format!(
            "failed to parse {label} as JSON: {err}; payload={content}"
        ))
        .into()
    })
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingResponseEnvelope {
    OpenAiLike { data: Vec<EmbeddingItem> },
    VoyageLike { embeddings: Vec<Vec<f32>> },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RerankerResponse {
    Scores(Vec<f32>),
    Wrapped { scores: Vec<f32> },
}

impl RerankerResponse {
    fn into_scores(self) -> Vec<f32> {
        match self {
            Self::Scores(scores) => scores,
            Self::Wrapped { scores } => scores,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ExpanderResponse {
    Variants(Vec<String>),
    Wrapped { variants: Vec<String> },
}

impl ExpanderResponse {
    fn into_variants(self) -> Vec<String> {
        match self {
            Self::Variants(variants) => variants,
            Self::Wrapped { variants } => variants,
        }
    }
}

impl EmbeddingResponseEnvelope {
    fn into_vectors(self, expected_len: usize) -> Result<Vec<Vec<f32>>> {
        let mut vectors = match self {
            Self::OpenAiLike { data } => {
                let all_indexed = data.iter().all(|item| item.index.is_some());
                let mut ordered = data
                    .into_iter()
                    .enumerate()
                    .map(|(position, item)| {
                        let order = item.index.unwrap_or(position);
                        (order, item.embedding.into_vec())
                    })
                    .collect::<Vec<_>>();
                if all_indexed {
                    ordered.sort_by_key(|(order, _)| *order);
                }
                ordered
                    .into_iter()
                    .map(|(_, embedding)| embedding)
                    .collect::<Vec<_>>()
            }
            Self::VoyageLike { embeddings } => embeddings,
        };

        if vectors.len() != expected_len {
            return Err(KboltError::Inference(format!(
                "embedding response size mismatch: expected {expected_len}, got {}",
                vectors.len()
            ))
            .into());
        }

        for (index, vector) in vectors.iter().enumerate() {
            if vector.is_empty() {
                return Err(KboltError::Inference(format!(
                    "embedding response contains empty vector at index {index}"
                ))
                .into());
            }
        }

        Ok(std::mem::take(&mut vectors))
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    #[serde(default)]
    index: Option<usize>,
    embedding: EmbeddingVector,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingVector {
    Direct(Vec<f32>),
    Wrapped { values: Vec<f32> },
}

impl EmbeddingVector {
    fn into_vec(self) -> Vec<f32> {
        match self {
            Self::Direct(values) => values,
            Self::Wrapped { values } => values,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    fn base_openai_config(base_url: String) -> EmbeddingConfig {
        EmbeddingConfig::OpenAiCompatible {
            model: "embed-model".to_string(),
            base_url,
            api_key_env: None,
            timeout_ms: 5_000,
            batch_size: 64,
            max_retries: 0,
        }
    }

    fn base_voyage_config(base_url: String) -> EmbeddingConfig {
        EmbeddingConfig::Voyage {
            model: "embed-model".to_string(),
            base_url,
            api_key_env: None,
            timeout_ms: 5_000,
            batch_size: 64,
            max_retries: 0,
        }
    }

    fn base_text_config(
        base_url: String,
        output_mode: TextInferenceOutputMode,
    ) -> TextInferenceConfig {
        TextInferenceConfig {
            provider: TextInferenceProvider::OpenAiCompatible {
                output_mode,
                model: "text-model".to_string(),
                base_url,
                api_key_env: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        }
    }

    fn base_local_llama_text_config() -> TextInferenceConfig {
        TextInferenceConfig {
            provider: TextInferenceProvider::LocalLlama {
                model_file: None,
                max_tokens: 128,
                n_ctx: 2048,
                n_gpu_layers: 0,
            },
        }
    }

    fn serve_once(status_code: u16, body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        let payload = body.to_string();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept client");

            let _ = read_full_request(&mut stream);

            let status_line = match status_code {
                200 => "HTTP/1.1 200 OK",
                429 => "HTTP/1.1 429 Too Many Requests",
                400 => "HTTP/1.1 400 Bad Request",
                500 => "HTTP/1.1 500 Internal Server Error",
                other => panic!("unsupported status code in test server: {other}"),
            };
            let response = format!(
                "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.as_bytes().len(),
                payload
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });

        format!("http://{addr}")
    }

    struct TestResponse {
        status_code: u16,
        body: &'static str,
        retry_after: Option<&'static str>,
    }

    fn serve_sequence(responses: Vec<TestResponse>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        std::thread::spawn(move || {
            for response_spec in responses {
                let (mut stream, _) = listener.accept().expect("accept client");
                let _ = read_full_request(&mut stream);
                let status_line = match response_spec.status_code {
                    200 => "HTTP/1.1 200 OK",
                    429 => "HTTP/1.1 429 Too Many Requests",
                    400 => "HTTP/1.1 400 Bad Request",
                    500 => "HTTP/1.1 500 Internal Server Error",
                    other => panic!("unsupported status code in test server: {other}"),
                };
                let retry_after = response_spec
                    .retry_after
                    .map(|value| format!("Retry-After: {value}\r\n"))
                    .unwrap_or_default();
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\n{retry_after}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_spec.body.as_bytes().len(),
                    response_spec.body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write response");
            }
        });

        format!("http://{addr}")
    }

    fn read_full_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        let mut raw = Vec::new();
        let mut header_end = None;
        while header_end.is_none() {
            let mut chunk = [0_u8; 1024];
            let read = stream.read(&mut chunk).expect("read request bytes");
            if read == 0 {
                break;
            }
            raw.extend_from_slice(&chunk[..read]);
            header_end = raw.windows(4).position(|window| window == b"\r\n\r\n");
        }

        let Some(header_end) = header_end else {
            return Vec::new();
        };
        let header_end = header_end + 4;
        let headers = String::from_utf8_lossy(&raw[..header_end]).to_ascii_lowercase();
        let mut content_length = 0usize;
        for line in headers.lines() {
            if let Some(value) = line.strip_prefix("content-length:") {
                content_length = value.trim().parse::<usize>().unwrap_or(0);
                break;
            }
        }

        let already_read_body = raw.len().saturating_sub(header_end);
        let mut remaining = content_length.saturating_sub(already_read_body);
        while remaining > 0 {
            let mut chunk = [0_u8; 1024];
            let read = stream.read(&mut chunk).expect("read request body");
            if read == 0 {
                break;
            }
            remaining = remaining.saturating_sub(read);
        }

        let body_end = header_end.saturating_add(content_length).min(raw.len());
        raw.get(header_end..body_end).unwrap_or_default().to_vec()
    }

    #[test]
    fn build_embedder_returns_none_when_not_configured() {
        let embedder = build_embedder(None).expect("build embedder");
        assert!(embedder.is_none());
    }

    #[test]
    fn openai_compatible_embedder_parses_openai_style_response() {
        let body = r#"{
  "data": [
    {"index": 1, "embedding": [0.3, 0.4]},
    {"index": 0, "embedding": [0.1, 0.2]}
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_openai_config(base_url);
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder
            .embed_batch(&["a".to_string(), "b".to_string()])
            .expect("embed");
        assert_eq!(vectors, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    }

    #[test]
    fn openai_compatible_embedder_parses_openai_style_response_without_index() {
        let body = r#"{
  "data": [
    {"embedding": [0.51, 0.52]},
    {"embedding": [0.61, 0.62]}
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_openai_config(base_url);
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder
            .embed_batch(&["a".to_string(), "b".to_string()])
            .expect("embed");
        assert_eq!(vectors, vec![vec![0.51, 0.52], vec![0.61, 0.62]]);
    }

    #[test]
    fn openai_compatible_embedder_parses_wrapped_embedding_values() {
        let body = r#"{
  "data": [
    {"index": 0, "embedding": {"values": [0.15, 0.25]}}
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_openai_config(base_url);
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder.embed_batch(&["a".to_string()]).expect("embed");
        assert_eq!(vectors, vec![vec![0.15, 0.25]]);
    }

    #[test]
    fn voyage_embedder_parses_voyage_style_response() {
        let body = r#"{
  "embeddings": [
    [0.11, 0.22],
    [0.33, 0.44]
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_voyage_config(base_url);
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder
            .embed_batch(&["a".to_string(), "b".to_string()])
            .expect("embed");
        assert_eq!(vectors, vec![vec![0.11, 0.22], vec![0.33, 0.44]]);
    }

    #[test]
    fn embedder_errors_when_api_key_env_is_missing() {
        let body = r#"{"data":[{"index":0,"embedding":[0.1,0.2]}]}"#;
        let base_url = serve_once(200, body);
        let config = EmbeddingConfig::OpenAiCompatible {
            model: "embed-model".to_string(),
            base_url,
            api_key_env: Some("KBOLT_TEST_MISSING_API_KEY".to_string()),
            timeout_ms: 5_000,
            batch_size: 64,
            max_retries: 0,
        };
        let embedder: Arc<dyn Embedder> = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        std::env::remove_var("KBOLT_TEST_MISSING_API_KEY");
        let err = embedder
            .embed_batch(&["hello".to_string()])
            .expect_err("missing key should fail");
        assert!(err
            .to_string()
            .contains("embedding API key env var is not set"));
    }

    #[test]
    fn local_onnx_embedder_requires_local_runtime_context() {
        let config = EmbeddingConfig::LocalOnnx {
            onnx_file: None,
            tokenizer_file: None,
            max_length: 256,
        };
        let err = match build_embedder(Some(&config)) {
            Ok(_) => panic!("local runtime context should be required"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("local_onnx embedder requires local runtime context"));
    }

    #[test]
    fn build_reranker_uses_heuristic_when_inference_config_is_missing() {
        let reranker = build_reranker(None).expect("build reranker");
        let scores = reranker
            .rerank(
                "rust traits",
                &[
                    "Rust traits and impl examples".to_string(),
                    "Python decorators".to_string(),
                ],
            )
            .expect("rerank docs");
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn local_llama_reranker_requires_local_runtime_context() {
        let config = base_local_llama_text_config();
        let err = match build_reranker(Some(&config)) {
            Ok(_) => panic!("local runtime context should be required"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("local_llama reranker requires local runtime context"));
    }

    #[test]
    fn openai_compatible_reranker_parses_json_scores() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "{\"scores\":[0.2,0.9]}"
      }
    }
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_text_config(base_url, TextInferenceOutputMode::JsonObject);
        let reranker = build_reranker(Some(&config)).expect("build reranker");

        let scores = reranker
            .rerank("query", &["doc one".to_string(), "doc two".to_string()])
            .expect("rerank docs");
        assert_eq!(scores, vec![0.2, 0.9]);
    }

    #[test]
    fn build_expander_uses_heuristic_when_inference_config_is_missing() {
        let expander = build_expander(None).expect("build expander");
        let variants = expander.expand("rust traits").expect("expand query");
        assert!(!variants.is_empty());
        assert_eq!(variants[0], "rust traits");
    }

    #[test]
    fn openai_compatible_expander_parses_json_variants() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "{\"variants\":[\"trait object rust\",\"explain rust traits\"]}"
      }
    }
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_text_config(base_url, TextInferenceOutputMode::JsonObject);
        let expander = build_expander(Some(&config)).expect("build expander");

        let variants = expander.expand("rust traits").expect("expand query");
        assert_eq!(
            variants,
            vec![
                "rust traits".to_string(),
                "trait object rust".to_string(),
                "explain rust traits".to_string(),
            ]
        );
    }

    #[test]
    fn json_object_mode_sets_response_format_in_chat_payload() {
        let payload = build_chat_payload(
            "model",
            "system",
            "user",
            TextInferenceOutputMode::JsonObject,
        );
        assert_eq!(payload["response_format"]["type"], "json_object");
    }

    #[test]
    fn text_mode_omits_response_format_in_chat_payload() {
        let payload = build_chat_payload("model", "system", "user", TextInferenceOutputMode::Text);
        assert!(payload.get("response_format").is_none());
    }

    #[test]
    fn text_mode_parses_fenced_json() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "```json\n{\"scores\":[0.2,0.9]}\n```"
      }
    }
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_text_config(base_url, TextInferenceOutputMode::Text);
        let reranker = build_reranker(Some(&config)).expect("build reranker");

        let scores = reranker
            .rerank("query", &["doc one".to_string(), "doc two".to_string()])
            .expect("rerank docs");
        assert_eq!(scores, vec![0.2, 0.9]);
    }

    #[test]
    fn text_mode_fails_fast_when_content_is_not_json() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "this is not json"
      }
    }
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_text_config(base_url, TextInferenceOutputMode::Text);
        let reranker = build_reranker(Some(&config)).expect("build reranker");

        let err = reranker
            .rerank("query", &["doc one".to_string(), "doc two".to_string()])
            .expect_err("non-json payload should fail");
        assert!(err
            .to_string()
            .contains("failed to parse reranker response as JSON"));
    }

    #[test]
    fn embedder_retries_on_rate_limit_status() {
        let base_url = serve_sequence(vec![
            TestResponse {
                status_code: 429,
                body: r#"{"error":"rate limit"}"#,
                retry_after: Some("0"),
            },
            TestResponse {
                status_code: 200,
                body: r#"{"data":[{"index":0,"embedding":[0.1,0.2]}]}"#,
                retry_after: None,
            },
        ]);
        let config = EmbeddingConfig::OpenAiCompatible {
            model: "embed-model".to_string(),
            base_url,
            api_key_env: None,
            timeout_ms: 5_000,
            batch_size: 64,
            max_retries: 1,
        };
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder
            .embed_batch(&["hello".to_string()])
            .expect("embed should retry then succeed");
        assert_eq!(vectors, vec![vec![0.1, 0.2]]);
    }

    #[test]
    fn parse_retry_after_seconds_accepts_delta_seconds() {
        assert_eq!(parse_retry_after_seconds(Some("7")), Some(7));
    }

    #[test]
    fn parse_retry_after_seconds_accepts_http_date() {
        let retry_at = std::time::SystemTime::now() + Duration::from_secs(60);
        let header = httpdate::fmt_http_date(retry_at);
        let parsed = parse_retry_after_seconds(Some(&header));
        assert!(parsed.is_some());
        let seconds = parsed.expect("parsed seconds");
        assert!(seconds <= 60);
    }
}
