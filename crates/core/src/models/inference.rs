use std::sync::Arc;
use std::time::Duration;

use kbolt_types::KboltError;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::{
    EmbeddingConfig, EmbeddingProvider, TextInferenceConfig, TextInferenceProvider,
};
use crate::models::expander::HeuristicExpander;
use crate::models::reranker::HeuristicReranker;
use crate::models::{Embedder, Expander, Reranker};
use crate::Result;

#[derive(Debug, Clone)]
struct OpenAiCompatibleEmbedder {
    config: EmbeddingConfig,
}

#[derive(Debug, Clone)]
struct VoyageEmbedder {
    config: EmbeddingConfig,
}

#[derive(Debug, Clone)]
struct OpenAiCompatibleReranker {
    config: TextInferenceConfig,
}

#[derive(Debug, Clone)]
struct OpenAiCompatibleExpander {
    config: TextInferenceConfig,
}

pub(crate) fn build_embedder(
    config: Option<&EmbeddingConfig>,
) -> Result<Option<Arc<dyn Embedder>>> {
    let Some(config) = config else {
        return Ok(None);
    };

    let embedder: Arc<dyn Embedder> = match config.provider {
        EmbeddingProvider::OpenAiCompatible => Arc::new(OpenAiCompatibleEmbedder {
            config: config.clone(),
        }),
        EmbeddingProvider::Voyage => Arc::new(VoyageEmbedder {
            config: config.clone(),
        }),
    };
    Ok(Some(embedder))
}

pub(crate) fn build_reranker(config: Option<&TextInferenceConfig>) -> Result<Arc<dyn Reranker>> {
    let reranker: Arc<dyn Reranker> = match config {
        Some(config) => match config.provider {
            TextInferenceProvider::OpenAiCompatible => Arc::new(OpenAiCompatibleReranker {
                config: config.clone(),
            }),
        },
        None => Arc::new(HeuristicReranker),
    };
    Ok(reranker)
}

pub(crate) fn build_expander(config: Option<&TextInferenceConfig>) -> Result<Arc<dyn Expander>> {
    let expander: Arc<dyn Expander> = match config {
        Some(config) => match config.provider {
            TextInferenceProvider::OpenAiCompatible => Arc::new(OpenAiCompatibleExpander {
                config: config.clone(),
            }),
        },
        None => Arc::new(HeuristicExpander),
    };
    Ok(expander)
}

impl Embedder for OpenAiCompatibleEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        embed_with_http_api(&self.config, "openai_compatible", texts)
    }
}

impl Embedder for VoyageEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        embed_with_http_api(&self.config, "voyage", texts)
    }
}

impl Reranker for OpenAiCompatibleReranker {
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

        let content = request_chat_completion(
            &self.config,
            "openai_compatible",
            system,
            &user,
            self.config.max_retries,
        )?;
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

impl Expander for OpenAiCompatibleExpander {
    fn expand(&self, query: &str) -> Result<Vec<String>> {
        let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        let system = "You generate search query rewrites. Return JSON only as {\"variants\":[string,...]} with 2 to 4 concise variants.";
        let user = format!(
            "Original query:\n{normalized}\n\nGenerate variants that improve lexical recall, semantic recall, and one HyDE-style descriptive query."
        );
        let content = request_chat_completion(
            &self.config,
            "openai_compatible",
            system,
            &user,
            self.config.max_retries,
        )?;
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
    config: &EmbeddingConfig,
    provider_name: &str,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let mut vectors = Vec::new();
    for batch in texts.chunks(config.batch_size) {
        let response_vectors =
            request_embedding_batch(config, provider_name, batch, config.max_retries)?;
        vectors.extend(response_vectors);
    }
    Ok(vectors)
}

fn request_embedding_batch(
    config: &EmbeddingConfig,
    provider_name: &str,
    texts: &[String],
    max_retries: u32,
) -> Result<Vec<Vec<f32>>> {
    let endpoint = embeddings_endpoint(&config.base_url);
    let payload = json!({
        "model": config.model,
        "input": texts,
    });
    let timeout = Duration::from_millis(config.timeout_ms);
    let agent = ureq::AgentBuilder::new().timeout(timeout).build();

    let mut attempt = 0_u32;
    loop {
        let mut request = agent
            .post(&endpoint)
            .set("content-type", "application/json");

        if let Some(api_key_env) = config.api_key_env.as_deref() {
            let api_key = std::env::var(api_key_env).map_err(|_| {
                KboltError::Inference(format!(
                    "embedding API key env var is not set: {api_key_env}"
                ))
            })?;
            request = request.set("authorization", &format!("Bearer {api_key}"));
        }

        match request.send_json(payload.clone()) {
            Ok(response) => {
                let parsed: EmbeddingResponseEnvelope = response.into_json().map_err(|err| {
                    KboltError::Inference(format!(
                        "failed to decode {provider_name} embedding response: {err}"
                    ))
                })?;
                let vectors = parsed.into_vectors(texts.len())?;
                return Ok(vectors);
            }
            Err(ureq::Error::Status(status, response)) => {
                let body = response
                    .into_string()
                    .unwrap_or_else(|_| "<unreadable body>".to_string());
                let can_retry = status >= 500 && attempt < max_retries;
                if can_retry {
                    attempt = attempt.saturating_add(1);
                    continue;
                }
                return Err(KboltError::Inference(format!(
                    "{provider_name} embedding request failed ({status}): {body}"
                ))
                .into());
            }
            Err(ureq::Error::Transport(err)) => {
                if attempt < max_retries {
                    attempt = attempt.saturating_add(1);
                    continue;
                }
                return Err(KboltError::Inference(format!(
                    "{provider_name} embedding transport error: {err}"
                ))
                .into());
            }
        }
    }
}

fn embeddings_endpoint(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/embeddings") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/embeddings")
    }
}

fn chat_completions_endpoint(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

fn request_chat_completion(
    config: &TextInferenceConfig,
    provider_name: &str,
    system_prompt: &str,
    user_prompt: &str,
    max_retries: u32,
) -> Result<String> {
    let endpoint = chat_completions_endpoint(&config.base_url);
    let payload = json!({
        "model": config.model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
    });
    let timeout = Duration::from_millis(config.timeout_ms);
    let agent = ureq::AgentBuilder::new().timeout(timeout).build();

    let mut attempt = 0_u32;
    loop {
        let mut request = agent
            .post(&endpoint)
            .set("content-type", "application/json");

        if let Some(api_key_env) = config.api_key_env.as_deref() {
            let api_key = std::env::var(api_key_env).map_err(|_| {
                KboltError::Inference(format!(
                    "inference API key env var is not set: {api_key_env}"
                ))
            })?;
            request = request.set("authorization", &format!("Bearer {api_key}"));
        }

        match request.send_json(payload.clone()) {
            Ok(response) => {
                let parsed: ChatCompletionResponse = response.into_json().map_err(|err| {
                    KboltError::Inference(format!(
                        "failed to decode {provider_name} chat completion response: {err}"
                    ))
                })?;
                let content = parsed.into_text()?;
                return Ok(content);
            }
            Err(ureq::Error::Status(status, response)) => {
                let body = response
                    .into_string()
                    .unwrap_or_else(|_| "<unreadable body>".to_string());
                let can_retry = status >= 500 && attempt < max_retries;
                if can_retry {
                    attempt = attempt.saturating_add(1);
                    continue;
                }
                return Err(KboltError::Inference(format!(
                    "{provider_name} chat completion request failed ({status}): {body}"
                ))
                .into());
            }
            Err(ureq::Error::Transport(err)) => {
                if attempt < max_retries {
                    attempt = attempt.saturating_add(1);
                    continue;
                }
                return Err(KboltError::Inference(format!(
                    "{provider_name} chat completion transport error: {err}"
                ))
                .into());
            }
        }
    }
}

fn parse_json_payload<T>(label: &str, content: &str) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_str(content).map_err(|err| {
        KboltError::Inference(format!("failed to parse {label} as JSON: {err}; payload={content}"))
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
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

impl ChatCompletionResponse {
    fn into_text(self) -> Result<String> {
        let Some(choice) = self.choices.into_iter().next() else {
            return Err(KboltError::Inference(
                "chat completion response is missing choices".to_string(),
            )
            .into());
        };

        extract_text(choice.message.content)
            .ok_or_else(|| {
                KboltError::Inference(
                    "chat completion response did not contain text content".to_string(),
                )
                .into()
            })
    }
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Value,
}

fn extract_text(value: Value) -> Option<String> {
    match value {
        Value::String(content) => Some(content),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    Value::String(segment) => text.push_str(&segment),
                    Value::Object(map) => {
                        if let Some(Value::String(segment)) = map.get("text") {
                            text.push_str(segment);
                        }
                    }
                    _ => {}
                }
            }

            if text.is_empty() { None } else { Some(text) }
        }
        Value::Object(map) => map.get("text").and_then(|item| item.as_str()).map(ToString::to_string),
        _ => None,
    }
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

    use super::*;

    fn base_config(provider: EmbeddingProvider, base_url: String) -> EmbeddingConfig {
        EmbeddingConfig {
            provider,
            model: "embed-model".to_string(),
            base_url,
            api_key_env: None,
            timeout_ms: 5_000,
            batch_size: 64,
            max_retries: 0,
        }
    }

    fn base_text_config(base_url: String) -> TextInferenceConfig {
        TextInferenceConfig {
            provider: TextInferenceProvider::OpenAiCompatible,
            model: "text-model".to_string(),
            base_url,
            api_key_env: None,
            timeout_ms: 5_000,
            max_retries: 0,
        }
    }

    fn serve_once(status_code: u16, body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        let payload = body.to_string();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept client");

            read_full_request(&mut stream);

            let status_line = match status_code {
                200 => "HTTP/1.1 200 OK",
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

    fn read_full_request(stream: &mut std::net::TcpStream) {
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
            return;
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
        let config = base_config(EmbeddingProvider::OpenAiCompatible, base_url);
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
        let config = base_config(EmbeddingProvider::OpenAiCompatible, base_url);
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
        let config = base_config(EmbeddingProvider::OpenAiCompatible, base_url);
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder
            .embed_batch(&["a".to_string()])
            .expect("embed");
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
        let config = base_config(EmbeddingProvider::Voyage, base_url);
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
        let config = EmbeddingConfig {
            api_key_env: Some("KBOLT_TEST_MISSING_API_KEY".to_string()),
            ..base_config(EmbeddingProvider::OpenAiCompatible, base_url)
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
        let config = base_text_config(base_url);
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
        let config = base_text_config(base_url);
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
}
