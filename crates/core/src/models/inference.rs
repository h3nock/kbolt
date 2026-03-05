use std::sync::Arc;
use std::thread;
use std::time::Duration;

use kbolt_types::KboltError;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::{
    EmbeddingConfig, TextInferenceConfig, TextInferenceOutputMode,
    TextInferenceProvider,
};
use crate::models::expander::HeuristicExpander;
use crate::models::reranker::HeuristicReranker;
use crate::models::text::strip_json_fences;
use crate::models::{Embedder, Expander, Reranker};
use crate::Result;

#[derive(Debug, Clone)]
struct HttpApiEmbedder {
    client: HttpJsonClient,
    model: String,
    batch_size: usize,
}

#[derive(Debug, Clone)]
struct OpenAiCompatibleReranker {
    chat: ChatClient,
}

#[derive(Debug, Clone)]
struct OpenAiCompatibleExpander {
    chat: ChatClient,
}

const MAX_RETRY_AFTER_SECONDS: u64 = 30;

#[derive(Debug, Clone, Copy)]
enum HttpOperation {
    Embedding,
    ChatCompletion,
}

impl HttpOperation {
    fn label(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::ChatCompletion => "chat completion",
        }
    }
}

#[derive(Debug, Clone)]
struct HttpJsonClient {
    agent: ureq::Agent,
    base_url: String,
    api_key_env: Option<String>,
    max_retries: u32,
    api_key_scope: &'static str,
    provider_name: &'static str,
}

impl HttpJsonClient {
    fn new(
        base_url: &str,
        api_key_env: Option<&str>,
        timeout_ms: u64,
        max_retries: u32,
        api_key_scope: &'static str,
        provider_name: &'static str,
    ) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_millis(timeout_ms))
                .build(),
            base_url: base_url.to_string(),
            api_key_env: api_key_env.map(ToString::to_string),
            max_retries,
            api_key_scope,
            provider_name,
        }
    }

    fn post_json<T>(&self, endpoint_suffix: &str, payload: &Value, operation: HttpOperation) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let endpoint = resolve_endpoint(&self.base_url, endpoint_suffix);
        let mut attempt = 0_u32;

        loop {
            let mut request = self
                .agent
                .post(&endpoint)
                .set("content-type", "application/json");

            if let Some(api_key_env) = self.api_key_env.as_deref() {
                let api_key = std::env::var(api_key_env).map_err(|_| {
                    KboltError::Inference(format!(
                        "{} API key env var is not set: {api_key_env}",
                        self.api_key_scope
                    ))
                })?;
                request = request.set("authorization", &format!("Bearer {api_key}"));
            }

            match request.send_json(payload.clone()) {
                Ok(response) => {
                    let decoded = response.into_json().map_err(|err| {
                        KboltError::Inference(format!(
                            "failed to decode {} {} response: {err}",
                            self.provider_name,
                            operation.label()
                        ))
                    })?;
                    return Ok(decoded);
                }
                Err(ureq::Error::Status(status, response)) => {
                    let retry_after_secs =
                        parse_retry_after_seconds(response.header("retry-after"));
                    let body = response
                        .into_string()
                        .unwrap_or_else(|_| "<unreadable body>".to_string());
                    let can_retry = should_retry_status(status) && attempt < self.max_retries;
                    if can_retry {
                        attempt = attempt.saturating_add(1);
                        if let Some(wait_seconds) = retry_after_secs {
                            thread::sleep(Duration::from_secs(
                                wait_seconds.min(MAX_RETRY_AFTER_SECONDS),
                            ));
                        }
                        continue;
                    }

                    return Err(KboltError::Inference(format!(
                        "{} {} request failed ({status}): {body}",
                        self.provider_name,
                        operation.label()
                    ))
                    .into());
                }
                Err(ureq::Error::Transport(err)) => {
                    if attempt < self.max_retries {
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    return Err(KboltError::Inference(format!(
                        "{} {} transport error: {err}",
                        self.provider_name,
                        operation.label()
                    ))
                    .into());
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ChatClient {
    http: HttpJsonClient,
    model: String,
    output_mode: TextInferenceOutputMode,
}

impl ChatClient {
    fn new(
        base_url: &str,
        api_key_env: Option<&str>,
        timeout_ms: u64,
        max_retries: u32,
        model: &str,
        output_mode: TextInferenceOutputMode,
        provider_name: &'static str,
    ) -> Self {
        Self {
            http: HttpJsonClient::new(
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
                "inference",
                provider_name,
            ),
            model: model.to_string(),
            output_mode,
        }
    }

    fn complete(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let payload = build_chat_payload(
            &self.model,
            system_prompt,
            user_prompt,
            self.output_mode.clone(),
        );

        let response: ChatCompletionResponse =
            self.http
                .post_json("chat/completions", &payload, HttpOperation::ChatCompletion)?;
        let content = response.into_text()?;
        let normalized = match self.output_mode {
            TextInferenceOutputMode::JsonObject => content.trim(),
            TextInferenceOutputMode::Text => strip_json_fences(&content),
        };
        Ok(normalized.to_string())
    }
}

fn build_chat_payload(
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
    output_mode: TextInferenceOutputMode,
) -> Value {
    let mut payload = json!({
        "model": model,
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
    if output_mode == TextInferenceOutputMode::JsonObject {
        payload["response_format"] = json!({ "type": "json_object" });
    }
    payload
}

pub(crate) fn build_embedder(
    config: Option<&EmbeddingConfig>,
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
            EmbeddingConfig::LocalOnnx { .. } => {
                return Err(
                    KboltError::Inference("local_onnx embedder is not implemented yet".to_string())
                        .into(),
                )
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

pub(crate) fn build_reranker(config: Option<&TextInferenceConfig>) -> Result<Arc<dyn Reranker>> {
    let reranker: Arc<dyn Reranker> = match config {
        Some(config) => match &config.provider {
            TextInferenceProvider::OpenAiCompatible {
                output_mode,
                model,
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
            } => Arc::new(OpenAiCompatibleReranker {
                chat: ChatClient::new(
                    base_url,
                    api_key_env.as_deref(),
                    *timeout_ms,
                    *max_retries,
                    model,
                    output_mode.clone(),
                    "openai_compatible",
                ),
            }),
            TextInferenceProvider::LocalLlama { .. } => {
                return Err(KboltError::Inference(
                    "local_llama reranker is not implemented yet".to_string(),
                )
                .into())
            }
        },
        None => Arc::new(HeuristicReranker),
    };
    Ok(reranker)
}

pub(crate) fn build_expander(config: Option<&TextInferenceConfig>) -> Result<Arc<dyn Expander>> {
    let expander: Arc<dyn Expander> = match config {
        Some(config) => match &config.provider {
            TextInferenceProvider::OpenAiCompatible {
                output_mode,
                model,
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
            } => Arc::new(OpenAiCompatibleExpander {
                chat: ChatClient::new(
                    base_url,
                    api_key_env.as_deref(),
                    *timeout_ms,
                    *max_retries,
                    model,
                    output_mode.clone(),
                    "openai_compatible",
                ),
            }),
            TextInferenceProvider::LocalLlama { .. } => {
                return Err(KboltError::Inference(
                    "local_llama expander is not implemented yet".to_string(),
                )
                .into())
            }
        },
        None => Arc::new(HeuristicExpander),
    };
    Ok(expander)
}

impl Embedder for HttpApiEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        embed_with_http_api(&self.client, &self.model, self.batch_size, texts)
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

fn resolve_endpoint(base_url: &str, suffix: &str) -> String {
    let trimmed_base = base_url.trim_end_matches('/');
    let normalized_suffix = suffix.trim_start_matches('/');
    if trimmed_base.ends_with(normalized_suffix) {
        trimmed_base.to_string()
    } else {
        format!("{trimmed_base}/{normalized_suffix}")
    }
}

fn should_retry_status(status: u16) -> bool {
    status == 429 || status >= 500
}

fn parse_retry_after_seconds(header_value: Option<&str>) -> Option<u64> {
    let raw = header_value?.trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(seconds) = raw.parse::<u64>() {
        return Some(seconds);
    }

    let retry_at = httpdate::parse_http_date(raw).ok()?;
    let now = std::time::SystemTime::now();
    let seconds = match retry_at.duration_since(now) {
        Ok(duration) => duration.as_secs(),
        Err(_) => 0,
    };
    Some(seconds)
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

        extract_text(choice.message.content).ok_or_else(|| {
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

            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        Value::Object(map) => map
            .get("text")
            .and_then(|item| item.as_str())
            .map(ToString::to_string),
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
