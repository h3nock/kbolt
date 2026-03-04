use std::sync::Arc;
use std::time::Duration;

use kbolt_types::KboltError;
use serde::Deserialize;
use serde_json::json;

use crate::config::{EmbeddingConfig, EmbeddingProvider};
use crate::models::Embedder;
use crate::Result;

#[derive(Debug, Clone)]
struct OpenAiCompatibleEmbedder {
    config: EmbeddingConfig,
}

#[derive(Debug, Clone)]
struct VoyageEmbedder {
    config: EmbeddingConfig,
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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingResponseEnvelope {
    OpenAiLike { data: Vec<EmbeddingItem> },
    VoyageLike { embeddings: Vec<Vec<f32>> },
}

impl EmbeddingResponseEnvelope {
    fn into_vectors(self, expected_len: usize) -> Result<Vec<Vec<f32>>> {
        let mut vectors = match self {
            Self::OpenAiLike { mut data } => {
                data.sort_by_key(|item| item.index);
                data.into_iter()
                    .map(|item| item.embedding)
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
    index: usize,
    embedding: Vec<f32>,
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
}
