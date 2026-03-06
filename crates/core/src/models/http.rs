use std::thread;
use std::time::Duration;

use kbolt_types::KboltError;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::Result;

const MAX_RETRY_AFTER_SECONDS: u64 = 30;

#[derive(Debug, Clone, Copy)]
pub(super) enum HttpOperation {
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
pub(super) struct HttpJsonClient {
    agent: ureq::Agent,
    base_url: String,
    api_key_env: Option<String>,
    max_retries: u32,
    api_key_scope: &'static str,
    provider_name: &'static str,
}

impl HttpJsonClient {
    pub(super) fn new(
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

    pub(super) fn post_json<T>(
        &self,
        endpoint_suffix: &str,
        payload: &Value,
        operation: HttpOperation,
    ) -> Result<T>
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

pub(super) fn parse_retry_after_seconds(header_value: Option<&str>) -> Option<u64> {
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
