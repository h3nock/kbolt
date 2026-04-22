use std::fmt;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use kbolt_types::KboltError;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{RecoveryNoticeSink, Result};

const MAX_RETRY_AFTER_SECONDS: u64 = 30;

#[derive(Debug, Clone, Copy)]
pub(super) enum HttpOperation {
    Embedding,
    Reranking,
    ChatCompletion,
    Tokenize,
}

impl HttpOperation {
    fn label(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Reranking => "reranking",
            Self::ChatCompletion => "chat completion",
            Self::Tokenize => "tokenize",
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
    provider_label: String,
    transport_recovery: Option<HttpTransportRecovery>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HttpEndpointReadiness {
    pub ready: bool,
    pub issue: Option<String>,
}

#[derive(Clone)]
pub(super) struct HttpTransportRecovery {
    label: String,
    callback: Arc<dyn Fn() -> Result<()> + Send + Sync>,
    notice: Option<RecoveryNoticeSink>,
}

impl fmt::Debug for HttpTransportRecovery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpTransportRecovery")
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

impl HttpTransportRecovery {
    pub(super) fn new(
        label: impl Into<String>,
        callback: Arc<dyn Fn() -> Result<()> + Send + Sync>,
        notice: Option<RecoveryNoticeSink>,
    ) -> Self {
        Self {
            label: label.into(),
            callback,
            notice,
        }
    }

    fn recover(&self) -> Result<()> {
        (self.callback)()
    }

    fn notify_restarting(&self) {
        if let Some(notice) = self.notice.as_ref() {
            notice(&format!(
                "{} unreachable; attempting automatic restart...",
                self.label
            ));
        }
    }

    fn notify_retrying(&self, operation: HttpOperation) {
        if let Some(notice) = self.notice.as_ref() {
            notice(&format!(
                "{} restarted; retrying {} request",
                self.label,
                operation.label()
            ));
        }
    }

    #[cfg(test)]
    pub(super) fn label(&self) -> &str {
        &self.label
    }
}

impl HttpJsonClient {
    pub(super) fn new(
        base_url: &str,
        api_key_env: Option<&str>,
        timeout_ms: u64,
        max_retries: u32,
        api_key_scope: &'static str,
        provider_label: &str,
        transport_recovery: Option<HttpTransportRecovery>,
    ) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_millis(timeout_ms))
                .build(),
            base_url: base_url.to_string(),
            api_key_env: api_key_env.map(ToString::to_string),
            max_retries,
            api_key_scope,
            provider_label: provider_label.to_string(),
            transport_recovery,
        }
    }

    pub(super) fn probe_readiness(&self) -> HttpEndpointReadiness {
        let endpoint = self.base_url.trim_end_matches('/').to_string();
        let mut request = self.agent.get(&endpoint);

        if let Some(api_key_env) = self.api_key_env.as_deref() {
            let api_key = match std::env::var(api_key_env) {
                Ok(value) => value,
                Err(_) => {
                    return HttpEndpointReadiness {
                        ready: false,
                        issue: Some(format!(
                            "{} API key env var is not set: {api_key_env}",
                            self.api_key_scope
                        )),
                    };
                }
            };
            request = request.set("authorization", &format!("Bearer {api_key}"));
        }

        match request.call() {
            Ok(_) | Err(ureq::Error::Status(_, _)) => HttpEndpointReadiness {
                ready: true,
                issue: None,
            },
            Err(ureq::Error::Transport(err)) => HttpEndpointReadiness {
                ready: false,
                issue: Some(format!(
                    "{} endpoint is unreachable: {err}",
                    self.provider_label
                )),
            },
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
        let mut recovery_attempted = false;

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
                            self.provider_label,
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
                        self.provider_label,
                        operation.label()
                    ))
                    .into());
                }
                Err(ureq::Error::Transport(err)) => {
                    if attempt < self.max_retries {
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    if !recovery_attempted {
                        if let Some(recovery) = self.transport_recovery.as_ref() {
                            recovery_attempted = true;
                            recovery.notify_restarting();
                            recovery.recover().map_err(|recovery_err| {
                                KboltError::Inference(format!(
                                    "{} {} transport error: {err}; automatic restart for {} failed: {}",
                                    self.provider_label,
                                    operation.label(),
                                    recovery.label,
                                    recovery_err
                                ))
                            })?;
                            recovery.notify_retrying(operation);
                            attempt = 0;
                            continue;
                        }
                    }
                    return Err(KboltError::Inference(format!(
                        "{} {} transport error{}: {err}",
                        self.provider_label,
                        operation.label(),
                        if recovery_attempted {
                            " after automatic restart attempt"
                        } else {
                            ""
                        }
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

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use serde::Deserialize;
    use serde_json::json;

    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct OkResponse {
        ok: bool,
    }

    fn read_raw_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
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
            return raw;
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
            raw.extend_from_slice(&chunk[..read]);
            remaining = remaining.saturating_sub(read);
        }

        raw
    }

    #[test]
    fn transport_recovery_retries_once_after_restart() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve port");
        let addr = listener.local_addr().expect("listener address");
        drop(listener);

        let recovery_calls = Arc::new(AtomicUsize::new(0));
        let notices = Arc::new(Mutex::new(Vec::<String>::new()));
        let recovery = {
            let recovery_calls = Arc::clone(&recovery_calls);
            let notices = Arc::clone(&notices);
            HttpTransportRecovery::new(
                "embedder",
                Arc::new(move || {
                    recovery_calls.fetch_add(1, Ordering::SeqCst);
                    let listener = TcpListener::bind(addr).expect("restart listener");
                    std::thread::spawn(move || {
                        let (mut stream, _) = listener.accept().expect("accept retried request");
                        let _ = read_raw_request(&mut stream);
                        let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}";
                        stream
                            .write_all(response.as_bytes())
                            .expect("write success response");
                    });
                    Ok(())
                }),
                Some(Arc::new(move |line| {
                    notices.lock().expect("notices lock").push(line.to_string());
                })),
            )
        };
        let client = HttpJsonClient::new(
            &format!("http://{addr}"),
            None,
            5_000,
            0,
            "embedding",
            "llama_cpp_server",
            Some(recovery),
        );

        let response: OkResponse = client
            .post_json(
                "v1/embeddings",
                &json!({"input":["hello"]}),
                HttpOperation::Embedding,
            )
            .expect("request should recover");

        assert_eq!(response, OkResponse { ok: true });
        assert_eq!(recovery_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            notices.lock().expect("notices lock").as_slice(),
            [
                "embedder unreachable; attempting automatic restart...",
                "embedder restarted; retrying embedding request",
            ]
        );
    }

    #[test]
    fn transport_recovery_failure_returns_restart_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve port");
        let addr = listener.local_addr().expect("listener address");
        drop(listener);

        let client = HttpJsonClient::new(
            &format!("http://{addr}"),
            None,
            5_000,
            0,
            "embedding",
            "llama_cpp_server",
            Some(HttpTransportRecovery::new(
                "embedder",
                Arc::new(|| Err(KboltError::Inference("restart failed".to_string()).into())),
                None,
            )),
        );

        let err = client
            .post_json::<OkResponse>(
                "v1/embeddings",
                &json!({"input":["hello"]}),
                HttpOperation::Embedding,
            )
            .expect_err("request should fail when restart fails");
        let message = err.to_string();
        assert!(message.contains("automatic restart"));
        assert!(message.contains("restart failed"));
    }

    #[test]
    fn transport_recovery_is_attempted_only_once() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve port");
        let addr = listener.local_addr().expect("listener address");
        drop(listener);

        let recovery_calls = Arc::new(AtomicUsize::new(0));
        let client = HttpJsonClient::new(
            &format!("http://{addr}"),
            None,
            5_000,
            0,
            "embedding",
            "llama_cpp_server",
            Some(HttpTransportRecovery::new(
                "embedder",
                Arc::new({
                    let recovery_calls = Arc::clone(&recovery_calls);
                    move || {
                        recovery_calls.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    }
                }),
                None,
            )),
        );

        let err = client
            .post_json::<OkResponse>(
                "v1/embeddings",
                &json!({"input":["hello"]}),
                HttpOperation::Embedding,
            )
            .expect_err("request should still fail");
        let message = err.to_string();

        assert_eq!(recovery_calls.load(Ordering::SeqCst), 1);
        assert!(message.contains("after automatic restart attempt"));
    }
}
