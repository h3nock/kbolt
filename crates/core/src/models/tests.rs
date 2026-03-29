use std::collections::HashMap;
use std::io::Read;
use std::io::Write;
use std::net::TcpListener;
use std::path::PathBuf;

use crate::config::{
    ChunkingConfig, Config, EmbedderRoleConfig, ProviderOperation, ProviderProfileConfig,
    RankingConfig, ReapingConfig, RoleBindingsConfig,
};
use crate::models::status;

fn base_config() -> Config {
    Config {
        config_dir: PathBuf::from("/tmp/kbolt-config"),
        cache_dir: PathBuf::from("/tmp/kbolt-cache"),
        default_space: None,
        providers: HashMap::new(),
        roles: RoleBindingsConfig::default(),
        reaping: ReapingConfig { days: 7 },
        chunking: ChunkingConfig::default(),
        ranking: RankingConfig::default(),
    }
}

fn serve_status(status_code: u16, body: &str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().expect("server address");
    let payload = body.to_string();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept client");
        let mut request = [0_u8; 1024];
        let _ = stream.read(&mut request);
        let status_line = match status_code {
            200 => "HTTP/1.1 200 OK",
            401 => "HTTP/1.1 401 Unauthorized",
            404 => "HTTP/1.1 404 Not Found",
            other => panic!("unsupported status code in test server: {other}"),
        };
        let response = format!(
            "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            payload.len(),
            payload
        );
        stream
            .write_all(response.as_bytes())
            .expect("write response");
    });

    format!("http://{addr}")
}

#[test]
fn status_marks_unbound_roles_as_unconfigured() {
    let config = base_config();
    let status = status(&config).expect("read status");

    assert!(!status.embedder.configured);
    assert!(!status.embedder.ready);
    assert_eq!(status.embedder.profile, None);
    assert!(!status.reranker.configured);
    assert!(!status.expander.configured);
}

#[test]
fn status_marks_http_endpoint_as_ready_when_server_responds() {
    let mut config = base_config();
    config.providers.insert(
        "local_embed".to_string(),
        ProviderProfileConfig::LlamaCppServer {
            operation: ProviderOperation::Embedding,
            base_url: serve_status(200, r#"{"ok":true}"#),
            model: "embeddinggemma".to_string(),
            timeout_ms: 5_000,
            max_retries: 0,
        },
    );
    config.roles.embedder = Some(EmbedderRoleConfig {
        provider: "local_embed".to_string(),
        batch_size: 16,
    });

    let status = status(&config).expect("read status");
    assert!(status.embedder.configured);
    assert!(status.embedder.ready);
    assert_eq!(status.embedder.profile.as_deref(), Some("local_embed"));
    assert_eq!(status.embedder.kind.as_deref(), Some("llama_cpp_server"));
    assert_eq!(status.embedder.operation.as_deref(), Some("embedding"));
    assert_eq!(status.embedder.model.as_deref(), Some("embeddinggemma"));
    assert!(status
        .embedder
        .endpoint
        .as_deref()
        .unwrap()
        .starts_with("http://127.0.0.1:"));
    assert_eq!(status.embedder.issue, None);
}

#[test]
fn status_marks_missing_api_key_as_not_ready() {
    let mut config = base_config();
    config.providers.insert(
        "remote_embed".to_string(),
        ProviderProfileConfig::OpenAiCompatible {
            operation: ProviderOperation::Embedding,
            base_url: "https://api.openai.com/v1".to_string(),
            model: "text-embedding-3-large".to_string(),
            api_key_env: Some("KBOLT_TEST_MISSING_STATUS_KEY".to_string()),
            timeout_ms: 5_000,
            max_retries: 0,
        },
    );
    config.roles.embedder = Some(EmbedderRoleConfig {
        provider: "remote_embed".to_string(),
        batch_size: 16,
    });
    std::env::remove_var("KBOLT_TEST_MISSING_STATUS_KEY");

    let status = status(&config).expect("read status");
    assert!(status.embedder.configured);
    assert!(!status.embedder.ready);
    assert!(status
        .embedder
        .issue
        .as_deref()
        .expect("issue should exist")
        .contains("KBOLT_TEST_MISSING_STATUS_KEY"));
}
