use std::sync::Arc;

use kbolt_types::KboltError;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::models::gateway::GatewayProviderKind;
use crate::models::http::{HttpJsonClient, HttpOperation};
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenizerRuntimeKind {
    RemoteLlamaCppServer,
    #[cfg(test)]
    Test,
}

#[derive(Debug, Clone)]
struct RemoteLlamaCppTokenizerRuntime {
    client: HttpJsonClient,
    endpoint_suffix: &'static str,
}

pub(crate) trait TokenizerRuntime: Send + Sync {
    fn kind(&self) -> TokenizerRuntimeKind;

    fn count_embedding_tokens(&self, text: &str) -> Result<usize>;

    fn count_embedding_tokens_batch(&self, texts: &[&str]) -> Result<Vec<usize>> {
        texts
            .iter()
            .map(|text| self.count_embedding_tokens(text))
            .collect()
    }
}

pub(crate) fn build_embedding_tokenizer_runtime(
    kind: GatewayProviderKind,
    remote_client: Option<HttpJsonClient>,
) -> Result<Option<Arc<dyn TokenizerRuntime>>> {
    match kind {
        GatewayProviderKind::LlamaCppServer => {
            let client = remote_client.ok_or_else(|| {
                KboltError::Internal(
                    "llama.cpp embedding tokenizer runtime requires an HTTP client".to_string(),
                )
            })?;
            Ok(Some(Arc::new(RemoteLlamaCppTokenizerRuntime {
                client,
                endpoint_suffix: llama_cpp_tokenize_endpoint_suffix(),
            })))
        }
        GatewayProviderKind::OpenAiCompatible => Ok(None),
    }
}

impl TokenizerRuntime for RemoteLlamaCppTokenizerRuntime {
    fn kind(&self) -> TokenizerRuntimeKind {
        TokenizerRuntimeKind::RemoteLlamaCppServer
    }

    fn count_embedding_tokens(&self, text: &str) -> Result<usize> {
        let payload = json!({
            "content": text,
            "add_special": true,
        });
        let response = self.client.post_json::<TokenizeResponseEnvelope>(
            self.endpoint_suffix,
            &payload,
            HttpOperation::Tokenize,
        )?;
        Ok(response.token_count())
    }
}

fn llama_cpp_tokenize_endpoint_suffix() -> &'static str {
    "tokenize"
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenizeResponseEnvelope {
    Wrapped { tokens: Vec<Value> },
    Direct(Vec<Value>),
}

impl TokenizeResponseEnvelope {
    fn token_count(self) -> usize {
        let tokens = match self {
            Self::Wrapped { tokens } => tokens,
            Self::Direct(tokens) => tokens,
        };
        tokens.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_compatible_embedding_tokenizer_is_unconfigured() {
        let runtime =
            build_embedding_tokenizer_runtime(GatewayProviderKind::OpenAiCompatible, None)
                .expect("build tokenizer runtime");

        assert!(runtime.is_none());
    }

    #[test]
    fn llama_cpp_embedding_tokenizer_requires_remote_client_for_now() {
        let err = match build_embedding_tokenizer_runtime(GatewayProviderKind::LlamaCppServer, None)
        {
            Ok(_) => panic!("missing remote client should fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("requires an HTTP client"),
            "unexpected error: {err}"
        );
    }
}
