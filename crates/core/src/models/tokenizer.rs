use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::models::gguf_tokenizer::LlamaSpmGgufTokenizerRuntime;
use crate::models::http::{HttpJsonClient, HttpOperation};
use crate::models::tiktoken_tokenizer::{TiktokenEncoding, TiktokenTokenizerRuntime};
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum TokenizerRuntimeKind {
    LlamaSpmGgufEmbedded,
    LlamaCppHttpTokenize,
    Tiktoken,
    #[cfg(test)]
    Test,
}

#[derive(Debug, Clone)]
struct LlamaCppHttpTokenizeRuntime {
    client: HttpJsonClient,
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

pub(crate) enum EmbeddingTokenizerSpec<'a> {
    NoLocalTokenizer,
    LlamaSpmGguf { model_path: &'a Path },
    LlamaCppHttpTokenize { client: HttpJsonClient },
    Tiktoken { encoding: TiktokenEncoding },
}

pub(crate) fn build_embedding_tokenizer_runtime(
    spec: EmbeddingTokenizerSpec<'_>,
) -> Result<Option<Arc<dyn TokenizerRuntime>>> {
    match spec {
        EmbeddingTokenizerSpec::NoLocalTokenizer => Ok(None),
        EmbeddingTokenizerSpec::LlamaSpmGguf { model_path } => Ok(Some(Arc::new(
            LlamaSpmGgufTokenizerRuntime::from_path(model_path)?,
        ))),
        EmbeddingTokenizerSpec::LlamaCppHttpTokenize { client } => {
            Ok(Some(Arc::new(LlamaCppHttpTokenizeRuntime { client })))
        }
        EmbeddingTokenizerSpec::Tiktoken { encoding } => {
            Ok(Some(Arc::new(TiktokenTokenizerRuntime::new(encoding))))
        }
    }
}

impl TokenizerRuntime for LlamaCppHttpTokenizeRuntime {
    fn kind(&self) -> TokenizerRuntimeKind {
        TokenizerRuntimeKind::LlamaCppHttpTokenize
    }

    fn count_embedding_tokens(&self, text: &str) -> Result<usize> {
        let payload = json!({
            "content": text,
            "add_special": true,
            "parse_special": true,
        });
        let response = self.client.post_json::<TokenizeResponseEnvelope>(
            "tokenize",
            &payload,
            HttpOperation::Tokenize,
        )?;
        Ok(response.token_count())
    }
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
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn openai_compatible_embedding_tokenizer_has_no_local_runtime() {
        let runtime = build_embedding_tokenizer_runtime(EmbeddingTokenizerSpec::NoLocalTokenizer)
            .expect("build tokenizer runtime");

        assert!(runtime.is_none());
    }

    #[test]
    fn llama_spm_gguf_tokenizer_spec_loads_from_model_path() {
        let dir = tempdir().expect("tempdir");
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"not gguf").expect("write invalid gguf");

        let err = match build_embedding_tokenizer_runtime(EmbeddingTokenizerSpec::LlamaSpmGguf {
            model_path: &model_path,
        }) {
            Ok(_) => panic!("invalid GGUF tokenizer should fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("not a GGUF file"),
            "unexpected error: {err}"
        );
    }
}
