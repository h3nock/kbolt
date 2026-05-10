use crate::models::tokenizer::{TokenizerRuntime, TokenizerRuntimeKind};
use crate::Result;

#[derive(Debug)]
pub(crate) struct TiktokenTokenizerRuntime {
    encoding: TiktokenEncoding,
}

impl TiktokenTokenizerRuntime {
    pub(crate) fn new(encoding: TiktokenEncoding) -> Self {
        Self { encoding }
    }
}

impl TokenizerRuntime for TiktokenTokenizerRuntime {
    fn kind(&self) -> TokenizerRuntimeKind {
        TokenizerRuntimeKind::Tiktoken
    }

    fn count_embedding_tokens(&self, text: &str) -> Result<usize> {
        Ok(match self.encoding {
            TiktokenEncoding::Cl100kBase => {
                tiktoken_rs::cl100k_base_singleton().count_ordinary(text)
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TiktokenEncoding {
    Cl100kBase,
}

impl TiktokenEncoding {
    pub(crate) fn for_official_openai_embedding_model(model: &str) -> Option<Self> {
        match model.trim() {
            "text-embedding-3-small" | "text-embedding-3-large" | "text-embedding-ada-002" => {
                Some(Self::Cl100kBase)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_openai_embedding_models_use_cl100k_base() {
        for model in [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ] {
            assert_eq!(
                TiktokenEncoding::for_official_openai_embedding_model(model),
                Some(TiktokenEncoding::Cl100kBase),
                "unexpected encoding for {model}"
            );
        }
    }

    #[test]
    fn non_embedding_or_unknown_models_do_not_get_inferred_encoding() {
        for model in [
            "gpt-5-mini",
            "embed-model",
            "qwen3-embedding",
            "text-embedding-3-large-custom",
        ] {
            assert_eq!(
                TiktokenEncoding::for_official_openai_embedding_model(model),
                None,
                "unexpected inferred encoding for {model}"
            );
        }
    }

    #[test]
    fn cl100k_runtime_counts_embedding_text_as_ordinary_text() {
        let runtime = TiktokenTokenizerRuntime::new(TiktokenEncoding::Cl100kBase);

        assert_eq!(runtime.count_embedding_tokens("").unwrap(), 0);
        assert_eq!(runtime.count_embedding_tokens("hello world").unwrap(), 2);
        assert_eq!(
            runtime
                .count_embedding_tokens("tiktoken is great!")
                .unwrap(),
            6
        );
        assert_eq!(runtime.count_embedding_tokens("<|endoftext|>").unwrap(), 7);
    }
}
