use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenizerRuntimeKind {
    RemoteLlamaCppServer,
    #[cfg(test)]
    Test,
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
