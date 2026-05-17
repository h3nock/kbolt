use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbeddingInputKind {
    Query,
    Document,
}

pub(crate) const DEFAULT_DOCUMENT_BATCH_WINDOW: usize = 64;

pub(crate) trait Embedder: Send + Sync {
    fn embed_batch(&self, kind: EmbeddingInputKind, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    fn preferred_document_batch_window(&self) -> usize {
        DEFAULT_DOCUMENT_BATCH_WINDOW
    }
}

pub(crate) trait EmbeddingDocumentSizer: Send + Sync {
    fn count_document_tokens(&self, text: &str) -> Result<usize>;
}
