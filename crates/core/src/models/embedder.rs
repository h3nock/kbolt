use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbeddingInputKind {
    Query,
    Document,
}

pub(crate) trait Embedder: Send + Sync {
    fn embed_batch(&self, kind: EmbeddingInputKind, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub(crate) trait EmbeddingDocumentSizer: Send + Sync {
    fn count_document_tokens(&self, text: &str) -> Result<usize>;
}
