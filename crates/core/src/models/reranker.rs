use crate::Result;

pub(crate) trait Reranker: Send + Sync {
    /// Returns one relevance score per candidate document, in the same order as the input.
    ///
    /// Higher scores mean more relevant for this rerank call. Scores only need to be
    /// comparable within the current query; callers must not assume cross-query or
    /// cross-model calibration.
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>>;
}
