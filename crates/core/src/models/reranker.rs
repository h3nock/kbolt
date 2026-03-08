use crate::Result;

pub(crate) trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>>;
}
