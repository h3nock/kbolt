use crate::Result;

pub(crate) fn normalize_query_text(query: &str) -> String {
    query.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) trait Expander: Send + Sync {
    /// Returns ordered, unique generated queries for the given query.
    ///
    /// Implementations must not return the original query. The engine owns that baseline query
    /// and routes it through keyword and dense retrieval when available.
    fn expand(&self, query: &str, max_variants: usize) -> Result<Vec<String>>;
}
