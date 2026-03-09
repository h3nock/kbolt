use crate::Result;

pub(crate) trait Expander: Send + Sync {
    /// Returns ordered, unique query variants for the given query.
    ///
    /// The first variant must be the normalized original query. Implementations should not
    /// return empty variants.
    fn expand(&self, query: &str) -> Result<Vec<String>>;
}
