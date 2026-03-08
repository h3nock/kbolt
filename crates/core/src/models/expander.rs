use crate::Result;

pub(crate) trait Expander: Send + Sync {
    fn expand(&self, query: &str) -> Result<Vec<String>>;
}
