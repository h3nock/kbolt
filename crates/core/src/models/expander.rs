use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpansionRoute {
    Both,
    KeywordOnly,
    DenseOnly,
}

impl ExpansionRoute {
    pub(crate) fn includes_keyword(self) -> bool {
        matches!(self, Self::Both | Self::KeywordOnly)
    }

    pub(crate) fn includes_dense(self) -> bool {
        matches!(self, Self::Both | Self::DenseOnly)
    }

    pub(crate) fn merged_with(self, other: Self) -> Self {
        match (self, other) {
            (Self::Both, _) | (_, Self::Both) => Self::Both,
            (Self::KeywordOnly, Self::DenseOnly) | (Self::DenseOnly, Self::KeywordOnly) => {
                Self::Both
            }
            (route, _) => route,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpandedQuery {
    pub text: String,
    pub route: ExpansionRoute,
}

pub(crate) fn normalize_query_text(query: &str) -> String {
    query.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) trait Expander: Send + Sync {
    /// Returns ordered, unique generated queries for the given query.
    ///
    /// Implementations must not return the original query. The engine owns that baseline query
    /// and routes it through both keyword and dense retrieval when available.
    fn expand(&self, query: &str) -> Result<Vec<ExpandedQuery>>;
}
