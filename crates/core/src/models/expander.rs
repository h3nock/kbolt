use std::collections::HashSet;

use crate::models::text::tokenize_terms;
use crate::Result;

pub(crate) trait Expander: Send + Sync {
    fn expand(&self, query: &str) -> Result<Vec<String>>;
}

#[derive(Debug, Default)]
pub(crate) struct HeuristicExpander;

impl Expander for HeuristicExpander {
    fn expand(&self, query: &str) -> Result<Vec<String>> {
        let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        let lexical = tokenize_terms(&normalized).join(" ");
        let semantic = format!("explain {normalized}");
        let hyde = format!("this document explains {normalized} with practical details");

        let mut seen = HashSet::new();
        let mut variants = Vec::new();
        for variant in [normalized, lexical, semantic, hyde] {
            let trimmed = variant.trim();
            if trimmed.is_empty() {
                continue;
            }

            let key = trimmed.to_ascii_lowercase();
            if seen.insert(key) {
                variants.push(trimmed.to_string());
            }
        }

        Ok(variants)
    }
}

#[cfg(test)]
mod tests {
    use super::{Expander, HeuristicExpander};

    #[test]
    fn expand_returns_unique_variants_in_stable_order() {
        let expander = HeuristicExpander;
        let variants = expander
            .expand("  Rust   trait object  ")
            .expect("expand query");

        assert_eq!(
            variants,
            vec![
                "Rust trait object".to_string(),
                "explain Rust trait object".to_string(),
                "this document explains Rust trait object with practical details".to_string(),
            ]
        );
    }

    #[test]
    fn expand_returns_empty_for_blank_query() {
        let expander = HeuristicExpander;
        let variants = expander.expand("   ").expect("expand blank query");
        assert!(variants.is_empty());
    }
}
