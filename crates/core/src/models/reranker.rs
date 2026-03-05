use std::collections::HashSet;

use crate::Result;
use crate::models::text::tokenize_terms;

pub(crate) trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>>;
}

#[derive(Debug, Default)]
pub(crate) struct HeuristicReranker;

impl Reranker for HeuristicReranker {
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let query_terms = tokenize_terms(query);
        if query_terms.is_empty() {
            return Ok(vec![0.0; docs.len()]);
        }

        let query_lower = query.trim().to_ascii_lowercase();
        let mut scores = Vec::with_capacity(docs.len());
        for doc in docs {
            let doc_terms = tokenize_terms(doc).into_iter().collect::<HashSet<_>>();
            let overlap = query_terms
                .iter()
                .filter(|term| doc_terms.contains(*term))
                .count() as f32
                / query_terms.len() as f32;
            let text_lower = doc.to_ascii_lowercase();
            let phrase_bonus = if !query_lower.is_empty() && text_lower.contains(&query_lower) {
                0.25
            } else {
                0.0
            };
            scores.push((0.75 * overlap + phrase_bonus).clamp(0.0, 1.0));
        }

        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::{HeuristicReranker, Reranker};

    #[test]
    fn rerank_scores_phrase_and_term_overlap() {
        let reranker = HeuristicReranker;
        let scores = reranker
            .rerank(
                "rust traits",
                &[
                    "Rust traits and impl examples".to_string(),
                    "Python decorators".to_string(),
                ],
            )
            .expect("rerank docs");

        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn rerank_returns_zeroes_for_blank_query_terms() {
        let reranker = HeuristicReranker;
        let scores = reranker
            .rerank("   ", &["anything".to_string(), "else".to_string()])
            .expect("rerank docs");

        assert_eq!(scores, vec![0.0, 0.0]);
    }
}
