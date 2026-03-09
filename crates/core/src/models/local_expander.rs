use std::collections::HashSet;
use std::sync::Arc;

use crate::models::completion::CompletionClient;
use crate::models::Expander;
use crate::Result;

use super::local_llama::LocalLlamaClient;

pub(super) struct LocalLlamaExpander {
    client: Arc<LocalLlamaClient>,
}

impl LocalLlamaExpander {
    pub(super) fn new(client: Arc<LocalLlamaClient>) -> Self {
        Self { client }
    }
}

impl Expander for LocalLlamaExpander {
    fn expand(&self, query: &str) -> Result<Vec<String>> {
        let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        let user_prompt = format!("/no_think Expand this search query: {normalized}");
        let content = self.client.complete("", &user_prompt)?;
        let raw_variants = parse_expansion_output(&content);

        let mut seen = HashSet::new();
        let mut variants = Vec::new();
        seen.insert(normalized.to_ascii_lowercase());
        variants.push(normalized);

        for variant in raw_variants {
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

/// Parse expansion output with flexible multi-format support.
///
/// Tries in order:
/// 1. Structured lines: `lex: ...`, `vec: ...`, `hyde: ...` — extract values
/// 2. JSON: `{"variants":[...]}` or `[...]`
/// 3. Plain text: split by newlines and take non-empty lines
fn parse_expansion_output(content: &str) -> Vec<String> {
    let structured = parse_structured_lines(content);
    if !structured.is_empty() {
        return structured;
    }

    if let Some(json_variants) = parse_json_variants(content) {
        return json_variants;
    }

    // Fall back to plain text lines.
    content
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect()
}

/// Parse `lex: ...`, `vec: ...`, `hyde: ...` structured output lines.
fn parse_structured_lines(content: &str) -> Vec<String> {
    let prefixes = ["lex:", "vec:", "hyde:"];
    let mut variants = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        for prefix in &prefixes {
            if let Some(value) = trimmed.strip_prefix(prefix) {
                let value = value.trim();
                if !value.is_empty() {
                    variants.push(value.to_string());
                }
            }
        }
    }

    variants
}

/// Try to parse JSON variants from content.
fn parse_json_variants(content: &str) -> Option<Vec<String>> {
    let trimmed = content.trim();

    // Try {"variants": [...]}
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(arr) = parsed.get("variants").and_then(|v| v.as_array()) {
            return Some(
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect(),
            );
        }
        if let Some(arr) = parsed.as_array() {
            return Some(
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect(),
            );
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_structured_lines_extracts_values() {
        let content = "lex: BM25 full text search\nvec: semantic vector retrieval\nhyde: A document about BM25 ranking algorithms";
        let variants = parse_expansion_output(content);
        assert_eq!(
            variants,
            vec![
                "BM25 full text search",
                "semantic vector retrieval",
                "A document about BM25 ranking algorithms",
            ]
        );
    }

    #[test]
    fn parse_structured_lines_handles_mixed_with_other_lines() {
        let content = "Here are expansions:\nlex: keyword search\nvec: dense retrieval\n";
        let variants = parse_expansion_output(content);
        assert_eq!(variants, vec!["keyword search", "dense retrieval"]);
    }

    #[test]
    fn parse_json_wrapped_variants() {
        let content = r#"{"variants":["trait object rust","explain rust traits"]}"#;
        let variants = parse_expansion_output(content);
        assert_eq!(variants, vec!["trait object rust", "explain rust traits"]);
    }

    #[test]
    fn parse_json_array_variants() {
        let content = r#"["keyword search","semantic search"]"#;
        let variants = parse_expansion_output(content);
        assert_eq!(variants, vec!["keyword search", "semantic search"]);
    }

    #[test]
    fn parse_plain_text_fallback() {
        let content = "keyword search\nsemantic search\n\nhypothetical document";
        let variants = parse_expansion_output(content);
        assert_eq!(
            variants,
            vec!["keyword search", "semantic search", "hypothetical document"]
        );
    }

    #[test]
    fn parse_empty_content() {
        let variants = parse_expansion_output("");
        assert!(variants.is_empty());
    }

    #[test]
    fn parse_structured_skips_empty_values() {
        let content = "lex:\nvec: semantic search\nhyde:  ";
        let variants = parse_expansion_output(content);
        assert_eq!(variants, vec!["semantic search"]);
    }
}
