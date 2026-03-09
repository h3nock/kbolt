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

        let system_prompt = "Return JSON only as {\"variants\":[string,...]}.";
        let user_prompt = format!(
            "/no_think Original query: {normalized}\nReturn 2 to 4 concise search rewrites in JSON only."
        );
        let content = self.client.complete(system_prompt, &user_prompt)?;
        let raw_variants = parse_expansion_output(&content)?;

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

fn parse_expansion_output(content: &str) -> Result<Vec<String>> {
    let trimmed = content.trim();
    let parsed = serde_json::from_str::<serde_json::Value>(trimmed).map_err(|err| {
        kbolt_types::KboltError::Inference(format!("local expander returned invalid JSON: {err}"))
    })?;

    let variants = if let Some(arr) = parsed.get("variants").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    } else if let Some(arr) = parsed.as_array() {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    } else {
        return Err(kbolt_types::KboltError::Inference(
            "local expander response must be a JSON object with variants or a JSON array"
                .to_string(),
        )
        .into());
    };

    Ok(variants)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_expansion_output_accepts_wrapped_json_variants() {
        let content = r#"{"variants":["trait object rust","explain rust traits"]}"#;
        let variants = parse_expansion_output(content).expect("wrapped json");
        assert_eq!(variants, vec!["trait object rust", "explain rust traits"]);
    }

    #[test]
    fn parse_expansion_output_rejects_non_json_content() {
        let content = "lex: keyword search\nvec: dense retrieval";
        let err = parse_expansion_output(content).expect_err("must reject fallback formats");
        assert!(err.to_string().contains("invalid JSON"));
    }

    #[test]
    fn parse_expansion_output_accepts_json_array_variants() {
        let content = r#"["keyword search","semantic search"]"#;
        let variants = parse_expansion_output(content).expect("json array");
        assert_eq!(variants, vec!["keyword search", "semantic search"]);
    }
}
