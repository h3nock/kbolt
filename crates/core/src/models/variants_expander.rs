use std::collections::HashSet;
use std::sync::Arc;

use kbolt_types::KboltError;
use serde::Deserialize;
use serde_json::from_str;

use crate::models::completion::CompletionClient;
use crate::models::local_llama::{
    LocalLlamaClient, LocalLlamaGenerationOptions, LocalLlamaGrammar, LocalLlamaPrompt,
};
use crate::models::text::strip_json_fences;
use crate::models::{normalize_query_text, Expander};
use crate::Result;

const VARIANTS_SYSTEM_PROMPT: &str = "You generate retrieval query variants. Return JSON only as an array of strings. Preserve the original intent, named entities, numbers, abbreviations, error text, config keys, and file paths. Keep each variant specific and retrieval-focused. Do not answer the query. Do not explain anything.";

const VARIANTS_GRAMMAR: &str = r#"root ::= ws array ws
array ::= "[" ws elements? ws "]"
elements ::= string (ws "," ws string)*
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\\x00-\x1F] | escape
escape ::= "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
ws ::= [ \t\n\r]*"#;

#[derive(Clone)]
pub(super) struct ChatVariantsExpander {
    chat: Arc<dyn CompletionClient>,
}

impl ChatVariantsExpander {
    pub(super) fn new(chat: Arc<dyn CompletionClient>) -> Self {
        Self { chat }
    }
}

#[derive(Clone)]
pub(super) struct LocalLlamaVariantsExpander {
    client: Arc<LocalLlamaClient>,
    options: LocalLlamaGenerationOptions,
}

impl LocalLlamaVariantsExpander {
    pub(super) fn new(
        client: Arc<LocalLlamaClient>,
        mut options: LocalLlamaGenerationOptions,
    ) -> Self {
        options.grammar = Some(LocalLlamaGrammar {
            grammar: VARIANTS_GRAMMAR.to_string(),
            root: "root".to_string(),
        });
        Self { client, options }
    }
}

impl Expander for ChatVariantsExpander {
    fn expand(&self, query: &str, max_variants: usize) -> Result<Vec<String>> {
        let normalized = normalize_query_text(query);
        if normalized.is_empty() || max_variants == 0 {
            return Ok(Vec::new());
        }

        let prompt = variants_user_prompt(&normalized, max_variants);
        let content = self.chat.complete(VARIANTS_SYSTEM_PROMPT, &prompt)?;
        parse_variants_output(&normalized, &content, max_variants)
    }
}

impl Expander for LocalLlamaVariantsExpander {
    fn expand(&self, query: &str, max_variants: usize) -> Result<Vec<String>> {
        let normalized = normalize_query_text(query);
        if normalized.is_empty() || max_variants == 0 {
            return Ok(Vec::new());
        }

        let prompt = variants_user_prompt(&normalized, max_variants);
        let content = self.client.generate(
            LocalLlamaPrompt::Chat {
                system_prompt: VARIANTS_SYSTEM_PROMPT,
                user_prompt: &prompt,
            },
            &self.options,
        )?;
        parse_variants_output(&normalized, &content, max_variants)
    }
}

fn variants_user_prompt(query: &str, max_variants: usize) -> String {
    format!(
        "Original query:\n{query}\n\nGenerate {max_variants} distinct retrieval-useful variants as a JSON array of strings only."
    )
}

fn parse_variants_output(
    original_query: &str,
    content: &str,
    max_variants: usize,
) -> Result<Vec<String>> {
    let trimmed = strip_json_fences(content).trim();
    let parsed: VariantsResponse = from_str(trimmed).map_err(|err| {
        KboltError::Inference(format!(
            "failed to parse expander response as JSON variants: {err}; payload={content}"
        ))
    })?;

    let mut variants = Vec::new();
    let mut seen = HashSet::new();
    for variant in parsed.into_variants() {
        let normalized = normalize_query_text(&variant);
        if normalized.is_empty() || normalized.eq_ignore_ascii_case(original_query) {
            continue;
        }

        let key = normalized.to_ascii_lowercase();
        if seen.insert(key) {
            variants.push(normalized);
        }

        if variants.len() >= max_variants {
            break;
        }
    }

    Ok(variants)
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum VariantsResponse {
    Variants(Vec<String>),
    Wrapped { variants: Vec<String> },
}

impl VariantsResponse {
    fn into_variants(self) -> Vec<String> {
        match self {
            Self::Variants(variants) => variants,
            Self::Wrapped { variants } => variants,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_variants_output;

    #[test]
    fn parse_variants_output_accepts_plain_array() {
        let variants = parse_variants_output(
            "rust traits",
            "[\"trait object rust\",\"explain rust traits\"]",
            4,
        )
        .expect("parse variants");

        assert_eq!(
            variants,
            vec![
                "trait object rust".to_string(),
                "explain rust traits".to_string(),
            ]
        );
    }

    #[test]
    fn parse_variants_output_filters_duplicates_and_original_query() {
        let variants = parse_variants_output(
            "rust traits",
            r#"{"variants":["rust traits","trait object rust","Trait Object Rust","  explain rust traits  "]}"#,
            4,
        )
        .expect("parse variants");

        assert_eq!(
            variants,
            vec![
                "trait object rust".to_string(),
                "explain rust traits".to_string(),
            ]
        );
    }

    #[test]
    fn parse_variants_output_truncates_to_requested_limit() {
        let variants = parse_variants_output(
            "rust traits",
            "[\"trait object rust\",\"explain rust traits\",\"borrow checker traits\"]",
            2,
        )
        .expect("parse variants");

        assert_eq!(
            variants,
            vec![
                "trait object rust".to_string(),
                "explain rust traits".to_string(),
            ]
        );
    }

    #[test]
    fn parse_variants_output_rejects_invalid_json() {
        let err =
            parse_variants_output("rust traits", "trait object rust", 3).expect_err("invalid json");
        assert!(err
            .to_string()
            .contains("failed to parse expander response as JSON variants"));
    }
}
