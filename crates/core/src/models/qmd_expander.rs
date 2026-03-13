use std::collections::HashMap;
use std::sync::Arc;

use kbolt_types::KboltError;

use crate::models::local_llama::{
    LocalLlamaClient, LocalLlamaGenerationOptions, LocalLlamaGrammar, LocalLlamaPrompt,
    LocalLlamaSamplerConfig, LocalLlamaSamplingParams,
};
use crate::models::{normalize_query_text, ExpandedQuery, Expander, ExpansionRoute};
use crate::Result;

const QMD_PROMPT_PREFIX: &str = "/no_think Expand this search query: ";
const QMD_EXPANSION_GRAMMAR: &str = r#"root ::= line+
line ::= type ": " content "\n"
type ::= "lex" | "vec" | "hyde"
content ::= [^\n]+"#;

pub(super) struct LocalQmdExpander {
    client: Arc<LocalLlamaClient>,
}

impl LocalQmdExpander {
    pub(super) fn new(client: Arc<LocalLlamaClient>) -> Self {
        Self { client }
    }
}

impl Expander for LocalQmdExpander {
    fn expand(&self, query: &str) -> Result<Vec<ExpandedQuery>> {
        let normalized = normalize_query_text(query);
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        let user_prompt = format!("{QMD_PROMPT_PREFIX}{normalized}");
        let content = self.client.generate(
            qmd_generation_prompt(&user_prompt),
            &LocalLlamaGenerationOptions {
                sampler: LocalLlamaSamplerConfig::Sample(LocalLlamaSamplingParams {
                    seed: 0,
                    temperature: 0.7,
                    top_k: 20,
                    top_p: 0.8,
                    repeat_last_n: 64,
                    repeat_penalty: 1.0,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.5,
                }),
                grammar: Some(LocalLlamaGrammar {
                    grammar: QMD_EXPANSION_GRAMMAR.to_string(),
                    root: "root".to_string(),
                }),
                ..LocalLlamaGenerationOptions::default()
            },
        )?;

        parse_qmd_output(&normalized, &content)
    }
}

fn qmd_generation_prompt(user_prompt: &str) -> LocalLlamaPrompt<'_> {
    LocalLlamaPrompt::Chat {
        system_prompt: "",
        user_prompt,
    }
}

fn parse_qmd_output(original_query: &str, content: &str) -> Result<Vec<ExpandedQuery>> {
    let mut variants: Vec<ExpandedQuery> = Vec::new();
    let mut index_by_text: HashMap<String, usize> = HashMap::new();

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let Some((kind, text)) = line.split_once(':') else {
            // Truncated trailing line from max_tokens cutoff — skip.
            continue;
        };
        let route = match kind.trim() {
            "lex" => ExpansionRoute::KeywordOnly,
            "vec" | "hyde" => ExpansionRoute::DenseOnly,
            _ => continue,
        };

        let text = normalize_query_text(text.trim());
        if text.is_empty() {
            continue;
        }
        if text.eq_ignore_ascii_case(original_query) {
            continue;
        }

        let key = text.to_ascii_lowercase();
        if let Some(index) = index_by_text.get(&key).copied() {
            let existing = variants
                .get_mut(index)
                .expect("tracked qmd expansion index must be valid");
            existing.route = existing.route.merged_with(route);
            continue;
        }

        index_by_text.insert(key, variants.len());
        variants.push(ExpandedQuery { text, route });
    }

    if variants.is_empty() {
        return Err(KboltError::Inference(
            "qmd expander returned no usable expansions".to_string(),
        )
        .into());
    }

    Ok(variants)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_qmd_output_maps_routes_and_merges_duplicates() {
        let parsed = parse_qmd_output(
            "rust borrow checker",
            "lex: rust borrow checker errors\nvec: rust ownership explanation\nhyde: rust ownership explanation\n",
        )
        .expect("valid qmd output");

        assert_eq!(
            parsed,
            vec![
                ExpandedQuery {
                    text: "rust borrow checker errors".to_string(),
                    route: ExpansionRoute::KeywordOnly,
                },
                ExpandedQuery {
                    text: "rust ownership explanation".to_string(),
                    route: ExpansionRoute::DenseOnly,
                },
            ]
        );
    }

    #[test]
    fn parse_qmd_output_skips_malformed_and_truncated_lines() {
        let parsed = parse_qmd_output(
            "query",
            "json: nope\nlex: keyword search\nvec: semantic search\nhy",
        )
        .expect("should skip invalid lines and use valid ones");
        assert_eq!(
            parsed,
            vec![
                ExpandedQuery {
                    text: "keyword search".to_string(),
                    route: ExpansionRoute::KeywordOnly,
                },
                ExpandedQuery {
                    text: "semantic search".to_string(),
                    route: ExpansionRoute::DenseOnly,
                },
            ]
        );
    }

    #[test]
    fn parse_qmd_output_rejects_empty_result() {
        let err = parse_qmd_output("same query", "lex: same query\n")
            .expect_err("must reject filtered-empty output");
        assert!(err.to_string().contains("no usable expansions"));
    }

    #[test]
    fn qmd_generation_uses_chat_prompt_path() {
        let prompt = format!("{QMD_PROMPT_PREFIX}database schema migration");

        match qmd_generation_prompt(&prompt) {
            LocalLlamaPrompt::Chat {
                system_prompt,
                user_prompt,
            } => {
                assert_eq!(system_prompt, "");
                assert_eq!(user_prompt, prompt);
            }
            LocalLlamaPrompt::Raw(_) => panic!("qmd expander must use the chat prompt path"),
        }
    }
}
