pub(crate) fn tokenize_terms(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter_map(|term| {
            let lowered = term.trim().to_ascii_lowercase();
            if lowered.is_empty() {
                None
            } else {
                Some(lowered)
            }
        })
        .collect::<Vec<_>>()
}

pub(crate) fn strip_json_fences(content: &str) -> &str {
    let trimmed = content.trim();
    let Some(without_prefix) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let Some(inner) = without_prefix.strip_suffix("```") else {
        return trimmed;
    };

    let Some((first_line, remainder)) = inner.split_once('\n') else {
        return inner.trim();
    };
    let first = first_line.trim();
    if first.is_empty() {
        return inner.trim();
    }
    if !first.eq_ignore_ascii_case("json") {
        return trimmed;
    }

    remainder.trim()
}

#[cfg(test)]
mod tests {
    use super::{strip_json_fences, tokenize_terms};

    #[test]
    fn tokenize_terms_splits_and_normalizes() {
        assert_eq!(
            tokenize_terms("Rust_trait-object!"),
            vec![
                "rust".to_string(),
                "trait".to_string(),
                "object".to_string()
            ]
        );
    }

    #[test]
    fn strip_json_fences_removes_json_markdown_wrapper() {
        let wrapped = "```json\n{\"scores\":[0.2,0.9]}\n```";
        assert_eq!(strip_json_fences(wrapped), "{\"scores\":[0.2,0.9]}");
    }

    #[test]
    fn strip_json_fences_keeps_non_fenced_payload() {
        let plain = "{\"scores\":[0.2,0.9]}";
        assert_eq!(strip_json_fences(plain), plain);
    }

    #[test]
    fn strip_json_fences_handles_incomplete_fence_without_panic() {
        assert_eq!(strip_json_fences("```"), "```");
    }

    #[test]
    fn strip_json_fences_keeps_non_json_language_fence() {
        let wrapped = "```yaml\nscores: [0.2, 0.9]\n```";
        assert_eq!(strip_json_fences(wrapped), wrapped);
    }
}
