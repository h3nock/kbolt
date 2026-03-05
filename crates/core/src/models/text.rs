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
    if !trimmed.starts_with("```") || !trimmed.ends_with("```") {
        return trimmed;
    }

    let inner = &trimmed[3..trimmed.len().saturating_sub(3)];
    let mut lines = inner.lines();
    let first = lines.next().unwrap_or_default().trim();
    if !first.is_empty() && !first.eq_ignore_ascii_case("json") {
        return trimmed;
    }

    let offset = first.len();
    inner.get(offset..).unwrap_or_default().trim()
}

#[cfg(test)]
mod tests {
    use super::{strip_json_fences, tokenize_terms};

    #[test]
    fn tokenize_terms_splits_and_normalizes() {
        assert_eq!(
            tokenize_terms("Rust_trait-object!"),
            vec!["rust".to_string(), "trait".to_string(), "object".to_string()]
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
}
