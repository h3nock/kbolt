use crate::config::{ChunkPolicy, ChunkingConfig};
use crate::ingest::extract::{BlockKind, ExtractedBlock};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalChunk {
    pub text: String,
    pub offset: usize,
    pub length: usize,
    pub heading: Option<String>,
    pub kind: FinalChunkKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalChunkKind {
    Section,
    Paragraph,
    Code,
    Table,
    Mixed,
}

pub trait TokenCounter {
    fn count(&self, text: &str) -> usize;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct WhitespaceTokenCounter;

impl TokenCounter for WhitespaceTokenCounter {
    fn count(&self, text: &str) -> usize {
        count_whitespace_tokens(text)
    }
}

impl FinalChunkKind {
    pub fn as_storage_kind(self) -> &'static str {
        match self {
            Self::Section => "section",
            Self::Paragraph => "paragraph",
            Self::Code => "code",
            Self::Table => "table",
            Self::Mixed => "mixed",
        }
    }
}

/// Resolves the effective chunk policy for a file profile.
/// Precedence: CLI override > profile > defaults.
pub fn resolve_policy(
    config: &ChunkingConfig,
    profile: Option<&str>,
    cli_override: Option<&ChunkPolicy>,
) -> ChunkPolicy {
    if let Some(override_policy) = cli_override {
        return override_policy.clone();
    }

    if let Some(profile_name) = profile {
        let key = normalize_profile_key(profile_name);
        if let Some(policy) = config.profiles.get(&key) {
            return policy.clone();
        }
    }

    config.defaults.clone()
}

fn normalize_profile_key(raw: &str) -> String {
    raw.trim().trim_start_matches('.').to_ascii_lowercase()
}

fn count_whitespace_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

pub fn derive_chunk_kind(blocks: &[ExtractedBlock]) -> FinalChunkKind {
    if blocks.is_empty() {
        return FinalChunkKind::Mixed;
    }

    if blocks.iter().all(|block| block.kind == BlockKind::CodeFence) {
        return FinalChunkKind::Code;
    }

    if blocks
        .iter()
        .all(|block| matches!(block.kind, BlockKind::TableHeader | BlockKind::TableRow))
    {
        return FinalChunkKind::Table;
    }

    if blocks.iter().all(|block| {
        matches!(
            block.kind,
            BlockKind::Paragraph | BlockKind::ListItem | BlockKind::BlockQuote
        )
    }) {
        return FinalChunkKind::Paragraph;
    }

    if blocks.iter().any(|block| block.kind == BlockKind::Heading)
        && blocks.iter().all(|block| {
            matches!(
                block.kind,
                BlockKind::Heading | BlockKind::Paragraph | BlockKind::ListItem | BlockKind::BlockQuote
            )
        })
    {
        return FinalChunkKind::Section;
    }

    FinalChunkKind::Mixed
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::config::{ChunkPolicy, ChunkingConfig};
    use crate::ingest::chunk::{
        derive_chunk_kind, resolve_policy, FinalChunkKind, TokenCounter, WhitespaceTokenCounter,
    };
    use crate::ingest::extract::{BlockKind, ExtractedBlock};

    fn baseline_config() -> ChunkingConfig {
        ChunkingConfig {
            defaults: ChunkPolicy {
                target_tokens: 450,
                soft_max_tokens: 550,
                hard_max_tokens: 750,
                boundary_overlap_tokens: 48,
                neighbor_window: 1,
                contextual_prefix: true,
            },
            profiles: HashMap::from([(
                "md".to_string(),
                ChunkPolicy {
                    target_tokens: 300,
                    soft_max_tokens: 360,
                    hard_max_tokens: 480,
                    boundary_overlap_tokens: 24,
                    neighbor_window: 2,
                    contextual_prefix: false,
                },
            )]),
        }
    }

    #[test]
    fn resolve_policy_prefers_cli_override() {
        let config = baseline_config();
        let override_policy = ChunkPolicy {
            target_tokens: 128,
            soft_max_tokens: 160,
            hard_max_tokens: 196,
            boundary_overlap_tokens: 16,
            neighbor_window: 3,
            contextual_prefix: false,
        };

        let resolved = resolve_policy(&config, Some("md"), Some(&override_policy));
        assert_eq!(resolved, override_policy);
    }

    #[test]
    fn resolve_policy_uses_normalized_profile_key() {
        let config = baseline_config();

        let resolved = resolve_policy(&config, Some(".MD"), None);
        assert_eq!(resolved.target_tokens, 300);
        assert_eq!(resolved.soft_max_tokens, 360);
        assert_eq!(resolved.hard_max_tokens, 480);
        assert_eq!(resolved.boundary_overlap_tokens, 24);
        assert_eq!(resolved.neighbor_window, 2);
        assert!(!resolved.contextual_prefix);
    }

    #[test]
    fn resolve_policy_falls_back_to_defaults() {
        let config = baseline_config();

        let resolved = resolve_policy(&config, Some("txt"), None);
        assert_eq!(resolved, config.defaults);
    }

    fn block(kind: BlockKind) -> ExtractedBlock {
        ExtractedBlock {
            text: "x".to_string(),
            offset: 0,
            length: 1,
            kind,
            heading_path: vec![],
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn derive_chunk_kind_code_only_is_code() {
        let blocks = vec![block(BlockKind::CodeFence), block(BlockKind::CodeFence)];
        assert_eq!(derive_chunk_kind(&blocks), FinalChunkKind::Code);
    }

    #[test]
    fn derive_chunk_kind_table_only_is_table() {
        let blocks = vec![block(BlockKind::TableHeader), block(BlockKind::TableRow)];
        assert_eq!(derive_chunk_kind(&blocks), FinalChunkKind::Table);
    }

    #[test]
    fn derive_chunk_kind_narrative_without_heading_is_paragraph() {
        let blocks = vec![block(BlockKind::Paragraph), block(BlockKind::ListItem)];
        assert_eq!(derive_chunk_kind(&blocks), FinalChunkKind::Paragraph);
    }

    #[test]
    fn derive_chunk_kind_heading_scoped_narrative_is_section() {
        let blocks = vec![block(BlockKind::Heading), block(BlockKind::Paragraph)];
        assert_eq!(derive_chunk_kind(&blocks), FinalChunkKind::Section);
    }

    #[test]
    fn derive_chunk_kind_mixed_content_is_mixed() {
        let blocks = vec![block(BlockKind::CodeFence), block(BlockKind::Paragraph)];
        assert_eq!(derive_chunk_kind(&blocks), FinalChunkKind::Mixed);
    }

    #[test]
    fn chunk_kind_storage_labels_are_stable() {
        assert_eq!(FinalChunkKind::Section.as_storage_kind(), "section");
        assert_eq!(FinalChunkKind::Paragraph.as_storage_kind(), "paragraph");
        assert_eq!(FinalChunkKind::Code.as_storage_kind(), "code");
        assert_eq!(FinalChunkKind::Table.as_storage_kind(), "table");
        assert_eq!(FinalChunkKind::Mixed.as_storage_kind(), "mixed");
    }

    #[test]
    fn whitespace_token_counter_counts_word_boundaries() {
        let counter = WhitespaceTokenCounter;
        assert_eq!(counter.count(""), 0);
        assert_eq!(counter.count("alpha"), 1);
        assert_eq!(counter.count("alpha beta\tgamma\n\ndelta"), 4);
    }
}
