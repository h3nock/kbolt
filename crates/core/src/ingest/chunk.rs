use crate::config::{ChunkPolicy, ChunkingConfig};
use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument};

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

pub fn chunk_document(document: &ExtractedDocument, policy: &ChunkPolicy) -> Vec<FinalChunk> {
    let counter = WhitespaceTokenCounter;
    chunk_document_with_counter(document, policy, &counter)
}

pub fn chunk_document_with_counter(
    document: &ExtractedDocument,
    policy: &ChunkPolicy,
    counter: &dyn TokenCounter,
) -> Vec<FinalChunk> {
    if document.blocks.is_empty() {
        return Vec::new();
    }

    let soft_max = normalized_soft_max(policy);
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut current_tokens = 0usize;

    for block in &document.blocks {
        let block_tokens = counter.count(block.text.as_str());
        let candidate_tokens = current_tokens.saturating_add(block_tokens);

        if current.is_empty() || candidate_tokens <= soft_max {
            current_tokens = candidate_tokens;
            current.push(block.clone());
            continue;
        }

        chunks.push(finalize_chunk(&current));
        current.clear();
        current.push(block.clone());
        current_tokens = block_tokens;
    }

    if !current.is_empty() {
        chunks.push(finalize_chunk(&current));
    }

    chunks
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

fn normalized_soft_max(policy: &ChunkPolicy) -> usize {
    let target = policy.target_tokens.max(1);
    policy.soft_max_tokens.max(target)
}

fn finalize_chunk(blocks: &[ExtractedBlock]) -> FinalChunk {
    let start = blocks.first().map(|block| block.offset).unwrap_or(0);
    let end = blocks
        .last()
        .map(|block| block.offset.saturating_add(block.length))
        .unwrap_or(start);
    let text = blocks
        .iter()
        .map(|block| block.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let heading = resolve_heading(blocks);
    let kind = derive_chunk_kind(blocks);

    FinalChunk {
        text,
        offset: start,
        length: end.saturating_sub(start),
        heading,
        kind,
    }
}

fn resolve_heading(blocks: &[ExtractedBlock]) -> Option<String> {
    blocks
        .iter()
        .rev()
        .find_map(|block| (!block.heading_path.is_empty()).then(|| block.heading_path.join(" > ")))
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
        chunk_document, chunk_document_with_counter, derive_chunk_kind, resolve_policy,
        FinalChunkKind, TokenCounter, WhitespaceTokenCounter,
    };
    use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument};

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

    fn block_with(
        kind: BlockKind,
        text: &str,
        offset: usize,
        heading_path: &[&str],
    ) -> ExtractedBlock {
        ExtractedBlock {
            text: text.to_string(),
            offset,
            length: text.len(),
            kind,
            heading_path: heading_path.iter().map(|value| value.to_string()).collect(),
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

    #[test]
    fn chunk_document_packs_adjacent_blocks_within_soft_max() {
        let policy = ChunkPolicy {
            target_tokens: 3,
            soft_max_tokens: 4,
            hard_max_tokens: 8,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![
                block_with(BlockKind::Paragraph, "alpha beta", 0, &[]),
                block_with(BlockKind::Paragraph, "gamma", 12, &[]),
                block_with(BlockKind::Paragraph, "delta epsilon", 20, &[]),
            ],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "alpha beta\n\ngamma");
        assert_eq!(chunks[0].offset, 0);
        assert_eq!(chunks[0].length, 17);
        assert_eq!(chunks[0].kind, FinalChunkKind::Paragraph);
        assert_eq!(chunks[1].text, "delta epsilon");
    }

    #[test]
    fn chunk_document_resolves_heading_from_structural_path() {
        let policy = ChunkPolicy {
            target_tokens: 2,
            soft_max_tokens: 4,
            hard_max_tokens: 8,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![block_with(
                BlockKind::Paragraph,
                "body text",
                100,
                &["Guide", "Intro"],
            )],
            metadata: HashMap::new(),
            title: Some("Doc".to_string()),
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading.as_deref(), Some("Guide > Intro"));
    }

    #[test]
    fn chunk_document_with_counter_handles_empty_input() {
        let policy = ChunkPolicy::default();
        let document = ExtractedDocument {
            blocks: vec![],
            metadata: HashMap::new(),
            title: None,
        };
        let counter = WhitespaceTokenCounter;

        let chunks = chunk_document_with_counter(&document, &policy, &counter);
        assert!(chunks.is_empty());
    }
}
