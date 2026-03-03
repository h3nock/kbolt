use crate::config::{ChunkPolicy, ChunkingConfig};
use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument};

const TABLE_HEADER_ATTR: &str = "__kbolt_table_header";

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackClass {
    Narrative,
    Code,
    Table,
    Opaque,
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
    let expanded = expand_blocks_for_hard_max(&document.blocks, policy, counter);
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut current_tokens = 0usize;

    for block in &expanded {
        let block_tokens = counter.count(block.text.as_str());
        let candidate_tokens = current_tokens.saturating_add(block_tokens);
        let structurally_compatible = current
            .last()
            .is_none_or(|last| can_pack_together(last, block));

        if current.is_empty() || (structurally_compatible && candidate_tokens <= soft_max) {
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

fn normalized_hard_max(policy: &ChunkPolicy) -> usize {
    let soft_max = normalized_soft_max(policy);
    policy.hard_max_tokens.max(soft_max)
}

fn normalized_overlap(policy: &ChunkPolicy, hard_max: usize) -> usize {
    policy
        .boundary_overlap_tokens
        .min(hard_max.saturating_sub(1))
}

fn expand_blocks_for_hard_max(
    blocks: &[ExtractedBlock],
    policy: &ChunkPolicy,
    counter: &dyn TokenCounter,
) -> Vec<ExtractedBlock> {
    let hard_max = normalized_hard_max(policy);
    let overlap = normalized_overlap(policy, hard_max);
    let mut expanded = Vec::new();
    let mut active_table_header: Option<String> = None;

    for block in blocks {
        match block.kind {
            BlockKind::TableHeader => {
                active_table_header = Some(block.text.clone());
            }
            BlockKind::TableRow => {}
            _ => {
                active_table_header = None;
            }
        }

        let tagged = attach_table_header_attr(block, active_table_header.as_deref());

        if counter.count(tagged.text.as_str()) <= hard_max {
            expanded.push(tagged);
            continue;
        }

        if is_narrative_block_kind(&tagged.kind) {
            if let Some(sentence_splits) =
                split_block_by_sentence_boundaries(&tagged, hard_max, overlap)
            {
                expanded.extend(sentence_splits);
                continue;
            }
        }

        expanded.extend(split_block_by_tokens(&tagged, hard_max, overlap));
    }

    expanded
}

fn attach_table_header_attr(block: &ExtractedBlock, table_header: Option<&str>) -> ExtractedBlock {
    let mut tagged = block.clone();
    if tagged.kind == BlockKind::TableRow {
        if let Some(header) = table_header {
            if !header.trim().is_empty() {
                tagged
                    .attrs
                    .insert(TABLE_HEADER_ATTR.to_string(), header.to_string());
            }
        }
    }
    tagged
}

fn is_narrative_block_kind(kind: &BlockKind) -> bool {
    matches!(kind, BlockKind::Paragraph | BlockKind::ListItem | BlockKind::BlockQuote)
}

fn pack_class_for_kind(kind: &BlockKind) -> PackClass {
    match kind {
        BlockKind::Heading
        | BlockKind::Paragraph
        | BlockKind::ListItem
        | BlockKind::BlockQuote => PackClass::Narrative,
        BlockKind::CodeFence => PackClass::Code,
        BlockKind::TableHeader | BlockKind::TableRow => PackClass::Table,
        BlockKind::HtmlBlock => PackClass::Opaque,
    }
}

fn can_pack_together(current: &ExtractedBlock, next: &ExtractedBlock) -> bool {
    let current_class = pack_class_for_kind(&current.kind);
    let next_class = pack_class_for_kind(&next.kind);
    current_class == next_class && current_class != PackClass::Opaque
}

fn split_block_by_tokens(block: &ExtractedBlock, hard_max: usize, overlap: usize) -> Vec<ExtractedBlock> {
    let spans = token_byte_spans(block.text.as_str());
    if spans.is_empty() {
        return vec![block.clone()];
    }

    let mut out = Vec::new();
    let mut start_token = 0usize;
    while start_token < spans.len() {
        let end_token = (start_token + hard_max).min(spans.len());
        out.push(split_block_range(block, &spans, start_token, end_token));

        if end_token == spans.len() {
            break;
        }

        start_token = next_start_token(start_token, end_token, overlap);
    }

    out
}

fn split_block_by_sentence_boundaries(
    block: &ExtractedBlock,
    hard_max: usize,
    overlap: usize,
) -> Option<Vec<ExtractedBlock>> {
    let spans = token_byte_spans(block.text.as_str());
    if spans.is_empty() {
        return Some(vec![block.clone()]);
    }

    let sentence_end_tokens = sentence_end_token_indices(block.text.as_str(), &spans);
    if sentence_end_tokens.is_empty() {
        return None;
    }

    let mut used_sentence_boundary = false;
    let mut out = Vec::new();
    let mut start_token = 0usize;
    while start_token < spans.len() {
        let max_end = (start_token + hard_max).min(spans.len());
        let end_token = match last_sentence_end_within(&sentence_end_tokens, start_token, max_end) {
            Some(boundary_end) => {
                used_sentence_boundary = true;
                boundary_end
            }
            None => max_end,
        };

        out.push(split_block_range(block, &spans, start_token, end_token));
        if end_token == spans.len() {
            break;
        }

        start_token = next_start_token(start_token, end_token, overlap);
    }

    used_sentence_boundary.then_some(out)
}

fn split_block_range(
    block: &ExtractedBlock,
    spans: &[(usize, usize)],
    start_token: usize,
    end_token: usize,
) -> ExtractedBlock {
    let byte_start = spans[start_token].0;
    let byte_end = spans[end_token - 1].1;

    let mut split = block.clone();
    split.offset = block.offset.saturating_add(byte_start);
    split.length = byte_end.saturating_sub(byte_start);
    split.text = block.text[byte_start..byte_end].to_string();
    split
}

fn next_start_token(start_token: usize, end_token: usize, overlap: usize) -> usize {
    let next = end_token.saturating_sub(overlap);
    if next <= start_token {
        end_token
    } else {
        next
    }
}

fn sentence_end_token_indices(text: &str, spans: &[(usize, usize)]) -> Vec<usize> {
    spans
        .iter()
        .enumerate()
        .filter_map(|(index, (start, end))| {
            token_ends_sentence(&text[*start..*end]).then_some(index)
        })
        .collect()
}

fn last_sentence_end_within(
    sentence_end_tokens: &[usize],
    start_token: usize,
    max_end: usize,
) -> Option<usize> {
    sentence_end_tokens
        .iter()
        .copied()
        .filter(|index| *index >= start_token && (*index + 1) <= max_end)
        .max()
        .map(|index| index + 1)
}

fn token_ends_sentence(token: &str) -> bool {
    let trimmed = token.trim_end_matches(|ch: char| matches!(ch, '"' | '\'' | ')' | ']' | '}'));
    trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?')
}

fn token_byte_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut token_start: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(start) = token_start.take() {
                spans.push((start, idx));
            }
        } else if token_start.is_none() {
            token_start = Some(idx);
        }
    }

    if let Some(start) = token_start {
        spans.push((start, text.len()));
    }

    spans
}

fn finalize_chunk(blocks: &[ExtractedBlock]) -> FinalChunk {
    let start = blocks.first().map(|block| block.offset).unwrap_or(0);
    let end = blocks
        .last()
        .map(|block| block.offset.saturating_add(block.length))
        .unwrap_or(start);
    let mut text = blocks
        .iter()
        .map(|block| block.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let heading = resolve_heading(blocks);
    let kind = derive_chunk_kind(blocks);
    if kind == FinalChunkKind::Table {
        let has_header = blocks.iter().any(|block| block.kind == BlockKind::TableHeader);
        if !has_header {
            if let Some(header) = blocks
                .first()
                .and_then(|block| block.attrs.get(TABLE_HEADER_ATTR))
                .map(String::as_str)
            {
                text = format!("{header}\n{text}");
            }
        }
    }

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
        can_pack_together, chunk_document, chunk_document_with_counter, derive_chunk_kind,
        resolve_policy,
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

    #[test]
    fn chunk_document_splits_oversized_block_at_hard_max_with_overlap() {
        let policy = ChunkPolicy {
            target_tokens: 4,
            soft_max_tokens: 4,
            hard_max_tokens: 4,
            boundary_overlap_tokens: 1,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![block_with(
                BlockKind::Paragraph,
                "one two three four five six seven eight nine ten",
                10,
                &["Doc"],
            )],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "one two three four");
        assert_eq!(chunks[1].text, "four five six seven");
        assert_eq!(chunks[2].text, "seven eight nine ten");

        assert_eq!(chunks[0].offset, 10);
        assert_eq!(chunks[0].length, 18);
        assert_eq!(chunks[1].offset, 24);
        assert_eq!(chunks[1].length, 19);
        assert_eq!(chunks[2].offset, 38);
        assert_eq!(chunks[2].length, 20);
    }

    #[test]
    fn chunk_document_prefers_sentence_boundaries_for_narrative_forced_split() {
        let policy = ChunkPolicy {
            target_tokens: 4,
            soft_max_tokens: 4,
            hard_max_tokens: 4,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![block_with(
                BlockKind::Paragraph,
                "alpha one. beta two three. gamma four five. delta six seven.",
                0,
                &["Doc"],
            )],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].text, "alpha one.");
        assert_eq!(chunks[1].text, "beta two three.");
        assert_eq!(chunks[2].text, "gamma four five.");
        assert_eq!(chunks[3].text, "delta six seven.");
    }

    #[test]
    fn chunk_document_keeps_token_window_split_for_non_narrative_blocks() {
        let policy = ChunkPolicy {
            target_tokens: 4,
            soft_max_tokens: 4,
            hard_max_tokens: 4,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![block_with(
                BlockKind::CodeFence,
                "alpha. beta gamma delta epsilon zeta",
                0,
                &[],
            )],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "alpha. beta gamma delta");
        assert_eq!(chunks[1].text, "epsilon zeta");
        assert!(chunks.iter().all(|chunk| chunk.kind == FinalChunkKind::Code));
    }

    #[test]
    fn chunk_document_carries_table_header_for_row_only_chunks() {
        let policy = ChunkPolicy {
            target_tokens: 4,
            soft_max_tokens: 4,
            hard_max_tokens: 4,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![
                block_with(BlockKind::TableHeader, "h1 h2 h3 h4", 0, &[]),
                block_with(BlockKind::TableRow, "r1a r1b r1c r1d", 20, &[]),
                block_with(BlockKind::TableRow, "r2a r2b r2c r2d", 40, &[]),
            ],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 3);

        assert_eq!(chunks[0].text, "h1 h2 h3 h4");
        assert_eq!(chunks[0].offset, 0);
        assert_eq!(chunks[0].length, 11);

        assert_eq!(chunks[1].text, "h1 h2 h3 h4\nr1a r1b r1c r1d");
        assert_eq!(chunks[1].offset, 20);
        assert_eq!(chunks[1].length, 15);

        assert_eq!(chunks[2].text, "h1 h2 h3 h4\nr2a r2b r2c r2d");
        assert_eq!(chunks[2].offset, 40);
        assert_eq!(chunks[2].length, 15);
        assert!(chunks.iter().all(|chunk| chunk.kind == FinalChunkKind::Table));
    }

    #[test]
    fn chunk_document_flushes_on_structural_boundary_even_under_budget() {
        let policy = ChunkPolicy {
            target_tokens: 20,
            soft_max_tokens: 30,
            hard_max_tokens: 30,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![
                block_with(BlockKind::Heading, "# Intro", 0, &[]),
                block_with(BlockKind::Paragraph, "alpha beta", 8, &["Intro"]),
                block_with(BlockKind::CodeFence, "fn alpha() {}", 20, &["Intro"]),
                block_with(BlockKind::Paragraph, "gamma delta", 34, &["Intro"]),
            ],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].kind, FinalChunkKind::Section);
        assert_eq!(chunks[1].kind, FinalChunkKind::Code);
        assert_eq!(chunks[2].kind, FinalChunkKind::Paragraph);
    }

    #[test]
    fn can_pack_together_allows_same_narrative_family() {
        let heading = block_with(BlockKind::Heading, "# Intro", 0, &[]);
        let paragraph = block_with(BlockKind::Paragraph, "text", 10, &[]);
        assert!(can_pack_together(&heading, &paragraph));
    }

    #[test]
    fn can_pack_together_rejects_cross_family_boundaries() {
        let paragraph = block_with(BlockKind::Paragraph, "text", 0, &[]);
        let code = block_with(BlockKind::CodeFence, "fn a() {}", 10, &[]);
        let table = block_with(BlockKind::TableRow, "|a|b|", 20, &[]);
        assert!(!can_pack_together(&paragraph, &code));
        assert!(!can_pack_together(&paragraph, &table));
        assert!(!can_pack_together(&code, &table));
    }

    #[test]
    fn can_pack_together_treats_html_as_opaque() {
        let html = block_with(BlockKind::HtmlBlock, "<div>x</div>", 0, &[]);
        let html2 = block_with(BlockKind::HtmlBlock, "<div>y</div>", 20, &[]);
        let paragraph = block_with(BlockKind::Paragraph, "text", 40, &[]);
        assert!(!can_pack_together(&html, &html2));
        assert!(!can_pack_together(&html, &paragraph));
    }
}
