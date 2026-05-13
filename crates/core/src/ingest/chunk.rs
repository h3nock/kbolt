use crate::config::{ChunkPolicy, ChunkingConfig};
use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument};
use crate::Result;
use kbolt_types::KboltError;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NarrativeBoundary {
    Sentence,
    Clause,
    TokenWindow,
}

pub trait TokenCounter {
    fn count(&self, text: &str) -> Result<usize>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct WhitespaceTokenCounter;

impl TokenCounter for WhitespaceTokenCounter {
    fn count(&self, text: &str) -> Result<usize> {
        Ok(count_whitespace_tokens(text))
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

impl TryFrom<&str> for FinalChunkKind {
    type Error = KboltError;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value {
            "section" => Ok(Self::Section),
            "paragraph" => Ok(Self::Paragraph),
            "code" => Ok(Self::Code),
            "table" => Ok(Self::Table),
            "mixed" => Ok(Self::Mixed),
            other => Err(KboltError::Internal(format!(
                "invalid stored chunk kind: {other}"
            ))),
        }
    }
}

pub fn chunk_document(document: &ExtractedDocument, policy: &ChunkPolicy) -> Vec<FinalChunk> {
    let counter = WhitespaceTokenCounter;
    chunk_document_with_counter(document, policy, &counter)
        .expect("whitespace token counter should be infallible")
}

pub fn chunk_canonical_document(
    document: &ExtractedDocument,
    policy: &ChunkPolicy,
) -> Vec<FinalChunk> {
    let counter = WhitespaceTokenCounter;
    chunk_canonical_document_with_counter(document, policy, &counter)
        .expect("whitespace token counter should be infallible")
}

pub fn chunk_document_with_counter(
    document: &ExtractedDocument,
    policy: &ChunkPolicy,
    counter: &dyn TokenCounter,
) -> Result<Vec<FinalChunk>> {
    chunk_document_with_counter_inner(document, policy, counter, TableHeaderMode::SourceBlocks)
}

pub fn chunk_canonical_document_with_counter(
    document: &ExtractedDocument,
    policy: &ChunkPolicy,
    counter: &dyn TokenCounter,
) -> Result<Vec<FinalChunk>> {
    chunk_document_with_counter_inner(document, policy, counter, TableHeaderMode::CanonicalBlocks)
}

fn chunk_document_with_counter_inner(
    document: &ExtractedDocument,
    policy: &ChunkPolicy,
    counter: &dyn TokenCounter,
    table_header_mode: TableHeaderMode,
) -> Result<Vec<FinalChunk>> {
    if document.blocks.is_empty() {
        return Ok(Vec::new());
    }

    debug_assert_valid_blocks(&document.blocks);

    let soft_max = normalized_soft_max(policy);
    let expanded =
        expand_blocks_for_hard_max(&document.blocks, policy, counter, table_header_mode)?;
    let mut chunks = Vec::new();
    let mut current = Vec::new();

    for block in &expanded {
        let structurally_compatible = current
            .last()
            .is_none_or(|last| can_pack_together(last, block));
        let candidate_tokens = if current.is_empty() || !structurally_compatible {
            0
        } else {
            count_candidate_chunk_tokens(&current, block, counter)?
        };

        if current.is_empty() || (structurally_compatible && candidate_tokens <= soft_max) {
            current.push(block.clone());
            continue;
        }

        chunks.push(finalize_chunk(&current));
        current.clear();
        current.push(block.clone());
    }

    if !current.is_empty() {
        chunks.push(finalize_chunk(&current));
    }

    Ok(chunks)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableHeaderMode {
    SourceBlocks,
    CanonicalBlocks,
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

fn countable_chunk_text(blocks: &[ExtractedBlock]) -> String {
    blocks
        .iter()
        .map(|block| block.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn count_finalized_chunk_tokens(
    blocks: &[ExtractedBlock],
    counter: &dyn TokenCounter,
) -> Result<usize> {
    counter.count(countable_chunk_text(blocks).as_str())
}

fn count_candidate_chunk_tokens(
    current: &[ExtractedBlock],
    next: &ExtractedBlock,
    counter: &dyn TokenCounter,
) -> Result<usize> {
    let mut candidate = current.to_vec();
    candidate.push(next.clone());
    count_finalized_chunk_tokens(&candidate, counter)
}

fn count_single_block_tokens(block: &ExtractedBlock, counter: &dyn TokenCounter) -> Result<usize> {
    count_finalized_chunk_tokens(std::slice::from_ref(block), counter)
}

fn normalized_soft_max(policy: &ChunkPolicy) -> usize {
    let target = policy.target_tokens.max(1);
    policy.soft_max_tokens.max(target)
}

fn normalized_target(policy: &ChunkPolicy) -> usize {
    policy.target_tokens.max(1)
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
    table_header_mode: TableHeaderMode,
) -> Result<Vec<ExtractedBlock>> {
    let hard_max = normalized_hard_max(policy);
    let target = normalized_target(policy);
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

        let tagged = match table_header_mode {
            TableHeaderMode::SourceBlocks => {
                attach_table_header_attr(block, active_table_header.as_deref())
            }
            TableHeaderMode::CanonicalBlocks => block.clone(),
        };

        if count_single_block_tokens(&tagged, counter)? <= hard_max {
            expanded.push(tagged);
            continue;
        }

        if is_narrative_block_kind(&tagged.kind) {
            if let Some(sentence_splits) =
                split_block_by_sentence_boundaries(&tagged, target, hard_max, overlap, counter)?
            {
                expanded.extend(sentence_splits);
                continue;
            }
        }

        if tagged.kind == BlockKind::CodeFence {
            if let Some(code_splits) =
                split_code_block_by_blank_lines(&tagged, hard_max, overlap, counter)?
            {
                expanded.extend(code_splits);
                continue;
            }
        }

        expanded.extend(split_block_by_tokens(&tagged, hard_max, overlap, counter)?);
    }

    Ok(expanded)
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
    matches!(
        kind,
        BlockKind::Paragraph | BlockKind::ListItem | BlockKind::BlockQuote
    )
}

fn pack_class_for_kind(kind: &BlockKind) -> PackClass {
    match kind {
        BlockKind::Heading | BlockKind::Paragraph | BlockKind::ListItem | BlockKind::BlockQuote => {
            PackClass::Narrative
        }
        BlockKind::CodeFence => PackClass::Code,
        BlockKind::TableHeader | BlockKind::TableRow => PackClass::Table,
        BlockKind::HtmlBlock => PackClass::Opaque,
    }
}

fn can_pack_together(current: &ExtractedBlock, next: &ExtractedBlock) -> bool {
    let current_class = pack_class_for_kind(&current.kind);
    let next_class = pack_class_for_kind(&next.kind);
    current_class == next_class
        && current_class != PackClass::Opaque
        && heading_scopes_compatible(current, next)
}

fn heading_scopes_compatible(current: &ExtractedBlock, next: &ExtractedBlock) -> bool {
    let current_class = pack_class_for_kind(&current.kind);
    let next_class = pack_class_for_kind(&next.kind);
    if current_class != PackClass::Narrative || next_class != PackClass::Narrative {
        return true;
    }

    // Start each heading in a fresh chunk to keep section-level locality.
    if next.kind == BlockKind::Heading {
        return false;
    }

    // Allow heading + body packing inside the newly started section.
    if current.kind == BlockKind::Heading {
        return true;
    }

    current.heading_path == next.heading_path
}

fn split_block_by_tokens(
    block: &ExtractedBlock,
    hard_max: usize,
    overlap: usize,
    counter: &dyn TokenCounter,
) -> Result<Vec<ExtractedBlock>> {
    if block.text.is_empty() {
        return Ok(vec![block.clone()]);
    }

    let mut out = Vec::new();
    let mut start_byte = 0usize;
    while start_byte < block.text.len() {
        let end_byte = find_largest_fitting_end_byte(block, start_byte, hard_max, counter)?;
        out.push(split_block_range_by_bytes(block, start_byte, end_byte));

        if end_byte == block.text.len() {
            break;
        }

        start_byte = next_start_byte(block, start_byte, end_byte, overlap, counter)?;
    }

    Ok(out)
}

fn split_code_block_by_blank_lines(
    block: &ExtractedBlock,
    hard_max: usize,
    overlap: usize,
    counter: &dyn TokenCounter,
) -> Result<Option<Vec<ExtractedBlock>>> {
    let groups = code_group_ranges(block.text.as_str());
    if groups.len() <= 1 {
        return Ok(None);
    }

    let mut packed_ranges = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    for (start, end) in groups {
        match current {
            None => {
                current = Some((start, end));
            }
            Some((current_start, current_end)) => {
                let candidate = split_block_range_by_bytes(block, current_start, end);
                if count_single_block_tokens(&candidate, counter)? <= hard_max {
                    current = Some((current_start, end));
                } else {
                    packed_ranges.push((current_start, current_end));
                    current = Some((start, end));
                }
            }
        }
    }
    if let Some((start, end)) = current {
        packed_ranges.push((start, end));
    }

    let mut out = Vec::new();
    for (start, end) in packed_ranges {
        let split = split_block_range_by_bytes(block, start, end);
        if count_single_block_tokens(&split, counter)? > hard_max {
            out.extend(split_block_by_tokens(&split, hard_max, overlap, counter)?);
        } else {
            out.push(split);
        }
    }

    Ok((out.len() > 1).then_some(out))
}

fn split_block_by_sentence_boundaries(
    block: &ExtractedBlock,
    target_tokens: usize,
    hard_max: usize,
    overlap: usize,
    counter: &dyn TokenCounter,
) -> Result<Option<Vec<ExtractedBlock>>> {
    let spans = token_byte_spans(block.text.as_str());
    if spans.is_empty() {
        return Ok(Some(vec![block.clone()]));
    }

    let sentence_end_tokens = sentence_end_token_indices(block.text.as_str(), &spans)
        .into_iter()
        .map(|index| spans[index].1)
        .collect::<Vec<_>>();
    let clause_end_tokens = clause_end_token_indices(block.text.as_str(), &spans)
        .into_iter()
        .map(|index| spans[index].1)
        .collect::<Vec<_>>();
    if sentence_end_tokens.is_empty() && clause_end_tokens.is_empty() {
        return Ok(None);
    }

    let mut used_structural_boundary = false;
    let mut out = Vec::new();
    let mut start_byte = 0usize;
    while start_byte < block.text.len() {
        let mut candidates = sentence_end_tokens
            .iter()
            .copied()
            .filter(|end_byte| *end_byte > start_byte)
            .map(|end_byte| (end_byte, NarrativeBoundary::Sentence))
            .collect::<Vec<_>>();
        candidates.extend(
            clause_end_tokens
                .iter()
                .copied()
                .filter(|end_byte| *end_byte > start_byte)
                .map(|end_byte| (end_byte, NarrativeBoundary::Clause)),
        );
        let fallback_end = find_largest_fitting_end_byte(block, start_byte, hard_max, counter)?;
        candidates.push((fallback_end, NarrativeBoundary::TokenWindow));

        let Some((end_byte, boundary)) = choose_best_narrative_boundary(
            block,
            start_byte,
            target_tokens,
            hard_max,
            &candidates,
            counter,
        )?
        else {
            return Err(KboltError::Inference(
                "failed to choose a fitting narrative split boundary".to_string(),
            )
            .into());
        };

        if matches!(
            boundary,
            NarrativeBoundary::Sentence | NarrativeBoundary::Clause
        ) {
            used_structural_boundary = true;
        }

        out.push(split_block_range_by_bytes(block, start_byte, end_byte));
        if end_byte == block.text.len() {
            break;
        }

        start_byte = next_start_byte(block, start_byte, end_byte, overlap, counter)?;
    }

    Ok(used_structural_boundary.then_some(out))
}

fn split_block_range_by_bytes(
    block: &ExtractedBlock,
    byte_start: usize,
    byte_end: usize,
) -> ExtractedBlock {
    debug_assert!(byte_start < byte_end, "byte range must be non-empty");
    debug_assert!(
        byte_end <= block.text.len(),
        "byte range exceeds block text"
    );

    let mut split = block.clone();
    split.offset = block.offset.saturating_add(byte_start);
    split.length = byte_end.saturating_sub(byte_start);
    split.text = block.text[byte_start..byte_end].to_string();
    split
}

fn choose_best_narrative_boundary(
    block: &ExtractedBlock,
    start_byte: usize,
    target_tokens: usize,
    hard_max: usize,
    candidates: &[(usize, NarrativeBoundary)],
    counter: &dyn TokenCounter,
) -> Result<Option<(usize, NarrativeBoundary)>> {
    let mut best: Option<(usize, NarrativeBoundary, i64)> = None;
    for (end_byte, boundary) in candidates {
        if *end_byte <= start_byte {
            continue;
        }

        let candidate = split_block_range_by_bytes(block, start_byte, *end_byte);
        let token_count = count_single_block_tokens(&candidate, counter)?;
        if token_count > hard_max {
            continue;
        }

        let boundary_score = match boundary {
            NarrativeBoundary::Sentence => 30,
            NarrativeBoundary::Clause => 15,
            NarrativeBoundary::TokenWindow => 0,
        };
        let distance = token_count.abs_diff(target_tokens) as i64;
        let score = boundary_score - (distance * 10);
        let replace = best
            .as_ref()
            .map(|(best_end_byte, _, best_score)| {
                score > *best_score || (score == *best_score && *end_byte > *best_end_byte)
            })
            .unwrap_or(true);
        if replace {
            best = Some((*end_byte, *boundary, score));
        }
    }

    Ok(best.map(|(end_byte, boundary, _)| (end_byte, boundary)))
}

fn find_largest_fitting_end_byte(
    block: &ExtractedBlock,
    start_byte: usize,
    hard_max: usize,
    counter: &dyn TokenCounter,
) -> Result<usize> {
    let text = block.text.as_str();
    let token_spans = token_byte_spans(text);
    let mut token_boundaries = token_spans
        .iter()
        .map(|(_, end)| *end)
        .filter(|end| *end > start_byte)
        .collect::<Vec<_>>();
    token_boundaries.dedup();

    if let Some(end_byte) =
        largest_fitting_boundary(block, start_byte, hard_max, &token_boundaries, counter)?
    {
        return Ok(end_byte);
    }

    let mut char_boundaries = text
        .char_indices()
        .map(|(idx, ch)| idx + ch.len_utf8())
        .filter(|end| *end > start_byte)
        .collect::<Vec<_>>();
    char_boundaries.dedup();
    largest_fitting_boundary(block, start_byte, hard_max, &char_boundaries, counter)?.ok_or_else(
        || {
            KboltError::Inference(format!(
                "failed to find a fitting split boundary for block at offset {}",
                block.offset
            ))
            .into()
        },
    )
}

fn largest_fitting_boundary(
    block: &ExtractedBlock,
    start_byte: usize,
    hard_max: usize,
    boundaries: &[usize],
    counter: &dyn TokenCounter,
) -> Result<Option<usize>> {
    if boundaries.is_empty() {
        return Ok(None);
    }

    let mut left = 0usize;
    let mut right = boundaries.len();
    let mut best = None;
    while left < right {
        let mid = left + (right - left) / 2;
        let end_byte = boundaries[mid];
        let candidate = split_block_range_by_bytes(block, start_byte, end_byte);
        let token_count = count_single_block_tokens(&candidate, counter)?;
        if token_count <= hard_max {
            best = Some(end_byte);
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    Ok(best)
}

fn next_start_byte(
    block: &ExtractedBlock,
    current_start_byte: usize,
    end_byte: usize,
    overlap: usize,
    counter: &dyn TokenCounter,
) -> Result<usize> {
    debug_assert!(
        end_byte > current_start_byte,
        "end byte must advance beyond start byte"
    );
    if overlap == 0 {
        return Ok(next_content_start_byte(block.text.as_str(), end_byte));
    }

    let text = block.text.as_str();
    let token_starts = token_byte_spans(text)
        .into_iter()
        .map(|(start, _)| start)
        .filter(|start| *start > current_start_byte && *start < end_byte)
        .collect::<Vec<_>>();
    if let Some(next_start) =
        earliest_fitting_overlap_start(block, end_byte, overlap, &token_starts, counter)?
    {
        return Ok(next_start);
    }

    let char_starts = text
        .char_indices()
        .map(|(idx, _)| idx)
        .filter(|idx| *idx > current_start_byte && *idx < end_byte)
        .collect::<Vec<_>>();
    Ok(
        earliest_fitting_overlap_start(block, end_byte, overlap, &char_starts, counter)?
            .map(|start| next_content_start_byte(text, start))
            .unwrap_or_else(|| next_content_start_byte(text, end_byte)),
    )
}

fn next_content_start_byte(text: &str, start_byte: usize) -> usize {
    if start_byte >= text.len() {
        return text.len();
    }

    for (idx, ch) in text[start_byte..].char_indices() {
        if !ch.is_whitespace() {
            return start_byte + idx;
        }
    }

    text.len()
}

fn earliest_fitting_overlap_start(
    block: &ExtractedBlock,
    end_byte: usize,
    overlap: usize,
    candidates: &[usize],
    counter: &dyn TokenCounter,
) -> Result<Option<usize>> {
    if candidates.is_empty() {
        return Ok(None);
    }

    let mut left = 0usize;
    let mut right = candidates.len();
    let mut best = None;
    while left < right {
        let mid = left + (right - left) / 2;
        let start_byte = candidates[mid];
        let candidate = split_block_range_by_bytes(block, start_byte, end_byte);
        let token_count = count_single_block_tokens(&candidate, counter)?;
        if token_count <= overlap {
            best = Some(start_byte);
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    Ok(best)
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

fn clause_end_token_indices(text: &str, spans: &[(usize, usize)]) -> Vec<usize> {
    spans
        .iter()
        .enumerate()
        .filter_map(|(index, (start, end))| token_ends_clause(&text[*start..*end]).then_some(index))
        .collect()
}

fn token_ends_sentence(token: &str) -> bool {
    let trimmed = token.trim_end_matches(['"', '\'', ')', ']', '}']);
    trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?')
}

fn token_ends_clause(token: &str) -> bool {
    let trimmed = token.trim_end_matches(['"', '\'', ')', ']', '}']);
    trimmed.ends_with(',') || trimmed.ends_with(';') || trimmed.ends_with(':')
}

#[cfg(test)]
fn best_narrative_cut(
    candidates: &[(usize, NarrativeBoundary)],
    target_tokens: usize,
    start_token: usize,
    total_tokens: usize,
) -> Option<usize> {
    candidates
        .iter()
        .copied()
        .max_by_key(|(end_token, boundary)| {
            (
                score_narrative_cut(
                    target_tokens,
                    start_token,
                    *end_token,
                    total_tokens,
                    *boundary,
                ),
                *end_token as i64,
            )
        })
        .map(|(end_token, _)| end_token)
}

#[cfg(test)]
fn score_narrative_cut(
    target_tokens: usize,
    start_token: usize,
    end_token: usize,
    total_tokens: usize,
    boundary: NarrativeBoundary,
) -> i64 {
    let chunk_tokens = end_token.saturating_sub(start_token) as i64;
    let distance = (chunk_tokens - target_tokens as i64).abs();
    let boundary_score = match boundary {
        NarrativeBoundary::Sentence => 30,
        NarrativeBoundary::Clause => 15,
        NarrativeBoundary::TokenWindow => 0,
    };
    let tiny_tail_penalty = {
        let tiny_tail_threshold = (target_tokens / 4).max(1);
        let tail = total_tokens.saturating_sub(end_token);
        if tail > 0 && tail < tiny_tail_threshold {
            20
        } else {
            0
        }
    };

    boundary_score - (distance * 10) - tiny_tail_penalty
}

fn debug_assert_valid_blocks(blocks: &[ExtractedBlock]) {
    for block in blocks {
        debug_assert_eq!(
            block.text.len(),
            block.length,
            "extractor invariant violated: text byte length and source length differ"
        );
    }
}

fn code_group_ranges(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut groups = Vec::new();
    let mut group_start: Option<usize> = None;
    let mut line_start = 0usize;

    while line_start < bytes.len() {
        let line_end = next_line_end_bytes(bytes, line_start);
        let content_end = trim_line_ending_bytes(bytes, line_start, line_end);
        let is_blank = is_blank_line_bytes(bytes, line_start, content_end);

        match (group_start, is_blank) {
            (None, false) => {
                group_start = Some(line_start);
            }
            (Some(start), true) => {
                let end = trim_trailing_newlines_bytes(bytes, line_start);
                if end > start {
                    groups.push((start, end));
                }
                group_start = None;
            }
            _ => {}
        }

        line_start = line_end;
    }

    if let Some(start) = group_start {
        let end = trim_trailing_newlines_bytes(bytes, bytes.len());
        if end > start {
            groups.push((start, end));
        }
    }

    groups
}

fn next_line_end_bytes(bytes: &[u8], start: usize) -> usize {
    let mut index = start;
    while index < bytes.len() {
        if bytes[index] == b'\n' {
            return index + 1;
        }
        index += 1;
    }
    bytes.len()
}

fn trim_line_ending_bytes(bytes: &[u8], start: usize, end: usize) -> usize {
    let mut content_end = end;
    while content_end > start && matches!(bytes[content_end - 1], b'\n' | b'\r') {
        content_end -= 1;
    }
    content_end
}

fn is_blank_line_bytes(bytes: &[u8], start: usize, end: usize) -> bool {
    bytes[start..end]
        .iter()
        .all(|byte| matches!(byte, b' ' | b'\t'))
}

fn trim_trailing_newlines_bytes(bytes: &[u8], end: usize) -> usize {
    let mut result = end;
    while result > 0 && matches!(bytes[result - 1], b'\n' | b'\r') {
        result -= 1;
    }
    result
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
        let has_header = blocks
            .iter()
            .any(|block| block.kind == BlockKind::TableHeader);
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

    if blocks
        .iter()
        .all(|block| block.kind == BlockKind::CodeFence)
    {
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
                BlockKind::Heading
                    | BlockKind::Paragraph
                    | BlockKind::ListItem
                    | BlockKind::BlockQuote
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
        best_narrative_cut, can_pack_together, chunk_document, chunk_document_with_counter,
        derive_chunk_kind, resolve_policy, score_narrative_cut, FinalChunkKind, NarrativeBoundary,
        TokenCounter, WhitespaceTokenCounter,
    };
    use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument};

    fn baseline_config() -> ChunkingConfig {
        ChunkingConfig {
            defaults: ChunkPolicy {
                target_tokens: 800,
                soft_max_tokens: 950,
                hard_max_tokens: 1200,
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
    fn chunk_kind_parses_storage_labels() {
        assert_eq!(
            FinalChunkKind::try_from("section").expect("parse section"),
            FinalChunkKind::Section
        );
        assert_eq!(
            FinalChunkKind::try_from("paragraph").expect("parse paragraph"),
            FinalChunkKind::Paragraph
        );
        assert_eq!(
            FinalChunkKind::try_from("code").expect("parse code"),
            FinalChunkKind::Code
        );
        assert_eq!(
            FinalChunkKind::try_from("table").expect("parse table"),
            FinalChunkKind::Table
        );
        assert_eq!(
            FinalChunkKind::try_from("mixed").expect("parse mixed"),
            FinalChunkKind::Mixed
        );
    }

    #[test]
    fn chunk_kind_rejects_unknown_storage_labels() {
        let err = FinalChunkKind::try_from("unknown").expect_err("unknown label should fail");
        assert!(err.to_string().contains("invalid stored chunk kind"));
    }

    #[test]
    fn whitespace_token_counter_counts_word_boundaries() {
        let counter = WhitespaceTokenCounter;
        assert_eq!(counter.count("").expect("count empty"), 0);
        assert_eq!(counter.count("alpha").expect("count token"), 1);
        assert_eq!(
            counter
                .count("alpha beta\tgamma\n\ndelta")
                .expect("count whitespace"),
            4
        );
    }

    struct SeparatorAwareCounter;

    impl TokenCounter for SeparatorAwareCounter {
        fn count(&self, text: &str) -> crate::Result<usize> {
            Ok(text.split_whitespace().count() + text.matches("\n\n").count())
        }
    }

    struct CharCountCounter;

    impl TokenCounter for CharCountCounter {
        fn count(&self, text: &str) -> crate::Result<usize> {
            Ok(text.chars().count())
        }
    }

    #[test]
    fn chunk_document_with_counter_sizes_candidate_chunk_text_not_additive_blocks() {
        let policy = ChunkPolicy {
            target_tokens: 2,
            soft_max_tokens: 2,
            hard_max_tokens: 8,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![
                block_with(BlockKind::Paragraph, "alpha", 0, &[]),
                block_with(BlockKind::Paragraph, "beta", 8, &[]),
            ],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document_with_counter(&document, &policy, &SeparatorAwareCounter)
            .expect("chunk with separator-aware counter");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "alpha");
        assert_eq!(chunks[1].text, "beta");
    }

    #[test]
    fn chunk_document_with_counter_falls_back_to_char_boundaries_for_single_oversized_token() {
        let policy = ChunkPolicy {
            target_tokens: 4,
            soft_max_tokens: 4,
            hard_max_tokens: 4,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![block_with(BlockKind::CodeFence, "abcdefghij", 0, &[])],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document_with_counter(&document, &policy, &CharCountCounter)
            .expect("chunk oversized single token");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "abcd");
        assert_eq!(chunks[1].text, "efgh");
        assert_eq!(chunks[2].text, "ij");
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

        let chunks =
            chunk_document_with_counter(&document, &policy, &counter).expect("chunk empty input");
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
        assert!(chunks
            .iter()
            .all(|chunk| chunk.kind == FinalChunkKind::Code));
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
        assert!(chunks
            .iter()
            .all(|chunk| chunk.kind == FinalChunkKind::Table));
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
    fn chunk_document_flushes_on_heading_transition_even_under_budget() {
        let policy = ChunkPolicy {
            target_tokens: 20,
            soft_max_tokens: 50,
            hard_max_tokens: 50,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![
                block_with(BlockKind::Heading, "# Intro", 0, &[]),
                block_with(BlockKind::Paragraph, "alpha beta", 8, &["Intro"]),
                block_with(BlockKind::Heading, "## Setup", 20, &["Intro"]),
                block_with(BlockKind::Paragraph, "gamma delta", 30, &["Intro", "Setup"]),
            ],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].kind, FinalChunkKind::Section);
        assert_eq!(chunks[0].heading.as_deref(), Some("Intro"));
        assert_eq!(chunks[1].kind, FinalChunkKind::Section);
        assert_eq!(chunks[1].heading.as_deref(), Some("Intro > Setup"));
    }

    #[test]
    fn chunk_document_flushes_when_narrative_heading_path_changes() {
        let policy = ChunkPolicy {
            target_tokens: 20,
            soft_max_tokens: 50,
            hard_max_tokens: 50,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![
                block_with(BlockKind::Paragraph, "alpha beta", 0, &["Intro"]),
                block_with(BlockKind::Paragraph, "gamma delta", 12, &["Setup"]),
            ],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 2);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.kind == FinalChunkKind::Paragraph));
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
    fn can_pack_together_rejects_heading_start_when_chunk_is_open() {
        let paragraph = block_with(BlockKind::Paragraph, "text", 0, &["Intro"]);
        let heading = block_with(BlockKind::Heading, "## Setup", 10, &["Intro"]);
        assert!(!can_pack_together(&paragraph, &heading));
    }

    #[test]
    fn can_pack_together_rejects_narrative_heading_path_mismatch() {
        let intro = block_with(BlockKind::Paragraph, "alpha", 0, &["Intro"]);
        let setup = block_with(BlockKind::Paragraph, "beta", 10, &["Setup"]);
        assert!(!can_pack_together(&intro, &setup));
    }

    #[test]
    fn can_pack_together_treats_html_as_opaque() {
        let html = block_with(BlockKind::HtmlBlock, "<div>x</div>", 0, &[]);
        let html2 = block_with(BlockKind::HtmlBlock, "<div>y</div>", 20, &[]);
        let paragraph = block_with(BlockKind::Paragraph, "text", 40, &[]);
        assert!(!can_pack_together(&html, &html2));
        assert!(!can_pack_together(&html, &paragraph));
    }

    #[test]
    fn token_split_makes_progress_when_overlap_exceeds_window() {
        let policy = ChunkPolicy {
            target_tokens: 2,
            soft_max_tokens: 2,
            hard_max_tokens: 2,
            boundary_overlap_tokens: 10,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![block_with(BlockKind::CodeFence, "a b c d e f", 0, &[])],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 5);
        assert!(chunks.iter().all(|chunk| !chunk.text.is_empty()));
    }

    #[test]
    fn narrative_sentence_split_makes_progress_with_high_overlap() {
        let policy = ChunkPolicy {
            target_tokens: 4,
            soft_max_tokens: 4,
            hard_max_tokens: 4,
            boundary_overlap_tokens: 10,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![block_with(
                BlockKind::Paragraph,
                "one two. three four. five six. seven eight.",
                0,
                &[],
            )],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|chunk| !chunk.text.is_empty()));
    }

    #[test]
    fn code_forced_split_prefers_blank_line_boundaries() {
        let policy = ChunkPolicy {
            target_tokens: 4,
            soft_max_tokens: 4,
            hard_max_tokens: 6,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![block_with(
                BlockKind::CodeFence,
                "a1 a2 a3 a4\n\na5 a6 a7 a8\n\na9 a10 a11 a12",
                0,
                &[],
            )],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "a1 a2 a3 a4");
        assert_eq!(chunks[1].text, "a5 a6 a7 a8");
        assert_eq!(chunks[2].text, "a9 a10 a11 a12");
        assert!(chunks
            .iter()
            .all(|chunk| chunk.kind == FinalChunkKind::Code));
    }

    #[test]
    fn narrative_forced_split_uses_clause_boundaries_when_sentences_absent() {
        let policy = ChunkPolicy {
            target_tokens: 3,
            soft_max_tokens: 3,
            hard_max_tokens: 3,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        };
        let document = ExtractedDocument {
            blocks: vec![block_with(
                BlockKind::Paragraph,
                "alpha beta, gamma delta, epsilon zeta, eta theta",
                0,
                &[],
            )],
            metadata: HashMap::new(),
            title: None,
        };

        let chunks = chunk_document(&document, &policy);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].text, "alpha beta,");
        assert_eq!(chunks[1].text, "gamma delta,");
        assert_eq!(chunks[2].text, "epsilon zeta,");
        assert_eq!(chunks[3].text, "eta theta");
    }

    #[test]
    fn score_narrative_cut_prefers_sentence_boundary_over_clause() {
        let sentence = score_narrative_cut(8, 0, 8, 20, NarrativeBoundary::Sentence);
        let clause = score_narrative_cut(8, 0, 8, 20, NarrativeBoundary::Clause);
        let token = score_narrative_cut(8, 0, 8, 20, NarrativeBoundary::TokenWindow);
        assert!(sentence > clause);
        assert!(clause > token);
    }

    #[test]
    fn score_narrative_cut_prefers_proximity_to_target() {
        let close = score_narrative_cut(8, 0, 8, 20, NarrativeBoundary::Sentence);
        let far = score_narrative_cut(8, 0, 5, 20, NarrativeBoundary::Sentence);
        assert!(close > far);
    }

    #[test]
    fn best_narrative_cut_penalizes_tiny_tail_when_choices_are_similar() {
        let selected = best_narrative_cut(
            &[
                (8, NarrativeBoundary::Sentence),
                (9, NarrativeBoundary::Sentence),
            ],
            8,
            0,
            10,
        )
        .expect("pick best cut");
        assert_eq!(selected, 8);
    }
}
