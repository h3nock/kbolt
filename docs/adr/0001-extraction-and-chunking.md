# ADR 0001: Extraction and Chunking Strategy

- Status: Accepted
- Date: 2026-03-02
- Decision owners: Core indexing and search

## Context

Kbolt needs a V1 extraction/chunking strategy that:

1. Improves retrieval quality for markdown-heavy collections.
2. Stays deterministic and easy to debug.
3. Extends cleanly to additional file types.
4. Preserves snippet fidelity (`offset`/`length` based reads from source files).

Prior draft behavior used a flat chunk budget and did not define a stable, file-independent chunking contract.

## Decision

Kbolt will use a two-stage pipeline:

1. Structure-aware extraction into a normalized intermediate representation (IR).
2. File-independent chunking with tiered token budgets.

### 1) Extraction contract

Extractors emit typed structural blocks with source spans and metadata.

Required fields per block:

- `text`
- `kind` (paragraph, list item, code fence, table row, etc.)
- `heading_path`
- `offset` and `length` (source-file byte span)
- optional attributes (for example code language, table header labels)

Extractor output is consumed by the chunker. The chunker does not parse raw file syntax.

### 2) Chunking policy model

Chunking uses a three-tier budget:

- `target_tokens`: preferred chunk size.
- `soft_max_tokens`: tolerated overrun to preserve structural boundaries.
- `hard_max_tokens`: absolute ceiling requiring split.

Additional policy fields:

- `boundary_overlap_tokens`: overlap only when a forced hard split occurs.
- `neighbor_window`: retrieval-time context expansion (`±N` neighboring chunks by sequence).
- `contextual_prefix`: whether to prepend deterministic context for retrieval/embedding text.

### 3) Policy precedence

Policy resolution order (target architecture):

1. CLI override (for experiments)
2. Collection-level override (deferred in V1)
3. File-type profile (`md`, `code`, `txt`)
4. Global default

V1 supports `CLI > profile > global`. Collection-level override storage is deferred until collection metadata/schema support is added.

### 4) Markdown-specific extraction and split behavior

Markdown extraction:

- Parse CommonMark/GFM constructs.
- Track heading hierarchy and attach breadcrumb to every block.
- Emit typed blocks for paragraphs, lists, quotes, code fences, table headers/rows, and HTML blocks.
- Read and store frontmatter metadata.

Markdown chunking:

1. Pack adjacent compatible blocks toward `target_tokens`.
   - Narrative packing is heading-scoped: a new heading always starts a fresh chunk.
   - Heading + narrative body may pack together inside that section.
2. Allow growth to `soft_max_tokens` if it preserves clean boundaries.
3. If still over `hard_max_tokens`, split by block-specific fallback:
   - paragraph/list/quote: sentence then clause boundaries
   - code fence: blank-line boundaries, then fixed token windows
   - table: row groups with table-header carryover in retrieval text
4. Apply `boundary_overlap_tokens` only in step 3 forced splits.

Storage-level chunk kind is derived from block composition:

- code-fence only → `Code`
- table-only (header/rows) → `Table`
- paragraph/list/quote-only → `Paragraph`
- heading-scoped narrative → `Section`
- forced heterogeneous merge/split → `Mixed`

### 5) Two text views per chunk

Each stored chunk has:

1. Source text view: exact file slice for snippet and citation fidelity.
2. Retrieval text view: optional contextual prefix + source text for BM25/embedding/reranking.

Contextual prefixes are deterministic (document title, heading path, selected frontmatter, code/table hints). Prefixes do not change source spans.
Retrieval text is computed from source text + metadata during indexing/query stages; it is not persisted as a second chunks-table text column.

### 6) Retrieval-time context expansion

Search result assembly may fetch neighboring chunks using `neighbor_window` (default `1`) by chunk sequence. This provides context without storing large blanket overlap across all chunks.
Default ranking behavior keeps reranker scoring on the primary hit chunk; neighbor-expanded context is for result context, not for changing the ranking unit.

## Initial V1 defaults

Markdown profile defaults:

- `target_tokens = 800`
- `soft_max_tokens = 950`
- `hard_max_tokens = 1200`
- `boundary_overlap_tokens = 48`
- `neighbor_window = 1`
- `contextual_prefix = true`

Rationale:

- `target_tokens = 800` keeps materially more local context without changing FiQA/SciFact retrieval quality in a meaningful way versus the previous 450-token baseline.
- `soft_max_tokens = 950` still preserves clean structural packing without forcing small overflow fragments.
- `hard_max_tokens = 1200` gives the packer enough headroom to avoid unnecessary splits while remaining well below the local embedding truncation ceiling.
- `boundary_overlap_tokens = 48` protects recall only at forced split boundaries while avoiding broad duplicate indexing.

## Consequences

Positive:

- Better section coherence than flat token windows.
- Cleaner extension model for new file types.
- Better auditability due to explicit spans and typed blocks.
- Lower duplication than broad storage-time overlap.

Tradeoffs:

- More extractor and policy logic than a simple splitter.
- Additional testing required for markdown edge cases (tables, large code fences, malformed markdown).

## Non-goals for this ADR

- Defining model architecture for reranking and expansion.
- Final benchmark thresholds.
- V2-only techniques (for example full late-chunking rollout, hierarchical retrieval trees).

## Implementation follow-ups

1. Add `ChunkingConfig` and per-profile policy loading in config.
2. Implement extractor IR in `core/ingest/extract`.
3. Implement file-independent chunker in `core/ingest/chunk`.
4. Update update/search flows to use neighbor expansion and contextual prefixes.
5. Add unit and integration coverage for markdown edge cases and policy precedence.
