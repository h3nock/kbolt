# Search Speed Baseline and Hot Paths

- Status: Current baseline
- Date: 2026-03-24
- Scope: Search latency and hot-path analysis for the current global ranking-window default:
  - `initial_candidate_limit_min = 40`
  - `rerank_candidates_min = 20`
  - `rerank_candidates_max = 30`

## Why This Exists

Kbolt's speed profile is not uniform across modes:

- `keyword`, `semantic`, and `auto --no-rerank` are fast.
- `auto+rerank` is materially slower because it builds rerank inputs and runs local model inference.
- `deep` is the slowest mode because it adds expansion, repeated first-stage retrieval, and often reranking.

This document records the measured baseline for `40/20/30` and the current best understanding of the main bottlenecks.

## Benchmark Method

Benchmarks were run against:

- FiQA mixed-budget public benchmark slices:
  - `fast_300`
  - `heavy_100`
  - `query_40`
- SciFact mixed-budget public benchmark slices:
  - `fast_300`
  - `heavy_100`
  - `query_40`

The `40/20/30` configuration was selected because it was the best global compromise found in the window sweep:

- `20/10/20` was faster but too narrow.
- `100/40/50` was much slower and often noisier.
- `60/30/40` improved SciFact further, but degraded FiQA enough that it was not a good shared default.

## Current Speed Snapshot

### FiQA

| Suite | Mode | nDCG@10 | Recall@10 | MRR@10 | p50 | p95 |
|---|---|---:|---:|---:|---:|---:|
| `fast_300` | `keyword` | 0.2378 | 0.2920 | 0.3038 | 24 ms | 34 ms |
| `fast_300` | `auto` | 0.4184 | 0.5082 | 0.4736 | 25 ms | 37 ms |
| `fast_300` | `semantic` | 0.4308 | 0.5082 | 0.5000 | 16 ms | 18 ms |
| `heavy_100` | `semantic` | 0.3762 | 0.4671 | 0.4237 | 16 ms | 18 ms |
| `heavy_100` | `auto+rerank` | 0.4243 | 0.5440 | 0.4728 | 2651 ms | 3391 ms |
| `query_40` | `auto+rerank` | 0.3374 | 0.4308 | 0.4008 | 2673 ms | 3658 ms |
| `query_40` | `deep-norerank` | 0.2270 | 0.3142 | 0.2749 | 1862 ms | 13562 ms |
| `query_40` | `deep` | 0.3153 | 0.3604 | 0.4219 | 4512 ms | 15637 ms |

### SciFact

| Suite | Mode | nDCG@10 | Recall@10 | MRR@10 | p50 | p95 |
|---|---|---:|---:|---:|---:|---:|
| `fast_300` | `keyword` | 0.5296 | 0.6317 | 0.5059 | 7 ms | 10 ms |
| `fast_300` | `auto` | 0.7378 | 0.9212 | 0.6870 | 13 ms | 17 ms |
| `fast_300` | `semantic` | 0.7792 | 0.9212 | 0.7425 | 10 ms | 12 ms |
| `heavy_100` | `semantic` | 0.7721 | 0.9370 | 0.7340 | 10 ms | 12 ms |
| `heavy_100` | `auto+rerank` | 0.7679 | 0.8935 | 0.7379 | 4334 ms | 5547 ms |
| `query_40` | `auto+rerank` | 0.7772 | 0.9075 | 0.7499 | 4161 ms | 5844 ms |
| `query_40` | `deep-norerank` | 0.5985 | 0.7942 | 0.5463 | 2054 ms | 13657 ms |
| `query_40` | `deep` | 0.7532 | 0.9042 | 0.7258 | 6986 ms | 18320 ms |

## What The Numbers Mean

1. The fast modes are already fast.

- `keyword`, `semantic`, and `auto --no-rerank` are already in the 7-25 ms p50 range on these public benchmarks.
- Micro-optimizing those paths is not where the big wins are.

2. The expensive modes are the real latency problem.

- `auto+rerank` moves from milliseconds to multi-second latency.
- `deep` adds another large step on top of that.
- `deep-norerank` is still expensive, which means the expander and variant search path already cost a lot even before the reranker runs.

3. Tail latency is especially bad for deep.

- Both FiQA and SciFact show `deep` and `deep-norerank` p95s in the 13-18 second range.
- That points to variability in model inference and/or deep query expansion rather than only index lookups.

## Current Hot Path Picture

### 1) Local reranker inference is the dominant hot path

`assemble_search_results()` calls the reranker on representative documents after first-stage retrieval.

Relevant code:

- `crates/core/src/engine/search_ops.rs`
- `crates/core/src/models/local_reranker.rs`

Observed behavior:

- A live `sample` on an active benchmark process showed most time under:
  - `LocalQwen3Reranker::rerank`
  - `score_docs`
  - `llama_context::synchronize()`
  - Metal command buffer waits

This matches the measured latency jump when reranking is enabled.

### 2) The reranker is serialized and scores documents one-by-one

Current reranker behavior in `local_reranker.rs`:

- guarded by `inference_lock: Mutex<()>`
- tokenizes every `(query, document)` prompt independently
- creates one context per rerank request
- sets `n_seq_max(1)`
- loops over documents sequentially
- clears KV cache and decodes each document independently

This means Kbolt does not currently exploit multi-document batching inside the local reranker path.

### 3) Deep adds an LLM expansion step before retrieval

Current deep behavior in `search_ops.rs`:

- always calls the expander when deep is requested
- generates up to `deep_variants_max`
- embeds dense variants
- runs per-variant first-stage retrieval
- aggregates variant results
- may rerank after that

Even without reranking, `deep-norerank` is far slower than `keyword` / `semantic` / `auto`.

### 4) Rerank input construction does on-demand file I/O and chunk loading

Current rerank input behavior:

- `build_rerank_input()` calls `load_candidate_bytes()` and may read the full source file from disk
- `candidate_neighbors()` may load all chunks for a document from SQLite
- `finalize_search_results()` reassembles final text with neighbors for returned results

This design preserves source fidelity, but it adds synchronous file and SQLite work inside the slow path.

### 5) BM25 and dense retrieval are not the main bottleneck

`query_bm25()` and `query_dense()` are called many times, especially in deep mode, but the benchmark timings show that:

- first-stage retrieval without reranking stays in the millisecond range
- the multi-second jump starts when model-based expansion or reranking is involved

BM25 and vector retrieval are still worth tightening, but they are not the primary speed problem today.

### 6) Search-only processes still open a Tantivy writer

Current storage behavior:

- `Storage::open_space()` eagerly opens a Tantivy `IndexWriter`
- the writer is stored inside `SpaceIndexes` even for read-mostly search/eval processes

Why this matters:

- it acquires the index writer lock even when the current process only wants to search
- it makes repeatable profiling and concurrent search/update workflows more fragile
- it adds startup work and couples read-only search too tightly to update concerns

Why this is important from first principles:

- search and update are different responsibilities
- a search path should not need exclusive writer state just to answer read queries
- even if this is not the dominant per-query latency hotspot, it is still the wrong architecture boundary

## Optimization Priorities

### Highest ROI: Batch reranker scoring

Current issue:

- Kbolt reranks one document at a time in a serialized loop.

Best near-term direction:

- change the local reranker path to score multiple sequences in one context instead of `n_seq_max(1)` + `clear_kv_cache()` per document
- treat reranking as batched model inference, not a repeated single-item loop

Why this matters:

- this is the hottest path in the current profile
- it improves both `auto+rerank` and `deep`
- it preserves the current retrieval architecture

### High ROI: Stop reading source files on the rerank path

Current issue:

- reranking builds document text by loading source bytes and reconstructing text on the fly

Best near-term direction:

- persist a compact rerank/search text representation for each chunk (or representative document chunk) in SQLite
- keep source-fidelity snippet reads for final presentation, but stop depending on raw file reads during reranking

Why this matters:

- removes file I/O from the latency-sensitive path
- simplifies rerank input construction
- reduces repeated neighbor assembly work

### High ROI: Make deep selective instead of always-on

Current issue:

- deep always expands
- expansion is expensive and often unnecessary

Best near-term direction:

- add a strong-signal gate so deep skips expansion on easy queries
- only pay expansion cost when the base query is weak or ambiguous

Why this matters:

- deep is the slowest mode
- many queries do not need expansion to perform well

### Medium ROI: Make rerank windows per-mode

Current issue:

- one global window setting is too blunt

Best near-term direction:

- separate `auto+rerank` and `deep` candidate/rerank windows
- keep `auto+rerank` moderate
- allow `deep` to be wider only if its quality justifies the cost

Why this matters:

- FiQA and SciFact do not agree on the same ideal global setting
- a shared default is a compromise, not an optimum

### Medium ROI: Avoid full-document neighbor loads

Current issue:

- `get_chunks_for_document()` loads all chunks for a document when neighbor expansion only needs `seq ± N`

Best near-term direction:

- add a storage query that fetches only the neighbor rows actually needed around a target chunk

Why this matters:

- reduces SQLite work and allocation pressure on rerank/finalize paths
- keeps the same output contract

### Secondary ROI: Reduce redundant BM25 reader reloads / repeated setup work

Current issue:

- `query_bm25()` builds and reloads reader state on every search call
- search-only processes also open Tantivy writer state eagerly during space initialization

Best near-term direction:

- instrument first before changing this
- if it shows up in traces, keep reusable reader/searcher state where safe

Why this is not first:

- the measured gap between fast modes and reranked modes says model inference is the real bottleneck
- but this is still worth fixing as part of a cleaner read/write storage split

## Redesign-Level Options

### Option A: Move to phased ranking with a batched external reranker

This keeps the current high-level architecture:

1. first-stage lexical/dense retrieval
2. bounded rerank window
3. final result assembly

But it moves expensive inference onto a runtime built for batching.

Relevant references:

- [Vespa phased ranking](https://docs.vespa.ai/en/ranking/phased-ranking.html)
- [Vespa ranking overview](https://docs.vespa.ai/en/basics/ranking.html)

Why this is attractive:

- strongest immediate performance upside without rewriting retrieval
- preserves the current ranking model

### Option B: Replace cross-encoder reranking with late interaction retrieval

Relevant references:

- [PLAID: An Efficient Engine for Late Interaction Retrieval](https://arxiv.org/abs/2205.09707)

Why this is attractive:

- pushes more quality into retrieval itself
- can narrow or even remove the need for a slow second-stage cross-encoder on every query

Why this is a redesign:

- different indexing model
- different serving path
- higher implementation cost

### Option C: Move to a unified sparse+dense retrieval model

Relevant references:

- [BGE-M3](https://arxiv.org/abs/2402.03216)

Why this is attractive:

- one model family can provide sparse, dense, and multi-vector retrieval capabilities
- reduces architectural split between lexical and dense branches

Why this is a redesign:

- changes indexing, scoring, and evaluation assumptions
- not a drop-in speed patch

### Option D: Split read-only search state from write/update state

Why this is attractive:

- search processes would open reader/searcher state only
- update processes would own the writer and commit lifecycle
- removes unnecessary writer locking from search and eval flows
- gives cleaner performance boundaries and safer concurrent behavior

Why this is a redesign:

- changes `Storage` responsibilities and lifecycle
- requires explicit read/write state ownership instead of the current shared `SpaceIndexes` shape

## What Not To Do First

- Do not spend the next optimization cycle micro-tuning BM25 weights for speed.
- Do not widen windows again globally.
- Do not treat `deep` as the first target for speed wins before fixing reranker batching and rerank input construction.

## Recommended Next Steps

1. Add stage-level timing instrumentation:
   - first-stage retrieval
   - expander inference
   - rerank input assembly
   - reranker inference
   - final result assembly
2. Rework the local reranker to support true multi-document batching.
3. Persist rerank/search text so reranking no longer depends on raw file reads.
4. Add deep gating before doing any larger deep-specific tuning.
