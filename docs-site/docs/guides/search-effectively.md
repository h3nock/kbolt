# Search effectively

Start with the default search, then switch modes only when the query needs it.

## Start with the default mode

```bash
kbolt search "rust error handling"
```

The default mode combines keyword and semantic retrieval without reranking. It is the right first pass for most searches.

## Use exact matching for identifiers

Use keyword search when exact text matters:

```bash
kbolt search "E0502" --keyword
kbolt search "rerank_candidates_max" --keyword
```

Good fits include compiler errors, function names, config keys, file names, and CLI flags.

## Use semantic search for concepts

Use semantic search when the query is a paraphrase or a natural-language description:

```bash
kbolt search "how does kbolt recover from partial indexing failures" --semantic
```

Semantic search depends on embeddings, so newly edited files may lag until semantic indexing has refreshed.

## Add reranking when ordering matters

If the default search finds plausible results but the order is weak, add reranking:

```bash
kbolt search "rust error handling" --rerank
```

Reranking improves ordering but adds latency.

## Use deep search as a second pass

Use deep search when the query may not share vocabulary with the best matching documents, or when a short/underspecified query needs broader recall:

```bash
kbolt search "index corruption recovery" --deep
```

Deep search expands the query and runs a broader multi-variant retrieval path. It runs expansion on every search and is slower than the default mode and `--rerank`.

Good fits include vocabulary mismatch, short queries, and searches where normal search or `--rerank` missed useful material. Poor fits include exact titles, named entities, identifiers, config keys, and other lexically clear lookups.

## Diagnose search behavior

Use debug output when you need to inspect the retrieval pipeline:

```bash
kbolt search "rust error handling" --debug
```

## Next steps

- Learn the modes in [Search modes](../concepts/search-modes.md).
- See exact flags in [Search reference](../reference/cli/search.md).
- If changed files are missing from search, see [Keep indexes fresh](keep-indexes-fresh.md).
