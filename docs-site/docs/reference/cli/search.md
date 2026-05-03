# Search

## Synopsis

```bash
kbolt search [OPTIONS] <QUERY>
```

## Arguments

- `<QUERY>`: the search query

## Options

- `--collection <COLLECTIONS>`: restrict search to specific collections as a comma-separated list
- `--limit <LIMIT>`: maximum number of results to return
- `--min-score <MIN_SCORE>`: filter out results below a score threshold
- `--deep`: query expansion plus multi-variant retrieval for vocabulary-mismatch or underspecified queries
- `--keyword`: keyword-only search
- `--semantic`: dense-vector-only search
- `--no-rerank`: skip cross-encoder reranking
- `--rerank`: enable reranking on the default auto mode
- `--debug`: show pipeline stages and per-signal scores

## Behavior notes

- `--deep`, `--keyword`, and `--semantic` are mutually exclusive.
- The default search mode is hybrid retrieval without reranking unless `--rerank` is set.
- `--deep` reranks by default; use `--no-rerank` if you want the broader retrieval path without the reranking pass.
- `--keyword` and `--semantic` bypass the default hybrid path.
- `--keyword` and `--semantic` always skip reranking.
- normal CLI output groups chunk matches by document and may show `+N more matching sections` when multiple hits came from the same file
- `--format json` returns the raw chunk-level `SearchResponse` from the engine without CLI grouping
- `--debug` changes the text presentation layer so you can inspect raw chunk hits, pipeline stages, and per-signal scores

## Examples

Default hybrid search:

```bash
kbolt search "rust error handling"
```

Default hybrid search with reranking:

```bash
kbolt search "rust error handling" --rerank
```

Keyword-only search:

```bash
kbolt search "E0502" --keyword
```

Semantic-only search:

```bash
kbolt search "partial index recovery" --semantic
```

Deep search:

```bash
kbolt search "index corruption recovery" --deep
```

Scoped search:

```bash
kbolt --space work search "deploy checklist" --collection runbooks
```

## Related pages

- [Search effectively](../../guides/search-effectively.md)
- [Search modes](../../concepts/search-modes.md)
- [Quickstart](../../quickstart.md)
- [Troubleshooting](../../operations/troubleshooting.md)
