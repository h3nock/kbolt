# Search modes

`kbolt` supports a small set of search modes. Start with the default mode, then switch only when the query calls for narrower matching or broader recall.

## Default mode

Run:

```bash
kbolt search "rust error handling"
```

The default mode is hybrid:

- keyword retrieval
- semantic retrieval
- no reranking unless you ask for it

This is the right first pass for most note and code search.

!!! note
    The default mode does not rerank. Pass `--rerank` when you want the extra ranking step.

## `--rerank`

Use `--rerank` when the result set is plausible but the ordering needs to be stronger:

```bash
kbolt search "rust error handling" --rerank
```

This keeps the default hybrid retrieval path and adds cross-encoder reranking on top.

Do not use `--rerank` for every query just because it produces better ordering. The cross-encoder pass adds real latency, so the default path should stay the fast path.

## `--keyword`

Use `--keyword` when exact tokens matter more than meaning:

```bash
kbolt search "E0502" --keyword
```

Good fits:

- compiler errors
- identifiers
- filenames
- config keys
- exact flag names

## `--semantic`

Use `--semantic` when the query is conceptual or paraphrased:

```bash
kbolt search "how to recover from partial index failures" --semantic
```

Good fits:

- conceptual questions
- paraphrases
- natural-language descriptions of behavior

## `--deep`

Use `--deep` when the first pass is too narrow and you need broader recall:

```bash
kbolt search "index corruption recovery" --deep
```

Deep mode expands the query and runs a broader retrieval path. It is slower than the default mode and belongs in a second pass, not the first one.

`--deep` reranks by default. If you want the broader retrieval path without the reranking step, pass `--no-rerank`:

```bash
kbolt search "index corruption recovery" --deep --no-rerank
```

If you use the managed local stack and want deep mode locally, enable the optional expander first:

```bash
kbolt local enable deep
```

## `--debug`

Use `--debug` when you need to inspect the retrieval pipeline rather than just read results:

```bash
kbolt search "rust error handling" --debug
```

This exposes pipeline stages and per-signal scores in the CLI output.

!!! note
    `--deep`, `--keyword`, and `--semantic` are mutually exclusive. Rerank flags apply only to the default mode and to `--deep`. `--keyword` and `--semantic` always skip reranking.

## Which mode should you pick?

- Use the default mode for general search.
- Add `--rerank` when ranking quality matters more than latency.
- Use `--keyword` for exact terms.
- Use `--semantic` for conceptual language.
- Use `--deep` when the first pass misses useful material.
- Use `--debug` when you are diagnosing search behavior.

## Next steps

- For the exact command contract, see [Search](../reference/cli/search.md).
- For local model setup, see [Local setup](local-setup.md).
- If results still look wrong, see [Troubleshooting](../operations/troubleshooting.md).
