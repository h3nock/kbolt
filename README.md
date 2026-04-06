# kbolt

`kbolt` is a local-first retrieval engine for indexing local notes and docs and searching them with keyword, semantic, reranked, and deep retrieval modes.

## What kbolt supports

- Index Markdown and plaintext files from one or more local directories
- Group collections into spaces and scope search with `--space` or `--collection`
- Search with keyword, semantic, hybrid reranked, and deep retrieval modes
- Fetch underlying source files with `kbolt get`, `kbolt multi-get`, and `kbolt ls`
- Check readiness with `kbolt doctor` and `kbolt models list`
- Serve the index to agents over MCP with `kbolt mcp`
- Run retrieval benchmarks with `kbolt eval ...`
- Schedule recurring re-indexing with `kbolt schedule ...`

## How it works

1. Register one or more directories as collections with `kbolt collection add`.
2. `kbolt` extracts content, chunks documents, builds the keyword index, and stores embeddings when configured.
3. Search across all indexed content or scope to a space or collection.
4. Use `get`, `multi-get`, and `ls` to pull the source files behind search results.

`kbolt collection add` indexes immediately by default. Use `--no-index` if you only want to register a collection first.

## Install

Recommended on macOS and Linux x86_64 with Homebrew:

```bash
brew install h3nock/kbolt/kbolt
```

For Rust users:

```bash
cargo install kbolt
```

Prebuilt binaries are also available from [GitHub Releases](https://github.com/h3nock/kbolt/releases).

If `llama-server` is not already available and you did not install via Homebrew, install `llama.cpp` by following the official [llama.cpp install guide](https://github.com/ggml-org/llama.cpp/wiki).

## Quick start

Once `kbolt` and `llama-server` are available, run:

```bash
kbolt setup local
kbolt doctor
kbolt collection add /path/to/docs
kbolt search "your query"
```

`kbolt setup local` downloads the default local embedder and reranker models, starts managed `llama-server` processes, and writes the local provider bindings into the kbolt config directory.

To enable deep search later:

```bash
kbolt local enable deep
```

## Search modes

Use the default search first:

```bash
kbolt search "rust error handling"
```

Other supported modes:

- `kbolt search "query" --keyword` for keyword-only retrieval
- `kbolt search "query" --semantic --no-rerank` for dense-only retrieval
- `kbolt search "query" --deep` for query expansion plus reranked retrieval

## Common workflows

### Organize content with spaces

```bash
kbolt space add work
kbolt --space work collection add /path/to/repo-docs
kbolt space default work
```

### Inspect indexed files

```bash
kbolt ls repo-docs
kbolt get docs/guide.md
kbolt multi-get docs/guide.md,docs/api.md
```

### Re-index after changes

```bash
kbolt update
kbolt --space work update --verbose
```

### Check setup and model readiness

```bash
kbolt doctor
kbolt models list
kbolt local status
```

## Other capabilities

- `kbolt mcp` starts the stdio MCP server for agent integrations
- `kbolt eval import beir ...` and `kbolt eval run` support retrieval evaluation
- `kbolt ignore ...` manages collection-specific ignore patterns
- `kbolt schedule ...` manages recurring re-indexing on macOS and Linux
- Advanced users can configure provider profiles manually and use `kbolt doctor` to verify them

## Configuration paths

Kbolt stores configuration and cache data in platform directories:

- config: `dirs::config_dir()/kbolt`
- cache: `dirs::cache_dir()/kbolt`

On macOS that resolves to:

- `~/Library/Application Support/kbolt`
- `~/Library/Caches/kbolt`

See `kbolt.md` for the detailed command and architecture reference.

## Development

From source:

```bash
cargo build --release
./target/release/kbolt --help
```
