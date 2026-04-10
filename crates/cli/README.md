# kbolt

`kbolt` is a local-first retrieval engine for indexing local notes and docs and searching them with keyword, semantic, reranked, and deep retrieval modes.

## Install

macOS and Linux x86_64 with Homebrew:

```bash
brew install h3nock/kbolt/kbolt
```

Rust users on macOS, Linux, or Windows:

```bash
cargo install kbolt
```

Prebuilt binaries are also available from [GitHub Releases](https://github.com/h3nock/kbolt/releases).

If `llama-server` is not already installed, follow the official [llama.cpp install guide](https://github.com/ggml-org/llama.cpp/wiki).

## Quick start

Set up the default local retrieval stack:
```bash
kbolt setup local
kbolt doctor
```
`kbolt setup local` downloads the default local embedder and reranker models, starts managed `llama-server` processes, and writes the local provider bindings into the kbolt config directory.

Add a folder of notes or docs:
```bash
kbolt collection add /path/to/docs --name my_docs
```
Search the indexed content:
```bash
kbolt search "rust error handling"
```

## Search modes

Default:
- `kbolt search "query"`
- hybrid retrieval with keyword + semantic + reranking

Other supported modes:

- `kbolt search "query" --keyword` for keyword-only retrieval
- `kbolt search "query" --semantic --no-rerank` for semantic-only retrieval
- `kbolt search "query" --deep` for query expansion plus reranked retrieval

`kbolt setup local` configures the default local embedder and reranker. To enable deep search later:

```bash
kbolt local enable deep
```

## What kbolt supports

- Index Markdown and plaintext files from one or more local directories
- Group collections into spaces and scope search with `--space` or `--collection`
- Search with keyword, semantic, hybrid reranked, and deep retrieval modes
- Read underlying source files with `kbolt get`, `kbolt multi-get`, and `kbolt ls`
- Check readiness with `kbolt doctor` and `kbolt models list`
- Serve the index to agents over MCP with `kbolt mcp`
- Run retrieval benchmarks with `kbolt eval ...`
- Schedule recurring re-indexing with `kbolt schedule ...`

This crate is the main user-facing package in the workspace. Most users should install and run `kbolt`, not the internal `kbolt-core`, `kbolt-mcp`, or `kbolt-types` crates directly.
