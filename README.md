# kbolt

`kbolt` is a local-first retrieval engine for indexing local notes and docs and searching them with keyword, semantic, reranked, and deep retrieval modes.

Full documentation: <https://h3nock.github.io/kbolt/>

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
On macOS and Linux, keep it fresh automatically:
```bash
kbolt watch enable
```
Search the indexed content:
```bash
kbolt search "rust error handling"
```

## Search modes

`kbolt search "query"` runs hybrid keyword + semantic retrieval. Reranking is opt-in for this default mode.

- `kbolt search "query"`: hybrid keyword + semantic
- `kbolt search "query" --rerank`: hybrid + reranking (higher quality, slower)
- `kbolt search "query" --keyword`: keyword only
- `kbolt search "query" --semantic`: dense only
- `kbolt search "query" --deep`: explicit query expansion + multi-variant retrieval, reranked by default

`kbolt setup local` configures the default local embedder and reranker. To enable deep search later:

```bash
kbolt local enable deep
```

Use `--deep` when the query may not share vocabulary with the best matching documents, or when a short/underspecified query needs broader recall. It runs query expansion on every search and is slower than the default search and `--rerank`; for exact titles, named entities, or lexically clear lookups, start with normal search or `--rerank`.

## What kbolt supports

- Index Markdown, plaintext, HTML, digital PDFs, and source code (Rust, Python, JS/TS, Go, Java, Kotlin, C/C++, C#, Ruby, PHP, Swift) from one or more local directories
- Group collections into spaces and scope search with `--space` or `--collection`
- Search with keyword, semantic, hybrid reranked, and deep retrieval modes
- Read indexed content with `kbolt get`, `kbolt multi-get`, and `kbolt ls`
- Check indexed content and disk usage with `kbolt status`
- Re-scan and re-index changed files with `kbolt update`
- Keep collections fresh automatically on macOS and Linux with `kbolt watch enable`
- Exclude files with gitignore-style patterns via `kbolt ignore`
- Check readiness with `kbolt doctor` and `kbolt models list`
- Run local models via `llama-server` or bind remote OpenAI-compatible endpoints through provider profiles
- Serve the index to agents over MCP with `kbolt mcp`
- Run retrieval benchmarks with `kbolt eval ...`
- Schedule recurring re-indexing with `kbolt schedule ...`
