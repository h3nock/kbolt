# kbolt

`kbolt` is the command-line application for local-first indexing and retrieval over local notes and docs.

## What it supports

- Register local directories as collections
- Search with keyword, semantic, reranked, and deep modes
- Organize collections into spaces
- Fetch source files with `get`, `multi-get`, and `ls`
- Check readiness with `doctor` and `models list`
- Serve the index over MCP with `kbolt mcp`

## Typical flow

Install with Cargo:

```bash
cargo install kbolt
```

Or on macOS and Linux x86_64 with Homebrew:

```bash
brew install h3nock/kbolt/kbolt
```

If `llama-server` is not already available and you did not install via Homebrew, follow the official [llama.cpp install guide](https://github.com/ggml-org/llama.cpp/wiki).

Then run:

```bash
kbolt setup local
kbolt doctor
kbolt collection add /path/to/docs
kbolt search "your query"
```

`kbolt collection add` indexes immediately by default.

To enable deep search later:

```bash
kbolt local enable deep
```

This crate is the main user-facing package in the workspace. Most users should install and run `kbolt`, not the internal `kbolt-core`, `kbolt-mcp`, or `kbolt-types` crates directly.
