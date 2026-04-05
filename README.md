# kbolt

`kbolt` is a local-first retrieval engine for indexing document collections and searching them with keyword, semantic, reranked, and deep retrieval modes.

## Current usage

Install `llama.cpp` so `llama-server` is available, then run:

```bash
kbolt setup local
kbolt collection add /path/to/docs
kbolt search "your query"
```

`kbolt setup local` downloads the default local embedder and reranker models, starts managed `llama-server` processes, and writes the local provider bindings into the kbolt config directory.

To enable deep search later:

```bash
kbolt local enable deep
```

## Configuration paths

Kbolt stores configuration and cache data in platform directories:

- config: `dirs::config_dir()/kbolt`
- cache: `dirs::cache_dir()/kbolt`

On macOS that resolves to:

- `~/Library/Application Support/kbolt`
- `~/Library/Caches/kbolt`

## Development

From source:

```bash
cargo build --release
./target/release/kbolt --help
```
