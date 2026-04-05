# kbolt

`kbolt` is the command-line application for local-first document indexing and retrieval.

## Typical flow

Install `llama.cpp` so `llama-server` is available, then run:

```bash
kbolt setup local
kbolt collection add /path/to/docs
kbolt search "your query"
```

To enable deep search later:

```bash
kbolt local enable deep
```

This crate is the main user-facing package in the workspace. Most users should install and run `kbolt`, not the internal `kbolt-core`, `kbolt-mcp`, or `kbolt-types` crates directly.
