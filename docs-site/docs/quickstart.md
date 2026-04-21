# Quickstart

This path assumes you want the default local stack: local embeddings, local reranking, and a first successful search against your own files.

## Prerequisites

Before you start:

- install `kbolt`
- install `llama-server` if your install path did not already provide it
- choose a directory you want to index

If you have not done those steps yet, start with [Install](install.md).

## 1. Set up the local stack

Run:

```bash
kbolt setup local
```

On the first run, this downloads the default local models, writes provider bindings into `index.toml`, and starts managed `llama-server` processes for the embedder and reranker.

The first run downloads roughly 1 GB of model files. On a slow connection, this step can take a while before the setup summary appears.

## 2. Verify readiness

Run:

```bash
kbolt doctor
```

On a healthy local setup, `doctor` should confirm that the config parses, the local roles are bound, and the configured services are reachable.

## 3. Add a directory as a collection

Run:

```bash
kbolt collection add /path/to/docs --name my_docs
```

This registers the directory as a collection and runs the initial indexing pass unless you add `--no-index`.

## 4. Search the indexed content

Run:

```bash
kbolt search "rust error handling"
```

If you want the same default retrieval path with stronger ranking quality, rerun with:

```bash
kbolt search "rust error handling" --rerank
```

## 5. Read the underlying files

Search gets you the matching documents and snippets. Use the read commands to inspect the source files directly:

```bash
kbolt ls my_docs
kbolt get my_docs/path/to/file.md
```

## What success looks like

At this point you should have:

- a working `index.toml`
- healthy local services
- at least one indexed collection
- successful search results from your own content

## If something fails

- If `doctor` says `kbolt is not set up`, run `kbolt setup local` first.
- If local services are not reachable, see [Local setup](concepts/local-setup.md) and [Troubleshooting](operations/troubleshooting.md).
- If indexing succeeds but search returns nothing useful, see [Search modes](concepts/search-modes.md).
- If `update` or initial indexing says more errors were omitted, rerun the suggested command with `--verbose`.

## Next steps

- Learn the retrieval options in [Search modes](concepts/search-modes.md).
- If you want to use the index from Claude Desktop, go to [Use with Claude Desktop](guides/use-with-claude-desktop.md).
- If you want to manage multiple indexes, read [Spaces and collections](concepts/spaces-and-collections.md).
