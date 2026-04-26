# Keep indexes fresh

`kbolt` searches its local index, not the filesystem directly. When files change, the index needs a refresh before new content appears in search.

For most local collections, use the watcher:

```bash
kbolt watch enable
```

The watcher runs in the background on macOS and Linux. It watches every configured collection and refreshes changed content automatically.

## Choose the right freshness path

| Use case | Command |
| --- | --- |
| Keep normal local notes, docs, or source trees fresh | `kbolt watch enable` |
| Refresh immediately after a large change | `kbolt update` |
| Refresh a specific collection now | `kbolt update --collection my_docs` |
| Run periodic batch refreshes instead of a watcher | `kbolt schedule add --every 1h` |
| Run under your own supervisor | `kbolt watch --foreground` |

`watch` is the recommended default for directories you edit regularly. `update` is still useful when you want an immediate refresh instead of waiting for the watcher quiet window. `schedule` is better for archive-style collections or machines where you prefer predictable batch work.

## What the watcher refreshes

The watcher separates cheap keyword freshness from more expensive semantic embeddings.

After files change:

- keyword search refreshes after the collection is quiet for a short window
- continuously changing files still get keyword refreshes periodically
- semantic embeddings wait for a longer quiet period so in-progress drafts are not embedded repeatedly
- periodic safety scans catch missed filesystem events

The practical effect is that exact-token and keyword-style searches become current first. Semantic-only searches may lag while a file is actively being edited.

## Check the watcher

Use status first:

```bash
kbolt watch status
```

Then inspect logs when status reports an error or the index does not look fresh:

```bash
kbolt watch logs
kbolt watch logs --lines 200
```

Use JSON when another tool needs structured state:

```bash
kbolt --format json watch status
```

## Force a refresh

If you need fresh results right now, run:

```bash
kbolt update
```

Use `--no-embed` when you only need metadata and keyword search refreshed:

```bash
kbolt update --no-embed
```

If semantic indexing is blocked by dense-vector repair, `kbolt watch status` shows the exact `kbolt --space <space> update` command to run.

## MCP and agent clients

`kbolt mcp` does not start the watcher. Start the watcher once from a terminal:

```bash
kbolt watch enable
```

After that, Claude Desktop and other MCP clients can search the same fresh index through `kbolt mcp`.

## Stop watching

Disable the background watcher:

```bash
kbolt watch disable
```

To stop tracking one directory entirely, remove its collection:

```bash
kbolt collection remove my_docs
```

## Next steps

- [Watch reference](../reference/cli/watch.md)
- [Content management](../reference/cli/content-management.md)
- [Troubleshooting](../operations/troubleshooting.md)
