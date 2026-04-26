# CLI overview

This page maps the main command groups to the job they do.

## Core workflow

Use these commands for the normal local path:

- `kbolt setup local`
- `kbolt doctor`
- `kbolt collection add`
- `kbolt watch enable`
- `kbolt search`

## Command map

### Search and retrieval

- `search` for ranked retrieval
- `ls` for listing indexed files in a collection
- `get` for reading one document
- `multi-get` for reading multiple documents within size limits

Reference:

- [Search](cli/search.md)
- [Read and integration](cli/read-and-integration.md)

### Content management

- `space` for namespaces
- `collection` for registered directories
- `ignore` for gitignore-style exclusions
- `watch` for automatic freshness as files change
- `update` for immediate manual re-indexing
- `schedule` for periodic batch re-indexing

Reference:

- [Content management](cli/content-management.md)
- [Watch](cli/watch.md)
- [Schedule](cli/schedule.md)

Freshness rule of thumb:

- use `watch` for directories you edit
- use `update` when you need a refresh now
- use `schedule` when periodic batch work fits better than a live watcher

### Local setup and health

- `setup local` for the default local stack
- `local` for managed `llama-server` processes
- `doctor` for readiness checks
- `status` for index and storage summary
- `models list` for configured role bindings

Reference:

- [Local and health](cli/local-and-health.md)

### Agent integration

- `mcp` to start the MCP server

Reference:

- [Read and integration](cli/read-and-integration.md)
- [MCP tools](mcp-tools.md)

### Benchmarking

- `eval` for retrieval benchmarks and dataset import

Reference:

- [Eval](cli/eval.md)

## Output formats

Most commands support:

```bash
kbolt --format cli ...
kbolt --format json ...
```

Use `cli` for humans and `json` when another tool needs structured output.

## Next steps

- For the end-to-end path, see [Quickstart](../quickstart.md).
- For freshness choices, see [Keep indexes fresh](../guides/keep-indexes-fresh.md).
- For concepts that shape the CLI, see [Search modes](../concepts/search-modes.md) and [Spaces and collections](../concepts/spaces-and-collections.md).
