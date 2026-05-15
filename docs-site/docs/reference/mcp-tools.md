# MCP tools

`kbolt mcp` exposes a small, retrieval-focused MCP surface.

## Tool list

### `search`

Search indexed content with optional space and collection filters.

Inputs:

- `query`
- `space`
- `collection`
- `limit`
- `mode`
- `no_rerank`

Notes:

- default limit is `10`
- supported modes are `auto`, `deep`, `keyword`, and `semantic`
- use `deep` for vocabulary-mismatch or underspecified queries; use `auto` for normal searches and lexically clear lookups

### `get`

Read one document by docid or collection-relative path.

Inputs:

- `identifier`
- `space`

### `multi_get`

Read multiple documents while enforcing file-count and byte budgets.

Inputs:

- `locators`
- `space`
- `max_files`
- `max_bytes`

Notes:

- default `max_files` is `20`
- default `max_bytes` is `51200`

### `list_files`

List indexed files in one collection, optionally filtered by path prefix.

Inputs:

- `space`
- `collection`
- `prefix`

### `status`

Return index status for the current or specified space.

Inputs:

- `space`

## When to use MCP vs the CLI

Use the CLI when you are working directly in a terminal.

Use MCP when another tool or agent needs to:

- search your indexed content
- read indexed content
- check status programmatically

## Related pages

- [Use with Claude Desktop](../guides/use-with-claude-desktop.md)
- [Read and integration](cli/read-and-integration.md)
