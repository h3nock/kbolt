# Use with Claude Desktop

This guide connects `kbolt` to Claude Desktop over MCP so Claude can search and read your indexed content.

## Prerequisites

Before you wire Claude Desktop to `kbolt`:

- `kbolt` must be installed and runnable from your shell
- at least one collection should already be indexed
- on macOS and Linux, `kbolt watch enable` should be running if you want changed files to stay searchable automatically
- Claude Desktop must support MCP stdio servers

If you have not indexed anything yet, start with [Quickstart](../quickstart.md).

## 1. Add the MCP server entry

Add a `kbolt` server entry under `mcpServers` in your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "kbolt": {
      "command": "kbolt",
      "args": ["mcp"]
    }
  }
}
```

If `kbolt` is not on your `PATH`, replace `command` with the absolute path to the binary.

!!! note
    The exact Claude Desktop config file location depends on platform and Claude Desktop version. Use the Claude Desktop documentation for the file path, but keep the `kbolt` server entry itself as shown above.

## 2. Restart Claude Desktop

Claude Desktop reads the MCP configuration on startup. Restart it after editing the config file.

## 3. Verify the connection

After restart, Claude should be able to use the `kbolt` MCP tools:

- `search`
- `get`
- `multi_get`
- `list_files`
- `status`

You can verify this by asking Claude to search your indexed content or by checking the MCP tools list in the client.

## 4. Keep the index current

Claude can only search what `kbolt` has already indexed.

For normal use on macOS and Linux, enable the watcher once from a terminal:

```bash
kbolt watch enable
```

`kbolt mcp` does not start the watcher itself. The MCP server and the watcher are separate processes that share the same local index.

When you need an immediate refresh, run:

```bash
kbolt update
```

If you use multiple spaces and want to force one space immediately, scope the manual update:

```bash
kbolt --space work update
```

## Common failures

### Claude Desktop cannot start the server

Check:

- the `command` path points to a real `kbolt` binary
- `kbolt --help` works in your shell
- the JSON syntax is valid

### Claude connects, but search returns nothing

Check:

- the collection was indexed successfully
- you are searching the right space
- the index is current

Use:

```bash
kbolt status
kbolt watch status
kbolt doctor
```

## Next steps

- For the exact tool surface, see [MCP tools](../reference/mcp-tools.md).
- For freshness behavior, see [Keep indexes fresh](keep-indexes-fresh.md).
- For the CLI command behind this integration, see [Read and integration](../reference/cli/read-and-integration.md).
