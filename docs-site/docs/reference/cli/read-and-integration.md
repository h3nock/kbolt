# Read and integration

This page groups the commands used to read indexed content and expose it to agent clients.

## `ls`

Use `ls` to list indexed files in a collection:

```bash
kbolt ls my_docs
kbolt ls my_docs guides/
```

Useful option:

- `--all`: include deactivated files

## `get`

Use `get` to read one indexed document by collection-relative path or docid:

```bash
kbolt get my_docs/path/to/file.md
kbolt get '#abc123'
```

Useful options:

- `--offset <OFFSET>`
- `--limit <LIMIT>`

## `multi-get`

Use `multi-get` to read multiple documents within file-count and byte budgets:

```bash
kbolt multi-get my_docs/a.md,my_docs/b.md
```

Useful options:

- `--max-files`
- `--max-bytes`

## `mcp`

Use `mcp` to start the MCP server for agent integration:

```bash
kbolt mcp
```

This command is usually launched by the MCP client rather than typed by hand in a normal terminal session.

## Related pages

- [Read source files](../../guides/read-source-files.md)
- [Use with Claude Desktop](../../guides/use-with-claude-desktop.md)
- [MCP tools](../mcp-tools.md)
- [Quickstart](../../quickstart.md)
