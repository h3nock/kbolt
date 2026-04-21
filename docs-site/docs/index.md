# kbolt

`kbolt` is a local-first retrieval engine for indexing notes, documentation, and source code, then searching them with keyword, semantic, reranked, and deep retrieval modes.

The default path is local: `kbolt setup local` downloads the local embedder and reranker models, starts managed `llama-server` processes, and writes the local provider bindings into your config.

`kbolt` also supports remote OpenAI-compatible providers. The product is local-first, not local-only.

## Start here

- If you have not installed `kbolt` yet, start with [Install](install.md).
- If you want a first successful search end to end, go to [Quickstart](quickstart.md).
- If you want to use the index from Claude Desktop, go to [Use with Claude Desktop](guides/use-with-claude-desktop.md).
- If something is already broken, start with [Troubleshooting](operations/troubleshooting.md).

Fast install on macOS or Linux with Homebrew:

```bash
brew install h3nock/kbolt/kbolt
```

## What kbolt does

- indexes Markdown, plaintext, and source code from one or more directories
- groups content into spaces and collections
- supports keyword, semantic, hybrid reranked, and deep retrieval modes
- exposes the index over MCP for agent workflows
- keeps local setup and health visible through `doctor`, `status`, and `models list`

## Common paths

### Index local files and search them

1. [Install](install.md)
2. [Quickstart](quickstart.md)
3. [Search modes](concepts/search-modes.md)

### Run the default local stack

1. [Quickstart](quickstart.md)
2. [Local setup](concepts/local-setup.md)
3. [Troubleshooting](operations/troubleshooting.md)

### Use kbolt from an agent client

1. [Use with Claude Desktop](guides/use-with-claude-desktop.md)
2. [MCP tools](reference/mcp-tools.md)
3. [Read and integration](reference/cli/read-and-integration.md)

### Manage multiple indexes

1. [Spaces and collections](concepts/spaces-and-collections.md)
2. [Content management](reference/cli/content-management.md)
3. [Exclude files](guides/exclude-files.md)

## What this site is for

This site is the public long-form documentation for `kbolt`.

Use it for:

- installation
- setup
- search usage
- CLI reference
- troubleshooting
- MCP integration

It is not the place for internal architecture records or maintainer release steps. Those stay in the repository.
