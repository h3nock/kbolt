# kbolt

`kbolt` indexes local notes, documentation, source code, HTML, and digital PDFs, then searches them with keyword, semantic, reranked, and deep retrieval modes.

Most users start with the local setup path. `kbolt setup local` downloads the default local models, starts managed `llama-server` processes, and writes provider bindings into your config.

Remote OpenAI-compatible providers are supported through `index.toml`. The quickstart documents the local model stack.

## Start here

| I want to... | Go to |
| --- | --- |
| Install kbolt | [Install](install.md) |
| Get one successful search | [Quickstart](quickstart.md) |
| Add more directories or spaces | [Add and organize content](guides/add-and-organize-content.md) |
| Keep changed files searchable | [Keep indexes fresh](guides/keep-indexes-fresh.md) |
| Search better | [Search effectively](guides/search-effectively.md) |
| Read indexed content behind search results | [Read indexed content](guides/read-source-files.md) |
| Use kbolt from Claude Desktop | [Use with Claude Desktop](guides/use-with-claude-desktop.md) |
| Fix setup or freshness problems | [Health and status](operations/health-and-status.md) |
| Check platform support | [Platform support](concepts/platform-support.md) |
| Look up command details | [CLI overview](reference/cli-overview.md) |

## Common paths

### First local index

1. [Install](install.md)
2. [Quickstart](quickstart.md)
3. [Keep indexes fresh](guides/keep-indexes-fresh.md)

### Day-to-day search

1. [Search effectively](guides/search-effectively.md)
2. [Search modes](concepts/search-modes.md)
3. [Read indexed content](guides/read-source-files.md)

### Agent workflow

1. [Use with Claude Desktop](guides/use-with-claude-desktop.md)
2. [Keep indexes fresh](guides/keep-indexes-fresh.md)
3. [MCP tools](reference/mcp-tools.md)

### Index management

1. [Add and organize content](guides/add-and-organize-content.md)
2. [Spaces and collections](concepts/spaces-and-collections.md)
3. [Exclude files](guides/exclude-files.md)

## What kbolt does

- indexes Markdown, plaintext, HTML, digital PDFs, and source code from local directories
- groups content into spaces and collections
- searches with keyword, semantic, hybrid reranked, and deep retrieval modes
- reads indexed content through CLI and MCP tools
- keeps collections fresh automatically on macOS and Linux
- runs the default local model stack through managed `llama-server` processes

Guides solve user jobs. Reference pages give exact command and config details.
