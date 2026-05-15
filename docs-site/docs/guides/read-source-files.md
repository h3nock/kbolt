# Read indexed content

Search returns matching documents and snippets. Use the read commands when you want the indexed document text. For extracted formats such as HTML and digital PDFs, this is the canonical text stored in the index rather than raw file bytes.

## List files in a collection

```bash
kbolt ls my_docs
kbolt ls my_docs guides/
```

Use a prefix to narrow the listing to one part of the collection.

## Read one file

Use a collection-relative path:

```bash
kbolt get my_docs/path/to/file.md
```

Or use a document id from search output:

```bash
kbolt get '#abc123'
```

Read a line range with `--offset` and `--limit`:

```bash
kbolt get my_docs/path/to/file.md --offset 40 --limit 80
```

## Read several files

Use `multi-get` when an agent or script needs several documents in one call:

```bash
kbolt multi-get my_docs/a.md,my_docs/b.md
```

Set limits when you need bounded output:

```bash
kbolt multi-get my_docs/a.md,my_docs/b.md --max-files 10 --max-bytes 30000
```

## When to use each command

| Need | Command |
| --- | --- |
| See what is indexed | `kbolt ls` |
| Read one document | `kbolt get` |
| Read several documents with output limits | `kbolt multi-get` |
| Let an agent call read tools directly | `kbolt mcp` |

## Next steps

- Use these commands from agents with [Use with Claude Desktop](use-with-claude-desktop.md).
- See exact arguments in [Read and integration](../reference/cli/read-and-integration.md).
- Check indexed content with [Health and status](../operations/health-and-status.md).
