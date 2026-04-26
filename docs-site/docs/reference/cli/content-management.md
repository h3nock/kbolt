# Content management

This page groups the commands that define what gets indexed and where it lives.

## `space`

Use `space` to create, inspect, and manage top-level namespaces.

Key subcommands:

- `add`
- `describe`
- `rename`
- `remove`
- `current`
- `default`
- `list`
- `info`

Examples:

```bash
kbolt space add work
kbolt space default work
kbolt space current
kbolt space list
```

`space add` can also register directories as collections in one step:

```bash
kbolt space add work ./api ./runbooks
```

## `collection`

Use `collection` to register directories and manage their metadata.

Key subcommands:

- `add`
- `list`
- `info`
- `describe`
- `rename`
- `remove`

Examples:

```bash
kbolt collection add /path/to/docs --name my_docs
kbolt collection list
kbolt collection info my_docs
```

Important options on `collection add`:

- `--name`
- `--description`
- `--extensions`
- `--no-index`

Use `--no-index` when you want to register first and run the initial indexing pass later.

## `update`

Use `update` to re-scan and re-index collections immediately.

If the watcher is enabled, normal file changes are picked up automatically. `update` is still useful after bulk changes or when you want fresh results right now.

Examples:

```bash
kbolt update
kbolt update --collection my_docs
kbolt update --dry-run
kbolt update --no-embed
```

Important options:

- `--collection <COLLECTIONS>`
- `--no-embed`
- `--dry-run`
- `--verbose`

## `watch`

Use `watch` to keep collections fresh automatically:

```bash
kbolt watch enable
kbolt watch status
kbolt watch logs
```

The watcher covers all configured collections. Use it for directories you edit regularly.

Use `schedule` instead when you prefer periodic batch refreshes over a live watcher.

## `ignore`

Use `ignore` to add gitignore-style patterns on top of the built-in default ignores and the collection's own `.gitignore`.

Key subcommands:

- `show`
- `add`
- `remove`
- `edit`
- `list`

Examples:

```bash
kbolt ignore add my_docs "*.pdf"
kbolt ignore show my_docs
kbolt ignore edit my_docs
```

After changing ignore rules, the watcher picks up the change automatically if it is running. Run `update` when you want the index refreshed immediately.

## Related pages

- [Add and organize content](../../guides/add-and-organize-content.md)
- [Spaces and collections](../../concepts/spaces-and-collections.md)
- [Keep indexes fresh](../../guides/keep-indexes-fresh.md)
- [Exclude files](../../guides/exclude-files.md)
- [Watch](watch.md)
- [Data locations](../../operations/data-locations.md)
