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

Use `update` to re-scan and re-index collections after files change.

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

The watcher covers all configured collections and reuses the same indexing behavior as `kbolt update`.

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

After changing ignore rules, run `update` so the index reflects the new exclusion set.

## Related pages

- [Spaces and collections](../../concepts/spaces-and-collections.md)
- [Exclude files](../../guides/exclude-files.md)
- [Watch](watch.md)
- [Data locations](../../operations/data-locations.md)
