# Spaces and collections

`kbolt` organizes indexed content into spaces and collections.

## Space

A space is the top-level namespace for your index.

Use spaces when you want to separate content sets such as:

- work vs personal notes
- product docs vs source code
- one client project vs another

Common commands:

```bash
kbolt space add work
kbolt space list
kbolt space default work
kbolt space current
```

## Collection

A collection is one registered directory inside a space.

Use collections when you want to index separate directories but still search them together within one space.

Common commands:

```bash
kbolt --space work collection add /path/to/api --name api
kbolt --space work collection add /path/to/runbooks --name runbooks
kbolt --space work collection list
```

## Active space resolution

When a command needs a space, `kbolt` resolves it in this order:

1. `--space`
2. `KBOLT_SPACE`
3. configured default space

Use `kbolt space current` if you are not sure which one is active.

## The default path

If no default space exists, `kbolt setup local` creates and uses `default`.

That keeps the first-run path simple: one space, one or more collections, then search.

## When to create another space

Create another space when:

- you want separate indexes
- you do not want search results mixed together
- you want different operational scopes for update and status

Do not create new spaces just to give one directory a label. That is a collection.

## Next steps

- For a practical workflow, see [Add and organize content](../guides/add-and-organize-content.md).
- For the grouped command reference, see [Content management](../reference/cli/content-management.md).
- For the read path once content is indexed, see [Read and integration](../reference/cli/read-and-integration.md).
