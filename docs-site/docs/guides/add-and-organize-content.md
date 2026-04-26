# Add and organize content

This guide shows the normal path for registering directories and deciding when to use spaces or collections.

## Recommended path

Start with one space and one collection:

```bash
kbolt collection add /path/to/docs --name my_docs
```

This registers the directory and runs the initial indexing pass.

Search it:

```bash
kbolt search "deployment checklist" --collection my_docs
```

## Add more collections

A collection is one registered directory. Add another collection when you want another directory searchable in the same space:

```bash
kbolt collection add /path/to/runbooks --name runbooks
kbolt collection list
```

Collections in the same space can be searched together. Use `--collection` only when you want to narrow the search.

## Create another space

A space is a separate namespace. Create another space when results should not mix:

```bash
kbolt space add work
kbolt --space work collection add /path/to/api --name api
```

Set a default when you want commands to use that space without passing `--space`:

```bash
kbolt space default work
```

Check the active space:

```bash
kbolt space current
```

## Register now, index later

Use `--no-index` when you want to register a collection first and run indexing later:

```bash
kbolt collection add /path/to/docs --name my_docs --no-index
kbolt update --collection my_docs
```

## Verify

Check the collection:

```bash
kbolt collection info my_docs
kbolt status
```

List indexed files:

```bash
kbolt ls my_docs
```

## Next steps

- Keep changed files searchable with [Keep indexes fresh](keep-indexes-fresh.md).
- Learn the mental model in [Spaces and collections](../concepts/spaces-and-collections.md).
- See exact commands in [Content management](../reference/cli/content-management.md).
