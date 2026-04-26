# Exclude files

Use ignore rules to keep generated files, build artifacts, and irrelevant directories out of the index.

## What kbolt ignores by default

`kbolt` already skips common noisy directories and files, including:

- `.git`
- `node_modules`
- `target`
- `dist`
- `build`
- `.next`
- `.turbo`
- `.venv`
- `venv`
- `__pycache__`
- `.pytest_cache`
- `.cache`
- `coverage`
- `.DS_Store`
- `*.lock`

These defaults apply even if you do not add custom ignore rules.

## `.gitignore` support

Inside a collection root, `kbolt` respects the collection's `.gitignore`.

That means:

- files ignored by the collection's own `.gitignore` are skipped
- parent `.gitignore` files outside the collection root are not applied

This keeps collection behavior local and predictable.

## Add a custom ignore rule

To add one pattern:

```bash
kbolt ignore add my_docs "*.pdf"
```

To inspect the current rules for a collection:

```bash
kbolt ignore show my_docs
```

To remove one exact pattern:

```bash
kbolt ignore remove my_docs "*.pdf"
```

To edit the ignore file directly:

```bash
kbolt ignore edit my_docs
```

To list all collections that currently have custom ignore rules:

```bash
kbolt ignore list
```

## Refresh after changing ignore rules

Ignore rules only affect future indexing work. If the watcher is running, it refreshes the affected collection automatically after the ignore file changes.

If you want the new exclusion set applied immediately, run:

```bash
kbolt update --collection my_docs
```

## Good candidates for custom ignore rules

Use custom rules for:

- generated API docs
- vendored snapshots
- export folders
- logs
- binary assets that do not help retrieval

## Next steps

- For the grouped command reference, see [Content management](../reference/cli/content-management.md).
- For automatic freshness, see [Keep indexes fresh](keep-indexes-fresh.md).
- For indexing behavior and default local setup, see [Quickstart](../quickstart.md).
