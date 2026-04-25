# Watch

## Synopsis

```bash
kbolt watch
kbolt watch enable
kbolt watch disable
kbolt watch status
kbolt watch logs [--lines <N>]
kbolt watch --foreground
```

## What watch does

`watch` keeps configured collections fresh after files change. It does not maintain a separate indexer path. File events are treated as signals, and the watcher calls the same `Engine::update` logic used by `kbolt update`.

The normal user path is:

```bash
kbolt watch enable
```

On macOS this installs a user `launchd` agent. On Linux this installs a user `systemd` service. No `sudo` is required.

## Freshness behavior

The watcher separates cheap freshness from expensive semantic work:

- file events are coalesced by the native watcher
- keyword-only refresh waits for a quiet window before running
- continuously changing files are still refreshed for keyword search after a cap
- semantic embeddings run after a longer quiet window
- safety scans run periodically with `--no-embed` to catch missed filesystem events

This means keyword search becomes fresh automatically during active editing. Semantic search waits until the collection has been quiet, avoiding repeated embeddings of in-progress drafts. If semantic indexing is blocked by space-level dense repair, status shows the exact `kbolt --space <space> update` command to run.

## Commands

### `enable`

Enable and start the background watcher:

```bash
kbolt watch enable
```

Running `enable` again is safe. It rewrites the service definition if the installed binary path changed, then keeps the service active.

### `disable`

Disable and stop the background watcher:

```bash
kbolt watch disable
```

### `status`

Show service and runtime state:

```bash
kbolt watch status
```

`kbolt watch` without a subcommand is the same as `kbolt watch status`.

Use JSON when another tool needs structured state:

```bash
kbolt --format json watch status
```

### `logs`

Show recent watcher activity:

```bash
kbolt watch logs
kbolt watch logs --lines 200
```

Logs are rotated by size and kept under the kbolt cache directory.

### `--foreground`

Run the watcher attached to the current terminal:

```bash
kbolt watch --foreground
```

This is for debugging, development, or custom supervision. For normal use, prefer `kbolt watch enable`.

If the managed background service is already running, foreground mode refuses to start so two watchers do not compete over the same work.

## Scope

`watch` always watches all configured collections. It rejects the top-level `--space` flag because watch freshness is a background property of the whole local index.

To stop watching a directory, remove the collection:

```bash
kbolt collection remove my_docs
```

## Related pages

- [CLI overview](../cli-overview.md)
- [Content management](content-management.md)
- [Data locations](../../operations/data-locations.md)
