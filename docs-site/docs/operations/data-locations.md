# Data locations

`kbolt` keeps config and cache data under the standard user config and cache directories for the current operating system.

## Default roots

### macOS

- config root: `~/Library/Application Support/kbolt`
- cache root: `~/Library/Caches/kbolt`

### Linux

- config root: `~/.config/kbolt`
- cache root: `~/.cache/kbolt`

### Windows

- config root: `%APPDATA%\\kbolt`
- cache root: `%LOCALAPPDATA%\\kbolt`

For the resolved paths on the current machine, run:

```bash
kbolt doctor
```

## Important files and directories

### Config root

- `index.toml`: main config file
- `ignores/<space>/<collection>.ignore`: custom ignore patterns
- `schedules.toml`: saved schedule definitions

### Cache root

- `meta.sqlite`: SQLite metadata store
- `spaces/`: per-space search indexes
- `models/`: downloaded local model files
- `run/`: pid files for managed local services and the watcher
- `watch/state.json`: watcher runtime state while `kbolt watch` is running
- `logs/`: managed local-service logs and `watch.log`

## What to check when debugging

Use the config root when you need to inspect:

- provider bindings
- default space
- ignore patterns
- schedules

Use the cache root when you need to inspect:

- whether the index exists yet
- local model downloads
- managed-service pid files and logs
- watcher state and logs

## Related pages

- [Health and status](health-and-status.md)
- [Troubleshooting](troubleshooting.md)
- [Keep indexes fresh](../guides/keep-indexes-fresh.md)
- [index.toml](../reference/config/index-toml.md)
