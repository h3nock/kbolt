# Platform support

`kbolt` is developed primarily on macOS and tested in CI on macOS and Linux. Windows support is best-effort through Cargo builds, and managed background services are not available there yet.

## At a glance

| Capability | macOS | Linux | Windows |
| --- | --- | --- | --- |
| Homebrew install | yes | yes, x86_64 | no |
| GitHub release binary | x86_64, aarch64 | x86_64 | no |
| Cargo install | yes | yes, including aarch64 when native dependencies build | yes, best effort |
| `kbolt setup local` | yes, with `llama-server` | yes, with `llama-server` | best effort; requires `llama-server` |
| `kbolt watch enable` | yes, via user `launchd` | yes, via user `systemd` | no managed service |
| `kbolt watch --foreground` | yes | yes | manual process |
| `kbolt schedule` | yes, via user `launchd` | yes, via user `systemd` | no managed scheduler |

## macOS

Homebrew is the shortest path:

```bash
brew install h3nock/kbolt/kbolt
```

The release workflow also builds x86_64 and aarch64 macOS archives.

Managed watcher and schedule commands install user-level `launchd` entries. They do not require `sudo`.

## Linux

Homebrew is supported on Linux x86_64 systems that use Homebrew:

```bash
brew install h3nock/kbolt/kbolt
```

The release workflow also builds a Linux x86_64 archive.

Managed watcher and schedule commands install user-level `systemd` units. They do not require `sudo`.

For Linux aarch64, use Cargo:

```bash
cargo install kbolt
```

There is no prebuilt Linux aarch64 release archive yet.

## Windows

Windows users should install with Cargo:

```bash
cargo install kbolt
```

Managed watcher and schedule commands are not supported on Windows. Use manual `kbolt update` or run `kbolt watch --foreground` under your own supervision.

## Next steps

- Install kbolt with [Install](../install.md).
- Keep files fresh with [Keep indexes fresh](../guides/keep-indexes-fresh.md).
- Check paths in [Data locations](../operations/data-locations.md).
