# Platform support

`kbolt` is developed and released primarily for macOS and Linux. Some commands also work on Windows through Cargo builds, but managed background services are not available there yet.

## At a glance

| Capability | macOS | Linux | Windows |
| --- | --- | --- | --- |
| Homebrew install | yes | yes, x86_64 | no |
| GitHub release binary | x86_64, aarch64 | x86_64 | no |
| Cargo install | yes | yes | yes |
| `kbolt setup local` | yes, with `llama-server` | yes, with `llama-server` | not covered by the public quickstart |
| `kbolt watch enable` | yes, via user `launchd` | yes, via user `systemd` | no managed service |
| `kbolt watch --foreground` | yes | yes | not covered as a public workflow |
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
