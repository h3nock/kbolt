# Health and status

`kbolt` has several status commands. Use the one that matches what you are checking.

## Which command should you run?

| Question | Command |
| --- | --- |
| Is kbolt configured and ready? | `kbolt doctor` |
| What is indexed in this space? | `kbolt status` |
| Are local model services running? | `kbolt local status` |
| Is automatic watching running? | `kbolt watch status` |
| Which model roles are configured? | `kbolt models list` |
| Are schedules installed? | `kbolt schedule status` |

## Start with `doctor`

Run:

```bash
kbolt doctor
```

Use `doctor` when setup or search looks wrong. It checks the config, provider bindings, local readiness, and index state.

## Check indexed content

Run:

```bash
kbolt status
```

If you use multiple spaces, scope it:

```bash
kbolt --space work status
```

Use this when you need to know whether the index exists, how much content is indexed, and whether model roles are ready.

## Check local services

Run:

```bash
kbolt local status
```

Use this when local inference endpoints are unreachable or a model-backed search mode fails.

## Check automatic freshness

Run:

```bash
kbolt watch status
```

Use this when changed files do not appear in search, or when you want to confirm the background watcher is running.

Inspect logs when status reports an issue:

```bash
kbolt watch logs --lines 200
```

## Check model bindings

Run:

```bash
kbolt models list
```

Use this when you need to know which providers are bound to embedding, reranking, or expansion.

## Next steps

- Fix common failures with [Troubleshooting](troubleshooting.md).
- Find state files and logs in [Data locations](data-locations.md).
- Learn local service behavior in [Local setup](../concepts/local-setup.md).
