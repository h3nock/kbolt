# Local and health

This page groups the commands used to bootstrap the default local stack and inspect its health.

## `setup local`

Use `setup local` to configure the default local stack:

```bash
kbolt setup local
```

This writes local provider bindings, downloads the default local models, and starts managed `llama-server` processes for the embedder and reranker.

## `local`

Use `local` to inspect and control the managed local services.

Subcommands:

- `status`
- `start`
- `stop`
- `enable`

Examples:

```bash
kbolt local status
kbolt local start
kbolt local stop
kbolt local enable deep
```

## `doctor`

Use `doctor` when you need a structured readiness check:

```bash
kbolt doctor
```

`doctor` is the first command to run when setup or search looks wrong.

## `status`

Use `status` to inspect the index state, storage usage, and model readiness:

```bash
kbolt status
```

If you use multiple spaces, scope it explicitly:

```bash
kbolt --space work status
```

## `models list`

Use `models list` to inspect the configured role bindings and whether each one is ready:

```bash
kbolt models list
```

## Related pages

- [Local setup](../../concepts/local-setup.md)
- [Quickstart](../../quickstart.md)
- [Troubleshooting](../../operations/troubleshooting.md)
