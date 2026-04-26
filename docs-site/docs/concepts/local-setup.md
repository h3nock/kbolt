# Local setup

The default `kbolt` path is a managed local stack built on `llama-server`.

## What `setup local` does

Run:

```bash
kbolt setup local
```

On the default path, this:

- creates `index.toml` if it does not exist yet
- sets the default space to `default` if needed
- writes local provider profiles and role bindings
- downloads the default local embedder and reranker models
- starts managed `llama-server` processes for those roles

## What stays managed

The managed local stack owns:

- model cache under the `kbolt` cache directory
- pid files for managed services
- service logs
- local provider endpoints written into config

If a managed local endpoint goes down later, `kbolt` now makes one automatic restart attempt before surfacing a transport error. During that recovery attempt, the CLI prints a short stderr notice. This applies only to the provider profiles written by `kbolt setup local` and `kbolt local enable deep`.

You can inspect the current state with:

```bash
kbolt local status
```

## Start and stop

Use:

```bash
kbolt local start
kbolt local stop
```

These commands manage the local `llama-server` processes without changing the rest of the index state.

## Optional deep search

Deep search uses a separate expander role. Enable it explicitly:

```bash
kbolt local enable deep
```

This downloads the optional expander model and wires the corresponding local role.

## Local-first, not local-only

`kbolt` also supports remote OpenAI-compatible provider profiles. The local setup path is the default because it is the shortest way to a working stack, not because it is the only supported configuration.

## Next steps

- For the first end-to-end path, go to [Quickstart](../quickstart.md).
- For platform-specific support, see [Platform support](platform-support.md).
- For the grouped command reference, see [Local and health](../reference/cli/local-and-health.md).
- If local services are not reachable, see [Troubleshooting](../operations/troubleshooting.md).
