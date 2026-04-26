# Troubleshooting

Start with the command that best matches the failure:

- `kbolt doctor` for setup and readiness problems
- `kbolt local status` for managed local-service problems
- `kbolt watch status` for automatic freshness problems
- `kbolt status` for index and storage state

## `doctor` says `kbolt is not set up`

On a fresh machine, `doctor` exits with:

```text
kbolt is not set up

get started:
  kbolt setup local
```

That means the config has not been created yet. Run:

```bash
kbolt setup local
```

## `setup local` fails

Check these first:

- `llama-server` is installed and on your `PATH`
- the machine can download the default model files
- local ports for the managed services are available

Then rerun:

```bash
kbolt setup local
```

If setup completes but later readiness checks fail, inspect:

```bash
kbolt local status
```

If you installed with Homebrew, `llama-server` should come from the `llama.cpp` dependency. For Cargo or GitHub Release installs, install `llama-server` separately.

## `doctor` warns that the index does not exist yet

If you already ran `setup local` but have not indexed any collection yet, storage warnings are expected. The config and model stack can be healthy before any index files exist.

Create the index by adding a collection or running an update:

```bash
kbolt collection add /path/to/docs --name my_docs
```

or:

```bash
kbolt update
```

## Local services are configured but not reachable

You may see error text like:

```text
Connection refused (os error 61)
```

or a provider transport error that includes a local URL such as `http://127.0.0.1:8101`.

Run:

```bash
kbolt local status
```

Then try:

```bash
kbolt local start
```

If the services still do not stay up, inspect the log files under the `kbolt` cache directory. See [Data locations](data-locations.md).

## Command says `no active space`

You may see:

```text
no active space: use --space, set KBOLT_SPACE, or configure a default
```

Fix it by choosing one of these:

```bash
kbolt --space work search "query"
kbolt space default work
export KBOLT_SPACE=work
```

Use `kbolt space list` if you are not sure which spaces exist.

## Search returns no results

Check these in order:

1. the collection exists
2. the collection was indexed successfully
3. you are searching the intended space
4. the query mode matches the query shape

Useful commands:

```bash
kbolt collection list
kbolt status
kbolt search "query" --keyword
kbolt search "query" --semantic
```

If files changed after the first indexing pass, make sure the watcher is running:

```bash
kbolt watch status
```

If you need a refresh immediately, run:

```bash
kbolt update
```

## Changed files do not appear in search

Check whether automatic watching is enabled and running:

```bash
kbolt watch status
```

On macOS and Linux, if the watcher is disabled, start it:

```bash
kbolt watch enable
```

If the watcher reports an error, inspect recent logs:

```bash
kbolt watch logs --lines 200
```

If you need fresh keyword and metadata results immediately, run:

```bash
kbolt update --no-embed
```

If semantic search is stale while a file is actively changing, wait for the collection to become quiet or run a full update:

```bash
kbolt update
```

If `watch status` says semantic indexing is blocked for a space, run the exact `kbolt --space <space> update` command shown in the status output.

## Indexing is slower than expected

The most common cause is indexing too much irrelevant material.

Check:

- collection-local `.gitignore`
- default ignored directories
- custom `kbolt ignore` rules

Use:

```bash
kbolt ignore show my_docs
```

Then tighten the exclusion set. If the watcher is running, it will refresh the affected collection automatically; run `kbolt update` when you want the change applied immediately.

## Update output says more errors were omitted

When `update` or initial indexing has many file errors, the default output shows a short list and a hint such as:

```text
run with --verbose for the full list
```

Rerun the update with `--verbose`:

```bash
kbolt update --verbose
```

If the failed collection was scoped, keep the same `--space` and `--collection` values in the verbose command.

## Still stuck?

- Re-run [Quickstart](../quickstart.md) against a small test directory first.
- Compare the current paths with [Data locations](data-locations.md).
- Check automatic freshness in [Keep indexes fresh](../guides/keep-indexes-fresh.md).
- Use [Search modes](../concepts/search-modes.md) if the problem is retrieval quality rather than availability.
