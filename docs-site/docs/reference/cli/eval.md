# Eval

## Synopsis

```bash
kbolt eval run [--file <path>]
kbolt eval import beir --dataset <name> --source <dir> --output <dir> [--collection <name>]
```

## What eval does

`eval` is the benchmark surface for retrieval quality.

Use it to:

- run an evaluation from an `eval.toml` manifest
- import a BEIR dataset into a local benchmark corpus plus manifest

## `run`

Use `run` to evaluate the current index against an eval manifest:

```bash
kbolt eval run
kbolt eval run --file /path/to/eval.toml
```

Without `--file`, `kbolt` loads `eval.toml` from the config directory.

Important rules:

- the top-level `--space` flag is rejected for `eval`; set scope inside each eval case instead
- each case must include a non-empty `query`
- each case must include at least one `judgments` entry
- each case must include at least one judgment with `relevance > 0`
- judgment paths must be unique within each case
- referenced collections must already exist and have indexed chunks

### Minimal manifest shape

```toml
[[cases]]
query = "trait object vs generic"
space = "bench"
collections = ["rust"]
judgments = [
  { path = "rust/traits.md", relevance = 2 },
  { path = "rust/generics.md", relevance = 1 },
]
```

Each run reports metrics per search mode, including:

- `keyword`
- `auto`
- `auto+rerank`
- `semantic` when an embedder is configured
- `deep-norerank`
- `deep`

## `import beir`

Use `import beir` to turn an extracted BEIR dataset into:

- a `corpus/` directory with materialized Markdown documents
- an `eval.toml` manifest

Example:

```bash
kbolt eval import beir --dataset scifact --source /path/to/scifact --output /tmp/scifact-bench
```

### Required source layout

The source directory must contain:

```text
corpus.jsonl
queries.jsonl
qrels/test.tsv
```

This command always imports the BEIR `test` split.

### Import rules

- `--output` must point to an empty directory, or to a directory that does not exist yet
- `--collection` defaults to the dataset name
- imported corpus files are written as `<document-id>.md`
- the generated eval cases use the default benchmark space `bench`

After import, the usual path is:

```bash
kbolt space add bench
kbolt --space bench collection add /tmp/scifact-bench/corpus --name scifact --no-index
kbolt --space bench update --collection scifact
kbolt eval run --file /tmp/scifact-bench/eval.toml
```

## Related pages

- [CLI overview](../cli-overview.md)
- [Content management](content-management.md)
- [Data locations](../../operations/data-locations.md)
