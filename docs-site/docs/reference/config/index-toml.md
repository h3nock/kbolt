# `index.toml`

`index.toml` is the main `kbolt` config file.

## Location

By default, `index.toml` lives under the `kbolt` config directory:

- macOS: `~/Library/Application Support/kbolt/index.toml`
- Linux: `~/.config/kbolt/index.toml`
- Windows: `%APPDATA%\\kbolt\\index.toml`

For the resolved paths on the current machine, run:

```bash
kbolt doctor
```

## What it controls

`index.toml` controls:

- the default active space
- inference provider profiles
- role bindings for embedding, reranking, and query expansion
- reaping, chunking, and ranking defaults

If a section is omitted, `kbolt` fills in built-in defaults for `reaping`, `chunking`, and
`ranking`. `providers` and `roles` default to empty tables, and `default_space` defaults to no
override.

## Top-level keys

`index.toml` supports these top-level keys:

- `default_space`
- `providers`
- `roles`
- `reaping`
- `chunking`
- `ranking`

## Minimal local shape

```toml
default_space = "default"

[providers.local_embed]
kind = "llama_cpp_server"
operation = "embedding"
base_url = "http://127.0.0.1:8101"
model = "embeddinggemma"
timeout_ms = 30000
max_retries = 2

[providers.local_rerank]
kind = "llama_cpp_server"
operation = "reranking"
base_url = "http://127.0.0.1:8102"
model = "qwen3-reranker"
timeout_ms = 30000
max_retries = 2

[roles.embedder]
provider = "local_embed"
batch_size = 32

[roles.reranker]
provider = "local_rerank"

[reaping]
days = 7
```

## Provider profiles

Each provider lives under `[providers.<name>]`.

Supported `kind` values:

| Kind | Required fields | Optional fields |
| --- | --- | --- |
| `llama_cpp_server` | `operation`, `base_url`, `model` | `timeout_ms`, `max_retries` |
| `openai_compatible` | `operation`, `base_url`, `model` | `api_key_env`, `timeout_ms`, `max_retries` |

Supported `operation` values:

- `embedding`
- `reranking`
- `chat_completion`

Shared provider rules:

- provider names must not be empty
- `base_url` must not be empty and must start with `http://` or `https://`
- `model` must not be empty
- `timeout_ms` must be greater than zero
- if `api_key_env` is set, it must not be empty

Defaults:

- `timeout_ms = 30000`
- `max_retries = 2`

Example remote provider:

```toml
[providers.remote_embed]
kind = "openai_compatible"
operation = "embedding"
base_url = "https://api.example.com/v1"
model = "embedding-model"
api_key_env = "KBOLT_API_KEY"
timeout_ms = 30000
max_retries = 2
```

## Roles

`roles` binds product responsibilities to provider profiles.

Supported roles:

| Role | Required fields | Allowed provider operations | Defaults |
| --- | --- | --- | --- |
| `roles.embedder` | `provider` | `embedding` | `batch_size = 32` |
| `roles.reranker` | `provider` | `reranking`, `chat_completion` | none |
| `roles.expander` | `provider` | `chat_completion` | see below |

Role rules:

- every role provider must reference an existing provider profile
- `roles.embedder.batch_size` must be greater than zero
- `roles.expander.max_tokens` must be greater than zero

`roles.expander` supports these sampling keys:

- `max_tokens = 600`
- `seed = 0`
- `temperature = 0.7`
- `top_k = 20`
- `top_p = 0.8`
- `min_p = 0.0`
- `repeat_last_n = 64`
- `repeat_penalty = 1.0`
- `frequency_penalty = 0.0`
- `presence_penalty = 0.5`

Validation rules for expander sampling:

- `temperature` must be finite and greater than zero
- `top_k` must be greater than zero
- `top_p` must be finite and in `(0, 1]`
- `min_p` must be finite and in `[0, 1]`
- `repeat_last_n` must be greater than or equal to `-1`
- `repeat_penalty` must be finite and greater than zero
- `frequency_penalty` and `presence_penalty` must be finite

Example expander binding:

```toml
[providers.remote_chat]
kind = "openai_compatible"
operation = "chat_completion"
base_url = "https://api.example.com/v1"
model = "chat-model"
api_key_env = "KBOLT_API_KEY"

[roles.expander]
provider = "remote_chat"
max_tokens = 600
temperature = 0.7
top_k = 20
top_p = 0.8
min_p = 0.0
repeat_last_n = 64
repeat_penalty = 1.0
frequency_penalty = 0.0
presence_penalty = 0.5
```

## Reaping

`reaping.days` controls when stale documents become eligible for cleanup.

Default:

- `days = 7`

## Chunking

`chunking` has two parts:

- `chunking.defaults` for the general chunk policy
- `chunking.profiles` for named overrides

Each chunk policy supports:

- `target_tokens`
- `soft_max_tokens`
- `hard_max_tokens`
- `boundary_overlap_tokens`
- `neighbor_window`
- `contextual_prefix`

Default `chunking.defaults` values:

| Key | Default |
| --- | ---: |
| `target_tokens` | 800 |
| `soft_max_tokens` | 950 |
| `hard_max_tokens` | 1200 |
| `boundary_overlap_tokens` | 48 |
| `neighbor_window` | 1 |
| `contextual_prefix` | `true` |

Built-in `chunking.profiles.code` values:

| Key | Default |
| --- | ---: |
| `target_tokens` | 320 |
| `soft_max_tokens` | 420 |
| `hard_max_tokens` | 560 |
| `boundary_overlap_tokens` | 24 |
| `neighbor_window` | 1 |
| `contextual_prefix` | `true` |

Chunking validation rules:

- all token caps must be greater than zero
- `target_tokens <= soft_max_tokens <= hard_max_tokens`

Example:

```toml
[chunking.defaults]
target_tokens = 900
soft_max_tokens = 1050
hard_max_tokens = 1300
boundary_overlap_tokens = 64
neighbor_window = 1
contextual_prefix = true

[chunking.profiles.code]
target_tokens = 320
soft_max_tokens = 420
hard_max_tokens = 560
boundary_overlap_tokens = 24
neighbor_window = 1
contextual_prefix = true
```

## Ranking

`ranking` controls deep-search fanout, hybrid fusion, rerank budgets, and BM25 field boosts.

Top-level `ranking` keys:

| Key | Default |
| --- | ---: |
| `deep_variant_rrf_k` | 60 |
| `deep_variants_max` | 4 |
| `initial_candidate_limit_min` | 40 |
| `rerank_candidates_min` | 20 |
| `rerank_candidates_max` | 30 |

`ranking.hybrid_fusion.mode` supports:

- `dbsf` (default)
- `linear`
- `rrf`

`ranking.hybrid_fusion.linear` defaults:

| Key | Default |
| --- | ---: |
| `dense_weight` | 0.7 |
| `bm25_weight` | 0.3 |

`ranking.hybrid_fusion.dbsf` defaults:

| Key | Default |
| --- | ---: |
| `dense_weight` | 1.0 |
| `bm25_weight` | 0.4 |
| `stddevs` | 3.0 |

`ranking.hybrid_fusion.rrf` defaults:

| Key | Default |
| --- | ---: |
| `k` | 60 |

`ranking.bm25_boosts` defaults:

| Key | Default |
| --- | ---: |
| `title` | 2.0 |
| `heading` | 1.5 |
| `body` | 1.0 |
| `filepath` | 0.5 |

Ranking validation rules:

- all candidate limits must be greater than zero
- `rerank_candidates_max` must be greater than or equal to `rerank_candidates_min`
- fusion weights must be finite, non-negative, and sum to greater than zero
- `ranking.hybrid_fusion.dbsf.stddevs` must be finite and greater than zero
- `ranking.hybrid_fusion.rrf.k` must be greater than zero
- every BM25 boost must be finite and greater than zero

Example:

```toml
[ranking]
deep_variant_rrf_k = 60
deep_variants_max = 4
initial_candidate_limit_min = 40
rerank_candidates_min = 20
rerank_candidates_max = 30

[ranking.hybrid_fusion]
mode = "dbsf"

[ranking.hybrid_fusion.dbsf]
dense_weight = 1.0
bm25_weight = 0.4
stddevs = 3.0

[ranking.bm25_boosts]
title = 2.0
heading = 1.5
body = 1.0
filepath = 0.5
```

## Validation rules that matter first

If the file is invalid, `kbolt doctor` reports the config error directly. The most important
failures are:

- empty provider names
- empty or invalid provider URLs
- empty model names
- role bindings that reference undefined providers
- role bindings that point at incompatible provider operations
- invalid chunk token limits
- invalid ranking weights or candidate budgets

## Related pages

- [Local setup](../../concepts/local-setup.md)
- [Troubleshooting](../../operations/troubleshooting.md)
