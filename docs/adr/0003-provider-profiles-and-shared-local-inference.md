# ADR 0003: Provider Profiles and Shared Local Inference

## Status
Accepted

## Supersedes
- [ADR 0002](./0002-role-inference-config.md)

## Context
Kbolt has three inference roles:
- embeddings
- reranking
- expansion

The architecture should satisfy two constraints at the same time:
- standardize local inference so multiple agents and humans can share the same local backend cleanly
- keep remote providers flexible and pluggable per role

The previous config model mixed deployment concerns with role concerns:
- local runtime files and runtime knobs lived directly in role config
- transport settings were repeated across roles
- local model ownership stayed inside kbolt instead of inside a shared local service

That shape does not match the target operating model.

## Decision
Use **provider profiles plus role bindings** as the target inference architecture.

### Principles
- one local backend family: `llama.cpp server`
- multiple local provider profiles of that same kind, one per deployed model
- many remote backends
- per-role provider choice

### Provider profiles
Provider profiles represent concrete inference deployments.

They own:
- provider kind
- operation contract
- base URL
- model identity
- auth source when needed
- timeout and retry policy

V1 provider kinds:
- `llama_cpp_server`
- `openai_compatible`

### Role bindings
Role bindings select a provider profile for a role and attach only role-owned behavior knobs.

Examples:
- embedder role: provider + client-side batch sizing
- reranker role: provider
- expander role: provider + generation settings

### Local lifecycle
The first implementation is **connect-only** for local server profiles.

Kbolt will:
- resolve the configured localhost deployment
- health-check it
- use it as a client

Kbolt will not, in this first architecture step:
- auto-start local servers
- supervise them
- own local GGUF runtime construction for normal inference

## Target Configuration Shape

```toml
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

[providers.openai_expand]
kind = "openai_compatible"
operation = "chat_completion"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5-mini"
timeout_ms = 30000
max_retries = 2

[roles.embedder]
provider = "local_embed"
batch_size = 32

[roles.reranker]
provider = "local_rerank"

[roles.expander]
provider = "openai_expand"
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

## Validation Rules
- every role provider reference must resolve to an existing provider profile
- `embedder` may bind only to `embedding`
- `expander` may bind only to `chat_completion`
- `reranker` may bind to `reranking` or `chat_completion`
- provider profiles must define non-empty `base_url` and `model`
- provider URLs must start with `http://` or `https://`
- `api_key_env` must be non-empty when set
- role-specific knobs keep their own validation rules

## Consequences
- local inference becomes service-backed rather than runtime-backed
- transport configuration stops being duplicated across roles
- local deployment ownership is separated from role behavior
- current in-process local runtime modules are no longer the target architecture center
- remote vendor support remains extensible without changing the core schema
