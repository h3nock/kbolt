# ADR 0003: Provider Profiles and Shared Local Inference

## Status
Accepted

## Amendment 2026-04-26
After this ADR was accepted, `kbolt setup local` grew a managed convenience path that downloads
default local models and starts local `llama-server` processes for configured roles.

That does not change the inference boundary selected here. Search and indexing code still bind
roles through provider profiles and talk to inference deployments over provider-backed clients.
The managed local lifecycle is a setup and operations convenience, not a return to in-process model
runtime ownership.

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

The design process also surfaced several negative constraints:
- this refactor is pre-release, so the architecture should not carry migration shims or dual paths
- local multi-user and multi-agent use needs one shared serving model, not per-process local runtimes
- remote provider flexibility matters, but local backend sprawl does not
- role adapters should be capability-oriented and provider-backed, not model-specific

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
Inference code is client-only with respect to provider profiles.

Kbolt resolves configured localhost deployments, health-checks them, and uses them through provider
clients. Product setup commands may create and manage default local `llama-server` processes, but
normal inference still crosses the provider boundary rather than constructing in-process local
runtimes.

## Rejected Alternatives

### Keep in-process local runtimes as the primary architecture
Rejected because it gives kbolt the wrong ownership boundary. It couples inference behavior to
artifact management, makes multi-agent/local concurrency harder to reason about, and keeps local
resource scheduling inside random client processes instead of inside a shared serving layer.

### Use provider profiles as connection-only records and keep model on the role
Rejected because the local deployment unit is a concrete server+model+operation combination.
Splitting that across provider and role makes role bindings harder to reason about and weakens
operation validation.

### Add a normalized `connections + deployments + roles` schema
Rejected for V1 because it solves hypothetical repetition before we have real repetition
pressure. Provider-profile-as-deployment gives the right boundary with much less machinery.

### Keep old and new inference schemas side by side during refactor
Rejected because this repo is still in active development with no migration requirement.
Supporting both would preserve the old mistakes, complicate engine construction, and hide which
schema is authoritative.

### Let kbolt continue to own local model pulls and local runtime setup
Rejected because it conflicts with the chosen local architecture. Once local inference is served
by `llama.cpp server` deployments, kbolt should keep the inference boundary at HTTP provider
profiles. Kbolt may still offer a convenience setup path that downloads default local models and
launches managed `llama-server` processes, but inference code should continue to bind roles
through provider profiles rather than rebuilding an in-process runtime.

### Force one remote vendor
Rejected because remote providers already own serving/scheduling and users will have different
preferences there. The architecture standardizes local backend family, not remote vendor.

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
- provider profiles and role bindings are the only authoritative inference schema
- readiness/status is now deployment-oriented rather than artifact-oriented
