# ADR 0003: Provider Profiles and Shared Local Inference

## Status
Accepted

## Supersedes
- [ADR 0002](./0002-role-inference-config.md)

## Context
Kbolt supports three distinct inference roles:
- embeddings
- reranking
- expansion

Those roles need different request contracts, but they do not need different configuration
shapes for transport, connection, or model ownership.

The previous architecture mixed two separate concerns:
- **provider/backend connection**: local runtime vs remote API, base URL, auth, timeouts, retries
- **role binding**: which backend a role uses and which model/options that role needs

That shape was acceptable while kbolt owned local runtimes in-process, but it is the wrong long
term boundary for the system we want:
- one standardized local backend
- many remote backends
- per-role provider choice
- correct behavior when multiple agents and humans issue requests on the same machine

The widely accepted solution for multi-client local inference is a shared localhost service that
owns models, queueing, batching, and concurrency. For kbolt's local path, the chosen backend is
`llama.cpp server`.

## Options Considered

### Option 1: Keep in-process local runtimes and continue extending role-specific config
- **Pros**:
  - smallest immediate change
  - preserves existing construction flow
- **Cons**:
  - local inference remains process-owned instead of machine-owned
  - concurrency and batching stay a client concern
  - config continues to reflect implementation details like local files and runtime knobs
  - duplicates transport config across roles

### Option 2: Standardize local inference on a shared `llama.cpp server`, but keep today's
role-shaped config
- **Pros**:
  - removes in-process local model ownership
  - local concurrency is delegated to the server
- **Cons**:
  - config remains split across implementation-specific sections
  - local and remote providers still use different conceptual models
  - transport and role binding stay coupled

### Option 3: Introduce provider profiles plus role bindings, with one standardized local backend
- **Pros**:
  - one clean abstraction for both local and remote providers
  - transport config is defined once per provider
  - roles stay independent and pluggable
  - local inference becomes service-backed instead of runtime-backed
  - simplest long-term architecture
- **Cons**:
  - larger refactor than a backend swap alone

## Decision
Adopt **provider profiles + role bindings** as the target inference architecture.

### Principles
- **One local backend**: `llama.cpp server`
- **Many remote backends**: remote providers remain pluggable per role
- **Per-role provider choice**: embedder, reranker, and expander bind independently

### Provider profiles
Provider profiles define connection/backend details only:
- provider kind
- base URL
- auth source
- timeout and retry policy
- provider-specific transport settings

Provider profiles include both:
- local server backends
- remote API backends

### Role bindings
Roles define only role-owned settings:
- which provider profile to use
- model name / alias for that role
- role-specific request options

Examples:
- embedder dimensions or input mode remain embedder concerns
- reranker output mode remains a reranker concern
- expander generation settings remain expander concerns

### Local inference
Local inference is standardized on a shared `llama.cpp server` process running on localhost.

Kbolt becomes a client of that service. It no longer owns:
- local GGUF runtime construction
- llama contexts
- local flash-attention tuning
- local model file resolution for normal inference
- local concurrency scheduling

Those become deployment/runtime concerns of the local inference server.

### Remote inference
Remote inference remains provider-pluggable per role.

Kbolt supports a small set of clean remote adapters rather than forcing one remote vendor. The
stable contract is the role interface, not the remote provider brand.

## Target Configuration Shape

```toml
default_space = "work"

[providers.local]
kind = "llama_cpp_server"
base_url = "http://127.0.0.1:8080"
timeout_ms = 30000
max_retries = 2

[providers.openai]
kind = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
timeout_ms = 30000
max_retries = 2

[roles.embedder]
provider = "local"
model = "embeddinggemma"

[roles.reranker]
provider = "local"
model = "qwen3-reranker"

[roles.expander]
provider = "openai"
model = "gpt-5-mini"
max_tokens = 600
temperature = 0.7
top_p = 0.8
```

## Consequences
- Local inference is opinionated and standardized.
- Remote inference remains flexible.
- Engine keeps depending only on role contracts (`Embedder`, `Reranker`, `Expander`).
- Transport configuration is no longer duplicated across roles.
- Local server lifecycle becomes part of local setup.
- Current in-process local runtime modules become transitional implementation detail and should not
  remain the architectural center of gravity.

## Impact Radius

### Configuration
- replace implementation-shaped local config with:
  - `providers.*`
  - `roles.*`
- remove local runtime knobs from normal kbolt inference config
- keep role-specific request options where they actually affect behavior

### Engine composition
- `Engine::new()` should build role adapters from provider profiles + role bindings
- engine should not know whether a role is local or remote beyond provider selection

### Inference modules
- add a `llama.cpp server` client as the standard local provider
- keep remote provider adapters behind role contracts
- stop centering inference architecture around in-process local runtime modules

### Model management
- local server-owned models are no longer normal kbolt-managed inference artifacts
- local model download/file discovery should leave the hot inference path
- model administration commands must be reconsidered in light of server-owned local models

### Testing
- add config validation coverage for provider profiles and role bindings
- add capability tests for provider/role compatibility
- move local inference integration tests to HTTP-level server contract tests

### Documentation
- update the main spec and setup docs to reflect:
  - one local backend
  - many remote backends
  - per-role provider choice

## Non-Goals
- This ADR does not define the migration path from the current implementation.
- This ADR does not require remote traffic to pass through a local proxy.
- This ADR does not require every remote provider to support every role.
