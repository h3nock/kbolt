# ADR 0002: Role-Specific Inference Config

## Status
Accepted

## Context
Kbolt already had:
- `[models.*]` for artifact download/source identity.
- `[embeddings]` for dense embedding inference.

Deep search and reranking needed independent inference settings so those features can use
provider-backed adapters without coupling to embedding transport config.

## Decision
Add a dedicated inference config namespace with per-role settings:

```toml
[inference.reranker]
provider = "openai_compatible"
output_mode = "json_object"
model = "rerank-model"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
timeout_ms = 30000
max_retries = 2

[inference.expander]
provider = "openai_compatible"
output_mode = "json_object"
model = "expand-model"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
timeout_ms = 30000
max_retries = 2
```

Validation rules match embedding config expectations:
- non-empty `model`
- explicit `output_mode` (`json_object` or `text`)
- `base_url` must be `http://` or `https://`
- `timeout_ms > 0`
- `api_key_env` must be non-empty when set

## Consequences
- Reranker and expander can be configured independently from embeddings.
- Providers are pluggable per role.
- If a role is not configured, kbolt currently uses deterministic heuristic implementations.
- V1 provider scope for these role adapters is `openai_compatible`.
