# ADR 0002: Role-Specific Inference Config

## Status
Accepted

## Context
Kbolt already had:
- `[models.*]` for artifact download/source identity.
- `[embeddings]` for dense embedding inference.

Deep search and reranking needed independent inference settings so those features can use
provider-backed adapters without coupling to embedding transport config.

The expander role also needs an explicit model-contract seam because the local qmd family does
not share the same prompt/output contract as JSON-backed expanders.

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
adapter = "json_variants"
output_mode = "json_object"
model = "expand-model"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
timeout_ms = 30000
max_retries = 2

[inference.reranker]
provider = "local_llama"
# optional when exactly one .gguf exists in models/reranker
# model_file = "qwen-reranker.gguf"
max_tokens = 256
n_ctx = 2048
# omit n_gpu_layers to auto-detect acceleration

[inference.expander]
provider = "local_llama"
adapter = "qmd"
# optional when exactly one .gguf exists in models/expander
# model_file = "qmd-query-expansion.gguf"
max_tokens = 600
n_ctx = 2048
# omit n_gpu_layers to auto-detect acceleration
```

Validation rules match embedding config expectations:
- `openai_compatible`: non-empty `model`, explicit `output_mode` (`json_object` or `text`), `base_url` must be `http://` or `https://`, `timeout_ms > 0`, and `api_key_env` non-empty when set
- `local_llama`: `max_tokens > 0`, `n_ctx > 0`, optional `model_file` must be non-empty when set
- `inference.expander.adapter`: V1 adapters are `json_variants` and `qmd`; `adapter = "qmd"` requires `provider = "local_llama"`

## Consequences
- Reranker and expander can be configured independently from embeddings.
- Providers are pluggable per role, and expander model contracts are explicit instead of inferred from provider or model id.
- Unset roles stay unset: kbolt does not synthesize heuristic replacements.
- Deep search therefore requires an expander configuration, while reranking is skipped and reported when no reranker is configured.
- V1 provider scope for these role adapters is `openai_compatible` and `local_llama`.
- V1 expander adapters are `json_variants` (generic JSON rewrites) and `qmd` (typed `lex` / `vec` / `hyde` local expansion).
