# ADR 0002: Role-Specific Inference Config

## Status
Accepted

## Context
Kbolt already had:
- `[models.*]` for artifact download/source identity.
- `[embeddings]` for dense embedding inference.

Deep search and reranking needed independent inference settings so those features can use
provider-backed adapters without coupling to embedding transport config.

The expander role also needs its own provider contract because query-variant generation has
different runtime knobs than reranking: local expander quality depends on sampler settings,
non-thinking template controls, and grammar-constrained output, while reranking does not.

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
flash_attention = "auto" # auto | enabled | disabled
# omit n_gpu_layers to auto-detect acceleration

[inference.expander]
provider = "local_llama"
# optional when exactly one .gguf exists in models/expander
# model_file = "Qwen3-1.7B-Q8_0.gguf"
max_tokens = 600
n_ctx = 2048
flash_attention = "auto" # auto | enabled | disabled
enable_thinking = false
reasoning_format = "none"
temperature = 0.7
top_k = 20
top_p = 0.8
min_p = 0.0
repeat_last_n = 64
repeat_penalty = 1.0
frequency_penalty = 0.0
presence_penalty = 0.5
# omit n_gpu_layers to auto-detect acceleration
```

Validation rules match embedding config expectations:
- `openai_compatible`: non-empty `model`, `base_url` must be `http://` or `https://`, `timeout_ms > 0`, and `api_key_env` non-empty when set
- `local_llama`: `max_tokens > 0`, `n_ctx > 0`, optional `model_file` must be non-empty when set, optional `reasoning_format` must be non-empty when set, optional `chat_template_kwargs` must be a JSON object, and sampler fields must stay within valid ranges
- `local_gguf` and `local_llama` share `flash_attention = "auto" | "enabled" | "disabled"`; default is `auto`

## Consequences
- Reranker and expander can be configured independently from embeddings.
- Providers are pluggable per role, and expander generation settings are explicit instead of hidden in model-specific adapters.
- Unset roles stay unset: kbolt does not synthesize heuristic replacements.
- Deep search therefore requires an expander configuration, while reranking is skipped and reported when no reranker is configured.
- V1 provider scope for these role adapters is `openai_compatible` and `local_llama`.
- The expander contract is plain query variants (`query -> Vec<String>`). Prompting, JSON parsing, duplicate/original filtering, and generation controls live behind the provider-specific expander implementation rather than a public `adapter` switch.
