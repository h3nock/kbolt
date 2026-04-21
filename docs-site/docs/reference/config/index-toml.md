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

## When you need to edit it

Most users do not need to edit `index.toml` before the first successful search.

Edit it when you want to:

- change the default space
- bind remote provider profiles
- tune chunking or ranking settings

## Top-level keys

`index.toml` supports these top-level keys:

- `default_space`
- `providers`
- `roles`
- `reaping`
- `chunking`
- `ranking`

## Minimal shape

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

## Validation rules that matter first

- provider names must not be empty
- provider URLs must start with `http://` or `https://`
- provider models must not be empty
- role bindings must reference existing provider profiles
- each role must bind to a compatible operation

If the file is invalid, `kbolt doctor` will report the config error directly.

## Related pages

- [Local setup](../../concepts/local-setup.md)
- [Troubleshooting](../../operations/troubleshooting.md)
