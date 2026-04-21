# Install

Choose one install path, then verify the binary is on your `PATH`.

=== "Homebrew"

    Homebrew is the shortest path on macOS and on Linux x86_64 systems that already use Homebrew:

    ```bash
    brew install h3nock/kbolt/kbolt
    ```

    The Homebrew formula installs `llama.cpp` as a required dependency, so this path also provides `llama-server` for `kbolt setup local`.

=== "Cargo"

    Use Cargo when you want the CLI directly from crates.io or when you are on Windows:

    ```bash
    cargo install kbolt
    ```

    If `cargo install` succeeds but `kbolt` is still not found, make sure `~/.cargo/bin` is on your `PATH`.

    Cargo installs only `kbolt`. Install `llama-server` separately if you plan to use `kbolt setup local`.

=== "GitHub Releases"

    Prebuilt release archives are published on [GitHub Releases](https://github.com/h3nock/kbolt/releases).

    Current release archives are built for:

    - Linux x86_64
    - macOS x86_64
    - macOS aarch64

    If you need Windows today, use `cargo install`.

    Release archives install only `kbolt`. Install `llama-server` separately if you plan to use `kbolt setup local`.

## Install `llama-server` when needed

`llama-server` is required only if you plan to use `kbolt setup local`.

Homebrew installs it through the `llama.cpp` dependency. Cargo and GitHub Release installs do not.

For non-Homebrew installs, follow the official [llama.cpp install guide](https://github.com/ggml-org/llama.cpp/wiki), then verify the binary is available:

```bash
llama-server --help
```

If you plan to use only remote OpenAI-compatible providers, `llama-server` is not required.

## Verify the install

Run:

```bash
kbolt --version
```

Then check the command surface:

```bash
kbolt --help
```

## Next steps

- For the default local setup path, continue to [Quickstart](quickstart.md).
- If you need installation recovery steps, see [Troubleshooting](operations/troubleshooting.md).
