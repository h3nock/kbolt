# Contributing

## One-Time Setup

Run the local hook installer once per clone:

```bash
./scripts/install-hooks.sh
```

This configures git to use the committed hook in `.githooks/`.

## Local Workflow

1. Edit code.
2. Run `cargo fmt --all`.
3. Run `cargo test`.
4. Review `git diff`.
5. Commit one logical slice at a time.

## Formatting Policy

- The repo has a baseline `cargo fmt` commit. Formatting drift should be small after normal edits.
- CI enforces `cargo fmt --all --check` and `cargo test`.
- The pre-commit hook runs the same formatting check locally when staged Rust files exist.
- Keep formatting-only changes separate from logic changes when possible.
