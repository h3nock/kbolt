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

## Docs Workflow

- The public docs site source lives under `docs-site/`.
- CI builds the docs site with MkDocs Material.
- Before changing docs-site content or config, build it locally:

```bash
python3 -m venv .local/docs-venv
source .local/docs-venv/bin/activate
python -m pip install -r docs-site/requirements.txt
cd docs-site
mkdocs build --strict
```
