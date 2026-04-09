# Releasing kbolt

This repository treats `kbolt` as an application first:

- GitHub Releases are the canonical release surface.
- crates.io and the Homebrew tap should stay in sync with the tagged release version.

## Versioning and tags

- The workspace uses one shared version.
- Release tags must be `vX.Y.Z`.
- The CLI crate version and the tag must match exactly.

## Release artifacts

The release workflow builds and uploads these archives:

- `kbolt-vX.Y.Z-linux-x86_64.tar.gz`
- `kbolt-vX.Y.Z-macos-x86_64.tar.gz`
- `kbolt-vX.Y.Z-macos-aarch64.tar.gz`

Each archive contains:

- `kbolt`
- `README.md`
- `LICENSE`

The release workflow also uploads a `SHA256SUMS` file covering all archives.

## Workflows

### CI

`.github/workflows/ci.yml` runs on pushes to `main` and pull requests.

It verifies:

- formatting
- tests on Linux and macOS
- release binary builds on Linux and macOS
- basic CLI smoke for the release binary

### GitHub Releases

`.github/workflows/release.yml` runs on `v*` tags.

It:

- builds release binaries for supported targets
- packages tarballs
- generates `SHA256SUMS`
- publishes a GitHub Release with generated notes

## Additional release surfaces

### crates.io

Publish in dependency order:

1. `kbolt-types`
2. `kbolt-core`
3. `kbolt-mcp`
4. `kbolt`

Use `cargo publish --workspace --locked --dry-run` before the real publish sequence.

### Homebrew tap

The custom tap lives at `h3nock/homebrew-kbolt`.

After the GitHub Release is live:

1. update `Formula/kbolt.rb` to the new versioned archive URLs
2. update the corresponding SHA256 values
3. commit and push the tap repo

## Recommended release order

1. Make sure CI is green on `main`.
2. Bump the workspace version if needed.
3. Create and push a `vX.Y.Z` tag.
4. Wait for `release.yml` to finish and verify the published GitHub Release.
5. Publish crates.io packages for the same version.
6. Update the Homebrew tap formula to the same version.
