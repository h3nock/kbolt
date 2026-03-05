# Kbolt — User Flows

Every flow is one actor performing one action to achieve one goal.

## Actors

- **CLI user** — human at the terminal
- **MCP client** — LLM/agent connected via stdio
- **System** — cron/launchd (automated)

---

## Setup

### 1. Install kbolt

**Actor**: CLI user
**Goal**: Get kbolt binary available on PATH

```
curl -fsSL https://... | sh
```

Or via Homebrew:

```
brew install kbolt
```

Or for Rust developers:

```
cargo install kbolt
```

After install, `kbolt` command is available. No configuration needed yet — the system creates `~/.config/kbolt/` and `~/.cache/kbolt/` on first use.

> **TODO (implementation)**: The install script / Homebrew post-install message should print a getting-started reminder after confirming kbolt is installed, e.g. "Run `kbolt models pull` to download ML models (~2.5GB) for semantic search and reranking. Keyword search works without models." This is a one-time message at install time, not on every command.

---

### 2. Download models

**Actor**: CLI user
**Goal**: Pre-download ML models before first use

```
kbolt models pull
```

Downloads embedding model (~600MB), reranker (~700MB), and query expander (~1.2GB) via the configured model provider (default: HuggingFace Hub) to `~/.cache/kbolt/models/{role}/` (`embedder`, `reranker`, `expander`). The command reports per-model progress events and a final download summary.

Embedding inference provider scope (V1):
- supported: `openai_compatible`, `voyage`, `local_onnx`
- deferred to V2: native Google embeddings API provider (add only if we need native-only controls/parity)

Text inference provider scope for reranker/expander (V1):
- supported: `openai_compatible`, `local_llama`

If models are missing when a command needs them (search, update with embedding), prompt/fallback behavior is CLI-only:
- interactive CLI (TTY): kbolt prompts "Models not downloaded. Download now and continue? (Y/n)".
- non-interactive CLI: no prompt; command fails with actionable guidance (`kbolt models pull`; for update also `--no-embed` when appropriate).
- MCP/agent usage: no prompt; return deterministic error behavior.

When the user declines the interactive prompt in auto search mode, kbolt falls back to keyword-only search. For explicitly requested semantic/deep modes, kbolt fails instead of silently changing mode.

---

### 3. Check model status

**Actor**: CLI user
**Goal**: See which models are downloaded and ready

```
kbolt models list
```

Shows each model: name, role (embed/rerank/expand), status (downloaded/missing), file size, path on disk.

---

## Spaces

### 4. Create a space

**Actor**: CLI user
**Goal**: Create a grouping for related collections

```
kbolt space add {name}
kbolt space add {name} --description {description}
kbolt space add {name} {dir1} {dir2} ...
kbolt space add {name} --description {description} {dir1} {dir2} ...
```

**Parameters**:
- **name** (required) — space name (e.g. `work`, `personal`, `notes`)
- **--description** (optional) — what this space contains
- **dirs** (optional) — one or more directories to add as collections immediately

When directories are provided, each becomes a collection named after its directory name. `--description` refers to the space. To set collection descriptions, use `kbolt collection describe` after creation.

A `default` space always exists (created on first use). Collections added without `--space` go into the `default` space.

---

### 5. Remove a space

**Actor**: CLI user
**Goal**: Remove a space and all its collections and indexed data

```
kbolt space remove {name}
```

**Parameters**:
- **name** (required) — space to remove

CASCADE deletes all collections, documents, chunks, embeddings. Removes the space's Tantivy index and USearch file from `~/.cache/kbolt/spaces/{name}/`. Does not touch actual files on disk. Cannot remove the `default` space.

---

### 6. Rename a space

**Actor**: CLI user
**Goal**: Change a space's name

```
kbolt space rename {old_name} {new_name}
```

**Parameters**:
- **old_name** (required)
- **new_name** (required)

Updates the name in SQLite and renames the per-space index directory. No re-indexing needed.

---

### 7. List spaces

**Actor**: CLI user
**Goal**: See all spaces

```
kbolt space list
```

Shows each space with: name, description, collection count, total document count, last updated. Marks the active space (from `--space` / env / default).

---

### 8. View space details

**Actor**: CLI user
**Goal**: See detailed information about one space

```
kbolt space info {name}
```

**Parameters**:
- **name** (required)

Shows: name, description, collections (with doc counts each), total documents, total chunks, embedded chunk count, disk usage for this space's indexes.

---

### 9. Set default space

**Actor**: CLI user
**Goal**: Set which space is used when no `--space` flag or `KBOLT_SPACE` env var is provided

```
kbolt space default {name}
kbolt space default
```

**Parameters**:
- **name** (optional) — if omitted, shows the current default

Persists the default in `~/.config/kbolt/index.toml`. This is the third level in the space resolution precedence: `--space` flag > `KBOLT_SPACE` env var > configured default > unique lookup > error.

---

### 10. Show current space

**Actor**: CLI user
**Goal**: See which space is currently active and why

```
kbolt space current
```

Shows the active space and how it was resolved (flag, env var, configured default, or none). Useful for debugging unexpected behavior.

---

### 11. Update space description

**Actor**: CLI user
**Goal**: Change a space's description

```
kbolt space describe {name} {text}
```

**Parameters**:
- **name** (required)
- **text** (required) — new description

---

## Collections

### 12. Add a collection

**Actor**: CLI user
**Goal**: Register a directory and index it within a space

```
kbolt collection add {path}
kbolt collection add {path} --name {name} --description {description}
kbolt collection add {path} --space {space}
kbolt collection add {path} --extensions {ext,ext,...}
kbolt collection add {path} --no-index
```

**Parameters**:
- **path** (required) — absolute path to directory
- **--name** (optional) — human-readable name, defaults to directory name
- **--description** (optional) — what this collection contains, shown in MCP instructions and CLI output
- **--space** (optional) — target space. If omitted, uses active space (from env/default). If the named space doesn't exist, creates it implicitly.
- **--extensions** (optional) — comma-separated list of file extensions to index (e.g. `rs,py,md`). If omitted, indexes all files with a supported extractor.
- **--no-index** (optional) — register the collection without triggering indexing

Collection names are unique within their space (not globally). Registers the collection in SQLite and immediately triggers `kbolt update --collection {name}` to index it.

Use `--no-index` when adding multiple collections before a single bulk `kbolt update`, or when you want to configure ignore patterns before the first index.

Ignore patterns are per-collection, stored internally at `~/.config/kbolt/ignores/{space}/{collection}.ignore`. Managed via `kbolt ignore` commands. Follows `.gitignore` syntax.

---

### 13. Remove a collection

**Actor**: CLI user
**Goal**: Unregister a collection and delete all its indexed data

```
kbolt collection remove {name}
```

**Parameters**:
- **name** (required) — collection to remove (resolved via space precedence)

CASCADE deletes all documents, chunks, embeddings for this collection. Removes entries from the space's Tantivy index and USearch file. Removes the collection's ignore file if one exists. Does not touch the actual files on disk.

---

### 14. Rename a collection

**Actor**: CLI user
**Goal**: Change a collection's display name

```
kbolt collection rename {old_name} {new_name}
```

**Parameters**:
- **old_name** (required)
- **new_name** (required)

Updates the name in SQLite. Renames the ignore file if one exists. No re-indexing needed — the name is just a label. Must be unique within the collection's space.

---

### 15. View collection details

**Actor**: CLI user
**Goal**: See detailed information about one collection

```
kbolt collection info {name}
```

**Parameters**:
- **name** (required) — resolved via space precedence

Shows: name, space, path, description, extensions filter, document count, chunk count, embedded chunk count, created/updated timestamps, whether ignore patterns are configured.

---

### 16. List all collections

**Actor**: CLI user
**Goal**: See all registered collections

```
kbolt collection list
kbolt collection list --space {name}
```

Without `--space`: shows collections across all spaces, grouped by space. With `--space`: shows collections in that space only. Each collection shows: name, path, document count, last updated. Summary row with totals.

---

### 17. Update collection description

**Actor**: CLI user
**Goal**: Change a collection's description

```
kbolt collection describe {name} {text}
```

**Parameters**:
- **name** (required) — resolved via space precedence
- **text** (required) — new description

---

## Ignore Patterns

### 18. Show ignore patterns

**Actor**: CLI user
**Goal**: See the current ignore patterns for a collection

```
kbolt ignore show {collection}
```

**Parameters**:
- **collection** (required) — resolved via space precedence

Prints the contents of the collection's ignore file. If no ignore patterns are configured, says so.

---

### 19. Edit ignore patterns

**Actor**: CLI user
**Goal**: Edit a collection's ignore patterns in a text editor

```
kbolt ignore edit {collection}
```

**Parameters**:
- **collection** (required) — resolved via space precedence

Opens `~/.config/kbolt/ignores/{space}/{collection}.ignore` in the user's preferred editor (`$VISUAL` > `$EDITOR` > `vi`). `$VISUAL` / `$EDITOR` may include command-line flags (for example: `code --wait`, `nvim -f`). Creates the file if it doesn't exist.

---

### 20. Add ignore pattern

**Actor**: CLI user
**Goal**: Add a single pattern to a collection's ignore file

```
kbolt ignore add {collection} {pattern}
```

**Parameters**:
- **collection** (required) — resolved via space precedence
- **pattern** (required) — `.gitignore` syntax pattern (e.g. `dist/`, `*.min.js`)

Appends the pattern to the collection's ignore file. Creates the file if it doesn't exist.

---

### 21. Remove ignore pattern

**Actor**: CLI user
**Goal**: Remove a pattern from a collection's ignore file

```
kbolt ignore remove {collection} {pattern}
```

**Parameters**:
- **collection** (required) — resolved via space precedence
- **pattern** (required) — exact pattern to remove

Removes the matching line from the ignore file. If the file becomes empty, deletes it.

---

### 22. List collections with ignore patterns

**Actor**: CLI user
**Goal**: See which collections have ignore patterns configured

```
kbolt ignore list
kbolt ignore list --space {name}
```

Shows collections that have an ignore file, grouped by space. Each entry shows: space, collection, number of patterns.

---

## Indexing

### 23. Index everything

**Actor**: CLI user or System (automated)
**Goal**: Scan all collections across all spaces, extract, chunk, index, and embed

```
kbolt update
kbolt update --space {name}
```

Without `--space`: updates all collections in all spaces. With `--space`: scopes to that space only.

For each collection: scans the directory and applies filters (hardcoded ignores → ignore patterns → extractor registry → extensions). For each file that passes filters, checks the file's modification time (mtime) against the stored value — if unchanged, skips the file entirely (no read needed). If mtime changed, reads the file and computes a SHA-256 hash — if the hash matches the stored hash, updates the stored mtime and skips re-indexing. Only when the hash actually changed does the full pipeline run: extract content, chunk, write to the space's FTS index, embed vectors into the space's USearch file. Chunking is structure-first and budget-constrained (`target`/`soft_max`/`hard_max`); forced splits can use small boundary overlap, while retrieval-time neighboring chunks provide additional context during search. Files that disappeared from disk are deactivated (soft-deleted). Files deactivated longer than the reaping period (default: 7 days) are hard-deleted.

This two-level change detection (mtime first, hash second) makes "nothing changed" scans extremely fast — ~10ms for 10K files. The SHA-256 hash is the source of truth; mtime is a fast pre-filter to avoid unnecessary file reads.

Returns a summary report: files scanned, skipped (unchanged), added, updated, deactivated, reactivated, reaped, chunks created, chunks embedded, duration.

---

### 24. Index specific collections

**Actor**: CLI user
**Goal**: Re-index one or more collections without updating all

```
kbolt update --collection {name}
kbolt update --collection {name},{name},...
kbolt update --space {space} --collection {name}
kbolt update --space {space} --collection {name},{name},...
```

**Parameters**:
- **--collection** (required for this flow) — comma-separated collection names (resolved via space precedence)
- **--space** (optional) — resolve collection names within this space

Same pipeline as flow 23, but scoped to the named collections. Unchanged collections are skipped entirely. Useful when a few collections have heavy changes and a full `kbolt update` would be slow.

---

### 25. Index without embedding

**Actor**: CLI user
**Goal**: Update FTS index only, skip dense vectors

```
kbolt update --no-embed
kbolt update --space {name} --no-embed
kbolt update --space {name} --collection {name} --no-embed
```

**Parameters**:
- **--no-embed** — skip embedding step
- **--space** (optional) — scope to a specific space
- **--collection** (optional) — scope to specific collections

Useful when models aren't downloaded yet, or for a quick index pass where keyword search is sufficient. Embedding can be done later by running `kbolt update` normally (it will embed any unembedded chunks). All flags compose independently — `--no-embed` can be combined with `--space`, `--collection`, `--dry-run`, and `--verbose`.

---

### 26. Dry run

**Actor**: CLI user
**Goal**: Preview what an update would do without changing anything

```
kbolt update --dry-run
kbolt update --collection {name} --dry-run
kbolt update --space {name} --dry-run
kbolt update --space {name} --collection {name} --dry-run
```

**Parameters**:
- **--dry-run** — simulate the update, report what would happen
- **--space** (optional) — scope to a specific space
- **--collection** (optional) — scope to specific collections

Shows: how many files would be added, updated, deactivated, reaped. No writes to SQLite, Tantivy, or USearch. Useful before first index of a large collection to understand scope.

---

### 27. Verbose update

**Actor**: CLI user
**Goal**: See per-file decisions during indexing (for troubleshooting)

```
kbolt update --verbose
kbolt update --space {name} --verbose
kbolt update --space {name} --collection {name} --verbose
kbolt update --collection {name} --dry-run --verbose
```

**Parameters**:
- **--verbose** — log per-file decisions
- **--space** (optional) — scope to a specific space
- **--collection** (optional) — scope to specific collections

Prints why each file was skipped (hash unchanged, filtered by ignore patterns, unsupported extension, extraction failed) or processed (new, changed). Useful when a file isn't appearing in search results and the user wants to know why. `--verbose` combines with `--dry-run` to preview per-file decisions without making changes.

---

## Search

### 28. Search

**Actor**: CLI user
**Goal**: Find relevant content in the index

```
kbolt search {query}
kbolt search {query} --space {name}
kbolt search {query} --collection {name}
kbolt search {query} --collection {name},{name},...
kbolt search {query} --space {name} --collection {name}
kbolt search {query} --space {name} --collection {name},{name},...
kbolt search {query} --limit {n}
kbolt search {query} --deep
kbolt search {query} --space {name} --deep
kbolt search {query} --space {name} --collection {name} --deep
kbolt search {query} --keyword
kbolt search {query} --semantic
kbolt search {query} --rerank
kbolt search {query} --no-rerank
```

**Parameters**:
- **query** (required) — the search string
- **--space** (optional) — scope to a specific space. If omitted, searches all spaces.
- **--collection** (optional) — one or more collections (comma-separated, resolved via space precedence)
- **--limit** (optional) — max results, default 10
- **--deep** (optional) — query expansion + all signals + reranking
- **--keyword** (optional) — BM25 only (diagnostic)
- **--semantic** (optional) — dense vector only (diagnostic)
- **--rerank** (optional) — enable cross-encoder reranking in auto mode
- **--no-rerank** (optional) — disable cross-encoder reranking (diagnostic)

Default behavior (no mode flag): auto mode — the system analyzes the query and routes to hybrid BM25+dense retrieval, with reranking off by default for speed. Use `--rerank` to opt in.

**Cross-space search**: When searching without `--space`, each space's Tantivy and USearch indexes are queried independently. Candidate lists from all spaces are concatenated, then fused and reranked together. The cross-encoder reranker normalizes scores across spaces.

Returns ranked results. Each result shows: rank, docid, space, collection, path, title, heading breadcrumb, snippet, score. The output footer includes a staleness hint: "Index last updated: {time ago}" based on the most recent update timestamp across the searched collections. This helps users understand why a recently saved file might not appear in results.

Search result assembly may expand each hit with neighboring chunks from the same document (`±N` by chunk sequence, default `±1`) to provide context without storing large overlap at indexing time.

`--keyword`, `--semantic`, and `--no-rerank` are diagnostic flags for troubleshooting retrieval quality — they let the user isolate which retrieval signal is working or failing.

---

## Document Access

### 29. Get document by docid

**Actor**: CLI user
**Goal**: Read a document found via search, using the short hash

```
kbolt get {docid}
kbolt get {docid} --offset {n} --limit {n}
```

**Parameters**:
- **docid** (required) — short hash like `#a1b2c3`
- **--offset** (optional) — start at line N
- **--limit** (optional) — max lines to return

Resolves the docid to a document in SQLite, reads the file from disk at the collection's path, returns the content. Always reads the **live file**, not a cached copy. If the file's current hash doesn't match the indexed hash, the output includes a `stale` indicator — the content has changed since indexing, and search snippets may not match. If the file has been deleted from disk, returns an error: "File deleted since indexing. Run `kbolt update` to refresh." Primarily useful as a quick reference after seeing a docid in search results.

---

### 30. Get document by path

**Actor**: CLI user
**Goal**: Read a document using its collection-relative path

```
kbolt get {collection/path}
kbolt get {collection/path} --offset {n} --limit {n}
kbolt get {collection/path} --space {name}
```

**Parameters**:
- **collection/path** (required) — collection-relative path like `{collection}/{relative/path}`
- **--offset** (optional) — start at line N
- **--limit** (optional) — max lines to return
- **--space** (optional) — resolve the collection name within this space

Resolves the path to a collection + relative path, reads the file from disk. Always reads the **live file**. If the file's current hash doesn't match the indexed hash, includes a `stale` indicator. If the file has been deleted, returns an error with guidance to run `kbolt update`. The main convenience over `cat` is that the user doesn't need to know the absolute path — kbolt resolves `{collection}/...` to the collection's root path via the collection registry. Use `--space` when the same collection name exists in multiple spaces.

---

### 31. Get multiple documents

**Actor**: CLI user or MCP client
**Goal**: Read several documents at once

```
kbolt multi-get {path},{path},...
kbolt multi-get {docid},{docid},...
kbolt multi-get {path},{docid},{path},...
kbolt multi-get {path},{path} --max-bytes {n} --max-files {n}
kbolt multi-get {path},{path} --space {name}
```

**Parameters**:
- **locators** (required) — comma-separated paths (`collection/path`) and/or docids (`#a1b2c3`), resolved in order given
- **--max-files** (optional) — max number of files returned, default 20
- **--max-bytes** (optional) — max total bytes returned, default 50KB
- **--space** (optional) — resolve collection references within this space

Locators are resolved in the order given. Budget (`--max-files`, `--max-bytes`) is consumed in that order, so the caller controls priority — put the most important files first. Files are returned whole, never truncated mid-file. When budget is exhausted, remaining files are reported as omitted with their sizes, so the caller can fetch them individually via `kbolt get` if needed.

Primarily exists for MCP clients (LLMs can't access the file system). CLI users can use it for scripting. Each returned document includes a `stale` indicator if its content has changed since indexing. Documents whose files have been deleted are omitted from results with a warning.

---

## Browsing

### 32. List files in a collection

**Actor**: CLI user
**Goal**: See what files are indexed in a collection

```
kbolt ls {collection}
kbolt ls {collection} {prefix}
kbolt ls {collection} --all
kbolt ls {collection} --space {name}
kbolt ls {collection} {prefix} --space {name}
```

**Parameters**:
- **collection** (required) — resolved via space precedence
- **prefix** (optional) — subdirectory filter
- **--all** (optional) — include deactivated (soft-deleted) files
- **--space** (optional) — resolve collection name within this space

Shows: path, title, docid, active status (if `--all`). Default shows only active files.

---

## Admin

### 33. Check system status

**Actor**: CLI user
**Goal**: Understand overall index health

```
kbolt status
kbolt status --space {name}
```

Without `--space`: shows all spaces with their collections and stats. With `--space`: scopes to that space.

Shows: list of spaces with collection counts, doc/chunk counts per collection, total documents, total chunks, total embedded chunks, disk usage breakdown (SQLite, per-space Tantivy/USearch, models).

---

### 34. Set up scheduled indexing

**Actor**: CLI user
**Goal**: Automate periodic re-indexing

```
kbolt schedule --every {interval}
kbolt schedule --every {interval} --no-embed
kbolt schedule --at {time}
kbolt schedule --every {interval} --space {name}
kbolt schedule --every {interval} --space {name} --collection {name}
kbolt schedule --every {interval} --space {name} --collection {name},{name},...
```

**Parameters**:
- **--every** — interval (e.g. `6h`, `30m`, `1d`)
- **--at** — daily at specific time (e.g. `03:00`)
- **--no-embed** (optional) — FTS-only re-index for speed
- **--space** (optional) — scope scheduled updates to a specific space
- **--collection** (optional) — scope scheduled updates to specific collections

Creates a launchd plist (macOS) or cron job / systemd timer (Linux) that runs `kbolt update` on the specified schedule. Without `--space`/`--collection`, schedules a full `kbolt update` across all spaces. With scoping flags, the scheduled command runs `kbolt update --space {name}` or `kbolt update --space {name} --collection {name}` accordingly. Multiple schedules can coexist (e.g. a fast `--no-embed` every 30m for one space, and a full update daily for everything).

---

### 35. Check schedule status

**Actor**: CLI user
**Goal**: See current scheduled indexing setup

```
kbolt schedule --status
```

Shows all active schedules. Each entry shows: scope (all / space / collection), interval/time, last run, next run, whether embedding is included.

---

### 36. Remove schedule

**Actor**: CLI user
**Goal**: Stop automated re-indexing

```
kbolt schedule --off
kbolt schedule --off --space {name}
kbolt schedule --off --space {name} --collection {name}
```

Without scoping flags, removes all schedules. With `--space` or `--collection`, removes only the matching schedule. Removes the corresponding launchd plist or cron job.

---

## MCP Client Flows

These flows are performed by an LLM/agent connected to kbolt via MCP stdio transport (`kbolt mcp`).

### 37. Discover spaces and collections

**Actor**: MCP client
**Goal**: Understand what's available to search

On MCP connection, the server injects dynamic instructions into the LLM's system prompt:
- Number of indexed documents, spaces, and collections
- Available spaces with descriptions and their collections (including collection descriptions)
- Guidance on when to use `mode: "deep"` vs default search

The LLM can also call the `status` tool for detailed information.

---

### 38. Search (MCP)

**Actor**: MCP client
**Goal**: Find relevant content

MCP tool call:
```json
{ "tool": "search", "query": "{query}", "space": "{name}", "collection": "{name}", "limit": "{n}", "mode": "{mode}" }
```

**Parameters**:
- **query** (required)
- **space** (optional) — scope to a space
- **collection** (optional) — scope to a collection
- **limit** (optional, default 10)
- **mode** (optional) — auto / deep / keyword / semantic

Same as CLI search (flow 28) but invoked as an MCP tool. When `space` is omitted, searches across all spaces (cross-space search). Returns structured results with docid, space, collection, path, title, heading, snippet, score.

---

### 39. Read one document (MCP)

**Actor**: MCP client
**Goal**: Read full content of a document found via search

MCP tool call:
```json
{ "tool": "get", "identifier": "{docid_or_path}", "space": "{name}" }
```

**Parameters**:
- **identifier** (required) — docid or collection-relative path
- **space** (optional) — resolve collection name within this space. Not needed when using docid (globally unique), but required when using a collection-relative path and the collection name exists in multiple spaces.

This is the primary reason `get` exists — the LLM has no file system access and needs this to read document content after finding it via search.

---

### 40. Read multiple documents (MCP)

**Actor**: MCP client
**Goal**: Read several related documents at once

MCP tool call:
```json
{ "tool": "multi_get", "locators": ["{path_or_docid}", ...], "space": "{name}", "max_files": "{n}", "max_bytes": "{n}" }
```

**Parameters**:
- **locators** (required) — array of paths (`collection/path`) and/or docids (`a1b2c3`), resolved in order given
- **space** (optional) — resolve collection references within this space
- **max_files** (optional, default 20)
- **max_bytes** (optional, default 50KB)

Lets the LLM pull several files in one call, respecting file count and byte budget caps. Locators are resolved in order, so the LLM can prioritize which files to read first. Response includes an `omitted` list with sizes for any files that didn't fit, so the LLM can fetch them individually via `get` if needed.

---

### 41. List files (MCP)

**Actor**: MCP client
**Goal**: Browse what's indexed

MCP tool call:
```json
{ "tool": "list_files", "space": "{name}", "collection": "{name}", "prefix": "{prefix}" }
```

**Parameters**:
- **space** (optional) — scope to a space
- **collection** (required)
- **prefix** (optional)

---

### 42. Check status (MCP)

**Actor**: MCP client
**Goal**: Understand index state

MCP tool call:
```json
{ "tool": "status", "space": "{name}" }
```

**Parameters**:
- **space** (optional) — scope to a specific space. If omitted, returns status for all spaces.

Returns space list, collection list per space, document/chunk counts, embedding status.

---

## Automated Flows

### 43. Scheduled re-index

**Actor**: System (cron/launchd)
**Goal**: Keep index fresh without human intervention

The system runs `kbolt update` (or `kbolt update --no-embed`) on the schedule configured via flow 34. Same pipeline as flow 23, no human interaction. Output is logged to system log.

---

## Deferred Flows (V2+)

The following flows are planned but deferred beyond V1. They are listed here so they don't get lost.

### Evaluation Framework

- **`kbolt eval add {query} {expected_paths}`** — Add a test case (query + expected relevant documents) to the eval dataset stored in `~/.config/kbolt/eval.toml`.
- **`kbolt eval run`** — Run all test cases against the current index. Measures MRR@10, Recall@K (K=1,5,10), and latency (p50, p95, p99).
- **`kbolt eval report`** — Show the most recent evaluation results. Useful for tracking retrieval quality over time as the index, models, or chunking strategy change.

### HTTP + MCP Server

- **`kbolt serve --port {n}`** — Start a long-running process that serves the HTTP API and MCP via streamable HTTP transport on a single port (default: 3777). Models are loaded into memory and shared across requests. The server keeps Tantivy and USearch indexes open for low-latency queries. Deferred because V1 uses `kbolt mcp` (stdio transport) for MCP and CLI for everything else.

### Cleanup

- **`kbolt cleanup`** — Remove orphaned content (content rows with no referencing documents), vacuum SQLite, and compact Tantivy segments. Returns a report: orphaned content removed, bytes reclaimed, vacuum duration. In V1, reaping of deactivated documents happens automatically during `kbolt update` (flow 23) based on the configured reaping period.

### Resource URIs (MCP)

- MCP clients can access documents via `kbolt://{space}/{collection}/{path}` resource URIs as an alternative to the `get` tool. Deferred until the MCP resource protocol stabilizes.

---

## Global CLI Options

These options apply across multiple commands, not to any specific flow:

- **-s, --space {name}** — set the active space for this command. Overrides `KBOLT_SPACE` env var and configured default. Applies to all commands that operate on collections.
- **-f, --format cli|json** — output format. `cli` is human-readable (default), `json` is machine-readable for scripting. Applies to: search, status, space list/info, collection list/info, models list, ls.
- **--help** — show usage for any command
- **--version** — show kbolt version

### Space Resolution Precedence

When a command needs to resolve which space to use:

1. **`--space` flag** — explicit, always wins
2. **`KBOLT_SPACE` env var** — set once, applies to all commands in that shell
3. **Configured default** — set via `kbolt space default {name}`, persisted in `~/.config/kbolt/index.toml`
4. **Unique lookup** — if a collection name exists in exactly one space, use that space automatically
5. **Error** — if ambiguous (collection exists in multiple spaces), error with guidance telling the user to specify `--space`

Commands that operate on all data by default (e.g. `kbolt update` without `--collection`, `kbolt search` without `--collection`, `kbolt status`) scan all spaces unless `--space` is provided.
