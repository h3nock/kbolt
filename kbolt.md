# Kbolt — V1 Specification

> *A local-first retrieval engine that brings your documents to light.*

## Overview

A best-in-class local-first retrieval engine: correct, fast, and extensible. V1 focuses on core retrieval quality with clean extension points. Model training, SPLADE, and advanced features are deferred to V2.

**Stack**: Rust, Tantivy (FTS), USearch (dense vectors), SQLite (metadata + entity storage), ONNX Runtime (embeddings), llama-cpp-rs (reranking/generation), tree-sitter (code), pulldown-cmark (markdown), clap (CLI), TOML (system config).

**Distribution**: Shell installer script (`curl -fsSL https://... | sh`) + Homebrew tap + GitHub Releases with prebuilt binaries for macOS (arm64, x86_64) and Linux (x86_64, arm64). One-liner install, no Rust toolchain required. `cargo install` as fallback for Rust developers.

---

## Crate Structure

```
crates/
  core/       # Engine struct: ingestion, retrieval, storage, models, config
  types/      # Shared request/response types, no logic
  cli/        # CLI adapter: clap -> Engine -> formatted output
  mcp/        # MCP server: stdio transport -> Engine -> MCP responses
```

CLI and MCP are thin adapters over the `Engine` struct. HTTP server (axum, Streamable HTTP MCP transport) deferred to V2.

---

## Architecture

### Process Model

Kbolt is a **single process** in V1. No daemon, no background service.

```
kbolt search "query"     → process starts, searches, prints, exits
kbolt update             → process starts, scans/indexes, exits
kbolt mcp                → process starts, stays alive for MCP client session, exits on disconnect
```

No process running between commands. Models live as long as the process — loaded on first use within a session, freed when the process exits. MCP sessions keep models warm naturally (process stays alive). CLI commands cold-start if dense/reranking is needed.

Scheduled indexing via OS-level schedulers (cron/launchd), set up by `kbolt schedule` helper command.

V2 extension path: `kbolt serve` daemon (HTTP + MCP). CLI would check "is daemon running?" and send requests to it instead of creating a local Engine. Core modules don't change — only the CLI adapter adds a daemon-check wrapper, plus a new HTTP crate.

### Crate Dependency Graph

```
types/          (no dependencies — pure data structs)
  ▲
  │
core/           (depends on: types)
  ▲
  │
  ├── cli/      (depends on: core, types)
  │
  └── mcp/      (depends on: core, types)
```

Dependencies flow strictly upward. CLI and MCP never depend on each other. Core never depends on CLI or MCP. Types depends on nothing.

- **types/** — every struct that crosses a module boundary (requests, responses, reports). No logic, no `impl` blocks with behavior. Just data definitions with `serde` derives.
- **core/** — all business logic, storage, model inference. 90% of the code. External dependencies: `rusqlite`, `tantivy`, `usearch`, `ort`, `llama-cpp-rs`, `tree-sitter`, `pulldown-cmark`, `tokenizers`, `hf-hub`.
- **cli/** — thin adapter. Parses args (clap), calls Engine, formats terminal output. External dependencies: `clap`.
- **mcp/** — thin adapter. MCP protocol over stdio, maps tool calls to Engine methods. External dependencies: `mcp-sdk` or hand-rolled stdio JSON-RPC.

### System Architecture

```
                          ┌───────┐  ┌───────┐
                          │  CLI  │  │  MCP  │
                          └───┬───┘  └───┬───┘
                              └────┬─────┘
                                   │ &Engine
───────────────────────────────────┼─────────────────────────────────
  Core                             │
                                   ▼
                      ┌──────────────────────┐
                      │        Engine         │
                      │  .storage    .models  │
                      │  .config              │
                      └───────────┬───────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                         ▼
   ┌───────────┐           ┌───────────┐             ┌───────────┐
   │  search/  │           │  ingest/  │             │  config/  │
   │ execute() │           │  update() │             │   load()  │
   │   deep()  │           │           │             │           │
   └─────┬─────┘           └──┬─────┬──┘             └─────┬─────┘
         │                    │     │                       │
         │  ┌─────────────────┘     │                       │
    ┌────┴──┴───────────────┐       │                       │
    ▼      (borrows)        ▼       │ (reads)               │
┌───────────┐        ┌───────────┐  │                       │
│  storage/ │        │  models/  │  │                       │
│           │        │           │  │                       │
│ sqlite.rs │        │  embed()  │  │                       │
│tantivy.rs │        │  rerank() │  │                       │
│usearch.rs │        │ generate()│  │                       │
└─────┬─────┘        └─────┬─────┘  │                       │
      │                    │        │                       │
──────┼────────────────────┼────────┼───────────────────────┼───────
      │  External          │        │                       │
      ▼                    ▼        ▼                       ▼
┌─────────────┐     ┌───────────┐ ┌───────────┐       ┌──────────┐
│  ~/.cache/   │     │ ONNX RT   │ │   Files   │       │~/.config/│
│              │     │ llama-cpp │ │  on Disk  │       │index.toml│
│ meta.sqlite  │     └─────┬─────┘ └───────────┘       │ ignores/ │
│ spaces/      │           ▼                            └──────────┘
│  {name}/     │     ┌──────────┐     ┌─────────────┐
│   tantivy/   │     │  Model   │ ◀───│ HuggingFace │
│   vectors    │     │  Files   │     │ Hub         │
└──────────────┘     └──────────┘     └─────────────┘
```

### External I/O

Every module that crosses the Core boundary:

| Module | External System | Direction | What |
|---|---|---|---|
| `storage/` | `~/.cache/meta.sqlite` | read/write | SQLite — all entity CRUD (spaces, collections, documents, chunks, embeddings, cache) |
| `storage/` | `~/.cache/spaces/{name}/tantivy/` | read/write | Per-space Tantivy — BM25 index |
| `storage/` | `~/.cache/spaces/{name}/vectors.usearch` | read/write | Per-space USearch — HNSW vector index |
| `models/` | ONNX Runtime | in-process | Embedding inference (CPU, thread-safe) |
| `models/` | llama-cpp | in-process | Reranking + generation (GPU, single-thread) |
| `models/` | `~/.cache/models/` | read | Load .onnx and .gguf files from disk |
| `models/` | HuggingFace Hub | download | Fetch model files on first use |
| `config/` | `~/.config/index.toml` | read/write | System settings (models, reaping, default space) |
| `config/` | `~/.config/ignores/` | read/write | Ignore patterns (per space/collection) |
| `ingest/` | Collection directories | read | Scan files, read bytes, compute hashes |

Only `storage/`, `models/`, `config/`, and `ingest/` touch the outside world. `search/` and `engine` are pure internal — they only call other modules.

### Component Connectivity

**Ownership** — Engine creates and holds as struct fields:
- Engine → Storage: creates at startup, holds for process lifetime
- Engine → Models: creates at startup (passing ModelConfig from Config), holds for process lifetime
- Engine → Config: loads at startup via `config::load()`, holds for process lifetime

**Delegation** — Engine calls module functions, passing borrowed references:
- Engine → `search::execute(&storage, &models, req)`: hybrid search
- Engine → `search::deep(&storage, &models, req)`: expanded search
- Engine → `ingest::update(&storage, &models, &config, opts)`: full ingestion pipeline
- Engine → `storage.*` directly: for simple passthrough (get, list, collection CRUD)

**Borrowing** — modules receive `&Storage` and `&Models` as function parameters:
- `search/` receives `&Storage` → calls: `query_bm25()`, `query_dense()`, `get_chunks()`, `get_document_meta()`
- `search/` receives `&Models` → calls: `embed()` (query vector), `rerank()` (cross-encoder)
- `ingest/` receives `&Storage` → calls: `upsert_document()`, `delete_chunks()`, `insert_chunks()`, `insert_tantivy()`, `insert_usearch()`, `insert_embedding()`, `deactivate_document()`, `reap_documents()`
- `ingest/` receives `&Models` → calls: `embed()` (chunk vectors)
- `ingest/` receives `&Config` → reads: `reaping.days` (for hard-delete threshold)

**Direct external I/O** — modules that cross the Core boundary beyond borrowing:
- `ingest/` → reads files from disk: scans collection directories, reads file bytes, computes SHA-256 hashes
- `search/` has no external I/O — all data access goes through borrowed `&Storage` and `&Models`

**Construction-time data flow** (once when Engine is created):
- `config::load(path)` → returns `Config` struct (includes `ModelConfig`)
- `Engine::new()` passes `config.models` → `Models::new()`
- `Engine::new()` passes `config.cache_dir` → `Storage::new()`
- After construction, Config and Models have no direct connection

**No connection** (by design):
- `config/` ↔ `storage/`: independent
- `config/` ↔ `models/`: Engine bridges them at construction
- `search/` ↔ `ingest/`: never call each other
- `storage/` ↔ `models/`: never call each other

**Module dependency rules** — no circular dependencies:
- `config/` → nothing (reads TOML, returns Config struct)
- `storage/` → nothing (receives operations, manages three stores)
- `models/` → nothing (loads models, runs inference)
- `search/` → uses Storage and Models (via borrowed references)
- `ingest/` → uses Storage, Models, and Config (via borrowed references)
- `engine` → all of the above (owns instances, delegates via borrowing)

**Public vs internal functions**: Each module exposes only its entry points via `mod.rs`. Internal sub-module functions (`query::parse()`, `fusion::fuse()`, etc.) are private. Engine only sees top-level public functions.

### Core Module Structure

```
core/
  lib.rs                 # pub mod engine, storage, search, ingest, models, config
  engine.rs              # Engine struct — thin orchestrator, delegates to modules

  storage/
    mod.rs               # Storage struct — public API, owns db + tantivy + usearch
    sqlite.rs            # SQLite operations (collections, documents, chunks, embeddings, cache)
    tantivy.rs           # Tantivy operations (index, query, field boosting)
    usearch.rs           # USearch operations (insert, search, quantization)

  search/
    mod.rs               # pub fn execute(), pub fn deep() — only public entry points
    query.rs             # query parsing (phrases, filters, negations) — internal
    routing.rs           # query routing (decide which signals to activate) — internal
    fusion.rs            # RRF fusion, agreement bonus, deduplication — internal
    rerank.rs            # cross-encoder reranking, score blending — internal

  ingest/
    mod.rs               # pub fn update() — only public entry point
    extract.rs           # Extractor trait + MarkdownExtractor, PlaintextExtractor, CodeExtractor
    chunk.rs             # chunking pipeline (budget enforcement, overlap, merging)
    embed.rs             # embedding pipeline (batch embed, USearch insert)
    ignore.rs            # ignore pattern parsing, hardcoded ignores, extension filtering

  models/
    mod.rs               # Models struct — facade, lazy loading, public API (embed/rerank/generate)
    embedder.rs          # Embedder trait + OnnxEmbedder implementation
    reranker.rs          # Reranker trait + GgufReranker implementation
    generator.rs         # Generator trait + GgufGenerator implementation

  config/
    mod.rs               # Config struct, TOML loading/saving (models + reaping settings only)
```

### Engine (composition root)

Engine owns Storage, Models, and Config as struct fields. When Engine is created, it creates all three. When Engine is dropped, all three are dropped (connections closed, models unloaded). Every operation goes through Engine.

```rust
pub struct Engine {
    storage: Storage,
    models: Models,
    config: Config,
}

impl Engine {
    // Construction — the only place where Config feeds into Storage and Models
    pub fn new(config_path: &Path) -> Result<Self> {
        let config = config::load(config_path)?;
        let storage = Storage::new(&config.cache_dir)?;
        let models = Models::new(config.models.clone(), &config.cache_dir.join("models"))?;
        Ok(Engine { storage, models, config })
    }

    // Delegation — Engine passes borrowed references to domain modules
    pub fn search(&self, req: SearchRequest) -> Result<SearchResponse> {
        search::execute(&self.storage, &self.models, req)
    }

    pub fn deep_search(&self, req: SearchRequest) -> Result<SearchResponse> {
        search::deep(&self.storage, &self.models, req)
    }

    pub fn update(&self, opts: UpdateOptions) -> Result<UpdateReport> {
        ingest::update(&self.storage, &self.models, &self.config, opts)
    }

    // Passthrough — simple operations go directly to storage
    pub fn get_document(&self, req: GetRequest) -> Result<DocumentResponse> {
        self.storage.get_document(req)
    }

    // ... other passthrough methods for collections, list_files, status, etc.
}
```

Adapters (CLI, MCP) take `&Engine` directly. They parse input into request types, call Engine methods, format the response.

### Storage (three-store model, owns concurrency)

Storage is the data layer. SQLite is global (one database for all spaces). Tantivy and USearch are per-space — each space gets its own index for BM25 IDF isolation. All connected by `chunk_id` as the universal join key.

```rust
pub struct Storage {
    db: Mutex<rusqlite::Connection>,                     // global, single writer, WAL
    spaces: HashMap<String, SpaceIndexes>,               // per-space search indexes
}

struct SpaceIndexes {
    tantivy_reader: tantivy::IndexReader,                // thread-safe (Clone + Send + Sync)
    tantivy_writer: Mutex<tantivy::IndexWriter>,         // single writer
    usearch: RwLock<usearch::Index>,                     // concurrent reads, exclusive writes
}
```

| Store | Role | Stores | Good at |
|---|---|---|---|
| SQLite | Source of truth | All entities (spaces, collections, documents, chunks, embeddings, cache) | Relational queries, joins, CRUD, metadata |
| Tantivy (per-space) | BM25 search | Denormalized chunk entries (chunk_id, filepath, title, heading, body) | Full-text search, tokenization, field boosting |
| USearch (per-space) | Dense search | Vectors keyed by chunk_id | Approximate nearest neighbor in high dimensions |

SQLite is the source of truth. Tantivy and USearch are derived indexes — if corrupted or deleted, they can be rebuilt from SQLite + files on disk + model inference. Per-space isolation means a code-heavy space won't skew BM25 IDF statistics for a notes space.

Search and ingest don't manage locks — they call `storage.query_bm25(space, ...)`, `storage.insert_chunks(...)`, etc. Storage handles synchronization internally. Cross-space search queries each space's indexes independently, then concatenates candidates for fusion and reranking.

### Models (facade, owns lifecycle)

Models is a facade that hides all ML complexity from the rest of the system. Engine calls `models.embed()`, `models.rerank()`, `models.generate()` without knowing about ONNX, llama-cpp, HuggingFace downloads, or thread-safety wrappers.

```rust
pub struct Models {
    embedder: Option<Arc<dyn Embedder>>,       // None until first use, ONNX: thread-safe
    reranker: Option<Mutex<dyn Reranker>>,     // None until first use, llama-cpp: NOT thread-safe
    generator: Option<Mutex<dyn Generator>>,   // None until first use, llama-cpp: NOT thread-safe
    config: ModelConfig,
    model_dir: PathBuf,
}
```

Models are loaded lazily on first use and stay loaded for the lifetime of the process. In V1 (single process), this means models are freed when the process exits. MCP sessions keep models warm naturally.

Three separate traits — each defines its own contract because they share nothing in common (different inputs, outputs, runtimes, thread-safety):

```rust
pub trait Extractor: Send + Sync {
    fn supports(&self) -> &[&str];   // file extensions: ["md", "markdown"]
    fn extract(&self, path: &Path, bytes: &[u8]) -> Result<ExtractedContent>;
}

pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}

pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>>;
}

pub trait Generator: Send + Sync {
    fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String>;
}
```

### Adapter Pattern (CLI and MCP)

Both adapters follow the same pattern — thin translation layers between external interface and Engine:

```
User input → Parse into Request type → Engine method → Response type → Format output
```

CLI parses command-line args (clap) into request types, calls Engine, prints formatted results.
MCP parses JSON-RPC tool calls into request types, calls Engine, returns MCP JSON responses.

The adapters contain no business logic. All validation, search, indexing, and error handling happens inside Core.

---

## Entity Model

### Entity Relationships

```
Space (SQLite)
    │
    │ space_id (integer FK, ON DELETE CASCADE)
    ▼
Collection (SQLite)
    │
    │ collection_id (integer FK, ON DELETE CASCADE)
    ▼
Document (SQLite)
    │
    │ doc_id (integer FK, ON DELETE CASCADE)
    ▼
Chunk (SQLite)
    │
    ├── chunk_id (integer FK, ON DELETE CASCADE) ──▶ Embedding (SQLite + USearch vector)
    │
    └── chunk_id (stored field) ──▶ Tantivy entry (denormalized copy for BM25 search)
```

Full CASCADE chain: deleting a Space removes all its Collections, which removes all their Documents, which removes all their Chunks, which removes all their Embeddings. Tantivy and USearch entries must be explicitly cleaned up by application code (not SQL — they're external indexes). Each space has its own Tantivy index and USearch file for BM25 IDF isolation.

### Entity Details

**Space** — A grouping of related collections. Provides organizational boundaries and BM25 IDF isolation (each space has its own Tantivy index and USearch file). Managed via CLI commands, stored in SQLite.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER PK | Stable FK target (survives renames) |
| `name` | TEXT UNIQUE | Human-readable identifier (e.g. `work`, `personal`, `notes`) |
| `description` | TEXT nullable | What this space contains (shown in MCP instructions, CLI) |
| `created` | TEXT | ISO 8601, when the space was created |

A `default` space always exists, created automatically on first use. It is the implicit destination for `kbolt collection add` when no `--space` is specified. The `default` space has no special weight in name resolution — it is just another space.

**Collection** — A directory the user wants indexed. Belongs to exactly one space. Managed via CLI commands, stored in SQLite.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER PK | Stable FK target (survives renames) |
| `space_id` | INTEGER FK | References `spaces.id`, CASCADE delete |
| `name` | TEXT | Human-readable identifier, unique within its space |
| `path` | TEXT | Absolute path to directory on disk |
| `description` | TEXT nullable | Human-readable description (shown in MCP instructions, CLI) |
| `extensions` | TEXT nullable | JSON array of allowed extensions (e.g. `["rs","py"]`), NULL = all supported |
| `created` | TEXT | ISO 8601, when the collection was added |
| `updated` | TEXT | ISO 8601, last config change or successful update |

**Document** — A file within a collection. Created/updated/deactivated during `kbolt update`.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER PK | Stable FK target for chunks |
| `collection_id` | INTEGER FK | References `collections.id`, CASCADE delete |
| `path` | TEXT | Relative path from collection root (e.g. `rust/error-handling.md`) |
| `title` | TEXT | Extracted from first heading, or filename if no heading |
| `hash` | TEXT | SHA-256 of file bytes — used for change detection |
| `modified` | TEXT | ISO 8601, file modification timestamp |
| `active` | INTEGER | 1 = live, 0 = deactivated (file disappeared from disk) |
| `deactivated_at` | TEXT nullable | ISO 8601 when deactivated, NULL when active |

UNIQUE constraint on `(collection_id, path)`. Soft delete with auto-reaping: documents deactivated for longer than the configured reaping period (default: 7 days) are hard-deleted during `kbolt update`.

**Space Resolution** — How kbolt determines which space to use when the user doesn't specify one explicitly. Four-level precedence:

1. **`--space` flag** — explicit, always wins
2. **`KBOLT_SPACE` env var** — set once, applies to all commands in that shell
3. **Configured default** — set via `kbolt space default {name}`, persisted in `index.toml`
4. **Unique lookup** — if a collection name exists in exactly one space, use that space. If it exists in multiple spaces, error with guidance (never silently default).

If none of the above resolve, the command either operates on all spaces (for `update`, `search`, `status`) or errors with a message telling the user to specify `--space`.

**Chunk** — A piece of a document. The unit of retrieval for both BM25 and dense search.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER PK | Key used in Tantivy and USearch indexes |
| `doc_id` | INTEGER FK | References `documents.id`, CASCADE delete |
| `seq` | INTEGER | Position within document (0, 1, 2...), for ordering |
| `offset` | INTEGER | Byte position where this chunk starts in the source file |
| `length` | INTEGER | Byte count of this chunk in the source file |
| `heading` | TEXT nullable | Heading breadcrumb (e.g. `# Intro > ## Setup`) |
| `kind` | TEXT | `section`, `function`, `class`, or `paragraph` |

UNIQUE constraint on `(doc_id, seq)`. Offset and length define a byte slice into the source file on disk — at query time, the snippet is extracted by reading `length` bytes starting at `offset` from the file at `{collection.path}/{document.path}`. Chunks are immutable: when a document changes, all its chunks are deleted and re-created.

**Embedding** — Metadata tracking that a chunk has been embedded by a specific model. The actual float vector lives in USearch.

| Column | Type | Purpose |
|---|---|---|
| `chunk_id` | INTEGER FK | References `chunks.id`, CASCADE delete |
| `model` | TEXT | Model identifier (e.g. `EmbeddingGemma-256`) |
| `embedded_at` | TEXT | ISO 8601, when the embedding was created |

PRIMARY KEY on `(chunk_id, model)`. Tracks which chunks have been embedded and by which model. When the embedding model changes, stale entries are detected by comparing the `model` column against the current model config.

**LLM Cache** — Internal cache for expensive model calls (query expansion). Not a domain entity.

| Column | Type | Purpose |
|---|---|---|
| `key` | TEXT PK | Hash of (model identifier + prompt) |
| `value` | TEXT | The model's response |
| `created` | TEXT | ISO 8601 |

**Docid** — Not an entity. A computed display value: `"#" + document.hash[0..6]`. Used for quick document reference in CLI output and `kbolt get #a1b2c3`. Looked up via `SELECT ... FROM documents WHERE hash LIKE 'a1b2c3%'`.

### SQLite Schema

```sql
CREATE TABLE spaces (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    created     TEXT NOT NULL
);

CREATE TABLE collections (
    id          INTEGER PRIMARY KEY,
    space_id    INTEGER NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    description TEXT,
    extensions  TEXT,
    created     TEXT NOT NULL,
    updated     TEXT NOT NULL,
    UNIQUE(space_id, name)
);

CREATE TABLE documents (
    id              INTEGER PRIMARY KEY,
    collection_id   INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    path            TEXT NOT NULL,
    title           TEXT NOT NULL,
    hash            TEXT NOT NULL,
    modified        TEXT NOT NULL,
    active          INTEGER NOT NULL DEFAULT 1,
    deactivated_at  TEXT,
    UNIQUE(collection_id, path)
);
CREATE INDEX idx_documents_collection ON documents(collection_id, active);
CREATE INDEX idx_documents_hash ON documents(hash);

CREATE TABLE chunks (
    id       INTEGER PRIMARY KEY,
    doc_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq      INTEGER NOT NULL,
    offset   INTEGER NOT NULL,
    length   INTEGER NOT NULL,
    heading  TEXT,
    kind     TEXT NOT NULL DEFAULT 'section',
    UNIQUE(doc_id, seq)
);
CREATE INDEX idx_chunks_doc ON chunks(doc_id);

CREATE TABLE embeddings (
    chunk_id    INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

CREATE TABLE llm_cache (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL,
    created TEXT NOT NULL
);
```

### Tantivy Schema (per-space)

Each space has its own Tantivy index at `~/.cache/kbolt/spaces/{space}/tantivy/`. Each Tantivy entry represents one chunk (not one document). Fields are passed explicitly by application code at index time — Tantivy has no awareness of our SQLite schema.

```
chunk_id:  u64  (stored, fast)          — join key back to SQLite chunks table
filepath:  TEXT (stored, indexed, 2x)   — "collection_name/relative/path.md"
title:     TEXT (stored, indexed, 3x)   — document title (copied from parent Document at index time)
heading:   TEXT (stored, indexed, 2x)   — chunk heading breadcrumb
body:      TEXT (indexed, NOT stored)   — chunk text (indexed for BM25, read from disk at query time)
```

Tokenizer: default with stemming + unicode.

Field boosting: title (3x) > filepath (2x) = heading (2x) > body (1x). A match in the title is a stronger relevance signal than a match in the body.

### USearch Config (per-space)

```
Metric:         Cosine
Dimensions:     256 (EmbeddingGemma) — configurable per model
Quantization:   f16
Connectivity:   M=16, ef_construction=200, ef_search=100
Key type:       u64 (chunk_id from SQLite)
```

### Storage Layout

```
~/.config/kbolt/
    index.toml              # system config (models, reaping, default space)
    ignores/                # ignore patterns (internal, not in user directories)
        {space}/
            {collection}.ignore   # .gitignore syntax, per-collection

~/.cache/kbolt/
    meta.sqlite             # all entities: spaces, collections, documents, chunks, embeddings, cache
    spaces/                 # per-space indexes (BM25 IDF isolation)
        {space}/
            tantivy/        # Tantivy index (managed by Tantivy)
            vectors.usearch # USearch HNSW file (managed by USearch)
    models/                 # downloaded via HuggingFace Hub (hf-hub crate)
        embed.onnx
        reranker.gguf
        expander.gguf
```

SQLite is shared globally (one database for all spaces). Tantivy and USearch are per-space — each space has its own index directory under `~/.cache/kbolt/spaces/{space_name}/`. This gives each space independent BM25 IDF statistics, preventing one space's content from skewing another's keyword search quality.

Ignore patterns are stored internally at `~/.config/kbolt/ignores/{space}/{collection}.ignore`, not in the user's directories. Kbolt does not own user directories and should not place files there. Users manage ignore patterns via `kbolt ignore` commands.

---

## File Filtering

Three filtering layers determine which files get indexed, applied in order:

### 1. Hardcoded Ignores (always active)

```
.git/
node_modules/
.DS_Store
*.lock
```

Skipped before any other filter runs. Never useful to index.

### 2. Ignore Patterns (user-defined, per collection)

Stored internally at `~/.config/kbolt/ignores/{space}/{collection}.ignore`. Uses `.gitignore` syntax. Managed via `kbolt ignore` commands — not placed in the user's directories.

```
# example ignore patterns for collection "api" in space "work"
# stored at: ~/.config/kbolt/ignores/work/api.ignore
dist/
build/
vendor/
*.generated.*
*.min.js
```

Users interact with ignore patterns through a virtual path structure: `{space}/{collection}` maps to the internal ignore file. For example, `kbolt ignore show api` shows the ignore patterns for the `api` collection (resolved via space precedence), and `kbolt ignore edit api` opens the file in `$VISUAL` / `$EDITOR` / `vi`.

### 3. Extractor Registry + Extensions Filter

A file is only indexed if:
1. An extractor supports its extension (MarkdownExtractor handles `.md`, CodeExtractor handles `.rs`, etc.)
2. If the collection has an `extensions` list, the file's extension is in that list

Both ignore patterns and `extensions` can filter by extension. They compose — both must pass. Ignore patterns are better for excluding paths/directories. `extensions` is better when the set of wanted types is small (e.g. `extensions = ["rs"]` is cleaner than ignoring every other supported extension in the ignore file).

---

## Extractor System

```rust
pub struct ExtractedContent {
    pub chunks: Vec<PreChunk>,               // structural splits from the source
    pub metadata: HashMap<String, String>,   // frontmatter, language, etc.
}

pub struct PreChunk {
    pub text: String,
    pub offset: usize,          // byte offset in source file
    pub length: usize,          // byte length in source file
    pub kind: ChunkKind,
    pub heading: Option<String>, // heading breadcrumb
}

pub enum ChunkKind {
    Section,     // markdown section under a heading
    Function,    // code function/method
    Class,       // code class/struct
    Paragraph,   // plaintext paragraph
}
```

### V1 Extractors

1. **MarkdownExtractor** (pulldown-cmark AST)
   - Splits on headings (H1-H6), tracks heading breadcrumb stack
   - Respects code fences — never splits inside
   - Handles YAML frontmatter (extracted to metadata)
   - Each section = one PreChunk with heading

2. **PlaintextExtractor** (paragraph-based)
   - Splits on double newlines
   - Groups small paragraphs up to chunk size

3. **CodeExtractor** (tree-sitter AST)
   - Splits on function/method/class boundaries
   - Preserves imports as preamble chunk
   - V1 languages: Rust, Python, TypeScript/JavaScript, Go, C/C++

### Chunking Pipeline

Extractors produce PreChunks (structural splits). The chunker enforces token budget:

1. PreChunk fits within 512 tokens → use as-is
2. PreChunk too large → sub-split at break points (headings > code blocks > blank lines > sentences), 15% overlap between sub-chunks
3. PreChunk too small → merge with adjacent PreChunks of same kind until budget is reached

Token budget: **512 tokens** per chunk (tighter chunks improve retrieval precision — each chunk represents a more focused semantic unit).

Token counting: HuggingFace `tokenizers` crate with the embedding model's actual tokenizer. Loaded from model config, exact counts, microsecond-fast (native Rust). Token budget matches the model's actual vocabulary.

---

## Ingestion Pipeline

`kbolt update` is a single command that scans, extracts, chunks, indexes (FTS), and embeds (dense vectors). The `--no-embed` flag skips embedding if models aren't available.

### Per-Document Flow

```
For each collection:
  Scan directory (apply hardcoded ignores → ignore patterns → extractor registry → extensions filter)
  │
  For each file that passes filters:
  │  Read file, compute SHA-256 hash
  │  Compare with stored document hash
  │  │
  │  ├── Hash unchanged → skip entirely
  │  │
  │  └── Hash changed OR new file:
  │      ├── Run extractor (Markdown/Code/Plaintext) → PreChunks
  │      ├── Run chunker (enforce 512-token budget) → final Chunks
  │      ├── UPSERT document row in SQLite (id, collection_id, path, title, hash, modified)
  │      ├── DELETE old chunks for this document (CASCADE clears embeddings)
  │      ├── Remove old entries from Tantivy and USearch (by old chunk_ids)
  │      ├── INSERT new chunk rows in SQLite
  │      ├── For each chunk:
  │      │   ├── Add to Tantivy (chunk_id, filepath, title, heading, body)
  │      │   └── [if models available and --no-embed not set]:
  │      │       ├── Embed chunk text → 256-dim vector
  │      │       ├── INSERT embedding row in SQLite (chunk_id, model, embedded_at)
  │      │       └── INSERT vector in USearch (key=chunk_id, vector)
  │      └── Done with this file
  │
  For files that disappeared from disk:
  │  ├── Deactivate document (active=0, deactivated_at=now)
  │  └── Leave chunks/embeddings intact (reusable if file returns)
  │
  For deactivated documents returning (same hash):
  │  └── Re-activate (active=1, deactivated_at=NULL, skip re-indexing)
  │
  Reaping phase:
  │  └── Hard-delete documents deactivated longer than reaping period (default: 7 days)
  │      CASCADE removes chunks → embeddings; explicitly remove from Tantivy + USearch
  │
  Commit Tantivy writer
```

The file is the unit of change. When a file's hash changes, all its chunks are deleted and re-created — no chunk-level diffing. This is correct because chunk boundaries shift when content changes (a new paragraph shifts every subsequent offset).

### Embedding Integrity

During `update`, the system automatically detects and corrects two embedding integrity problems. No user intervention or flags needed.

**Model mismatch detection**: The `embeddings` table records which model produced each embedding (the `model` column). On each update, the system compares stored model names against the current model in `index.toml`. If they differ — the user changed their embedding model — all existing embedding rows for the old model are deleted and the corresponding USearch vectors are removed. The affected chunks are then re-embedded with the new model. This happens transparently during the normal embedding phase.

**USearch sync check**: The `embeddings` table in SQLite is the ledger of what *should* be in USearch. USearch stores the actual float vectors. These can diverge — USearch file deleted, corrupted, or partially written. On each update, the system compares the count of embedding rows in SQLite against the count of vectors in USearch. If they disagree, the system clears both (embeddings rows + USearch vectors) and re-embeds all chunks from scratch. This is a coarse check — it catches file-level corruption, not individual vector corruption, which is sufficient because USearch either works or it doesn't.

**Why no `--force-embed` flag**: Both real scenarios (model change, USearch corruption) are handled by auto-detection. A manual `--force-embed` flag would require the user to understand that embeddings exist as a separate layer, that they can go stale, and that the system might have missed something — three layers of internals. If auto-detection has a bug, the fix is to fix auto-detection, not to expose an escape hatch.

---

## Retrieval Pipeline

### Stage 1: Query Understanding

Parse the raw query string to extract:
- Quoted phrases → exact match requirements
- File filters → `file:*.rs`, `collection:notes`
- Negations → `-excluded`
- Plain terms → everything else

Pure parsing, no LLM.

### Stage 2: Query Routing

Decides which retrieval signals to activate based on query characteristics:

| Query type | BM25 | Dense | Reranker |
|---|---|---|---|
| Exact phrase (`"error handling"`) | yes | no | no |
| Short keyword (`rust trait`) | yes | yes | yes |
| Natural question (`how does X work?`) | yes | yes | yes |
| File filter only (`file:*.rs`) | yes | no | no |

Rule-based (no LLM). `--no-rerank` flag skips reranker. `--keyword` forces BM25-only. `--semantic` forces dense-only.

### Stage 3: Query Expansion (--deep only)

Expander model (Qwen3 1.7B GGUF) generates three query variants:
- **Lexical variant**: rephrased with different vocabulary (for BM25 recall)
- **Semantic variant**: describes the concept differently (for dense recall)
- **HyDE variant**: a hypothetical answer paragraph (embedded for vector search)

Each variant is fed to Stage 4 independently, producing its own BM25 and dense candidate lists. All candidates from all variants are collected into a single pool, then enter Stage 5 (fusion) together. Results are cached in `llm_cache` to avoid repeating expensive model calls.

### Stage 4: Multi-Signal Retrieval (parallel)

Both signals operate on the same unit: individual chunks.

**BM25 via Tantivy**:
- Query across `body` (1x), `title` (3x), `heading` (2x), `filepath` (2x) with field boosting
- Top-K=100 candidates, scores normalized to [0, 1]

**Dense via USearch**:
- Embed query via ONNX, search HNSW for top-K=100 nearest neighbors
- Score = 1 - cosine_distance

Both run in parallel threads. Because both operate at chunk granularity, RRF fusion compares equivalent units.

**Cross-space search**: When searching across multiple spaces (no `--space` specified), each space's Tantivy index and USearch file are queried independently. Candidate lists from all spaces are concatenated before entering fusion. The reranker (Stage 7) normalizes scores across spaces — since it scores each (query, chunk) pair independently, it doesn't matter which space the chunk came from.

### Stage 5: Fusion (RRF)

Reciprocal Rank Fusion combines BM25 and dense rankings:

```
RRF(d) = w_bm25 / (k + rank_bm25(d)) + w_dense / (k + rank_dense(d))
```

Defaults: `k=60`, `w_bm25=1.0`, `w_dense=1.0`. Agreement bonus: 1.2x multiplier for chunks appearing in both result sets.

### Stage 6: Document Deduplication

Multiple chunks from the same document → keep only the highest-scoring chunk per document. Remaining slots filled by next-best results from other documents.

### Stage 7: Reranking

Cross-encoder (Qwen3-Reranker 0.6B GGUF) scores top-20 candidates:
- Input: (query, chunk_text) pairs — chunk text read from disk via offset/length
- Output: relevance score per pair
- Final score: `0.7 * reranker_score + 0.3 * rrf_score` (blended, weights tunable)

### Stage 8: Result Assembly

Per result:
- Document metadata from SQLite (title, path, collection name, space name via join)
- Collection and space descriptions from SQLite
- Heading breadcrumb from chunk
- Snippet extracted by reading source file at chunk offset/length
- Short docid (`#` + first 6 chars of document hash)

Return top-10 results (default, configurable via `--limit`).

---

## Configuration (TOML)

System-level settings only. Spaces and collections are stored in SQLite, managed via CLI.

```toml
# ~/.config/kbolt/index.toml

default_space = "work"    # optional, set via `kbolt space default`

[models]
embed = "google/EmbeddingGemma-256"
reranker = "ExpedientFalcon/qwen3-reranker-0.6b-q8"
expander = "Qwen/Qwen3-1.7B-q4"

[reaping]
days = 7    # hard-delete documents deactivated longer than this
```

---

## CLI Commands

```
USAGE: kbolt <command> [options]

GLOBAL FLAGS
  -s, --space <name>               Set active space (overrides KBOLT_SPACE and default)
  -f, --format <fmt>               Output: cli|json (default: cli)

SEARCH (one command, mode flags)
  kbolt search <query>              Smart hybrid search (routes by query type)
    -d, --deep                     Expand query + multi-variant + rerank
    -k, --keyword                  BM25 only
    --semantic                     Dense vector only
    --no-rerank                    Skip reranking
    -c, --collection <name,...>    Scope to one or more collections (comma-separated)
    -n, --limit <N>                Max results (default: 10)

DOCUMENTS
  kbolt get <path|docid>            Get document by path or #docid
    --offset <N>                   Start at line N
    --limit <N>                    Max lines
  kbolt multi-get <glob|paths>      Batch retrieve (glob or comma-separated)
    --max-bytes <N>                Max total bytes (default: 10KB)
  kbolt ls [collection] [prefix]    List files in collection

SPACES
  kbolt space add <name> [dirs...]  Create space, optionally add directories as collections
    --description <text>           Space description
  kbolt space remove <name>         Remove space and all its collections/data
  kbolt space rename <old> <new>
  kbolt space list                  List all spaces with collection counts
  kbolt space info <name>           Show space details and stats
  kbolt space default [name]        Get or set the default space
  kbolt space current               Show the currently active space (flag > env > default)
  kbolt space describe <name> <text>  Update space description

COLLECTIONS
  kbolt collection add <path>       Add directory as collection
    --name <name>                  Collection name (default: dir name)
    --description <text>           Human-readable description
    --extensions <ext,...>          Comma-separated list of extensions to index
    --no-index                     Register without triggering indexing
  kbolt collection remove <name>    Remove collection and all its data
  kbolt collection rename <old> <new>
  kbolt collection info <name>      Show collection details and stats
  kbolt collection list             List all collections in active space (or all spaces)
  kbolt collection describe <name> <text>  Update collection description

IGNORE PATTERNS
  kbolt ignore show <collection>    Show ignore patterns for a collection
  kbolt ignore edit <collection>    Open ignore file in $VISUAL / $EDITOR / vi
  kbolt ignore add <collection> <pattern>   Add a pattern
  kbolt ignore remove <collection> <pattern>  Remove a pattern
  kbolt ignore list                 List all collections that have ignore patterns

INDEXING
  kbolt update                      Scan, extract, chunk, FTS index, and embed
    --no-embed                     Skip embedding (FTS-only indexing)
    --collection <name,...>        Scope to one or more collections (comma-separated)
    --dry-run                      Preview what would change without writing
    --verbose                      Log per-file decisions

MODELS
  kbolt models pull                 Download required models
  kbolt models list                 Show model status

SCHEDULING
  kbolt schedule --every <interval> Set up recurring re-index (e.g. 6h, 30m)
    --no-embed                     FTS-only re-index (faster)
    --at <time>                    Daily at specific time (e.g. 03:00)
  kbolt schedule --off              Remove the schedule
  kbolt schedule --status           Show current schedule

ADMIN
  kbolt status                      Index health, space/collection stats, disk usage
```

Scheduling uses OS-level mechanisms: launchd plist on macOS, cron/systemd timer on Linux. `kbolt schedule` creates the appropriate config for the platform.

Space resolution for commands that need a space context (collection add/remove/rename, search with `--collection`, ignore, ls): `--space` flag > `KBOLT_SPACE` env var > configured default > unique lookup > error. Commands that operate on all data by default (update without `--collection`, search without `--collection`, status) scan all spaces unless `--space` is provided.

---

## MCP Server

Stdio transport (`kbolt mcp` command). Streamable HTTP transport deferred to V2 (with `kbolt serve`).

### Tools (5)

1. **search** — Smart hybrid search with optional `mode` param (deep/keyword/semantic), optional `space` and `collection` filters
2. **get** — Single document by path or docid
3. **multi_get** — Batch retrieve by glob or path list
4. **list_files** — List files in a collection with optional prefix filter, optional `space` filter
5. **status** — Index health, space/collection info, document counts

All tools accept an optional `space` parameter. If omitted, search and status operate across all spaces. Collection-scoped tools use the same resolution logic as the CLI.

### Resources

Documents accessible via `kbolt://{space}/{collection}/{path}` URIs. MCP clients can read documents directly via resource URIs.

### Dynamic Instructions

Injected into LLM system prompt on connection:
- Number of indexed documents, spaces, and collections
- Available spaces with descriptions and their collections
- Search strategy guidance (when to use --deep vs default)

---

## Models (V1 — Off-the-shelf)

| Role | Model | Format | Size | Runtime |
|---|---|---|---|---|
| Embedding | EmbeddingGemma (256d) | ONNX | ~600MB | ONNX Runtime (CPU) |
| Reranker | Qwen3-Reranker 0.6B | GGUF Q8 | ~700MB | llama-cpp-rs (GPU) |
| Expander | Qwen3 1.7B | GGUF Q4 | ~1.2GB | llama-cpp-rs (GPU) |

Download: HuggingFace Hub via `hf-hub` crate. Handles resumable downloads, SHA-256 checksum verification, local caching. Auto-download on first use (`kbolt models pull` for explicit pre-download).

Models loaded lazily — embedder loads on first `kbolt update` (with embedding), reranker loads on first search that triggers reranking, expander loads on first `--deep` search. Unloaded after configurable inactivity timeout (default: 5 minutes).

---

## Evaluation Framework

```
kbolt eval run                    Run evaluation suite
kbolt eval add <query> <expected> Add test case
kbolt eval report                 Show metrics
```

Metrics: MRR@10, Recall@K (K=1,5,10), latency (p50/p95/p99).

Dataset in `~/.config/kbolt/eval.toml`:
```toml
[[queries]]
query = "how to handle errors in rust"
expected = ["notes/rust/error-handling.md"]
space = "personal"
collection = "notes"
```

---

## Implementation Order

1. **types/** — shared request/response structs (including Space types)
2. **core/config** — TOML loading/saving (models, reaping, default space)
3. **core/storage** — Storage struct, SQLite schema (spaces, collections, documents, chunks, embeddings, cache), per-space Tantivy/USearch indexes
4. **core/ingest/ignore** — hardcoded ignores, ignore pattern parser (internal storage), extension filtering
5. **core/ingest/extract** — Extractor trait + Markdown, plaintext, code extractors
6. **core/ingest/chunk** — chunking pipeline (token budget, overlap, merging)
7. **core/ingest** — `update` flow (scan → filter → extract → chunk → store → FTS index)
8. **core/models** — model registry, HuggingFace download, Embedder/Reranker/Generator traits + impls
9. **core/ingest/embed** — embedding within update flow (chunks → embed → USearch insert)
10. **core/search** — query parsing, routing, BM25, dense, RRF fusion, reranking, result assembly, cross-space search
11. **core/search (deep)** — query expansion + multi-variant retrieval
12. **core/engine** — Engine struct wiring everything together (space resolution logic)
13. **cli/** — all commands wired to Engine (space, collection, ignore, search, update, admin)
14. **mcp/** — MCP stdio server (space-aware tools)
15. **eval** — evaluation framework
16. **distribution** — CI/CD, release binaries, Homebrew, install script

---

## Verification Plan

**Unit tests**: Each extractor (chunk boundaries, heading breadcrumbs, edge cases), chunking (token budget, overlap, break points), storage CRUD (all tables including spaces, collections), ignore rules (hardcoded + internal ignore patterns + extensions), RRF scoring, query parsing, space resolution logic.

**Integration tests**: Full ingest pipeline (file on disk → filter → extract → chunk → store → search → result), hybrid search fusion, deep search expansion, cross-space search, space management (add/remove/rename with CASCADE to collections), collection management (add/remove/rename with CASCADE), soft delete + reaping lifecycle, snippet extraction from disk, ignore pattern management.

**End-to-end tests**: Index ~100 sample files across multiple spaces, run search queries (within space, across spaces), verify recall against known-relevant docs, MRR@10 > 0.7, all CLI commands produce expected output, MCP tools respond correctly via stdio.

**Performance targets**: < 500MB RSS, < 100ms keyword search, < 3s deep search.

---

## V2 Roadmap (deferred)

- **Daemon** (`kbolt serve`) — long-running process with HTTP API + MCP-over-HTTP. Warm models shared across CLI and multiple MCP clients. Background indexing. CLI auto-connects to daemon when running.
- SPLADE sparse vectors (4th retrieval signal)
- Model training and fine-tuning pipeline
- Context-aware retrieval (description embedded into vectors)
- Vision embedder for images
- Document graph (link-aware retrieval)
- Relevance feedback loop
- Per-user model adaptation (LoRA)
