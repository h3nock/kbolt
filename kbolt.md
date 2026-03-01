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

### Cross-Process Locking

"Single process" means no daemon — it does not mean only one kbolt instance can exist. Multiple CLI commands, scheduled jobs, and MCP sessions can run concurrently. To prevent corruption of Tantivy and USearch indexes, kbolt uses a global operation-scoped file lock at `~/.cache/kbolt/kbolt.lock`.

**Lock semantics**:
- **Write operations** (`update`, `collection add/remove`, `space remove`) acquire an **exclusive** flock for the duration of the operation, not the Engine lifetime. This is critical because `kbolt mcp` is long-lived — an Engine-scoped lock would block all other CLI commands for the entire MCP session.
- **Read operations** (`search`, `get`, `multi-get`, `ls`, `status`) acquire a **shared** flock for the duration of the operation. Multiple readers can proceed concurrently, but all readers block while a writer holds the exclusive lock.
- If the lock is unavailable, the command **fails fast** with: "Another kbolt process is active. Try again shortly." No queuing, no retry. (The message is intentionally generic — an exclusive lock request can fail because another writer holds it, or because readers hold shared locks.)
- The OS releases flocks automatically on process death (including SIGKILL), so stale locks cannot occur.

Default search uses both BM25 and dense retrieval (USearch), so all search modes require the shared lock. A keyword-only carveout (Tantivy and SQLite both support concurrent readers during writes) is a possible V2 optimization but not worth the complexity for V1.

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
| `models/` | Model artifact provider (default: HuggingFace Hub via `hf-hub`) | download | Fetch model files on first use |
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
- `search/` receives `&Storage` → calls: `query_bm25()`, `query_dense()`, `get_chunks()`, `get_document_by_hash_prefix()`
- `search/` receives `&Models` → calls: `embed()` (query vector), `rerank()` (cross-encoder)
- `ingest/` receives `&Storage` → calls: `upsert_document()`, `delete_chunks_for_document()`, `insert_chunks()`, `index_tantivy()`, `insert_usearch()`, `insert_embeddings()`, `deactivate_document()`, `reap_documents()`
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
    pub fn new(config_path: Option<&Path>) -> Result<Self>;

    // Search — delegates to search module
    pub fn search(&self, req: SearchRequest) -> Result<SearchResponse>;

    // Indexing — delegates to ingest module
    pub fn update(&self, opts: UpdateOptions) -> Result<UpdateReport>;

    // Documents — resolves target, reads live file, computes stale flag
    pub fn get_document(&self, req: GetRequest) -> Result<DocumentResponse>;
    pub fn multi_get(&self, req: MultiGetRequest) -> Result<MultiGetResponse>;
    pub fn list_files(&self, space: Option<&str>, collection: &str,
                       prefix: Option<&str>) -> Result<Vec<FileEntry>>;

    // Spaces — thin wrappers over storage
    pub fn add_space(&self, name: &str, description: Option<&str>) -> Result<SpaceInfo>;
    pub fn remove_space(&self, name: &str) -> Result<()>;
    pub fn rename_space(&self, old: &str, new: &str) -> Result<()>;
    pub fn describe_space(&self, name: &str, description: &str) -> Result<()>;
    pub fn list_spaces(&self) -> Result<Vec<SpaceInfo>>;
    pub fn space_info(&self, name: &str) -> Result<SpaceInfo>;
    pub fn set_default_space(&self, name: Option<&str>) -> Result<Option<String>>;
    pub fn resolve_space(&self, explicit: Option<&str>) -> Result<String>;

    // Collections — resolves space, delegates to storage, optionally triggers update
    pub fn add_collection(&self, req: AddCollectionRequest) -> Result<CollectionInfo>;
    pub fn remove_collection(&self, space: Option<&str>, name: &str) -> Result<()>;
    pub fn rename_collection(&self, space: Option<&str>, old: &str, new: &str) -> Result<()>;
    pub fn describe_collection(&self, space: Option<&str>, name: &str, desc: &str) -> Result<()>;
    pub fn list_collections(&self, space: Option<&str>) -> Result<Vec<CollectionInfo>>;
    pub fn collection_info(&self, space: Option<&str>, name: &str) -> Result<CollectionInfo>;

    // Models
    pub fn pull_models(&self) -> Result<PullReport>;
    pub fn model_status(&self) -> Result<ModelStatus>;

    // Admin
    pub fn status(&self, space: Option<&str>) -> Result<StatusResponse>;
}
```

Engine is the only type adapters interact with. Every public method maps directly to a CLI command or MCP tool. Space resolution (flag > env > default > unique lookup) happens inside Engine methods that accept `space: Option<&str>` — adapters pass through whatever the user provided, Engine resolves it.

Adapters (CLI, MCP) take `&Engine` directly. They parse input into request types, call Engine methods, format the response.

### Storage (three-store model, owns concurrency)

Storage is the data layer. SQLite is global (one database for all spaces). Tantivy and USearch are per-space — each space gets its own index for BM25 IDF isolation. All connected by `chunk_id` as the universal join key.

```rust
pub struct Storage {
    db: Mutex<rusqlite::Connection>,                     // global, single writer, WAL
    spaces: RwLock<HashMap<String, Arc<SpaceIndexes>>>,  // per-space search indexes
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

**Space index lifecycle**: All known spaces are eagerly opened at `Storage::new()` — the spaces table is scanned and each space's Tantivy index and USearch file are loaded. The `RwLock<HashMap<String, Arc<SpaceIndexes>>>` allows `open_space` and `close_space` to take `&self` (write lock on the map). Read operations (search, get) take a read lock on the map, clone the `Arc<SpaceIndexes>`, then drop the map lock before touching the indexes. This means space add/remove only blocks reads for the brief duration of a HashMap lookup, not for entire search operations.

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

## Component Interfaces

This section defines the contracts between components — the types, method signatures, and error types that form the API boundaries. Types that cross crate boundaries live in `types/`. Types internal to `core/` live alongside the modules that define them.

### Error Types

Defined in `types/`. A single error enum for the entire system. Each variant carries enough context for the caller (CLI, MCP) to produce a useful message without inspecting inner errors.

```rust
#[derive(Debug, thiserror::Error)]
pub enum KboltError {
    // Storage
    #[error("database error: {0}")]
    Database(String),

    #[error("tantivy error: {0}")]
    Tantivy(String),

    #[error("usearch error: {0}")]
    USearch(String),

    // Entity resolution
    #[error("space not found: {name}")]
    SpaceNotFound { name: String },

    #[error("collection not found: {name}")]
    CollectionNotFound { name: String },

    #[error("document not found: {path}")]
    DocumentNotFound { path: String },

    #[error("ambiguous space: collection '{collection}' exists in spaces: {spaces:?}")]
    AmbiguousSpace { collection: String, spaces: Vec<String> },

    #[error("space already exists: {name}")]
    SpaceAlreadyExists { name: String },

    #[error("collection already exists: {name} (in space {space})")]
    CollectionAlreadyExists { name: String, space: String },

    #[error("no active space: use --space, set KBOLT_SPACE, or configure a default")]
    NoActiveSpace,

    // Filesystem
    #[error("file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("file deleted since indexing: {0}. Run `kbolt update` to refresh.")]
    FileDeleted(PathBuf),

    // Models
    #[error("model not available: {name}. Run `kbolt models pull` to download.")]
    ModelNotAvailable { name: String },

    #[error("model download failed: {0}")]
    ModelDownload(String),

    #[error("inference error: {0}")]
    Inference(String),

    // Config
    #[error("config error: {0}")]
    Config(String),

    #[error("invalid path: {0}")]
    InvalidPath(PathBuf),

    #[error("internal error: {0}")]
    Internal(String),

    // I/O
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, KboltError>;
```

`KboltError` is the public error contract used by adapters and boundary types. Core modules use an internal error type (`CoreError`) with `#[from]` conversions for infrastructure errors (SQLite, TOML, JSON, etc.), then map to `KboltError` at the core/adapter boundary. Adapters translate `KboltError` into their output format: CLI prints the error message to stderr with exit code 1, MCP returns it as a tool error response. No adapter catches and re-wraps — they use the `Display` impl from `thiserror`.

### Types Crate (`types/`)

All types that cross crate boundaries — used by `core/` and adapters (`cli/`, `mcp/`). Pure data structs, no logic, no `impl` blocks with behavior. All types derive `Debug, Clone, Serialize, Deserialize`.

#### Search Types

```rust
pub struct SearchRequest {
    pub query: String,
    pub mode: SearchMode,
    pub space: Option<String>,             // scope to space (None = all spaces)
    pub collections: Vec<String>,          // scope to collections (empty = all in space)
    pub limit: usize,                      // default: 10
    pub min_score: f32,                    // default: 0.0
    pub no_rerank: bool,                   // skip cross-encoder reranker
    pub debug: bool,                       // populate signals in results
}

pub enum SearchMode {
    Auto,       // route by query characteristics (default)
    Deep,       // query expansion + multi-variant + full rerank
    Keyword,    // BM25 only
    Semantic,   // dense vector only
}

pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub query: String,                     // echo back the query
    pub mode: SearchMode,                  // actual mode used
    pub staleness_hint: Option<String>,    // e.g. "Index last updated: 3h ago"
    pub elapsed_ms: u64,
}

pub struct SearchResult {
    pub docid: String,                     // "#a1b2c3"
    pub path: String,                      // "collection/relative/path.md"
    pub title: String,
    pub space: String,
    pub collection: String,
    pub heading: Option<String>,           // chunk heading breadcrumb within the document
    pub text: String,                      // chunk content (read from disk at offset/length)
    pub score: f32,                        // final relevance score [0, 1]
    pub signals: Option<SearchSignals>,    // per-signal breakdown (populated when debug requested)
}

pub struct SearchSignals {
    pub bm25: Option<f32>,                 // normalized BM25 score
    pub dense: Option<f32>,                // cosine similarity
    pub rrf: f32,                          // RRF fusion score
    pub reranker: Option<f32>,             // cross-encoder score
}
```

#### Document Types

```rust
pub enum Locator {
    Path(String),                          // "collection/path.md" or just "path.md"
    DocId(String),                         // "a1b2c3" (without the # prefix)
}

pub struct GetRequest {
    pub locator: Locator,
    pub space: Option<String>,
    pub offset: Option<usize>,             // start at line N
    pub limit: Option<usize>,              // max lines to return
}

pub struct DocumentResponse {
    pub docid: String,
    pub path: String,
    pub title: String,
    pub space: String,
    pub collection: String,
    pub content: String,                   // live file content (read from disk)
    pub stale: bool,                       // true if current file hash != indexed hash
    pub total_lines: usize,
    pub returned_lines: usize,             // lines returned (after offset/limit)
}

pub struct MultiGetRequest {
    pub locators: Vec<Locator>,            // paths and/or docids, resolved in order given
    pub space: Option<String>,
    pub max_files: usize,                  // default: 20
    pub max_bytes: usize,                  // default: 51_200 (50KB)
}

pub struct MultiGetResponse {
    pub documents: Vec<DocumentResponse>,  // files returned (whole, never truncated mid-file)
    pub omitted: Vec<OmittedFile>,         // files that matched but didn't fit within budget
    pub resolved_count: usize,             // total files matched before budget applied
}

pub struct OmittedFile {
    pub path: String,
    pub docid: String,
    pub size_bytes: usize,                 // so the caller can decide whether to fetch individually via get
    pub reason: OmitReason,
}

pub enum OmitReason {
    MaxFiles,
    MaxBytes,
}

pub struct FileEntry {
    pub path: String,                      // relative to collection root
    pub title: String,
    pub docid: String,
    pub active: bool,
    pub chunk_count: usize,
    pub embedded: bool,                    // has at least one embedding
}
```

#### Space & Collection Types

```rust
pub struct SpaceInfo {
    pub name: String,
    pub description: Option<String>,
    pub collection_count: usize,
    pub document_count: usize,
    pub chunk_count: usize,
    pub created: String,                   // ISO 8601
}

pub struct AddCollectionRequest {
    pub path: PathBuf,                     // absolute directory path
    pub space: Option<String>,             // resolved via space resolution
    pub name: Option<String>,              // default: directory name
    pub description: Option<String>,
    pub extensions: Option<Vec<String>>,   // e.g. ["rs", "py"]
    pub no_index: bool,                    // register without triggering initial indexing
}

pub struct CollectionInfo {
    pub name: String,
    pub space: String,
    pub path: PathBuf,
    pub description: Option<String>,
    pub extensions: Option<Vec<String>>,
    pub document_count: usize,
    pub active_document_count: usize,
    pub chunk_count: usize,
    pub embedded_chunk_count: usize,
    pub created: String,
    pub updated: String,
}
```

#### Indexing Types

```rust
pub struct UpdateOptions {
    pub space: Option<String>,             // scope to space (None = all spaces)
    pub collections: Vec<String>,          // scope to collections (empty = all)
    pub no_embed: bool,                    // skip embedding (FTS-only indexing)
    pub dry_run: bool,                     // preview what would change, no writes
    pub verbose: bool,                     // log per-file decisions to stderr
}

pub struct UpdateReport {
    pub scanned: usize,                    // files examined (stat'd)
    pub skipped_mtime: usize,              // unchanged mtime → fast skip
    pub skipped_hash: usize,               // mtime changed but hash unchanged
    pub added: usize,                      // new files indexed
    pub updated: usize,                    // changed files re-indexed
    pub deactivated: usize,                // files disappeared from disk
    pub reactivated: usize,                // previously deactivated files returned
    pub reaped: usize,                     // hard-deleted past reaping period
    pub embedded: usize,                   // chunks that received embeddings
    pub errors: Vec<FileError>,            // per-file errors (non-fatal, indexing continues)
    pub elapsed_ms: u64,
}

pub struct FileError {
    pub path: String,
    pub error: String,
}
```

#### Status Types

```rust
pub struct StatusResponse {
    pub spaces: Vec<SpaceStatus>,
    pub models: ModelStatus,
    pub cache_dir: PathBuf,
    pub config_dir: PathBuf,
    pub total_documents: usize,
    pub total_chunks: usize,
    pub total_embedded: usize,
    pub disk_usage: DiskUsage,
}

pub struct SpaceStatus {
    pub name: String,
    pub description: Option<String>,
    pub collections: Vec<CollectionStatus>,
    pub last_updated: Option<String>,      // most recent update across collections
}

pub struct CollectionStatus {
    pub name: String,
    pub path: PathBuf,
    pub documents: usize,
    pub active_documents: usize,
    pub chunks: usize,
    pub embedded_chunks: usize,
    pub last_updated: String,
}

pub struct ModelStatus {
    pub embedder: ModelInfo,
    pub reranker: ModelInfo,
    pub expander: ModelInfo,
}

pub struct ModelInfo {
    pub name: String,
    pub downloaded: bool,
    pub size_bytes: Option<u64>,
    pub path: Option<PathBuf>,
}

pub struct DiskUsage {
    pub sqlite_bytes: u64,
    pub tantivy_bytes: u64,
    pub usearch_bytes: u64,
    pub models_bytes: u64,
    pub total_bytes: u64,
}
```

### Storage API

The Storage struct's public methods — the internal contract that `search/`, `ingest/`, and `engine` depend on. All methods take `&self` — Storage handles its own locking internally via `Mutex<Connection>`, `Mutex<IndexWriter>`, and `RwLock<usearch::Index>`.

```rust
impl Storage {
    // --- Construction ---
    pub fn new(cache_dir: &Path) -> Result<Self>;

    // --- Space index lifecycle ---
    pub fn open_space(&self, name: &str) -> Result<()>;    // load/create Tantivy + USearch for space (write-locks spaces map)
    pub fn close_space(&self, name: &str) -> Result<()>;   // unload space indexes from memory (write-locks spaces map)

    // --- Space CRUD (SQLite) ---
    pub fn create_space(&self, name: &str, description: Option<&str>) -> Result<i64>;
    pub fn delete_space(&self, name: &str) -> Result<()>;      // CASCADE to collections
    pub fn rename_space(&self, old: &str, new: &str) -> Result<()>;
    pub fn get_space(&self, name: &str) -> Result<SpaceRow>;
    pub fn list_spaces(&self) -> Result<Vec<SpaceRow>>;
    pub fn update_space_description(&self, name: &str, description: &str) -> Result<()>;
    pub fn find_space_for_collection(&self, collection: &str) -> Result<SpaceResolution>;

    // --- Collection CRUD (SQLite) ---
    pub fn create_collection(&self, space_id: i64, name: &str, path: &Path,
                              description: Option<&str>, extensions: Option<&[String]>) -> Result<i64>;
    pub fn delete_collection(&self, space_id: i64, name: &str) -> Result<()>;
    pub fn rename_collection(&self, space_id: i64, old: &str, new: &str) -> Result<()>;
    pub fn get_collection(&self, space_id: i64, name: &str) -> Result<CollectionRow>;
    pub fn list_collections(&self, space_id: Option<i64>) -> Result<Vec<CollectionRow>>;
    pub fn update_collection_description(&self, space_id: i64, name: &str, desc: &str) -> Result<()>;
    pub fn update_collection_timestamp(&self, collection_id: i64) -> Result<()>;

    // --- Document CRUD (SQLite) ---
    pub fn upsert_document(&self, collection_id: i64, path: &str, title: &str,
                            hash: &str, modified: &str) -> Result<i64>;
    pub fn get_document_by_path(&self, collection_id: i64, path: &str) -> Result<Option<DocumentRow>>;
    pub fn get_document_by_hash_prefix(&self, prefix: &str) -> Result<Vec<DocumentRow>>;
    pub fn list_documents(&self, collection_id: i64, active_only: bool) -> Result<Vec<DocumentRow>>;
    pub fn deactivate_document(&self, doc_id: i64) -> Result<()>;
    pub fn reactivate_document(&self, doc_id: i64) -> Result<()>;
    pub fn reap_documents(&self, older_than_days: u32) -> Result<Vec<i64>>;
    pub fn get_fts_dirty_documents(&self) -> Result<Vec<FtsDirtyRecord>>;  // documents with fts_dirty = 1, joined to collection for path resolution
    pub fn batch_clear_fts_dirty(&self, doc_ids: &[i64]) -> Result<()>;   // set fts_dirty = 0 for a batch of documents

    // --- Chunk operations (SQLite) ---
    pub fn insert_chunks(&self, doc_id: i64, chunks: &[ChunkInsert]) -> Result<Vec<i64>>;
    pub fn delete_chunks_for_document(&self, doc_id: i64) -> Result<Vec<i64>>;
    pub fn get_chunks(&self, chunk_ids: &[i64]) -> Result<Vec<ChunkRow>>;
    pub fn get_chunks_for_document(&self, doc_id: i64) -> Result<Vec<ChunkRow>>;

    // --- Embedding tracking (SQLite) ---
    pub fn insert_embeddings(&self, entries: &[(i64, &str)]) -> Result<()>;
    pub fn get_unembedded_chunks(&self, model: &str, limit: usize) -> Result<Vec<EmbedRecord>>;  // chunks needing embedding, active documents only, with context for disk reads
    pub fn delete_embeddings_for_model(&self, model: &str) -> Result<usize>;
    pub fn count_embeddings(&self) -> Result<usize>;

    // --- Tantivy operations (per-space) ---
    pub fn index_tantivy(&self, space: &str, entries: &[TantivyEntry]) -> Result<()>;
    pub fn delete_tantivy(&self, space: &str, chunk_ids: &[i64]) -> Result<()>;
    pub fn delete_tantivy_by_doc(&self, space: &str, doc_id: i64) -> Result<()>;  // remove all entries for a document
    pub fn query_bm25(&self, space: &str, query: &str,
                       fields: &[(&str, f32)], limit: usize) -> Result<Vec<BM25Hit>>;
    pub fn commit_tantivy(&self, space: &str) -> Result<()>;

    // --- USearch operations (per-space) ---
    pub fn insert_usearch(&self, space: &str, key: i64, vector: &[f32]) -> Result<()>;
    pub fn batch_insert_usearch(&self, space: &str, entries: &[(i64, &[f32])]) -> Result<()>;
    pub fn delete_usearch(&self, space: &str, keys: &[i64]) -> Result<()>;
    pub fn query_dense(&self, space: &str, vector: &[f32], limit: usize) -> Result<Vec<DenseHit>>;
    pub fn count_usearch(&self, space: &str) -> Result<usize>;
    pub fn clear_usearch(&self, space: &str) -> Result<()>;

    // --- LLM Cache ---
    pub fn cache_get(&self, key: &str) -> Result<Option<String>>;
    pub fn cache_set(&self, key: &str, value: &str) -> Result<()>;

    // --- Aggregate queries (for status) ---
    pub fn count_documents(&self, space_id: Option<i64>) -> Result<usize>;
    pub fn count_chunks(&self, space_id: Option<i64>) -> Result<usize>;
    pub fn count_embedded_chunks(&self, space_id: Option<i64>) -> Result<usize>;
    pub fn disk_usage(&self) -> Result<DiskUsage>;
}
```

Internal row types (defined in `core/storage/`, not in `types/` — these are storage implementation details):

```rust
pub struct SpaceRow {
    pub id: i64, pub name: String, pub description: Option<String>, pub created: String,
}
pub struct CollectionRow {
    pub id: i64, pub space_id: i64, pub name: String, pub path: PathBuf,
    pub description: Option<String>, pub extensions: Option<Vec<String>>,
    pub created: String, pub updated: String,
}
pub struct DocumentRow {
    pub id: i64, pub collection_id: i64, pub path: String, pub title: String,
    pub hash: String, pub modified: String, pub active: bool,
    pub deactivated_at: Option<String>, pub fts_dirty: bool,
}
pub struct ChunkRow {
    pub id: i64, pub doc_id: i64, pub seq: i32, pub offset: usize,
    pub length: usize, pub heading: Option<String>, pub kind: ChunkKind,
}
pub struct ChunkInsert {
    pub seq: i32, pub offset: usize, pub length: usize,
    pub heading: Option<String>, pub kind: ChunkKind,
}
pub struct TantivyEntry {
    pub chunk_id: i64, pub doc_id: i64, pub filepath: String, pub title: String,
    pub heading: Option<String>, pub body: String,
}
pub struct BM25Hit { pub chunk_id: i64, pub score: f32 }
pub struct DenseHit { pub chunk_id: i64, pub distance: f32 }

// Returned by get_fts_dirty_documents — everything needed to replay Tantivy writes from disk
pub struct FtsDirtyRecord {
    pub doc_id: i64,
    pub doc_path: String,                  // relative path within collection
    pub doc_title: String,
    pub doc_hash: String,                  // stored hash, for verifying file hasn't changed
    pub collection_path: PathBuf,          // absolute path to collection root
    pub space_name: String,
    pub chunks: Vec<ChunkRow>,             // all chunks for this document
}

// Returned by get_unembedded_chunks — everything needed to embed from disk
pub struct EmbedRecord {
    pub chunk_id: i64,
    pub doc_path: String,                  // relative path within collection
    pub collection_path: PathBuf,          // absolute path to collection root
    pub space_name: String,
    pub offset: usize,                     // byte offset in source file
    pub length: usize,                     // byte length in source file
}

pub enum SpaceResolution {
    Found(SpaceRow),
    Ambiguous(Vec<String>),                // collection exists in multiple spaces
    NotFound,
}
```

### Search Module API

Two public functions — the only entry points from Engine into the search module:

```rust
// search/mod.rs
pub fn execute(
    storage: &Storage,
    models: &Models,
    req: SearchRequest,
) -> Result<SearchResponse>;

pub fn deep(
    storage: &Storage,
    models: &Models,
    req: SearchRequest,
) -> Result<SearchResponse>;
```

`execute` handles Auto, Keyword, and Semantic modes. `deep` handles Deep mode (query expansion + multi-variant retrieval + full rerank). Engine dispatches based on `req.mode`.

Internal types (private to `search/`, not in `types/`):

```rust
// query.rs — query parsing
struct ParsedQuery {
    terms: Vec<String>,                    // plain search terms
    phrases: Vec<String>,                  // quoted exact-match phrases
    file_filters: Vec<String>,             // file:*.rs patterns
    collection_filter: Option<String>,     // collection:notes
    negations: Vec<String>,                // -excluded terms
}

// routing.rs — signal activation
struct RouteDecision {
    use_bm25: bool,
    use_dense: bool,
    use_reranker: bool,
}

// fusion.rs — RRF fusion
struct FusionConfig {
    k: u32,                                // default: 60
    w_bm25: f32,                           // default: 1.0
    w_dense: f32,                          // default: 1.0
    agreement_bonus: f32,                  // default: 1.2
}

struct FusedCandidate {
    chunk_id: i64,
    doc_id: i64,                           // for deduplication
    rrf_score: f32,
    bm25_score: Option<f32>,
    dense_score: Option<f32>,
}

// rerank.rs — cross-encoder reranking
struct RankedCandidate {
    chunk_id: i64,
    final_score: f32,                      // 0.7 * reranker + 0.3 * rrf
    reranker_score: Option<f32>,
    rrf_score: f32,
}
```

### Ingest Module API

One public function — the only entry point from Engine into the ingest module:

```rust
// ingest/mod.rs
pub fn update(
    storage: &Storage,
    models: &Models,
    config: &Config,
    opts: UpdateOptions,
) -> Result<UpdateReport>;
```

`update` runs the full pipeline: scan → filter → mtime check → hash check → extract → chunk → store → FTS index → embed. Returns an `UpdateReport` summarizing what happened. Non-fatal per-file errors are collected in `report.errors` — the pipeline continues processing remaining files.

Internal types (private to `ingest/`):

```rust
// chunk.rs — finalized chunks ready for storage
struct FinalChunk {
    text: String,
    offset: usize,
    length: usize,
    heading: Option<String>,
    kind: ChunkKind,
}

// ignore.rs — compiled ignore rules for a collection
struct IgnoreRules {
    hardcoded: Vec<GlobPattern>,           // .git/, node_modules/, etc.
    user_patterns: Vec<GlobPattern>,       // from .ignore file
    extensions: Option<Vec<String>>,       // from collection config
}
```

### Config API

```rust
// config/mod.rs

pub struct Config {
    pub config_dir: PathBuf,               // ~/.config/kbolt/
    pub cache_dir: PathBuf,                // ~/.cache/kbolt/
    pub default_space: Option<String>,
    pub models: ModelConfig,
    pub reaping: ReapingConfig,
}

pub struct ModelConfig {
    pub embedder: ModelSourceConfig,
    pub reranker: ModelSourceConfig,
    pub expander: ModelSourceConfig,
}

pub struct ModelSourceConfig {
    pub provider: ModelProvider,           // "huggingface" in V1
    pub id: String,                        // provider model identifier
    pub revision: Option<String>,          // optional pinned revision/tag
}

pub enum ModelProvider {
    HuggingFace,
}

pub struct ReapingConfig {
    pub days: u32,                         // default: 7
}

pub fn load(config_path: Option<&Path>) -> Result<Config>;
pub fn save(config: &Config) -> Result<()>;
```

`load` reads `~/.config/kbolt/index.toml` (or creates it with defaults if missing). `save` writes back to disk. Config is loaded once at Engine construction time and held as a struct field. Adapter commands that modify config (like `kbolt space default`) call `save` after mutation.

### Models API

Trait definitions already specified in the Architecture section (Embedder, Reranker, Generator, Extractor). The Models facade provides the public API that Engine and domain modules call:

```rust
// models/mod.rs

impl Models {
    pub fn new(config: ModelConfig, model_dir: &Path) -> Result<Self>;

    // Inference — lazy-loads the backing model on first call
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    pub fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>>;
    pub fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String>;

    // Metadata
    pub fn dimensions(&self) -> usize;          // embedding dimensions (e.g. 256)
    pub fn embedder_name(&self) -> &str;        // current embedder model identifier

    // Lifecycle
    pub fn pull(&self) -> Result<PullReport>;   // download all configured models
    pub fn status(&self) -> Result<ModelStatus>; // which models are downloaded
}

pub struct PullReport {
    pub downloaded: Vec<String>,           // model names freshly downloaded
    pub already_present: Vec<String>,      // model names already on disk
    pub total_bytes: u64,                  // bytes downloaded this run
}
```

`embed()`, `rerank()`, and `generate()` trigger lazy model loading on first invocation. The loaded model stays in memory for the process lifetime. Thread safety: embedder (ONNX) is `Arc<dyn Embedder>` (thread-safe, concurrent calls OK). Reranker and generator (llama-cpp) are `Mutex<dyn Reranker>` / `Mutex<dyn Generator>` (serialized access — llama-cpp is not thread-safe).

Model download is delegated through a provider abstraction (for example, HuggingFace, local filesystem mirrors, or cloud object storage). The core model module must not hardcode provider-specific assumptions in orchestration logic.

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
| `fts_dirty` | INTEGER | 1 = chunks written to SQLite but Tantivy not yet confirmed, 0 = FTS in sync |

UNIQUE constraint on `(collection_id, path)`. Soft delete with auto-reaping: documents deactivated for longer than the configured reaping period (default: 7 days) are hard-deleted during `kbolt update`. Deactivated documents (`active = 0`) are excluded from the embedding backlog (`get_unembedded_chunks` filters `active = 1`) and from search results (result assembly filters out chunks belonging to inactive documents). Chunks and embeddings are retained during deactivation so that if the file reappears (same hash), re-activation skips re-indexing.

The `fts_dirty` flag is a BM25/FTS reconciliation mechanism. It is set to 1 in the same SQLite transaction that writes chunk mutations. It is cleared to 0 only via `batch_clear_fts_dirty()`, which runs after each Tantivy commit point (every N documents and at end of phase). This batched clearing ensures the flag is never cleared before the Tantivy entries are durably committed. On each `kbolt update`, Phase 0 replays any documents with `fts_dirty = 1` through Tantivy indexing before the normal mtime/hash fast path runs. This closes a crash-safety gap: without it, if the process dies after SQLite commits chunks but before Tantivy commits, the next update would skip the file (unchanged hash) and the BM25 entry would stay missing forever. Dense/embedding recovery is handled separately by the existing `embeddings` ledger + USearch sync check.

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
    fts_dirty       INTEGER NOT NULL DEFAULT 0,
    UNIQUE(collection_id, path)
);
CREATE INDEX idx_documents_collection ON documents(collection_id, active);
CREATE INDEX idx_documents_hash ON documents(hash);
CREATE INDEX idx_documents_fts_dirty ON documents(fts_dirty) WHERE fts_dirty = 1;

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
doc_id:    u64  (stored, fast)          — join key back to SQLite documents table, used for delete-by-document
filepath:  TEXT (stored, indexed, 2x)   — "collection_name/relative/path.md"
title:     TEXT (stored, indexed, 3x)   — document title (copied from parent Document at index time)
heading:   TEXT (stored, indexed, 2x)   — chunk heading breadcrumb
body:      TEXT (indexed, NOT stored)   — chunk text (indexed for BM25, read from disk at query time)
```

The `doc_id` field enables delete-by-document: when a document is re-indexed or replayed after a crash, all of its old Tantivy entries can be removed by `doc_id` before adding new ones. This is necessary because old chunk IDs are not preserved across re-indexing (chunks are deleted and re-created with new IDs), so there is no other way to remove stale entries from a previous version.

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
    models/                 # model artifacts (provider-managed cache)
        embedder/           # embedder role artifacts
        reranker/           # reranker role artifacts
        expander/           # query-expander role artifacts
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

### Concurrency Model

Document-level ingestion is **single-threaded** in V1. Files are processed one at a time through scan → extract → chunk → SQLite → Tantivy. Embedding runs as a separate second pass after all FTS indexing is complete. The pass queries `get_unembedded_chunks()` in bounded batches (default: 64 chunks), which returns chunk metadata plus the file paths needed to read chunk text from disk. This two-phase design is simple, testable, and matches the `--no-embed` flag naturally (just skip phase 2). The extraction/chunking phase is fast (~1ms per file for text processing) — embedding is the bottleneck, and batching it efficiently matters more than parallelizing extraction.

Note: the `Mutex`/`RwLock` wrappers on Storage fields are justified by search parallelism (BM25 and dense retrieval run in parallel threads during search), not by ingest.

### Per-Document Flow

```
Phase 0: FTS Reconciliation (replay dirty documents from a previous crash)
  Query get_fts_dirty_documents() → Vec<FtsDirtyRecord>
  For each dirty document:
  │  Resolve file path: collection_path / doc_path
  │  Read file from disk, compute SHA-256 hash
  │  │
  │  ├── File exists AND hash matches stored doc_hash:
  │  │   delete_tantivy_by_doc(space, doc_id)  ← remove any stale old entries first
  │  │   For each chunk in record.chunks:
  │  │   │  Read chunk text from file at offset/length
  │  │   │  Add to Tantivy (chunk_id, doc_id, filepath, title, heading, body)
  │  │   (Document will be included in the batch_clear_fts_dirty below)
  │  │
  │  ├── File exists but hash differs:
  │  │   Skip — file changed since indexing. Phase 1 will detect
  │  │   the hash mismatch and fully re-process this document.
  │  │
  │  └── File deleted from disk:
  │      Skip — Phase 1 will deactivate this document.
  │
  Commit Tantivy writer
  batch_clear_fts_dirty() for all successfully replayed documents

Phase 1: Scan + Extract + Chunk + FTS Index (single-threaded, per file)
  Track a batch of document IDs whose Tantivy entries are uncommitted.
  For each collection:
    Scan directory (apply hardcoded ignores → ignore patterns → extractor registry → extensions filter)
    │
    For each file that passes filters:
    │  stat() file → get mtime (modification time)
    │  Compare mtime with stored `modified` in SQLite
    │  │
    │  ├── mtime unchanged → skip entirely (fast path, no file read)
    │  │
    │  └── mtime changed OR new file:
    │      Read file, compute SHA-256 hash
    │      Compare with stored document hash
    │      │
    │      ├── Hash unchanged → update stored mtime, skip re-indexing
    │      │
    │      └── Hash changed OR new file:
    │          Run extractor (Markdown/Code/Plaintext) → PreChunks
    │          Run chunker (enforce 512-token budget) → final Chunks
    │          In a single SQLite transaction:
    │          │  UPSERT document row (set fts_dirty = 1)
    │          │  DELETE old chunks for this document (CASCADE clears embeddings)
    │          │  INSERT new chunk rows
    │          delete_tantivy_by_doc(space, doc_id)  ← remove all old entries by document
    │          delete_usearch for old chunk_ids (if known) or by doc lookup
    │          Add new entries to Tantivy (chunk_id, doc_id, filepath, title, heading, body)
    │          Add doc_id to uncommitted batch
    │          Done with this file
    │
    │  Every N documents (default: 100):
    │      Commit Tantivy writer
    │      batch_clear_fts_dirty() for all doc_ids in uncommitted batch
    │      Reset uncommitted batch
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
    Final: Commit Tantivy writer for any remaining uncommitted entries
    batch_clear_fts_dirty() for remaining doc_ids in uncommitted batch

Phase 2: Embedding (batched, skipped if --no-embed)
  [if models available and --no-embed not set]:
    Loop:
    │  Query get_unembedded_chunks(current_model, limit=64) → Vec<EmbedRecord>
    │  If empty → done
    │  For each chunk in batch:
    │  │  Resolve file path: collection_path / doc_path
    │  │  Read chunk text from file at offset/length
    │  │  (If file deleted or unreadable, skip chunk and log warning)
    │  embed() batch texts → vectors
    │  Group by space_name:
    │  │  batch_insert_usearch(space, entries) for each space
    │  insert_embeddings(entries) in SQLite
```

The file is the unit of change. When a file's hash changes, all its chunks are deleted and re-created — no chunk-level diffing. This is correct because chunk boundaries shift when content changes (a new paragraph shifts every subsequent offset).

The mtime pre-check makes "nothing changed" scans extremely fast — just a `stat()` syscall per file (~1μs each). For a 10K file collection, a full scan where nothing changed takes ~10ms. The SHA-256 hash remains the source of truth for actual content changes; mtime is only used to avoid unnecessary file reads.

### Crash Safety

The invariant: if the process crashes at any point during `kbolt update`, re-running `kbolt update` will converge all data to the correct state.

**FTS recovery** uses the `fts_dirty` flag on the documents table. The flag is set to 1 in the same SQLite transaction that writes chunk mutations, and cleared to 0 only after the Tantivy commit that includes those chunks succeeds (via `batch_clear_fts_dirty` at each commit point). This ensures the flag accurately reflects whether Tantivy is in sync.

- **Crash after SQLite commit but before Tantivy commit**: The document has `fts_dirty = 1` and chunks in SQLite. On next update, Phase 0 reads chunk metadata from SQLite, reads chunk text from the live file on disk, and replays the Tantivy writes. If the file has changed since indexing (hash mismatch), Phase 0 skips it — Phase 1 will detect the change and fully reprocess. If the file was deleted, Phase 0 skips it — Phase 1 will deactivate it.
- **Crash before SQLite commit**: SQLite rolls back the transaction. The document keeps its old hash. On next update, the file is re-detected as changed and fully reprocessed. `fts_dirty` was never set.
- **Crash during Phase 0 itself**: The dirty documents remain dirty (`batch_clear_fts_dirty` hasn't run). Next update replays them again. Phase 0 is idempotent — re-indexing the same chunks into Tantivy is a no-op (same chunk_ids overwrite the same entries).

**Dense/embedding recovery** is entirely separate. The `embeddings` table is the ledger for what should be in USearch. `get_unembedded_chunks()` picks up where it left off. The USearch sync check (count comparison) handles file-level divergence. If a file was deleted between Phase 1 and Phase 2, the embedding pass skips those chunks with a warning — the next update will deactivate the document and CASCADE-delete its chunks and embeddings.

### Embedding Integrity

During `update`, the system automatically detects and corrects two embedding integrity problems. No user intervention or flags needed.

**Model mismatch detection**: The `embeddings` table records which model produced each embedding (the `model` column). On each update, the system compares stored model names against the current model in `index.toml`. If they differ — the user changed their embedding model — all existing embedding rows for the old model are deleted and the corresponding USearch vectors are removed. The affected chunks are then re-embedded with the new model. This happens transparently during the normal embedding phase.

**USearch sync check**: The `embeddings` table in SQLite is the ledger of what *should* be in USearch. USearch stores the actual float vectors. These can diverge — USearch file deleted, corrupted, or partially written. On each update, the system compares the count of embedding rows in SQLite against the count of vectors in USearch. If they disagree, the system clears both (embeddings rows + USearch vectors) and re-embeds all chunks from scratch. This is a coarse check — it catches file-level corruption, not individual vector corruption, which is sufficient because USearch either works or it doesn't.

**Why no `--force-embed` flag**: Both real scenarios (model change, USearch corruption) are handled by auto-detection. A manual `--force-embed` flag would require the user to understand that embeddings exist as a separate layer, that they can go stale, and that the system might have missed something — three layers of internals. If auto-detection has a bug, the fix is to fix auto-detection, not to expose an escape hatch.

---

## Filesystem Sync Strategy

Kbolt indexes files from the user's filesystem. Those files can change at any time — edited, deleted, renamed, moved. The index is a snapshot that can become stale relative to the live filesystem.

### V1 Approach: Explicit Update

In V1, `kbolt update` is the **only** mechanism for syncing the index with the filesystem. There are no file watchers, no background daemons, and no auto-scan at search time. This matches the approach used by Cursor (~5 min periodic sync), Khoj (user-triggered), and Zoekt (periodic poll) — the dominant pattern in local RAG and code search tools.

**Change detection pipeline** (per file during `kbolt update`):

1. `stat()` the file → get mtime (~1μs per file)
2. Compare mtime against stored `modified` in SQLite
3. If mtime unchanged → **skip** (no file read needed)
4. If mtime changed → read file, compute SHA-256 hash
5. Compare hash against stored `hash` in SQLite
6. If hash unchanged → update stored mtime, skip re-indexing
7. If hash changed → full re-extract, re-chunk, re-index

This two-level check (mtime first, hash second) means a "nothing changed" scan is just a batch of `stat()` calls — ~10ms for 10K files, ~50ms for 50K files. The SHA-256 hash is the source of truth for actual content changes; mtime is only a fast pre-filter to avoid unnecessary file reads.

**Freshness mechanisms**:

- **Manual**: `kbolt update` (or scoped: `--space`, `--collection`)
- **Scheduled**: `kbolt schedule --every {interval}` creates a cron job / launchd plist
- **On collection add**: `kbolt collection add` triggers indexing immediately (unless `--no-index`)

### Staleness Transparency

The system is transparent about index freshness rather than hiding it:

- **Search output** includes a staleness hint: "Index last updated: 3h ago" (based on the most recent update timestamp across searched collections). Helps users understand why a recently saved file might not appear in results.
- **`get` / `multi-get`**: Always reads the **live file** from disk (not a cached copy). If the file's current hash doesn't match the indexed hash, the response includes a `stale: true` metadata field indicating the content has changed since indexing. If the file has been deleted, returns an error: "File deleted since indexing. Run `kbolt update` to refresh."
- **`status`**: Shows per-collection last update timestamps so users can see which collections are fresh.

### Content Storage

Kbolt does **not** store file content — only metadata, FTS index, and vectors. `get` and `multi-get` read directly from the live filesystem. This means:

- No storage overhead (no content duplication)
- `get` always returns the current version of the file (fresh reads)
- If a file is deleted, `get` fails with a clear error (no stale fallback)
- Search snippets may not match the live file if content changed since indexing (signaled by `stale: true`)

### V2 Roadmap

- **Auto-scan before search**: On each search, run a quick mtime scan of the searched collections. If files changed, re-index them into Tantivy (FTS only — embedding is too slow for inline). This would make keyword search always fresh, with vectors lagging until the next full update. The ~10-50ms overhead is imperceptible for typical collection sizes.
- **Optional file watcher daemon**: `kbolt watch` starts a background process using platform-native filesystem events (FSEvents on macOS, inotify on Linux) via the `notify` crate. Debounces changes and runs incremental FTS-only updates. Additive — if the daemon isn't running, explicit `kbolt update` still works.
- **Git-aware optimization**: For git-backed collections, use `git diff` / `git status` instead of scanning the full tree. Extremely fast and precise for codebases, but only applicable to git repos.
- **MCP session-aware updates**: Trigger a scan when an MCP client connects, so the LLM starts with a fresh index.

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

[models.embedder]
provider = "huggingface"
id = "google/EmbeddingGemma-256"
# revision = "main"

[models.reranker]
provider = "huggingface"
id = "ExpedientFalcon/qwen3-reranker-0.6b-q8"

[models.expander]
provider = "huggingface"
id = "Qwen/Qwen3-1.7B-q4"

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
    --debug                        Include per-signal scores in results
    -c, --collection <name,...>    Scope to one or more collections (comma-separated)
    -n, --limit <N>                Max results (default: 10)

DOCUMENTS
  kbolt get <path|docid>            Get document by path or #docid
    --offset <N>                   Start at line N
    --limit <N>                    Max lines
  kbolt multi-get <path|docid,...>   Batch retrieve by paths and/or docids
    --max-files <N>                Max files returned (default: 20)
    --max-bytes <N>                Max total bytes (default: 50KB)
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
3. **multi_get** — Batch retrieve by paths and/or docids, with file count and byte budget caps
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

Download path is provider-agnostic in core: model orchestration depends on a provider abstraction, and artifact download is delegated to provider adapters. Default adapter is HuggingFace Hub via `hf-hub` (resumable downloads, checksum verification, local caching). `kbolt models pull` is the explicit pre-download path.

Prompting is CLI-only: interactive terminal sessions may prompt to pull models when missing; non-interactive CLI and MCP never prompt.

Models loaded lazily — embedder loads on first `kbolt update` (with embedding), reranker loads on first search that triggers reranking, expander loads on first `--deep` search. Once loaded, models stay in memory for the process lifetime (freed when the process exits). No inactivity timeout in V1 — it would add timer/reload complexity and cause unpredictable latency spikes during MCP sessions. Memory reclamation via timeout is a V2/daemon-mode feature.

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
