# Kbolt — V1 Specification

> *A local-first retrieval engine that brings your documents to light.*

## Overview

A best-in-class local-first retrieval engine: correct, fast, and extensible. V1 focuses on core retrieval quality with clean extension points. Model training, SPLADE, and advanced features are deferred to V2.

**Stack**: Rust, Tantivy (FTS), USearch (dense vectors), SQLite (metadata + entity storage), HTTP inference clients against local `llama.cpp server` and remote OpenAI-compatible deployments, tree-sitter (code), pulldown-cmark (markdown), clap (CLI), TOML (system config).

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

Kbolt itself is a **single process** in V1. No kbolt daemon, no kbolt-owned background
inference service.

```
kbolt search "query"     → process starts, searches, prints, exits
kbolt update             → process starts, scans/indexes, exits
kbolt mcp                → process starts, stays alive for MCP client session, exits on disconnect
```

No kbolt process runs between commands. MCP sessions keep the `Engine` warm because the process
stays alive for the client session, but local inference deployments are external to kbolt and may
outlive any given kbolt process.

Scheduled indexing uses OS-managed jobs: launchd on macOS and systemd user timers on Linux. `kbolt schedule` manages those artifacts from persisted schedule definitions.

V2 extension path: `kbolt serve` daemon (HTTP + MCP) or a kbolt-owned inference broker. CLI would
check "is daemon running?" and send requests to it instead of creating a local Engine. Core
modules do not need their role contracts rewritten for that.

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
- **core/** — all business logic, storage, model inference. 90% of the code. External dependencies: `rusqlite`, `tantivy`, `usearch`, HTTP client libraries, `tree-sitter`, `pulldown-cmark`, `tokenizers`.
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
                      ┌────────────────────────────┐
                      │           Engine           │
                      │ storage, config, gateway  │
                      └─────────────┬──────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
        ┌───────────────┐   ┌───────────────┐   ┌──────────────┐
        │ engine/search │   │ engine/update │   │ engine/ignore│
        │   search()    │   │   update()    │   │   commands   │
        └───────┬───────┘   └───────┬───────┘   └──────┬───────┘
                │                   │                  │
                ├──────────┬────────┴──────────┬───────┤
                ▼          ▼                   ▼       ▼
            storage/     models/             ingest/  config/

──────┼────────────────────┼─────────────────────────────────┼──────────────
      │ External           │                                 │
      ▼                    ▼                                 ▼
┌───────────────┐     ┌──────────────────────────────┐  ┌──────────────┐
│ ~/.cache/     │     │ Inference deployments        │  │ ~/.config/   │
│ meta.sqlite   │     │ - local llama.cpp server     │  │ index.toml   │
│ spaces/{name} │     │ - remote OpenAI-compatible   │  │ ignores/     │
└───────────────┘     └──────────────────────────────┘  └──────────────┘
```

### External I/O

Every module that crosses the Core boundary:

| Module | External System | Direction | What |
|---|---|---|---|
| `storage/` | `~/.cache/meta.sqlite` | read/write | SQLite — all entity CRUD (spaces, collections, documents, chunks, embeddings, cache) |
| `storage/` | `~/.cache/spaces/{name}/tantivy/` | read/write | Per-space Tantivy — BM25 index |
| `storage/` | `~/.cache/spaces/{name}/vectors.usearch` | read/write | Per-space USearch — HNSW vector index |
| `models/` | `llama.cpp server` deployments | HTTP client | Local embedding, reranking, and expansion inference |
| `models/` | OpenAI-compatible deployments | HTTP client | Remote embedding, reranking, and expansion inference |
| `config/` | `~/.config/index.toml` | read/write | System settings (provider profiles, role bindings, reaping, default space) |
| `config/` | `~/.config/ignores/` | read/write | Ignore patterns (per space/collection) |
| `ingest/` | Collection directories | read | Scan files, read bytes, compute hashes |

Only `storage/`, `models/`, `config/`, and `engine/update_ops` touch the outside world. Other engine submodules and `ingest/` extraction/chunking are internal logic.

### Component Connectivity

**Ownership** — Engine creates and holds as struct fields:
- Engine -> Storage: creates at startup, holds for process lifetime
- Engine -> Models role adapters: built at startup from Config (`Embedder`, `EmbeddingDocumentSizer`, `Reranker`, `Expander`)
- Engine -> Config: loads at startup via `config::load()`, holds for process lifetime

**Delegation** — Engine dispatches into internal engine modules:
- `engine/search_ops.rs`: keyword, semantic, auto, deep ranking + result assembly
- `engine/update_ops.rs`: scan/update/index/embed pipeline
- `engine/ignore_ops.rs`: collection ignore CRUD

**Borrowing** — internal modules operate through `&self` on Engine and call:
- `storage.*` for CRUD + Tantivy/USearch operations
- role adapters (`embedder`, `reranker`, `expander`) for inference
- `ingest::extract` + `ingest::chunk` for extraction/chunking during update

**Direct external I/O**:
- `engine/update_ops` reads files from disk: scans collection directories, reads bytes, computes SHA-256 hashes
- `engine/search_ops` reads source files for snippet assembly

**Construction-time data flow** (once when Engine is created):
- `config::load(path)` -> returns `Config` struct (provider profiles + role bindings + storage policy)
- `Engine::new()` resolves role bindings through the inference gateway and builds provider-backed role adapters
- `Engine::new()` passes `config.cache_dir` -> `Storage::new()`

**No connection** (by design):
- `config/` <-> `storage/`: independent
- `storage/` <-> `models/`: never call each other
- provider clients do not know search/update/storage orchestration

**Module dependency rules** — no circular dependencies:
- `config/` -> nothing (reads TOML, returns Config struct)
- `storage/` -> nothing (receives operations, manages three stores)
- `models/` -> nothing (resolves provider bindings, talks to inference deployments)
- `ingest/` -> extraction/chunking only (no storage/model orchestration)
- `engine` -> orchestrates storage/models/config/ingest

**Public vs internal functions**: adapters call Engine public methods; internal engine submodules remain private implementation detail.

### Core Module Structure

```
core/
  lib.rs                 # pub mod engine, storage, ingest, models, config
  engine.rs              # Engine facade (public API entry points)
  engine/
    search_ops.rs        # search ranking + deep search + result assembly
    update_ops.rs        # update scan/index/embed/reconcile pipeline
    ignore_ops.rs        # collection ignore CRUD operations
    scoring.rs           # score normalization/fusion helpers
    text_helpers.rs      # snippet + contextual-prefix text assembly helpers
    path_utils.rs        # path/docid normalization helpers
    file_utils.rs        # hashing/title/file-error helpers
    ignore_helpers.rs    # ignore matcher + ignore path/pattern helpers
    tests.rs             # engine integration tests

  storage/
    mod.rs               # Storage struct — public API, owns db + tantivy + usearch
    sqlite.rs            # SQLite operations (collections, documents, chunks, embeddings, cache)
    tantivy.rs           # Tantivy operations (index, query, field boosting)
    usearch.rs           # USearch operations (insert, search, quantization)

  ingest/
    mod.rs               # ingest submodule exports
    extract.rs           # extractor registry + shared extraction contracts
    chunk.rs             # file-independent chunking pipeline
    markdown.rs          # markdown extraction implementation
    plaintext.rs         # plaintext extraction implementation
    code.rs              # code extraction implementation

  models/
    mod.rs               # inference entry points + readiness reporting
    gateway.rs           # role -> provider deployment resolution
    embedder.rs          # Embedder trait
    reranker.rs          # Reranker trait
    expander.rs          # Expander trait
    inference.rs         # provider-backed role implementations
    http.rs              # shared HTTP transport/retry/readiness behavior
    chat.rs              # chat completion request/response shaping
    completion.rs        # shared completion client contract
    variants_expander.rs # query-variant expander implementation
    text.rs              # shared text normalization helpers

  config/
    mod.rs               # Config struct, provider profiles, role bindings, TOML load/save
```

### Engine (composition root)

Engine owns `Storage`, `Config`, and role adapters as struct fields. Every operation flows through Engine.

```rust
pub struct Engine {
    storage: Storage,
    config: Config,
    embedder: Option<Arc<dyn Embedder>>,
    embedding_document_sizer: Option<Arc<dyn EmbeddingDocumentSizer>>,
    reranker: Option<Arc<dyn Reranker>>,
    expander: Option<Arc<dyn Expander>>,
}

impl Engine {
    pub fn new(config_path: Option<&Path>) -> Result<Self>;

    // Search / indexing orchestration
    pub fn search(&self, req: SearchRequest) -> Result<SearchResponse>;
    pub fn update(&self, opts: UpdateOptions) -> Result<UpdateReport>;

    // Documents
    pub fn get_document(&self, req: GetRequest) -> Result<DocumentResponse>;
    pub fn multi_get(&self, req: MultiGetRequest) -> Result<MultiGetResponse>;
    pub fn list_files(&self, space: Option<&str>, collection: &str,
                       prefix: Option<&str>) -> Result<Vec<FileEntry>>;

    // Spaces
    pub fn add_space(&self, name: &str, description: Option<&str>) -> Result<SpaceInfo>;
    pub fn remove_space(&self, name: &str) -> Result<()>;
    pub fn rename_space(&self, old: &str, new: &str) -> Result<()>;
    pub fn describe_space(&self, name: &str, description: &str) -> Result<()>;
    pub fn list_spaces(&self) -> Result<Vec<SpaceInfo>>;
    pub fn space_info(&self, name: &str) -> Result<SpaceInfo>;
    pub fn set_default_space(&mut self, name: Option<&str>) -> Result<Option<String>>;
    pub fn resolve_space(&self, explicit: Option<&str>) -> Result<String>;

    // Collections
    pub fn add_collection(&self, req: AddCollectionRequest) -> Result<AddCollectionResult>;
    pub fn remove_collection(&self, space: Option<&str>, name: &str) -> Result<()>;
    pub fn rename_collection(&self, space: Option<&str>, old: &str, new: &str) -> Result<()>;
    pub fn describe_collection(&self, space: Option<&str>, name: &str, desc: &str) -> Result<()>;
    pub fn list_collections(&self, space: Option<&str>) -> Result<Vec<CollectionInfo>>;
    pub fn collection_info(&self, space: Option<&str>, name: &str) -> Result<CollectionInfo>;

    // Inference / readiness
    pub fn model_status(&self) -> Result<ModelStatus>;
    pub fn status(&self, space: Option<&str>) -> Result<StatusResponse>;
}
```

Engine is the only type adapters interact with. Public methods map to CLI/MCP operations; internal complexity is delegated to private engine submodules.

### Storage (three-store model, owns concurrency)

Storage is the data layer. SQLite is global (one database for all spaces). Tantivy and USearch are per-space — each space gets its own index for BM25 IDF isolation. All connected by `chunk_id` as the universal join key.

```rust
pub struct Storage {
    db: Mutex<rusqlite::Connection>,                     // global, single writer, WAL
    cache_dir: PathBuf,
    spaces: RwLock<HashMap<String, SpaceIndexes>>,       // per-space search indexes
}

struct SpaceIndexes {
    _tantivy_dir: PathBuf,
    usearch_path: PathBuf,
    tantivy_index: tantivy::Index,
    tantivy_writer: Mutex<tantivy::IndexWriter>,         // single writer
    usearch_index: RwLock<usearch::Index>,               // concurrent reads, exclusive writes
}
```

| Store | Role | Stores | Good at |
|---|---|---|---|
| SQLite | Source of truth | All entities (spaces, collections, documents, chunks, embeddings) | Relational queries, joins, CRUD, metadata |
| Tantivy (per-space) | BM25 search | Denormalized chunk entries (chunk_id, filepath, title, heading, body) | Full-text search, tokenization, field boosting |
| USearch (per-space) | Dense search | Vectors keyed by chunk_id | Approximate nearest neighbor in high dimensions |

SQLite is the source of truth. Tantivy and USearch are derived indexes — if corrupted or deleted, they can be rebuilt from SQLite + files on disk + model inference. Per-space isolation means a code-heavy space won't skew BM25 IDF statistics for a notes space.

Engine search/update modules do not manage index locks — they call `storage.query_bm25(space, ...)`, `storage.insert_chunks(...)`, etc. Storage handles synchronization internally.

**Space index lifecycle**: All known spaces are eagerly opened at `Storage::new()` by scanning the spaces table and loading each space's Tantivy/USearch handles. `open_space`/`close_space` take a write lock on the spaces map. Read operations take a read lock and use the selected entry for the query/index mutation. This keeps index ownership centralized in Storage and prevents adapter-level lock coupling.

### Models (gateway, provider clients, role adapters)

`models/` hides backend protocol details behind role interfaces. Engine owns role instances
(`Embedder`, `Reranker`, `Expander`) and builds them from config at construction time through:
- a gateway that resolves `roles.* -> providers.*`
- provider clients that know backend protocols (`llama.cpp server`, `openai_compatible`)
- role adapters that implement kbolt's role traits on top of those provider capabilities

Kbolt does not own local model download or local runtime construction in this architecture.

Three separate traits — each defines its own contract because they share nothing in common (different inputs, outputs, runtimes, thread-safety):

```rust
pub trait Extractor: Send + Sync {
    fn supports(&self) -> &[&str];   // file extensions: ["md", "markdown"]
    fn supports_path(&self, _path: &Path) -> bool { false } // optional fallback for path-based matching
    fn extract(&self, path: &Path, bytes: &[u8]) -> Result<ExtractedDocument>;
}

pub trait Embedder: Send + Sync {
    fn embed_batch(&self, kind: EmbeddingInputKind, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>>;
}

pub trait Expander: Send + Sync {
    fn expand(&self, query: &str, max_variants: usize) -> Result<Vec<String>>;
}

pub enum EmbeddingInputKind {
    Query,
    Document,
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
    #[error("model not available: {name}")]
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
    pub score: f32,                        // final query-local ranking score [0, 1]
    pub signals: Option<SearchSignals>,    // per-signal breakdown (populated when debug requested)
}

pub struct SearchSignals {
    pub bm25: Option<f32>,                 // normalized BM25 score
    pub dense: Option<f32>,                // dense distance-derived score
    pub fusion: f32,                       // normalized first-stage fusion score
    pub reranker: Option<f32>,             // raw cross-encoder score used for rerank ordering
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
    pub warnings: Vec<String>,             // non-fatal issues (for example, files deleted since indexing)
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

pub struct AddCollectionResult {
    pub collection: CollectionInfo,
    pub initial_indexing: InitialIndexingOutcome,
}

pub enum InitialIndexingOutcome {
    Skipped,                            // collection registered with --no-index
    Indexed(UpdateReport),              // initial indexing ran; completeness comes from report
    Blocked(InitialIndexingBlock),      // collection registered, but indexing could not start
}

pub enum InitialIndexingBlock {
    SpaceDenseRepairRequired {
        space: String,
        reason: String,
    },
    ModelNotAvailable {
        name: String,
    },
}
```

#### Indexing Types

```rust
pub struct UpdateOptions {
    pub space: Option<String>,             // scope to space (None = all spaces)
    pub collections: Vec<String>,          // scope to collections (empty = all)
    pub no_embed: bool,                    // skip embedding (FTS-only indexing)
    pub dry_run: bool,                     // preview what would change, no writes
    pub verbose: bool,                     // include per-file decisions in update output
}

pub struct UpdateReport {
    pub scanned_docs: usize,               // candidate files examined as documents
    pub skipped_mtime_docs: usize,         // active docs skipped by mtime fast path
    pub skipped_hash_docs: usize,          // active docs skipped after hash match
    pub added_docs: usize,                 // new docs indexed (or that would be indexed in dry-run)
    pub updated_docs: usize,               // changed docs re-indexed (or that would be in dry-run)
    pub failed_docs: usize,                // unique docs with non-fatal failures this run
    pub deactivated_docs: usize,           // docs no longer seen in the collection scan
    pub reactivated_docs: usize,           // previously inactive docs made active again
    pub reaped_docs: usize,                // inactive docs hard-deleted by reaping
    pub embedded_chunks: usize,            // chunks that received embeddings
    pub decisions: Vec<UpdateDecision>,    // verbose per-doc decisions (only when verbose)
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
    pub configured: bool,
    pub ready: bool,
    pub profile: Option<String>,
    pub kind: Option<String>,
    pub operation: Option<String>,
    pub model: Option<String>,
    pub endpoint: Option<String>,
    pub issue: Option<String>,
}

pub struct DiskUsage {
    pub sqlite_bytes: u64,
    pub tantivy_bytes: u64,
    pub usearch_bytes: u64,
    pub models_bytes: u64,                // counts files under ~/.cache/kbolt/models/ if present
    pub total_bytes: u64,
}
```

### Storage API

The Storage struct's public methods — the internal contract that engine orchestration modules depend on. All methods take `&self` — Storage handles its own locking internally via `Mutex<Connection>`, `Mutex<IndexWriter>`, and `RwLock<usearch::Index>`.

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
    pub fn get_fts_dirty_documents_in_space(&self, space_id: i64) -> Result<Vec<FtsDirtyRecord>>;
    pub fn get_fts_dirty_documents_in_collections(&self, collection_ids: &[i64]) -> Result<Vec<FtsDirtyRecord>>;
    pub fn batch_clear_fts_dirty(&self, doc_ids: &[i64]) -> Result<()>;   // set fts_dirty = 0 for a batch of documents
    pub fn list_reapable_documents_in_space(&self, older_than_days: u32, space_id: i64) -> Result<Vec<ReapableDocument>>;
    pub fn list_reapable_documents_in_collections(&self, older_than_days: u32, collection_ids: &[i64]) -> Result<Vec<ReapableDocument>>;

    // --- Chunk operations (SQLite) ---
    pub fn insert_chunks(&self, doc_id: i64, chunks: &[ChunkInsert]) -> Result<Vec<i64>>;
    pub fn delete_chunks_for_document(&self, doc_id: i64) -> Result<Vec<i64>>;
    pub fn get_chunks(&self, chunk_ids: &[i64]) -> Result<Vec<ChunkRow>>;
    pub fn get_chunks_for_document(&self, doc_id: i64) -> Result<Vec<ChunkRow>>;

    // --- Embedding tracking (SQLite) ---
    pub fn insert_embeddings(&self, entries: &[(i64, &str)]) -> Result<()>;
    pub fn get_unembedded_chunks(&self, model: &str, after_chunk_id: i64, limit: usize) -> Result<Vec<EmbedRecord>>;  // chunks needing embedding, active documents only, with context for disk reads
    pub fn get_unembedded_chunks_in_space(&self, model: &str, space_id: i64, after_chunk_id: i64, limit: usize) -> Result<Vec<EmbedRecord>>;
    pub fn get_unembedded_chunks_in_collections(&self, model: &str, collection_ids: &[i64], after_chunk_id: i64, limit: usize) -> Result<Vec<EmbedRecord>>;
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
    pub title_source: DocumentTitleSource,
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
pub enum ChunkKind {
    Section,                            // section-scoped narrative chunk
    Paragraph,                          // paragraph/list/quote narrative chunk
    Code,                               // markdown code-fence chunk
    Table,                              // markdown table row-group chunk
    Mixed,                              // forced merge/split across heterogeneous block kinds
    Function,                           // code-file function/method chunk
    Class,                              // code-file class/struct chunk
}
pub struct TantivyEntry {
    pub chunk_id: i64, pub doc_id: i64, pub filepath: String,
    pub semantic_title: Option<String>,
    pub heading: Option<String>, pub body: String,
}
pub struct BM25Hit { pub chunk_id: i64, pub score: f32 }
pub struct DenseHit { pub chunk_id: i64, pub distance: f32 }

// Returned by get_fts_dirty_documents — everything needed to replay Tantivy writes from disk
pub struct FtsDirtyRecord {
    pub doc_id: i64,
    pub doc_path: String,                  // relative path within collection
    pub doc_title: String,
    pub doc_title_source: DocumentTitleSource,
    pub doc_hash: String,                  // stored hash, for verifying file hasn't changed
    pub collection_path: PathBuf,          // absolute path to collection root
    pub space_name: String,
    pub chunks: Vec<ChunkRow>,             // all chunks for this document
}

pub enum DocumentTitleSource {
    Extracted,
    FilenameFallback,
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

### Engine Internal Search/Update Ops

Search and update internals live under `core/src/engine/` and are private implementation modules, not separate top-level public APIs.

- `search_ops.rs`: keyword/semantic/auto/deep ranking and final result assembly.
- `update_ops.rs`: update pipeline (scan -> filter -> extract -> chunk -> store -> FTS -> embed), plus replay/reconciliation helpers.
- `ignore_ops.rs`: collection ignore CRUD behavior.
- helper modules (`scoring.rs`, `text_helpers.rs`, `path_utils.rs`, `file_utils.rs`, `ignore_helpers.rs`) keep pure utility logic isolated.

Extraction/chunking remain in `core/src/ingest/` and are invoked from `engine/update_ops.rs`.

### Config API

```rust
// config/mod.rs

pub struct Config {
    pub config_dir: PathBuf,               // ~/.config/kbolt/
    pub cache_dir: PathBuf,                // ~/.cache/kbolt/
    pub default_space: Option<String>,
    pub providers: HashMap<String, ProviderProfileConfig>,
    pub roles: RoleBindingsConfig,
    pub reaping: ReapingConfig,
    pub chunking: ChunkingConfig,
    pub ranking: RankingConfig,
}

pub enum ProviderProfileConfig {
    LlamaCppServer {
        operation: ProviderOperation,
        base_url: String,
        model: String,
        timeout_ms: u64,
        max_retries: u32,
    },
    OpenAiCompatible {
        operation: ProviderOperation,
        base_url: String,
        model: String,
        api_key_env: Option<String>,
        timeout_ms: u64,
        max_retries: u32,
    },
}

pub enum ProviderOperation {
    Embedding,
    Reranking,
    ChatCompletion,
}

pub struct RoleBindingsConfig {
    pub embedder: Option<EmbedderRoleConfig>,
    pub reranker: Option<RerankerRoleConfig>,
    pub expander: Option<ExpanderRoleConfig>,
}

pub struct EmbedderRoleConfig {
    pub provider: String,
    pub batch_size: usize,
}

pub struct RerankerRoleConfig {
    pub provider: String,
}

pub struct ExpanderRoleConfig {
    pub provider: String,
    pub max_tokens: usize,
    pub sampling: ExpanderSamplingConfig,
}

pub struct ExpanderSamplingConfig {
    pub seed: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_last_n: i32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

pub struct ReapingConfig {
    pub days: u32,                         // default: 7
}

pub struct ChunkingConfig {
    pub defaults: ChunkPolicy,
    pub profiles: HashMap<String, ChunkPolicy>,   // md, code, txt
}

pub struct RankingConfig {
    pub deep_variant_rrf_k: usize,
    pub deep_variants_max: usize,
    pub initial_candidate_limit_min: usize,
    pub rerank_candidates_min: usize,
    pub rerank_candidates_max: usize,
    pub hybrid_fusion: HybridFusionConfig,
    pub bm25_boosts: Bm25BoostsConfig,
}

pub struct HybridFusionConfig {
    pub mode: HybridFusionMode,
    pub linear: LinearHybridFusionConfig,
    pub dbsf: DbsfHybridFusionConfig,
    pub rrf: RrfHybridFusionConfig,
}

pub struct LinearHybridFusionConfig {
    pub dense_weight: f32,
    pub bm25_weight: f32,
}

pub struct DbsfHybridFusionConfig {
    pub dense_weight: f32,
    pub bm25_weight: f32,
    pub stddevs: f32,
}

pub struct RrfHybridFusionConfig {
    pub k: usize,
}

pub enum HybridFusionMode {
    Rrf,
    Dbsf,
    Linear,
}

pub fn load(config_path: Option<&Path>) -> Result<Config>;
pub fn save(config: &Config) -> Result<()>;
```

`load` reads `~/.config/kbolt/index.toml` (or creates it with defaults if missing). `save` writes back to disk. Config is loaded once at Engine construction time and held as a struct field. Adapter commands that modify config (like `kbolt space default`) call `save` after mutation.

### Models API

Trait definitions are listed in the Architecture section (`Embedder`, `Reranker`, `Expander`, `Extractor`). Runtime model orchestration exposed from `models/mod.rs` is:

Engine composes role adapters through one inference gateway:
- `Config`
- `Gateway` (role -> provider deployment resolution)
- provider client (`llama.cpp server` or `openai_compatible`)
- role adapter (`Embedder`, `Reranker`, `Expander`)

Unbound roles stay unconfigured: deep search requires an expander, reranking is skipped with an
explicit pipeline notice when no reranker is configured, and semantic search requires an
embedder binding.

Inference provider scope is deployment-based:
- local backend family: `llama.cpp server`
- remote backend family: `openai_compatible`

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
| `title` | TEXT | Display title stored for the document |
| `title_source` | TEXT | `extracted` for semantic titles, `filename_fallback` when only the basename is available |
| `hash` | TEXT | SHA-256 of file bytes — used for change detection |
| `modified` | TEXT | ISO 8601, file modification timestamp |
| `active` | INTEGER | 1 = live, 0 = deactivated (file disappeared from disk) |
| `deactivated_at` | TEXT nullable | ISO 8601 when deactivated, NULL when active |
| `fts_dirty` | INTEGER | 1 = chunks written to SQLite but Tantivy not yet confirmed, 0 = FTS in sync |

UNIQUE constraint on `(collection_id, path)`. Soft delete with auto-reaping: documents deactivated for longer than the configured reaping period (default: 7 days) are hard-deleted during `kbolt update`. Deactivated documents (`active = 0`) are excluded from the embedding backlog (`get_unembedded_chunks` filters `active = 1`) and from search results (result assembly filters out chunks belonging to inactive documents). Chunks and embeddings are retained during deactivation so that if the file reappears (same hash), re-activation skips re-indexing.

`title` remains the display value surfaced in CLI/file listings. Ranking and rerank contextual prefixes only treat `title` as semantic metadata when `title_source = extracted`. `filepath` stays a separate retrieval signal with its own weight.

Older caches created before `title_source` existed must be refreshed with `kbolt update` to repopulate title provenance accurately. The schema migration adds the column, but it does not infer old provenance heuristically.

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
| `kind` | TEXT | `section`, `paragraph`, `code`, `table`, `mixed`, `function`, or `class` |

UNIQUE constraint on `(doc_id, seq)`. Offset and length define a byte slice into the source file on disk — at query time, the snippet is extracted by reading `length` bytes starting at `offset` from the file at `{collection.path}/{document.path}`. Chunks are immutable: when a document changes, all its chunks are deleted and re-created.

**Embedding** — Metadata tracking that a chunk has been embedded by a specific model. The actual float vector lives in USearch.

| Column | Type | Purpose |
|---|---|---|
| `chunk_id` | INTEGER FK | References `chunks.id`, CASCADE delete |
| `model` | TEXT | Model identifier (e.g. `ggml-org/embeddinggemma-300M-GGUF`) |
| `embedded_at` | TEXT | ISO 8601, when the embedding was created |

PRIMARY KEY on `(chunk_id, model)`. Tracks which chunks have been embedded and by which model. When the embedding model changes, stale entries are detected by comparing the `model` column against the current model config.

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
    title_source    TEXT NOT NULL DEFAULT 'extracted',
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

```

### Tantivy Schema (per-space)

Each space has its own Tantivy index at `~/.cache/kbolt/spaces/{space}/tantivy/`. Each Tantivy entry represents one chunk (not one document). Fields are passed explicitly by application code at index time — Tantivy has no awareness of our SQLite schema.

```
chunk_id:  u64  (stored, fast)          — join key back to SQLite chunks table
doc_id:    u64  (stored, fast)          — join key back to SQLite documents table, used for delete-by-document
filepath:  TEXT (stored, indexed, 2x)   — "collection_name/relative/path.md"
title:     TEXT (stored, indexed, 3x)   — semantic document title only when `title_source = extracted`
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
    schedules.toml          # persisted schedule definitions (machine-managed)
    ignores/                # ignore patterns (internal, not in user directories)
        {space}/
            {collection}.ignore   # .gitignore syntax, per-collection

~/.cache/kbolt/
    meta.sqlite             # all entities: spaces, collections, documents, chunks, embeddings
    spaces/                 # per-space indexes (BM25 IDF isolation)
        {space}/
            tantivy/        # Tantivy index (managed by Tantivy)
            vectors.usearch # USearch HNSW file (managed by USearch)
    models/                 # optional residual cache directory; not required by provider-profile inference
    schedules/              # per-schedule run state
        s1.json             # last_started / last_finished / last_result / last_error
```

SQLite is shared globally (one database for all spaces). Tantivy and USearch are per-space — each space has its own index directory under `~/.cache/kbolt/spaces/{space_name}/`. This gives each space independent BM25 IDF statistics, preventing one space's content from skewing another's keyword search quality. The `models/` directory is no longer an active part of inference ownership; disk-usage reporting still counts it if present so old local artifacts remain visible to operators.

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

Users interact with ignore patterns through a virtual path structure: `{space}/{collection}` maps to the internal ignore file. For example, `kbolt ignore show api` shows the ignore patterns for the `api` collection (resolved via space precedence), and `kbolt ignore edit api` opens the file in `$VISUAL` / `$EDITOR` / `vi` (`$VISUAL` / `$EDITOR` may include flags like `code --wait`).

### 3. Extractor Registry + Extensions Filter

A file is only indexed if:
1. An extractor supports its extension (MarkdownExtractor handles `.md`, CodeExtractor handles `.rs`, etc.)
2. If the collection has an `extensions` list, the file's extension is in that list

Both ignore patterns and `extensions` can filter by extension. They compose — both must pass. Ignore patterns are better for excluding paths/directories. `extensions` is better when the set of wanted types is small (e.g. `extensions = ["rs"]` is cleaner than ignoring every other supported extension in the ignore file).

---

## Extractor System

Decision record: `docs/adr/0001-extraction-and-chunking.md`.

```rust
pub struct ExtractedDocument {
    pub blocks: Vec<ExtractedBlock>,         // structural blocks from the source
    pub metadata: HashMap<String, String>,   // frontmatter, language, etc.
    pub title: Option<String>,
}

pub struct ExtractedBlock {
    pub text: String,
    pub offset: usize,                       // byte offset in source file
    pub length: usize,                       // byte length in source file
    pub kind: BlockKind,
    pub heading_path: Vec<String>,           // heading breadcrumb stack
    pub attrs: HashMap<String, String>,      // code language, table headers, etc.
}

pub enum BlockKind {
    Heading,
    Paragraph,
    ListItem,
    BlockQuote,
    CodeFence,
    TableHeader,
    TableRow,
    HtmlBlock,
}
```

Extractor output is a stable intermediate representation (IR). The chunker is file-independent and consumes this IR, not raw file bytes.

### Extractor Abstraction

```rust
pub trait Extractor: Send + Sync {
    fn supports(&self) -> &[&str]; // fast extension dispatch
    fn supports_path(&self, _path: &Path) -> bool { false } // optional fallback
    fn extract(&self, path: &Path, bytes: &[u8]) -> Result<ExtractedDocument>;
}
```

Registry dispatch uses `supports()` as the O(1) fast path (`HashMap<extension, extractor>`). `supports_path()` is an optional slower fallback for extractors needing path-aware checks.

### V1 Extractors

1. **MarkdownExtractor** (CommonMark/GFM parser)
   - Parses heading hierarchy (H1-H6) and maintains heading breadcrumb stack
   - Emits typed blocks for paragraphs, list items, quotes, code fences, tables, HTML blocks
   - Handles frontmatter and surfaces fields in metadata
   - Preserves source spans (`offset`/`length`) for all emitted blocks

2. **PlaintextExtractor** (paragraph-based fallback)
   - Splits on double newlines
   - Emits paragraph blocks with source spans
   - Used for extensions with no richer extractor

3. **CodeExtractor** (baseline V1)
   - Baseline V1 behavior emits code blocks with language metadata
   - Language-specific via extension mapping
   - Structural function/class AST boundaries are deferred beyond baseline V1

### Chunking Pipeline

Chunking is structure-first and budget-constrained.

```rust
pub struct ChunkPolicy {
    pub target_tokens: usize,              // preferred packing target
    pub soft_max_tokens: usize,            // tolerated overrun for clean boundaries
    pub hard_max_tokens: usize,            // absolute ceiling
    pub boundary_overlap_tokens: usize,    // overlap only for forced hard splits
    pub neighbor_window: usize,            // retrieval-time expansion window (default: 1)
    pub contextual_prefix: bool,           // prepend deterministic context for retrieval/embedding text
}
```

Policy defaults for markdown (`.md`) in V1:
- `target_tokens = 800`
- `soft_max_tokens = 950`
- `hard_max_tokens = 1200`
- `boundary_overlap_tokens = 48`
- `neighbor_window = 1`
- `contextual_prefix = true`

Policy resolution precedence (target architecture):
1. CLI override (experimental)
2. Collection-level override (deferred in V1)
3. File-type profile (for example `md`, `code`, `txt`)
4. Global default

V1 supports levels `1`, `3`, and `4`. Collection-level policy storage is deferred until collection metadata/schema support is added.

Chunking rules:
1. Pack adjacent extracted blocks from the same structural neighborhood toward `target_tokens`.
2. Allow overrun to `soft_max_tokens` to avoid splitting at unnatural boundaries.
3. If a block still exceeds `hard_max_tokens`, split by block-specific fallback order:
   - paragraph/list/quote: sentence boundaries, then clause boundaries
   - code fence: blank-line boundaries, then fixed token windows
   - table: row groups with header carryover in retrieval text
4. Apply `boundary_overlap_tokens` only when step 3 forces a hard split.
5. Merge undersized fragments with adjacent compatible fragments when possible.

Final chunk kinds are storage-level labels derived from block composition:
- code-fence only chunks → `ChunkKind::Code`
- table-only chunks (header/rows) → `ChunkKind::Table`
- paragraph/list/quote-only chunks → `ChunkKind::Paragraph`
- heading-scoped narrative chunks → `ChunkKind::Section`
- forced heterogeneous merges/splits → `ChunkKind::Mixed`

Each final chunk has two text views:
1. **source text** — exact file slice from `offset`/`length` for snippets and citation fidelity.
2. **retrieval text** — optional contextualized prefix + source text for BM25/reranker/embedding.

Contextualized prefixes are deterministic and derived from indexed metadata (extracted document title only, heading path, selected frontmatter fields, code language, table header summary). Filename fallback titles are display-only and do not participate in semantic prefixing. Prefixes never mutate stored source spans.
The retrieval text view is computed from source text + metadata at indexing/query time and is not persisted as a second chunks-table text column.

Token counting:
- when the embedder binding resolves to `llama_cpp_server`, chunk packing uses the provider's `/tokenize` endpoint as the document-token authority
- the chunker still owns boundary selection, but candidate chunk text is evaluated in the embedder document-token space instead of whitespace counts
- embedding preflight re-counts the actual payload just before `/v1/embeddings` and rejects oversized payloads locally instead of surfacing a backend 500
- `openai_compatible` embedders do not yet expose an exact token-count capability through the gateway, so they continue to use the deterministic whitespace counter for chunk packing

Operational requirement for local embedding deployments:
- the `llama.cpp server` embedder must be launched with enough batch capacity to accept at least the configured `hard_max_tokens` document payloads; the tested local baseline is `-ub 2048` for `embeddinggemma`

---

## Ingestion Pipeline

`kbolt update` is a single command that scans, extracts, chunks, indexes (FTS), and embeds (dense vectors). The `--no-embed` flag skips embedding if models aren't available.

Update has three scope forms:

- unscoped: no `--space`, no `--collection` → every collection in every space
- space-scoped: `--space work` with no `--collection` → every collection in that space
- collection-targeted: one or more `--collection` values, optionally constrained by `--space` → exactly the resolved target collections

Phase 0, Phase 1, Phase 2 backlog embedding, and reaping all operate on that resolved target set. A scoped update must not replay or repair unrelated collections.

### Concurrency Model

Document-level ingestion is **single-threaded** in V1. Files are processed one at a time through scan → extract → chunk → SQLite → Tantivy. Embedding runs as a separate second pass after all FTS indexing is complete. The pass queries `get_unembedded_chunks()` in bounded batches (default: 64 chunks), which returns chunk metadata plus the file paths needed to read chunk text from disk. For exact-size embedders, kbolt runs a token-count preflight on each pending payload before the batch call. This two-phase design is simple, testable, and matches the `--no-embed` flag naturally (just skip phase 2). The extraction/chunking phase is fast (~1ms per file for text processing) — embedding is the bottleneck, and batching it efficiently matters more than parallelizing extraction.

Note: the `Mutex`/`RwLock` wrappers on Storage fields are justified by search parallelism (BM25 and dense retrieval run in parallel threads during search), not by ingest.

### Per-Document Flow

```
Phase 0: FTS Reconciliation (replay dirty documents from a previous crash)
  Query the dirty-document backlog for the resolved target set:
  - unscoped → get_fts_dirty_documents()
  - space-scoped → get_fts_dirty_documents_in_space(space_id)
  - collection-targeted → get_fts_dirty_documents_in_collections(collection_ids)
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
    │          Run extractor (Markdown/Code/Plaintext) → ExtractedBlocks
    │          Run file-independent chunker (target/soft_max/hard_max policy) → final Chunks
    │          Build replacement chunk rows and embedding payloads in memory
    │          If an exact document-token sizer exists:
    │          │  preflight replacement embedding payloads before mutating storage
    │          │  if an existing document has any rejected replacement payload, keep the old
    │          │  chunks/Tantivy/USearch state intact, report the failure, and skip replacement
    │          Persist the validated replacement:
    │          │  UPSERT document row (set fts_dirty = 1)
    │          │  DELETE old chunks for this document (CASCADE clears embeddings)
    │          │  INSERT new chunk rows
    │          │  mark any rejected new-document chunk ids so the Phase 2 backlog pass does not
    │          │  retry and double-report the same preflight failure
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
    │      within the same resolved target set only
    │      CASCADE removes chunks → embeddings; explicitly remove from Tantivy + USearch
    │
    Final: Commit Tantivy writer for any remaining uncommitted entries
    batch_clear_fts_dirty() for remaining doc_ids in uncommitted batch

Phase 2: Embedding (batched, skipped if --no-embed)
  [if models available and --no-embed not set]:
    Loop:
    │  Query the embedding backlog for the resolved target set:
    │  - unscoped → get_unembedded_chunks(current_model, after_chunk_id, limit=64)
    │  - space-scoped → get_unembedded_chunks_in_space(current_model, space_id, after_chunk_id, limit=64)
    │  - collection-targeted → get_unembedded_chunks_in_collections(current_model, collection_ids, after_chunk_id, limit=64)
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

- **Crash after SQLite commit but before Tantivy commit**: The document has `fts_dirty = 1` and chunks in SQLite. On the next update that includes this document's collection in scope, Phase 0 reads chunk metadata from SQLite, reads chunk text from the live file on disk, and replays the Tantivy writes. If the file has changed since indexing (hash mismatch), Phase 0 skips it — Phase 1 will detect the change and fully reprocess. If the file was deleted, Phase 0 skips it — Phase 1 will deactivate it.
- **Crash before SQLite commit**: SQLite rolls back the transaction. The document keeps its old hash. On next update, the file is re-detected as changed and fully reprocessed. `fts_dirty` was never set.
- **Crash during Phase 0 itself**: The dirty documents remain dirty (`batch_clear_fts_dirty` hasn't run). Next update replays them again. Phase 0 is idempotent — re-indexing the same chunks into Tantivy is a no-op (same chunk_ids overwrite the same entries).

**Dense/embedding recovery** is entirely separate. The `embeddings` table is the ledger for what should be in USearch. The Phase 2 backlog queries pick up missing embeddings only within the current update scope. If a file was deleted between Phase 1 and Phase 2, the embedding pass skips those chunks with a warning — the next update that includes that collection in scope will deactivate the document and CASCADE-delete its chunks and embeddings.

### Embedding Integrity

During `update`, the system detects two embedding integrity problems. Their repair scope is intentionally different from Phase 2 backlog embedding.

**Model mismatch detection**: The `embeddings` table records which model produced each embedding (the `model` column). On each update, the system compares stored model names against the current model in `index.toml` for each touched space. If they differ — the user changed their embedding model — the dense state for that space is invalid and must be rebuilt at the space level.

**USearch sync check**: The `embeddings` table in SQLite is the ledger of what *should* be in USearch. USearch stores the actual float vectors. These can diverge — USearch file deleted, corrupted, or partially written. On each update, the system compares the count of embedding rows in SQLite against the count of vectors in USearch for each touched space. If they disagree, the dense state for that space is invalid and must be rebuilt at the space level. This is a coarse check — it catches space-level ledger/index divergence, not individual vector corruption.

**Repair contract**:

- unscoped update: may rebuild dense state for any touched space automatically
- space-scoped update: may rebuild dense state for that space automatically
- collection-targeted update: must **not** auto-repair whole-space dense state. If model drift or a SQLite-vs-USearch count mismatch is detected in the touched space, the command fails clearly and instructs the user to run `kbolt --space <space> update`

**Why no `--force-embed` flag**: Both real scenarios (model change, USearch corruption) are handled by auto-detection. A manual `--force-embed` flag would require the user to understand that embeddings exist as a separate layer, that they can go stale, and that the system might have missed something — three layers of internals. If auto-detection has a bug, the fix is to fix auto-detection, not to expose an escape hatch.

**Current V1 constraint (deferred)**: Dense state is split between SQLite (embedding ledger) and USearch (vectors), so V1 does not have an atomic dense commit boundary. We keep reconcile-on-update for now and defer a stronger design until it becomes a real problem.

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
- **Scheduled**: `kbolt schedule add ...` persists a schedule and reconciles launchd/systemd user timer artifacts
- **On collection add**: `kbolt collection add` triggers indexing immediately (unless `--no-index`)
  - the command reports the initial indexing outcome from a collection/document point of view
  - if indexing is blocked after registration (for example, space-level dense repair is required or a model is unavailable), the collection remains registered and the command tells the user what to run next

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

Rule-based (no LLM). Auto mode defaults to BM25+dense with reranker off for speed; `--rerank` opts in. `--no-rerank` forces reranker off. `--keyword` forces BM25-only. `--semantic` forces dense-only.

### Stage 3: Query Expansion (--deep only)

Deep search always includes the normalized original query. The expander then adds generated
query variants as plain strings. The engine, not the expander, decides how variants are searched.

In V1, the original query and every generated variant follow the same Stage 4 path independently:
keyword retrieval plus dense retrieval when embeddings are available. Each variant produces its
own candidate list, and those lists are then aggregated before reranking.

### Stage 4: Multi-Signal Retrieval (parallel)

Both signals operate on the same unit: individual chunks.

**BM25 via Tantivy**:
- Query across `body` (1.0x), `title` (2.0x), `heading` (1.5x), `filepath` (0.5x) with configurable field boosting
- Retrieves up to the current candidate limit, then scores are normalized to `[0, 1]`

**Dense via USearch**:
- Embed query via ONNX/GGUF, search HNSW for up to the current candidate limit nearest neighbors
- Score = `1 / (1 + distance)` at retrieval time, then query-local normalization is applied during score-based hybrid fusion modes

Both run in parallel threads. Because both operate at chunk granularity, fusion compares equivalent units.

**Cross-space search**: When searching across multiple spaces (no `--space` specified), each space's Tantivy index and USearch file are queried independently. Candidate lists from all spaces are concatenated before entering fusion. Stage 8 then reranks the combined document pool without regard to source space.

### Stage 5: Fusion (Hybrid)

Hybrid fusion is configurable per query leg and runs before any optional reranking. Kbolt
supports three standard fusion modes.

Linear fusion:

```
fusion(d) = wd * dense_norm(d) + wb * bm25_norm(d)
```

Where `dense_norm` and `bm25_norm` are query-local max-normalized scores in `[0, 1]`.
After fusion, scores are normalized again so the top fused chunk in the query is `1.0`.

Defaults: `mode = "dbsf"`, `dense_weight = 1.0`, `bm25_weight = 0.4`, `stddevs = 3.0`.

DBSF fusion (distribution-based score fusion):

```
fusion(d) = wd * dense_dbsf(d) + wb * bm25_dbsf(d)
```

Each branch is normalized from its own returned score distribution using `mean ± stddevs * stddev`,
clipped to `[0, 1]`, then combined with a weighted sum. This reduces the tendency of flat lexical
result sets to behave like full-strength evidence.
Defaults for that mode: `dense_weight = 1.0`, `bm25_weight = 0.4`, `stddevs = 3.0`.

RRF remains available as an alternative mode:

```
RRF(d) = 1 / (k + rank_bm25(d)) + 1 / (k + rank_dense(d))
```

This optional RRF mode is plain equal-weight rank fusion over chunk positions. It does not use BM25 or dense score magnitude, and a chunk missing from one list simply omits that reciprocal term.
Hybrid RRF uses `ranking.hybrid_fusion.rrf.k`.
Deep-search variant aggregation uses `ranking.deep_variant_rrf_k` and is configured separately.

### Stage 6: Candidate Carry-Through

Retrieval remains chunk-based through the ranking pipeline:
- no document-level deduplication is applied before final result assembly
- when reranking is enabled, Stage 8 selects one highest-fusion representative chunk per document (MaxP) for reranker input construction only
- the final result set may still contain multiple chunks from the same document

### Stage 7: Context Expansion

For each surviving hit chunk, fetch neighboring chunks from the same document by `seq`:
- default window: `±1` chunk (`neighbor_window = 1`)
- configurable per chunking profile
- expansion happens at retrieval time (not as storage-time overlap)

Neighbor expansion is used for answer context in result assembly. It does not change the primary ranking unit.

### Stage 8: Reranking

Cross-encoder (Qwen3-Reranker 0.6B GGUF) scores top-20–30 candidates:
- Input: (query, primary-hit-chunk text) pairs — reranker scores one representative chunk per document (optionally with deterministic contextual prefix), not neighbor-expanded context
- Output: raw query-local relevance score per representative document
- Score application:
  - reranker rank establishes between-document priority
  - chunk-level normalized fusion keeps ordering within each reranked document
  - final result score is the product of a document-rank prior and the chunk's relative first-stage fusion score within that document
- Debug signals keep the raw reranker score and first-stage retrieval signals; the final result score is not a calibrated probability.

### Stage 9: Result Assembly

Per result:
- Document metadata from SQLite (title, path, collection name, space name via join)
- Collection and space descriptions from SQLite
- Heading breadcrumb from chunk
- Snippet extracted by reading source file at chunk offset/length
- Short docid (`#` + first 6 chars of document hash)

Return top results up to `--limit` (default `10`).

---

## Configuration (TOML)

System-level settings only. Spaces and collections are stored in SQLite, managed via CLI.

**Architecture direction**: inference is moving toward provider profiles plus role bindings.
- one local backend family: `llama.cpp server`
- multiple local deployment profiles of that same kind
- many remote backends
- per-role provider choice

See [ADR 0003](./docs/adr/0003-provider-profiles-and-shared-local-inference.md).

```toml
# ~/.config/kbolt/index.toml

default_space = "work"    # optional, set via `kbolt space default`

[providers.local_embed]
kind = "llama_cpp_server"
operation = "embedding"
base_url = "http://127.0.0.1:8101"
model = "embeddinggemma"

[providers.local_rerank]
kind = "llama_cpp_server"
operation = "reranking"
base_url = "http://127.0.0.1:8102"
model = "qwen3-reranker"

[providers.local_expand]
kind = "llama_cpp_server"
operation = "chat_completion"
base_url = "http://127.0.0.1:8103"
model = "qwen3-1.7b"

[providers.remote_expand]
kind = "openai_compatible"
operation = "chat_completion"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5-mini"

[roles.embedder]
provider = "local_embed"
batch_size = 32

[roles.reranker]
provider = "local_rerank"

[roles.expander]
provider = "local_expand"
max_tokens = 600
temperature = 0.7
top_k = 20
top_p = 0.8
min_p = 0.0
repeat_last_n = 64
repeat_penalty = 1.0
frequency_penalty = 0.0
presence_penalty = 0.5

[reaping]
days = 7    # hard-delete documents deactivated longer than this

[ranking]
deep_variant_rrf_k = 60
deep_variants_max = 4
initial_candidate_limit_min = 40
rerank_candidates_min = 20
rerank_candidates_max = 30

[ranking.hybrid_fusion]
mode = "dbsf" # linear | dbsf | rrf

[ranking.hybrid_fusion.linear]
dense_weight = 0.7
bm25_weight = 0.3

[ranking.hybrid_fusion.dbsf]
dense_weight = 1.0
bm25_weight = 0.4
stddevs = 3.0

[ranking.hybrid_fusion.rrf]
k = 60

[ranking.bm25_boosts]
title = 2.0
heading = 1.5
body = 1.0
filepath = 0.5

[chunking.defaults]
# Defaults are tuned for markdown-heavy collections.
target_tokens = 800
soft_max_tokens = 950
hard_max_tokens = 1200
boundary_overlap_tokens = 48
neighbor_window = 1
contextual_prefix = true

# Optional per-type override; only specify fields that differ.
[chunking.profiles.code]
target_tokens = 320
soft_max_tokens = 420
hard_max_tokens = 560
boundary_overlap_tokens = 24
```

Collection-level chunking overrides are deferred in V1. When implemented, they will be stored with collection metadata in SQLite and resolved between CLI overrides and profile/default settings.

Notes:
- Google embedding access through OpenAI-compatible endpoints is covered by `provider = "openai_compatible"` with the corresponding Google-compatible base URL.
- Native Google embeddings API support is intentionally deferred to V2.
- Local inference profiles point at externally managed `llama.cpp server` deployments; kbolt does not download or supervise those server processes in V1.
- Roles can bind to different deployments even when they share the same backend family, for example separate local embedding/reranking/chat endpoints or different remote OpenAI-compatible endpoints.
- The embedder path still uses EmbeddingGemma query/document normalization internally; the reranker path still uses the Qwen3-style reranker contract; the expander path still generates plain query variants as JSON strings. Those are role-adapter behaviors, not user-configurable local runtime ownership.

---

## CLI Commands

```
USAGE: kbolt <command> [options]

GLOBAL FLAGS
  -s, --space <name>               Set active space (overrides KBOLT_SPACE and default)
                                   Does not apply to `schedule`, which has its own subcommand-level scope flags
  -f, --format <fmt>               Output: cli|json (default: cli)
                                   Applies to all standard commands except `kbolt mcp`

SEARCH (one command, mode flags)
  kbolt search <query>              Smart hybrid search (routes by query type)
    -d, --deep                     Expand query + multi-variant + rerank
    -k, --keyword                  BM25 only
    --semantic                     Dense vector only
    --rerank                       Enable reranking in auto mode
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
  kbolt space add <name> [dirs...]  Create space, optionally register directories as collections
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
  kbolt ignore edit <collection>    Open ignore file in $VISUAL / $EDITOR / vi (supports flags)
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
  kbolt models list                 Show configured role bindings and provider readiness

EVALUATION
  kbolt eval run                    Run the default evaluation suite
    --file <path>                  Run a specific eval manifest instead of ~/.config/kbolt/eval.toml
  kbolt eval import beir
    --dataset <name>               Dataset identifier used in the report and default collection name
    --source <dir>                 Extracted canonical BEIR dataset directory
    --output <dir>                 Empty directory where corpus/ + eval.toml will be written
    --collection <name>            Override the collection name written into eval paths

SCHEDULING
  kbolt schedule add --every <interval>
                                   Set up recurring re-index (minutes/hours only, minimum 5m)
  kbolt schedule add --at <time>
                                   Run daily at a local time (`HH:MM`, `3pm`, `3:00pm`)
  kbolt schedule add --on <day,...> --at <time>
                                   Run weekly on selected weekdays
    --space <name>                 Scope to a space
    --collection <name>            Scope to one collection; repeat for more than one
  kbolt schedule status            Show saved schedules, backend state, run state, and orphans
  kbolt schedule remove <id>       Remove one schedule by short id (`s1`, `s2`, ...)
  kbolt schedule remove --all      Remove all schedules

ADMIN
  kbolt status                      Index health, space/collection stats, disk usage
```

Scheduling uses OS-level mechanisms: launchd jobs on macOS and systemd user timers on Linux. `kbolt schedule add/remove` reconciles those managed artifacts from `~/.config/kbolt/schedules.toml`, and `kbolt schedule status` reports drift or orphaned backend artifacts.

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

Deferred to V2: MCP resource URIs (`kbolt://{space}/{collection}/{path}`) are not exposed in V1. Document reads are available through the `get` and `multi_get` tools.

### Dynamic Instructions

Injected into LLM system prompt on connection:
- Number of indexed documents, spaces, and collections
- Available spaces with descriptions and their collections
- Search strategy guidance (when to use --deep vs default)

---

## Inference Deployments

| Role | Model | Format | Size | Runtime |
|---|---|---|---|---|
| Embedding | embeddinggemma 300M | GGUF Q8 | ~314MB | `llama.cpp server` |
| Reranker | Qwen3-Reranker 0.6B | GGUF Q8 | ~640MB | `llama.cpp server` |
| Expander | Qwen3 1.7B | GGUF Q8 | ~1.7GB | `llama.cpp server` |

In the provider-profile architecture, kbolt does not own local model artifact download or process supervision. Local inference is served by one or more externally managed `llama.cpp server` deployments, and kbolt binds roles to those deployments through `providers.*` + `roles.*`.

Remote inference uses `openai_compatible` provider profiles. Different roles may bind to different remote deployments without changing engine/search code.

`kbolt models list` reports configured role bindings plus endpoint readiness. Update/collection-add guidance now points users to configure role bindings or opt out of embedding when appropriate.

---

## Evaluation Framework

```
kbolt eval run
kbolt eval run --file /path/to/eval.toml
kbolt eval import beir --dataset fiqa --source /path/to/fiqa --output /path/to/fiqa-bench
```

Retrieval metrics: nDCG@10, Recall@10, MRR@10, latency (p50/p95).
Operational metrics such as update throughput and index size are tracked separately from the eval report.

Dataset in `~/.config/kbolt/eval.toml`:
```toml
[[cases]]
query = "how to handle errors in rust"
space = "personal"
collections = ["notes"]
judgments = [
  { path = "notes/error-handling.md", relevance = 1 },
]
```

The eval runner compares fixed retrieval modes:
- `keyword`
- `auto`
- `deep`
- `semantic` only when embeddings are configured

CLI output stays lean: per-mode summary metrics plus the queries that still need attention. JSON output returns the full structured report.

`kbolt eval import beir` imports the canonical BEIR test split from an extracted dataset directory. It expects:
- `corpus.jsonl`
- `queries.jsonl`
- `qrels/test.tsv`

The importer materializes:
- `corpus/{doc_id}.md`
- `eval.toml`

under the requested output directory, using the benchmark default `space = "bench"` and `collections = ["{dataset}"]` unless `--collection` overrides it. The command prints the next steps to register that corpus as a normal collection, run `kbolt update`, and then run `kbolt eval run --file ...`.

---

## Implementation Order

1. **types/** — shared request/response structs (spaces, collections, search/update/status)
2. **core/config** — TOML load/save + validation (provider profiles, role bindings, reaping, chunking, default space)
3. **core/storage** — SQLite schema + per-space Tantivy/USearch ownership and lifecycle
4. **core/ingest/extract** — extractor contracts + markdown/plaintext/code extractors
5. **core/ingest/chunk** — profile-aware chunking (packing, forced split, overlap, structural kinds)
6. **core/models (gateway)** — role binding resolution and readiness reporting
7. **core/models (provider clients + role adapters)** — embedder/reranker/expander builders over `llama.cpp server` and `openai_compatible`
8. **core/engine (facade)** — public API surface and shared operation lock orchestration
9. **core/engine/update_ops** — update pipeline + FTS replay + embedding backlog reconciliation
10. **core/engine/search_ops** — keyword/semantic/auto/deep retrieval + scoring + rerank/expand integration
11. **core/engine/ignore_ops** — collection ignore pattern CRUD
12. **cli/** — command mapping and human-facing UX
13. **mcp/** — stdio MCP protocol framing + tool handlers
14. **eval + distribution** — eval harness, CI/release artifacts, Homebrew/install script

---

## Verification Plan

**Unit tests**: Each extractor (typed block emission, heading breadcrumbs, source spans, edge cases), chunking (target/soft/hard budget behavior, block-specific forced-split rules, boundary overlap on forced splits only), storage CRUD (all tables including spaces, collections), ignore rules (hardcoded + internal ignore patterns + extensions), hybrid fusion scoring, query parsing, neighbor expansion logic, space resolution logic.

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
