pub mod protocol;
pub mod stdio;

use kbolt_core::engine::Engine;
use kbolt_core::{CoreError, Result};
use kbolt_types::{
    CollectionInfo, DocumentResponse, FileEntry, GetRequest, KboltError, Locator, ModelStatus,
    MultiGetRequest, MultiGetResponse, SearchMode, SearchRequest, SearchResponse, SpaceInfo,
    StatusResponse, UpdateOptions, UpdateReport,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

const DEFAULT_SEARCH_LIMIT: usize = 10;
const DEFAULT_MULTI_GET_MAX_FILES: usize = 20;
const DEFAULT_MULTI_GET_MAX_BYTES: usize = 50 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum McpToolCall {
    Search {
        query: String,
        space: Option<String>,
        collection: Option<String>,
        limit: Option<usize>,
        mode: Option<String>,
        no_rerank: Option<bool>,
    },
    Get {
        identifier: String,
        space: Option<String>,
    },
    MultiGet {
        locators: Vec<String>,
        space: Option<String>,
        max_files: Option<usize>,
        max_bytes: Option<usize>,
    },
    ListFiles {
        space: Option<String>,
        collection: String,
        prefix: Option<String>,
    },
    Status {
        space: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum McpToolResponse {
    Search(SearchResponse),
    Get(DocumentResponse),
    MultiGet(MultiGetResponse),
    ListFiles(Vec<FileEntry>),
    Status(StatusResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

pub struct McpAdapter {
    pub engine: Engine,
}

impl McpAdapter {
    pub fn new(engine: Engine) -> Self {
        Self { engine }
    }

    pub fn list_spaces(&self) -> Result<Vec<SpaceInfo>> {
        self.engine.list_spaces()
    }

    pub fn list_collections(&self, space: Option<&str>) -> Result<Vec<CollectionInfo>> {
        self.engine.list_collections(space)
    }

    pub fn update(&self, options: UpdateOptions) -> Result<UpdateReport> {
        self.engine.update(options)
    }

    pub fn list_files(
        &self,
        space: Option<&str>,
        collection: &str,
        prefix: Option<&str>,
    ) -> Result<Vec<FileEntry>> {
        self.engine.list_files(space, collection, prefix)
    }

    pub fn status(&self, space: Option<&str>) -> Result<StatusResponse> {
        self.engine.status(space)
    }

    pub fn get_document(&self, req: GetRequest) -> Result<DocumentResponse> {
        self.engine.get_document(req)
    }

    pub fn multi_get(&self, req: MultiGetRequest) -> Result<MultiGetResponse> {
        self.engine.multi_get(req)
    }

    pub fn model_status(&self) -> Result<ModelStatus> {
        self.engine.model_status()
    }

    pub fn search(&self, req: SearchRequest) -> Result<SearchResponse> {
        self.engine.search(req)
    }

    fn search_with_auto_fallback(&self, req: SearchRequest) -> Result<SearchResponse> {
        match self.search(req.clone()) {
            Ok(response) => Ok(response),
            Err(err) if should_fallback_to_keyword_search(&req, &err) => {
                let mut fallback = req;
                fallback.mode = SearchMode::Keyword;
                fallback.no_rerank = true;
                self.search(fallback)
            }
            Err(err) => Err(err),
        }
    }

    pub fn tool_definitions() -> Vec<McpToolDefinition> {
        vec![
            McpToolDefinition {
                name: "search".to_string(),
                description: "Search indexed content with optional space/collection filters."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "required": ["query"],
                    "additionalProperties": false,
                    "properties": {
                        "query": { "type": "string" },
                        "space": { "type": "string" },
                        "collection": { "type": "string" },
                        "limit": { "type": "integer", "minimum": 1 },
                        "mode": { "type": "string", "enum": ["auto", "deep", "keyword", "semantic"] },
                        "no_rerank": { "type": "boolean" }
                    }
                }),
            },
            McpToolDefinition {
                name: "get".to_string(),
                description: "Read one document by docid or collection-relative path.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "required": ["identifier"],
                    "additionalProperties": false,
                    "properties": {
                        "identifier": { "type": "string" },
                        "space": { "type": "string" }
                    }
                }),
            },
            McpToolDefinition {
                name: "multi_get".to_string(),
                description: "Read multiple documents with file-count and byte budgets."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "required": ["locators"],
                    "additionalProperties": false,
                    "properties": {
                        "locators": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "space": { "type": "string" },
                        "max_files": { "type": "integer", "minimum": 1 },
                        "max_bytes": { "type": "integer", "minimum": 1 }
                    }
                }),
            },
            McpToolDefinition {
                name: "list_files".to_string(),
                description: "List indexed files in a collection, optionally filtered by prefix."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "required": ["collection"],
                    "additionalProperties": false,
                    "properties": {
                        "space": { "type": "string" },
                        "collection": { "type": "string" },
                        "prefix": { "type": "string" }
                    }
                }),
            },
            McpToolDefinition {
                name: "status".to_string(),
                description: "Show index status and collection health.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "space": { "type": "string" }
                    }
                }),
            },
        ]
    }

    pub fn call_tool(&self, call: McpToolCall) -> Result<McpToolResponse> {
        match call {
            McpToolCall::Search {
                query,
                space,
                collection,
                limit,
                mode,
                no_rerank,
            } => {
                let mode = parse_tool_search_mode(mode.as_deref())?;
                let no_rerank = resolve_tool_no_rerank(&mode, no_rerank);
                let collections = match collection {
                    Some(name) => {
                        let trimmed = name.trim();
                        if trimmed.is_empty() {
                            return Err(KboltError::InvalidInput(
                                "collection cannot be empty".to_string(),
                            )
                            .into());
                        }
                        vec![trimmed.to_string()]
                    }
                    None => Vec::new(),
                };

                let response = self.search_with_auto_fallback(SearchRequest {
                    query,
                    mode,
                    space,
                    collections,
                    limit: limit.unwrap_or(DEFAULT_SEARCH_LIMIT),
                    min_score: 0.0,
                    no_rerank,
                    debug: false,
                })?;
                Ok(McpToolResponse::Search(response))
            }
            McpToolCall::Get { identifier, space } => {
                let response = self.get_document(GetRequest {
                    locator: Locator::parse(&identifier),
                    space,
                    offset: None,
                    limit: None,
                })?;
                Ok(McpToolResponse::Get(response))
            }
            McpToolCall::MultiGet {
                locators,
                space,
                max_files,
                max_bytes,
            } => {
                if locators.is_empty() {
                    return Err(
                        KboltError::InvalidInput("locators cannot be empty".to_string()).into(),
                    );
                }

                let response = self.multi_get(MultiGetRequest {
                    locators: locators
                        .iter()
                        .map(|item| Locator::parse(item))
                        .collect::<Vec<_>>(),
                    space,
                    max_files: max_files.unwrap_or(DEFAULT_MULTI_GET_MAX_FILES),
                    max_bytes: max_bytes.unwrap_or(DEFAULT_MULTI_GET_MAX_BYTES),
                })?;
                Ok(McpToolResponse::MultiGet(response))
            }
            McpToolCall::ListFiles {
                space,
                collection,
                prefix,
            } => {
                let collection = collection.trim();
                if collection.is_empty() {
                    return Err(
                        KboltError::InvalidInput("collection cannot be empty".to_string()).into(),
                    );
                }

                let response = self.list_files(space.as_deref(), collection, prefix.as_deref())?;
                Ok(McpToolResponse::ListFiles(response))
            }
            McpToolCall::Status { space } => {
                let response = self.status(space.as_deref())?;
                Ok(McpToolResponse::Status(response))
            }
        }
    }

    pub fn call_tool_json(&self, name: &str, args: Value) -> Result<Value> {
        let call = parse_tool_call_json(name, args)?;
        let response = self.call_tool(call)?;
        serialize_tool_response(response)
    }

    pub fn dynamic_instructions(&self) -> Result<String> {
        let status = self.status(None)?;
        let spaces = self.list_spaces()?;
        let mut total_collections = 0usize;
        let mut lines = Vec::new();

        lines.push("kbolt context:".to_string());
        lines.push(format!("- indexed_documents: {}", status.total_documents));
        lines.push(format!("- spaces: {}", spaces.len()));

        lines.push("spaces:".to_string());
        for space in spaces {
            let space_description = space
                .description
                .clone()
                .unwrap_or_else(|| "no description".to_string());
            lines.push(format!("- {}: {}", space.name, space_description));

            let collections = self.list_collections(Some(&space.name))?;
            total_collections += collections.len();
            if collections.is_empty() {
                lines.push("  - no collections".to_string());
                continue;
            }

            for collection in collections {
                let collection_description = collection
                    .description
                    .clone()
                    .unwrap_or_else(|| "no description".to_string());
                lines.push(format!(
                    "  - {}: {}",
                    collection.name, collection_description
                ));
            }
        }
        lines.insert(3, format!("- collections: {total_collections}"));

        lines.push("search_guidance:".to_string());
        lines.push("- Use mode \"auto\" first for most queries.".to_string());
        lines.push(
            "- Use mode \"deep\" when results are sparse or the query is broad/ambiguous."
                .to_string(),
        );

        Ok(lines.join("\n"))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchToolArgs {
    query: String,
    #[serde(default)]
    space: Option<String>,
    #[serde(default)]
    collection: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    no_rerank: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GetToolArgs {
    identifier: String,
    #[serde(default)]
    space: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MultiGetToolArgs {
    locators: Vec<String>,
    #[serde(default)]
    space: Option<String>,
    #[serde(default)]
    max_files: Option<usize>,
    #[serde(default)]
    max_bytes: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListFilesToolArgs {
    #[serde(default)]
    space: Option<String>,
    collection: String,
    #[serde(default)]
    prefix: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StatusToolArgs {
    #[serde(default)]
    space: Option<String>,
}

fn parse_tool_call_json(name: &str, args: Value) -> Result<McpToolCall> {
    match name {
        "search" => {
            let parsed: SearchToolArgs = parse_tool_args(name, args)?;
            Ok(McpToolCall::Search {
                query: parsed.query,
                space: parsed.space,
                collection: parsed.collection,
                limit: parsed.limit,
                mode: parsed.mode,
                no_rerank: parsed.no_rerank,
            })
        }
        "get" => {
            let parsed: GetToolArgs = parse_tool_args(name, args)?;
            Ok(McpToolCall::Get {
                identifier: parsed.identifier,
                space: parsed.space,
            })
        }
        "multi_get" => {
            let parsed: MultiGetToolArgs = parse_tool_args(name, args)?;
            Ok(McpToolCall::MultiGet {
                locators: parsed.locators,
                space: parsed.space,
                max_files: parsed.max_files,
                max_bytes: parsed.max_bytes,
            })
        }
        "list_files" => {
            let parsed: ListFilesToolArgs = parse_tool_args(name, args)?;
            Ok(McpToolCall::ListFiles {
                space: parsed.space,
                collection: parsed.collection,
                prefix: parsed.prefix,
            })
        }
        "status" => {
            let parsed: StatusToolArgs = parse_tool_args(name, args)?;
            Ok(McpToolCall::Status {
                space: parsed.space,
            })
        }
        _ => Err(KboltError::InvalidInput(format!("unknown tool: {name}")).into()),
    }
}

fn parse_tool_args<T>(tool: &str, args: Value) -> Result<T>
where
    T: DeserializeOwned,
{
    let normalized = if args.is_null() {
        Value::Object(Map::new())
    } else {
        args
    };

    serde_json::from_value(normalized).map_err(|err| {
        KboltError::InvalidInput(format!("invalid arguments for '{tool}': {err}")).into()
    })
}

fn serialize_tool_response(response: McpToolResponse) -> Result<Value> {
    match response {
        McpToolResponse::Search(data) => serialize_response_value(data),
        McpToolResponse::Get(data) => serialize_response_value(data),
        McpToolResponse::MultiGet(data) => serialize_response_value(data),
        McpToolResponse::ListFiles(data) => serialize_response_value(data),
        McpToolResponse::Status(data) => serialize_response_value(data),
    }
}

fn serialize_response_value<T>(response: T) -> Result<Value>
where
    T: Serialize,
{
    serde_json::to_value(response).map_err(|err| {
        KboltError::Internal(format!("failed to serialize tool response: {err}")).into()
    })
}

fn parse_tool_search_mode(raw_mode: Option<&str>) -> Result<SearchMode> {
    let Some(raw_mode) = raw_mode else {
        return Ok(SearchMode::Auto);
    };

    let normalized = raw_mode.trim().to_ascii_lowercase();
    let mode = match normalized.as_str() {
        "auto" => SearchMode::Auto,
        "deep" => SearchMode::Deep,
        "keyword" => SearchMode::Keyword,
        "semantic" => SearchMode::Semantic,
        _ => {
            return Err(KboltError::InvalidInput(
                "mode must be one of: auto, deep, keyword, semantic".to_string(),
            )
            .into())
        }
    };

    Ok(mode)
}

fn resolve_tool_no_rerank(mode: &SearchMode, no_rerank: Option<bool>) -> bool {
    match no_rerank {
        Some(value) => value,
        None => matches!(mode, SearchMode::Auto),
    }
}

fn should_fallback_to_keyword_search(req: &SearchRequest, err: &CoreError) -> bool {
    matches!(req.mode, SearchMode::Auto)
        && matches!(err, CoreError::Domain(KboltError::ModelNotAvailable { .. }))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::ffi::OsString;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    use kbolt_core::engine::Engine;
    use kbolt_types::{
        AddCollectionRequest, GetRequest, Locator, MultiGetRequest, SearchMode, SearchRequest,
        UpdateOptions,
    };
    use serde_json::json;
    use tempfile::tempdir;

    use super::{McpAdapter, McpToolCall, McpToolResponse};

    struct EnvRestore {
        home: Option<OsString>,
        config_home: Option<OsString>,
        cache_home: Option<OsString>,
    }

    impl EnvRestore {
        fn capture() -> Self {
            Self {
                home: std::env::var_os("HOME"),
                config_home: std::env::var_os("XDG_CONFIG_HOME"),
                cache_home: std::env::var_os("XDG_CACHE_HOME"),
            }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match &self.home {
                Some(path) => std::env::set_var("HOME", path),
                None => std::env::remove_var("HOME"),
            }
            match &self.config_home {
                Some(path) => std::env::set_var("XDG_CONFIG_HOME", path),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
            match &self.cache_home {
                Some(path) => std::env::set_var("XDG_CACHE_HOME", path),
                None => std::env::remove_var("XDG_CACHE_HOME"),
            }
        }
    }

    fn with_isolated_xdg_dirs<T>(run: impl FnOnce() -> T) -> T {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let _restore = EnvRestore::capture();

        let root = tempdir().expect("create temp root");
        std::env::set_var("HOME", root.path());
        let config_home = root.path().join("config-home");
        let cache_home = root.path().join("cache-home");
        std::env::set_var("XDG_CONFIG_HOME", &config_home);
        std::env::set_var("XDG_CACHE_HOME", &cache_home);

        run()
    }

    fn new_collection_dir(root: &Path, name: &str) -> PathBuf {
        let path = root.join(name);
        fs::create_dir_all(&path).expect("create collection directory");
        path
    }

    #[test]
    fn list_spaces_and_collections_delegate_to_engine() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            engine.add_space("notes", None).expect("add notes");

            let work_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");

            let adapter = McpAdapter::new(engine);
            let spaces = adapter.list_spaces().expect("list spaces");
            assert!(
                spaces.iter().any(|space| space.name == "work"),
                "expected work in spaces"
            );
            assert!(
                spaces.iter().any(|space| space.name == "notes"),
                "expected notes in spaces"
            );

            let collections = adapter
                .list_collections(Some("work"))
                .expect("list work collections");
            assert_eq!(collections.len(), 1);
            assert_eq!(collections[0].name, "api");
            assert_eq!(collections[0].space, "work");
        });
    }

    #[test]
    fn update_status_and_list_files_are_scoped() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            engine.add_space("notes", None).expect("add notes");

            let work_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");

            let notes_path = new_collection_dir(root.path(), "notes-wiki");
            engine
                .add_collection(AddCollectionRequest {
                    path: notes_path.clone(),
                    space: Some("notes".to_string()),
                    name: Some("wiki".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add notes collection");

            fs::create_dir_all(work_path.join("src")).expect("create work src");
            fs::write(work_path.join("src/lib.rs"), "fn alpha() {}\n").expect("write work file");
            fs::write(notes_path.join("guide.md"), "notes guide\n").expect("write notes file");

            let adapter = McpAdapter::new(engine);
            let report = adapter
                .update(UpdateOptions {
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    no_embed: true,
                    dry_run: false,
                    verbose: false,
                })
                .expect("update work collection");
            assert_eq!(report.added, 1);

            let files = adapter
                .list_files(Some("work"), "api", Some("src"))
                .expect("list files");
            assert_eq!(files.len(), 1);
            assert_eq!(files[0].path, "src/lib.rs");

            let status = adapter.status(Some("work")).expect("status work");
            assert_eq!(status.spaces.len(), 1);
            assert_eq!(status.spaces[0].name, "work");
            assert_eq!(status.total_documents, 1);
        });
    }

    #[test]
    fn get_document_wrapper_returns_content() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let work_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");

            fs::create_dir_all(work_path.join("src")).expect("create src dir");
            fs::write(work_path.join("src/lib.rs"), "line-a\nline-b\n").expect("write file");

            let adapter = McpAdapter::new(engine);
            adapter
                .update(UpdateOptions {
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    no_embed: true,
                    dry_run: false,
                    verbose: false,
                })
                .expect("run update");

            let document = adapter
                .get_document(GetRequest {
                    locator: Locator::Path("api/src/lib.rs".to_string()),
                    space: Some("work".to_string()),
                    offset: Some(1),
                    limit: Some(1),
                })
                .expect("get document");
            assert_eq!(document.path, "api/src/lib.rs");
            assert_eq!(document.content, "line-b");
            assert_eq!(document.returned_lines, 1);
        });
    }

    #[test]
    fn multi_get_wrapper_respects_budgets() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let work_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");

            fs::write(work_path.join("a.md"), "alpha\n").expect("write a");
            fs::write(work_path.join("b.md"), "beta\n").expect("write b");
            fs::write(work_path.join("c.md"), "gamma\n").expect("write c");

            let adapter = McpAdapter::new(engine);
            adapter
                .update(UpdateOptions {
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    no_embed: true,
                    dry_run: false,
                    verbose: false,
                })
                .expect("run update");

            let response = adapter
                .multi_get(MultiGetRequest {
                    locators: vec![
                        Locator::Path("api/a.md".to_string()),
                        Locator::Path("api/b.md".to_string()),
                        Locator::Path("api/c.md".to_string()),
                    ],
                    space: Some("work".to_string()),
                    max_files: 2,
                    max_bytes: 51_200,
                })
                .expect("run multi_get");
            assert_eq!(response.documents.len(), 2);
            assert_eq!(response.omitted.len(), 1);
            assert_eq!(response.resolved_count, 3);
        });
    }

    #[test]
    fn multi_get_wrapper_reports_deleted_files_as_warnings() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let work_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");

            let existing = work_path.join("a.md");
            let deleted = work_path.join("b.md");
            fs::write(&existing, "alpha\n").expect("write a");
            fs::write(&deleted, "beta\n").expect("write b");

            let adapter = McpAdapter::new(engine);
            adapter
                .update(UpdateOptions {
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    no_embed: true,
                    dry_run: false,
                    verbose: false,
                })
                .expect("run update");

            fs::remove_file(&deleted).expect("remove b");

            let response = adapter
                .multi_get(MultiGetRequest {
                    locators: vec![
                        Locator::Path("api/a.md".to_string()),
                        Locator::Path("api/b.md".to_string()),
                    ],
                    space: Some("work".to_string()),
                    max_files: 10,
                    max_bytes: 51_200,
                })
                .expect("run multi_get");
            assert_eq!(response.documents.len(), 1);
            assert!(response.omitted.is_empty());
            assert_eq!(response.resolved_count, 1);
            assert_eq!(response.warnings.len(), 1);
            assert!(response.warnings[0].contains("file deleted since indexing:"));
            assert!(response.warnings[0].contains("b.md"));
        });
    }

    #[test]
    fn model_status_wrapper_exposes_configured_models() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);

            let status = adapter.model_status().expect("read model status");
            assert!(!status.embedder.name.is_empty());
            assert!(!status.reranker.name.is_empty());
            assert!(!status.expander.name.is_empty());
            assert!(!status.embedder.downloaded);
            assert!(!status.reranker.downloaded);
            assert!(!status.expander.downloaded);
        });
    }

    #[test]
    fn search_wrapper_returns_keyword_results() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let work_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");
            fs::write(work_path.join("a.md"), "search-token\n").expect("write file");

            let adapter = McpAdapter::new(engine);
            adapter
                .update(UpdateOptions {
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    no_embed: true,
                    dry_run: false,
                    verbose: false,
                })
                .expect("run update");

            let response = adapter
                .search(SearchRequest {
                    query: "search-token".to_string(),
                    mode: SearchMode::Keyword,
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    limit: 5,
                    min_score: 0.0,
                    no_rerank: false,
                    debug: false,
                })
                .expect("run search");
            assert_eq!(response.mode, SearchMode::Keyword);
            assert_eq!(response.query, "search-token");
            assert_eq!(response.results.len(), 1);
            assert_eq!(response.results[0].space, "work");
        });
    }

    #[test]
    fn call_tool_dispatches_search_with_defaults() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let work_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");
            fs::write(work_path.join("a.md"), "search-token\n").expect("write file");

            let adapter = McpAdapter::new(engine);
            adapter
                .update(UpdateOptions {
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    no_embed: true,
                    dry_run: false,
                    verbose: false,
                })
                .expect("run update");

            let response = adapter
                .call_tool(McpToolCall::Search {
                    query: "search-token".to_string(),
                    space: Some("work".to_string()),
                    collection: Some("api".to_string()),
                    limit: None,
                    mode: None,
                    no_rerank: None,
                })
                .expect("run tool search");

            match response {
                McpToolResponse::Search(search) => {
                    assert_eq!(search.mode, SearchMode::Keyword);
                    assert_eq!(search.query, "search-token");
                    assert_eq!(search.results.len(), 1);
                    assert_eq!(search.results[0].space, "work");
                }
                other => panic!("unexpected response: {other:?}"),
            }
        });
    }

    #[test]
    fn call_tool_get_accepts_bare_docid() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let work_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");
            fs::write(work_path.join("a.md"), "search-token\n").expect("write file");

            let adapter = McpAdapter::new(engine);
            adapter
                .update(UpdateOptions {
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    no_embed: true,
                    dry_run: false,
                    verbose: false,
                })
                .expect("run update");

            let search = adapter
                .search(SearchRequest {
                    query: "search-token".to_string(),
                    mode: SearchMode::Keyword,
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    limit: 5,
                    min_score: 0.0,
                    no_rerank: false,
                    debug: false,
                })
                .expect("run search");
            let bare_docid = search.results[0].docid.trim_start_matches('#').to_string();

            let response = adapter
                .call_tool(McpToolCall::Get {
                    identifier: bare_docid,
                    space: Some("work".to_string()),
                })
                .expect("run tool get");

            match response {
                McpToolResponse::Get(document) => {
                    assert_eq!(document.path, "api/a.md");
                    assert_eq!(document.space, "work");
                }
                other => panic!("unexpected response: {other:?}"),
            }
        });
    }

    #[test]
    fn call_tool_rejects_invalid_search_mode() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);

            let err = adapter
                .call_tool(McpToolCall::Search {
                    query: "alpha".to_string(),
                    space: None,
                    collection: None,
                    limit: None,
                    mode: Some("invalid".to_string()),
                    no_rerank: None,
                })
                .expect_err("invalid mode should fail");
            assert!(
                err.to_string()
                    .contains("mode must be one of: auto, deep, keyword, semantic"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn call_tool_json_search_returns_structured_data() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let work_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");
            fs::write(work_path.join("a.md"), "search-token\n").expect("write file");

            let adapter = McpAdapter::new(engine);
            adapter
                .update(UpdateOptions {
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    no_embed: true,
                    dry_run: false,
                    verbose: false,
                })
                .expect("run update");

            let response = adapter
                .call_tool_json(
                    "search",
                    json!({
                        "query": "search-token",
                        "space": "work",
                        "collection": "api"
                    }),
                )
                .expect("run search via json bridge");

            assert_eq!(response["query"], "search-token");
            assert_eq!(response["mode"], "Keyword");
            let results = response["results"]
                .as_array()
                .expect("results should be an array");
            assert_eq!(results.len(), 1);
            assert_eq!(results[0]["space"], "work");
            assert_eq!(results[0]["path"], "api/a.md");
        });
    }

    #[test]
    fn parse_tool_call_json_search_accepts_no_rerank_flag() {
        let parsed = super::parse_tool_call_json(
            "search",
            json!({
                "query": "alpha",
                "no_rerank": true
            }),
        )
        .expect("parse search args");

        assert_eq!(
            parsed,
            McpToolCall::Search {
                query: "alpha".to_string(),
                space: None,
                collection: None,
                limit: None,
                mode: None,
                no_rerank: Some(true),
            }
        );
    }

    #[test]
    fn resolve_tool_no_rerank_defaults_auto_to_true() {
        assert!(super::resolve_tool_no_rerank(&SearchMode::Auto, None));
        assert!(!super::resolve_tool_no_rerank(&SearchMode::Deep, None));
        assert!(!super::resolve_tool_no_rerank(&SearchMode::Keyword, None));
        assert!(!super::resolve_tool_no_rerank(&SearchMode::Semantic, None));
    }

    #[test]
    fn resolve_tool_no_rerank_honors_explicit_override() {
        assert!(!super::resolve_tool_no_rerank(
            &SearchMode::Auto,
            Some(false)
        ));
        assert!(super::resolve_tool_no_rerank(&SearchMode::Deep, Some(true)));
    }

    #[test]
    fn call_tool_json_rejects_unknown_tool() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);

            let err = adapter
                .call_tool_json("unknown_tool", json!({}))
                .expect_err("unknown tool should fail");
            assert!(
                err.to_string().contains("unknown tool"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn call_tool_json_rejects_invalid_arguments() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);

            let err = adapter
                .call_tool_json("search", json!({ "query": "alpha", "limit": "five" }))
                .expect_err("invalid arguments should fail");
            assert!(
                err.to_string().contains("invalid arguments for 'search'"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn dynamic_instructions_include_space_and_collection_descriptions() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine
                .add_space("work", Some("work documents"))
                .expect("add work");

            let work_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: Some("api reference".to_string()),
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");

            let adapter = McpAdapter::new(engine);
            let output = adapter
                .dynamic_instructions()
                .expect("build dynamic instructions");

            assert!(output.contains("kbolt context:"));
            assert!(output.contains("work documents"));
            assert!(output.contains("api reference"));
            assert!(output.contains("search_guidance:"));
            assert!(output.contains("mode \"deep\""));
        });
    }

    #[test]
    fn tool_definitions_match_spec_tools() {
        let tools = McpAdapter::tool_definitions();
        assert_eq!(tools.len(), 5);

        let names = tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<HashSet<_>>();
        assert!(names.contains("search"));
        assert!(names.contains("get"));
        assert!(names.contains("multi_get"));
        assert!(names.contains("list_files"));
        assert!(names.contains("status"));

        let search_tool = tools
            .iter()
            .find(|tool| tool.name == "search")
            .expect("search tool should exist");
        let required = search_tool.input_schema["required"]
            .as_array()
            .expect("search required should be array");
        assert_eq!(required[0], "query");
    }
}
