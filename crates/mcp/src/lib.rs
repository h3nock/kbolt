use kbolt_core::engine::Engine;
use kbolt_core::Result;
use kbolt_types::{
    CollectionInfo, DocumentResponse, FileEntry, GetRequest, KboltError, Locator, ModelStatus,
    MultiGetRequest, MultiGetResponse, SearchMode, SearchRequest, SearchResponse, SpaceInfo,
    StatusResponse, UpdateOptions, UpdateReport,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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

    pub fn call_tool(&self, call: McpToolCall) -> Result<McpToolResponse> {
        match call {
            McpToolCall::Search {
                query,
                space,
                collection,
                limit,
                mode,
            } => {
                let mode = parse_tool_search_mode(mode.as_deref())?;
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

                let response = self.search(SearchRequest {
                    query,
                    mode,
                    space,
                    collections,
                    limit: limit.unwrap_or(DEFAULT_SEARCH_LIMIT),
                    min_score: 0.0,
                    no_rerank: false,
                    debug: false,
                })?;
                Ok(McpToolResponse::Search(response))
            }
            McpToolCall::Get { identifier, space } => {
                let response = self.get_document(GetRequest {
                    locator: parse_tool_locator(&identifier),
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
                        .map(|item| parse_tool_locator(item))
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

fn parse_tool_locator(raw: &str) -> Locator {
    let trimmed = raw.trim();
    if trimmed.contains('/') {
        return Locator::Path(trimmed.to_string());
    }

    Locator::DocId(trimmed.trim_start_matches('#').to_string())
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::fs;
    use std::path::PathBuf;
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
        let _guard = lock.lock().expect("lock env mutex");
        let _restore = EnvRestore::capture();

        let root = tempdir().expect("create temp root");
        std::env::set_var("HOME", root.path());
        let config_home = root.path().join("config-home");
        let cache_home = root.path().join("cache-home");
        std::env::set_var("XDG_CONFIG_HOME", &config_home);
        std::env::set_var("XDG_CACHE_HOME", &cache_home);

        run()
    }

    fn new_collection_dir(root: &PathBuf, name: &str) -> PathBuf {
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

            let work_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
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

            let work_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
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

            let notes_path = new_collection_dir(&root.path().to_path_buf(), "notes-wiki");
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

            let work_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
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

            let work_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
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
            assert_eq!(response.resolved_count, 2);
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

            let work_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
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

            let work_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
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

            let work_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
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

            let work_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
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
}
