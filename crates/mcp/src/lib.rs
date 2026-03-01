use kbolt_core::engine::Engine;
use kbolt_core::Result;
use kbolt_types::{
    CollectionInfo, DocumentResponse, FileEntry, GetRequest, SpaceInfo, StatusResponse,
    UpdateOptions, UpdateReport,
};

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
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};

    use kbolt_core::engine::Engine;
    use kbolt_types::{AddCollectionRequest, GetRequest, Locator, UpdateOptions};
    use tempfile::tempdir;

    use super::McpAdapter;

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
}
