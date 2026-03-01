pub mod args;

use kbolt_core::engine::Engine;
use kbolt_core::Result;
use kbolt_types::{
    ActiveSpaceSource, AddCollectionRequest, GetRequest, KboltError, Locator, MultiGetRequest,
    OmitReason, SearchMode, SearchRequest, UpdateOptions,
};

pub struct CliAdapter {
    pub engine: Engine,
}

impl CliAdapter {
    pub fn new(engine: Engine) -> Self {
        Self { engine }
    }

    pub fn space_add(
        &self,
        name: &str,
        description: Option<&str>,
        strict: bool,
        dirs: &[std::path::PathBuf],
    ) -> Result<String> {
        if strict {
            use std::collections::HashSet;

            let mut validation_errors = Vec::new();
            let mut derived_names = HashSet::new();
            for dir in dirs {
                if !dir.is_absolute() || !dir.is_dir() {
                    validation_errors.push(format!("- {} -> invalid path", dir.display()));
                    continue;
                }

                let collection_name = dir.file_name().and_then(|item| item.to_str());
                match collection_name {
                    Some(name) => {
                        if !derived_names.insert(name.to_string()) {
                            validation_errors.push(format!(
                                "- {} -> duplicate derived collection name '{name}'",
                                dir.display()
                            ));
                        }
                    }
                    None => validation_errors.push(format!(
                        "- {} -> cannot derive collection name from path",
                        dir.display()
                    )),
                }
            }

            if !validation_errors.is_empty() {
                let mut lines = Vec::new();
                lines.push(
                    "strict mode aborted: one or more directories are invalid".to_string(),
                );
                lines.extend(validation_errors);
                return Err(kbolt_types::KboltError::InvalidInput(lines.join("\n")).into());
            }
        }

        let added = self.engine.add_space(name, description)?;
        let description = added.description.unwrap_or_default();
        let suffix = if description.is_empty() {
            String::new()
        } else {
            format!(" - {description}")
        };

        if dirs.is_empty() {
            return Ok(format!("space added: {}{suffix}", added.name));
        }

        let mut successes = Vec::new();
        let mut failures = Vec::new();
        for dir in dirs {
            let collection_name = dir
                .file_name()
                .and_then(|item| item.to_str())
                .map(ToString::to_string);

            let result = self.engine.add_collection(AddCollectionRequest {
                path: dir.clone(),
                space: Some(name.to_string()),
                name: collection_name,
                description: None,
                extensions: None,
                no_index: true,
            });

            match result {
                Ok(info) => successes.push(format!(
                    "- {} -> {}/{}",
                    dir.display(),
                    info.space,
                    info.name
                )),
                Err(err) => {
                    if strict {
                        let rollback_result = self.engine.remove_space(name);
                        return match rollback_result {
                            Ok(()) => Err(err),
                            Err(rollback_err) => Err(kbolt_types::KboltError::Internal(format!(
                                "strict mode rollback failed: add error: {err}; rollback error: {rollback_err}"
                            ))
                            .into()),
                        };
                    }
                    failures.push(format!("- {} -> {}", dir.display(), err));
                }
            }
        }

        let mut lines = Vec::new();
        lines.push(format!("space added: {}{suffix}", added.name));
        lines.push(format!("collections added: {}", successes.len()));
        lines.extend(successes);
        if !failures.is_empty() {
            lines.push(format!("collections failed: {}", failures.len()));
            lines.extend(failures);
        }
        lines.push(
            "note: collections were registered without indexing; run `kbolt update` to index them"
                .to_string(),
        );

        Ok(lines.join("\n"))
    }

    pub fn space_describe(&self, name: &str, text: &str) -> Result<String> {
        self.engine.describe_space(name, text)?;
        Ok(format!("space description updated: {name}"))
    }

    pub fn space_rename(&self, old: &str, new: &str) -> Result<String> {
        self.engine.rename_space(old, new)?;
        Ok(format!("space renamed: {old} -> {new}"))
    }

    pub fn space_remove(&self, name: &str) -> Result<String> {
        self.engine.remove_space(name)?;
        if name == "default" {
            return Ok("default space cleared".to_string());
        }
        Ok(format!("space removed: {name}"))
    }

    pub fn space_default(&mut self, name: Option<&str>) -> Result<String> {
        if let Some(space_name) = name {
            let updated = self.engine.set_default_space(Some(space_name))?;
            let value = updated.unwrap_or_default();
            return Ok(format!("default space: {value}"));
        }

        let current = self.engine.config().default_space.as_deref();
        let output = match current {
            Some(value) => format!("default space: {value}"),
            None => "default space: none".to_string(),
        };
        Ok(output)
    }

    pub fn space_current(&self, explicit: Option<&str>) -> Result<String> {
        let active = self.engine.current_space(explicit)?;
        let output = match active {
            Some(active) => {
                let source = match active.source {
                    ActiveSpaceSource::Flag => "flag",
                    ActiveSpaceSource::EnvVar => "env",
                    ActiveSpaceSource::ConfigDefault => "default",
                };
                format!("active space: {} ({source})", active.name)
            }
            None => "active space: none".to_string(),
        };
        Ok(output)
    }

    pub fn space_list(&self) -> Result<String> {
        let spaces = self.engine.list_spaces()?;
        let mut lines = Vec::with_capacity(spaces.len() + 1);
        lines.push("spaces:".to_string());
        for space in spaces {
            let description = space.description.unwrap_or_default();
            let suffix = if description.is_empty() {
                String::new()
            } else {
                format!(" - {description}")
            };
            lines.push(format!(
                "- {} (collections: {}, documents: {}, chunks: {}){}",
                space.name, space.collection_count, space.document_count, space.chunk_count, suffix
            ));
        }
        Ok(lines.join("\n"))
    }

    pub fn space_info(&self, name: &str) -> Result<String> {
        let space = self.engine.space_info(name)?;
        let description = space.description.unwrap_or_default();
        let description_line = if description.is_empty() {
            "description:".to_string()
        } else {
            format!("description: {description}")
        };

        Ok(format!(
            "name: {}\n{description_line}\ncollections: {}\ndocuments: {}\nchunks: {}\ncreated: {}",
            space.name, space.collection_count, space.document_count, space.chunk_count, space.created
        ))
    }

    pub fn collection_list(&self, space: Option<&str>) -> Result<String> {
        let collections = self.engine.list_collections(space)?;
        let mut lines = Vec::with_capacity(collections.len() + 1);
        lines.push("collections:".to_string());
        if collections.is_empty() {
            lines.push("- none".to_string());
            return Ok(lines.join("\n"));
        }

        for collection in collections {
            lines.push(format!(
                "- {}/{} ({})",
                collection.space,
                collection.name,
                collection.path.display()
            ));
        }
        Ok(lines.join("\n"))
    }

    pub fn collection_add(
        &self,
        space: Option<&str>,
        path: &std::path::Path,
        name: Option<&str>,
        description: Option<&str>,
        extensions: Option<&[String]>,
        no_index: bool,
    ) -> Result<String> {
        let added = self.engine.add_collection(AddCollectionRequest {
            path: path.to_path_buf(),
            space: space.map(ToString::to_string),
            name: name.map(ToString::to_string),
            description: description.map(ToString::to_string),
            extensions: extensions.map(|items| items.to_vec()),
            no_index,
        })?;

        Ok(format!("collection added: {}/{}", added.space, added.name))
    }

    pub fn collection_info(&self, space: Option<&str>, name: &str) -> Result<String> {
        let collection = self.engine.collection_info(space, name)?;
        let description = collection.description.unwrap_or_default();
        let extensions = collection
            .extensions
            .map(|items| items.join(","))
            .unwrap_or_default();
        let description_line = if description.is_empty() {
            "description:".to_string()
        } else {
            format!("description: {description}")
        };
        let extensions_line = if extensions.is_empty() {
            "extensions:".to_string()
        } else {
            format!("extensions: {extensions}")
        };

        Ok(format!(
            "name: {}\nspace: {}\npath: {}\n{description_line}\n{extensions_line}\ndocuments: {}\nactive_documents: {}\nchunks: {}\nembedded_chunks: {}\ncreated: {}\nupdated: {}",
            collection.name,
            collection.space,
            collection.path.display(),
            collection.document_count,
            collection.active_document_count,
            collection.chunk_count,
            collection.embedded_chunk_count,
            collection.created,
            collection.updated
        ))
    }

    pub fn collection_describe(&self, space: Option<&str>, name: &str, text: &str) -> Result<String> {
        self.engine.describe_collection(space, name, text)?;
        Ok(format!("collection description updated: {name}"))
    }

    pub fn collection_rename(&self, space: Option<&str>, old: &str, new: &str) -> Result<String> {
        self.engine.rename_collection(space, old, new)?;
        Ok(format!("collection renamed: {old} -> {new}"))
    }

    pub fn collection_remove(&self, space: Option<&str>, name: &str) -> Result<String> {
        self.engine.remove_collection(space, name)?;
        Ok(format!("collection removed: {name}"))
    }

    pub fn models_list(&self) -> Result<String> {
        let status = self.engine.model_status()?;
        let mut lines = Vec::new();
        lines.push("models:".to_string());

        for (label, info) in [
            ("embedder", status.embedder),
            ("reranker", status.reranker),
            ("expander", status.expander),
        ] {
            let availability = if info.downloaded { "downloaded" } else { "missing" };
            let mut line = format!("- {label}: {} ({availability})", info.name);
            if let Some(size_bytes) = info.size_bytes {
                line.push_str(&format!(", size_bytes: {size_bytes}"));
            }
            if let Some(path) = info.path {
                line.push_str(&format!(", path: {}", path.display()));
            }
            lines.push(line);
        }

        Ok(lines.join("\n"))
    }

    pub fn models_pull(&self) -> Result<String> {
        let report = self.engine.pull_models()?;
        let mut lines = Vec::new();
        lines.push(format!("downloaded: {}", report.downloaded.len()));
        lines.push("downloaded_models:".to_string());
        if report.downloaded.is_empty() {
            lines.push("- none".to_string());
        } else {
            for model in report.downloaded {
                lines.push(format!("- {model}"));
            }
        }

        lines.push(format!("already_present: {}", report.already_present.len()));
        lines.push("already_present_models:".to_string());
        if report.already_present.is_empty() {
            lines.push("- none".to_string());
        } else {
            for model in report.already_present {
                lines.push(format!("- {model}"));
            }
        }

        lines.push(format!("total_bytes: {}", report.total_bytes));
        Ok(lines.join("\n"))
    }

    pub fn search(
        &self,
        space: Option<&str>,
        query: &str,
        collections: &[String],
        limit: usize,
        min_score: f32,
        deep: bool,
        keyword: bool,
        semantic: bool,
        no_rerank: bool,
        debug: bool,
    ) -> Result<String> {
        let mode_flags = deep as u8 + keyword as u8 + semantic as u8;
        if mode_flags > 1 {
            return Err(
                KboltError::InvalidInput(
                    "only one of --deep, --keyword, or --semantic can be set".to_string(),
                )
                .into(),
            );
        }

        let mode = if deep {
            SearchMode::Deep
        } else if keyword {
            SearchMode::Keyword
        } else if semantic {
            SearchMode::Semantic
        } else {
            SearchMode::Auto
        };

        let response = self.engine.search(SearchRequest {
            query: query.to_string(),
            mode,
            space: space.map(ToString::to_string),
            collections: collections.to_vec(),
            limit,
            min_score,
            no_rerank,
            debug,
        })?;

        let mut lines = Vec::new();
        lines.push(format!("query: {}", response.query));
        lines.push(format!("mode: {}", format_search_mode(&response.mode)));
        lines.push(format!("results: {}", response.results.len()));
        for (index, item) in response.results.iter().enumerate() {
            lines.push(format!(
                "{}. {} {} score={:.3}",
                index + 1,
                item.docid,
                item.path,
                item.score
            ));
            lines.push(format!("title: {}", item.title));
            lines.push(format!("space: {} | collection: {}", item.space, item.collection));
            if let Some(heading) = &item.heading {
                lines.push(format!("heading: {heading}"));
            }
            lines.push(format!("text: {}", item.text));
            if let Some(signals) = &item.signals {
                lines.push(format!(
                    "signals: bm25={:?} dense={:?} rrf={:.3} reranker={:?}",
                    signals.bm25, signals.dense, signals.rrf, signals.reranker
                ));
            }
        }
        if let Some(hint) = response.staleness_hint {
            lines.push(hint);
        }
        lines.push(format!("elapsed_ms: {}", response.elapsed_ms));
        Ok(lines.join("\n"))
    }

    pub fn update(
        &self,
        space: Option<&str>,
        collections: &[String],
        no_embed: bool,
        dry_run: bool,
        verbose: bool,
    ) -> Result<String> {
        let report = self.engine.update(UpdateOptions {
            space: space.map(ToString::to_string),
            collections: collections.to_vec(),
            no_embed,
            dry_run,
            verbose,
        })?;

        let mut lines = Vec::new();
        lines.push(format!("scanned: {}", report.scanned));
        lines.push(format!("skipped_mtime: {}", report.skipped_mtime));
        lines.push(format!("skipped_hash: {}", report.skipped_hash));
        lines.push(format!("added: {}", report.added));
        lines.push(format!("updated: {}", report.updated));
        lines.push(format!("deactivated: {}", report.deactivated));
        lines.push(format!("reactivated: {}", report.reactivated));
        lines.push(format!("reaped: {}", report.reaped));
        lines.push(format!("embedded: {}", report.embedded));
        lines.push(format!("errors: {}", report.errors.len()));
        if verbose {
            for error in report.errors {
                lines.push(format!("- {}: {}", error.path, error.error));
            }
        }
        lines.push(format!("elapsed_ms: {}", report.elapsed_ms));
        Ok(lines.join("\n"))
    }

    pub fn status(&self, space: Option<&str>) -> Result<String> {
        let status = self.engine.status(space)?;
        let mut lines = Vec::new();
        lines.push("spaces:".to_string());
        if status.spaces.is_empty() {
            lines.push("- none".to_string());
        } else {
            for space in status.spaces {
                let collection_count = space.collections.len();
                let description = space.description.unwrap_or_default();
                let description_suffix = if description.is_empty() {
                    String::new()
                } else {
                    format!(" - {description}")
                };
                let last_updated = space
                    .last_updated
                    .as_deref()
                    .map(|value| format!(", last_updated: {value}"))
                    .unwrap_or_default();
                lines.push(format!(
                    "- {} (collections: {}{}){}",
                    space.name, collection_count, last_updated, description_suffix
                ));

                for collection in space.collections {
                    lines.push(format!(
                        "  - {} ({}) documents: {}, active: {}, chunks: {}, embedded: {}, last_updated: {}",
                        collection.name,
                        collection.path.display(),
                        collection.documents,
                        collection.active_documents,
                        collection.chunks,
                        collection.embedded_chunks,
                        collection.last_updated
                    ));
                }
            }
        }

        lines.push(format!("total_documents: {}", status.total_documents));
        lines.push(format!("total_chunks: {}", status.total_chunks));
        lines.push(format!("total_embedded: {}", status.total_embedded));
        lines.push(format!("sqlite_bytes: {}", status.disk_usage.sqlite_bytes));
        lines.push(format!("tantivy_bytes: {}", status.disk_usage.tantivy_bytes));
        lines.push(format!("usearch_bytes: {}", status.disk_usage.usearch_bytes));
        lines.push(format!("models_bytes: {}", status.disk_usage.models_bytes));
        lines.push(format!("total_bytes: {}", status.disk_usage.total_bytes));
        lines.push(format!(
            "model_embedder: {} ({})",
            status.models.embedder.name,
            if status.models.embedder.downloaded {
                "downloaded"
            } else {
                "missing"
            }
        ));
        lines.push(format!(
            "model_reranker: {} ({})",
            status.models.reranker.name,
            if status.models.reranker.downloaded {
                "downloaded"
            } else {
                "missing"
            }
        ));
        lines.push(format!(
            "model_expander: {} ({})",
            status.models.expander.name,
            if status.models.expander.downloaded {
                "downloaded"
            } else {
                "missing"
            }
        ));
        lines.push(format!("cache_dir: {}", status.cache_dir.display()));
        lines.push(format!("config_dir: {}", status.config_dir.display()));

        Ok(lines.join("\n"))
    }

    pub fn ls(
        &self,
        space: Option<&str>,
        collection: &str,
        prefix: Option<&str>,
        all: bool,
    ) -> Result<String> {
        let mut files = self.engine.list_files(space, collection, prefix)?;
        if !all {
            files.retain(|file| file.active);
        }

        let mut lines = Vec::new();
        lines.push("files:".to_string());
        if files.is_empty() {
            lines.push("- none".to_string());
            return Ok(lines.join("\n"));
        }

        for file in files {
            if all {
                lines.push(format!(
                    "- {} | {} | {} | active: {}",
                    file.path, file.title, file.docid, file.active
                ));
            } else {
                lines.push(format!("- {} | {} | {}", file.path, file.title, file.docid));
            }
        }

        Ok(lines.join("\n"))
    }

    pub fn get(
        &self,
        space: Option<&str>,
        identifier: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<String> {
        let locator = parse_cli_locator(identifier);

        let document = self.engine.get_document(GetRequest {
            locator,
            space: space.map(ToString::to_string),
            offset,
            limit,
        })?;

        Ok(format!(
            "docid: {}\npath: {}\ntitle: {}\nspace: {}\ncollection: {}\nstale: {}\ntotal_lines: {}\nreturned_lines: {}\ncontent:\n{}",
            document.docid,
            document.path,
            document.title,
            document.space,
            document.collection,
            document.stale,
            document.total_lines,
            document.returned_lines,
            document.content
        ))
    }

    pub fn multi_get(
        &self,
        space: Option<&str>,
        locators: &[String],
        max_files: usize,
        max_bytes: usize,
    ) -> Result<String> {
        let locators = locators
            .iter()
            .map(|item| parse_cli_locator(item))
            .collect::<Vec<_>>();

        let response = self.engine.multi_get(MultiGetRequest {
            locators,
            space: space.map(ToString::to_string),
            max_files,
            max_bytes,
        })?;

        let mut lines = Vec::new();
        lines.push(format!("documents: {}", response.documents.len()));
        for document in response.documents {
            lines.push(format!(
                "--- {} {} (stale: {}, lines: {}/{}) ---",
                document.docid,
                document.path,
                document.stale,
                document.returned_lines,
                document.total_lines
            ));
            lines.push(document.content);
        }

        lines.push(format!("omitted: {}", response.omitted.len()));
        for omitted in response.omitted {
            let reason = match omitted.reason {
                OmitReason::MaxFiles => "max_files",
                OmitReason::MaxBytes => "max_bytes",
            };
            lines.push(format!(
                "- {} {} ({} bytes, reason: {reason})",
                omitted.docid, omitted.path, omitted.size_bytes
            ));
        }
        lines.push(format!("resolved_count: {}", response.resolved_count));
        Ok(lines.join("\n"))
    }
}

fn format_search_mode(mode: &SearchMode) -> &'static str {
    match mode {
        SearchMode::Auto => "auto",
        SearchMode::Deep => "deep",
        SearchMode::Keyword => "keyword",
        SearchMode::Semantic => "semantic",
    }
}

fn parse_cli_locator(raw: &str) -> Locator {
    let trimmed = raw.trim();
    if trimmed.contains('/') {
        return Locator::Path(trimmed.to_string());
    }

    let docid = trimmed.trim_start_matches('#').to_string();
    Locator::DocId(docid)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};
    use std::{fs, path::PathBuf};

    use tempfile::tempdir;

    use super::CliAdapter;
    use kbolt_core::engine::Engine;
    use kbolt_types::AddCollectionRequest;

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

    #[test]
    fn space_current_reports_none_when_no_source_is_set() {
        with_isolated_xdg_dirs(|| {
            let mut engine = Engine::new(None).expect("create engine");
            engine
                .set_default_space(None)
                .expect("clear default space for test");
            let adapter = CliAdapter::new(engine);

            let output = adapter.space_current(None).expect("run space current");
            assert_eq!(output, "active space: none");
        });
    }

    #[test]
    fn space_current_reports_default_source() {
        with_isolated_xdg_dirs(|| {
            let mut engine = Engine::new(None).expect("create engine");
            engine
                .set_default_space(Some("default"))
                .expect("set default space");
            let adapter = CliAdapter::new(engine);

            let output = adapter.space_current(None).expect("run space current");
            assert_eq!(output, "active space: default (default)");
        });
    }

    #[test]
    fn space_default_reports_none_when_unset() {
        with_isolated_xdg_dirs(|| {
            let mut engine = Engine::new(None).expect("create engine");
            engine
                .set_default_space(None)
                .expect("clear default space for test");
            let mut adapter = CliAdapter::new(engine);

            let output = adapter.space_default(None).expect("show default space");
            assert_eq!(output, "default space: none");
        });
    }

    #[test]
    fn space_default_sets_and_reports_value() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let mut adapter = CliAdapter::new(engine);

            let set_output = adapter
                .space_default(Some("default"))
                .expect("set default space");
            assert_eq!(set_output, "default space: default");

            let get_output = adapter.space_default(None).expect("read default space");
            assert_eq!(get_output, "default space: default");
        });
    }

    #[test]
    fn space_list_reports_default_space() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let output = adapter.space_list().expect("list spaces");
            assert!(
                output.contains("- default (collections: 0, documents: 0, chunks: 0)"),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn space_list_includes_newly_added_spaces() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            engine
                .add_space("work", Some("work docs"))
                .expect("add work space");
            let adapter = CliAdapter::new(engine);

            let output = adapter.space_list().expect("list spaces");
            assert!(output.contains("- work (collections: 0, documents: 0, chunks: 0) - work docs"));
        });
    }

    #[test]
    fn space_info_reports_selected_space() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            engine
                .add_space("work", Some("work docs"))
                .expect("add work space");
            let adapter = CliAdapter::new(engine);

            let output = adapter.space_info("work").expect("show space info");
            assert!(output.contains("name: work"), "unexpected output: {output}");
            assert!(
                output.contains("description: work docs"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("collections: 0"),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn space_add_creates_space_without_description() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let output = adapter
                .space_add("work", None, false, &[])
                .expect("add space");
            assert_eq!(output, "space added: work");

            let info = adapter.space_info("work").expect("space info");
            assert!(info.contains("name: work"), "unexpected output: {info}");
        });
    }

    #[test]
    fn space_add_includes_description_in_output() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let output = adapter
                .space_add("work", Some("work docs"), false, &[])
                .expect("add space");
            assert_eq!(output, "space added: work - work docs");
        });
    }

    #[test]
    fn space_add_with_dirs_is_best_effort_and_reports_failures() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let valid_api = root.path().join("api");
            let valid_wiki = root.path().join("wiki");
            let missing = root.path().join("missing");
            fs::create_dir_all(&valid_api).expect("create api dir");
            fs::create_dir_all(&valid_wiki).expect("create wiki dir");

            let output = adapter
                .space_add(
                    "work",
                    Some("work docs"),
                    false,
                    &[valid_api.clone(), missing.clone(), valid_wiki.clone()],
                )
                .expect("add space with dirs");
            assert!(output.contains("space added: work - work docs"), "unexpected output: {output}");
            assert!(output.contains("collections added: 2"), "unexpected output: {output}");
            assert!(output.contains("collections failed: 1"), "unexpected output: {output}");
            assert!(
                output.contains(&format!("- {} ->", missing.display())),
                "unexpected output: {output}"
            );

            let api = adapter
                .collection_info(Some("work"), "api")
                .expect("api should be added");
            assert!(api.contains("name: api"), "unexpected output: {api}");
            let wiki = adapter
                .collection_info(Some("work"), "wiki")
                .expect("wiki should be added");
            assert!(wiki.contains("name: wiki"), "unexpected output: {wiki}");
        });
    }

    #[test]
    fn space_add_with_dirs_strict_aborts_without_side_effects() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let valid_api = root.path().join("api");
            let missing = root.path().join("missing");
            fs::create_dir_all(&valid_api).expect("create api dir");

            let err = adapter
                .space_add(
                    "work",
                    Some("work docs"),
                    true,
                    &[valid_api.clone(), missing.clone()],
                )
                .expect_err("strict mode should fail");
            assert!(
                err.to_string()
                    .contains("strict mode aborted: one or more directories are invalid"),
                "unexpected error: {err}"
            );

            let missing_space = adapter
                .space_info("work")
                .expect_err("space should not be created");
            assert!(
                missing_space.to_string().contains("space not found"),
                "unexpected error: {missing_space}"
            );
        });
    }

    #[test]
    fn space_add_with_dirs_strict_succeeds_when_all_valid() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let valid_api = root.path().join("api");
            let valid_wiki = root.path().join("wiki");
            fs::create_dir_all(&valid_api).expect("create api dir");
            fs::create_dir_all(&valid_wiki).expect("create wiki dir");

            let output = adapter
                .space_add(
                    "work",
                    Some("work docs"),
                    true,
                    &[valid_api.clone(), valid_wiki.clone()],
                )
                .expect("strict mode should succeed");
            assert!(output.contains("collections added: 2"), "unexpected output: {output}");
            assert!(
                !output.contains("collections failed:"),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn space_describe_updates_space_description() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);
            adapter
                .space_add("work", Some("old docs"), false, &[])
                .expect("add work space");

            let output = adapter
                .space_describe("work", "updated docs")
                .expect("describe space");
            assert_eq!(output, "space description updated: work");

            let info = adapter.space_info("work").expect("space info");
            assert!(
                info.contains("description: updated docs"),
                "unexpected output: {info}"
            );
        });
    }

    #[test]
    fn space_rename_moves_space_name() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);
            adapter
                .space_add("work", None, false, &[])
                .expect("add work");

            let output = adapter
                .space_rename("work", "team")
                .expect("rename space");
            assert_eq!(output, "space renamed: work -> team");

            let info = adapter.space_info("team").expect("team space info");
            assert!(info.contains("name: team"), "unexpected output: {info}");
        });
    }

    #[test]
    fn space_remove_deletes_non_default_space() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);
            adapter
                .space_add("work", None, false, &[])
                .expect("add work");

            let output = adapter.space_remove("work").expect("remove work");
            assert_eq!(output, "space removed: work");

            let err = adapter
                .space_info("work")
                .expect_err("work should be removed");
            assert!(
                err.to_string().contains("space not found"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn space_remove_clears_default_space_without_deleting_it() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let output = adapter.space_remove("default").expect("clear default");
            assert_eq!(output, "default space cleared");

            let info = adapter.space_info("default").expect("default should remain");
            assert!(
                info.contains("name: default"),
                "unexpected output: {info}"
            );
        });
    }

    fn new_collection_dir(root: &PathBuf, name: &str) -> PathBuf {
        let path = root.join(name);
        fs::create_dir_all(&path).expect("create collection directory");
        path
    }

    #[test]
    fn collection_list_reports_none_when_empty() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let output = adapter.collection_list(None).expect("list collections");
            assert_eq!(output, "collections:\n- none");
        });
    }

    #[test]
    fn collection_list_supports_space_scoping() {
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

            let adapter = CliAdapter::new(engine);

            let scoped = adapter
                .collection_list(Some("work"))
                .expect("list scoped collections");
            assert!(
                scoped.contains("- work/api"),
                "unexpected scoped output: {scoped}"
            );
            assert!(
                !scoped.contains("notes/wiki"),
                "unexpected scoped output: {scoped}"
            );

            let all = adapter
                .collection_list(None)
                .expect("list all collections");
            assert!(all.contains("- work/api"), "unexpected all output: {all}");
            assert!(all.contains("- notes/wiki"), "unexpected all output: {all}");
        });
    }

    #[test]
    fn collection_info_resolves_and_formats_fields() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: Some("API docs".to_string()),
                    extensions: Some(vec!["rs".to_string(), "md".to_string()]),
                    no_index: true,
                })
                .expect("add work collection");

            let adapter = CliAdapter::new(engine);
            let output = adapter
                .collection_info(Some("work"), "api")
                .expect("collection info");
            assert!(output.contains("name: api"), "unexpected output: {output}");
            assert!(output.contains("space: work"), "unexpected output: {output}");
            assert!(
                output.contains("description: API docs"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("extensions: rs,md"),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn collection_describe_updates_collection_description() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: Some("old docs".to_string()),
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");

            let adapter = CliAdapter::new(engine);
            let output = adapter
                .collection_describe(Some("work"), "api", "new docs")
                .expect("describe collection");
            assert_eq!(output, "collection description updated: api");

            let info = adapter
                .collection_info(Some("work"), "api")
                .expect("collection info");
            assert!(
                info.contains("description: new docs"),
                "unexpected output: {info}"
            );
        });
    }

    #[test]
    fn collection_rename_updates_collection_name() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");

            let adapter = CliAdapter::new(engine);
            let output = adapter
                .collection_rename(Some("work"), "api", "backend")
                .expect("rename collection");
            assert_eq!(output, "collection renamed: api -> backend");

            let info = adapter
                .collection_info(Some("work"), "backend")
                .expect("collection info");
            assert!(info.contains("name: backend"), "unexpected output: {info}");
        });
    }

    #[test]
    fn collection_remove_deletes_collection() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");

            let adapter = CliAdapter::new(engine);
            let output = adapter
                .collection_remove(Some("work"), "api")
                .expect("remove collection");
            assert_eq!(output, "collection removed: api");

            let err = adapter
                .collection_info(Some("work"), "api")
                .expect_err("collection should be missing");
            assert!(
                err.to_string().contains("collection not found"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn collection_add_registers_collection_with_no_index() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            let adapter = CliAdapter::new(engine);

            let output = adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, true)
                .expect("add collection");
            assert_eq!(output, "collection added: work/api");

            let info = adapter
                .collection_info(Some("work"), "api")
                .expect("collection info");
            assert!(info.contains("name: api"), "unexpected output: {info}");
        });
    }

    #[test]
    fn collection_add_without_no_index_triggers_index_update() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            fs::create_dir_all(collection_path.join("src")).expect("create src dir");
            fs::write(collection_path.join("src/lib.rs"), "fn alpha() {}\n").expect("write file");
            let adapter = CliAdapter::new(engine);

            let output = adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, false)
                .expect("collection add should trigger update");
            assert_eq!(output, "collection added: work/api");

            let info = adapter
                .collection_info(Some("work"), "api")
                .expect("collection info");
            assert!(
                info.contains("documents: 1"),
                "expected indexed document count in output: {info}"
            );
        });
    }

    #[test]
    fn models_list_reports_configured_models() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);
            let embed_model = adapter.engine.config().models.embed.clone();
            let reranker_model = adapter.engine.config().models.reranker.clone();
            let expander_model = adapter.engine.config().models.expander.clone();

            let output = adapter.models_list().expect("list models");
            assert!(output.contains("models:"), "unexpected output: {output}");
            assert!(
                output.contains(&format!("- embedder: {embed_model} (missing)")),
                "unexpected output: {output}"
            );
            assert!(
                output.contains(&format!("- reranker: {reranker_model} (missing)")),
                "unexpected output: {output}"
            );
            assert!(
                output.contains(&format!("- expander: {expander_model} (missing)")),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn models_pull_reports_already_present_models() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);
            let model_dir = adapter.engine.config().cache_dir.join("models");

            fs::create_dir_all(model_dir.join("embedder")).expect("create embedder dir");
            fs::create_dir_all(model_dir.join("reranker")).expect("create reranker dir");
            fs::create_dir_all(model_dir.join("expander")).expect("create expander dir");
            fs::write(model_dir.join("embedder/model.bin"), b"e").expect("seed embedder");
            fs::write(model_dir.join("reranker/model.bin"), b"r").expect("seed reranker");
            fs::write(model_dir.join("expander/model.bin"), b"x").expect("seed expander");

            let embed_model = adapter.engine.config().models.embed.clone();
            let reranker_model = adapter.engine.config().models.reranker.clone();
            let expander_model = adapter.engine.config().models.expander.clone();

            let output = adapter.models_pull().expect("pull models");
            assert!(output.contains("downloaded: 0"), "unexpected output: {output}");
            assert!(
                output.contains("downloaded_models:\n- none"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("already_present: 3"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains(&format!("- {embed_model}")),
                "unexpected output: {output}"
            );
            assert!(
                output.contains(&format!("- {reranker_model}")),
                "unexpected output: {output}"
            );
            assert!(
                output.contains(&format!("- {expander_model}")),
                "unexpected output: {output}"
            );
            assert!(output.contains("total_bytes: 0"), "unexpected output: {output}");
        });
    }

    #[test]
    fn search_rejects_conflicting_mode_flags() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let err = adapter
                .search(None, "alpha", &[], 10, 0.0, true, true, false, false, false)
                .expect_err("conflicting search flags should fail");
            assert!(
                err.to_string().contains("only one of --deep, --keyword, or --semantic"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn search_keyword_formats_ranked_output() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, true)
                .expect("add collection");

            fs::write(collection_path.join("a.md"), "alpha query token\n").expect("write file");
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            let output = adapter
                .search(
                    Some("work"),
                    "alpha",
                    &["api".to_string()],
                    5,
                    0.0,
                    false,
                    true,
                    false,
                    false,
                    true,
                )
                .expect("run search");
            assert!(output.contains("query: alpha"), "unexpected output: {output}");
            assert!(output.contains("mode: keyword"), "unexpected output: {output}");
            assert!(output.contains("results: 1"), "unexpected output: {output}");
            assert!(output.contains("1. #"), "unexpected output: {output}");
            assert!(output.contains("api/a.md"), "unexpected output: {output}");
            assert!(output.contains("signals:"), "unexpected output: {output}");
        });
    }

    #[test]
    fn update_reports_added_and_skipped_counts() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, true)
                .expect("add collection");

            let file_path = collection_path.join("src/lib.rs");
            fs::create_dir_all(file_path.parent().expect("file parent")).expect("create parent");
            fs::write(&file_path, "fn alpha() {}\n").expect("write file");

            let first = adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");
            assert!(first.contains("added: 1"), "unexpected output: {first}");
            assert!(first.contains("errors: 0"), "unexpected output: {first}");

            let second = adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run second update");
            assert!(
                second.contains("skipped_mtime: 1"),
                "unexpected output: {second}"
            );
        });
    }

    #[test]
    fn update_dry_run_does_not_persist_documents() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, true)
                .expect("add collection");

            let file_path = collection_path.join("src/lib.rs");
            fs::create_dir_all(file_path.parent().expect("file parent")).expect("create parent");
            fs::write(&file_path, "fn alpha() {}\n").expect("write file");

            let output = adapter
                .update(Some("work"), &["api".to_string()], true, true, false)
                .expect("run dry-run update");
            assert!(output.contains("added: 1"), "unexpected output: {output}");

            let space = adapter.engine.storage().get_space("work").expect("get space");
            let collection = adapter
                .engine
                .storage()
                .get_collection(space.id, "api")
                .expect("get collection");
            let docs = adapter
                .engine
                .storage()
                .list_documents(collection.id, false)
                .expect("list docs");
            assert!(docs.is_empty(), "dry run should not persist docs");
        });
    }

    #[test]
    fn ls_filters_inactive_by_default_and_includes_it_with_all() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, true)
                .expect("add collection");

            fs::create_dir_all(collection_path.join("src")).expect("create src dir");
            fs::write(collection_path.join("src/lib.rs"), "fn alpha() {}\n").expect("write src");
            fs::create_dir_all(collection_path.join("docs")).expect("create docs dir");
            fs::write(collection_path.join("docs/guide.md"), "guide\n").expect("write docs");
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            let space = adapter.engine.storage().get_space("work").expect("get space");
            let collection = adapter
                .engine
                .storage()
                .get_collection(space.id, "api")
                .expect("get collection");
            let docs_entry = adapter
                .engine
                .storage()
                .get_document_by_path(collection.id, "docs/guide.md")
                .expect("lookup docs file")
                .expect("docs file should exist");
            adapter
                .engine
                .storage()
                .deactivate_document(docs_entry.id)
                .expect("deactivate docs file");

            let default_output = adapter
                .ls(Some("work"), "api", None, false)
                .expect("run ls");
            assert!(default_output.contains("files:"), "unexpected output: {default_output}");
            assert!(
                default_output.contains("- src/lib.rs | lib.rs | #"),
                "unexpected output: {default_output}"
            );
            assert!(
                !default_output.contains("docs/guide.md"),
                "unexpected output: {default_output}"
            );

            let all_output = adapter
                .ls(Some("work"), "api", None, true)
                .expect("run ls --all");
            assert!(
                all_output.contains("- docs/guide.md | guide.md | #"),
                "unexpected output: {all_output}"
            );
            assert!(
                all_output.contains("active: false"),
                "unexpected output: {all_output}"
            );
        });
    }

    #[test]
    fn ls_supports_prefix_and_unique_space_resolution() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, true)
                .expect("add collection");

            fs::create_dir_all(collection_path.join("src")).expect("create src dir");
            fs::write(collection_path.join("src/lib.rs"), "fn alpha() {}\n").expect("write src");
            fs::create_dir_all(collection_path.join("docs")).expect("create docs dir");
            fs::write(collection_path.join("docs/guide.md"), "guide\n").expect("write docs");
            adapter
                .update(None, &["api".to_string()], true, false, false)
                .expect("run update");

            let output = adapter.ls(None, "api", Some("src"), false).expect("run ls");
            assert!(
                output.contains("- src/lib.rs | lib.rs | #"),
                "unexpected output: {output}"
            );
            assert!(
                !output.contains("docs/guide.md"),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn get_by_path_formats_sliced_document_fields() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, true)
                .expect("add collection");

            fs::create_dir_all(collection_path.join("src")).expect("create src dir");
            fs::write(
                collection_path.join("src/lib.rs"),
                "line-a\nline-b\nline-c\n",
            )
            .expect("write src");
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            let output = adapter
                .get(Some("work"), "api/src/lib.rs", Some(1), Some(1))
                .expect("get sliced file");
            assert!(output.contains("docid: #"), "unexpected output: {output}");
            assert!(
                output.contains("path: api/src/lib.rs"),
                "unexpected output: {output}"
            );
            assert!(output.contains("stale: false"), "unexpected output: {output}");
            assert!(output.contains("total_lines: 3"), "unexpected output: {output}");
            assert!(
                output.contains("returned_lines: 1"),
                "unexpected output: {output}"
            );
            assert!(output.ends_with("line-b"), "unexpected output: {output}");
        });
    }

    #[test]
    fn get_by_docid_reports_stale_after_file_change() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, true)
                .expect("add collection");

            fs::create_dir_all(collection_path.join("src")).expect("create src dir");
            let file_path = collection_path.join("src/lib.rs");
            fs::write(&file_path, "fn alpha() {}\n").expect("write src");
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            let files = adapter
                .engine
                .list_files(Some("work"), "api", None)
                .expect("list files");
            let docid = files[0].docid.clone();

            fs::write(&file_path, "fn beta() {}\n").expect("modify src");
            let output = adapter
                .get(Some("work"), &docid, None, None)
                .expect("get by docid");
            assert!(output.contains("stale: true"), "unexpected output: {output}");
            assert!(output.contains("content:\nfn beta() {}"), "unexpected output: {output}");

            let bare_docid = docid.trim_start_matches('#').to_string();
            let bare_output = adapter
                .get(Some("work"), &bare_docid, None, None)
                .expect("get by bare docid");
            assert!(
                bare_output.contains("path: api/src/lib.rs"),
                "unexpected output: {bare_output}"
            );
        });
    }

    #[test]
    fn multi_get_formats_documents_and_omissions() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, true)
                .expect("add collection");

            fs::write(collection_path.join("a.md"), "alpha\n").expect("write a");
            fs::write(collection_path.join("b.md"), "beta\n").expect("write b");
            fs::write(collection_path.join("c.md"), "gamma\n").expect("write c");
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            let output = adapter
                .multi_get(
                    Some("work"),
                    &[
                        "api/a.md".to_string(),
                        "api/b.md".to_string(),
                        "api/c.md".to_string(),
                    ],
                    2,
                    51_200,
                )
                .expect("run multi-get");
            assert!(output.contains("documents: 2"), "unexpected output: {output}");
            assert!(output.contains("omitted: 1"), "unexpected output: {output}");
            assert!(
                output.contains("reason: max_files"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("resolved_count: 2"),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn status_reports_spaces_totals_and_models() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", Some("work docs"), false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, true)
                .expect("add collection");

            let file_path = collection_path.join("src/lib.rs");
            fs::create_dir_all(file_path.parent().expect("file parent")).expect("create parent");
            fs::write(&file_path, "fn alpha() {}\n").expect("write file");
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            let output = adapter.status(None).expect("run status");
            assert!(output.contains("spaces:"), "unexpected output: {output}");
            assert!(
                output.contains("- work (collections: 1"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("  - api ("),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("total_documents: 1"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("total_chunks: 1"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("model_embedder:"),
                "unexpected output: {output}"
            );
            assert!(output.contains("cache_dir:"), "unexpected output: {output}");
            assert!(output.contains("config_dir:"), "unexpected output: {output}");
        });
    }

    #[test]
    fn status_supports_space_scope() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            adapter
                .space_add("notes", None, false, &[])
                .expect("add notes");

            let work_collection = new_collection_dir(&root.path().to_path_buf(), "work-api");
            adapter
                .collection_add(Some("work"), &work_collection, Some("api"), None, None, true)
                .expect("add work collection");
            let notes_collection = new_collection_dir(&root.path().to_path_buf(), "notes-wiki");
            adapter
                .collection_add(Some("notes"), &notes_collection, Some("wiki"), None, None, true)
                .expect("add notes collection");

            fs::write(work_collection.join("a.md"), "alpha\n").expect("write work file");
            fs::write(notes_collection.join("b.md"), "beta\n").expect("write notes file");
            adapter
                .update(None, &[], true, false, false)
                .expect("run update");

            let output = adapter.status(Some("work")).expect("run scoped status");
            assert!(
                output.contains("- work (collections: 1"),
                "unexpected output: {output}"
            );
            assert!(
                !output.contains("- notes (collections: 1"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("total_documents: 1"),
                "unexpected output: {output}"
            );
        });
    }
}
