pub mod args;

use kbolt_core::engine::Engine;
use kbolt_core::Result;
use kbolt_types::{ActiveSpaceSource, AddCollectionRequest, UpdateOptions};

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
    fn collection_add_without_no_index_surfaces_current_core_error() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(&root.path().to_path_buf(), "work-api");
            let adapter = CliAdapter::new(engine);

            let err = adapter
                .collection_add(Some("work"), &collection_path, Some("api"), None, None, false)
                .expect_err("collection add should fail without --no-index for now");
            assert!(
                err.to_string()
                    .contains("automatic indexing on collection add is not wired yet"),
                "unexpected error: {err}"
            );
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
}
