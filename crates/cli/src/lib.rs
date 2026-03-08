pub mod args;

use kbolt_core::engine::Engine;
use kbolt_core::ModelPullEvent;
use kbolt_core::Result;
use kbolt_types::{
    ActiveSpaceSource, AddCollectionRequest, AddScheduleRequest, EvalModeReport, EvalRunReport,
    GetRequest, KboltError, Locator, MultiGetRequest, OmitReason, RemoveScheduleRequest,
    ScheduleAddResponse, ScheduleBackend, ScheduleInterval, ScheduleIntervalUnit,
    ScheduleRunResult, ScheduleScope, ScheduleState, ScheduleStatusResponse, ScheduleTrigger,
    ScheduleWeekday, SearchMode, SearchRequest, UpdateDecision, UpdateDecisionKind, UpdateOptions,
    UpdateReport,
};

pub struct CliAdapter {
    pub engine: Engine,
}

pub struct CliSearchOptions<'a> {
    pub space: Option<&'a str>,
    pub query: &'a str,
    pub collections: &'a [String],
    pub limit: usize,
    pub min_score: f32,
    pub deep: bool,
    pub keyword: bool,
    pub semantic: bool,
    pub rerank: bool,
    pub no_rerank: bool,
    pub debug: bool,
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
                lines.push("strict mode aborted: one or more directories are invalid".to_string());
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
            space.name,
            space.collection_count,
            space.document_count,
            space.chunk_count,
            space.created
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

    pub fn collection_describe(
        &self,
        space: Option<&str>,
        name: &str,
        text: &str,
    ) -> Result<String> {
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

    pub fn ignore_show(&self, space: Option<&str>, collection: &str) -> Result<String> {
        let (resolved_space, content) = self.engine.read_collection_ignore(space, collection)?;
        if let Some(content) = content {
            return Ok(format!(
                "ignore patterns for {resolved_space}/{collection}:\n{content}"
            ));
        }
        Ok(format!(
            "no ignore patterns configured for {resolved_space}/{collection}"
        ))
    }

    pub fn ignore_add(
        &self,
        space: Option<&str>,
        collection: &str,
        pattern: &str,
    ) -> Result<String> {
        let (resolved_space, normalized_pattern) = self
            .engine
            .add_collection_ignore_pattern(space, collection, pattern)?;
        Ok(format!(
            "ignore pattern added for {resolved_space}/{collection}: {normalized_pattern}"
        ))
    }

    pub fn ignore_remove(
        &self,
        space: Option<&str>,
        collection: &str,
        pattern: &str,
    ) -> Result<String> {
        let (resolved_space, removed_count) = self
            .engine
            .remove_collection_ignore_pattern(space, collection, pattern)?;
        if removed_count == 0 {
            return Ok(format!(
                "ignore pattern not found for {resolved_space}/{collection}: {pattern}"
            ));
        }

        Ok(format!(
            "ignore pattern removed for {resolved_space}/{collection}: {pattern} ({removed_count} match(es))"
        ))
    }

    pub fn ignore_list(&self, space: Option<&str>) -> Result<String> {
        let entries = self.engine.list_collection_ignores(space)?;
        let mut lines = Vec::new();
        lines.push("ignore patterns:".to_string());
        if entries.is_empty() {
            lines.push("- none".to_string());
            return Ok(lines.join("\n"));
        }

        let mut current_space: Option<String> = None;
        for entry in entries {
            if current_space.as_deref() != Some(entry.space.as_str()) {
                lines.push(format!("{}:", entry.space));
                current_space = Some(entry.space.clone());
            }
            lines.push(format!(
                "- {} (patterns: {})",
                entry.collection, entry.pattern_count
            ));
        }

        Ok(lines.join("\n"))
    }

    pub fn ignore_edit(&self, space: Option<&str>, collection: &str) -> Result<String> {
        let (resolved_space, path) = self
            .engine
            .prepare_collection_ignore_edit(space, collection)?;
        let editor_command = resolve_editor_command()?;

        let mut process = std::process::Command::new(&editor_command[0]);
        if editor_command.len() > 1 {
            process.args(&editor_command[1..]);
        }
        process.arg(&path);

        let status = process.status().map_err(|err| {
            KboltError::Internal(format!(
                "failed to launch editor '{}': {err}",
                editor_command[0]
            ))
        })?;
        if !status.success() {
            return Err(
                KboltError::Internal(format!("editor exited with status: {status}")).into(),
            );
        }

        Ok(format!(
            "ignore patterns updated for {resolved_space}/{collection}: {}",
            path.display()
        ))
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
            let availability = if info.downloaded {
                "downloaded"
            } else {
                "missing"
            };
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
        let mut lines = Vec::new();
        let report = self.engine.pull_models_with_progress(|event| match event {
            ModelPullEvent::DownloadStarted { role, model } => {
                lines.push(format!("downloading {role}: {model}"));
            }
            ModelPullEvent::DownloadCompleted { role, model, bytes } => {
                lines.push(format!("downloaded {role}: {model} ({bytes} bytes)"));
            }
            ModelPullEvent::AlreadyPresent { role, model, bytes } => {
                lines.push(format!("already present {role}: {model} ({bytes} bytes)"));
            }
        })?;

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

    pub fn eval_run(&self) -> Result<String> {
        let report = self.engine.run_eval()?;
        Ok(format_eval_run_report(&report))
    }

    pub fn search(&self, options: CliSearchOptions<'_>) -> Result<String> {
        let CliSearchOptions {
            space,
            query,
            collections,
            limit,
            min_score,
            deep,
            keyword,
            semantic,
            rerank,
            no_rerank,
            debug,
        } = options;
        let mode_flags = deep as u8 + keyword as u8 + semantic as u8;
        if mode_flags > 1 {
            return Err(KboltError::InvalidInput(
                "only one of --deep, --keyword, or --semantic can be set".to_string(),
            )
            .into());
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
        let effective_no_rerank = resolve_no_rerank_for_mode(mode.clone(), rerank, no_rerank);

        let response = self.engine.search(SearchRequest {
            query: query.to_string(),
            mode,
            space: space.map(ToString::to_string),
            collections: collections.to_vec(),
            limit,
            min_score,
            no_rerank: effective_no_rerank,
            debug,
        })?;

        let mut lines = Vec::new();
        lines.push(format!("query: {}", response.query));
        lines.push(format!(
            "mode: {}",
            format_search_mode(&response.effective_mode)
        ));
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
            lines.push(format!(
                "space: {} | collection: {}",
                item.space, item.collection
            ));
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

        Ok(format_update_report(&report, verbose))
    }

    pub fn schedule_add(&self, req: AddScheduleRequest) -> Result<String> {
        let response = self.engine.add_schedule(req)?;
        Ok(format_schedule_add_response(&response))
    }

    pub fn schedule_status(&self) -> Result<String> {
        let response = self.engine.schedule_status()?;
        Ok(format_schedule_status_response(&response))
    }

    pub fn schedule_remove(&self, req: RemoveScheduleRequest) -> Result<String> {
        let response = self.engine.remove_schedule(req)?;
        Ok(format_schedule_remove_response(&response))
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
        lines.push(format!(
            "tantivy_bytes: {}",
            status.disk_usage.tantivy_bytes
        ));
        lines.push(format!(
            "usearch_bytes: {}",
            status.disk_usage.usearch_bytes
        ));
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
        let locator = Locator::parse(identifier);

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
            .map(|item| Locator::parse(item))
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
        if !response.warnings.is_empty() {
            lines.push(format!("warnings: {}", response.warnings.len()));
            for warning in response.warnings {
                lines.push(format!("- {warning}"));
            }
        }
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

fn format_update_report(report: &UpdateReport, verbose: bool) -> String {
    let mut lines = Vec::new();
    if verbose {
        for decision in &report.decisions {
            lines.push(format_update_decision(decision));
        }

        for error in unreported_update_errors(report) {
            lines.push(format!("error: {}: {}", error.path, error.error));
        }
    }

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
    lines.push(format!("elapsed_ms: {}", report.elapsed_ms));
    lines.join("\n")
}

fn format_update_decision(decision: &UpdateDecision) -> String {
    let locator = format!(
        "{}/{}/{}",
        decision.space, decision.collection, decision.path
    );
    match decision.detail.as_deref() {
        Some(detail) => format!(
            "{locator}: {} ({detail})",
            format_update_decision_kind(&decision.kind)
        ),
        None => format!("{locator}: {}", format_update_decision_kind(&decision.kind)),
    }
}

fn format_update_decision_kind(kind: &UpdateDecisionKind) -> &'static str {
    match kind {
        UpdateDecisionKind::New => "new",
        UpdateDecisionKind::Changed => "changed",
        UpdateDecisionKind::SkippedMtime => "skipped_mtime",
        UpdateDecisionKind::SkippedHash => "skipped_hash",
        UpdateDecisionKind::Ignored => "ignored",
        UpdateDecisionKind::Unsupported => "unsupported",
        UpdateDecisionKind::ReadFailed => "read_failed",
        UpdateDecisionKind::ExtractFailed => "extract_failed",
        UpdateDecisionKind::Reactivated => "reactivated",
        UpdateDecisionKind::Deactivated => "deactivated",
    }
}

fn unreported_update_errors(report: &UpdateReport) -> Vec<&kbolt_types::FileError> {
    report
        .errors
        .iter()
        .filter(|error| {
            !report.decisions.iter().any(|decision| {
                matches!(
                    decision.kind,
                    UpdateDecisionKind::ReadFailed | UpdateDecisionKind::ExtractFailed
                ) && std::path::Path::new(&error.path)
                    .ends_with(std::path::Path::new(&decision.path))
            })
        })
        .collect()
}

pub fn resolve_no_rerank_for_mode(mode: SearchMode, rerank: bool, no_rerank: bool) -> bool {
    match mode {
        SearchMode::Auto => !rerank,
        SearchMode::Deep => no_rerank,
        SearchMode::Keyword | SearchMode::Semantic => true,
    }
}

fn resolve_editor_command() -> Result<Vec<String>> {
    let raw = std::env::var("VISUAL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::env::var("EDITOR")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .unwrap_or_else(|| "vi".to_string());

    parse_editor_command(&raw)
}

fn parse_editor_command(raw: &str) -> Result<Vec<String>> {
    let args = shell_words::split(raw).map_err(|err| {
        KboltError::InvalidInput(format!("invalid editor command '{raw}': {err}"))
    })?;
    if args.is_empty() {
        return Err(KboltError::InvalidInput("editor command cannot be empty".to_string()).into());
    }
    Ok(args)
}

fn format_schedule_add_response(response: &ScheduleAddResponse) -> String {
    format!(
        "schedule added: {}\ntrigger: {}\nscope: {}\nbackend: {}",
        response.schedule.id,
        format_schedule_trigger(&response.schedule.trigger),
        format_schedule_scope(&response.schedule.scope),
        format_schedule_backend(response.backend),
    )
}

fn format_schedule_status_response(response: &ScheduleStatusResponse) -> String {
    let mut lines = Vec::new();
    lines.push("schedules:".to_string());
    if response.schedules.is_empty() {
        lines.push("- none".to_string());
    } else {
        for entry in &response.schedules {
            lines.push(format!(
                "- {} | {} | {} | {} | {}",
                entry.schedule.id,
                format_schedule_trigger(&entry.schedule.trigger),
                format_schedule_scope(&entry.schedule.scope),
                format_schedule_backend(entry.backend),
                format_schedule_state(entry.state),
            ));
            lines.push(format!(
                "  last_started: {}",
                entry.run_state.last_started.as_deref().unwrap_or("never")
            ));
            lines.push(format!(
                "  last_finished: {}",
                entry.run_state.last_finished.as_deref().unwrap_or("never")
            ));
            lines.push(format!(
                "  last_result: {}",
                format_schedule_run_result(entry.run_state.last_result)
            ));
            if let Some(error) = entry.run_state.last_error.as_deref() {
                lines.push(format!("  last_error: {error}"));
            }
        }
    }

    lines.push("orphans:".to_string());
    if response.orphans.is_empty() {
        lines.push("- none".to_string());
    } else {
        for orphan in &response.orphans {
            lines.push(format!(
                "- {} ({})",
                orphan.id,
                format_schedule_backend(orphan.backend)
            ));
        }
    }

    lines.join("\n")
}

fn format_schedule_remove_response(response: &kbolt_types::ScheduleRemoveResponse) -> String {
    if response.removed_ids.is_empty() {
        return "removed schedules: none".to_string();
    }

    format!("removed schedules: {}", response.removed_ids.join(", "))
}

fn format_schedule_trigger(trigger: &ScheduleTrigger) -> String {
    match trigger {
        ScheduleTrigger::Every { interval } => format_schedule_interval(interval),
        ScheduleTrigger::Daily { time } => format!("daily at {}", format_schedule_time(time)),
        ScheduleTrigger::Weekly { weekdays, time } => format!(
            "{} at {}",
            format_schedule_weekdays(weekdays),
            format_schedule_time(time)
        ),
    }
}

fn format_schedule_interval(interval: &ScheduleInterval) -> String {
    let suffix = match interval.unit {
        ScheduleIntervalUnit::Minutes => "m",
        ScheduleIntervalUnit::Hours => "h",
    };
    format!("every {}{suffix}", interval.value)
}

fn format_schedule_scope(scope: &ScheduleScope) -> String {
    match scope {
        ScheduleScope::All => "all spaces".to_string(),
        ScheduleScope::Space { space } => format!("space {space}"),
        ScheduleScope::Collections { space, collections } => collections
            .iter()
            .map(|collection| format!("{space}/{collection}"))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

fn format_schedule_backend(backend: ScheduleBackend) -> &'static str {
    match backend {
        ScheduleBackend::Launchd => "launchd",
        ScheduleBackend::SystemdUser => "systemd-user",
    }
}

fn format_schedule_state(state: ScheduleState) -> &'static str {
    match state {
        ScheduleState::Installed => "installed",
        ScheduleState::Drifted => "drifted",
        ScheduleState::TargetMissing => "target_missing",
    }
}

fn format_schedule_run_result(result: Option<ScheduleRunResult>) -> &'static str {
    match result {
        Some(ScheduleRunResult::Success) => "success",
        Some(ScheduleRunResult::SkippedLock) => "skipped_lock",
        Some(ScheduleRunResult::Failed) => "failed",
        None => "never",
    }
}

fn format_schedule_weekdays(weekdays: &[ScheduleWeekday]) -> String {
    weekdays
        .iter()
        .map(|weekday| match weekday {
            ScheduleWeekday::Mon => "mon",
            ScheduleWeekday::Tue => "tue",
            ScheduleWeekday::Wed => "wed",
            ScheduleWeekday::Thu => "thu",
            ScheduleWeekday::Fri => "fri",
            ScheduleWeekday::Sat => "sat",
            ScheduleWeekday::Sun => "sun",
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn format_schedule_time(time: &str) -> String {
    let Some((hour, minute)) = time.split_once(':') else {
        return time.to_string();
    };
    let Ok(mut hour) = hour.parse::<u32>() else {
        return time.to_string();
    };
    let Ok(minute) = minute.parse::<u32>() else {
        return time.to_string();
    };

    let meridiem = if hour >= 12 { "PM" } else { "AM" };
    if hour == 0 {
        hour = 12;
    } else if hour > 12 {
        hour -= 12;
    }

    format!("{hour}:{minute:02} {meridiem}")
}

fn format_eval_run_report(report: &EvalRunReport) -> String {
    let mut lines = vec!["eval:".to_string()];
    for mode in &report.modes {
        lines.push(format!(
            "- {}: recall@5 {:.3}, mrr@10 {:.3}, p50 {}ms, p95 {}ms",
            format_eval_mode(mode),
            mode.recall_at_5,
            mode.mrr_at_10,
            mode.latency_p50_ms,
            mode.latency_p95_ms
        ));
    }

    let findings = report
        .modes
        .iter()
        .flat_map(|mode| {
            mode.queries.iter().filter_map(|query| {
                let perfect_recall = query.matched_paths.len() == query.expected_paths.len();
                let perfect_rank = query.first_relevant_rank == Some(1);
                if perfect_recall && perfect_rank {
                    return None;
                }

                Some(format!(
                    "- [{}] {} | first relevant: {} | expected: {} | returned: {}",
                    format_eval_mode(mode),
                    query.query,
                    query
                        .first_relevant_rank
                        .map(|rank| rank.to_string())
                        .unwrap_or_else(|| "none".to_string()),
                    query.expected_paths.join(", "),
                    if query.returned_paths.is_empty() {
                        "none".to_string()
                    } else {
                        query.returned_paths.join(", ")
                    }
                ))
            })
        })
        .collect::<Vec<_>>();

    if findings.is_empty() {
        lines.push("queries needing attention: none".to_string());
    } else {
        lines.push("queries needing attention:".to_string());
        lines.extend(findings);
    }

    lines.join("\n")
}

fn format_eval_mode(report: &EvalModeReport) -> &'static str {
    match report.mode {
        SearchMode::Keyword => "keyword",
        SearchMode::Auto => "auto",
        SearchMode::Deep => "deep",
        SearchMode::Semantic => "semantic",
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use tempfile::tempdir;

    use super::{
        format_eval_run_report, format_schedule_add_response, format_schedule_status_response,
        parse_editor_command, resolve_editor_command, resolve_no_rerank_for_mode, CliAdapter,
        CliSearchOptions,
    };
    use kbolt_core::engine::Engine;
    use kbolt_types::{
        AddCollectionRequest, EvalModeReport, EvalQueryReport, EvalRunReport, ScheduleAddResponse,
        ScheduleBackend, ScheduleDefinition, ScheduleInterval, ScheduleIntervalUnit,
        ScheduleOrphan, ScheduleRunResult, ScheduleRunState, ScheduleScope, ScheduleState,
        ScheduleStatusEntry, ScheduleStatusResponse, ScheduleTrigger, ScheduleWeekday, SearchMode,
    };

    const MODEL_MANIFEST_FILENAME: &str = ".kbolt-model-manifest.json";

    struct EnvRestore {
        home: Option<OsString>,
        config_home: Option<OsString>,
        cache_home: Option<OsString>,
        visual: Option<OsString>,
        editor: Option<OsString>,
    }

    impl EnvRestore {
        fn capture() -> Self {
            Self {
                home: std::env::var_os("HOME"),
                config_home: std::env::var_os("XDG_CONFIG_HOME"),
                cache_home: std::env::var_os("XDG_CACHE_HOME"),
                visual: std::env::var_os("VISUAL"),
                editor: std::env::var_os("EDITOR"),
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
            match &self.visual {
                Some(value) => std::env::set_var("VISUAL", value),
                None => std::env::remove_var("VISUAL"),
            }
            match &self.editor {
                Some(value) => std::env::set_var("EDITOR", value),
                None => std::env::remove_var("EDITOR"),
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
            assert!(
                output.contains("space added: work - work docs"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("collections added: 2"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("collections failed: 1"),
                "unexpected output: {output}"
            );
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
            assert!(
                output.contains("collections added: 2"),
                "unexpected output: {output}"
            );
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

            let output = adapter.space_rename("work", "team").expect("rename space");
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

            let info = adapter
                .space_info("default")
                .expect("default should remain");
            assert!(info.contains("name: default"), "unexpected output: {info}");
        });
    }

    fn new_collection_dir(root: &Path, name: &str) -> PathBuf {
        let path = root.join(name);
        fs::create_dir_all(&path).expect("create collection directory");
        path
    }

    fn json_escape(value: &str) -> String {
        value.replace('\\', "\\\\").replace('"', "\\\"")
    }

    fn seed_model_artifact(model_root: &Path, role: &str, model_id: &str, payload: &[u8]) {
        let role_dir = model_root.join(role);
        fs::create_dir_all(&role_dir).expect("create model role dir");
        fs::write(role_dir.join("model.bin"), payload).expect("write model payload");

        let manifest = format!(
            "{{\n  \"provider\": \"huggingface\",\n  \"id\": \"{}\",\n  \"revision\": null\n}}\n",
            json_escape(model_id)
        );
        fs::write(role_dir.join(MODEL_MANIFEST_FILENAME), manifest).expect("write model manifest");
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

            let all = adapter.collection_list(None).expect("list all collections");
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

            let collection_path = new_collection_dir(root.path(), "work-api");
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
            assert!(
                output.contains("space: work"),
                "unexpected output: {output}"
            );
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

            let collection_path = new_collection_dir(root.path(), "work-api");
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

            let collection_path = new_collection_dir(root.path(), "work-api");
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

            let collection_path = new_collection_dir(root.path(), "work-api");
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
            let collection_path = new_collection_dir(root.path(), "work-api");
            let adapter = CliAdapter::new(engine);

            let output = adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
                .expect("add collection");
            assert_eq!(output, "collection added: work/api");

            let info = adapter
                .collection_info(Some("work"), "api")
                .expect("collection info");
            assert!(info.contains("name: api"), "unexpected output: {info}");
        });
    }

    #[test]
    fn collection_add_then_update_no_embed_triggers_index_update() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            fs::create_dir_all(collection_path.join("src")).expect("create src dir");
            fs::write(collection_path.join("src/lib.rs"), "fn alpha() {}\n").expect("write file");
            let adapter = CliAdapter::new(engine);

            let output = adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
                .expect("add collection without initial indexing");
            assert_eq!(output, "collection added: work/api");

            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("update should index collection without embeddings");

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
    fn ignore_show_reports_when_no_patterns_are_configured() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add collection");
            let adapter = CliAdapter::new(engine);

            let output = adapter
                .ignore_show(Some("work"), "api")
                .expect("show ignore patterns");
            assert_eq!(output, "no ignore patterns configured for work/api");
        });
    }

    #[test]
    fn ignore_show_prints_collection_patterns() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add collection");

            fs::create_dir_all(engine.config().config_dir.join("ignores").join("work"))
                .expect("create ignore dir");
            fs::write(
                engine
                    .config()
                    .config_dir
                    .join("ignores")
                    .join("work")
                    .join("api.ignore"),
                "dist/\n*.tmp\n",
            )
            .expect("write ignore file");

            let adapter = CliAdapter::new(engine);
            let output = adapter
                .ignore_show(None, "api")
                .expect("show ignore patterns");
            assert_eq!(output, "ignore patterns for work/api:\ndist/\n*.tmp");
        });
    }

    #[test]
    fn ignore_add_appends_pattern_to_collection_file() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add collection");
            let adapter = CliAdapter::new(engine);

            let first = adapter
                .ignore_add(None, "api", "dist/")
                .expect("add first ignore pattern");
            assert_eq!(first, "ignore pattern added for work/api: dist/");

            let second = adapter
                .ignore_add(Some("work"), "api", "*.tmp")
                .expect("add second ignore pattern");
            assert_eq!(second, "ignore pattern added for work/api: *.tmp");

            let saved = fs::read_to_string(
                adapter
                    .engine
                    .config()
                    .config_dir
                    .join("ignores")
                    .join("work")
                    .join("api.ignore"),
            )
            .expect("read ignore file");
            assert_eq!(saved, "dist/\n*.tmp\n");
        });
    }

    #[test]
    fn ignore_add_rejects_empty_pattern() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add collection");
            let adapter = CliAdapter::new(engine);

            let err = adapter
                .ignore_add(Some("work"), "api", "   ")
                .expect_err("empty pattern should fail");
            assert!(
                err.to_string().contains("ignore pattern cannot be empty"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn ignore_remove_deletes_matching_patterns() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add collection");
            fs::create_dir_all(engine.config().config_dir.join("ignores").join("work"))
                .expect("create ignore dir");
            fs::write(
                engine
                    .config()
                    .config_dir
                    .join("ignores")
                    .join("work")
                    .join("api.ignore"),
                "dist/\n*.tmp\ndist/\n",
            )
            .expect("write ignore file");
            let adapter = CliAdapter::new(engine);

            let output = adapter
                .ignore_remove(Some("work"), "api", "dist/")
                .expect("remove pattern");
            assert_eq!(
                output,
                "ignore pattern removed for work/api: dist/ (2 match(es))"
            );

            let saved = fs::read_to_string(
                adapter
                    .engine
                    .config()
                    .config_dir
                    .join("ignores")
                    .join("work")
                    .join("api.ignore"),
            )
            .expect("read updated ignore file");
            assert_eq!(saved, "*.tmp\n");
        });
    }

    #[test]
    fn ignore_remove_reports_when_pattern_is_not_found() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add collection");
            let adapter = CliAdapter::new(engine);

            let output = adapter
                .ignore_remove(None, "api", "dist/")
                .expect("remove missing pattern");
            assert_eq!(output, "ignore pattern not found for work/api: dist/");
        });
    }

    #[test]
    fn ignore_list_reports_none_when_no_ignore_files_exist() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add collection");
            let adapter = CliAdapter::new(engine);

            let output = adapter.ignore_list(None).expect("list ignores");
            assert_eq!(output, "ignore patterns:\n- none");
        });
    }

    #[test]
    fn ignore_list_groups_entries_by_space_and_honors_scope() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            engine.add_space("notes", None).expect("add notes");
            let work_collection = new_collection_dir(root.path(), "work-api");
            let notes_collection = new_collection_dir(root.path(), "notes-wiki");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_collection,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");
            engine
                .add_collection(AddCollectionRequest {
                    path: notes_collection,
                    space: Some("notes".to_string()),
                    name: Some("wiki".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add notes collection");

            fs::create_dir_all(engine.config().config_dir.join("ignores").join("work"))
                .expect("create work ignore dir");
            fs::write(
                engine
                    .config()
                    .config_dir
                    .join("ignores")
                    .join("work")
                    .join("api.ignore"),
                "dist/\n*.tmp\n",
            )
            .expect("write work ignore file");

            fs::create_dir_all(engine.config().config_dir.join("ignores").join("notes"))
                .expect("create notes ignore dir");
            fs::write(
                engine
                    .config()
                    .config_dir
                    .join("ignores")
                    .join("notes")
                    .join("wiki.ignore"),
                "# comment\nbuild/\n",
            )
            .expect("write notes ignore file");

            let adapter = CliAdapter::new(engine);
            let output = adapter.ignore_list(None).expect("list all ignores");
            assert!(
                output.contains("ignore patterns:"),
                "unexpected output: {output}"
            );
            assert!(output.contains("notes:"), "unexpected output: {output}");
            assert!(
                output.contains("- wiki (patterns: 1)"),
                "unexpected output: {output}"
            );
            assert!(output.contains("work:"), "unexpected output: {output}");
            assert!(
                output.contains("- api (patterns: 2)"),
                "unexpected output: {output}"
            );

            let scoped = adapter
                .ignore_list(Some("work"))
                .expect("list scoped ignores");
            assert!(scoped.contains("work:"), "unexpected output: {scoped}");
            assert!(
                scoped.contains("- api (patterns: 2)"),
                "unexpected output: {scoped}"
            );
            assert!(!scoped.contains("notes:"), "unexpected output: {scoped}");
        });
    }

    #[test]
    fn editor_command_resolution_prefers_visual_then_editor_then_vi() {
        with_isolated_xdg_dirs(|| {
            std::env::set_var("VISUAL", "nvim -f");
            std::env::set_var("EDITOR", "vim");
            let from_visual = resolve_editor_command().expect("resolve visual");
            assert_eq!(from_visual, vec!["nvim".to_string(), "-f".to_string()]);

            std::env::remove_var("VISUAL");
            let from_editor = resolve_editor_command().expect("resolve editor");
            assert_eq!(from_editor, vec!["vim".to_string()]);

            std::env::remove_var("EDITOR");
            let fallback = resolve_editor_command().expect("resolve fallback");
            assert_eq!(fallback, vec!["vi".to_string()]);
        });
    }

    #[test]
    fn parse_editor_command_rejects_invalid_shell_words() {
        let err = parse_editor_command("'").expect_err("invalid shell words should fail");
        assert!(
            err.to_string().contains("invalid editor command"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn eval_run_report_formats_summary_and_attention_queries() {
        let output = format_eval_run_report(&EvalRunReport {
            total_cases: 1,
            modes: vec![
                EvalModeReport {
                    mode: SearchMode::Keyword,
                    recall_at_5: 1.0,
                    mrr_at_10: 1.0,
                    latency_p50_ms: 2,
                    latency_p95_ms: 3,
                    queries: vec![EvalQueryReport {
                        query: "trait object generic".to_string(),
                        space: Some("default".to_string()),
                        collections: vec!["rust".to_string()],
                        expected_paths: vec!["rust/guides/traits.md".to_string()],
                        returned_paths: vec!["rust/guides/traits.md".to_string()],
                        matched_paths: vec!["rust/guides/traits.md".to_string()],
                        first_relevant_rank: Some(1),
                        elapsed_ms: 2,
                    }],
                },
                EvalModeReport {
                    mode: SearchMode::Deep,
                    recall_at_5: 0.0,
                    mrr_at_10: 0.0,
                    latency_p50_ms: 8,
                    latency_p95_ms: 12,
                    queries: vec![EvalQueryReport {
                        query: "trait object generic".to_string(),
                        space: Some("default".to_string()),
                        collections: vec!["rust".to_string()],
                        expected_paths: vec!["rust/guides/traits.md".to_string()],
                        returned_paths: vec!["rust/overview.md".to_string()],
                        matched_paths: vec![],
                        first_relevant_rank: None,
                        elapsed_ms: 8,
                    }],
                },
            ],
        });

        assert!(output.contains("- keyword: recall@5 1.000, mrr@10 1.000, p50 2ms, p95 3ms"));
        assert!(output.contains("- deep: recall@5 0.000, mrr@10 0.000, p50 8ms, p95 12ms"));
        assert!(output.contains("queries needing attention:"));
        assert!(output.contains("[deep] trait object generic | first relevant: none"));
    }

    #[test]
    fn ignore_edit_creates_file_and_runs_configured_editor_command() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path,
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add collection");

            std::env::set_var("VISUAL", "true --wait");
            let adapter = CliAdapter::new(engine);
            let output = adapter.ignore_edit(None, "api").expect("run ignore edit");
            assert!(
                output.contains("ignore patterns updated for work/api:"),
                "unexpected output: {output}"
            );

            let ignore_path = adapter
                .engine
                .config()
                .config_dir
                .join("ignores")
                .join("work")
                .join("api.ignore");
            assert!(ignore_path.exists(), "ignore file should exist");
        });
    }

    #[test]
    fn models_list_reports_configured_models() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);
            let embed_model = adapter.engine.config().models.embedder.id.clone();
            let reranker_model = adapter.engine.config().models.reranker.id.clone();
            let expander_model = adapter.engine.config().models.expander.id.clone();

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

            let embed_model = adapter.engine.config().models.embedder.id.clone();
            let reranker_model = adapter.engine.config().models.reranker.id.clone();
            let expander_model = adapter.engine.config().models.expander.id.clone();
            seed_model_artifact(&model_dir, "embedder", &embed_model, b"e");
            seed_model_artifact(&model_dir, "reranker", &reranker_model, b"r");
            seed_model_artifact(&model_dir, "expander", &expander_model, b"x");

            let output = adapter.models_pull().expect("pull models");
            assert!(
                output.contains("downloaded: 0"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("downloaded_models:\n- none"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("already_present: 3"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains(&format!(
                    "already present embedder: {embed_model} (1 bytes)"
                )),
                "unexpected output: {output}"
            );
            assert!(
                output.contains(&format!(
                    "already present reranker: {reranker_model} (1 bytes)"
                )),
                "unexpected output: {output}"
            );
            assert!(
                output.contains(&format!(
                    "already present expander: {expander_model} (1 bytes)"
                )),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("total_bytes: 0"),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn search_rejects_conflicting_mode_flags() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let err = adapter
                .search(CliSearchOptions {
                    space: None,
                    query: "alpha",
                    collections: &[],
                    limit: 10,
                    min_score: 0.0,
                    deep: true,
                    keyword: true,
                    semantic: false,
                    rerank: false,
                    no_rerank: false,
                    debug: false,
                })
                .expect_err("conflicting search flags should fail");
            assert!(
                err.to_string()
                    .contains("only one of --deep, --keyword, or --semantic"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn resolve_no_rerank_for_mode_defaults_auto_off_and_allows_opt_in() {
        assert!(resolve_no_rerank_for_mode(SearchMode::Auto, false, false));
        assert!(!resolve_no_rerank_for_mode(SearchMode::Auto, true, false));
    }

    #[test]
    fn resolve_no_rerank_for_mode_keeps_existing_mode_specific_behavior() {
        assert!(!resolve_no_rerank_for_mode(SearchMode::Deep, false, false));
        assert!(resolve_no_rerank_for_mode(SearchMode::Deep, false, true));
        assert!(resolve_no_rerank_for_mode(SearchMode::Keyword, true, false));
        assert!(resolve_no_rerank_for_mode(
            SearchMode::Semantic,
            true,
            false
        ));
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
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
                .expect("add collection");

            fs::write(collection_path.join("a.md"), "alpha query token\n").expect("write file");
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            let output = adapter
                .search(CliSearchOptions {
                    space: Some("work"),
                    query: "alpha",
                    collections: &["api".to_string()],
                    limit: 5,
                    min_score: 0.0,
                    deep: false,
                    keyword: true,
                    semantic: false,
                    rerank: false,
                    no_rerank: false,
                    debug: true,
                })
                .expect("run search");
            assert!(
                output.contains("query: alpha"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("mode: keyword"),
                "unexpected output: {output}"
            );
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
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
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
    fn update_verbose_reports_buffered_decisions_before_summary() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let collection_path = new_collection_dir(root.path(), "work-api");
            engine
                .add_collection(AddCollectionRequest {
                    path: collection_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: Some(vec!["rs".to_string()]),
                    no_index: true,
                })
                .expect("add collection");
            let adapter = CliAdapter::new(engine);

            fs::create_dir_all(collection_path.join("src")).expect("create src dir");
            fs::write(collection_path.join("src/lib.rs"), "fn alpha() {}\n")
                .expect("write valid file");
            fs::write(collection_path.join("src/bad.rs"), [0xff, 0xfe, 0xfd])
                .expect("write invalid file");

            let output = adapter
                .update(Some("work"), &["api".to_string()], true, false, true)
                .expect("run verbose update");

            let first_line = output.lines().next().expect("expected output lines");
            assert!(
                first_line.starts_with("work/api/"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("work/api/src/lib.rs: new"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("work/api/src/bad.rs: extract_failed (extract failed:"),
                "unexpected output: {output}"
            );

            let summary_index = output
                .lines()
                .position(|line| line.starts_with("scanned: "))
                .expect("expected summary output");
            assert!(summary_index > 0, "unexpected output: {output}");
            assert_eq!(
                output.match_indices("src/bad.rs").count(),
                1,
                "extract failure should not be duplicated: {output}"
            );
            assert!(output.contains("added: 2"), "unexpected output: {output}");
            assert!(output.contains("errors: 1"), "unexpected output: {output}");
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
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
                .expect("add collection");

            let file_path = collection_path.join("src/lib.rs");
            fs::create_dir_all(file_path.parent().expect("file parent")).expect("create parent");
            fs::write(&file_path, "fn alpha() {}\n").expect("write file");

            let output = adapter
                .update(Some("work"), &["api".to_string()], true, true, false)
                .expect("run dry-run update");
            assert!(output.contains("added: 1"), "unexpected output: {output}");

            let space = adapter
                .engine
                .storage()
                .get_space("work")
                .expect("get space");
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
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
                .expect("add collection");

            fs::create_dir_all(collection_path.join("src")).expect("create src dir");
            fs::write(collection_path.join("src/lib.rs"), "fn alpha() {}\n").expect("write src");
            fs::create_dir_all(collection_path.join("docs")).expect("create docs dir");
            fs::write(collection_path.join("docs/guide.md"), "guide\n").expect("write docs");
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            let space = adapter
                .engine
                .storage()
                .get_space("work")
                .expect("get space");
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
            assert!(
                default_output.contains("files:"),
                "unexpected output: {default_output}"
            );
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
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
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
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
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
            assert!(
                output.contains("stale: false"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("total_lines: 3"),
                "unexpected output: {output}"
            );
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
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
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
            assert!(
                output.contains("stale: true"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("content:\nfn beta() {}"),
                "unexpected output: {output}"
            );

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
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
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
            assert!(
                output.contains("documents: 2"),
                "unexpected output: {output}"
            );
            assert!(output.contains("omitted: 1"), "unexpected output: {output}");
            assert!(
                output.contains("reason: max_files"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("resolved_count: 3"),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn multi_get_reports_deleted_files_as_warnings() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
                .expect("add collection");

            let existing = collection_path.join("a.md");
            let deleted = collection_path.join("b.md");
            fs::write(&existing, "alpha\n").expect("write a");
            fs::write(&deleted, "beta\n").expect("write b");
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            fs::remove_file(&deleted).expect("remove b");

            let output = adapter
                .multi_get(
                    Some("work"),
                    &["api/a.md".to_string(), "api/b.md".to_string()],
                    10,
                    51_200,
                )
                .expect("run multi-get");
            assert!(
                output.contains("documents: 1"),
                "unexpected output: {output}"
            );
            assert!(output.contains("omitted: 0"), "unexpected output: {output}");
            assert!(
                output.contains("resolved_count: 1"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("warnings: 1"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("file deleted since indexing:"),
                "unexpected output: {output}"
            );
            assert!(output.contains("b.md"), "unexpected output: {output}");
        });
    }

    #[test]
    fn multi_get_reports_missing_and_invalid_locators_as_warnings() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            adapter
                .space_add("work", None, false, &[])
                .expect("add work");
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
                .expect("add collection");

            fs::write(collection_path.join("a.md"), "alpha\n").expect("write a");
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            let output = adapter
                .multi_get(
                    Some("work"),
                    &[
                        "api/a.md".to_string(),
                        "api/missing.md".to_string(),
                        "api/../bad.md".to_string(),
                    ],
                    10,
                    51_200,
                )
                .expect("run multi-get");
            assert!(
                output.contains("documents: 1"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("resolved_count: 1"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("warnings: 2"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("document not found: api/missing.md"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("invalid locator: path locator must not traverse directories"),
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
            let collection_path = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &collection_path,
                    Some("api"),
                    None,
                    None,
                    true,
                )
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
            assert!(output.contains("  - api ("), "unexpected output: {output}");
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
            assert!(
                output.contains("config_dir:"),
                "unexpected output: {output}"
            );
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

            let work_collection = new_collection_dir(root.path(), "work-api");
            adapter
                .collection_add(
                    Some("work"),
                    &work_collection,
                    Some("api"),
                    None,
                    None,
                    true,
                )
                .expect("add work collection");
            let notes_collection = new_collection_dir(root.path(), "notes-wiki");
            adapter
                .collection_add(
                    Some("notes"),
                    &notes_collection,
                    Some("wiki"),
                    None,
                    None,
                    true,
                )
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

    #[test]
    fn format_schedule_add_response_renders_trigger_scope_and_backend() {
        let output = format_schedule_add_response(&ScheduleAddResponse {
            schedule: ScheduleDefinition {
                id: "s1".to_string(),
                trigger: ScheduleTrigger::Every {
                    interval: ScheduleInterval {
                        value: 30,
                        unit: ScheduleIntervalUnit::Minutes,
                    },
                },
                scope: ScheduleScope::All,
            },
            backend: ScheduleBackend::Launchd,
        });

        assert_eq!(
            output,
            "schedule added: s1\ntrigger: every 30m\nscope: all spaces\nbackend: launchd"
        );
    }

    #[test]
    fn format_schedule_status_response_renders_entries_and_orphans() {
        let output = format_schedule_status_response(&ScheduleStatusResponse {
            schedules: vec![ScheduleStatusEntry {
                schedule: ScheduleDefinition {
                    id: "s2".to_string(),
                    trigger: ScheduleTrigger::Weekly {
                        weekdays: vec![ScheduleWeekday::Mon, ScheduleWeekday::Fri],
                        time: "15:00".to_string(),
                    },
                    scope: ScheduleScope::Collections {
                        space: "work".to_string(),
                        collections: vec!["api".to_string(), "docs".to_string()],
                    },
                },
                backend: ScheduleBackend::Launchd,
                state: ScheduleState::Drifted,
                run_state: ScheduleRunState {
                    last_started: Some("2026-03-07T20:00:00Z".to_string()),
                    last_finished: Some("2026-03-07T20:00:05Z".to_string()),
                    last_result: Some(ScheduleRunResult::SkippedLock),
                    last_error: None,
                },
            }],
            orphans: vec![ScheduleOrphan {
                id: "s9".to_string(),
                backend: ScheduleBackend::Launchd,
            }],
        });

        assert!(output.contains(
            "schedules:\n- s2 | mon,fri at 3:00 PM | work/api, work/docs | launchd | drifted"
        ));
        assert!(output.contains("last_result: skipped_lock"));
        assert!(output.contains("orphans:\n- s9 (launchd)"));
    }
}
