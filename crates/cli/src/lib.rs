pub mod args;

use std::path::Path;

use kbolt_core::engine::Engine;
use kbolt_core::Result;
use kbolt_types::{
    ActiveSpaceSource, AddCollectionRequest, AddCollectionResult, AddScheduleRequest,
    DoctorCheckStatus, DoctorReport, DoctorSetupStatus, EvalImportReport, EvalRunReport,
    GetRequest, InitialIndexingBlock, InitialIndexingOutcome, KboltError, LocalAction, LocalReport,
    Locator, ModelInfo, MultiGetRequest, OmitReason, RemoveScheduleRequest, ScheduleAddResponse,
    ScheduleBackend, ScheduleInterval, ScheduleIntervalUnit, ScheduleRunResult, ScheduleScope,
    ScheduleState, ScheduleStatusResponse, ScheduleTrigger, ScheduleWeekday, SearchMode,
    SearchPipeline, SearchPipelineNotice, SearchPipelineStep, SearchPipelineUnavailableReason,
    SearchRequest, UpdateDecision, UpdateDecisionKind, UpdateOptions, UpdateReport,
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
        &mut self,
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
                    info.collection.space,
                    info.collection.name
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
        lines.push(format!("collections registered: {}", successes.len()));
        lines.extend(successes);
        if !failures.is_empty() {
            lines.push(format!("collections failed: {}", failures.len()));
            lines.extend(failures);
        }
        lines.push(format!(
            "note: collections were registered without indexing; run `kbolt --space {} update` to index them",
            added.name
        ));

        Ok(lines.join("\n"))
    }

    pub fn space_describe(&self, name: &str, text: &str) -> Result<String> {
        self.engine.describe_space(name, text)?;
        Ok(format!("space description updated: {name}"))
    }

    pub fn space_rename(&mut self, old: &str, new: &str) -> Result<String> {
        self.engine.rename_space(old, new)?;
        Ok(format!("space renamed: {old} -> {new}"))
    }

    pub fn space_remove(&mut self, name: &str) -> Result<String> {
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

        Ok(format_collection_add_result(&added))
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
            ("embedder", &status.embedder),
            ("reranker", &status.reranker),
            ("expander", &status.expander),
        ] {
            lines.push(format!("- {label}: {}", format_model_binding_summary(info)));
        }

        Ok(lines.join("\n"))
    }

    pub fn eval_run(&self, eval_file: Option<&Path>) -> Result<String> {
        let report = self.engine.run_eval(eval_file)?;
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
            "requested_mode: {}",
            format_search_mode(&response.requested_mode)
        ));
        lines.push(format!(
            "effective_mode: {}",
            format_search_mode(&response.effective_mode)
        ));
        lines.push(format!(
            "pipeline: {}",
            format_search_pipeline(&response.pipeline)
        ));
        for notice in &response.pipeline.notices {
            lines.push(format!("note: {}", format_search_pipeline_notice(notice)));
        }
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
                    "signals: bm25={:?} dense={:?} fusion={:.3} reranker={:?}",
                    signals.bm25, signals.dense, signals.fusion, signals.reranker
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
            "model_embedder: {}",
            format_model_binding_summary(&status.models.embedder)
        ));
        lines.push(format!(
            "model_reranker: {}",
            format_model_binding_summary(&status.models.reranker)
        ));
        lines.push(format!(
            "model_expander: {}",
            format_model_binding_summary(&status.models.expander)
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

pub fn format_doctor_report(report: &DoctorReport) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "setup: {}",
        format_doctor_setup_status(report.setup_status)
    ));
    lines.push(format!("ready: {}", report.ready));
    if let Some(path) = report.config_file.as_ref() {
        lines.push(format!("config_file: {}", path.display()));
    }
    if let Some(path) = report.config_dir.as_ref() {
        lines.push(format!("config_dir: {}", path.display()));
    }
    if let Some(path) = report.cache_dir.as_ref() {
        lines.push(format!("cache_dir: {}", path.display()));
    }
    lines.push("checks:".to_string());
    for check in &report.checks {
        lines.push(format!(
            "- [{}] {} {} ({}ms): {}",
            format_doctor_check_status(check.status),
            check.scope,
            check.id,
            check.elapsed_ms,
            check.message
        ));
        if let Some(fix) = check.fix.as_deref() {
            lines.push(format!("  fix: {fix}"));
        }
    }
    lines.join("\n")
}

pub fn format_local_report(report: &LocalReport) -> String {
    let mut lines = Vec::new();
    lines.push(format!("action: {}", format_local_action(report.action)));
    lines.push(format!("ready: {}", report.ready));
    lines.push(format!("config_file: {}", report.config_file.display()));
    lines.push(format!("cache_dir: {}", report.cache_dir.display()));
    if let Some(path) = report.llama_server_path.as_ref() {
        lines.push(format!("llama_server: {}", path.display()));
    } else {
        lines.push("llama_server: missing".to_string());
    }
    if !report.notes.is_empty() {
        lines.push("notes:".to_string());
        for note in &report.notes {
            lines.push(format!("- {note}"));
        }
    }
    lines.push("services:".to_string());
    for service in &report.services {
        lines.push(format!(
            "- {}: {} | configured={} | enabled={} | managed={} | running={} | ready={} | model={} | endpoint={} | model_path={} | pid={} | pid_file={} | log_file={}",
            service.name,
            service.provider,
            service.configured,
            service.enabled,
            service.managed,
            service.running,
            service.ready,
            service.model,
            service.endpoint,
            service.model_path.display(),
            service
                .pid
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            service.pid_file.display(),
            service.log_file.display()
        ));
        if let Some(issue) = service.issue.as_deref() {
            lines.push(format!("  issue: {issue}"));
        }
    }
    lines.join("\n")
}

fn format_doctor_setup_status(status: DoctorSetupStatus) -> &'static str {
    match status {
        DoctorSetupStatus::ConfigMissing => "config_missing",
        DoctorSetupStatus::ConfigInvalid => "config_invalid",
        DoctorSetupStatus::NotConfigured => "not_configured",
        DoctorSetupStatus::Configured => "configured",
    }
}

fn format_local_action(action: LocalAction) -> &'static str {
    match action {
        LocalAction::Setup => "setup",
        LocalAction::Start => "start",
        LocalAction::Stop => "stop",
        LocalAction::Status => "status",
        LocalAction::EnableDeep => "enable_deep",
    }
}

fn format_doctor_check_status(status: DoctorCheckStatus) -> &'static str {
    match status {
        DoctorCheckStatus::Pass => "PASS",
        DoctorCheckStatus::Warn => "WARN",
        DoctorCheckStatus::Fail => "FAIL",
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

fn format_model_binding_summary(info: &ModelInfo) -> String {
    let mut parts = vec![if !info.configured {
        "unconfigured".to_string()
    } else if info.ready {
        "ready".to_string()
    } else {
        "not_ready".to_string()
    }];

    if let Some(profile) = info.profile.as_deref() {
        parts.push(format!("profile={profile}"));
    }
    if let Some(kind) = info.kind.as_deref() {
        parts.push(format!("kind={kind}"));
    }
    if let Some(operation) = info.operation.as_deref() {
        parts.push(format!("operation={operation}"));
    }
    if let Some(model) = info.model.as_deref() {
        parts.push(format!("model={model}"));
    }
    if let Some(endpoint) = info.endpoint.as_deref() {
        parts.push(format!("endpoint={endpoint}"));
    }
    if let Some(issue) = info.issue.as_deref() {
        parts.push(format!("issue={issue}"));
    }

    parts.join(" | ")
}

fn format_search_pipeline(pipeline: &SearchPipeline) -> String {
    let mut parts = Vec::new();
    if pipeline.expansion {
        parts.push("expansion");
    }
    if pipeline.keyword {
        parts.push("keyword");
    }
    if pipeline.dense {
        parts.push("dense");
    }
    if pipeline.rerank {
        parts.push("rerank");
    }

    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(" + ")
    }
}

fn format_search_pipeline_notice(notice: &SearchPipelineNotice) -> String {
    let step = match notice.step {
        SearchPipelineStep::Dense => "dense retrieval",
        SearchPipelineStep::Rerank => "rerank",
    };
    let reason = match notice.reason {
        SearchPipelineUnavailableReason::NotConfigured => "not configured",
        SearchPipelineUnavailableReason::ModelNotAvailable => "required provider is not ready",
    };
    format!("{step} unavailable: {reason}")
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

    lines.push(format!("scanned_docs: {}", report.scanned_docs));
    lines.push(format!("skipped_mtime_docs: {}", report.skipped_mtime_docs));
    lines.push(format!("skipped_hash_docs: {}", report.skipped_hash_docs));
    lines.push(format!("added_docs: {}", report.added_docs));
    lines.push(format!("updated_docs: {}", report.updated_docs));
    lines.push(format!("failed_docs: {}", report.failed_docs));
    lines.push(format!("deactivated_docs: {}", report.deactivated_docs));
    lines.push(format!("reactivated_docs: {}", report.reactivated_docs));
    lines.push(format!("reaped_docs: {}", report.reaped_docs));
    lines.push(format!("embedded_chunks: {}", report.embedded_chunks));
    lines.push(format!("errors: {}", report.errors.len()));
    lines.push(format!("elapsed_ms: {}", report.elapsed_ms));
    lines.join("\n")
}

fn format_collection_add_result(result: &AddCollectionResult) -> String {
    let collection = &result.collection;
    let locator = format!("{}/{}", collection.space, collection.name);

    match &result.initial_indexing {
        InitialIndexingOutcome::Skipped => {
            format!("collection added without indexing: {locator}")
        }
        InitialIndexingOutcome::Indexed(report) => {
            format_collection_add_indexing_report(collection, &locator, report)
        }
        InitialIndexingOutcome::Blocked(block) => {
            format_collection_add_block(collection, &locator, block)
        }
    }
}

fn format_collection_add_indexing_report(
    collection: &kbolt_types::CollectionInfo,
    locator: &str,
    report: &UpdateReport,
) -> String {
    let mut lines = Vec::new();
    if report.failed_docs == 0 {
        lines.push(format!("collection added and indexed: {locator}"));
    } else {
        lines.push(format!("collection added: {locator}"));
        lines.push("initial indexing incomplete".to_string());
    }

    lines.push(format!("scanned_docs: {}", report.scanned_docs));
    lines.push(format!("added_docs: {}", report.added_docs));
    lines.push(format!("updated_docs: {}", report.updated_docs));
    lines.push(format!("failed_docs: {}", report.failed_docs));

    if report.failed_docs > 0 {
        lines.push(format!(
            "rerun: kbolt --space {} update --collection {}",
            collection.space, collection.name
        ));
    }

    lines.join("\n")
}

fn format_collection_add_block(
    collection: &kbolt_types::CollectionInfo,
    locator: &str,
    block: &InitialIndexingBlock,
) -> String {
    let mut lines = Vec::new();
    lines.push(format!("collection added: {locator}"));

    match block {
        InitialIndexingBlock::SpaceDenseRepairRequired { space, reason } => {
            lines.push(format!(
                "initial indexing blocked by space-level dense integrity issue in '{space}'"
            ));
            lines.push(format!("reason: {reason}"));
            lines.push(format!("run: kbolt --space {space} update"));
        }
        InitialIndexingBlock::ModelNotAvailable { name } => {
            lines.push(format!(
                "initial indexing blocked: model '{name}' is not available"
            ));
            lines.push("run: kbolt setup local".to_string());
            lines.push("or configure [roles.embedder] in index.toml".to_string());
            lines.push(format!(
                "then run: kbolt --space {} update --collection {}",
                collection.space, collection.name
            ));
        }
    }

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
            "- {}: ndcg@10 {:.3}, recall@10 {:.3}, mrr@10 {:.3}, p50 {}ms, p95 {}ms",
            format_eval_mode_label(&mode.mode, mode.no_rerank),
            mode.ndcg_at_10,
            mode.recall_at_10,
            mode.mrr_at_10,
            mode.latency_p50_ms,
            mode.latency_p95_ms
        ));
    }
    for failure in &report.failed_modes {
        lines.push(format!(
            "- {}: failed ({})",
            format_eval_mode_label(&failure.mode, failure.no_rerank),
            failure.error
        ));
    }

    let findings = report
        .modes
        .iter()
        .flat_map(|mode| {
            mode.queries.iter().filter_map(|query| {
                let perfect_recall = query.matched_paths.len() == relevant_judgment_count(query);
                let perfect_rank = query.first_relevant_rank == Some(1);
                if perfect_recall && perfect_rank {
                    return None;
                }

                Some(format!(
                    "- [{}] {} | first relevant: {} | expected: {} | returned: {}",
                    format_eval_mode_label(&mode.mode, mode.no_rerank),
                    query.query,
                    query
                        .first_relevant_rank
                        .map(|rank| rank.to_string())
                        .unwrap_or_else(|| "none".to_string()),
                    format_eval_judgments(&query.judgments),
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

pub fn format_eval_import_report(report: &EvalImportReport) -> String {
    [
        format!("imported benchmark: {}", report.dataset),
        format!("source: {}", report.source),
        format!("output: {}", report.output_dir),
        format!("corpus_dir: {}", report.corpus_dir),
        format!("manifest: {}", report.manifest_path),
        format!("documents: {}", report.document_count),
        format!("queries: {}", report.query_count),
        format!("judgments: {}", report.judgment_count),
        "next:".to_string(),
        format!(
            "- create the benchmark space if needed: kbolt space add {}",
            report.default_space
        ),
        format!(
            "- register the corpus: kbolt --space {} collection add {} --name {} --no-index",
            report.default_space, report.corpus_dir, report.collection
        ),
        format!(
            "- index it: kbolt --space {} update --collection {}",
            report.default_space, report.collection
        ),
        format!("- run eval: kbolt eval run --file {}", report.manifest_path),
    ]
    .join("\n")
}

fn relevant_judgment_count(query: &kbolt_types::EvalQueryReport) -> usize {
    query
        .judgments
        .iter()
        .filter(|judgment| judgment.relevance > 0)
        .count()
}

fn format_eval_judgments(judgments: &[kbolt_types::EvalJudgment]) -> String {
    judgments
        .iter()
        .map(|judgment| format!("{}(rel={})", judgment.path, judgment.relevance))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_eval_mode_label(mode: &SearchMode, no_rerank: bool) -> &'static str {
    match (mode, no_rerank) {
        (SearchMode::Keyword, _) => "keyword",
        (SearchMode::Auto, true) => "auto",
        (SearchMode::Auto, false) => "auto+rerank",
        (SearchMode::Semantic, _) => "semantic",
        (SearchMode::Deep, true) => "deep-norerank",
        (SearchMode::Deep, false) => "deep",
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
        format_collection_add_result, format_doctor_report, format_eval_import_report,
        format_eval_run_report, format_local_report, format_schedule_add_response,
        format_schedule_status_response, parse_editor_command, resolve_editor_command,
        resolve_no_rerank_for_mode, CliAdapter, CliSearchOptions,
    };
    use kbolt_core::engine::Engine;
    use kbolt_types::{
        AddCollectionRequest, AddCollectionResult, CollectionInfo, DoctorCheck, DoctorCheckStatus,
        DoctorReport, DoctorSetupStatus, EvalImportReport, EvalJudgment, EvalModeReport,
        EvalQueryReport, EvalRunReport, InitialIndexingBlock, InitialIndexingOutcome, LocalAction,
        LocalReport, ScheduleAddResponse, ScheduleBackend, ScheduleDefinition, ScheduleInterval,
        ScheduleIntervalUnit, ScheduleOrphan, ScheduleRunResult, ScheduleRunState, ScheduleScope,
        ScheduleState, ScheduleStatusEntry, ScheduleStatusResponse, ScheduleTrigger,
        ScheduleWeekday, SearchMode, UpdateReport,
    };

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
        std::env::set_var("XDG_CONFIG_HOME", root.path().join("config-home"));
        std::env::set_var("XDG_CACHE_HOME", root.path().join("cache-home"));

        run()
    }

    fn new_collection_dir(root: &Path, name: &str) -> PathBuf {
        let path = root.join(name);
        fs::create_dir_all(&path).expect("create collection directory");
        path
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
                    no_rerank: true,
                    ndcg_at_10: 1.0,
                    recall_at_10: 1.0,
                    mrr_at_10: 1.0,
                    latency_p50_ms: 2,
                    latency_p95_ms: 3,
                    queries: vec![EvalQueryReport {
                        query: "trait object generic".to_string(),
                        space: Some("default".to_string()),
                        collections: vec!["rust".to_string()],
                        judgments: vec![EvalJudgment {
                            path: "rust/guides/traits.md".to_string(),
                            relevance: 1,
                        }],
                        returned_paths: vec!["rust/guides/traits.md".to_string()],
                        matched_paths: vec!["rust/guides/traits.md".to_string()],
                        first_relevant_rank: Some(1),
                        elapsed_ms: 2,
                    }],
                },
                EvalModeReport {
                    mode: SearchMode::Deep,
                    no_rerank: false,
                    ndcg_at_10: 0.0,
                    recall_at_10: 0.0,
                    mrr_at_10: 0.0,
                    latency_p50_ms: 8,
                    latency_p95_ms: 12,
                    queries: vec![EvalQueryReport {
                        query: "trait object generic".to_string(),
                        space: Some("default".to_string()),
                        collections: vec!["rust".to_string()],
                        judgments: vec![EvalJudgment {
                            path: "rust/guides/traits.md".to_string(),
                            relevance: 1,
                        }],
                        returned_paths: vec!["rust/overview.md".to_string()],
                        matched_paths: vec![],
                        first_relevant_rank: None,
                        elapsed_ms: 8,
                    }],
                },
            ],
            failed_modes: vec![kbolt_types::EvalModeFailure {
                mode: SearchMode::Semantic,
                no_rerank: true,
                error: "model not available".to_string(),
            }],
        });

        assert!(output
            .contains("- keyword: ndcg@10 1.000, recall@10 1.000, mrr@10 1.000, p50 2ms, p95 3ms"));
        assert!(output
            .contains("- deep: ndcg@10 0.000, recall@10 0.000, mrr@10 0.000, p50 8ms, p95 12ms"));
        assert!(output.contains("- semantic: failed (model not available)"));
        assert!(output.contains("queries needing attention:"));
        assert!(output.contains("[deep] trait object generic | first relevant: none"));
    }

    #[test]
    fn eval_import_report_formats_next_steps() {
        let output = format_eval_import_report(&EvalImportReport {
            dataset: "scifact".to_string(),
            source: "/tmp/scifact-source".to_string(),
            output_dir: "/tmp/scifact-bench".to_string(),
            corpus_dir: "/tmp/scifact-bench/corpus".to_string(),
            manifest_path: "/tmp/scifact-bench/eval.toml".to_string(),
            default_space: "bench".to_string(),
            collection: "scifact".to_string(),
            document_count: 2,
            query_count: 2,
            judgment_count: 3,
        });

        assert!(output.contains("imported benchmark: scifact"));
        assert!(output.contains("documents: 2"));
        assert!(output.contains("queries: 2"));
        assert!(output.contains("judgments: 3"));
        assert!(output.contains("kbolt space add bench"));
        assert!(output.contains("kbolt eval run --file /tmp/scifact-bench/eval.toml"));
    }

    #[test]
    fn models_list_reports_role_binding_readiness() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = CliAdapter::new(engine);

            let output = adapter.models_list().expect("list models");
            assert!(output.contains("models:"), "unexpected output: {output}");
            assert!(
                output.contains("- embedder: unconfigured"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("- reranker: unconfigured"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("- expander: unconfigured"),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn doctor_report_formats_status_checks_and_fixes() {
        let output = format_doctor_report(&DoctorReport {
            setup_status: DoctorSetupStatus::Configured,
            config_file: Some(PathBuf::from("/tmp/kbolt/index.toml")),
            config_dir: Some(PathBuf::from("/tmp/kbolt")),
            cache_dir: Some(PathBuf::from("/tmp/cache/kbolt")),
            ready: false,
            checks: vec![
                DoctorCheck {
                    id: "config.file_parses".to_string(),
                    scope: "config".to_string(),
                    status: DoctorCheckStatus::Pass,
                    elapsed_ms: 2,
                    message: "ok".to_string(),
                    fix: None,
                },
                DoctorCheck {
                    id: "roles.embedder.reachable".to_string(),
                    scope: "roles.embedder".to_string(),
                    status: DoctorCheckStatus::Fail,
                    elapsed_ms: 17,
                    message: "llama_cpp_server endpoint is unreachable".to_string(),
                    fix: Some("Start the embedding server.".to_string()),
                },
            ],
        });

        assert!(output.contains("setup: configured"));
        assert!(output.contains("ready: false"));
        assert!(output.contains("config_file: /tmp/kbolt/index.toml"));
        assert!(
            output.contains("- [PASS] config config.file_parses (2ms): ok"),
            "unexpected output: {output}"
        );
        assert!(
            output.contains(
                "- [FAIL] roles.embedder roles.embedder.reachable (17ms): llama_cpp_server endpoint is unreachable"
            ),
            "unexpected output: {output}"
        );
        assert!(output.contains("  fix: Start the embedding server."));
    }

    #[test]
    fn local_report_formats_service_state_and_notes() {
        let output = format_local_report(&LocalReport {
            action: LocalAction::Setup,
            config_file: PathBuf::from("/tmp/kbolt/index.toml"),
            cache_dir: PathBuf::from("/tmp/cache/kbolt"),
            llama_server_path: Some(PathBuf::from("/opt/homebrew/bin/llama-server")),
            ready: false,
            notes: vec!["started embedder on http://127.0.0.1:8101".to_string()],
            services: vec![kbolt_types::LocalServiceReport {
                name: "embedder".to_string(),
                provider: "kbolt_local_embed".to_string(),
                enabled: true,
                configured: true,
                managed: true,
                running: true,
                ready: false,
                model: "embeddinggemma".to_string(),
                model_path: PathBuf::from("/tmp/cache/kbolt/models/embedder/model.gguf"),
                endpoint: "http://127.0.0.1:8101".to_string(),
                port: 8101,
                pid: Some(42),
                pid_file: PathBuf::from("/tmp/cache/kbolt/run/embedder.pid"),
                log_file: PathBuf::from("/tmp/cache/kbolt/logs/embedder.log"),
                issue: Some("service is not ready".to_string()),
            }],
        });

        assert!(output.contains("action: setup"));
        assert!(output.contains("llama_server: /opt/homebrew/bin/llama-server"));
        assert!(output.contains("- started embedder on http://127.0.0.1:8101"));
        assert!(output.contains("configured=true"));
        assert!(output.contains("issue: service is not ready"));
    }

    #[test]
    fn search_rejects_conflicting_mode_flags() {
        with_isolated_xdg_dirs(|| {
            let adapter = CliAdapter::new(Engine::new(None).expect("create engine"));

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
    fn resolve_no_rerank_for_mode_matches_cli_contract() {
        assert!(resolve_no_rerank_for_mode(SearchMode::Auto, false, false));
        assert!(!resolve_no_rerank_for_mode(SearchMode::Auto, true, false));
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
    fn search_reports_requested_and_effective_mode_for_auto_keyword_fallback() {
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
                .expect("add collection");
            fs::write(work_path.join("a.md"), "fallback token\n").expect("write file");

            let adapter = CliAdapter::new(engine);
            adapter
                .update(Some("work"), &["api".to_string()], true, false, false)
                .expect("run update");

            let output = adapter
                .search(CliSearchOptions {
                    space: Some("work"),
                    query: "fallback",
                    collections: &["api".to_string()],
                    limit: 5,
                    min_score: 0.0,
                    deep: false,
                    keyword: false,
                    semantic: false,
                    rerank: false,
                    no_rerank: false,
                    debug: false,
                })
                .expect("run auto search");

            assert!(
                output.contains("requested_mode: auto"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("effective_mode: keyword"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("pipeline: keyword"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("note: dense retrieval unavailable: not configured"),
                "unexpected output: {output}"
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

            let summary_index = output
                .lines()
                .position(|line| line.starts_with("scanned_docs: "))
                .expect("expected summary output");
            assert!(summary_index > 0, "unexpected output: {output}");
            assert!(
                output
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .starts_with("work/api/"),
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
        });
    }

    #[test]
    fn collection_add_result_formats_no_index_message() {
        let output = format_collection_add_result(&AddCollectionResult {
            collection: CollectionInfo {
                name: "api".to_string(),
                space: "work".to_string(),
                path: PathBuf::from("/tmp/work-api"),
                description: None,
                extensions: None,
                document_count: 0,
                active_document_count: 0,
                chunk_count: 0,
                embedded_chunk_count: 0,
                created: "2026-03-31T00:00:00Z".to_string(),
                updated: "2026-03-31T00:00:00Z".to_string(),
            },
            initial_indexing: InitialIndexingOutcome::Skipped,
        });

        assert_eq!(output, "collection added without indexing: work/api");
    }

    #[test]
    fn collection_add_result_formats_incomplete_initial_indexing() {
        let output = format_collection_add_result(&AddCollectionResult {
            collection: CollectionInfo {
                name: "api".to_string(),
                space: "work".to_string(),
                path: PathBuf::from("/tmp/work-api"),
                description: None,
                extensions: None,
                document_count: 3,
                active_document_count: 3,
                chunk_count: 3,
                embedded_chunk_count: 2,
                created: "2026-03-31T00:00:00Z".to_string(),
                updated: "2026-03-31T00:00:00Z".to_string(),
            },
            initial_indexing: InitialIndexingOutcome::Indexed(UpdateReport {
                scanned_docs: 3,
                skipped_mtime_docs: 0,
                skipped_hash_docs: 0,
                added_docs: 2,
                updated_docs: 0,
                failed_docs: 1,
                deactivated_docs: 0,
                reactivated_docs: 0,
                reaped_docs: 0,
                embedded_chunks: 2,
                decisions: Vec::new(),
                errors: Vec::new(),
                elapsed_ms: 5,
            }),
        });

        assert!(output.contains("collection added: work/api"));
        assert!(output.contains("initial indexing incomplete"));
        assert!(output.contains("scanned_docs: 3"));
        assert!(output.contains("added_docs: 2"));
        assert!(output.contains("failed_docs: 1"));
        assert!(output.contains("rerun: kbolt --space work update --collection api"));
    }

    #[test]
    fn collection_add_result_formats_model_block_with_resume_steps() {
        let output = format_collection_add_result(&AddCollectionResult {
            collection: CollectionInfo {
                name: "api".to_string(),
                space: "work".to_string(),
                path: PathBuf::from("/tmp/work-api"),
                description: None,
                extensions: None,
                document_count: 0,
                active_document_count: 0,
                chunk_count: 0,
                embedded_chunk_count: 0,
                created: "2026-03-31T00:00:00Z".to_string(),
                updated: "2026-03-31T00:00:00Z".to_string(),
            },
            initial_indexing: InitialIndexingOutcome::Blocked(
                InitialIndexingBlock::ModelNotAvailable {
                    name: "embed-model".to_string(),
                },
            ),
        });

        assert!(output.contains("collection added: work/api"));
        assert!(output.contains("initial indexing blocked: model 'embed-model' is not available"));
        assert!(output.contains("run: kbolt setup local"));
        assert!(output.contains("configure [roles.embedder] in index.toml"));
        assert!(output.contains("then run: kbolt --space work update --collection api"));
    }

    #[test]
    fn space_add_with_directories_reports_registration_without_indexing() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            let mut adapter = CliAdapter::new(engine);

            let work_path = new_collection_dir(root.path(), "work-api");
            let notes_path = new_collection_dir(root.path(), "work-notes");

            let output = adapter
                .space_add("work", Some("work docs"), false, &[work_path, notes_path])
                .expect("add space with directories");

            assert!(output.contains("space added: work - work docs"));
            assert!(output.contains("collections registered: 2"));
            assert!(output.contains("run `kbolt --space work update` to index them"));
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
