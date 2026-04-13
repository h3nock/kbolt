pub mod args;

use std::path::Path;

use kbolt_core::engine::Engine;
use kbolt_core::Result;
use kbolt_types::{
    ActiveSpaceSource, AddCollectionRequest, AddCollectionResult, AddScheduleRequest,
    DoctorCheckStatus, DoctorReport, DoctorSetupStatus, EvalImportReport, EvalRunReport, FileEntry,
    GetRequest, InitialIndexingBlock, InitialIndexingOutcome, KboltError, LocalAction, LocalReport,
    Locator, ModelInfo, MultiGetRequest, OmitReason, RemoveScheduleRequest, ScheduleAddResponse,
    ScheduleBackend, ScheduleInterval, ScheduleIntervalUnit, ScheduleRunResult, ScheduleScope,
    ScheduleState, ScheduleStatusResponse, ScheduleTrigger, ScheduleWeekday, SearchMode,
    SearchPipeline, SearchPipelineNotice, SearchPipelineStep, SearchPipelineUnavailableReason,
    SearchRequest, StatusResponse, UpdateDecision, UpdateDecisionKind, UpdateOptions, UpdateReport,
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
        Ok(format_models_list(&status))
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

        if debug {
            lines.push(format!("query: {}", response.query));
            lines.push(format!(
                "mode: {} -> {}",
                format_search_mode(&response.requested_mode),
                format_search_mode(&response.effective_mode)
            ));
            lines.push(format!(
                "pipeline: {}",
                format_search_pipeline(&response.pipeline)
            ));
            for notice in &response.pipeline.notices {
                lines.push(format!("note: {}", format_search_pipeline_notice(notice)));
            }
            lines.push(String::new());
        }

        lines.push(format!(
            "{} result{}",
            response.results.len(),
            if response.results.len() == 1 { "" } else { "s" }
        ));

        for (index, item) in response.results.iter().enumerate() {
            lines.push(String::new());
            if debug {
                lines.push(format!(
                    "{}. {} score={:.3}",
                    index + 1,
                    item.docid,
                    item.score
                ));
            } else {
                lines.push(format!("{}. {}", index + 1, item.title));
            }
            lines.push(format!(
                "   {}",
                format_search_result_path(&item.space, &item.path)
            ));
            if !debug {
                lines.push(format!("   score: {:.2}", item.score));
            }
            if let Some(heading) = &item.heading {
                lines.push(format!("   heading: {heading}"));
            }
            lines.push(String::new());
            let snippet = truncate_snippet(&item.text, 4);
            for snippet_line in snippet.lines() {
                lines.push(format!("   {snippet_line}"));
            }
            if let Some(signals) = &item.signals {
                lines.push(String::new());
                lines.push(format!(
                    "   signals: bm25={} dense={} fusion={} reranker={}",
                    format_optional_search_signal(signals.bm25),
                    format_optional_search_signal(signals.dense),
                    format_search_signal(signals.fusion),
                    format_optional_search_signal(signals.reranker)
                ));
            }
        }

        if let Some(hint) = response.staleness_hint {
            lines.push(String::new());
            lines.push(hint);
        }

        if debug {
            lines.push(format!("elapsed: {}ms", response.elapsed_ms));
        }

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
        let active_space = active_space_name_for_status(&self.engine, space);
        Ok(format_status_response(&status, active_space.as_deref()))
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

        Ok(format_file_list(&files, all))
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

    let failures: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == DoctorCheckStatus::Fail)
        .collect();
    let warnings: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == DoctorCheckStatus::Warn)
        .collect();

    match report.setup_status {
        DoctorSetupStatus::ConfigMissing => {
            lines.push("kbolt is not set up".to_string());
            lines.push(String::new());
            lines.push("get started:".to_string());
            lines.push("  kbolt setup local".to_string());
            return lines.join("\n");
        }
        DoctorSetupStatus::ConfigInvalid => {
            lines.push("kbolt configuration is invalid".to_string());
            for check in &failures {
                lines.push(String::new());
                lines.push(format!("  {}", check.message));
                if let Some(fix) = check.fix.as_deref() {
                    lines.push(format!("  fix: {fix}"));
                }
            }
            if let Some(path) = report.config_file.as_ref() {
                lines.push(String::new());
                lines.push(format!("config: {}", path.display()));
            }
            return lines.join("\n");
        }
        DoctorSetupStatus::NotConfigured => {
            lines.push("kbolt is installed but no inference roles are configured".to_string());
            lines.push(String::new());
            lines.push("get started:".to_string());
            lines.push("  kbolt setup local".to_string());
            return lines.join("\n");
        }
        DoctorSetupStatus::Configured => {}
    }

    if report.ready && failures.is_empty() {
        lines.push("kbolt is ready".to_string());
    } else {
        lines.push("kbolt has issues".to_string());
    }

    let configured_roles: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.id.ends_with(".bound") && c.status == DoctorCheckStatus::Pass)
        .collect();
    if !configured_roles.is_empty() {
        lines.push(String::new());
        lines.push("configured:".to_string());
        for check in &configured_roles {
            let role = check.scope.strip_prefix("roles.").unwrap_or(&check.scope);
            lines.push(format!("  {role}"));
        }
    }

    let not_enabled: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.id.ends_with(".bound") && c.status == DoctorCheckStatus::Warn)
        .collect();
    if !not_enabled.is_empty() {
        lines.push(String::new());
        lines.push("not enabled:".to_string());
        for check in &not_enabled {
            let role = check.scope.strip_prefix("roles.").unwrap_or(&check.scope);
            lines.push(format!("  {role}"));
        }
    }

    if !failures.is_empty() {
        lines.push(String::new());
        lines.push("failures:".to_string());
        for check in &failures {
            lines.push(format!("  {}: {}", check.id, check.message));
            if let Some(fix) = check.fix.as_deref() {
                lines.push(format!("  fix: {fix}"));
            }
        }
    }

    if !warnings.is_empty() && failures.is_empty() {
        // Only show non-bound warnings if there are no failures
        let other_warnings: Vec<_> = warnings
            .iter()
            .filter(|c| !c.id.ends_with(".bound") && !c.id.ends_with(".reachable"))
            .collect();
        if !other_warnings.is_empty() {
            lines.push(String::new());
            lines.push("warnings:".to_string());
            for check in &other_warnings {
                lines.push(format!("  {}: {}", check.id, check.message));
                if let Some(fix) = check.fix.as_deref() {
                    lines.push(format!("  fix: {fix}"));
                }
            }
        }
    }

    lines.join("\n")
}

pub fn format_local_report(report: &LocalReport) -> String {
    let mut lines = Vec::new();

    let action_label = match report.action {
        LocalAction::Setup => "local setup complete",
        LocalAction::Start => "local servers started",
        LocalAction::Stop => "local servers stopped",
        LocalAction::Status => "local server status",
        LocalAction::EnableDeep => "deep search enabled",
    };

    if report.action == LocalAction::Stop || report.ready {
        lines.push(action_label.to_string());
    } else {
        lines.push(format!("{action_label} (not ready)"));
    }

    if !report.notes.is_empty() {
        lines.push(String::new());
        lines.push("notes:".to_string());
        for note in &report.notes {
            lines.push(format!("  {note}"));
        }
    }

    let ready_services: Vec<_> = if report.action == LocalAction::Stop {
        Vec::new()
    } else {
        report.services.iter().filter(|s| s.ready).collect()
    };
    let issue_services: Vec<_> = if report.action == LocalAction::Stop {
        report
            .services
            .iter()
            .filter(|s| s.configured && (s.running || s.ready))
            .collect()
    } else {
        report
            .services
            .iter()
            .filter(|s| s.configured && !s.ready)
            .collect()
    };
    let unconfigured_services: Vec<_> = report.services.iter().filter(|s| !s.configured).collect();

    if !ready_services.is_empty() {
        lines.push(String::new());
        lines.push("ready:".to_string());
        for service in &ready_services {
            lines.push(format!("  {} ({})", service.name, service.model));
        }
    }

    if !issue_services.is_empty() {
        lines.push(String::new());
        lines.push("issues:".to_string());
        for service in &issue_services {
            let issue = service.issue.as_deref().unwrap_or("not ready");
            lines.push(format!("  {}: {issue}", service.name));
        }
    }

    if !unconfigured_services.is_empty() && report.action != LocalAction::Stop {
        lines.push(String::new());
        lines.push("not configured:".to_string());
        for service in &unconfigured_services {
            lines.push(format!("  {}", service.name));
        }
    }

    lines.push(String::new());
    lines.push("config:".to_string());
    lines.push(format!("  {}", report.config_file.display()));

    if report.action == LocalAction::Setup && report.ready {
        lines.push(String::new());
        lines.push("next:".to_string());
        lines.push("  kbolt collection add /path/to/docs".to_string());
        lines.push("  kbolt doctor".to_string());
    }

    lines.join("\n")
}

fn format_search_mode(mode: &SearchMode) -> &'static str {
    match mode {
        SearchMode::Auto => "auto",
        SearchMode::Deep => "deep",
        SearchMode::Keyword => "keyword",
        SearchMode::Semantic => "semantic",
    }
}

fn format_search_result_path(space: &str, path: &str) -> String {
    format!("{space}/{path}")
}

fn format_models_list(status: &kbolt_types::ModelStatus) -> String {
    let mut lines = Vec::new();
    lines.push("models:".to_string());
    append_model_status_lines(&mut lines, "embedder", &status.embedder);
    append_model_status_lines(&mut lines, "reranker", &status.reranker);
    append_model_status_lines(&mut lines, "expander", &status.expander);
    lines.join("\n")
}

fn format_status_response(status: &StatusResponse, active_space: Option<&str>) -> String {
    let mut lines = Vec::new();

    lines.push("spaces:".to_string());
    if status.spaces.is_empty() {
        lines.push("- none".to_string());
    } else {
        for space in &status.spaces {
            let active_suffix = if Some(space.name.as_str()) == active_space {
                " (active)"
            } else {
                ""
            };
            lines.push(format!("- {}{}", space.name, active_suffix));
            if let Some(description) = space.description.as_deref() {
                if !description.is_empty() {
                    lines.push(format!("  description: {description}"));
                }
            }
            if let Some(last_updated) = space.last_updated.as_deref() {
                lines.push(format!("  updated: {last_updated}"));
            }

            if space.collections.is_empty() {
                lines.push("  collections: none".to_string());
            } else {
                lines.push("  collections:".to_string());
                for collection in &space.collections {
                    lines.push(format!("    - {}", collection.name));
                    lines.push(format!("      path: {}", collection.path.display()));
                    lines.push(format!(
                        "      documents: {} active / {} total",
                        collection.active_documents, collection.documents
                    ));
                    lines.push(format!("      chunks: {}", collection.chunks));
                    lines.push(format!("      embedded: {}", collection.embedded_chunks));
                    lines.push(format!("      updated: {}", collection.last_updated));
                }
            }
        }
    }

    lines.push(String::new());
    lines.push("totals:".to_string());
    lines.push(format!("- documents: {}", status.total_documents));
    lines.push(format!("- chunks: {}", status.total_chunks));
    lines.push(format!("- embedded: {}", status.total_embedded));

    lines.push(String::new());
    lines.push("storage:".to_string());
    lines.push(format!(
        "- sqlite: {}",
        format_bytes_human(status.disk_usage.sqlite_bytes)
    ));
    lines.push(format!(
        "- tantivy: {}",
        format_bytes_human(status.disk_usage.tantivy_bytes)
    ));
    lines.push(format!(
        "- vectors: {}",
        format_bytes_human(status.disk_usage.usearch_bytes)
    ));
    lines.push(format!(
        "- models: {}",
        format_bytes_human(status.disk_usage.models_bytes)
    ));
    lines.push(format!(
        "- total: {}",
        format_bytes_human(status.disk_usage.total_bytes)
    ));

    lines.push(String::new());
    lines.push("models:".to_string());
    append_model_status_lines(&mut lines, "embedder", &status.models.embedder);
    append_model_status_lines(&mut lines, "reranker", &status.models.reranker);
    append_model_status_lines(&mut lines, "expander", &status.models.expander);

    lines.push(String::new());
    lines.push("paths:".to_string());
    lines.push(format!("- cache: {}", status.cache_dir.display()));
    lines.push(format!("- config: {}", status.config_dir.display()));

    lines.join("\n")
}

fn format_file_list(files: &[FileEntry], all: bool) -> String {
    let mut lines = Vec::new();
    lines.push("files:".to_string());
    if files.is_empty() {
        lines.push("- none".to_string());
        return lines.join("\n");
    }

    for file in files {
        if all && !file.active {
            lines.push(format!("- {} (inactive)", file.path));
        } else {
            lines.push(format!("- {}", file.path));
        }
    }

    lines.join("\n")
}

fn append_model_status_lines(lines: &mut Vec<String>, label: &str, info: &ModelInfo) {
    let mut summary = format!("- {label}: {}", model_state_label(info));
    if let Some(model) = info.model.as_deref() {
        summary.push_str(&format!(" ({model})"));
    }
    lines.push(summary);

    if info.configured && !info.ready {
        if let Some(issue) = info.issue.as_deref() {
            lines.push(format!("  issue: {issue}"));
        }
    }
}

fn model_state_label(info: &ModelInfo) -> &'static str {
    if !info.configured {
        "not configured"
    } else if info.ready {
        "ready"
    } else {
        "not ready"
    }
}

fn format_bytes_human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];

    if bytes < 1024 {
        return format!("{bytes} B");
    }

    let mut value = bytes as f64;
    let mut unit_index = 0usize;
    while value >= 1024.0 && unit_index < UNITS.len() - 1 {
        value /= 1024.0;
        unit_index += 1;
    }

    if value >= 10.0 {
        format!("{value:.0} {}", UNITS[unit_index])
    } else {
        format!("{value:.1} {}", UNITS[unit_index])
    }
}

fn active_space_name_for_status(engine: &Engine, explicit: Option<&str>) -> Option<String> {
    if let Some(space_name) = explicit {
        return Some(space_name.to_string());
    }

    if let Ok(space_name) = std::env::var("KBOLT_SPACE") {
        let trimmed = space_name.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    engine.config().default_space.clone()
}

fn format_optional_search_signal(value: Option<f32>) -> String {
    value
        .map(format_search_signal)
        .unwrap_or_else(|| "-".to_string())
}

fn format_search_signal(value: f32) -> String {
    format!("{value:.2}")
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

        if !lines.is_empty() {
            lines.push(String::new());
        }
    }

    lines.push("update complete".to_string());
    append_update_summary_lines(&mut lines, report);

    if !report.errors.is_empty() && !verbose {
        append_update_error_lines(&mut lines, report, 3);
    }

    lines.join("\n")
}

fn format_collection_add_result(result: &AddCollectionResult) -> String {
    let collection = &result.collection;
    let locator = format!("{}/{}", collection.space, collection.name);

    match &result.initial_indexing {
        InitialIndexingOutcome::Skipped => [
            format!("collection added: {locator}"),
            "indexing skipped (--no-index)".to_string(),
            String::new(),
            "next:".to_string(),
            format!(
                "  kbolt --space {} update --collection {}",
                collection.space, collection.name
            ),
        ]
        .join("\n"),
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

    append_update_summary_lines(&mut lines, report);

    if !report.errors.is_empty() {
        append_update_error_lines(&mut lines, report, 3);
    }

    if report.failed_docs > 0 {
        lines.push(String::new());
        lines.push("next:".to_string());
        lines.push(format!(
            "  kbolt --space {} update --collection {}",
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
                "indexing blocked by a dense integrity issue in space '{space}'"
            ));
            lines.push(format!("reason: {reason}"));
            lines.push(String::new());
            lines.push("next:".to_string());
            lines.push(format!("  kbolt --space {space} update"));
        }
        InitialIndexingBlock::ModelNotAvailable { name } => {
            lines.push(format!("indexing blocked: model '{name}' is not available"));
            lines.push(String::new());
            lines.push("next:".to_string());
            lines.push("  kbolt setup local".to_string());
            lines.push("  or configure [roles.embedder] in index.toml".to_string());
            lines.push(format!(
                "  then run: kbolt --space {} update --collection {}",
                collection.space, collection.name
            ));
        }
    }

    lines.join("\n")
}

fn append_update_error_lines(lines: &mut Vec<String>, report: &UpdateReport, limit: usize) {
    lines.push(String::new());
    lines.push("errors:".to_string());
    for error in report.errors.iter().take(limit) {
        lines.push(format!("- {}: {}", error.path, error.error));
    }
    if report.errors.len() > limit {
        lines.push(format!("- {} more error(s)", report.errors.len() - limit));
    }
}

fn append_update_summary_lines(lines: &mut Vec<String>, report: &UpdateReport) {
    lines.push(format!("- {} document(s) scanned", report.scanned_docs));

    let unchanged = report.skipped_mtime_docs + report.skipped_hash_docs;
    if unchanged > 0 {
        lines.push(format!("- {} unchanged", unchanged));
    }
    if report.added_docs > 0 {
        lines.push(format!("- {} added", report.added_docs));
    }
    if report.updated_docs > 0 {
        lines.push(format!("- {} updated", report.updated_docs));
    }
    if report.failed_docs > 0 {
        lines.push(format!("- {} failed", report.failed_docs));
    }
    if report.deactivated_docs > 0 {
        lines.push(format!("- {} deactivated", report.deactivated_docs));
    }
    if report.reactivated_docs > 0 {
        lines.push(format!("- {} reactivated", report.reactivated_docs));
    }
    if report.reaped_docs > 0 {
        lines.push(format!("- {} reaped", report.reaped_docs));
    }
    if report.embedded_chunks > 0 {
        lines.push(format!("- {} chunk(s) embedded", report.embedded_chunks));
    }

    lines.push(format!(
        "- completed in {}",
        format_elapsed_ms(report.elapsed_ms)
    ));
}

fn format_elapsed_ms(elapsed_ms: u64) -> String {
    if elapsed_ms < 1_000 {
        format!("{elapsed_ms}ms")
    } else if elapsed_ms < 60_000 {
        format!("{:.1}s", elapsed_ms as f64 / 1_000.0)
    } else {
        let total_seconds = elapsed_ms as f64 / 1_000.0;
        format!("{:.1}m", total_seconds / 60.0)
    }
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

fn truncate_snippet(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max_lines {
        return text.to_string();
    }
    let truncated: Vec<&str> = lines[..max_lines].to_vec();
    let remaining = lines.len() - max_lines;
    format!(
        "{}\n(+{remaining} more line{})",
        truncated.join("\n"),
        if remaining == 1 { "" } else { "s" }
    )
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
        active_space_name_for_status, format_collection_add_result, format_doctor_report,
        format_elapsed_ms, format_eval_import_report, format_eval_run_report, format_file_list,
        format_local_report, format_models_list, format_optional_search_signal,
        format_schedule_add_response, format_schedule_status_response, format_search_result_path,
        format_status_response, format_update_report, parse_editor_command, resolve_editor_command,
        resolve_no_rerank_for_mode, truncate_snippet, CliAdapter, CliSearchOptions,
    };
    use kbolt_core::engine::Engine;
    use kbolt_types::{
        AddCollectionRequest, AddCollectionResult, CollectionInfo, CollectionStatus, DiskUsage,
        DoctorCheck, DoctorCheckStatus, DoctorReport, DoctorSetupStatus, EvalImportReport,
        EvalJudgment, EvalModeReport, EvalQueryReport, EvalRunReport, FileEntry,
        InitialIndexingBlock, InitialIndexingOutcome, LocalAction, LocalReport, ModelInfo,
        ScheduleAddResponse, ScheduleBackend, ScheduleDefinition, ScheduleInterval,
        ScheduleIntervalUnit, ScheduleOrphan, ScheduleRunResult, ScheduleRunState, ScheduleScope,
        ScheduleState, ScheduleStatusEntry, ScheduleStatusResponse, ScheduleTrigger,
        ScheduleWeekday, SearchMode, SpaceStatus, StatusResponse, UpdateReport,
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
                output.contains("- embedder: not configured"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("- reranker: not configured"),
                "unexpected output: {output}"
            );
            assert!(
                output.contains("- expander: not configured"),
                "unexpected output: {output}"
            );
        });
    }

    #[test]
    fn models_list_surfaces_not_ready_issue_without_provider_dump() {
        let output = format_models_list(&kbolt_types::ModelStatus {
            embedder: ModelInfo {
                configured: true,
                ready: false,
                profile: Some("kbolt_local_embed".to_string()),
                kind: Some("llama_cpp_server".to_string()),
                operation: Some("embedding".to_string()),
                model: Some("embeddinggemma".to_string()),
                endpoint: Some("http://127.0.0.1:8101".to_string()),
                issue: Some("endpoint is unreachable".to_string()),
            },
            reranker: ModelInfo {
                configured: true,
                ready: true,
                profile: Some("kbolt_local_rerank".to_string()),
                kind: Some("llama_cpp_server".to_string()),
                operation: Some("reranking".to_string()),
                model: Some("qwen3-reranker".to_string()),
                endpoint: Some("http://127.0.0.1:8102".to_string()),
                issue: None,
            },
            expander: ModelInfo {
                configured: false,
                ready: false,
                profile: None,
                kind: None,
                operation: None,
                model: None,
                endpoint: None,
                issue: None,
            },
        });

        assert!(output.contains("- embedder: not ready (embeddinggemma)"));
        assert!(output.contains("  issue: endpoint is unreachable"));
        assert!(output.contains("- reranker: ready (qwen3-reranker)"));
        assert!(output.contains("- expander: not configured"));
        assert!(!output.contains("profile="), "unexpected output:\n{output}");
        assert!(
            !output.contains("endpoint=http"),
            "unexpected output:\n{output}"
        );
    }

    #[test]
    fn doctor_report_success_is_concise() {
        let output = format_doctor_report(&DoctorReport {
            setup_status: DoctorSetupStatus::Configured,
            config_file: Some(PathBuf::from("/tmp/kbolt/index.toml")),
            config_dir: Some(PathBuf::from("/tmp/kbolt")),
            cache_dir: Some(PathBuf::from("/tmp/cache/kbolt")),
            ready: true,
            checks: vec![
                DoctorCheck {
                    id: "roles.embedder.bound".to_string(),
                    scope: "roles.embedder".to_string(),
                    status: DoctorCheckStatus::Pass,
                    elapsed_ms: 0,
                    message: "bound".to_string(),
                    fix: None,
                },
                DoctorCheck {
                    id: "roles.expander.bound".to_string(),
                    scope: "roles.expander".to_string(),
                    status: DoctorCheckStatus::Warn,
                    elapsed_ms: 0,
                    message: "role is not configured".to_string(),
                    fix: Some("configure expander".to_string()),
                },
            ],
        });

        assert!(
            output.contains("kbolt is ready"),
            "unexpected output:\n{output}"
        );
        assert!(output.contains("configured:"));
        assert!(output.contains("  embedder"));
        assert!(output.contains("not enabled:"));
        assert!(output.contains("  expander"));
        assert!(
            !output.contains("PASS"),
            "should not show raw check status in success case"
        );
    }

    #[test]
    fn doctor_report_shows_failures_with_fixes() {
        let output = format_doctor_report(&DoctorReport {
            setup_status: DoctorSetupStatus::Configured,
            config_file: Some(PathBuf::from("/tmp/kbolt/index.toml")),
            config_dir: Some(PathBuf::from("/tmp/kbolt")),
            cache_dir: Some(PathBuf::from("/tmp/cache/kbolt")),
            ready: false,
            checks: vec![DoctorCheck {
                id: "roles.embedder.reachable".to_string(),
                scope: "roles.embedder".to_string(),
                status: DoctorCheckStatus::Fail,
                elapsed_ms: 17,
                message: "endpoint is unreachable".to_string(),
                fix: Some("Start the embedding server.".to_string()),
            }],
        });

        assert!(
            output.contains("kbolt has issues"),
            "unexpected output:\n{output}"
        );
        assert!(output.contains("failures:"));
        assert!(output.contains("endpoint is unreachable"));
        assert!(output.contains("fix: Start the embedding server."));
    }

    #[test]
    fn doctor_report_missing_config_guides_to_setup() {
        let output = format_doctor_report(&DoctorReport {
            setup_status: DoctorSetupStatus::ConfigMissing,
            config_file: None,
            config_dir: None,
            cache_dir: None,
            ready: false,
            checks: vec![],
        });

        assert!(
            output.contains("kbolt is not set up"),
            "unexpected output:\n{output}"
        );
        assert!(output.contains("kbolt setup local"));
    }

    #[test]
    fn local_report_shows_ready_services_and_hides_internals() {
        let output = format_local_report(&LocalReport {
            action: LocalAction::Setup,
            config_file: PathBuf::from("/tmp/kbolt/index.toml"),
            cache_dir: PathBuf::from("/tmp/cache/kbolt"),
            llama_server_path: Some(PathBuf::from("/opt/homebrew/bin/llama-server")),
            ready: true,
            notes: vec![],
            services: vec![
                kbolt_types::LocalServiceReport {
                    name: "embedder".to_string(),
                    provider: "kbolt_local_embed".to_string(),
                    enabled: true,
                    configured: true,
                    managed: true,
                    running: true,
                    ready: true,
                    model: "embeddinggemma".to_string(),
                    model_path: PathBuf::from("/tmp/cache/kbolt/models/embedder/model.gguf"),
                    endpoint: "http://127.0.0.1:8101".to_string(),
                    port: 8101,
                    pid: Some(42),
                    pid_file: PathBuf::from("/tmp/cache/kbolt/run/embedder.pid"),
                    log_file: PathBuf::from("/tmp/cache/kbolt/logs/embedder.log"),
                    issue: None,
                },
                kbolt_types::LocalServiceReport {
                    name: "expander".to_string(),
                    provider: "kbolt_local_expand".to_string(),
                    enabled: false,
                    configured: false,
                    managed: false,
                    running: false,
                    ready: false,
                    model: "qmd".to_string(),
                    model_path: PathBuf::from("/tmp/cache/kbolt/models/expander/model.gguf"),
                    endpoint: "http://127.0.0.1:8103".to_string(),
                    port: 8103,
                    pid: None,
                    pid_file: PathBuf::from("/tmp/cache/kbolt/run/expander.pid"),
                    log_file: PathBuf::from("/tmp/cache/kbolt/logs/expander.log"),
                    issue: Some("not configured".to_string()),
                },
            ],
        });

        assert!(
            output.contains("local setup complete"),
            "unexpected output:\n{output}"
        );
        assert!(output.contains("  embedder (embeddinggemma)"));
        assert!(output.contains("not configured:"));
        assert!(output.contains("  expander"));
        assert!(output.contains("/tmp/kbolt/index.toml"));
        assert!(output.contains("kbolt collection add"));
        assert!(
            !output.contains("pid"),
            "should not expose pid in default output"
        );
        assert!(
            !output.contains("log_file"),
            "should not expose log_file in default output"
        );
        assert!(
            !output.contains("model_path"),
            "should not expose model_path in default output"
        );
    }

    #[test]
    fn local_report_surfaces_notes() {
        let output = format_local_report(&LocalReport {
            action: LocalAction::Setup,
            config_file: PathBuf::from("/tmp/kbolt/index.toml"),
            cache_dir: PathBuf::from("/tmp/cache/kbolt"),
            llama_server_path: Some(PathBuf::from("/opt/homebrew/bin/llama-server")),
            ready: true,
            notes: vec![
                "moved incompatible old config to /tmp/index.toml.invalid.bak".to_string(),
                "started embedder on http://127.0.0.1:8101".to_string(),
            ],
            services: vec![],
        });

        assert!(output.contains("notes:"), "unexpected output:\n{output}");
        assert!(
            output.contains("moved incompatible old config"),
            "unexpected output:\n{output}"
        );
        assert!(
            output.contains("started embedder on http://127.0.0.1:8101"),
            "unexpected output:\n{output}"
        );
    }

    #[test]
    fn local_report_shows_issues_when_not_ready() {
        let output = format_local_report(&LocalReport {
            action: LocalAction::Setup,
            config_file: PathBuf::from("/tmp/kbolt/index.toml"),
            cache_dir: PathBuf::from("/tmp/cache/kbolt"),
            llama_server_path: Some(PathBuf::from("/opt/homebrew/bin/llama-server")),
            ready: false,
            notes: vec![],
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

        assert!(
            output.contains("(not ready)"),
            "unexpected output:\n{output}"
        );
        assert!(output.contains("issues:"));
        assert!(output.contains("embedder: service is not ready"));
    }

    #[test]
    fn local_stop_report_treats_stopped_services_as_expected() {
        let output = format_local_report(&LocalReport {
            action: LocalAction::Stop,
            config_file: PathBuf::from("/tmp/kbolt/index.toml"),
            cache_dir: PathBuf::from("/tmp/cache/kbolt"),
            llama_server_path: Some(PathBuf::from("/opt/homebrew/bin/llama-server")),
            ready: false,
            notes: vec![
                "stopped embedder".to_string(),
                "stopped reranker".to_string(),
            ],
            services: vec![
                kbolt_types::LocalServiceReport {
                    name: "embedder".to_string(),
                    provider: "kbolt_local_embed".to_string(),
                    enabled: true,
                    configured: true,
                    managed: false,
                    running: false,
                    ready: false,
                    model: "embeddinggemma".to_string(),
                    model_path: PathBuf::from("/tmp/cache/kbolt/models/embedder/model.gguf"),
                    endpoint: "http://127.0.0.1:8101".to_string(),
                    port: 8101,
                    pid: None,
                    pid_file: PathBuf::from("/tmp/cache/kbolt/run/embedder.pid"),
                    log_file: PathBuf::from("/tmp/cache/kbolt/logs/embedder.log"),
                    issue: Some("service is not ready".to_string()),
                },
                kbolt_types::LocalServiceReport {
                    name: "reranker".to_string(),
                    provider: "kbolt_local_rerank".to_string(),
                    enabled: true,
                    configured: true,
                    managed: false,
                    running: false,
                    ready: false,
                    model: "qwen3-reranker".to_string(),
                    model_path: PathBuf::from("/tmp/cache/kbolt/models/reranker/model.gguf"),
                    endpoint: "http://127.0.0.1:8102".to_string(),
                    port: 8102,
                    pid: None,
                    pid_file: PathBuf::from("/tmp/cache/kbolt/run/reranker.pid"),
                    log_file: PathBuf::from("/tmp/cache/kbolt/logs/reranker.log"),
                    issue: Some("service is not ready".to_string()),
                },
                kbolt_types::LocalServiceReport {
                    name: "expander".to_string(),
                    provider: "kbolt_local_expand".to_string(),
                    enabled: false,
                    configured: false,
                    managed: false,
                    running: false,
                    ready: false,
                    model: "qmd".to_string(),
                    model_path: PathBuf::from("/tmp/cache/kbolt/models/expander/model.gguf"),
                    endpoint: "http://127.0.0.1:8103".to_string(),
                    port: 8103,
                    pid: None,
                    pid_file: PathBuf::from("/tmp/cache/kbolt/run/expander.pid"),
                    log_file: PathBuf::from("/tmp/cache/kbolt/logs/expander.log"),
                    issue: Some("service is not configured".to_string()),
                },
            ],
        });

        assert!(
            output.starts_with("local servers stopped\n"),
            "unexpected output:\n{output}"
        );
        assert!(
            !output.contains("(not ready)"),
            "unexpected output:\n{output}"
        );
        assert!(!output.contains("issues:"), "unexpected output:\n{output}");
        assert!(
            !output.contains("not configured:"),
            "unexpected output:\n{output}"
        );
        assert!(
            output.contains("stopped embedder"),
            "unexpected output:\n{output}"
        );
        assert!(
            output.contains("stopped reranker"),
            "unexpected output:\n{output}"
        );
    }

    #[test]
    fn truncate_snippet_preserves_short_text() {
        assert_eq!(
            truncate_snippet("line one\nline two", 4),
            "line one\nline two"
        );
    }

    #[test]
    fn truncate_snippet_truncates_long_text() {
        let text = "one\ntwo\nthree\nfour\nfive\nsix";
        let result = truncate_snippet(text, 3);
        assert!(result.contains("one\ntwo\nthree\n"), "unexpected: {result}");
        assert!(result.contains("(+3 more lines)"), "unexpected: {result}");
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
                    debug: true,
                })
                .expect("run auto search");

            assert!(
                output.contains("mode: auto -> keyword"),
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
    fn search_result_paths_include_space_context() {
        assert_eq!(
            format_search_result_path("work", "api/guide.md"),
            "work/api/guide.md"
        );
    }

    #[test]
    fn status_response_formats_human_storage_and_model_summary() {
        let output = format_status_response(
            &StatusResponse {
                spaces: vec![SpaceStatus {
                    name: "default".to_string(),
                    description: Some("main workspace".to_string()),
                    last_updated: Some("2026-04-11T16:49:07Z".to_string()),
                    collections: vec![CollectionStatus {
                        name: "kbolt".to_string(),
                        path: PathBuf::from("/Users/macbook/kbolt"),
                        documents: 98,
                        active_documents: 98,
                        chunks: 1218,
                        embedded_chunks: 1218,
                        last_updated: "2026-04-11T16:49:07Z".to_string(),
                    }],
                }],
                models: kbolt_types::ModelStatus {
                    embedder: ModelInfo {
                        configured: true,
                        ready: false,
                        profile: Some("kbolt_local_embed".to_string()),
                        kind: Some("llama_cpp_server".to_string()),
                        operation: Some("embedding".to_string()),
                        model: Some("embeddinggemma".to_string()),
                        endpoint: Some("http://127.0.0.1:8103".to_string()),
                        issue: Some("endpoint is unreachable".to_string()),
                    },
                    reranker: ModelInfo {
                        configured: true,
                        ready: true,
                        profile: Some("kbolt_local_rerank".to_string()),
                        kind: Some("llama_cpp_server".to_string()),
                        operation: Some("reranking".to_string()),
                        model: Some("qwen3-reranker".to_string()),
                        endpoint: Some("http://127.0.0.1:8104".to_string()),
                        issue: None,
                    },
                    expander: ModelInfo {
                        configured: false,
                        ready: false,
                        profile: None,
                        kind: None,
                        operation: None,
                        model: None,
                        endpoint: None,
                        issue: None,
                    },
                },
                cache_dir: PathBuf::from("/Users/macbook/Library/Caches/kbolt"),
                config_dir: PathBuf::from("/Users/macbook/Library/Application Support/kbolt"),
                total_documents: 98,
                total_chunks: 1218,
                total_embedded: 1218,
                disk_usage: DiskUsage {
                    sqlite_bytes: 348_160,
                    tantivy_bytes: 520_111,
                    usearch_bytes: 4_581_056,
                    models_bytes: 1_935_460_432,
                    total_bytes: 1_940_909_759,
                },
            },
            Some("default"),
        );

        assert!(output.contains("spaces:"));
        assert!(output.contains("- default (active)"));
        assert!(output.contains("  description: main workspace"));
        assert!(output.contains("  collections:"));
        assert!(output.contains("    - kbolt"));
        assert!(output.contains("storage:"));
        assert!(
            output.contains("- sqlite: 340 KB"),
            "unexpected output:\n{output}"
        );
        assert!(
            output.contains("- tantivy: 508 KB"),
            "unexpected output:\n{output}"
        );
        assert!(
            output.contains("- vectors: 4.4 MB"),
            "unexpected output:\n{output}"
        );
        assert!(
            output.contains("- models: 1.8 GB"),
            "unexpected output:\n{output}"
        );
        assert!(
            output.contains("- total: 1.8 GB"),
            "unexpected output:\n{output}"
        );
        assert!(output.contains("- embedder: not ready (embeddinggemma)"));
        assert!(output.contains("  issue: endpoint is unreachable"));
        assert!(!output.contains("profile="), "unexpected output:\n{output}");
    }

    #[test]
    fn active_space_name_for_status_follows_cli_precedence_without_validation() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create config root");
            std::fs::write(
                root.path().join("index.toml"),
                "default_space = \"default\"\n",
            )
            .expect("write config");
            let engine = Engine::new(Some(root.path())).expect("create engine");

            std::env::remove_var("KBOLT_SPACE");
            assert_eq!(
                active_space_name_for_status(&engine, Some("work")).as_deref(),
                Some("work")
            );

            std::env::set_var("KBOLT_SPACE", "ops");
            assert_eq!(
                active_space_name_for_status(&engine, None).as_deref(),
                Some("ops")
            );
            std::env::remove_var("KBOLT_SPACE");

            assert_eq!(
                active_space_name_for_status(&engine, None).as_deref(),
                Some("default")
            );
        });
    }

    #[test]
    fn optional_search_signal_uses_human_values() {
        assert_eq!(format_optional_search_signal(None), "-");
        assert_eq!(format_optional_search_signal(Some(0.824)), "0.82");
    }

    #[test]
    fn file_list_hides_docids_by_default() {
        let output = format_file_list(
            &[
                FileEntry {
                    path: "docs/keep.md".to_string(),
                    title: "keep.md".to_string(),
                    docid: "#3c96dd".to_string(),
                    active: true,
                    chunk_count: 1,
                    embedded: false,
                },
                FileEntry {
                    path: "docs/old.md".to_string(),
                    title: "old.md".to_string(),
                    docid: "#deadbe".to_string(),
                    active: false,
                    chunk_count: 1,
                    embedded: false,
                },
            ],
            false,
        );

        assert!(
            output.contains("- docs/keep.md"),
            "unexpected output:\n{output}"
        );
        assert!(!output.contains("#3c96dd"), "unexpected output:\n{output}");
        assert!(
            !output.contains("keep.md |"),
            "unexpected output:\n{output}"
        );
    }

    #[test]
    fn file_list_marks_inactive_files_with_all() {
        let output = format_file_list(
            &[FileEntry {
                path: "docs/old.md".to_string(),
                title: "old.md".to_string(),
                docid: "#deadbe".to_string(),
                active: false,
                chunk_count: 1,
                embedded: false,
            }],
            true,
        );

        assert!(output.contains("- docs/old.md (inactive)"));
        assert!(!output.contains("#deadbe"));
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
                .position(|line| line == "update complete")
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
            assert!(
                output.contains("- 2 document(s) scanned"),
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

        assert!(output.contains("collection added: work/api"));
        assert!(output.contains("indexing skipped (--no-index)"));
        assert!(output.contains("next:"));
        assert!(output.contains("  kbolt --space work update --collection api"));
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
                errors: vec![kbolt_types::FileError {
                    path: "work/api/bad.md".to_string(),
                    error: "extract failed".to_string(),
                }],
                elapsed_ms: 5,
            }),
        });

        assert!(output.contains("collection added: work/api"));
        assert!(output.contains("initial indexing incomplete"));
        assert!(output.contains("- 3 document(s) scanned"));
        assert!(output.contains("- 2 added"));
        assert!(output.contains("- 1 failed"));
        assert!(output.contains("- 2 chunk(s) embedded"));
        assert!(output.contains("- completed in 5ms"));
        assert!(output.contains("errors:"));
        assert!(output.contains("- work/api/bad.md: extract failed"));
        assert!(output.contains("next:"));
        assert!(output.contains("  kbolt --space work update --collection api"));
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
        assert!(output.contains("indexing blocked: model 'embed-model' is not available"));
        assert!(output.contains("next:"));
        assert!(output.contains("  kbolt setup local"));
        assert!(output.contains("  or configure [roles.embedder] in index.toml"));
        assert!(output.contains("  then run: kbolt --space work update --collection api"));
    }

    #[test]
    fn update_report_is_human_readable() {
        let output = format_update_report(
            &UpdateReport {
                scanned_docs: 12,
                skipped_mtime_docs: 5,
                skipped_hash_docs: 1,
                added_docs: 3,
                updated_docs: 2,
                failed_docs: 1,
                deactivated_docs: 0,
                reactivated_docs: 1,
                reaped_docs: 0,
                embedded_chunks: 8,
                decisions: Vec::new(),
                errors: vec![kbolt_types::FileError {
                    path: "work/api/src/bad.rs".to_string(),
                    error: "extract failed".to_string(),
                }],
                elapsed_ms: 1_250,
            },
            false,
        );

        assert!(output.starts_with("update complete"));
        assert!(output.contains("- 12 document(s) scanned"));
        assert!(output.contains("- 6 unchanged"));
        assert!(output.contains("- 3 added"));
        assert!(output.contains("- 2 updated"));
        assert!(output.contains("- 1 failed"));
        assert!(output.contains("- 1 reactivated"));
        assert!(output.contains("- 8 chunk(s) embedded"));
        assert!(output.contains("- completed in 1.2s"));
        assert!(output.contains("errors:"));
        assert!(output.contains("- work/api/src/bad.rs: extract failed"));
        assert!(
            !output.contains("scanned_docs:"),
            "unexpected output:\n{output}"
        );
    }

    #[test]
    fn format_elapsed_ms_uses_human_units() {
        assert_eq!(format_elapsed_ms(8), "8ms");
        assert_eq!(format_elapsed_ms(1_250), "1.2s");
        assert_eq!(format_elapsed_ms(125_000), "2.1m");
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
