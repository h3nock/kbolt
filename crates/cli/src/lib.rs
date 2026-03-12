pub mod args;

use kbolt_core::engine::Engine;
use kbolt_core::ModelPullEvent;
use kbolt_core::Result;
use kbolt_types::{
    ActiveSpaceSource, AddCollectionRequest, AddScheduleRequest, EvalRunReport, GetRequest,
    KboltError, Locator, MultiGetRequest, OmitReason, RemoveScheduleRequest, ScheduleAddResponse,
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
            let availability = format_model_availability(info.downloaded, info.path.is_some());
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
            format_model_availability(
                status.models.embedder.downloaded,
                status.models.embedder.path.is_some(),
            )
        ));
        lines.push(format!(
            "model_reranker: {} ({})",
            status.models.reranker.name,
            format_model_availability(
                status.models.reranker.downloaded,
                status.models.reranker.path.is_some(),
            )
        ));
        lines.push(format!(
            "model_expander: {} ({})",
            status.models.expander.name,
            format_model_availability(
                status.models.expander.downloaded,
                status.models.expander.path.is_some(),
            )
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

fn format_model_availability(downloaded: bool, has_local_path: bool) -> &'static str {
    if downloaded {
        "downloaded"
    } else if has_local_path {
        "missing"
    } else {
        "not_applicable"
    }
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
        SearchPipelineUnavailableReason::ModelNotAvailable => "required model not downloaded",
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
            format_eval_mode_label(&mode.mode, mode.no_rerank),
            mode.recall_at_5,
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
                let perfect_recall = query.matched_paths.len() == query.expected_paths.len();
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
                    no_rerank: false,
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
            failed_modes: vec![kbolt_types::EvalModeFailure {
                mode: SearchMode::Semantic,
                no_rerank: true,
                error: "model not available".to_string(),
            }],
        });

        assert!(output.contains("- keyword: recall@5 1.000, mrr@10 1.000, p50 2ms, p95 3ms"));
        assert!(output.contains("- deep: recall@5 0.000, mrr@10 0.000, p50 8ms, p95 12ms"));
        assert!(output.contains("- semantic: failed (model not available)"));
        assert!(output.contains("queries needing attention:"));
        assert!(output.contains("[deep] trait object generic | first relevant: none"));
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
                output.contains(&format!("- embedder: {embed_model} (not_applicable)")),
                "unexpected output: {output}"
            );
            assert!(
                output.contains(&format!("- reranker: {reranker_model} (not_applicable)")),
                "unexpected output: {output}"
            );
            assert!(
                output.contains(&format!("- expander: {expander_model} (not_applicable)")),
                "unexpected output: {output}"
            );
        });
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
                .position(|line| line.starts_with("scanned: "))
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
