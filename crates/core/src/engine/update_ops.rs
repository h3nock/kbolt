use super::*;
use kbolt_types::{UpdateDecision, UpdateDecisionKind};

impl Engine {
    pub fn update(&self, options: UpdateOptions) -> Result<UpdateReport> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        self.update_unlocked(options)
    }

    pub(super) fn update_unlocked(&self, options: UpdateOptions) -> Result<UpdateReport> {
        let started = Instant::now();
        let mut report = UpdateReport {
            scanned: 0,
            skipped_mtime: 0,
            skipped_hash: 0,
            added: 0,
            updated: 0,
            deactivated: 0,
            reactivated: 0,
            reaped: 0,
            embedded: 0,
            decisions: Vec::new(),
            errors: Vec::new(),
            elapsed_ms: 0,
        };

        if !options.dry_run {
            self.replay_fts_dirty_documents(&mut report)?;
        }

        let targets = self.resolve_targets(TargetScope {
            space: options.space.as_deref(),
            collections: &options.collections,
        })?;
        if targets.is_empty() {
            report.elapsed_ms = started.elapsed().as_millis() as u64;
            return Ok(report);
        }

        self.reconcile_dense_integrity(&targets, &options)?;

        let mut fts_dirty_by_space: HashMap<String, HashSet<i64>> = HashMap::new();
        let mut pending_embeddings = Vec::new();
        let mut failed_embedding_chunk_ids = HashSet::new();
        for target in &targets {
            self.update_collection_target(
                target,
                &options,
                &mut report,
                &mut fts_dirty_by_space,
                &mut pending_embeddings,
                &mut failed_embedding_chunk_ids,
            )?;
        }

        if !options.dry_run {
            self.flush_buffered_embeddings(
                &mut pending_embeddings,
                &mut failed_embedding_chunk_ids,
                &mut report,
            )?;

            for (space, doc_ids) in fts_dirty_by_space {
                if doc_ids.is_empty() {
                    continue;
                }

                self.storage.commit_tantivy(&space)?;
                let mut ids = doc_ids.into_iter().collect::<Vec<_>>();
                ids.sort_unstable();
                self.storage.batch_clear_fts_dirty(&ids)?;
            }

            self.embed_pending_chunks(&options, &mut failed_embedding_chunk_ids, &mut report)?;

            let reaped = self
                .storage
                .list_reapable_documents(self.config.reaping.days)?;
            let mut reaped_doc_ids = Vec::with_capacity(reaped.len());
            let mut chunk_ids_by_space: HashMap<String, Vec<i64>> = HashMap::new();
            for document in reaped {
                reaped_doc_ids.push(document.doc_id);
                if !document.chunk_ids.is_empty() {
                    chunk_ids_by_space
                        .entry(document.space_name)
                        .or_default()
                        .extend(document.chunk_ids);
                }
            }
            for (space, chunk_ids) in chunk_ids_by_space {
                self.purge_space_chunks(&space, &chunk_ids)?;
            }
            self.storage.delete_documents(&reaped_doc_ids)?;
            report.reaped = reaped_doc_ids.len();
        }

        report.elapsed_ms = started.elapsed().as_millis() as u64;
        Ok(report)
    }

    fn reconcile_dense_integrity(
        &self,
        targets: &[UpdateTarget],
        options: &UpdateOptions,
    ) -> Result<()> {
        if options.no_embed || options.dry_run {
            return Ok(());
        }

        let expected_model = self.embedding_model_key();
        let mut visited_spaces = HashSet::new();
        for target in targets {
            if !visited_spaces.insert(target.collection.space_id) {
                continue;
            }

            let models = self
                .storage
                .list_embedding_models_in_space(target.collection.space_id)?;
            if models.iter().any(|model| model != expected_model) {
                self.storage
                    .delete_embeddings_for_space(target.collection.space_id)?;
                self.storage.clear_usearch(&target.space)?;
                continue;
            }

            let sqlite_count = self
                .storage
                .count_embedded_chunks(Some(target.collection.space_id))?;
            let usearch_count = self.storage.count_usearch(&target.space)?;

            if sqlite_count == usearch_count {
                continue;
            }

            self.storage
                .delete_embeddings_for_space(target.collection.space_id)?;
            self.storage.clear_usearch(&target.space)?;
        }

        Ok(())
    }

    fn embed_pending_chunks(
        &self,
        options: &UpdateOptions,
        failed_chunk_ids: &mut HashSet<i64>,
        report: &mut UpdateReport,
    ) -> Result<()> {
        if options.no_embed || options.dry_run {
            return Ok(());
        }

        let Some(embedder) = self.embedder.as_ref() else {
            return Ok(());
        };

        let model = self.embedding_model_key();
        let mut after_chunk_id = 0_i64;
        loop {
            let backlog = self
                .storage
                .get_unembedded_chunks(model, after_chunk_id, 64)?;
            if backlog.is_empty() {
                break;
            }

            after_chunk_id = backlog
                .last()
                .map(|record| record.chunk_id)
                .expect("non-empty backlog should have a last chunk id");

            let mut pending = Vec::new();
            let mut bytes_by_path: HashMap<std::path::PathBuf, Option<Vec<u8>>> = HashMap::new();
            for record in backlog {
                if failed_chunk_ids.contains(&record.chunk_id) {
                    continue;
                }

                let full_path = record.collection_path.join(&record.doc_path);
                if !bytes_by_path.contains_key(&full_path) {
                    let bytes = match std::fs::read(&full_path) {
                        Ok(bytes) => Some(bytes),
                        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
                        Err(err) => {
                            report.errors.push(file_error(
                                Some(full_path.clone()),
                                format!("embed read failed: {err}"),
                            ));
                            None
                        }
                    };
                    bytes_by_path.insert(full_path.clone(), bytes);
                }

                let Some(bytes) = bytes_by_path
                    .get(&full_path)
                    .and_then(|bytes| bytes.as_deref())
                else {
                    continue;
                };

                let mut text = chunk_text_from_bytes(&bytes, record.offset, record.length);
                if text.trim().is_empty() {
                    text = " ".to_string();
                }

                pending.push(PendingChunkEmbedding {
                    chunk_id: record.chunk_id,
                    space_name: record.space_name,
                    path: full_path,
                    text,
                });
            }

            if pending.is_empty() {
                continue;
            }

            let result =
                self.embed_pending_batch_with_partial_failures(embedder.as_ref(), pending, report)?;
            failed_chunk_ids.extend(result.failed_chunk_ids);
            if !result.embeddings.is_empty() {
                self.store_chunk_embeddings(model, result.embeddings, report)?;
            }
        }

        Ok(())
    }

    fn flush_buffered_embeddings(
        &self,
        pending_embeddings: &mut Vec<PendingChunkEmbedding>,
        failed_chunk_ids: &mut HashSet<i64>,
        report: &mut UpdateReport,
    ) -> Result<()> {
        if pending_embeddings.is_empty() {
            return Ok(());
        }

        let Some(embedder) = self.embedder.as_ref() else {
            pending_embeddings.clear();
            return Ok(());
        };

        let pending = std::mem::take(pending_embeddings);
        let result =
            self.embed_pending_batch_with_partial_failures(embedder.as_ref(), pending, report)?;
        failed_chunk_ids.extend(result.failed_chunk_ids);
        if result.embeddings.is_empty() {
            return Ok(());
        }

        self.store_chunk_embeddings(self.embedding_model_key(), result.embeddings, report)
    }

    fn embed_pending_batch_with_partial_failures(
        &self,
        embedder: &dyn crate::models::Embedder,
        pending: Vec<PendingChunkEmbedding>,
        report: &mut UpdateReport,
    ) -> Result<EmbeddedPendingBatch> {
        if pending.is_empty() {
            return Ok(EmbeddedPendingBatch::default());
        }

        let texts = pending
            .iter()
            .map(|embedding| embedding.text.clone())
            .collect::<Vec<_>>();
        match embedder.embed_batch(crate::models::EmbeddingInputKind::Document, &texts) {
            Ok(vectors) => {
                if let Some(detail) = invalid_pending_embedding_batch_detail(&pending, &vectors) {
                    return self.split_pending_embedding_batch(embedder, pending, report, detail);
                }

                let embeddings = pending
                    .into_iter()
                    .zip(vectors)
                    .map(|(embedding, vector)| (embedding.chunk_id, embedding.space_name, vector))
                    .collect::<Vec<_>>();
                Ok(EmbeddedPendingBatch {
                    embeddings,
                    failed_chunk_ids: Vec::new(),
                })
            }
            Err(err) => {
                self.split_pending_embedding_batch(embedder, pending, report, err.to_string())
            }
        }
    }

    fn split_pending_embedding_batch(
        &self,
        embedder: &dyn crate::models::Embedder,
        pending: Vec<PendingChunkEmbedding>,
        report: &mut UpdateReport,
        detail: String,
    ) -> Result<EmbeddedPendingBatch> {
        if pending.len() == 1 {
            let embedding = pending
                .into_iter()
                .next()
                .expect("single pending embedding should exist");
            report.errors.push(file_error(
                Some(embedding.path),
                format!("embed failed: {detail}"),
            ));
            return Ok(EmbeddedPendingBatch {
                embeddings: Vec::new(),
                failed_chunk_ids: vec![embedding.chunk_id],
            });
        }

        let mid = pending.len() / 2;
        let mut right = pending;
        let left = right.drain(..mid).collect::<Vec<_>>();

        let mut left_result =
            self.embed_pending_batch_with_partial_failures(embedder, left, report)?;
        let right_result =
            self.embed_pending_batch_with_partial_failures(embedder, right, report)?;
        left_result.embeddings.extend(right_result.embeddings);
        left_result
            .failed_chunk_ids
            .extend(right_result.failed_chunk_ids);
        Ok(left_result)
    }

    fn store_chunk_embeddings(
        &self,
        model: &str,
        embeddings: Vec<(i64, String, Vec<f32>)>,
        report: &mut UpdateReport,
    ) -> Result<()> {
        let mut grouped_vectors: HashMap<String, Vec<(i64, Vec<f32>)>> = HashMap::new();
        let mut embedding_rows = Vec::with_capacity(embeddings.len());
        for (chunk_id, space_name, vector) in embeddings {
            if vector.is_empty() {
                return Err(KboltError::Inference(format!(
                    "embedder returned an empty vector for chunk {chunk_id}"
                ))
                .into());
            }

            grouped_vectors
                .entry(space_name)
                .or_default()
                .push((chunk_id, vector));
            embedding_rows.push((chunk_id, model));
        }

        for (space, vectors) in grouped_vectors {
            let refs = vectors
                .iter()
                .map(|(chunk_id, vector)| (*chunk_id, vector.as_slice()))
                .collect::<Vec<_>>();
            self.storage.batch_insert_usearch(&space, &refs)?;
        }

        self.storage.insert_embeddings(&embedding_rows)?;
        report.embedded = report.embedded.saturating_add(embedding_rows.len());
        Ok(())
    }

    fn replay_fts_dirty_documents(&self, report: &mut UpdateReport) -> Result<()> {
        let records = self.storage.get_fts_dirty_documents()?;
        if records.is_empty() {
            return Ok(());
        }

        let mut cleared_by_space: HashMap<String, Vec<i64>> = HashMap::new();
        for record in records {
            let space_name = record.space_name;
            let doc_id = record.doc_id;

            if record.chunks.is_empty() {
                cleared_by_space.entry(space_name).or_default().push(doc_id);
                continue;
            }

            let full_path = record.collection_path.join(&record.doc_path);
            let bytes = match std::fs::read(&full_path) {
                Ok(bytes) => bytes,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                Err(err) => {
                    report.errors.push(file_error(
                        Some(full_path),
                        format!("fts replay read failed: {err}"),
                    ));
                    continue;
                }
            };
            if sha256_hex(&bytes) != record.doc_hash {
                continue;
            }

            self.storage.delete_tantivy_by_doc(&space_name, doc_id)?;

            let file_body = String::from_utf8_lossy(&bytes).into_owned();
            let entries = record
                .chunks
                .iter()
                .map(|chunk| {
                    let chunk_body = chunk_text_from_bytes(&bytes, chunk.offset, chunk.length);
                    let source_body = if chunk_body.is_empty() {
                        file_body.as_str()
                    } else {
                        chunk_body.as_str()
                    };
                    TantivyEntry {
                        chunk_id: chunk.id,
                        doc_id,
                        filepath: record.doc_path.clone(),
                        semantic_title: record
                            .doc_title_source
                            .semantic_title(record.doc_title.as_str())
                            .map(ToString::to_string),
                        heading: chunk.heading.clone(),
                        body: retrieval_text_with_prefix(
                            source_body,
                            record
                                .doc_title_source
                                .semantic_title(record.doc_title.as_str()),
                            chunk.heading.as_deref(),
                            self.config.chunking.defaults.contextual_prefix,
                        ),
                    }
                })
                .collect::<Vec<_>>();

            self.storage.index_tantivy(&space_name, &entries)?;
            cleared_by_space.entry(space_name).or_default().push(doc_id);
        }

        for (space_name, mut doc_ids) in cleared_by_space {
            if doc_ids.is_empty() {
                continue;
            }

            doc_ids.sort_unstable();
            doc_ids.dedup();
            self.storage.commit_tantivy(&space_name)?;
            self.storage.batch_clear_fts_dirty(&doc_ids)?;
        }

        Ok(())
    }

    pub fn resolve_update_targets(&self, options: &UpdateOptions) -> Result<Vec<UpdateTarget>> {
        self.resolve_targets(TargetScope {
            space: options.space.as_deref(),
            collections: &options.collections,
        })
    }

    pub(super) fn resolve_targets(&self, scope: TargetScope<'_>) -> Result<Vec<UpdateTarget>> {
        let mut targets = Vec::new();

        if scope.collections.is_empty() {
            return self.resolve_update_targets_for_all_collections(scope.space);
        }

        let mut seen = std::collections::HashSet::new();
        for raw_collection_name in scope.collections {
            let collection_name = raw_collection_name.trim();
            if collection_name.is_empty() {
                return Err(KboltError::InvalidInput(
                    "collection names cannot be empty".to_string(),
                )
                .into());
            }

            let resolved_space = self.resolve_space_row(scope.space, Some(collection_name))?;
            let collection = self
                .storage
                .get_collection(resolved_space.id, collection_name)?;

            if seen.insert((collection.space_id, collection.name.clone())) {
                targets.push(UpdateTarget {
                    space: resolved_space.name,
                    collection,
                });
            }
        }

        Ok(targets)
    }

    fn resolve_update_targets_for_all_collections(
        &self,
        space: Option<&str>,
    ) -> Result<Vec<UpdateTarget>> {
        let (space_id_filter, spaces_by_id) = if let Some(space_name) = space {
            let resolved = self.resolve_space_row(Some(space_name), None)?;
            let mut map = std::collections::HashMap::new();
            map.insert(resolved.id, resolved.name.clone());
            (Some(resolved.id), map)
        } else {
            let spaces = self.storage.list_spaces()?;
            let map = spaces
                .into_iter()
                .map(|space| (space.id, space.name))
                .collect::<std::collections::HashMap<_, _>>();
            (None, map)
        };

        let collections = self.storage.list_collections(space_id_filter)?;
        let mut targets = Vec::with_capacity(collections.len());
        for collection in collections {
            let space_name = spaces_by_id
                .get(&collection.space_id)
                .ok_or_else(|| {
                    KboltError::Internal(format!(
                        "missing space mapping for collection '{}'",
                        collection.name
                    ))
                })?
                .clone();
            targets.push(UpdateTarget {
                space: space_name,
                collection,
            });
        }

        Ok(targets)
    }

    fn update_collection_target(
        &self,
        target: &UpdateTarget,
        options: &UpdateOptions,
        report: &mut UpdateReport,
        fts_dirty_by_space: &mut HashMap<String, HashSet<i64>>,
        pending_embeddings: &mut Vec<PendingChunkEmbedding>,
        failed_chunk_ids: &mut HashSet<i64>,
    ) -> Result<()> {
        let all_documents = self.storage.list_documents(target.collection.id, false)?;
        let mut docs_by_path: HashMap<String, DocumentRow> = all_documents
            .into_iter()
            .map(|doc| (doc.path.clone(), doc))
            .collect();
        let mut seen_paths = HashSet::new();
        let extension_filter = normalized_extension_filter(target.collection.extensions.as_deref());
        let ignore_matcher = load_collection_ignore_matcher(
            &self.config.config_dir,
            &target.collection.path,
            &target.space,
            &target.collection.name,
        )?;
        let extractor_registry = default_registry();
        let mut touched_collection = false;

        for entry in WalkDir::new(&target.collection.path)
            .follow_links(false)
            .into_iter()
            .filter_entry(|entry| {
                !entry.file_type().is_dir() || !is_hard_ignored_dir_name(entry.file_name())
            })
        {
            let entry = match entry {
                Ok(item) => item,
                Err(err) => {
                    report.errors.push(file_error(
                        err.path().map(Path::to_path_buf),
                        format!("walkdir error: {err}"),
                    ));
                    continue;
                }
            };

            if !entry.file_type().is_file() {
                continue;
            }

            if is_hard_ignored_file(entry.path()) {
                continue;
            }

            let relative_path =
                match collection_relative_path(&target.collection.path, entry.path()) {
                    Ok(path) => path,
                    Err(err) => {
                        report.errors.push(file_error(
                            Some(entry.path().to_path_buf()),
                            err.to_string(),
                        ));
                        continue;
                    }
                };

            if !extension_allowed(entry.path(), extension_filter.as_ref()) {
                push_update_decision(
                    report,
                    options,
                    target,
                    &relative_path,
                    UpdateDecisionKind::Unsupported,
                    Some("extension not allowed".to_string()),
                );
                continue;
            }

            if let Some(matcher) = ignore_matcher.as_ref() {
                if matcher
                    .matched(Path::new(&relative_path), false)
                    .is_ignore()
                {
                    push_update_decision(
                        report,
                        options,
                        target,
                        &relative_path,
                        UpdateDecisionKind::Ignored,
                        Some("matched ignore patterns".to_string()),
                    );
                    continue;
                }
            }

            let Some(extractor) = extractor_registry.resolve_for_path(entry.path()) else {
                push_update_decision(
                    report,
                    options,
                    target,
                    &relative_path,
                    UpdateDecisionKind::Unsupported,
                    Some("no extractor available".to_string()),
                );
                continue;
            };

            report.scanned += 1;
            seen_paths.insert(relative_path.clone());

            let metadata = match entry.metadata() {
                Ok(data) => data,
                Err(err) => {
                    let detail = format!("metadata error: {err}");
                    push_update_decision(
                        report,
                        options,
                        target,
                        &relative_path,
                        UpdateDecisionKind::ReadFailed,
                        Some(detail.clone()),
                    );
                    report
                        .errors
                        .push(file_error(Some(entry.path().to_path_buf()), detail));
                    continue;
                }
            };

            let modified = match modified_token(&metadata) {
                Ok(value) => value,
                Err(err) => {
                    let detail = format!("modified timestamp error: {err}");
                    push_update_decision(
                        report,
                        options,
                        target,
                        &relative_path,
                        UpdateDecisionKind::ReadFailed,
                        Some(detail.clone()),
                    );
                    report
                        .errors
                        .push(file_error(Some(entry.path().to_path_buf()), detail));
                    continue;
                }
            };

            if let Some(existing) = docs_by_path.get(&relative_path) {
                if existing.active && existing.modified == modified {
                    report.skipped_mtime += 1;
                    push_update_decision(
                        report,
                        options,
                        target,
                        &relative_path,
                        UpdateDecisionKind::SkippedMtime,
                        None,
                    );
                    continue;
                }
            }

            let bytes = match std::fs::read(entry.path()) {
                Ok(data) => data,
                Err(err) => {
                    let detail = err.to_string();
                    push_update_decision(
                        report,
                        options,
                        target,
                        &relative_path,
                        UpdateDecisionKind::ReadFailed,
                        Some(detail.clone()),
                    );
                    report
                        .errors
                        .push(file_error(Some(entry.path().to_path_buf()), detail));
                    continue;
                }
            };
            let hash = sha256_hex(&bytes);
            let mut title = file_title(entry.path());
            let mut title_source = DocumentTitleSource::FilenameFallback;

            let existing = docs_by_path.get(&relative_path).cloned();
            let pending_decision;
            if let Some(doc) = existing.as_ref() {
                if doc.hash == hash {
                    if doc.active {
                        report.skipped_hash += 1;
                        push_update_decision(
                            report,
                            options,
                            target,
                            &relative_path,
                            UpdateDecisionKind::SkippedHash,
                            None,
                        );
                    } else {
                        report.reactivated += 1;
                        push_update_decision(
                            report,
                            options,
                            target,
                            &relative_path,
                            UpdateDecisionKind::Reactivated,
                            None,
                        );
                    }

                    if !options.dry_run {
                        self.storage.refresh_document_activity(doc.id, &modified)?;
                    }
                    continue;
                }

                report.updated += 1;
                if !doc.active {
                    report.reactivated += 1;
                }
                pending_decision = (
                    UpdateDecisionKind::Changed,
                    (!doc.active).then_some("reactivated".to_string()),
                );
            } else {
                report.added += 1;
                pending_decision = (UpdateDecisionKind::New, None);
            }

            if options.dry_run {
                let (kind, detail) = pending_decision;
                push_update_decision(report, options, target, &relative_path, kind, detail);
                continue;
            }

            let extracted = match extractor.extract(entry.path(), &bytes) {
                Ok(document) => document,
                Err(err) => {
                    let detail = format!("extract failed: {err}");
                    push_update_decision(
                        report,
                        options,
                        target,
                        &relative_path,
                        UpdateDecisionKind::ExtractFailed,
                        Some(detail.clone()),
                    );
                    report
                        .errors
                        .push(file_error(Some(entry.path().to_path_buf()), detail));
                    continue;
                }
            };
            if let Some(extracted_title) = extracted
                .title
                .as_deref()
                .map(str::trim)
                .filter(|title| !title.is_empty())
            {
                title = extracted_title.to_string();
                title_source = DocumentTitleSource::Extracted;
            }

            let doc_id = self.storage.upsert_document(
                target.collection.id,
                &relative_path,
                &title,
                title_source,
                &hash,
                &modified,
            )?;

            if let Some(doc) = existing.as_ref() {
                let old_chunk_ids = self.storage.delete_chunks_for_document(doc.id)?;
                if !old_chunk_ids.is_empty() {
                    self.storage.delete_tantivy(&target.space, &old_chunk_ids)?;
                    self.storage.delete_usearch(&target.space, &old_chunk_ids)?;
                }
            }

            let policy = resolve_policy(&self.config.chunking, Some(extractor.profile_key()), None);
            let final_chunks = chunk_document(&extracted, &policy);

            let chunk_inserts = final_chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| ChunkInsert {
                    seq: index as i32,
                    offset: chunk.offset,
                    length: chunk.length,
                    heading: chunk.heading.clone(),
                    kind: chunk.kind,
                })
                .collect::<Vec<_>>();
            let body = String::from_utf8_lossy(&bytes).into_owned();
            let chunk_ids = self.storage.insert_chunks(doc_id, &chunk_inserts)?;

            if !chunk_ids.is_empty() {
                if !options.no_embed && self.embedder.is_some() {
                    for (chunk_id, chunk) in chunk_ids.iter().zip(final_chunks.iter()) {
                        let mut text = chunk_text_from_bytes(&bytes, chunk.offset, chunk.length);
                        if text.trim().is_empty() {
                            text = " ".to_string();
                        }
                        pending_embeddings.push(PendingChunkEmbedding {
                            chunk_id: *chunk_id,
                            space_name: target.space.clone(),
                            path: entry.path().to_path_buf(),
                            text,
                        });
                    }
                    if pending_embeddings.len() >= 64 {
                        self.flush_buffered_embeddings(
                            pending_embeddings,
                            failed_chunk_ids,
                            report,
                        )?;
                    }
                }

                let entries = chunk_ids
                    .iter()
                    .zip(final_chunks.iter())
                    .map(|(chunk_id, chunk)| TantivyEntry {
                        chunk_id: *chunk_id,
                        doc_id,
                        filepath: relative_path.clone(),
                        semantic_title: title_source
                            .semantic_title(title.as_str())
                            .map(ToString::to_string),
                        heading: chunk.heading.clone(),
                        body: retrieval_text_with_prefix(
                            if chunk.text.is_empty() {
                                body.as_str()
                            } else {
                                chunk.text.as_str()
                            },
                            title_source.semantic_title(title.as_str()),
                            chunk.heading.as_deref(),
                            policy.contextual_prefix,
                        ),
                    })
                    .collect::<Vec<_>>();
                self.storage.index_tantivy(&target.space, &entries)?;
                fts_dirty_by_space
                    .entry(target.space.clone())
                    .or_default()
                    .insert(doc_id);
            }

            docs_by_path.insert(
                relative_path.clone(),
                self.storage
                    .get_document_by_path(target.collection.id, &relative_path)?
                    .ok_or_else(|| {
                        KboltError::Internal(format!(
                            "document missing after upsert: collection={}, path={relative_path}",
                            target.collection.id
                        ))
                    })?,
            );
            let (kind, detail) = pending_decision;
            push_update_decision(report, options, target, &relative_path, kind, detail);
            touched_collection = true;
        }

        let mut missing_docs = docs_by_path
            .values()
            .filter(|doc| doc.active && !seen_paths.contains(&doc.path))
            .cloned()
            .collect::<Vec<_>>();
        missing_docs.sort_by(|left, right| left.path.cmp(&right.path));

        for doc in missing_docs {
            if doc.active && !seen_paths.contains(&doc.path) {
                report.deactivated += 1;
                push_update_decision(
                    report,
                    options,
                    target,
                    &doc.path,
                    UpdateDecisionKind::Deactivated,
                    None,
                );
                if !options.dry_run {
                    self.storage.deactivate_document(doc.id)?;
                    touched_collection = true;
                }
            }
        }

        if touched_collection && !options.dry_run {
            self.storage
                .update_collection_timestamp(target.collection.id)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
struct PendingChunkEmbedding {
    chunk_id: i64,
    space_name: String,
    path: std::path::PathBuf,
    text: String,
}

#[derive(Default)]
struct EmbeddedPendingBatch {
    embeddings: Vec<(i64, String, Vec<f32>)>,
    failed_chunk_ids: Vec<i64>,
}

fn invalid_pending_embedding_batch_detail(
    pending: &[PendingChunkEmbedding],
    vectors: &[Vec<f32>],
) -> Option<String> {
    if vectors.len() != pending.len() {
        return Some(format!(
            "embedder returned {} vectors for {} chunks",
            vectors.len(),
            pending.len()
        ));
    }

    if vectors.iter().any(|vector| vector.is_empty()) {
        if pending.len() == 1 {
            return Some(format!(
                "embedder returned an empty vector for chunk {}",
                pending[0].chunk_id
            ));
        }
        return Some("embedder returned an empty vector".to_string());
    }

    None
}

fn push_update_decision(
    report: &mut UpdateReport,
    options: &UpdateOptions,
    target: &UpdateTarget,
    path: &str,
    kind: UpdateDecisionKind,
    detail: Option<String>,
) {
    if !options.verbose {
        return;
    }

    report.decisions.push(UpdateDecision {
        space: target.space.clone(),
        collection: target.collection.name.clone(),
        path: path.to_string(),
        kind,
        detail,
    });
}
