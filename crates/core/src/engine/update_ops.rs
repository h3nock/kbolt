use super::*;
use kbolt_types::{UpdateDecision, UpdateDecisionKind};

const CANONICAL_TEXT_GENERATION: u32 = 1;
const CHUNKER_GENERATION: u32 = 1;

impl Engine {
    pub fn update(&self, options: UpdateOptions) -> Result<UpdateReport> {
        let _lock = self.acquire_operation_lock(LockMode::Exclusive)?;
        self.update_unlocked(options)
    }

    pub(super) fn update_unlocked(&self, options: UpdateOptions) -> Result<UpdateReport> {
        let _profile = crate::profile::UpdateProfileGuard::start();
        let started = Instant::now();
        let mut report = UpdateReport {
            scanned_docs: 0,
            skipped_mtime_docs: 0,
            skipped_hash_docs: 0,
            added_docs: 0,
            updated_docs: 0,
            failed_docs: 0,
            deactivated_docs: 0,
            reactivated_docs: 0,
            reaped_docs: 0,
            embedded_chunks: 0,
            decisions: Vec::new(),
            errors: Vec::new(),
            elapsed_ms: 0,
        };
        let mut failed_docs = HashSet::new();

        let targets = crate::profile::timed_update_stage("resolve_targets", || {
            self.resolve_targets(TargetScope {
                space: options.space.as_deref(),
                collections: &options.collections,
            })
        })?;
        if targets.is_empty() {
            report.elapsed_ms = started.elapsed().as_millis() as u64;
            return Ok(report);
        }
        let repair_scope = UpdateRepairScope::from_options_and_targets(&options, &targets);

        crate::profile::timed_update_stage("dense_integrity", || {
            self.reconcile_dense_integrity(&targets, &repair_scope, &options)
        })?;

        if !options.dry_run {
            crate::profile::timed_update_stage("fts_dirty_replay", || {
                self.replay_fts_dirty_documents(&repair_scope, &mut report, &mut failed_docs)
            })?;
        }

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
                &mut failed_docs,
            )?;
        }

        if !options.dry_run {
            crate::profile::timed_update_stage("embedding_flush_buffered", || {
                self.flush_buffered_embeddings(
                    &mut pending_embeddings,
                    &mut failed_embedding_chunk_ids,
                    &mut report,
                    &mut failed_docs,
                )
            })?;

            for (space, doc_ids) in fts_dirty_by_space {
                if doc_ids.is_empty() {
                    continue;
                }

                crate::profile::timed_update_stage("tantivy_commit", || {
                    self.storage.commit_tantivy(&space)
                })?;
                let mut ids = doc_ids.into_iter().collect::<Vec<_>>();
                ids.sort_unstable();
                crate::profile::timed_update_stage("sqlite_clear_fts_dirty", || {
                    self.storage.batch_clear_fts_dirty(&ids)
                })?;
            }

            crate::profile::timed_update_stage("embedding_backlog", || {
                self.embed_pending_chunks(
                    &repair_scope,
                    &options,
                    &mut failed_embedding_chunk_ids,
                    &mut report,
                    &mut failed_docs,
                )
            })?;

            let reaped = crate::profile::timed_update_stage("sqlite_list_reapable", || {
                self.list_reapable_documents_for_scope(self.config.reaping.days, &repair_scope)
            })?;
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
                crate::profile::timed_update_stage("purge_reaped_indexes", || {
                    self.purge_space_chunks(&space, &chunk_ids)
                })?;
            }
            crate::profile::timed_update_stage("sqlite_delete_reaped_documents", || {
                self.storage.delete_documents(&reaped_doc_ids)
            })?;
            report.reaped_docs = reaped_doc_ids.len();
        }

        report.failed_docs = failed_docs.len();
        report.elapsed_ms = started.elapsed().as_millis() as u64;
        Ok(report)
    }

    fn reconcile_dense_integrity(
        &self,
        targets: &[UpdateTarget],
        repair_scope: &UpdateRepairScope,
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
            let mut reasons = Vec::new();
            if models.iter().any(|model| model != expected_model) {
                reasons.push(format!(
                    "stored embedding models {:?} do not match current model '{}'",
                    models, expected_model
                ));
            }

            let sqlite_count = self
                .storage
                .count_embedded_chunks(Some(target.collection.space_id))?;
            let usearch_count = self.storage.count_usearch(&target.space)?;
            if sqlite_count != usearch_count {
                reasons.push(format!(
                    "sqlite embedded chunk count {sqlite_count} does not match USearch vector count {usearch_count}"
                ));
            }

            if reasons.is_empty() {
                continue;
            }

            if !repair_scope.allows_space_dense_repair() {
                return Err(KboltError::SpaceDenseRepairRequired {
                    space: target.space.clone(),
                    reason: reasons.join("; "),
                }
                .into());
            }

            self.storage
                .delete_embeddings_for_space(target.collection.space_id)?;
            self.storage.clear_usearch(&target.space)?;
        }

        Ok(())
    }

    fn embed_pending_chunks(
        &self,
        repair_scope: &UpdateRepairScope,
        options: &UpdateOptions,
        failed_chunk_ids: &mut HashSet<i64>,
        report: &mut UpdateReport,
        failed_docs: &mut HashSet<UpdateDocKey>,
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
            let backlog =
                crate::profile::timed_update_stage("sqlite_get_unembedded_chunks", || {
                    self.get_unembedded_chunks_for_scope(model, repair_scope, after_chunk_id, 64)
                })?;
            if backlog.is_empty() {
                break;
            }

            after_chunk_id = backlog
                .last()
                .map(|record| record.chunk_id)
                .expect("non-empty backlog should have a last chunk id");

            let mut pending = Vec::new();
            for record in backlog {
                if failed_chunk_ids.contains(&record.chunk_id) {
                    continue;
                }

                let full_path = record.collection_path.join(&record.doc_path);
                let read_started = Instant::now();
                let chunk_text = match self.storage.get_chunk_text(record.chunk_id) {
                    Ok(chunk_text) => chunk_text,
                    Err(err) => {
                        report.errors.push(file_error(
                            Some(full_path.clone()),
                            format!("embed canonical text load failed: {err}"),
                        ));
                        failed_docs.insert(update_doc_key(
                            &record.space_name,
                            &record.collection_path,
                            &record.doc_path,
                        ));
                        continue;
                    }
                };
                crate::profile::record_update_stage(
                    "embedding_backlog_read",
                    read_started.elapsed(),
                );
                let mut text = chunk_text.text;
                if text.trim().is_empty() {
                    text = " ".to_string();
                }
                let policy = resolve_policy(
                    &self.config.chunking,
                    Some(chunk_text.extractor_key.as_str()),
                    None,
                );
                let max_document_tokens = effective_chunk_hard_max(&policy);

                pending.push(PendingChunkEmbedding {
                    chunk_id: record.chunk_id,
                    doc_key: update_doc_key(
                        &record.space_name,
                        &record.collection_path,
                        &record.doc_path,
                    ),
                    space_name: record.space_name,
                    path: full_path,
                    text,
                    max_document_tokens,
                });
            }

            if pending.is_empty() {
                continue;
            }

            let mut preflight_failed_chunk_ids = Vec::new();
            let pending = self.preflight_pending_embeddings(
                pending,
                report,
                failed_docs,
                &mut preflight_failed_chunk_ids,
            )?;
            failed_chunk_ids.extend(preflight_failed_chunk_ids);
            if pending.is_empty() {
                continue;
            }

            let result = self.embed_preflighted_pending_batch_with_partial_failures(
                embedder.as_ref(),
                pending,
                report,
                failed_docs,
            )?;
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
        failed_docs: &mut HashSet<UpdateDocKey>,
    ) -> Result<()> {
        if pending_embeddings.is_empty() {
            return Ok(());
        }

        let Some(embedder) = self.embedder.as_ref() else {
            pending_embeddings.clear();
            return Ok(());
        };

        let pending = std::mem::take(pending_embeddings);
        let result = self.embed_preflighted_pending_batch_with_partial_failures(
            embedder.as_ref(),
            pending,
            report,
            failed_docs,
        )?;
        failed_chunk_ids.extend(result.failed_chunk_ids);
        if result.embeddings.is_empty() {
            return Ok(());
        }

        self.store_chunk_embeddings(self.embedding_model_key(), result.embeddings, report)
    }

    fn embed_preflighted_pending_batch_with_partial_failures(
        &self,
        embedder: &dyn crate::models::Embedder,
        pending: Vec<PendingChunkEmbedding>,
        report: &mut UpdateReport,
        failed_docs: &mut HashSet<UpdateDocKey>,
    ) -> Result<EmbeddedPendingBatch> {
        if pending.is_empty() {
            return Ok(EmbeddedPendingBatch::default());
        }

        let texts = pending
            .iter()
            .map(|embedding| embedding.text.clone())
            .collect::<Vec<_>>();
        crate::profile::increment_update_count("embedding_batch_calls", 1);
        crate::profile::increment_update_count("embedding_input_texts", texts.len() as u64);
        match crate::profile::timed_update_stage("embedding_http", || {
            embedder.embed_batch(crate::models::EmbeddingInputKind::Document, &texts)
        }) {
            Ok(vectors) => {
                if let Some(detail) = invalid_pending_embedding_batch_detail(&pending, &vectors) {
                    return self.split_pending_embedding_batch(
                        embedder,
                        pending,
                        report,
                        failed_docs,
                        detail,
                    );
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
            Err(err) => self.split_pending_embedding_batch(
                embedder,
                pending,
                report,
                failed_docs,
                err.to_string(),
            ),
        }
    }

    fn split_pending_embedding_batch(
        &self,
        embedder: &dyn crate::models::Embedder,
        pending: Vec<PendingChunkEmbedding>,
        report: &mut UpdateReport,
        failed_docs: &mut HashSet<UpdateDocKey>,
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
            failed_docs.insert(embedding.doc_key);
            return Ok(EmbeddedPendingBatch {
                embeddings: Vec::new(),
                failed_chunk_ids: vec![embedding.chunk_id],
            });
        }

        let mid = pending.len() / 2;
        let mut right = pending;
        let left = right.drain(..mid).collect::<Vec<_>>();

        let mut left_result = self.embed_preflighted_pending_batch_with_partial_failures(
            embedder,
            left,
            report,
            failed_docs,
        )?;
        let right_result = self.embed_preflighted_pending_batch_with_partial_failures(
            embedder,
            right,
            report,
            failed_docs,
        )?;
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
            crate::profile::increment_update_count("usearch_vectors", refs.len() as u64);
            self.storage.batch_insert_usearch(&space, &refs)?;
        }

        crate::profile::timed_update_stage("sqlite_insert_embeddings", || {
            self.storage.insert_embeddings(&embedding_rows)
        })?;
        report.embedded_chunks = report.embedded_chunks.saturating_add(embedding_rows.len());
        Ok(())
    }

    fn preflight_pending_embeddings(
        &self,
        pending: Vec<PendingChunkEmbedding>,
        report: &mut UpdateReport,
        failed_docs: &mut HashSet<UpdateDocKey>,
        failed_chunk_ids: &mut Vec<i64>,
    ) -> Result<Vec<PendingChunkEmbedding>> {
        let Some(sizer) = self.embedding_document_sizer.as_ref() else {
            return Ok(pending);
        };

        let mut accepted = Vec::with_capacity(pending.len());
        for embedding in pending {
            match count_document_tokens_profiled(
                sizer.as_ref(),
                &embedding.text,
                "tokenize_preflight",
                "tokenize_preflight_calls",
            ) {
                Ok(token_count) if token_count <= embedding.max_document_tokens => {
                    accepted.push(embedding);
                }
                Ok(token_count) => {
                    report.errors.push(file_error(
                        Some(embedding.path.clone()),
                        format!(
                            "embed preflight failed: payload has {token_count} tokens, exceeding hard_max_tokens {}",
                            embedding.max_document_tokens
                        ),
                    ));
                    failed_docs.insert(embedding.doc_key);
                    failed_chunk_ids.push(embedding.chunk_id);
                }
                Err(err) => {
                    report.errors.push(file_error(
                        Some(embedding.path.clone()),
                        format!("embed preflight token count failed: {err}"),
                    ));
                    failed_docs.insert(embedding.doc_key);
                    failed_chunk_ids.push(embedding.chunk_id);
                }
            }
        }

        Ok(accepted)
    }

    fn preflight_prepared_embeddings(
        &self,
        prepared: Vec<PreparedChunkEmbedding>,
        report: &mut UpdateReport,
        failed_docs: &mut HashSet<UpdateDocKey>,
    ) -> Result<PreparedEmbeddingPreflight> {
        let Some(sizer) = self.embedding_document_sizer.as_ref() else {
            return Ok(PreparedEmbeddingPreflight {
                accepted: prepared,
                rejected_chunk_indexes: Vec::new(),
            });
        };

        let mut preflight = PreparedEmbeddingPreflight {
            accepted: Vec::with_capacity(prepared.len()),
            rejected_chunk_indexes: Vec::new(),
        };
        for embedding in prepared {
            match count_document_tokens_profiled(
                sizer.as_ref(),
                &embedding.text,
                "tokenize_preflight",
                "tokenize_preflight_calls",
            ) {
                Ok(token_count) if token_count <= embedding.max_document_tokens => {
                    preflight.accepted.push(embedding);
                }
                Ok(token_count) => {
                    report.errors.push(file_error(
                        Some(embedding.path.clone()),
                        format!(
                            "embed preflight failed: payload has {token_count} tokens, exceeding hard_max_tokens {}",
                            embedding.max_document_tokens
                        ),
                    ));
                    failed_docs.insert(embedding.doc_key);
                    preflight.rejected_chunk_indexes.push(embedding.chunk_index);
                }
                Err(err) => {
                    report.errors.push(file_error(
                        Some(embedding.path.clone()),
                        format!("embed preflight token count failed: {err}"),
                    ));
                    failed_docs.insert(embedding.doc_key);
                    preflight.rejected_chunk_indexes.push(embedding.chunk_index);
                }
            }
        }

        Ok(preflight)
    }

    fn document_has_current_canonical_text(
        &self,
        doc_id: i64,
        generation_key: &str,
    ) -> Result<bool> {
        self.storage
            .has_current_document_text(doc_id, generation_key)
    }

    fn replay_fts_dirty_documents(
        &self,
        repair_scope: &UpdateRepairScope,
        report: &mut UpdateReport,
        failed_docs: &mut HashSet<UpdateDocKey>,
    ) -> Result<()> {
        let records = self.get_fts_dirty_documents_for_scope(repair_scope)?;
        if records.is_empty() {
            return Ok(());
        }

        let mut cleared_by_space: HashMap<String, Vec<i64>> = HashMap::new();
        for record in records {
            let space_name = record.space_name;
            let doc_id = record.doc_id;

            if record.chunks.is_empty() {
                self.storage.delete_tantivy_by_doc(&space_name, doc_id)?;
                cleared_by_space.entry(space_name).or_default().push(doc_id);
                continue;
            }

            let document_text = match self.storage.get_document_text(doc_id) {
                Ok(row) => row,
                Err(err) => {
                    failed_docs.insert(update_doc_key(
                        &space_name,
                        &record.collection_path,
                        &record.doc_path,
                    ));
                    report.errors.push(file_error(
                        Some(record.collection_path.join(&record.doc_path)),
                        format!("fts replay canonical text load failed: {err}"),
                    ));
                    continue;
                }
            };
            let policy = resolve_policy(
                &self.config.chunking,
                Some(document_text.extractor_key.as_str()),
                None,
            );

            let entries = record
                .chunks
                .iter()
                .map(|chunk| -> Result<TantivyEntry> {
                    let chunk_body = self.storage.get_chunk_text(chunk.id)?.text;
                    Ok(TantivyEntry {
                        chunk_id: chunk.id,
                        doc_id,
                        filepath: record.doc_path.clone(),
                        semantic_title: record
                            .doc_title_source
                            .semantic_title(record.doc_title.as_str())
                            .map(ToString::to_string),
                        heading: chunk.heading.clone(),
                        body: retrieval_text_with_prefix(
                            chunk_body.as_str(),
                            record
                                .doc_title_source
                                .semantic_title(record.doc_title.as_str()),
                            chunk.heading.as_deref(),
                            policy.contextual_prefix,
                        ),
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            self.storage.delete_tantivy_by_doc(&space_name, doc_id)?;
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

    fn get_fts_dirty_documents_for_scope(
        &self,
        repair_scope: &UpdateRepairScope,
    ) -> Result<Vec<crate::storage::FtsDirtyRecord>> {
        match repair_scope {
            UpdateRepairScope::Global => self.storage.get_fts_dirty_documents(),
            UpdateRepairScope::Space { space_id } => {
                self.storage.get_fts_dirty_documents_in_space(*space_id)
            }
            UpdateRepairScope::Collections { collection_ids } => self
                .storage
                .get_fts_dirty_documents_in_collections(collection_ids),
        }
    }

    fn get_unembedded_chunks_for_scope(
        &self,
        model: &str,
        repair_scope: &UpdateRepairScope,
        after_chunk_id: i64,
        limit: usize,
    ) -> Result<Vec<crate::storage::EmbedRecord>> {
        match repair_scope {
            UpdateRepairScope::Global => {
                self.storage
                    .get_unembedded_chunks(model, after_chunk_id, limit)
            }
            UpdateRepairScope::Space { space_id } => {
                self.storage
                    .get_unembedded_chunks_in_space(model, *space_id, after_chunk_id, limit)
            }
            UpdateRepairScope::Collections { collection_ids } => self
                .storage
                .get_unembedded_chunks_in_collections(model, collection_ids, after_chunk_id, limit),
        }
    }

    fn list_reapable_documents_for_scope(
        &self,
        older_than_days: u32,
        repair_scope: &UpdateRepairScope,
    ) -> Result<Vec<crate::storage::ReapableDocument>> {
        match repair_scope {
            UpdateRepairScope::Global => self.storage.list_reapable_documents(older_than_days),
            UpdateRepairScope::Space { space_id } => self
                .storage
                .list_reapable_documents_in_space(older_than_days, *space_id),
            UpdateRepairScope::Collections { collection_ids } => self
                .storage
                .list_reapable_documents_in_collections(older_than_days, collection_ids),
        }
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
        failed_docs: &mut HashSet<UpdateDocKey>,
    ) -> Result<()> {
        let all_documents = crate::profile::timed_update_stage("sqlite_list_documents", || {
            self.storage.list_documents(target.collection.id, false)
        })?;
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
        let mut failed_walk_prefixes = Vec::new();
        let mut collection_walk_incomplete = false;

        for entry in super::ignore_helpers::build_collection_walk(&target.collection.path) {
            let entry = match entry {
                Ok(item) => item,
                Err(err) => {
                    let error_scope = collect_walk_error_scope(&err);
                    let error_path = error_scope.paths.first().cloned();
                    if error_scope.collection_incomplete || error_scope.paths.is_empty() {
                        collection_walk_incomplete = true;
                    }
                    for path in &error_scope.paths {
                        let failed_path =
                            match collection_relative_path(&target.collection.path, path) {
                                Ok(relative) => {
                                    if relative.is_empty() || relative == "." {
                                        collection_walk_incomplete = true;
                                    } else {
                                        failed_walk_prefixes.push(relative.clone());
                                    }
                                    relative
                                }
                                Err(_) => {
                                    collection_walk_incomplete = true;
                                    path.display().to_string()
                                }
                            };
                        failed_docs.insert(update_doc_key(
                            &target.space,
                            &target.collection.path,
                            &failed_path,
                        ));
                    }
                    report
                        .errors
                        .push(file_error(error_path, format!("walk error: {err}")));
                    continue;
                }
            };

            if !entry
                .file_type()
                .is_some_and(|file_type| file_type.is_file())
            {
                continue;
            }

            if is_hard_ignored_file(entry.path()) {
                continue;
            }

            let relative_path =
                match collection_relative_path(&target.collection.path, entry.path()) {
                    Ok(path) => path,
                    Err(err) => {
                        failed_docs.insert(update_doc_key(
                            &target.space,
                            &target.collection.path,
                            &entry.path().display().to_string(),
                        ));
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
            let policy = resolve_policy(&self.config.chunking, Some(extractor.profile_key()), None);
            let generation_key =
                ingestion_generation_key(extractor.profile_key(), extractor.version(), &policy);

            report.scanned_docs += 1;
            crate::profile::increment_update_count("docs_scanned", 1);
            seen_paths.insert(relative_path.clone());

            let scan_started = Instant::now();
            let metadata = match entry.metadata() {
                Ok(data) => data,
                Err(err) => {
                    let detail = format!("metadata error: {err}");
                    failed_docs.insert(update_doc_key(
                        &target.space,
                        &target.collection.path,
                        &relative_path,
                    ));
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
                    crate::profile::record_update_stage(
                        "scan_metadata_mtime",
                        scan_started.elapsed(),
                    );
                    continue;
                }
            };

            let modified = match modified_token(&metadata) {
                Ok(value) => value,
                Err(err) => {
                    let detail = format!("modified timestamp error: {err}");
                    failed_docs.insert(update_doc_key(
                        &target.space,
                        &target.collection.path,
                        &relative_path,
                    ));
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
                    crate::profile::record_update_stage(
                        "scan_metadata_mtime",
                        scan_started.elapsed(),
                    );
                    continue;
                }
            };

            if let Some(existing) = docs_by_path.get(&relative_path) {
                let has_current_canonical_text =
                    self.document_has_current_canonical_text(existing.id, generation_key.as_str())?;
                if existing.active && existing.modified == modified && has_current_canonical_text {
                    report.skipped_mtime_docs += 1;
                    push_update_decision(
                        report,
                        options,
                        target,
                        &relative_path,
                        UpdateDecisionKind::SkippedMtime,
                        None,
                    );
                    crate::profile::record_update_stage(
                        "scan_metadata_mtime",
                        scan_started.elapsed(),
                    );
                    continue;
                }
            }
            crate::profile::record_update_stage("scan_metadata_mtime", scan_started.elapsed());

            let read_hash_started = Instant::now();
            let bytes = match std::fs::read(entry.path()) {
                Ok(data) => data,
                Err(err) => {
                    let detail = err.to_string();
                    failed_docs.insert(update_doc_key(
                        &target.space,
                        &target.collection.path,
                        &relative_path,
                    ));
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
                    crate::profile::record_update_stage("read_hash", read_hash_started.elapsed());
                    continue;
                }
            };
            let hash = sha256_hex(&bytes);
            crate::profile::record_update_stage("read_hash", read_hash_started.elapsed());
            let mut title = file_title(entry.path());
            let mut title_source = DocumentTitleSource::FilenameFallback;

            let existing = docs_by_path.get(&relative_path).cloned();
            let pending_decision;
            let pending_indexing;
            if let Some(doc) = existing.as_ref() {
                let has_current_canonical_text =
                    self.document_has_current_canonical_text(doc.id, generation_key.as_str())?;
                if doc.hash == hash && has_current_canonical_text {
                    if doc.active {
                        report.skipped_hash_docs += 1;
                        push_update_decision(
                            report,
                            options,
                            target,
                            &relative_path,
                            UpdateDecisionKind::SkippedHash,
                            None,
                        );
                    } else {
                        report.reactivated_docs += 1;
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
                        crate::profile::timed_update_stage("sqlite_refresh_document", || {
                            self.storage.refresh_document_activity(doc.id, &modified)
                        })?;
                    }
                    continue;
                }

                pending_indexing = PendingDocumentIndexing::Updated {
                    reactivated: !doc.active,
                };
                pending_decision = (
                    UpdateDecisionKind::Changed,
                    (!doc.active).then_some("reactivated".to_string()),
                );
            } else {
                pending_indexing = PendingDocumentIndexing::Added;
                pending_decision = (UpdateDecisionKind::New, None);
            }

            if options.dry_run {
                pending_indexing.record(report);
                let (kind, detail) = pending_decision;
                push_update_decision(report, options, target, &relative_path, kind, detail);
                continue;
            }

            let extracted = match crate::profile::timed_update_stage("extract", || {
                extractor.extract(entry.path(), &bytes)
            }) {
                Ok(document) => document,
                Err(err) => {
                    let detail = format!("extract failed: {err}");
                    failed_docs.insert(update_doc_key(
                        &target.space,
                        &target.collection.path,
                        &relative_path,
                    ));
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

            let canonical = crate::profile::timed_update_stage("canonical_text", || {
                build_canonical_document(&extracted)
            });
            let text_hash = sha256_hex(canonical.text.as_bytes());
            let max_document_tokens = effective_chunk_hard_max(&policy);
            let chunk_started = Instant::now();
            let final_chunks_result = match self.embedding_document_sizer.as_ref() {
                Some(sizer) => {
                    let sizer_counter = EmbeddingDocumentSizerCounter {
                        inner: sizer.as_ref(),
                    };
                    chunk_canonical_document_with_counter(
                        &canonical.document,
                        &policy,
                        &sizer_counter,
                    )
                }
                None => Ok(chunk_canonical_document(&canonical.document, &policy)),
            };
            crate::profile::record_update_stage("chunk", chunk_started.elapsed());
            let final_chunks = match final_chunks_result {
                Ok(chunks) => chunks,
                Err(err) => {
                    let detail = format!("chunking failed: {err}");
                    failed_docs.insert(update_doc_key(
                        &target.space,
                        &target.collection.path,
                        &relative_path,
                    ));
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
            crate::profile::increment_update_count("chunks_created", final_chunks.len() as u64);

            let doc_key = update_doc_key(&target.space, &target.collection.path, &relative_path);
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
            let mut prepared_embeddings = Vec::new();
            let mut rejected_chunk_indexes = Vec::new();
            if !chunk_inserts.is_empty() && !options.no_embed && self.embedder.is_some() {
                let prepared = crate::profile::timed_update_stage("embedding_prepare_text", || {
                    prepare_chunk_embeddings(
                        &final_chunks,
                        &doc_key,
                        target,
                        entry.path(),
                        max_document_tokens,
                    )
                });
                let preflight =
                    self.preflight_prepared_embeddings(prepared, report, failed_docs)?;
                prepared_embeddings = preflight.accepted;
                rejected_chunk_indexes = preflight.rejected_chunk_indexes;
                if existing.is_some() && !rejected_chunk_indexes.is_empty() {
                    push_update_decision(
                        report,
                        options,
                        target,
                        &relative_path,
                        UpdateDecisionKind::ExtractFailed,
                        Some("embed preflight failed".to_string()),
                    );
                    continue;
                }
            }

            let replacement =
                crate::profile::timed_update_stage("sqlite_replace_document_generation", || {
                    self.storage
                        .replace_document_generation(DocumentGenerationReplace {
                            collection_id: target.collection.id,
                            path: &relative_path,
                            title: &title,
                            title_source,
                            hash: &hash,
                            modified: &modified,
                            extractor_key: extractor.profile_key(),
                            source_hash: &hash,
                            text_hash: &text_hash,
                            generation_key: &generation_key,
                            text: canonical.text.as_str(),
                            chunks: &chunk_inserts,
                        })
                })?;
            let doc_id = replacement.doc_id;
            let chunk_ids = replacement.chunk_ids;

            if !replacement.old_chunk_ids.is_empty() {
                crate::profile::timed_update_stage("tantivy_delete", || {
                    self.storage
                        .delete_tantivy(&target.space, &replacement.old_chunk_ids)
                })?;
                crate::profile::timed_update_stage("usearch_delete_save", || {
                    self.storage
                        .delete_usearch(&target.space, &replacement.old_chunk_ids)
                })?;
            }

            fts_dirty_by_space
                .entry(target.space.clone())
                .or_default()
                .insert(doc_id);
            for chunk_index in rejected_chunk_indexes {
                if let Some(chunk_id) = chunk_ids.get(chunk_index) {
                    failed_chunk_ids.insert(*chunk_id);
                }
            }

            if !chunk_ids.is_empty() {
                if !options.no_embed && self.embedder.is_some() {
                    for prepared in prepared_embeddings {
                        pending_embeddings.push(PendingChunkEmbedding {
                            chunk_id: chunk_ids[prepared.chunk_index],
                            doc_key: prepared.doc_key,
                            space_name: prepared.space_name,
                            path: prepared.path,
                            text: prepared.text,
                            max_document_tokens: prepared.max_document_tokens,
                        });
                    }
                    if pending_embeddings.len() >= 64 {
                        self.flush_buffered_embeddings(
                            pending_embeddings,
                            failed_chunk_ids,
                            report,
                            failed_docs,
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
                            chunk.text.as_str(),
                            title_source.semantic_title(title.as_str()),
                            chunk.heading.as_deref(),
                            policy.contextual_prefix,
                        ),
                    })
                    .collect::<Vec<_>>();
                crate::profile::increment_update_count("tantivy_entries", entries.len() as u64);
                crate::profile::timed_update_stage("tantivy_add", || {
                    self.storage.index_tantivy(&target.space, &entries)
                })?;
            }

            docs_by_path.insert(
                relative_path.clone(),
                crate::profile::timed_update_stage("sqlite_get_document_after_write", || {
                    self.storage
                        .get_document_by_path(target.collection.id, &relative_path)
                })?
                .ok_or_else(|| {
                    KboltError::Internal(format!(
                        "document missing after upsert: collection={}, path={relative_path}",
                        target.collection.id
                    ))
                })?,
            );
            pending_indexing.record(report);
            let (kind, detail) = pending_decision;
            push_update_decision(report, options, target, &relative_path, kind, detail);
            touched_collection = true;
        }

        let mut missing_docs = docs_by_path
            .values()
            .filter(|doc| {
                doc.active
                    && !seen_paths.contains(&doc.path)
                    && !path_is_under_failed_walk(
                        doc.path.as_str(),
                        &failed_walk_prefixes,
                        collection_walk_incomplete,
                    )
            })
            .cloned()
            .collect::<Vec<_>>();
        missing_docs.sort_by(|left, right| left.path.cmp(&right.path));

        for doc in missing_docs {
            if doc.active && !seen_paths.contains(&doc.path) {
                report.deactivated_docs += 1;
                push_update_decision(
                    report,
                    options,
                    target,
                    &doc.path,
                    UpdateDecisionKind::Deactivated,
                    None,
                );
                if !options.dry_run {
                    crate::profile::timed_update_stage("sqlite_deactivate_document", || {
                        self.storage.deactivate_document(doc.id)
                    })?;
                    touched_collection = true;
                }
            }
        }

        if touched_collection && !options.dry_run {
            crate::profile::timed_update_stage("sqlite_update_collection_timestamp", || {
                self.storage
                    .update_collection_timestamp(target.collection.id)
            })?;
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
struct WalkErrorScope {
    paths: Vec<std::path::PathBuf>,
    collection_incomplete: bool,
}

fn collect_walk_error_scope(err: &ignore::Error) -> WalkErrorScope {
    let mut scope = WalkErrorScope::default();
    collect_walk_error_scope_into(err, &mut scope, false);
    scope
}

fn collect_walk_error_scope_into(
    err: &ignore::Error,
    scope: &mut WalkErrorScope,
    has_path_context: bool,
) {
    match err {
        ignore::Error::Partial(errors) => {
            if errors.is_empty() {
                scope.collection_incomplete = true;
            }
            for err in errors {
                collect_walk_error_scope_into(err, scope, has_path_context);
            }
        }
        ignore::Error::WithLineNumber { err, .. } | ignore::Error::WithDepth { err, .. } => {
            collect_walk_error_scope_into(err, scope, has_path_context);
        }
        ignore::Error::WithPath { path, err } => {
            scope.paths.push(path.clone());
            collect_walk_error_scope_into(err, scope, true);
        }
        ignore::Error::Loop { child, .. } => {
            scope.paths.push(child.clone());
        }
        ignore::Error::Io(_) => {
            if !has_path_context {
                scope.collection_incomplete = true;
            }
        }
        ignore::Error::Glob { .. }
        | ignore::Error::UnrecognizedFileType(_)
        | ignore::Error::InvalidDefinition => {
            scope.collection_incomplete = true;
        }
    }
}

fn path_is_under_failed_walk(
    doc_path: &str,
    failed_walk_prefixes: &[String],
    collection_walk_incomplete: bool,
) -> bool {
    if collection_walk_incomplete {
        return true;
    }

    failed_walk_prefixes.iter().any(|prefix| {
        prefix.is_empty()
            || prefix == "."
            || doc_path == prefix
            || doc_path
                .strip_prefix(prefix.as_str())
                .is_some_and(|suffix| suffix.starts_with('/'))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::path::PathBuf;

    #[test]
    fn walk_error_scope_collects_every_partial_path() {
        let err = ignore::Error::Partial(vec![
            ignore::Error::WithPath {
                path: PathBuf::from("/collection/blocked-a"),
                err: Box::new(ignore::Error::Io(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "blocked",
                ))),
            },
            ignore::Error::WithDepth {
                depth: 2,
                err: Box::new(ignore::Error::Loop {
                    ancestor: PathBuf::from("/collection"),
                    child: PathBuf::from("/collection/blocked-b"),
                }),
            },
        ]);

        let scope = collect_walk_error_scope(&err);

        assert_eq!(
            scope.paths,
            vec![
                PathBuf::from("/collection/blocked-a"),
                PathBuf::from("/collection/blocked-b")
            ]
        );
        assert!(!scope.collection_incomplete);
    }

    #[test]
    fn walk_error_scope_marks_pathless_errors_incomplete() {
        let err = ignore::Error::Partial(vec![
            ignore::Error::WithPath {
                path: PathBuf::from("/collection/blocked"),
                err: Box::new(ignore::Error::Io(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "blocked",
                ))),
            },
            ignore::Error::Io(io::Error::new(io::ErrorKind::Other, "unknown")),
        ]);

        let scope = collect_walk_error_scope(&err);

        assert_eq!(scope.paths, vec![PathBuf::from("/collection/blocked")]);
        assert!(scope.collection_incomplete);
    }

    #[test]
    fn walk_error_scope_treats_ignore_rule_errors_as_incomplete() {
        let err = ignore::Error::WithPath {
            path: PathBuf::from("/collection/.gitignore"),
            err: Box::new(ignore::Error::WithLineNumber {
                line: 4,
                err: Box::new(ignore::Error::Glob {
                    glob: Some("[".to_string()),
                    err: "invalid glob".to_string(),
                }),
            }),
        };

        let scope = collect_walk_error_scope(&err);

        assert_eq!(scope.paths, vec![PathBuf::from("/collection/.gitignore")]);
        assert!(scope.collection_incomplete);
    }
}

#[derive(Debug)]
struct PreparedChunkEmbedding {
    chunk_index: usize,
    doc_key: UpdateDocKey,
    space_name: String,
    path: std::path::PathBuf,
    text: String,
    max_document_tokens: usize,
}

#[derive(Debug, Default)]
struct PreparedEmbeddingPreflight {
    accepted: Vec<PreparedChunkEmbedding>,
    rejected_chunk_indexes: Vec<usize>,
}

#[derive(Debug)]
struct PendingChunkEmbedding {
    chunk_id: i64,
    doc_key: UpdateDocKey,
    space_name: String,
    path: std::path::PathBuf,
    text: String,
    max_document_tokens: usize,
}

#[derive(Default)]
struct EmbeddedPendingBatch {
    embeddings: Vec<(i64, String, Vec<f32>)>,
    failed_chunk_ids: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct UpdateDocKey {
    space: String,
    collection_path: std::path::PathBuf,
    path: String,
}

struct EmbeddingDocumentSizerCounter<'a> {
    inner: &'a dyn crate::models::EmbeddingDocumentSizer,
}

impl crate::ingest::chunk::TokenCounter for EmbeddingDocumentSizerCounter<'_> {
    fn count(&self, text: &str) -> Result<usize> {
        count_document_tokens_profiled(
            self.inner,
            text,
            "tokenize_chunking",
            "tokenize_chunking_calls",
        )
    }
}

fn count_document_tokens_profiled(
    sizer: &dyn crate::models::EmbeddingDocumentSizer,
    text: &str,
    stage: &'static str,
    count: &'static str,
) -> Result<usize> {
    let started = Instant::now();
    let result = sizer.count_document_tokens(text);
    crate::profile::record_update_stage(stage, started.elapsed());
    crate::profile::increment_update_count(count, 1);
    result
}

fn prepare_chunk_embeddings(
    final_chunks: &[crate::ingest::chunk::FinalChunk],
    doc_key: &UpdateDocKey,
    target: &UpdateTarget,
    path: &Path,
    max_document_tokens: usize,
) -> Vec<PreparedChunkEmbedding> {
    final_chunks
        .iter()
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let mut text = chunk.text.clone();
            if text.trim().is_empty() {
                text = " ".to_string();
            }
            PreparedChunkEmbedding {
                chunk_index,
                doc_key: doc_key.clone(),
                space_name: target.space.clone(),
                path: path.to_path_buf(),
                text,
                max_document_tokens,
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
enum UpdateRepairScope {
    Global,
    Space { space_id: i64 },
    Collections { collection_ids: Vec<i64> },
}

impl UpdateRepairScope {
    fn from_options_and_targets(options: &UpdateOptions, targets: &[UpdateTarget]) -> Self {
        if options.collections.is_empty() {
            if let Some(target) = targets.first() {
                if options.space.is_some() {
                    return Self::Space {
                        space_id: target.collection.space_id,
                    };
                }
            }
            return Self::Global;
        }

        let mut collection_ids = targets
            .iter()
            .map(|target| target.collection.id)
            .collect::<Vec<_>>();
        collection_ids.sort_unstable();
        collection_ids.dedup();
        Self::Collections { collection_ids }
    }

    fn allows_space_dense_repair(&self) -> bool {
        !matches!(self, Self::Collections { .. })
    }
}

#[derive(Debug, Clone, Copy)]
enum PendingDocumentIndexing {
    Added,
    Updated { reactivated: bool },
}

impl PendingDocumentIndexing {
    fn record(self, report: &mut UpdateReport) {
        match self {
            Self::Added => {
                report.added_docs += 1;
            }
            Self::Updated { reactivated } => {
                report.updated_docs += 1;
                if reactivated {
                    report.reactivated_docs += 1;
                }
            }
        }
    }
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

fn update_doc_key(space: &str, collection_path: &Path, path: &str) -> UpdateDocKey {
    UpdateDocKey {
        space: space.to_string(),
        collection_path: collection_path.to_path_buf(),
        path: path.to_string(),
    }
}

fn effective_chunk_hard_max(policy: &crate::config::ChunkPolicy) -> usize {
    policy
        .hard_max_tokens
        .max(policy.soft_max_tokens)
        .max(policy.target_tokens)
        .max(1)
}

fn ingestion_generation_key(
    extractor_key: &str,
    extractor_version: u32,
    policy: &crate::config::ChunkPolicy,
) -> String {
    format!(
        "canonical=v{CANONICAL_TEXT_GENERATION};chunker=v{CHUNKER_GENERATION};extractor={extractor_key}:v{extractor_version};chunk=target:{}:soft:{}:hard:{}:overlap:{}:prefix:{}",
        policy.target_tokens,
        policy.soft_max_tokens,
        policy.hard_max_tokens,
        policy.boundary_overlap_tokens,
        policy.contextual_prefix
    )
}
