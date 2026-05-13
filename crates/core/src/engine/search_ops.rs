use super::*;

const DEEP_SELECTED_GENERATED_VARIANTS_MAX: usize = 2;
const DEEP_VARIANT_NEAR_DUPLICATE_SIMILARITY: f32 = 0.98;

impl Engine {
    pub(super) fn initial_search_pipeline(&self, requested_mode: &SearchMode) -> SearchPipeline {
        let mut pipeline = SearchPipeline {
            keyword: matches!(
                requested_mode,
                SearchMode::Keyword | SearchMode::Auto | SearchMode::Deep
            ),
            dense: matches!(requested_mode, SearchMode::Semantic)
                || matches!(requested_mode, SearchMode::Auto | SearchMode::Deep)
                    && self.embedder.is_some(),
            expansion: matches!(requested_mode, SearchMode::Deep),
            rerank: false,
            notices: Vec::new(),
        };

        if matches!(requested_mode, SearchMode::Auto | SearchMode::Deep) && !self.embedder.is_some()
        {
            pipeline.dense = false;
            add_search_pipeline_notice(
                &mut pipeline,
                SearchPipelineStep::Dense,
                SearchPipelineUnavailableReason::NotConfigured,
            );
        }

        pipeline
    }

    pub(super) fn effective_search_mode(
        &self,
        requested_mode: &SearchMode,
        pipeline: &SearchPipeline,
    ) -> SearchMode {
        match requested_mode {
            SearchMode::Auto => {
                if pipeline.dense || pipeline.rerank {
                    SearchMode::Auto
                } else {
                    SearchMode::Keyword
                }
            }
            SearchMode::Keyword => SearchMode::Keyword,
            SearchMode::Semantic => SearchMode::Semantic,
            SearchMode::Deep => SearchMode::Deep,
        }
    }

    pub(super) fn initial_search_candidate_limit(
        &self,
        mode: &SearchMode,
        requested_limit: usize,
        rerank_enabled: bool,
    ) -> usize {
        match mode {
            SearchMode::Keyword | SearchMode::Semantic => requested_limit,
            SearchMode::Auto => {
                if rerank_enabled {
                    requested_limit.max(self.config.ranking.initial_candidate_limit_min)
                } else {
                    requested_limit
                }
            }
            SearchMode::Deep => {
                requested_limit.max(self.config.ranking.initial_candidate_limit_min)
            }
        }
    }

    pub(super) fn search_target_scopes(
        &self,
        targets: &[UpdateTarget],
    ) -> Result<Vec<SearchTargetScope>> {
        let mut grouped: Vec<(String, Vec<i64>)> = Vec::new();
        for target in targets {
            if let Some((_, collection_ids)) =
                grouped.iter_mut().find(|(space, _)| space == &target.space)
            {
                collection_ids.push(target.collection.id);
            } else {
                grouped.push((target.space.clone(), vec![target.collection.id]));
            }
        }

        let mut scopes = Vec::with_capacity(grouped.len());
        for (space, mut collection_ids) in grouped {
            collection_ids.sort_unstable();
            collection_ids.dedup();

            let summary = self
                .storage
                .get_active_search_scope_summary_in_collections(&collection_ids)?;
            scopes.push(SearchTargetScope {
                space,
                collection_ids,
                document_ids: summary.document_ids,
                chunk_count: summary.chunk_count,
                chunk_ids: Mutex::new(None),
            });
        }
        Ok(scopes)
    }

    pub(super) fn max_search_candidates(&self, scopes: &[SearchTargetScope]) -> usize {
        scopes
            .iter()
            .map(|scope| scope.chunk_count)
            .fold(0usize, usize::saturating_add)
    }

    fn chunk_ids_for_scope(&self, scope: &SearchTargetScope) -> Result<Vec<i64>> {
        let mut chunk_ids = scope
            .chunk_ids
            .lock()
            .map_err(|_| CoreError::poisoned("search scope chunk ids"))?;
        if let Some(chunk_ids) = chunk_ids.as_ref() {
            return Ok(chunk_ids.clone());
        }

        let loaded = self
            .storage
            .get_active_chunk_ids_in_collections(&scope.collection_ids)?;
        *chunk_ids = Some(loaded.clone());
        Ok(loaded)
    }

    pub(super) fn rank_chunks_for_mode(
        &self,
        mode: &SearchMode,
        scopes: &[SearchTargetScope],
        query: &str,
        limit: usize,
        min_score: f32,
        pipeline: &mut SearchPipeline,
    ) -> Result<Vec<RankedChunk>> {
        match mode {
            SearchMode::Keyword => self.rank_keyword_chunks(scopes, query, limit, min_score),
            SearchMode::Auto => self.rank_auto_chunks(scopes, query, limit, min_score, pipeline),
            SearchMode::Semantic => self.rank_semantic_chunks(scopes, query, limit, min_score),
            SearchMode::Deep => self.rank_deep_chunks(scopes, query, limit, min_score, pipeline),
        }
    }

    pub(super) fn rank_keyword_chunks(
        &self,
        scopes: &[SearchTargetScope],
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<RankedChunk>> {
        let boosts = &self.config.ranking.bm25_boosts;
        let mut candidates = Vec::new();
        for scope in scopes {
            let hits = self.storage.query_bm25_in_documents(
                &scope.space,
                query,
                &[
                    ("title", boosts.title),
                    ("heading", boosts.heading),
                    ("body", boosts.body),
                    ("filepath", boosts.filepath),
                ],
                &scope.document_ids,
                limit,
            )?;
            for hit in hits {
                candidates.push(SearchHitCandidate {
                    chunk_id: hit.chunk_id,
                    bm25_score: hit.score,
                });
            }
        }

        candidates.sort_by(|left, right| right.bm25_score.total_cmp(&left.bm25_score));
        let max_bm25 = candidates
            .iter()
            .map(|candidate| candidate.bm25_score)
            .fold(0.0_f32, f32::max);

        let mut ranked = Vec::new();
        let mut seen_chunks = HashSet::new();
        for candidate in candidates {
            if !seen_chunks.insert(candidate.chunk_id) {
                continue;
            }

            let normalized_score = if max_bm25 > 0.0 {
                candidate.bm25_score / max_bm25
            } else {
                0.0
            };
            if normalized_score < min_score {
                continue;
            }

            ranked.push(RankedChunk {
                chunk_id: candidate.chunk_id,
                score: normalized_score,
                fusion: normalized_score,
                reranker: None,
                bm25: Some(normalized_score),
                dense: None,
                original_rank: None,
            });
            if ranked.len() >= limit {
                break;
            }
        }

        Ok(ranked)
    }

    pub(super) fn rank_semantic_chunks(
        &self,
        scopes: &[SearchTargetScope],
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<RankedChunk>> {
        let Some(embedder) = self.embedder.as_ref() else {
            return Err(KboltError::InvalidInput(
                "semantic search requires a configured embedder role. bind [roles.embedder] to an embedding provider profile in index.toml".to_string(),
            )
            .into());
        };

        let vectors = embedder.embed_batch(
            crate::models::EmbeddingInputKind::Query,
            &[query.to_string()],
        )?;
        if vectors.len() != 1 || vectors[0].is_empty() {
            return Err(KboltError::Inference(
                "embedder must return one non-empty query vector".to_string(),
            )
            .into());
        }

        self.rank_semantic_chunks_with_embedding(scopes, &vectors[0], limit, min_score)
    }

    fn rank_semantic_chunks_with_embedding(
        &self,
        scopes: &[SearchTargetScope],
        query_vector: &[f32],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<RankedChunk>> {
        if query_vector.is_empty() {
            return Err(
                KboltError::Inference("query embedding must not be empty".to_string()).into(),
            );
        }

        let mut candidates = Vec::new();
        for scope in scopes {
            let chunk_ids = self.chunk_ids_for_scope(scope)?;
            let hits = self.storage.query_dense_in_chunks(
                &scope.space,
                query_vector,
                &chunk_ids,
                limit,
            )?;
            for hit in hits {
                candidates.push(hit);
            }
        }
        candidates.sort_by(|left, right| left.distance.total_cmp(&right.distance));

        let mut ranked = Vec::new();
        let mut seen_chunks = HashSet::new();
        for candidate in candidates {
            if !seen_chunks.insert(candidate.chunk_id) {
                continue;
            }

            let dense_score = dense_distance_to_score(candidate.distance);
            if dense_score < min_score {
                continue;
            }

            ranked.push(RankedChunk {
                chunk_id: candidate.chunk_id,
                score: dense_score,
                fusion: dense_score,
                reranker: None,
                bm25: None,
                dense: Some(dense_score),
                original_rank: None,
            });
            if ranked.len() >= limit {
                break;
            }
        }

        Ok(ranked)
    }

    pub(super) fn rank_auto_chunks(
        &self,
        scopes: &[SearchTargetScope],
        query: &str,
        limit: usize,
        min_score: f32,
        pipeline: &mut SearchPipeline,
    ) -> Result<Vec<RankedChunk>> {
        let keyword = if self.embedder.is_some() {
            let (keyword_result, semantic_result) = std::thread::scope(|scope| {
                let keyword_handle =
                    scope.spawn(|| self.rank_keyword_chunks(scopes, query, limit, 0.0));
                let semantic_handle =
                    scope.spawn(|| self.rank_semantic_chunks(scopes, query, limit, 0.0));
                (keyword_handle.join(), semantic_handle.join())
            });

            let keyword = match keyword_result {
                Ok(result) => result?,
                Err(_) => {
                    return Err(
                        KboltError::Internal("keyword search worker panicked".to_string()).into(),
                    )
                }
            };
            let semantic = match semantic_result {
                Ok(Ok(ranked)) => {
                    pipeline.dense = true;
                    ranked
                }
                Ok(Err(err)) if is_model_not_available_error(&err) => {
                    pipeline.dense = false;
                    add_search_pipeline_notice(
                        pipeline,
                        SearchPipelineStep::Dense,
                        SearchPipelineUnavailableReason::ModelNotAvailable,
                    );
                    Vec::new()
                }
                Ok(Err(err)) => return Err(err),
                Err(_) => {
                    return Err(
                        KboltError::Internal("semantic search worker panicked".to_string()).into(),
                    )
                }
            };

            return Ok(fuse_ranked_chunks(
                keyword,
                semantic,
                &self.config.ranking.hybrid_fusion,
                limit,
                min_score,
            ));
        } else {
            self.rank_keyword_chunks(scopes, query, limit, min_score)?
        };

        Ok(keyword)
    }

    pub(super) fn rank_deep_chunks(
        &self,
        scopes: &[SearchTargetScope],
        query: &str,
        limit: usize,
        min_score: f32,
        pipeline: &mut SearchPipeline,
    ) -> Result<Vec<RankedChunk>> {
        let Some(expander) = self.expander.as_ref() else {
            return Err(KboltError::InvalidInput(
                "deep search needs the optional expander. run `kbolt local enable deep`."
                    .to_string(),
            )
            .into());
        };

        let normalized_query = crate::models::normalize_query_text(query);
        let mut variants = vec![normalized_query.clone()];
        let max_generated = self.config.ranking.deep_variants_max.saturating_sub(1);
        if max_generated > 0 {
            let generated = expander.expand(query, max_generated)?;
            let mut seen = HashSet::from([normalized_query.to_ascii_lowercase()]);
            for generated_query in generated {
                let text = crate::models::normalize_query_text(&generated_query);
                if text.is_empty() {
                    continue;
                }

                let key = text.to_ascii_lowercase();
                if !seen.insert(key) {
                    continue;
                }

                variants.push(text);
                if variants.len() >= self.config.ranking.deep_variants_max {
                    break;
                }
            }
        }
        pipeline.expansion = true;
        let variant_vectors = if let Some(embedder) = self.embedder.as_ref() {
            match embedder.embed_batch(crate::models::EmbeddingInputKind::Query, &variants) {
                Ok(vectors) => {
                    if vectors.len() != variants.len() {
                        return Err(KboltError::Inference(format!(
                            "embedder returned {} vectors for {} deep variants",
                            vectors.len(),
                            variants.len()
                        ))
                        .into());
                    }
                    if let Some((index, _)) = vectors
                        .iter()
                        .enumerate()
                        .find(|(_, vector)| vector.is_empty())
                    {
                        return Err(KboltError::Inference(format!(
                            "embedder returned an empty vector for deep variant {index}"
                        ))
                        .into());
                    }
                    pipeline.dense = true;
                    Some(vectors)
                }
                Err(err) if is_model_not_available_error(&err) => {
                    pipeline.dense = false;
                    add_search_pipeline_notice(
                        pipeline,
                        SearchPipelineStep::Dense,
                        SearchPipelineUnavailableReason::ModelNotAvailable,
                    );
                    None
                }
                Err(err) => return Err(err),
            }
        } else {
            None
        };
        let selected_variant_indices = select_deep_variant_indices(
            &variants,
            variant_vectors.as_deref(),
            max_generated.min(DEEP_SELECTED_GENERATED_VARIANTS_MAX),
        );
        let variant_results: Vec<Result<Vec<RankedChunk>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = selected_variant_indices
                .iter()
                .map(|&variant_index| {
                    let vv = &variant_vectors;
                    let variant = variants[variant_index].clone();
                    scope.spawn(move || {
                        let keyword = self.rank_keyword_chunks(scopes, &variant, limit, 0.0)?;
                        let semantic = vv
                            .as_ref()
                            .and_then(|vectors| vectors.get(variant_index))
                            .map(|vector| {
                                self.rank_semantic_chunks_with_embedding(scopes, vector, limit, 0.0)
                            })
                            .transpose()?
                            .unwrap_or_default();
                        Ok(fuse_ranked_chunks(
                            keyword,
                            semantic,
                            &self.config.ranking.hybrid_fusion,
                            limit,
                            0.0,
                        ))
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|handle| {
                    handle.join().unwrap_or_else(|_| {
                        Err(
                            KboltError::Internal("deep variant search worker panicked".to_string())
                                .into(),
                        )
                    })
                })
                .collect()
        });

        let variant_results = variant_results.into_iter().collect::<Result<Vec<_>>>()?;
        Ok(aggregate_deep_variant_rankings(
            variant_results,
            self.config.ranking.deep_variant_rrf_k,
            limit,
            min_score,
        ))
    }

    pub(super) fn assemble_search_results(
        &self,
        query: &str,
        mode: &SearchMode,
        ranked_chunks: Vec<RankedChunk>,
        collections_by_id: &HashMap<i64, SearchCollectionMeta>,
        debug: bool,
        apply_rerank: bool,
        pipeline: &mut SearchPipeline,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let chunk_ids = ranked_chunks
            .iter()
            .map(|candidate| candidate.chunk_id)
            .collect::<Vec<_>>();
        let chunk_rows = self.storage.get_chunks(&chunk_ids)?;
        let chunk_by_id = chunk_rows
            .into_iter()
            .map(|chunk| (chunk.id, chunk))
            .collect::<HashMap<_, _>>();

        let unique_doc_ids = chunk_by_id
            .values()
            .map(|chunk| chunk.doc_id)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let docs_by_id: HashMap<i64, DocumentRow> = self
            .storage
            .get_documents_by_ids(&unique_doc_ids)?
            .into_iter()
            .map(|doc| (doc.id, doc))
            .collect();

        let mut candidates = Vec::new();
        for ranked in ranked_chunks {
            let Some(chunk) = chunk_by_id.get(&ranked.chunk_id) else {
                continue;
            };

            let Some(document) = docs_by_id.get(&chunk.doc_id) else {
                continue;
            };
            if !document.active {
                continue;
            }

            let Some(collection) = collections_by_id.get(&document.collection_id) else {
                continue;
            };

            candidates.push(PendingSearchCandidate {
                doc_id: chunk.doc_id,
                docid: short_docid(&document.hash),
                path: format!("{}/{}", collection.collection, document.path),
                title: document.title.clone(),
                title_source: document.title_source,
                space: collection.space.clone(),
                collection: collection.collection.clone(),
                heading: chunk.heading.clone(),
                chunk: chunk.clone(),
                bm25: ranked.bm25,
                dense: ranked.dense,
                fusion: ranked.fusion,
                reranker: ranked.reranker,
                original_rank: ranked.original_rank,
                final_score: ranked.score,
            });
        }

        let mut text_by_doc: HashMap<i64, DocumentTextRow> = HashMap::new();
        let mut chunks_by_doc: HashMap<i64, Vec<ChunkRow>> = HashMap::new();

        if apply_rerank && !candidates.is_empty() {
            let Some(reranker) = self.reranker.as_ref() else {
                pipeline.rerank = false;
                add_search_pipeline_notice(
                    pipeline,
                    SearchPipelineStep::Rerank,
                    SearchPipelineUnavailableReason::NotConfigured,
                );
                return finalize_search_results(
                    candidates,
                    &self.storage,
                    &mut text_by_doc,
                    &mut chunks_by_doc,
                    &self.config.chunking,
                    debug,
                    limit,
                );
            };
            let rerank_count = rerank_candidate_count(
                limit,
                candidates.len(),
                self.config.ranking.rerank_candidates_min,
                self.config.ranking.rerank_candidates_max,
            );
            let protected_original_docs = if matches!(mode, SearchMode::Deep) {
                protected_original_doc_count(rerank_count)
            } else {
                0
            };
            let representative_indices =
                select_rerank_representatives(&candidates, rerank_count, protected_original_docs);
            let mut rerank_doc_ids = Vec::new();
            for &idx in &representative_indices {
                rerank_doc_ids.push(candidates[idx].doc_id);
            }
            ensure_document_texts_loaded(&self.storage, &rerank_doc_ids, &mut text_by_doc)?;

            let mut rerank_inputs = Vec::new();
            for &idx in &representative_indices {
                let candidate = &candidates[idx];
                let rerank_input = build_rerank_input(candidate, &text_by_doc)?;
                rerank_inputs.push(rerank_input);
            }
            let raw_scores = match reranker.rerank(query, &rerank_inputs) {
                Ok(scores) => {
                    pipeline.rerank = true;
                    Some(scores)
                }
                Err(err) if is_model_not_available_error(&err) => {
                    pipeline.rerank = false;
                    add_search_pipeline_notice(
                        pipeline,
                        SearchPipelineStep::Rerank,
                        SearchPipelineUnavailableReason::ModelNotAvailable,
                    );
                    None
                }
                Err(err) => return Err(err),
            };
            let Some(raw_scores) = raw_scores else {
                return finalize_search_results(
                    candidates,
                    &self.storage,
                    &mut text_by_doc,
                    &mut chunks_by_doc,
                    &self.config.chunking,
                    debug,
                    limit,
                );
            };
            if raw_scores.len() != rerank_inputs.len() {
                return Err(KboltError::Inference(format!(
                    "reranker returned {} scores for {} candidates",
                    raw_scores.len(),
                    rerank_inputs.len()
                ))
                .into());
            }
            let doc_reranker_scores: HashMap<i64, f32> = rerank_doc_ids
                .into_iter()
                .zip(raw_scores.into_iter())
                .collect();
            apply_reranker_scores(&mut candidates, &doc_reranker_scores);
        }

        finalize_search_results(
            candidates,
            &self.storage,
            &mut text_by_doc,
            &mut chunks_by_doc,
            &self.config.chunking,
            debug,
            limit,
        )
    }
}

#[derive(Debug, Clone)]
struct PendingSearchCandidate {
    doc_id: i64,
    docid: String,
    path: String,
    title: String,
    title_source: DocumentTitleSource,
    space: String,
    collection: String,
    heading: Option<String>,
    chunk: ChunkRow,
    bm25: Option<f32>,
    dense: Option<f32>,
    fusion: f32,
    reranker: Option<f32>,
    original_rank: Option<usize>,
    final_score: f32,
}

fn build_rerank_input(
    candidate: &PendingSearchCandidate,
    text_by_doc: &HashMap<i64, DocumentTextRow>,
) -> Result<String> {
    let document_text = candidate_document_text(candidate, text_by_doc)?;
    let canonical_body =
        crate::storage::chunk_text_from_canonical(document_text.text.as_str(), &candidate.chunk)?;
    let rerank_body = crate::ingest::chunk::chunk_retrieval_body(
        canonical_body.as_str(),
        candidate.chunk.retrieval_prefix.as_deref(),
    );

    Ok(retrieval_text_with_prefix(
        rerank_body.as_str(),
        candidate
            .title_source
            .semantic_title(candidate.title.as_str()),
        candidate.heading.as_deref(),
        true,
    ))
}

/// Selects one representative candidate per unique document for reranking.
/// Candidates are already sorted by first-stage fusion score, so the first candidate for
/// each document is its highest-scoring chunk (MaxP strategy).
fn select_rerank_representatives(
    candidates: &[PendingSearchCandidate],
    max_docs: usize,
    protected_original_docs: usize,
) -> Vec<usize> {
    let mut seen_docs = HashSet::new();
    let mut indices = Vec::new();

    if protected_original_docs > 0 {
        let mut original_candidates = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                candidate
                    .original_rank
                    .map(|rank| (rank, index, candidate.doc_id))
            })
            .collect::<Vec<_>>();
        original_candidates
            .sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

        for (_, index, doc_id) in original_candidates {
            if seen_docs.insert(doc_id) {
                indices.push(index);
                if indices.len() >= protected_original_docs.min(max_docs) {
                    break;
                }
            }
        }
    }

    for (i, candidate) in candidates.iter().enumerate() {
        if seen_docs.insert(candidate.doc_id) {
            indices.push(i);
            if indices.len() >= max_docs {
                break;
            }
        }
    }
    indices
}

fn reranked_doc_prior(doc_rank: usize) -> f32 {
    1.0 / ((doc_rank.max(1) + 1) as f32).log2()
}

/// Applies doc-level reranker ordering to the candidate pool. Reranked
/// documents are ranked by reranker score, while chunk-level fusion preserves
/// ordering within each document. Chunks from non-reranked documents get a
/// fallback score below the reranked pool.
fn apply_reranker_scores(
    candidates: &mut [PendingSearchCandidate],
    doc_reranker_scores: &HashMap<i64, f32>,
) {
    if doc_reranker_scores.is_empty() {
        return;
    }

    let mut reranked_docs = Vec::new();
    let mut seen_reranked_docs = HashSet::new();
    let mut max_fusion_by_doc = HashMap::new();

    for (index, candidate) in candidates.iter().enumerate() {
        let Some(&reranker_score) = doc_reranker_scores.get(&candidate.doc_id) else {
            continue;
        };

        if seen_reranked_docs.insert(candidate.doc_id) {
            reranked_docs.push((candidate.doc_id, reranker_score, index));
        }

        max_fusion_by_doc
            .entry(candidate.doc_id)
            .and_modify(|max_fusion: &mut f32| {
                *max_fusion = f32::max(*max_fusion, candidate.fusion)
            })
            .or_insert(candidate.fusion);
    }

    reranked_docs.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.2.cmp(&right.2))
    });

    let doc_rank_by_doc: HashMap<i64, usize> = reranked_docs
        .iter()
        .enumerate()
        .map(|(index, (doc_id, _, _))| (*doc_id, index + 1))
        .collect();

    let mut lowest_reranked_score = f32::INFINITY;

    for candidate in candidates.iter_mut() {
        if let Some(reranker_score) = doc_reranker_scores.get(&candidate.doc_id).copied() {
            candidate.reranker = Some(reranker_score);

            let doc_rank = *doc_rank_by_doc
                .get(&candidate.doc_id)
                .expect("reranked document rank missing");
            let doc_prior = reranked_doc_prior(doc_rank);
            let max_fusion = *max_fusion_by_doc
                .get(&candidate.doc_id)
                .expect("reranked document max fusion missing");
            let chunk_scale = if max_fusion > 0.0 {
                (candidate.fusion / max_fusion).clamp(0.0, 1.0)
            } else {
                1.0
            };

            candidate.final_score = doc_prior * chunk_scale;
            lowest_reranked_score = lowest_reranked_score.min(candidate.final_score);
        }
    }

    let mut fallback_score = lowest_reranked_score.next_down();
    for candidate in candidates.iter_mut() {
        if candidate.reranker.is_none() {
            candidate.final_score = fallback_score;
            fallback_score = fallback_score.next_down();
        }
    }
}

fn finalize_search_results(
    mut candidates: Vec<PendingSearchCandidate>,
    storage: &Storage,
    text_by_doc: &mut HashMap<i64, DocumentTextRow>,
    chunks_by_doc: &mut HashMap<i64, Vec<ChunkRow>>,
    chunking: &crate::config::ChunkingConfig,
    debug: bool,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    candidates.sort_by(|left, right| right.final_score.total_cmp(&left.final_score));
    let candidates = candidates.into_iter().take(limit).collect::<Vec<_>>();
    let final_doc_ids = candidates
        .iter()
        .map(|candidate| candidate.doc_id)
        .collect::<Vec<_>>();
    ensure_document_texts_loaded(storage, &final_doc_ids, text_by_doc)?;
    let mut neighbor_doc_ids = Vec::new();
    for candidate in &candidates {
        let document_text = candidate_document_text(candidate, text_by_doc)?;
        if result_neighbor_window(chunking, document_text) > 0 {
            neighbor_doc_ids.push(candidate.doc_id);
        }
    }
    neighbor_doc_ids.sort_unstable();
    neighbor_doc_ids.dedup();
    ensure_neighbor_chunks_loaded(storage, &neighbor_doc_ids, chunks_by_doc)?;

    let mut results = Vec::new();
    for candidate in candidates {
        let document_text = candidate_document_text(&candidate, text_by_doc)?;
        let neighbor_window = result_neighbor_window(chunking, document_text);
        let text = search_text_with_canonical_neighbors(
            document_text.text.as_str(),
            &candidate.chunk,
            candidate_neighbors(&candidate, chunks_by_doc, neighbor_window),
            neighbor_window,
        )?;

        results.push(SearchResult {
            docid: candidate.docid,
            path: candidate.path,
            title: candidate.title,
            space: candidate.space,
            collection: candidate.collection,
            heading: candidate.heading,
            text,
            score: candidate.final_score,
            signals: if debug {
                Some(SearchSignals {
                    bm25: candidate.bm25,
                    dense: candidate.dense,
                    fusion: candidate.fusion,
                    reranker: candidate.reranker,
                })
            } else {
                None
            },
        });
    }

    Ok(results)
}

fn ensure_document_texts_loaded(
    storage: &Storage,
    doc_ids: &[i64],
    text_by_doc: &mut HashMap<i64, DocumentTextRow>,
) -> Result<()> {
    let missing = doc_ids
        .iter()
        .copied()
        .filter(|doc_id| !text_by_doc.contains_key(doc_id))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    text_by_doc.extend(storage.get_document_texts(&missing)?);
    Ok(())
}

fn ensure_neighbor_chunks_loaded(
    storage: &Storage,
    doc_ids: &[i64],
    chunks_by_doc: &mut HashMap<i64, Vec<ChunkRow>>,
) -> Result<()> {
    let missing = doc_ids
        .iter()
        .copied()
        .filter(|doc_id| !chunks_by_doc.contains_key(doc_id))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    chunks_by_doc.extend(storage.get_chunks_for_documents(&missing)?);
    Ok(())
}

fn result_neighbor_window(
    chunking: &crate::config::ChunkingConfig,
    document_text: &DocumentTextRow,
) -> usize {
    resolve_policy(chunking, Some(document_text.extractor_key.as_str()), None).neighbor_window
}

fn candidate_neighbors<'a>(
    candidate: &PendingSearchCandidate,
    chunks_by_doc: &'a HashMap<i64, Vec<ChunkRow>>,
    neighbor_window: usize,
) -> Option<&'a Vec<ChunkRow>> {
    if neighbor_window == 0 {
        return None;
    }

    chunks_by_doc.get(&candidate.doc_id)
}

fn candidate_document_text<'a>(
    candidate: &PendingSearchCandidate,
    text_by_doc: &'a HashMap<i64, DocumentTextRow>,
) -> Result<&'a DocumentTextRow> {
    text_by_doc.get(&candidate.doc_id).ok_or_else(|| {
        KboltError::Internal(format!(
            "document text cache missing for {}",
            candidate.doc_id
        ))
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DbsfHybridFusionConfig, HybridFusionConfig, HybridFusionMode, LinearHybridFusionConfig,
        RrfHybridFusionConfig,
    };
    use crate::ingest::chunk::FinalChunkKind;

    fn candidate(doc_id: i64, chunk_id: i64, fusion: f32) -> PendingSearchCandidate {
        PendingSearchCandidate {
            doc_id,
            docid: format!("#{doc_id}"),
            path: format!("doc-{doc_id}.md"),
            title: format!("Doc {doc_id}"),
            title_source: DocumentTitleSource::Extracted,
            space: "work".to_string(),
            collection: "docs".to_string(),
            heading: None,
            chunk: ChunkRow {
                id: chunk_id,
                doc_id,
                seq: chunk_id as i32,
                offset: 0,
                length: 0,
                heading: None,
                kind: FinalChunkKind::Paragraph,
                retrieval_prefix: None,
            },
            bm25: None,
            dense: None,
            fusion,
            reranker: None,
            original_rank: None,
            final_score: fusion,
        }
    }

    fn candidate_with_original_rank(
        doc_id: i64,
        chunk_id: i64,
        fusion: f32,
        original_rank: usize,
    ) -> PendingSearchCandidate {
        let mut candidate = candidate(doc_id, chunk_id, fusion);
        candidate.original_rank = Some(original_rank);
        candidate
    }

    fn linear_hybrid_fusion(dense_weight: f32, bm25_weight: f32) -> HybridFusionConfig {
        HybridFusionConfig {
            mode: HybridFusionMode::Linear,
            linear: LinearHybridFusionConfig {
                dense_weight,
                bm25_weight,
            },
            dbsf: DbsfHybridFusionConfig::default(),
            rrf: RrfHybridFusionConfig::default(),
        }
    }

    fn rrf_hybrid_fusion(k: usize) -> HybridFusionConfig {
        HybridFusionConfig {
            mode: HybridFusionMode::Rrf,
            linear: LinearHybridFusionConfig::default(),
            dbsf: DbsfHybridFusionConfig::default(),
            rrf: RrfHybridFusionConfig { k },
        }
    }

    fn ranked_chunk(
        chunk_id: i64,
        score: f32,
        bm25: Option<f32>,
        dense: Option<f32>,
    ) -> RankedChunk {
        RankedChunk {
            chunk_id,
            score,
            fusion: score,
            reranker: None,
            bm25,
            dense,
            original_rank: None,
        }
    }

    #[test]
    fn apply_reranker_scores_uses_doc_rank_and_within_doc_fusion_scale() {
        let mut candidates = vec![
            candidate(10, 100, 0.90),
            candidate(20, 200, 1.00),
            candidate(10, 101, 0.45),
            candidate(30, 300, 0.30),
        ];
        let doc_reranker_scores = HashMap::from([(10, 0.80), (20, 0.95)]);

        apply_reranker_scores(&mut candidates, &doc_reranker_scores);

        assert_eq!(candidates[0].reranker, Some(0.80));
        assert_eq!(candidates[1].reranker, Some(0.95));
        assert_eq!(candidates[2].reranker, Some(0.80));
        assert_eq!(candidates[3].reranker, None);

        assert!((candidates[1].final_score - 1.0).abs() < 1e-6);
        assert!((candidates[0].final_score - reranked_doc_prior(2)).abs() < 1e-6);
        assert!((candidates[2].final_score - (reranked_doc_prior(2) * 0.5)).abs() < 1e-6);
        assert!(candidates[3].final_score < candidates[2].final_score);
    }

    #[test]
    fn select_rerank_representatives_protects_original_docs_before_fill() {
        let candidates = vec![
            candidate(10, 100, 0.99),
            candidate(20, 200, 0.98),
            candidate_with_original_rank(30, 300, 0.70, 1),
            candidate_with_original_rank(40, 400, 0.60, 2),
            candidate_with_original_rank(30, 301, 0.55, 3),
        ];

        let selected = select_rerank_representatives(&candidates, 3, 2);

        assert_eq!(selected, vec![2, 3, 0]);
    }

    #[test]
    fn linear_fusion_prefers_strong_dense_only_hit_over_weaker_overlap() {
        let keyword = vec![
            ranked_chunk(10, 1.0, Some(1.0), None),
            ranked_chunk(20, 0.6, Some(0.6), None),
        ];
        let semantic = vec![
            ranked_chunk(30, 0.95, None, Some(0.95)),
            ranked_chunk(10, 0.40, None, Some(0.40)),
        ];

        let fused = fuse_ranked_chunks(keyword, semantic, &linear_hybrid_fusion(0.7, 0.3), 10, 0.0);

        assert_eq!(fused[0].chunk_id, 30);
        assert_eq!(fused[1].chunk_id, 10);
        assert!(fused[0].score > fused[1].score);
        assert_eq!(fused[0].dense, Some(0.95));
        assert_eq!(fused[1].bm25, Some(1.0));
        assert_eq!(fused[1].dense, Some(0.40));
    }

    #[test]
    fn linear_fusion_falls_back_to_single_signal_when_dense_is_missing() {
        let keyword = vec![
            ranked_chunk(10, 1.0, Some(1.0), None),
            ranked_chunk(20, 0.5, Some(0.5), None),
        ];

        let fused = fuse_ranked_chunks(
            keyword,
            Vec::new(),
            &linear_hybrid_fusion(0.7, 0.3),
            10,
            0.0,
        );

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].chunk_id, 10);
        assert_eq!(fused[0].score, 1.0);
        assert_eq!(fused[1].chunk_id, 20);
    }

    #[test]
    fn dbsf_normalization_downweights_flat_branch_head() {
        let keyword = vec![
            ranked_chunk(10, 1.00, Some(1.00), None),
            ranked_chunk(30, 0.99, Some(0.99), None),
            ranked_chunk(40, 0.98, Some(0.98), None),
        ];

        let normalized = normalize_scores_by_dbsf(&keyword, 3.0);

        assert!(normalized[&10] < 1.0);
        assert!(normalized[&10] > normalized[&30]);
        assert!(normalized[&30] > normalized[&40]);
    }

    #[test]
    fn rrf_fusion_uses_rank_positions_only() {
        let keyword = vec![
            ranked_chunk(10, 0.30, Some(0.30), None),
            ranked_chunk(20, 0.20, Some(0.20), None),
        ];
        let semantic = vec![
            ranked_chunk(20, 0.99, None, Some(0.99)),
            ranked_chunk(30, 0.98, None, Some(0.98)),
        ];

        let fused = fuse_ranked_chunks(keyword, semantic, &rrf_hybrid_fusion(60), 10, 0.0);

        assert_eq!(fused[0].chunk_id, 20);
        assert_eq!(fused[1].chunk_id, 10);
        assert_eq!(fused[2].chunk_id, 30);
    }

    #[test]
    fn deep_variant_aggregation_combines_variant_rrf_scores() {
        let variant_results = vec![
            vec![
                ranked_chunk(10, 1.0, Some(1.0), None),
                ranked_chunk(20, 0.8, Some(0.8), None),
            ],
            vec![
                ranked_chunk(10, 0.9, None, Some(0.9)),
                ranked_chunk(40, 0.75, None, Some(0.75)),
                ranked_chunk(30, 0.7, None, Some(0.7)),
            ],
        ];

        let fused = aggregate_deep_variant_rankings(variant_results, 1, 10, 0.0);

        assert_eq!(fused[0].chunk_id, 10);
        assert_eq!(fused.last().map(|item| item.chunk_id), Some(30));
        assert_eq!(fused[0].original_rank, Some(1));
        assert_eq!(
            fused
                .iter()
                .find(|item| item.chunk_id == 20)
                .and_then(|item| item.original_rank),
            Some(2)
        );
    }

    #[test]
    fn select_deep_variant_indices_prefers_non_duplicate_generated_variants() {
        let variants = vec![
            "original".to_string(),
            "close one".to_string(),
            "close two".to_string(),
            "diverse".to_string(),
        ];
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.01],
            vec![0.98, 0.02],
            vec![0.0, 1.0],
        ];

        let selected = select_deep_variant_indices(&variants, Some(&vectors), 2);

        assert_eq!(selected, vec![0, 1, 3]);
    }
}

fn fuse_ranked_chunks(
    keyword: Vec<RankedChunk>,
    semantic: Vec<RankedChunk>,
    hybrid_fusion: &config::HybridFusionConfig,
    limit: usize,
    min_score: f32,
) -> Vec<RankedChunk> {
    if semantic.is_empty() {
        return finalize_ranked_chunks(keyword, limit, min_score);
    }

    if keyword.is_empty() {
        return finalize_ranked_chunks(semantic, limit, min_score);
    }

    match hybrid_fusion.mode {
        config::HybridFusionMode::Linear => fuse_ranked_chunks_linear(
            keyword,
            semantic,
            hybrid_fusion.linear.dense_weight,
            hybrid_fusion.linear.bm25_weight,
            limit,
            min_score,
        ),
        config::HybridFusionMode::Dbsf => fuse_ranked_chunks_dbsf(
            keyword,
            semantic,
            hybrid_fusion.dbsf.dense_weight,
            hybrid_fusion.dbsf.bm25_weight,
            hybrid_fusion.dbsf.stddevs,
            limit,
            min_score,
        ),
        config::HybridFusionMode::Rrf => {
            fuse_ranked_chunks_rrf(keyword, semantic, hybrid_fusion.rrf.k, limit, min_score)
        }
    }
}

fn fuse_ranked_chunks_rrf(
    keyword: Vec<RankedChunk>,
    semantic: Vec<RankedChunk>,
    rrf_k: usize,
    limit: usize,
    min_score: f32,
) -> Vec<RankedChunk> {
    let mut bm25_rank = HashMap::new();
    let mut bm25_score = HashMap::new();
    for (index, item) in keyword.iter().enumerate() {
        bm25_rank.insert(item.chunk_id, index + 1);
        bm25_score.insert(item.chunk_id, item.score);
    }

    let mut dense_rank = HashMap::new();
    let mut dense_score = HashMap::new();
    for (index, item) in semantic.iter().enumerate() {
        dense_rank.insert(item.chunk_id, index + 1);
        dense_score.insert(item.chunk_id, item.score);
    }

    let mut all_chunk_ids = HashSet::new();
    for item in &keyword {
        all_chunk_ids.insert(item.chunk_id);
    }
    for item in &semantic {
        all_chunk_ids.insert(item.chunk_id);
    }

    let mut fused = Vec::new();
    for chunk_id in all_chunk_ids {
        let mut fusion = 0.0_f32;
        if let Some(rank) = bm25_rank.get(&chunk_id) {
            fusion += 1.0 / (rrf_k as f32 + *rank as f32);
        }
        if let Some(rank) = dense_rank.get(&chunk_id) {
            fusion += 1.0 / (rrf_k as f32 + *rank as f32);
        }
        fused.push(RankedChunk {
            chunk_id,
            score: fusion,
            fusion,
            reranker: None,
            bm25: bm25_score.get(&chunk_id).copied(),
            dense: dense_score.get(&chunk_id).copied(),
            original_rank: None,
        });
    }

    finalize_ranked_chunks(fused, limit, min_score)
}

fn fuse_ranked_chunks_linear(
    keyword: Vec<RankedChunk>,
    semantic: Vec<RankedChunk>,
    dense_weight: f32,
    bm25_weight: f32,
    limit: usize,
    min_score: f32,
) -> Vec<RankedChunk> {
    let bm25_norm = normalize_scores_by_max(&keyword);
    let dense_norm = normalize_scores_by_max(&semantic);
    fuse_ranked_chunks_weighted(
        keyword,
        semantic,
        &bm25_norm,
        &dense_norm,
        dense_weight,
        bm25_weight,
        limit,
        min_score,
    )
}

fn fuse_ranked_chunks_dbsf(
    keyword: Vec<RankedChunk>,
    semantic: Vec<RankedChunk>,
    dense_weight: f32,
    bm25_weight: f32,
    stddevs: f32,
    limit: usize,
    min_score: f32,
) -> Vec<RankedChunk> {
    let bm25_norm = normalize_scores_by_dbsf(&keyword, stddevs);
    let dense_norm = normalize_scores_by_dbsf(&semantic, stddevs);
    fuse_ranked_chunks_weighted(
        keyword,
        semantic,
        &bm25_norm,
        &dense_norm,
        dense_weight,
        bm25_weight,
        limit,
        min_score,
    )
}

fn fuse_ranked_chunks_weighted(
    keyword: Vec<RankedChunk>,
    semantic: Vec<RankedChunk>,
    bm25_norm: &HashMap<i64, f32>,
    dense_norm: &HashMap<i64, f32>,
    dense_weight: f32,
    bm25_weight: f32,
    limit: usize,
    min_score: f32,
) -> Vec<RankedChunk> {
    let bm25_score = keyword
        .iter()
        .map(|item| (item.chunk_id, item.score))
        .collect::<HashMap<_, _>>();
    let dense_score = semantic
        .iter()
        .map(|item| (item.chunk_id, item.score))
        .collect::<HashMap<_, _>>();

    let mut all_chunk_ids = HashSet::new();
    for item in &keyword {
        all_chunk_ids.insert(item.chunk_id);
    }
    for item in &semantic {
        all_chunk_ids.insert(item.chunk_id);
    }

    let mut fused = Vec::new();
    for chunk_id in all_chunk_ids {
        let fusion = dense_weight * dense_norm.get(&chunk_id).copied().unwrap_or(0.0)
            + bm25_weight * bm25_norm.get(&chunk_id).copied().unwrap_or(0.0);

        fused.push(RankedChunk {
            chunk_id,
            score: fusion,
            fusion,
            reranker: None,
            bm25: bm25_score.get(&chunk_id).copied(),
            dense: dense_score.get(&chunk_id).copied(),
            original_rank: None,
        });
    }

    finalize_ranked_chunks(fused, limit, min_score)
}

fn aggregate_deep_variant_rankings(
    variant_results: Vec<Vec<RankedChunk>>,
    variant_rrf_k: usize,
    limit: usize,
    min_score: f32,
) -> Vec<RankedChunk> {
    let mut aggregates: HashMap<i64, RankedChunk> = HashMap::new();
    for (variant_index, ranked) in variant_results.into_iter().enumerate() {
        for (index, item) in ranked.into_iter().enumerate() {
            let variant_rrf = 1.0 / (variant_rrf_k as f32 + (index + 1) as f32);
            let entry = aggregates
                .entry(item.chunk_id)
                .or_insert_with(|| RankedChunk {
                    chunk_id: item.chunk_id,
                    score: 0.0,
                    fusion: 0.0,
                    reranker: None,
                    bm25: None,
                    dense: None,
                    original_rank: None,
                });
            entry.score += variant_rrf;
            entry.bm25 = max_option(entry.bm25, item.bm25);
            entry.dense = max_option(entry.dense, item.dense);
            if variant_index == 0 {
                entry.original_rank = Some(index + 1);
            }
        }
    }

    finalize_ranked_chunks(aggregates.into_values().collect(), limit, min_score)
}

fn finalize_ranked_chunks(
    mut ranked: Vec<RankedChunk>,
    limit: usize,
    min_score: f32,
) -> Vec<RankedChunk> {
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    normalize_ranked_chunks_scores(&mut ranked);
    ranked
        .into_iter()
        .filter(|item| item.score >= min_score)
        .take(limit)
        .collect()
}

fn select_deep_variant_indices(
    variants: &[String],
    variant_vectors: Option<&[Vec<f32>]>,
    max_selected_generated: usize,
) -> Vec<usize> {
    let mut selected = vec![0];
    if variants.len() <= 1 || max_selected_generated == 0 {
        return selected;
    }

    let generated_indices = (1..variants.len()).collect::<Vec<_>>();
    let selected_generated = match variant_vectors {
        Some(vectors) if vectors.len() == variants.len() && !vectors[0].is_empty() => {
            let original = &vectors[0];
            let mut ranked_generated = generated_indices
                .iter()
                .map(|&index| (index, cosine_similarity(original, &vectors[index])))
                .collect::<Vec<_>>();
            ranked_generated.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });

            let mut selected_generated: Vec<usize> = Vec::new();
            for (index, _) in &ranked_generated {
                let too_close = selected_generated.iter().any(|&selected_index| {
                    cosine_similarity(
                        vectors[*index].as_slice(),
                        vectors[selected_index].as_slice(),
                    ) >= DEEP_VARIANT_NEAR_DUPLICATE_SIMILARITY
                });
                if !too_close {
                    selected_generated.push(*index);
                    if selected_generated.len() >= max_selected_generated {
                        break;
                    }
                }
            }

            if selected_generated.len() < max_selected_generated {
                for (index, _) in ranked_generated {
                    if selected_generated.contains(&index) {
                        continue;
                    }
                    selected_generated.push(index);
                    if selected_generated.len() >= max_selected_generated {
                        break;
                    }
                }
            }

            selected_generated
        }
        _ => generated_indices
            .into_iter()
            .take(max_selected_generated)
            .collect::<Vec<_>>(),
    };

    selected.extend(selected_generated);
    selected
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        return 0.0;
    }

    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn protected_original_doc_count(rerank_count: usize) -> usize {
    rerank_count.div_ceil(2)
}

fn normalize_ranked_chunks_scores(ranked: &mut [RankedChunk]) {
    let max_score = ranked.iter().map(|item| item.score).fold(0.0_f32, f32::max);
    if max_score > 0.0 {
        for item in ranked {
            item.score /= max_score;
            item.fusion = item.score;
        }
    }
}

fn normalize_score_by_max(score: f32, max_score: f32) -> f32 {
    if max_score > 0.0 {
        score / max_score
    } else {
        0.0
    }
}

fn normalize_scores_by_max(ranked: &[RankedChunk]) -> HashMap<i64, f32> {
    let max_score = ranked.iter().map(|item| item.score).fold(0.0_f32, f32::max);
    ranked
        .iter()
        .map(|item| (item.chunk_id, normalize_score_by_max(item.score, max_score)))
        .collect()
}

fn normalize_scores_by_dbsf(ranked: &[RankedChunk], stddevs: f32) -> HashMap<i64, f32> {
    if ranked.is_empty() {
        return HashMap::new();
    }

    let scores = ranked.iter().map(|item| item.score).collect::<Vec<_>>();
    let count = scores.len() as f32;
    let mean = scores.iter().sum::<f32>() / count;
    let variance = scores
        .iter()
        .map(|score| {
            let delta = *score - mean;
            delta * delta
        })
        .sum::<f32>()
        / count;
    let stddev = variance.sqrt();
    if !stddev.is_finite() || stddev <= f32::EPSILON {
        return normalize_scores_by_max(ranked);
    }

    let low = mean - stddevs * stddev;
    let high = mean + stddevs * stddev;
    if !low.is_finite() || !high.is_finite() || high <= low {
        return normalize_scores_by_max(ranked);
    }

    ranked
        .iter()
        .map(|item| {
            (
                item.chunk_id,
                ((item.score - low) / (high - low)).clamp(0.0, 1.0),
            )
        })
        .collect()
}

fn rerank_candidate_count(
    requested_limit: usize,
    candidate_count: usize,
    min_candidates: usize,
    max_candidates: usize,
) -> usize {
    let target = requested_limit.max(min_candidates).min(max_candidates);
    candidate_count.min(target)
}

fn add_search_pipeline_notice(
    pipeline: &mut SearchPipeline,
    step: SearchPipelineStep,
    reason: SearchPipelineUnavailableReason,
) {
    if pipeline
        .notices
        .iter()
        .any(|notice| notice.step == step && notice.reason == reason)
    {
        return;
    }

    pipeline.notices.push(SearchPipelineNotice { step, reason });
}

fn is_model_not_available_error(err: &CoreError) -> bool {
    matches!(err, CoreError::Domain(KboltError::ModelNotAvailable { .. }))
}
