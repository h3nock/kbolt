use super::*;

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

    pub(super) fn max_search_candidates(&self, targets: &[UpdateTarget]) -> Result<usize> {
        let mut total = 0usize;
        for target in targets {
            total = total.saturating_add(
                self.storage
                    .count_chunks_in_collection(target.collection.id)?,
            );
        }
        Ok(total)
    }

    pub(super) fn rank_chunks_for_mode(
        &self,
        mode: &SearchMode,
        targets: &[UpdateTarget],
        query: &str,
        limit: usize,
        min_score: f32,
        pipeline: &mut SearchPipeline,
    ) -> Result<Vec<RankedChunk>> {
        match mode {
            SearchMode::Keyword => self.rank_keyword_chunks(targets, query, limit, min_score),
            SearchMode::Auto => self.rank_auto_chunks(targets, query, limit, min_score, pipeline),
            SearchMode::Semantic => self.rank_semantic_chunks(targets, query, limit, min_score),
            SearchMode::Deep => self.rank_deep_chunks(targets, query, limit, min_score, pipeline),
        }
    }

    pub(super) fn rank_keyword_chunks(
        &self,
        targets: &[UpdateTarget],
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<RankedChunk>> {
        let boosts = &self.config.ranking.bm25_boosts;
        let mut candidates = Vec::new();
        for target in targets {
            let hits = self.storage.query_bm25(
                &target.space,
                query,
                &[
                    ("title", boosts.title),
                    ("heading", boosts.heading),
                    ("body", boosts.body),
                    ("filepath", boosts.filepath),
                ],
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
                rrf: normalized_score,
                reranker: None,
                bm25: Some(normalized_score),
                dense: None,
            });
            if ranked.len() >= limit {
                break;
            }
        }

        Ok(ranked)
    }

    pub(super) fn rank_semantic_chunks(
        &self,
        targets: &[UpdateTarget],
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<RankedChunk>> {
        let Some(embedder) = self.embedder.as_ref() else {
            return Err(KboltError::InvalidInput(
                "semantic search requires embeddings configuration. add [embeddings] to index.toml with provider = \"openai_compatible\", \"voyage\", \"local_onnx\", or \"local_gguf\"".to_string(),
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

        self.rank_semantic_chunks_with_embedding(targets, &vectors[0], limit, min_score)
    }

    fn rank_semantic_chunks_with_embedding(
        &self,
        targets: &[UpdateTarget],
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
        for target in targets {
            let hits = self
                .storage
                .query_dense(&target.space, query_vector, limit)?;
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
                rrf: dense_score,
                reranker: None,
                bm25: None,
                dense: Some(dense_score),
            });
            if ranked.len() >= limit {
                break;
            }
        }

        Ok(ranked)
    }

    pub(super) fn rank_auto_chunks(
        &self,
        targets: &[UpdateTarget],
        query: &str,
        limit: usize,
        min_score: f32,
        pipeline: &mut SearchPipeline,
    ) -> Result<Vec<RankedChunk>> {
        let keyword = if self.embedder.is_some() {
            let (keyword_result, semantic_result) = std::thread::scope(|scope| {
                let keyword_handle =
                    scope.spawn(|| self.rank_keyword_chunks(targets, query, limit, 0.0));
                let semantic_handle =
                    scope.spawn(|| self.rank_semantic_chunks(targets, query, limit, 0.0));
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
                self.config.ranking.rrf_k,
                limit,
                min_score,
            ));
        } else {
            self.rank_keyword_chunks(targets, query, limit, min_score)?
        };

        Ok(keyword)
    }

    pub(super) fn rank_deep_chunks(
        &self,
        targets: &[UpdateTarget],
        query: &str,
        limit: usize,
        min_score: f32,
        pipeline: &mut SearchPipeline,
    ) -> Result<Vec<RankedChunk>> {
        let Some(expander) = self.expander.as_ref() else {
            return Err(KboltError::InvalidInput(
                "deep search requires expander configuration. add [inference.expander] to index.toml".to_string(),
            )
            .into());
        };

        let normalized_query = crate::models::normalize_query_text(query);
        let mut variants = vec![crate::models::ExpandedQuery {
            text: normalized_query,
            route: crate::models::ExpansionRoute::Both,
        }];
        let max_generated = self.config.ranking.deep_variants_max.saturating_sub(1);
        if max_generated > 0 {
            let generated = expander.expand(query)?;
            let mut variant_index_by_key =
                HashMap::from([(variants[0].text.to_ascii_lowercase(), 0usize)]);
            for generated_query in generated {
                let text = crate::models::normalize_query_text(&generated_query.text);
                if text.is_empty() {
                    return Err(KboltError::Inference(
                        "expander returned an empty deep-search query".to_string(),
                    )
                    .into());
                }

                let key = text.to_ascii_lowercase();
                if let Some(existing_index) = variant_index_by_key.get(&key).copied() {
                    variants[existing_index].route = variants[existing_index]
                        .route
                        .merged_with(generated_query.route);
                    continue;
                }

                variant_index_by_key.insert(key, variants.len());
                variants.push(crate::models::ExpandedQuery {
                    text,
                    route: generated_query.route,
                });
                if variants.len() >= self.config.ranking.deep_variants_max {
                    break;
                }
            }
        }
        pipeline.expansion = true;
        let dense_variant_indices = variants
            .iter()
            .enumerate()
            .filter_map(|(index, variant)| variant.route.includes_dense().then_some(index))
            .collect::<Vec<_>>();
        let dense_variant_texts = dense_variant_indices
            .iter()
            .map(|&index| variants[index].text.clone())
            .collect::<Vec<_>>();
        let variant_vectors = if dense_variant_texts.is_empty() {
            None
        } else if let Some(embedder) = self.embedder.as_ref() {
            match embedder.embed_batch(
                crate::models::EmbeddingInputKind::Query,
                &dense_variant_texts,
            ) {
                Ok(vectors) => {
                    if vectors.len() != dense_variant_texts.len() {
                        return Err(KboltError::Inference(format!(
                            "embedder returned {} vectors for {} deep variants",
                            vectors.len(),
                            dense_variant_texts.len()
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
                    let mut vectors_by_variant = vec![None; variants.len()];
                    for (dense_index, vector) in vectors.into_iter().enumerate() {
                        let variant_index = dense_variant_indices[dense_index];
                        vectors_by_variant[variant_index] = Some(vector);
                    }
                    Some(vectors_by_variant)
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
        let variant_results: Vec<Result<Vec<RankedChunk>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = variants
                .iter()
                .enumerate()
                .map(|(index, variant)| {
                    let vv = &variant_vectors;
                    scope.spawn(move || {
                        let keyword = if variant.route.includes_keyword() {
                            self.rank_keyword_chunks(targets, &variant.text, limit, 0.0)?
                        } else {
                            Vec::new()
                        };
                        let semantic = if variant.route.includes_dense() {
                            vv.as_ref()
                                .and_then(|vectors| vectors.get(index))
                                .and_then(|vector| vector.as_ref())
                                .map(|vector| {
                                    self.rank_semantic_chunks_with_embedding(
                                        targets, vector, limit, 0.0,
                                    )
                                })
                                .transpose()?
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        };
                        Ok(fuse_ranked_chunks(
                            keyword,
                            semantic,
                            self.config.ranking.rrf_k,
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

        let mut aggregates: HashMap<i64, RankedChunk> = HashMap::new();
        for ranked in variant_results {
            for (index, item) in ranked?.into_iter().enumerate() {
                let variant_rrf = 1.0 / (self.config.ranking.rrf_k as f32 + (index + 1) as f32);
                let entry = aggregates
                    .entry(item.chunk_id)
                    .or_insert_with(|| RankedChunk {
                        chunk_id: item.chunk_id,
                        score: 0.0,
                        rrf: 0.0,
                        reranker: None,
                        bm25: None,
                        dense: None,
                    });
                entry.score += variant_rrf;
                entry.bm25 = max_option(entry.bm25, item.bm25);
                entry.dense = max_option(entry.dense, item.dense);
            }
        }

        let mut fused = aggregates.into_values().collect::<Vec<_>>();
        fused.sort_by(|left, right| right.score.total_cmp(&left.score));

        let max_score = fused.iter().map(|item| item.score).fold(0.0_f32, f32::max);
        if max_score > 0.0 {
            for item in &mut fused {
                item.score /= max_score;
                item.rrf = item.score;
            }
        }

        Ok(fused
            .into_iter()
            .filter(|item| item.score >= min_score)
            .take(limit)
            .collect())
    }

    pub(super) fn assemble_search_results(
        &self,
        query: &str,
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
                chunk_id: ranked.chunk_id,
                doc_id: chunk.doc_id,
                docid: short_docid(&document.hash),
                path: format!("{}/{}", collection.collection, document.path),
                title: document.title.clone(),
                space: collection.space.clone(),
                collection: collection.collection.clone(),
                heading: chunk.heading.clone(),
                chunk: chunk.clone(),
                full_path: collection.path.join(&document.path),
                bm25: ranked.bm25,
                dense: ranked.dense,
                rrf: ranked.rrf,
                reranker: ranked.reranker,
                final_score: ranked.score,
            });
        }

        let mut bytes_by_doc: HashMap<i64, Vec<u8>> = HashMap::new();
        let neighbor_window = self.config.chunking.defaults.neighbor_window;
        let contextual_prefix = self.config.chunking.defaults.contextual_prefix;

        let mut chunks_by_doc: HashMap<i64, Vec<ChunkRow>> = HashMap::new();

        if apply_rerank && !candidates.is_empty() {
            let rerank_count = rerank_candidate_count(
                limit,
                candidates.len(),
                self.config.ranking.rerank_candidates_min,
                self.config.ranking.rerank_candidates_max,
            );
            let representative_indices = select_rerank_representatives(&candidates, rerank_count);
            let mut rerank_doc_ids = Vec::new();
            let mut rerank_inputs = Vec::new();
            let mut invalid_chunk_ids = HashSet::new();
            for &idx in &representative_indices {
                let candidate = &candidates[idx];
                match build_rerank_input(
                    candidate,
                    &self.storage,
                    &mut bytes_by_doc,
                    &mut chunks_by_doc,
                    neighbor_window,
                    contextual_prefix,
                )? {
                    Some(rerank_input) => {
                        rerank_doc_ids.push(candidate.doc_id);
                        rerank_inputs.push(rerank_input);
                    }
                    None => {
                        invalid_chunk_ids.insert(candidate.chunk_id);
                    }
                }
            }
            if !invalid_chunk_ids.is_empty() {
                candidates.retain(|candidate| !invalid_chunk_ids.contains(&candidate.chunk_id));
            }
            let raw_scores = match self.reranker.as_ref() {
                Some(reranker) => match reranker.rerank(query, &rerank_inputs) {
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
                },
                None => {
                    pipeline.rerank = false;
                    add_search_pipeline_notice(
                        pipeline,
                        SearchPipelineStep::Rerank,
                        SearchPipelineUnavailableReason::NotConfigured,
                    );
                    None
                }
            };
            let Some(raw_scores) = raw_scores else {
                return finalize_search_results(
                    candidates,
                    &self.storage,
                    &mut bytes_by_doc,
                    &mut chunks_by_doc,
                    neighbor_window,
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
            &mut bytes_by_doc,
            &mut chunks_by_doc,
            neighbor_window,
            debug,
            limit,
        )
    }
}

#[derive(Debug, Clone)]
struct PendingSearchCandidate {
    chunk_id: i64,
    doc_id: i64,
    docid: String,
    path: String,
    title: String,
    space: String,
    collection: String,
    heading: Option<String>,
    chunk: ChunkRow,
    full_path: std::path::PathBuf,
    bm25: Option<f32>,
    dense: Option<f32>,
    rrf: f32,
    reranker: Option<f32>,
    final_score: f32,
}

fn build_rerank_input(
    candidate: &PendingSearchCandidate,
    storage: &Storage,
    bytes_by_doc: &mut HashMap<i64, Vec<u8>>,
    chunks_by_doc: &mut HashMap<i64, Vec<ChunkRow>>,
    neighbor_window: usize,
    contextual_prefix: bool,
) -> Result<Option<String>> {
    let Some(bytes) = load_candidate_bytes(candidate, bytes_by_doc)? else {
        return Ok(None);
    };
    let primary_text = chunk_text_from_bytes(bytes, candidate.chunk.offset, candidate.chunk.length);
    let rerank_body = if primary_text.trim().is_empty() {
        search_text_with_neighbors(
            bytes,
            &candidate.chunk,
            candidate_neighbors(storage, candidate, chunks_by_doc, neighbor_window)?,
            neighbor_window,
        )
    } else {
        primary_text
    };

    Ok(Some(retrieval_text_with_prefix(
        rerank_body.as_str(),
        candidate.title.as_str(),
        candidate.heading.as_deref(),
        contextual_prefix,
    )))
}

/// Selects one representative candidate per unique document for reranking.
/// Candidates are already sorted by RRF score, so the first candidate for
/// each document is its highest-scoring chunk (MaxP strategy).
fn select_rerank_representatives(
    candidates: &[PendingSearchCandidate],
    max_docs: usize,
) -> Vec<usize> {
    let mut seen_docs = HashSet::new();
    let mut indices = Vec::new();
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

/// Applies per-document reranker scores to all candidates. Every chunk from
/// a reranked document inherits that document's reranker score. Chunks from
/// non-reranked documents get a fallback score below all reranked scores.
fn apply_reranker_scores(
    candidates: &mut [PendingSearchCandidate],
    doc_reranker_scores: &HashMap<i64, f32>,
) {
    if doc_reranker_scores.is_empty() {
        return;
    }

    let mut fallback_score = doc_reranker_scores
        .values()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .next_down();

    for candidate in candidates {
        if let Some(reranker_score) = doc_reranker_scores.get(&candidate.doc_id).copied() {
            candidate.reranker = Some(reranker_score);
            candidate.final_score = reranker_score;
        } else {
            candidate.final_score = fallback_score;
            fallback_score = fallback_score.next_down();
        }
    }
}

fn finalize_search_results(
    mut candidates: Vec<PendingSearchCandidate>,
    storage: &Storage,
    bytes_by_doc: &mut HashMap<i64, Vec<u8>>,
    chunks_by_doc: &mut HashMap<i64, Vec<ChunkRow>>,
    neighbor_window: usize,
    debug: bool,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    candidates.sort_by(|left, right| right.final_score.total_cmp(&left.final_score));

    let mut results = Vec::new();
    for candidate in candidates {
        if results.len() >= limit {
            break;
        }

        let Some(bytes) = load_candidate_bytes(&candidate, bytes_by_doc)? else {
            continue;
        };
        let text = search_text_with_neighbors(
            bytes,
            &candidate.chunk,
            candidate_neighbors(storage, &candidate, chunks_by_doc, neighbor_window)?,
            neighbor_window,
        );

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
                    rrf: candidate.rrf,
                    reranker: candidate.reranker,
                })
            } else {
                None
            },
        });
    }

    Ok(results)
}

fn candidate_neighbors<'a>(
    storage: &Storage,
    candidate: &PendingSearchCandidate,
    chunks_by_doc: &'a mut HashMap<i64, Vec<ChunkRow>>,
    neighbor_window: usize,
) -> Result<Option<&'a Vec<ChunkRow>>> {
    if neighbor_window == 0 {
        return Ok(None);
    }

    if !chunks_by_doc.contains_key(&candidate.doc_id) {
        chunks_by_doc.insert(
            candidate.doc_id,
            storage.get_chunks_for_document(candidate.doc_id)?,
        );
    }

    Ok(chunks_by_doc.get(&candidate.doc_id))
}

fn load_candidate_bytes<'a>(
    candidate: &PendingSearchCandidate,
    bytes_by_doc: &'a mut HashMap<i64, Vec<u8>>,
) -> Result<Option<&'a [u8]>> {
    if !bytes_by_doc.contains_key(&candidate.doc_id) {
        match std::fs::read(&candidate.full_path) {
            Ok(bytes) => {
                bytes_by_doc.insert(candidate.doc_id, bytes);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err.into()),
        }
    }

    Ok(bytes_by_doc.get(&candidate.doc_id).map(Vec::as_slice))
}

fn fuse_ranked_chunks(
    keyword: Vec<RankedChunk>,
    semantic: Vec<RankedChunk>,
    rrf_k: usize,
    limit: usize,
    min_score: f32,
) -> Vec<RankedChunk> {
    if semantic.is_empty() {
        return keyword
            .into_iter()
            .filter(|item| item.score >= min_score)
            .take(limit)
            .collect();
    }

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
        let mut rrf = 0.0_f32;
        if let Some(rank) = bm25_rank.get(&chunk_id) {
            rrf += 1.0 / (rrf_k as f32 + *rank as f32);
        }
        if let Some(rank) = dense_rank.get(&chunk_id) {
            rrf += 1.0 / (rrf_k as f32 + *rank as f32);
        }
        fused.push(RankedChunk {
            chunk_id,
            score: rrf,
            rrf,
            reranker: None,
            bm25: bm25_score.get(&chunk_id).copied(),
            dense: dense_score.get(&chunk_id).copied(),
        });
    }
    fused.sort_by(|left, right| right.score.total_cmp(&left.score));

    let max_score = fused.iter().map(|item| item.score).fold(0.0_f32, f32::max);
    if max_score > 0.0 {
        for item in &mut fused {
            item.score /= max_score;
            item.rrf = item.score;
        }
    }

    fused
        .into_iter()
        .filter(|item| item.score >= min_score)
        .take(limit)
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
