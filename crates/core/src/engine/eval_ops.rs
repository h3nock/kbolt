use kbolt_types::{
    EvalCase, EvalJudgment, EvalModeFailure, EvalModeReport, EvalQueryReport, EvalRunReport,
    SearchMode, SearchRequest, SearchResult,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::eval_store::load_eval_dataset_with_file;

use super::*;

impl Engine {
    pub fn run_eval(&self, eval_file: Option<&Path>) -> Result<EvalRunReport> {
        let dataset = load_eval_dataset_with_file(&self.config.config_dir, eval_file)?;
        self.validate_eval_cases(&dataset.cases)?;
        let eval_runs = self.eval_runs();
        let total_cases = dataset.cases.len();
        let mut reports = Vec::with_capacity(eval_runs.len());
        let mut failed_modes = Vec::new();

        for (mode, no_rerank) in eval_runs {
            match self.run_eval_mode(&dataset.cases, mode.clone(), no_rerank) {
                Ok(report) => reports.push(report),
                Err(err) => failed_modes.push(EvalModeFailure {
                    mode,
                    no_rerank,
                    error: err.to_string(),
                }),
            }
        }

        Ok(EvalRunReport {
            total_cases,
            modes: reports,
            failed_modes,
        })
    }

    fn validate_eval_cases(&self, cases: &[EvalCase]) -> Result<()> {
        let mut seen_spaces = HashSet::new();
        let mut seen_collections = HashSet::new();

        for case in cases {
            if let Some(space_name) = case.space.as_deref() {
                if seen_spaces.insert(space_name.to_string()) {
                    self.resolve_space_row(Some(space_name), None)?;
                }
            }

            for collection in &case.collections {
                let key = (case.space.clone(), collection.clone());
                if seen_collections.insert(key) {
                    self.validate_eval_collection(case.space.as_deref(), collection)?;
                }
            }
        }

        Ok(())
    }

    fn validate_eval_collection(&self, space: Option<&str>, collection: &str) -> Result<()> {
        let resolved_space = self.resolve_space_row(space, Some(collection))?;
        let collection_row = self.storage.get_collection(resolved_space.id, collection)?;
        let chunk_count = self.storage.count_chunks_in_collection(collection_row.id)?;
        if chunk_count == 0 {
            return Err(KboltError::InvalidInput(format!(
                "eval collection '{collection}' in space '{}' has no indexed chunks; run `kbolt --space {} update --collection {collection}`",
                resolved_space.name, resolved_space.name
            ))
            .into());
        }

        Ok(())
    }

    fn eval_runs(&self) -> Vec<(SearchMode, bool)> {
        let mut runs = vec![
            (SearchMode::Keyword, true),
            (SearchMode::Auto, true),
            (SearchMode::Auto, false),
        ];
        if self.embedder.is_some() {
            runs.push((SearchMode::Semantic, true));
        }
        runs.push((SearchMode::Deep, true));
        runs.push((SearchMode::Deep, false));
        runs
    }

    fn run_eval_mode(
        &self,
        cases: &[EvalCase],
        mode: SearchMode,
        no_rerank: bool,
    ) -> Result<EvalModeReport> {
        let mut query_reports = Vec::with_capacity(cases.len());
        let mut ndcg_sum = 0.0_f32;
        let mut recall_sum = 0.0_f32;
        let mut mrr_sum = 0.0_f32;
        let mut latencies = Vec::with_capacity(cases.len());

        for case in cases {
            let response = self.search(SearchRequest {
                query: case.query.clone(),
                mode: mode.clone(),
                space: case.space.clone(),
                collections: case.collections.clone(),
                limit: 10,
                min_score: 0.0,
                no_rerank,
                debug: false,
            })?;

            let returned_paths = dedupe_result_paths(case, &response.results);
            let judgment_map = judgment_map(&case.judgments);
            let relevant_path_count = case
                .judgments
                .iter()
                .filter(|judgment| judgment.relevance > 0)
                .count();
            let matched_paths = returned_paths
                .iter()
                .filter(|path| relevance_for_path(&judgment_map, path.as_str()) > 0)
                .cloned()
                .collect::<Vec<_>>();
            let first_relevant_rank = returned_paths
                .iter()
                .position(|path| relevance_for_path(&judgment_map, path.as_str()) > 0)
                .map(|index| index + 1);
            let matched_top_10 = returned_paths
                .iter()
                .take(10)
                .filter(|path| relevance_for_path(&judgment_map, path.as_str()) > 0)
                .count();
            ndcg_sum += ndcg_at_k(&returned_paths, &judgment_map, 10);
            recall_sum += matched_top_10 as f32 / relevant_path_count as f32;
            mrr_sum += first_relevant_rank
                .map(|rank| 1.0_f32 / rank as f32)
                .unwrap_or(0.0);
            latencies.push(response.elapsed_ms);
            query_reports.push(EvalQueryReport {
                query: case.query.clone(),
                space: case.space.clone(),
                collections: case.collections.clone(),
                judgments: case.judgments.clone(),
                returned_paths,
                matched_paths,
                first_relevant_rank,
                elapsed_ms: response.elapsed_ms,
            });
        }

        let case_count = cases.len() as f32;
        Ok(EvalModeReport {
            mode,
            no_rerank,
            ndcg_at_10: ndcg_sum / case_count,
            recall_at_10: recall_sum / case_count,
            mrr_at_10: mrr_sum / case_count,
            latency_p50_ms: percentile_ms(&latencies, 0.50),
            latency_p95_ms: percentile_ms(&latencies, 0.95),
            queries: query_reports,
        })
    }
}

fn dedupe_result_paths(case: &EvalCase, results: &[SearchResult]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();

    for result in results {
        let path = if case.space.is_some() {
            result.path.clone()
        } else {
            format!("{}/{}", result.space, result.path)
        };

        if seen.insert(path.clone()) {
            deduped.push(path);
        }
    }

    deduped
}

fn judgment_map<'a>(judgments: &'a [EvalJudgment]) -> HashMap<&'a str, u8> {
    judgments
        .iter()
        .map(|judgment| (judgment.path.as_str(), judgment.relevance))
        .collect()
}

fn relevance_for_path(judgments: &HashMap<&str, u8>, path: &str) -> u8 {
    judgments.get(path).copied().unwrap_or(0)
}

fn ndcg_at_k(returned_paths: &[String], judgments: &HashMap<&str, u8>, k: usize) -> f32 {
    let dcg = dcg_at_k(
        &returned_paths
            .iter()
            .take(k)
            .map(|path| relevance_for_path(judgments, path.as_str()))
            .collect::<Vec<_>>(),
    );
    let mut ideal_relevances = judgments.values().copied().collect::<Vec<_>>();
    ideal_relevances.sort_unstable_by(|left, right| right.cmp(left));
    let ideal_dcg = dcg_at_k(&ideal_relevances.into_iter().take(k).collect::<Vec<_>>());
    if ideal_dcg == 0.0 {
        0.0
    } else {
        dcg / ideal_dcg
    }
}

fn dcg_at_k(relevances: &[u8]) -> f32 {
    relevances
        .iter()
        .enumerate()
        .map(|(index, relevance)| {
            let gain = 2_f32.powi(i32::from(*relevance)) - 1.0;
            let discount = (index as f32 + 2.0).log2();
            gain / discount
        })
        .sum()
}

fn percentile_ms(samples: &[u64], percentile: f32) -> u64 {
    if samples.is_empty() {
        return 0;
    }

    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() as f32 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;

    use tempfile::tempdir;

    use crate::config::{
        ChunkingConfig, Config, EmbeddingConfig, InferenceConfig, ModelConfig, ModelProvider,
        ModelSourceConfig, RankingConfig, ReapingConfig,
    };
    use crate::models::{Embedder, Expander};
    use crate::storage::Storage;
    use kbolt_types::{AddCollectionRequest, SearchMode, UpdateOptions};

    use super::*;

    #[derive(Default)]
    struct DeterministicEmbedder;

    impl Embedder for DeterministicEmbedder {
        fn embed_batch(
            &self,
            _kind: crate::models::EmbeddingInputKind,
            texts: &[String],
        ) -> crate::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    let token_count = text.split_whitespace().count().max(1) as f32;
                    let byte_count = text.len().max(1) as f32;
                    vec![token_count, byte_count]
                })
                .collect())
        }
    }

    #[derive(Default)]
    struct DeterministicExpander;

    impl Expander for DeterministicExpander {
        fn expand(&self, query: &str) -> crate::Result<Vec<crate::models::ExpandedQuery>> {
            Ok(vec![crate::models::ExpandedQuery {
                text: format!("explain {query}"),
                route: crate::models::ExpansionRoute::Both,
            }])
        }
    }

    struct FailingExpander;

    impl Expander for FailingExpander {
        fn expand(&self, _query: &str) -> crate::Result<Vec<crate::models::ExpandedQuery>> {
            Err(KboltError::Inference("expander unavailable".to_string()).into())
        }
    }

    #[test]
    fn run_eval_reports_keyword_auto_and_deep_without_embedder() {
        let root = tempdir().expect("create temp root");
        let collection_dir = seed_collection(root.path(), "rust", "guides/traits.md", TRAITS_DOC);
        let engine = test_engine(None, Some(Arc::new(DeterministicExpander)));
        seed_eval_file(
            &engine,
            r#"
[[cases]]
query = "trait object generic"
space = "default"
collections = ["rust"]
judgments = [{ path = "rust/guides/traits.md", relevance = 1 }]
"#,
        );

        engine
            .add_collection(AddCollectionRequest {
                path: collection_dir,
                space: Some("default".to_string()),
                name: Some("rust".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add collection");
        engine
            .update(UpdateOptions {
                space: Some("default".to_string()),
                collections: vec!["rust".to_string()],
                no_embed: true,
                dry_run: false,
                verbose: false,
            })
            .expect("update collection");

        let report = engine.run_eval(None).expect("run eval");

        assert_eq!(report.total_cases, 1);
        let mode_labels: Vec<_> = report
            .modes
            .iter()
            .map(|m| (m.mode.clone(), m.no_rerank))
            .collect();
        assert_eq!(
            mode_labels,
            vec![
                (SearchMode::Keyword, true),
                (SearchMode::Auto, true),
                (SearchMode::Auto, false),
                (SearchMode::Deep, true),
                (SearchMode::Deep, false),
            ]
        );
        for mode in &report.modes {
            assert_eq!(mode.queries.len(), 1);
            assert_eq!(mode.queries[0].first_relevant_rank, Some(1));
            assert_eq!(mode.ndcg_at_10, 1.0);
            assert_eq!(mode.recall_at_10, 1.0);
            assert_eq!(mode.mrr_at_10, 1.0);
        }
    }

    #[test]
    fn run_eval_includes_semantic_mode_when_embedder_is_available() {
        let root = tempdir().expect("create temp root");
        let collection_dir = seed_collection(root.path(), "rust", "guides/traits.md", TRAITS_DOC);
        let engine = test_engine(
            Some(Arc::new(DeterministicEmbedder)),
            Some(Arc::new(DeterministicExpander)),
        );
        seed_eval_file(
            &engine,
            r#"
[[cases]]
query = "trait object generic"
space = "default"
collections = ["rust"]
judgments = [{ path = "rust/guides/traits.md", relevance = 1 }]
"#,
        );

        engine
            .add_collection(AddCollectionRequest {
                path: collection_dir,
                space: Some("default".to_string()),
                name: Some("rust".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add collection");
        engine
            .update(UpdateOptions {
                space: Some("default".to_string()),
                collections: vec!["rust".to_string()],
                no_embed: false,
                dry_run: false,
                verbose: false,
            })
            .expect("update collection");

        let report = engine.run_eval(None).expect("run eval");

        assert!(report
            .modes
            .iter()
            .any(|mode| mode.mode == SearchMode::Semantic));
        let semantic = report
            .modes
            .iter()
            .find(|mode| mode.mode == SearchMode::Semantic)
            .expect("semantic report");
        assert_eq!(semantic.queries[0].first_relevant_rank, Some(1));
        assert_eq!(semantic.ndcg_at_10, 1.0);
        assert_eq!(semantic.recall_at_10, 1.0);
        assert_eq!(semantic.mrr_at_10, 1.0);
    }

    #[test]
    fn run_eval_keeps_successful_modes_when_later_mode_fails() {
        let root = tempdir().expect("create temp root");
        let collection_dir = seed_collection(root.path(), "rust", "guides/traits.md", TRAITS_DOC);
        let engine = test_engine(None, Some(Arc::new(FailingExpander)));
        seed_eval_file(
            &engine,
            r#"
[[cases]]
query = "trait object generic"
space = "default"
collections = ["rust"]
judgments = [{ path = "rust/guides/traits.md", relevance = 1 }]
"#,
        );

        engine
            .add_collection(AddCollectionRequest {
                path: collection_dir,
                space: Some("default".to_string()),
                name: Some("rust".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add collection");
        engine
            .update(UpdateOptions {
                space: Some("default".to_string()),
                collections: vec!["rust".to_string()],
                no_embed: true,
                dry_run: false,
                verbose: false,
            })
            .expect("update collection");

        let report = engine.run_eval(None).expect("run eval");

        assert!(report
            .modes
            .iter()
            .any(|mode| mode.mode == SearchMode::Keyword));
        assert!(report.failed_modes.iter().any(
            |mode| mode.mode == SearchMode::Deep && mode.error.contains("expander unavailable")
        ));
    }

    #[test]
    fn ndcg_at_10_is_zero_for_irrelevant_results() {
        let cases = [
            EvalJudgment {
                path: "rust/a.md".to_string(),
                relevance: 2,
            },
            EvalJudgment {
                path: "rust/b.md".to_string(),
                relevance: 1,
            },
        ];
        let judgments = judgment_map(&cases);

        let score = ndcg_at_k(&["rust/c.md".to_string()], &judgments, 10);

        assert_eq!(score, 0.0);
    }

    #[test]
    fn ndcg_at_10_is_one_for_perfect_ranking() {
        let cases = [
            EvalJudgment {
                path: "rust/a.md".to_string(),
                relevance: 2,
            },
            EvalJudgment {
                path: "rust/b.md".to_string(),
                relevance: 1,
            },
        ];
        let judgments = judgment_map(&cases);

        let score = ndcg_at_k(
            &["rust/a.md".to_string(), "rust/b.md".to_string()],
            &judgments,
            10,
        );

        assert!((score - 1.0).abs() < 1e-6, "unexpected score: {score}");
    }

    #[test]
    fn ndcg_at_10_uses_graded_relevance_ordering() {
        let cases = [
            EvalJudgment {
                path: "rust/a.md".to_string(),
                relevance: 2,
            },
            EvalJudgment {
                path: "rust/b.md".to_string(),
                relevance: 1,
            },
        ];
        let judgments = judgment_map(&cases);

        let perfect = ndcg_at_k(
            &["rust/a.md".to_string(), "rust/b.md".to_string()],
            &judgments,
            10,
        );
        let swapped = ndcg_at_k(
            &["rust/b.md".to_string(), "rust/a.md".to_string()],
            &judgments,
            10,
        );

        assert!(perfect > swapped, "perfect={perfect}, swapped={swapped}");
    }

    #[test]
    fn ndcg_at_10_handles_fewer_results_than_k() {
        let cases = [
            EvalJudgment {
                path: "rust/a.md".to_string(),
                relevance: 2,
            },
            EvalJudgment {
                path: "rust/b.md".to_string(),
                relevance: 1,
            },
            EvalJudgment {
                path: "rust/c.md".to_string(),
                relevance: 1,
            },
        ];
        let judgments = judgment_map(&cases);

        let score = ndcg_at_k(
            &["rust/a.md".to_string(), "rust/b.md".to_string()],
            &judgments,
            10,
        );

        assert!(score > 0.0 && score < 1.0, "unexpected score: {score}");
    }

    #[test]
    fn run_eval_supports_explicit_manifest_path() {
        let root = tempdir().expect("create temp root");
        let collection_dir = seed_collection(root.path(), "rust", "guides/traits.md", TRAITS_DOC);
        let engine = test_engine(None, Some(Arc::new(DeterministicExpander)));
        let eval_file = root.path().join("bench").join("scifact.toml");
        if let Some(parent) = eval_file.parent() {
            fs::create_dir_all(parent).expect("create bench dir");
        }
        fs::write(
            &eval_file,
            r#"
[[cases]]
query = "trait object generic"
space = "default"
collections = ["rust"]
judgments = [{ path = "rust/guides/traits.md", relevance = 1 }]
"#,
        )
        .expect("write eval file");

        engine
            .add_collection(AddCollectionRequest {
                path: collection_dir,
                space: Some("default".to_string()),
                name: Some("rust".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add collection");
        engine
            .update(UpdateOptions {
                space: Some("default".to_string()),
                collections: vec!["rust".to_string()],
                no_embed: true,
                dry_run: false,
                verbose: false,
            })
            .expect("update collection");

        let report = engine.run_eval(Some(&eval_file)).expect("run eval");

        assert_eq!(report.total_cases, 1);
        assert!(report.modes.iter().all(|mode| mode.recall_at_10 >= 0.0));
    }

    #[test]
    fn run_eval_fails_when_manifest_references_missing_collection() {
        let engine = test_engine(None, Some(Arc::new(DeterministicExpander)));
        seed_eval_file(
            &engine,
            r#"
[[cases]]
query = "trait object generic"
space = "default"
collections = ["rust"]
judgments = [{ path = "rust/guides/traits.md", relevance = 1 }]
"#,
        );

        let err = engine
            .run_eval(None)
            .expect_err("missing collection should fail");
        assert!(
            err.to_string().contains("collection not found: rust"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn run_eval_fails_when_collection_has_not_been_indexed() {
        let root = tempdir().expect("create temp root");
        let collection_dir = seed_collection(root.path(), "rust", "guides/traits.md", TRAITS_DOC);
        let engine = test_engine(None, Some(Arc::new(DeterministicExpander)));
        seed_eval_file(
            &engine,
            r#"
[[cases]]
query = "trait object generic"
space = "default"
collections = ["rust"]
judgments = [{ path = "rust/guides/traits.md", relevance = 1 }]
"#,
        );

        engine
            .add_collection(AddCollectionRequest {
                path: collection_dir,
                space: Some("default".to_string()),
                name: Some("rust".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add collection");

        let err = engine
            .run_eval(None)
            .expect_err("unindexed collection should fail");
        assert!(
            err.to_string().contains("has no indexed chunks"),
            "unexpected error: {err}"
        );
        assert!(
            err.to_string()
                .contains("kbolt --space default update --collection rust"),
            "unexpected error: {err}"
        );
    }

    fn test_engine(
        embedder: Option<Arc<dyn Embedder>>,
        expander: Option<Arc<dyn Expander>>,
    ) -> Engine {
        let root = tempdir().expect("create temp root");
        let root_path = root.path().to_path_buf();
        std::mem::forget(root);
        let config_dir = root_path.join("config");
        let cache_dir = root_path.join("cache");
        let storage = Storage::new(&cache_dir).expect("create storage");
        let config = Config {
            config_dir,
            cache_dir,
            default_space: None,
            models: ModelConfig {
                embedder: ModelSourceConfig {
                    provider: ModelProvider::HuggingFace,
                    id: "embed-model".to_string(),
                    revision: None,
                },
                reranker: ModelSourceConfig {
                    provider: ModelProvider::HuggingFace,
                    id: "reranker-model".to_string(),
                    revision: None,
                },
                expander: ModelSourceConfig {
                    provider: ModelProvider::HuggingFace,
                    id: "expander-model".to_string(),
                    revision: None,
                },
            },
            embeddings: embedder
                .as_ref()
                .map(|_| EmbeddingConfig::OpenAiCompatible {
                    model: "embed-model".to_string(),
                    base_url: "https://example.test/v1".to_string(),
                    api_key_env: None,
                    timeout_ms: 30_000,
                    batch_size: 32,
                    max_retries: 0,
                }),
            inference: InferenceConfig::default(),
            reaping: ReapingConfig { days: 7 },
            chunking: ChunkingConfig::default(),
            ranking: RankingConfig::default(),
        };
        Engine::from_parts_with_models(storage, config, embedder, None, expander)
    }

    fn seed_collection(
        root: &Path,
        collection: &str,
        relative_path: &str,
        content: &str,
    ) -> std::path::PathBuf {
        let collection_dir = root.join(collection);
        let full_path = collection_dir.join(relative_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).expect("create file parent");
        }
        fs::write(full_path, content).expect("write test document");
        collection_dir
    }

    fn seed_eval_file(engine: &Engine, content: &str) {
        fs::create_dir_all(&engine.config().config_dir).expect("create config dir");
        fs::write(engine.config().config_dir.join("eval.toml"), content).expect("write eval file");
    }

    const TRAITS_DOC: &str = r#"
Trait objects use dynamic dispatch, while generics use monomorphization.
Choose trait objects for heterogenous collections and generics for zero-cost abstraction.
"#;
}
