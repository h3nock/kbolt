use kbolt_types::{
    EvalCase, EvalModeReport, EvalQueryReport, EvalRunReport, SearchMode, SearchRequest,
    SearchResult,
};
use std::collections::HashSet;

use crate::eval_store::load_eval_dataset;

use super::*;

impl Engine {
    pub fn run_eval(&self) -> Result<EvalRunReport> {
        let dataset = load_eval_dataset(&self.config.config_dir)?;
        let modes = self.eval_modes();
        let total_cases = dataset.cases.len();
        let mut reports = Vec::with_capacity(modes.len());

        for mode in modes {
            reports.push(self.run_eval_mode(&dataset.cases, mode)?);
        }

        Ok(EvalRunReport {
            total_cases,
            modes: reports,
        })
    }

    fn eval_modes(&self) -> Vec<SearchMode> {
        let mut modes = vec![SearchMode::Keyword, SearchMode::Auto, SearchMode::Deep];
        if self.embedder.is_some() {
            modes.push(SearchMode::Semantic);
        }
        modes
    }

    fn run_eval_mode(&self, cases: &[EvalCase], mode: SearchMode) -> Result<EvalModeReport> {
        let mut query_reports = Vec::with_capacity(cases.len());
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
                no_rerank: false,
                debug: false,
            })?;

            let returned_paths = dedupe_result_paths(case, &response.results);
            let expected_paths = case.expected_paths.iter().cloned().collect::<HashSet<_>>();
            let matched_paths = returned_paths
                .iter()
                .filter(|path| expected_paths.contains(path.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            let first_relevant_rank = returned_paths
                .iter()
                .position(|path| expected_paths.contains(path))
                .map(|index| index + 1);
            let matched_top_5 = returned_paths
                .iter()
                .take(5)
                .filter(|path| expected_paths.contains(path.as_str()))
                .count();
            recall_sum += matched_top_5 as f32 / case.expected_paths.len() as f32;
            mrr_sum += first_relevant_rank
                .map(|rank| 1.0_f32 / rank as f32)
                .unwrap_or(0.0);
            latencies.push(response.elapsed_ms);
            query_reports.push(EvalQueryReport {
                query: case.query.clone(),
                space: case.space.clone(),
                collections: case.collections.clone(),
                expected_paths: case.expected_paths.clone(),
                returned_paths,
                matched_paths,
                first_relevant_rank,
                elapsed_ms: response.elapsed_ms,
            });
        }

        let case_count = cases.len() as f32;
        Ok(EvalModeReport {
            mode,
            recall_at_5: recall_sum / case_count,
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
        ModelSourceConfig, ReapingConfig,
    };
    use crate::models::Embedder;
    use crate::storage::Storage;
    use kbolt_types::{AddCollectionRequest, SearchMode, UpdateOptions};

    use super::*;

    #[derive(Default)]
    struct DeterministicEmbedder;

    impl Embedder for DeterministicEmbedder {
        fn embed_batch(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
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

    #[test]
    fn run_eval_reports_keyword_auto_and_deep_without_embedder() {
        let root = tempdir().expect("create temp root");
        let collection_dir = seed_collection(root.path(), "rust", "guides/traits.md", TRAITS_DOC);
        let engine = test_engine(None);
        seed_eval_file(
            &engine,
            r#"
[[cases]]
query = "trait object generic"
space = "default"
collections = ["rust"]
expected_paths = ["rust/guides/traits.md"]
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

        let report = engine.run_eval().expect("run eval");

        assert_eq!(report.total_cases, 1);
        assert_eq!(
            report
                .modes
                .iter()
                .map(|mode| mode.mode.clone())
                .collect::<Vec<_>>(),
            vec![SearchMode::Keyword, SearchMode::Auto, SearchMode::Deep]
        );
        for mode in &report.modes {
            assert_eq!(mode.queries.len(), 1);
            assert_eq!(mode.queries[0].first_relevant_rank, Some(1));
            assert_eq!(mode.recall_at_5, 1.0);
            assert_eq!(mode.mrr_at_10, 1.0);
        }
    }

    #[test]
    fn run_eval_includes_semantic_mode_when_embedder_is_available() {
        let root = tempdir().expect("create temp root");
        let collection_dir = seed_collection(root.path(), "rust", "guides/traits.md", TRAITS_DOC);
        let engine = test_engine(Some(Arc::new(DeterministicEmbedder)));
        seed_eval_file(
            &engine,
            r#"
[[cases]]
query = "trait object generic"
space = "default"
collections = ["rust"]
expected_paths = ["rust/guides/traits.md"]
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

        let report = engine.run_eval().expect("run eval");

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
        assert_eq!(semantic.recall_at_5, 1.0);
        assert_eq!(semantic.mrr_at_10, 1.0);
    }

    fn test_engine(embedder: Option<Arc<dyn Embedder>>) -> Engine {
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
        };
        Engine::from_parts_with_embedder(storage, config, embedder)
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
