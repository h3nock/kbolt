use serde::{Deserialize, Serialize};

use crate::SearchMode;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalJudgment {
    pub path: String,
    pub relevance: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalImportReport {
    pub dataset: String,
    pub source: String,
    pub output_dir: String,
    pub corpus_dir: String,
    pub manifest_path: String,
    pub default_space: String,
    pub collection: String,
    pub document_count: usize,
    pub query_count: usize,
    pub judgment_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalDataset {
    pub cases: Vec<EvalCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalCase {
    pub query: String,
    #[serde(default)]
    pub space: Option<String>,
    #[serde(default)]
    pub collections: Vec<String>,
    #[serde(default)]
    pub judgments: Vec<EvalJudgment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalRunReport {
    pub total_cases: usize,
    pub modes: Vec<EvalModeReport>,
    pub failed_modes: Vec<EvalModeFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalModeReport {
    pub mode: SearchMode,
    pub no_rerank: bool,
    pub ndcg_at_10: f32,
    pub recall_at_10: f32,
    pub mrr_at_10: f32,
    pub latency_p50_ms: u64,
    pub latency_p95_ms: u64,
    pub queries: Vec<EvalQueryReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalModeFailure {
    pub mode: SearchMode,
    pub no_rerank: bool,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalQueryReport {
    pub query: String,
    pub space: Option<String>,
    pub collections: Vec<String>,
    pub judgments: Vec<EvalJudgment>,
    pub returned_paths: Vec<String>,
    pub matched_paths: Vec<String>,
    pub first_relevant_rank: Option<usize>,
    pub elapsed_ms: u64,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        EvalCase, EvalDataset, EvalImportReport, EvalJudgment, EvalModeFailure, EvalModeReport,
        EvalQueryReport, EvalRunReport,
    };
    use crate::SearchMode;

    #[test]
    fn eval_dataset_serializes_minimal_case_shape() {
        let value = serde_json::to_value(EvalDataset {
            cases: vec![EvalCase {
                query: "trait object vs generic".to_string(),
                space: Some("bench".to_string()),
                collections: vec!["rust".to_string()],
                judgments: vec![
                    EvalJudgment {
                        path: "rust/traits.md".to_string(),
                        relevance: 2,
                    },
                    EvalJudgment {
                        path: "rust/generics.md".to_string(),
                        relevance: 1,
                    },
                ],
            }],
        })
        .expect("serialize eval dataset");

        assert_eq!(
            value,
            json!({
                "cases": [
                    {
                        "query": "trait object vs generic",
                        "space": "bench",
                        "collections": ["rust"],
                        "judgments": [
                            {"path": "rust/traits.md", "relevance": 2},
                            {"path": "rust/generics.md", "relevance": 1}
                        ]
                    }
                ]
            })
        );
    }

    #[test]
    fn eval_run_report_serializes_mode_metrics_and_queries() {
        let value = serde_json::to_value(EvalRunReport {
            total_cases: 1,
            modes: vec![EvalModeReport {
                mode: SearchMode::Keyword,
                no_rerank: true,
                ndcg_at_10: 1.0,
                recall_at_10: 1.0,
                mrr_at_10: 1.0,
                latency_p50_ms: 3,
                latency_p95_ms: 4,
                queries: vec![EvalQueryReport {
                    query: "trait object vs generic".to_string(),
                    space: Some("bench".to_string()),
                    collections: vec!["rust".to_string()],
                    judgments: vec![EvalJudgment {
                        path: "rust/traits.md".to_string(),
                        relevance: 1,
                    }],
                    returned_paths: vec!["rust/traits.md".to_string()],
                    matched_paths: vec!["rust/traits.md".to_string()],
                    first_relevant_rank: Some(1),
                    elapsed_ms: 3,
                }],
            }],
            failed_modes: vec![EvalModeFailure {
                mode: SearchMode::Deep,
                no_rerank: false,
                error: "model not available".to_string(),
            }],
        })
        .expect("serialize eval report");

        assert_eq!(
            value,
            json!({
                "total_cases": 1,
                "modes": [
                    {
                        "mode": "Keyword",
                        "no_rerank": true,
                        "ndcg_at_10": 1.0,
                        "recall_at_10": 1.0,
                        "mrr_at_10": 1.0,
                        "latency_p50_ms": 3,
                        "latency_p95_ms": 4,
                        "queries": [
                            {
                                "query": "trait object vs generic",
                                "space": "bench",
                                "collections": ["rust"],
                                "judgments": [
                                    {
                                        "path": "rust/traits.md",
                                        "relevance": 1
                                    }
                                ],
                                "returned_paths": ["rust/traits.md"],
                                "matched_paths": ["rust/traits.md"],
                                "first_relevant_rank": 1,
                                "elapsed_ms": 3
                            }
                        ]
                    }
                ],
                "failed_modes": [
                    {
                        "mode": "Deep",
                        "no_rerank": false,
                        "error": "model not available"
                    }
                ]
            })
        );
    }

    #[test]
    fn eval_import_report_serializes_paths_and_counts() {
        let value = serde_json::to_value(EvalImportReport {
            dataset: "scifact".to_string(),
            source: "/tmp/scifact-source".to_string(),
            output_dir: "/tmp/scifact-bench".to_string(),
            corpus_dir: "/tmp/scifact-bench/corpus".to_string(),
            manifest_path: "/tmp/scifact-bench/eval.toml".to_string(),
            default_space: "bench".to_string(),
            collection: "scifact".to_string(),
            document_count: 5_183,
            query_count: 300,
            judgment_count: 1_109,
        })
        .expect("serialize import report");

        assert_eq!(
            value,
            json!({
                "dataset": "scifact",
                "source": "/tmp/scifact-source",
                "output_dir": "/tmp/scifact-bench",
                "corpus_dir": "/tmp/scifact-bench/corpus",
                "manifest_path": "/tmp/scifact-bench/eval.toml",
                "default_space": "bench",
                "collection": "scifact",
                "document_count": 5183,
                "query_count": 300,
                "judgment_count": 1109
            })
        );
    }
}
