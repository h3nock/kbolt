use serde::{Deserialize, Serialize};

use crate::SearchMode;

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
    pub expected_paths: Vec<String>,
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
    pub recall_at_5: f32,
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
    pub expected_paths: Vec<String>,
    pub returned_paths: Vec<String>,
    pub matched_paths: Vec<String>,
    pub first_relevant_rank: Option<usize>,
    pub elapsed_ms: u64,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        EvalCase, EvalDataset, EvalModeFailure, EvalModeReport, EvalQueryReport, EvalRunReport,
    };
    use crate::SearchMode;

    #[test]
    fn eval_dataset_serializes_minimal_case_shape() {
        let value = serde_json::to_value(EvalDataset {
            cases: vec![EvalCase {
                query: "trait object vs generic".to_string(),
                space: Some("bench".to_string()),
                collections: vec!["rust".to_string()],
                expected_paths: vec!["rust/traits.md".to_string(), "rust/generics.md".to_string()],
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
                        "expected_paths": ["rust/traits.md", "rust/generics.md"]
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
                recall_at_5: 1.0,
                mrr_at_10: 1.0,
                latency_p50_ms: 3,
                latency_p95_ms: 4,
                queries: vec![EvalQueryReport {
                    query: "trait object vs generic".to_string(),
                    space: Some("bench".to_string()),
                    collections: vec!["rust".to_string()],
                    expected_paths: vec!["rust/traits.md".to_string()],
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
                        "recall_at_5": 1.0,
                        "mrr_at_10": 1.0,
                        "latency_p50_ms": 3,
                        "latency_p95_ms": 4,
                        "queries": [
                            {
                                "query": "trait object vs generic",
                                "space": "bench",
                                "collections": ["rust"],
                                "expected_paths": ["rust/traits.md"],
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
}
