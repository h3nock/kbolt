use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use kbolt_types::{EvalDataset, EvalJudgment, KboltError};

use crate::error::Result;

const EVAL_FILENAME: &str = "eval.toml";

pub(crate) fn load_eval_dataset(config_dir: &Path) -> Result<EvalDataset> {
    load_eval_dataset_from_path(&eval_file_path(config_dir), true)
}

pub(crate) fn load_eval_dataset_with_file(
    config_dir: &Path,
    eval_file: Option<&Path>,
) -> Result<EvalDataset> {
    match eval_file {
        Some(path) => load_eval_dataset_from_path(path, false),
        None => load_eval_dataset(config_dir),
    }
}

fn load_eval_dataset_from_path(eval_file: &Path, ensure_config_dir: bool) -> Result<EvalDataset> {
    if ensure_config_dir {
        let config_dir = eval_file.parent().ok_or_else(|| {
            KboltError::Config(format!(
                "invalid eval path {}: missing parent directory",
                eval_file.display()
            ))
        })?;
        fs::create_dir_all(config_dir)?;
    }

    if eval_file.is_dir() {
        return Err(KboltError::Config(format!(
            "invalid eval file {}: expected a file, found a directory",
            eval_file.display()
        ))
        .into());
    }

    let raw = fs::read_to_string(eval_file).map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound if ensure_config_dir => KboltError::Config(format!(
            "eval file not found: {}. create eval.toml under the config directory.",
            eval_file.display()
        )),
        std::io::ErrorKind::NotFound => {
            KboltError::Config(format!("eval file not found: {}", eval_file.display()))
        }
        _ => KboltError::Io(err),
    })?;

    let mut dataset: EvalDataset = toml::from_str(&raw).map_err(|err| {
        KboltError::Config(format!("invalid eval file {}: {err}", eval_file.display()))
    })?;
    normalize_dataset(&mut dataset, &eval_file)?;
    Ok(dataset)
}

pub(crate) fn eval_file_path(config_dir: &Path) -> PathBuf {
    config_dir.join(EVAL_FILENAME)
}

fn normalize_dataset(dataset: &mut EvalDataset, eval_file: &Path) -> Result<()> {
    if dataset.cases.is_empty() {
        return Err(KboltError::Config(format!(
            "invalid eval file {}: at least one [[cases]] entry is required",
            eval_file.display()
        ))
        .into());
    }

    for (index, case) in dataset.cases.iter_mut().enumerate() {
        let case_number = index + 1;
        case.query = normalize_required_text(case.query.trim(), eval_file, case_number, "query")?;
        case.space = normalize_optional_text(case.space.take());
        case.collections = normalize_string_list(
            std::mem::take(&mut case.collections),
            eval_file,
            case_number,
            "collections",
        )?;
        case.judgments =
            normalize_judgments(std::mem::take(&mut case.judgments), eval_file, case_number)?;
        if case.judgments.is_empty() {
            return Err(KboltError::Config(format!(
                "invalid eval file {}: case {} must include at least one judgments entry",
                eval_file.display(),
                case_number
            ))
            .into());
        }
        if !case.judgments.iter().any(|judgment| judgment.relevance > 0) {
            return Err(KboltError::Config(format!(
                "invalid eval file {}: case {} must include at least one judgment with relevance > 0",
                eval_file.display(),
                case_number
            ))
            .into());
        }
    }

    Ok(())
}

fn normalize_required_text(
    value: &str,
    eval_file: &Path,
    case_number: usize,
    field: &str,
) -> Result<String> {
    if value.is_empty() {
        return Err(KboltError::Config(format!(
            "invalid eval file {}: case {} field '{field}' must not be empty",
            eval_file.display(),
            case_number
        ))
        .into());
    }

    Ok(value.to_string())
}

fn normalize_optional_text(value: Option<String>) -> Option<String> {
    value.and_then(|item| {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn normalize_string_list(
    values: Vec<String>,
    eval_file: &Path,
    case_number: usize,
    field: &str,
) -> Result<Vec<String>> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(KboltError::Config(format!(
                "invalid eval file {}: case {} field '{field}' contains an empty value",
                eval_file.display(),
                case_number
            ))
            .into());
        }
        if seen.insert(trimmed.to_string()) {
            normalized.push(trimmed.to_string());
        }
    }
    Ok(normalized)
}

fn normalize_judgments(
    judgments: Vec<EvalJudgment>,
    eval_file: &Path,
    case_number: usize,
) -> Result<Vec<EvalJudgment>> {
    let mut seen_paths = HashSet::new();
    let mut normalized = Vec::with_capacity(judgments.len());
    for mut judgment in judgments {
        let trimmed_path = judgment.path.trim();
        if trimmed_path.is_empty() {
            return Err(KboltError::Config(format!(
                "invalid eval file {}: case {} field 'judgments' contains an empty path",
                eval_file.display(),
                case_number
            ))
            .into());
        }
        if !seen_paths.insert(trimmed_path.to_string()) {
            return Err(KboltError::Config(format!(
                "invalid eval file {}: case {} field 'judgments' contains duplicate path '{}'",
                eval_file.display(),
                case_number,
                trimmed_path
            ))
            .into());
        }
        judgment.path = trimmed_path.to_string();
        normalized.push(judgment);
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use kbolt_types::{EvalCase, EvalDataset, EvalJudgment};
    use tempfile::tempdir;

    use super::{load_eval_dataset, load_eval_dataset_with_file};

    const EVAL_FILENAME: &str = "eval.toml";

    #[test]
    fn load_roundtrips_trimmed_eval_cases() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        fs::create_dir_all(&config_dir).expect("create config dir");
        fs::write(
            config_dir.join(EVAL_FILENAME),
            r#"
[[cases]]
query = "  trait object vs generic  "
space = " bench "
collections = ["rust", "rust"]
judgments = [
  { path = " rust/traits.md ", relevance = 2 },
  { path = "rust/generics.md", relevance = 1 },
]
"#,
        )
        .expect("write eval file");

        let dataset = load_eval_dataset(&config_dir).expect("load eval dataset");

        assert_eq!(
            dataset,
            EvalDataset {
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
                }]
            }
        );
    }

    #[test]
    fn load_rejects_missing_eval_file() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");

        let err = load_eval_dataset(&config_dir).expect_err("missing eval file should fail");

        assert!(
            err.to_string().contains("eval file not found"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_empty_cases() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        fs::create_dir_all(&config_dir).expect("create config dir");
        fs::write(config_dir.join(EVAL_FILENAME), "cases = []\n").expect("write eval file");

        let err = load_eval_dataset(&config_dir).expect_err("empty cases should fail");

        assert!(
            err.to_string()
                .contains("at least one [[cases]] entry is required"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_empty_judgment_path_entries() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        fs::create_dir_all(&config_dir).expect("create config dir");
        fs::write(
            config_dir.join(EVAL_FILENAME),
            r#"
[[cases]]
query = "trait object vs generic"
judgments = [{ path = "rust/traits.md", relevance = 1 }, { path = "   ", relevance = 1 }]
"#,
        )
        .expect("write eval file");

        let err = load_eval_dataset(&config_dir).expect_err("empty judgment path should fail");

        assert!(
            err.to_string()
                .contains("field 'judgments' contains an empty path"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_duplicate_judgment_paths() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        fs::create_dir_all(&config_dir).expect("create config dir");
        fs::write(
            config_dir.join(EVAL_FILENAME),
            r#"
[[cases]]
query = "trait object vs generic"
judgments = [
  { path = "rust/traits.md", relevance = 2 },
  { path = "rust/traits.md", relevance = 1 },
]
"#,
        )
        .expect("write eval file");

        let err = load_eval_dataset(&config_dir).expect_err("duplicate judgments should fail");

        assert!(
            err.to_string()
                .contains("contains duplicate path 'rust/traits.md'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_cases_without_positive_judgments() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        fs::create_dir_all(&config_dir).expect("create config dir");
        fs::write(
            config_dir.join(EVAL_FILENAME),
            r#"
[[cases]]
query = "trait object vs generic"
judgments = [{ path = "rust/traits.md", relevance = 0 }]
"#,
        )
        .expect("write eval file");

        let err =
            load_eval_dataset(&config_dir).expect_err("missing positive judgments should fail");

        assert!(
            err.to_string()
                .contains("must include at least one judgment with relevance > 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_supports_explicit_eval_file_path() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        let eval_file = tmp.path().join("bench").join("scifact.toml");
        fs::create_dir_all(eval_file.parent().expect("bench dir")).expect("create bench dir");
        fs::write(
            &eval_file,
            r#"
[[cases]]
query = "trait object vs generic"
judgments = [{ path = "rust/traits.md", relevance = 1 }]
"#,
        )
        .expect("write eval file");

        let dataset = load_eval_dataset_with_file(&config_dir, Some(&eval_file))
            .expect("load explicit eval file");

        assert_eq!(dataset.cases.len(), 1);
        assert_eq!(dataset.cases[0].judgments[0].path, "rust/traits.md");
    }

    #[test]
    fn load_rejects_directory_explicit_eval_path() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        let eval_dir = tmp.path().join("bench");
        fs::create_dir_all(&eval_dir).expect("create bench dir");

        let err = load_eval_dataset_with_file(&config_dir, Some(&eval_dir))
            .expect_err("directory path should fail");

        assert!(
            err.to_string()
                .contains("expected a file, found a directory"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_missing_explicit_eval_file_path() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        let eval_file = tmp.path().join("bench").join("missing.toml");

        let err = load_eval_dataset_with_file(&config_dir, Some(&eval_file))
            .expect_err("missing explicit file should fail");

        assert!(
            err.to_string()
                .contains(&format!("eval file not found: {}", eval_file.display())),
            "unexpected error: {err}"
        );
    }
}
