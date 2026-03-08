use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use kbolt_types::{EvalDataset, KboltError};

use crate::error::Result;

const EVAL_FILENAME: &str = "eval.toml";

pub(crate) fn load_eval_dataset(config_dir: &Path) -> Result<EvalDataset> {
    fs::create_dir_all(config_dir)?;

    let eval_file = eval_file_path(config_dir);
    let raw = fs::read_to_string(&eval_file).map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => KboltError::Config(format!(
            "eval file not found: {}. create eval.toml under the config directory.",
            eval_file.display()
        )),
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
        case.expected_paths = normalize_string_list(
            std::mem::take(&mut case.expected_paths),
            eval_file,
            case_number,
            "expected_paths",
        )?;
        if case.expected_paths.is_empty() {
            return Err(KboltError::Config(format!(
                "invalid eval file {}: case {} must include at least one expected_paths entry",
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

#[cfg(test)]
mod tests {
    use std::fs;

    use kbolt_types::{EvalCase, EvalDataset};
    use tempfile::tempdir;

    use super::load_eval_dataset;

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
expected_paths = ["rust/traits.md", "rust/traits.md", "rust/generics.md"]
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
                    expected_paths: vec![
                        "rust/traits.md".to_string(),
                        "rust/generics.md".to_string()
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
    fn load_rejects_empty_expected_path_entries() {
        let tmp = tempdir().expect("create tempdir");
        let config_dir = tmp.path().join("config");
        fs::create_dir_all(&config_dir).expect("create config dir");
        fs::write(
            config_dir.join(EVAL_FILENAME),
            r#"
[[cases]]
query = "trait object vs generic"
expected_paths = ["rust/traits.md", "   "]
"#,
        )
        .expect("write eval file");

        let err = load_eval_dataset(&config_dir).expect_err("empty expected path should fail");

        assert!(
            err.to_string()
                .contains("field 'expected_paths' contains an empty value"),
            "unexpected error: {err}"
        );
    }
}
