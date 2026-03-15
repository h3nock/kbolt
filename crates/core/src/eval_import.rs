use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use kbolt_types::{EvalCase, EvalDataset, EvalImportReport, EvalJudgment, KboltError};
use serde::Deserialize;

use crate::Result;

const SCIFACT_DATASET: &str = "scifact";
const DEFAULT_SPACE: &str = "bench";
const DEFAULT_COLLECTION: &str = "scifact";
const CORPUS_DIRNAME: &str = "corpus";
const MANIFEST_FILENAME: &str = "eval.toml";

#[derive(Debug, Deserialize)]
struct BeirCorpusRecord {
    #[serde(rename = "_id")]
    id: String,
    title: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct BeirQueryRecord {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

pub fn import_scifact(source: &Path, output: &Path) -> Result<EvalImportReport> {
    let layout = validate_scifact_layout(source)?;
    prepare_output_dir(output)?;

    let corpus_dir = output.join(CORPUS_DIRNAME);
    fs::create_dir_all(&corpus_dir)?;

    let document_ids = materialize_scifact_corpus(&layout.corpus, &corpus_dir)?;
    let queries = load_scifact_queries(&layout.queries)?;
    let (judgments_by_query, judgment_count) =
        load_scifact_qrels(&layout.qrels, &document_ids, &queries)?;
    let dataset = build_eval_dataset(queries, judgments_by_query)?;
    let query_count = dataset.cases.len();
    let manifest_path = output.join(MANIFEST_FILENAME);
    fs::write(&manifest_path, toml::to_string_pretty(&dataset)?)?;

    Ok(EvalImportReport {
        dataset: SCIFACT_DATASET.to_string(),
        source: source.display().to_string(),
        output_dir: output.display().to_string(),
        corpus_dir: corpus_dir.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
        default_space: DEFAULT_SPACE.to_string(),
        collection: DEFAULT_COLLECTION.to_string(),
        document_count: document_ids.len(),
        query_count,
        judgment_count,
    })
}

struct ScifactLayout {
    corpus: PathBuf,
    queries: PathBuf,
    qrels: PathBuf,
}

fn validate_scifact_layout(source: &Path) -> Result<ScifactLayout> {
    if !source.is_dir() {
        return Err(KboltError::InvalidInput(format!(
            "scifact source must be a directory: {}",
            source.display()
        ))
        .into());
    }

    let corpus = source.join("corpus.jsonl");
    let queries = source.join("queries.jsonl");
    let qrels = source.join("qrels").join("test.tsv");

    for (label, path) in [
        ("corpus.jsonl", &corpus),
        ("queries.jsonl", &queries),
        ("qrels/test.tsv", &qrels),
    ] {
        if !path.is_file() {
            return Err(KboltError::InvalidInput(format!(
                "invalid scifact source {}: missing {label}",
                source.display()
            ))
            .into());
        }
    }

    Ok(ScifactLayout {
        corpus,
        queries,
        qrels,
    })
}

fn prepare_output_dir(output: &Path) -> Result<()> {
    if output.exists() {
        if !output.is_dir() {
            return Err(KboltError::InvalidInput(format!(
                "eval import output must be a directory: {}",
                output.display()
            ))
            .into());
        }
        if fs::read_dir(output)?.next().transpose()?.is_some() {
            return Err(KboltError::InvalidInput(format!(
                "eval import output directory must be empty: {}",
                output.display()
            ))
            .into());
        }
        return Ok(());
    }

    fs::create_dir_all(output)?;
    Ok(())
}

fn materialize_scifact_corpus(corpus_file: &Path, corpus_dir: &Path) -> Result<HashSet<String>> {
    let records = read_jsonl::<BeirCorpusRecord>(corpus_file, "corpus")?;
    if records.is_empty() {
        return Err(KboltError::InvalidInput(format!(
            "scifact corpus is empty: {}",
            corpus_file.display()
        ))
        .into());
    }

    let mut ids = HashSet::with_capacity(records.len());
    for record in records {
        validate_record_id("corpus document", &record.id)?;
        if !ids.insert(record.id.clone()) {
            return Err(KboltError::InvalidInput(format!(
                "duplicate corpus document id '{}'",
                record.id
            ))
            .into());
        }
        let document_path = corpus_dir.join(format!("{}.md", record.id));
        fs::write(document_path, render_corpus_document(&record))?;
    }

    Ok(ids)
}

fn render_corpus_document(record: &BeirCorpusRecord) -> String {
    let title = record.title.trim();
    let text = record.text.trim();
    if title.is_empty() {
        format!("{text}\n")
    } else {
        format!("# {title}\n\n{text}\n")
    }
}

fn load_scifact_queries(queries_file: &Path) -> Result<Vec<BeirQueryRecord>> {
    let queries = read_jsonl::<BeirQueryRecord>(queries_file, "queries")?;
    if queries.is_empty() {
        return Err(KboltError::InvalidInput(format!(
            "scifact queries are empty: {}",
            queries_file.display()
        ))
        .into());
    }

    let mut seen = HashSet::with_capacity(queries.len());
    for query in &queries {
        validate_record_id("query", &query.id)?;
        if !seen.insert(query.id.clone()) {
            return Err(
                KboltError::InvalidInput(format!("duplicate query id '{}'", query.id)).into(),
            );
        }
        if query.text.trim().is_empty() {
            return Err(
                KboltError::InvalidInput(format!("query '{}' has empty text", query.id)).into(),
            );
        }
    }

    Ok(queries)
}

fn load_scifact_qrels(
    qrels_file: &Path,
    document_ids: &HashSet<String>,
    queries: &[BeirQueryRecord],
) -> Result<(HashMap<String, Vec<EvalJudgment>>, usize)> {
    let query_ids = queries
        .iter()
        .map(|query| query.id.as_str())
        .collect::<HashSet<_>>();
    let file = fs::File::open(qrels_file)?;
    let reader = BufReader::new(file);
    let mut judgments_by_query: HashMap<String, Vec<EvalJudgment>> = HashMap::new();
    let mut seen_pairs = HashSet::new();
    let mut judgment_count = 0;

    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if line_number == 1 && trimmed.eq_ignore_ascii_case("query-id\tcorpus-id\tscore") {
            continue;
        }

        let fields = trimmed.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(KboltError::InvalidInput(format!(
                "invalid scifact qrels line {} in {}: expected 3 tab-separated fields",
                line_number,
                qrels_file.display()
            ))
            .into());
        }

        let query_id = fields[0].trim();
        let document_id = fields[1].trim();
        let relevance = fields[2].trim().parse::<u8>().map_err(|err| {
            KboltError::InvalidInput(format!(
                "invalid relevance '{}' on qrels line {} in {}: {err}",
                fields[2].trim(),
                line_number,
                qrels_file.display()
            ))
        })?;

        validate_record_id("query", query_id)?;
        validate_record_id("corpus document", document_id)?;

        if !query_ids.contains(query_id) {
            return Err(KboltError::InvalidInput(format!(
                "qrels references unknown query id '{}' in {}",
                query_id,
                qrels_file.display()
            ))
            .into());
        }
        if !document_ids.contains(document_id) {
            return Err(KboltError::InvalidInput(format!(
                "qrels references unknown corpus document id '{}' in {}",
                document_id,
                qrels_file.display()
            ))
            .into());
        }
        if relevance == 0 {
            continue;
        }
        if !seen_pairs.insert((query_id.to_string(), document_id.to_string())) {
            return Err(KboltError::InvalidInput(format!(
                "duplicate qrels pair '{} -> {}' in {}",
                query_id,
                document_id,
                qrels_file.display()
            ))
            .into());
        }

        judgments_by_query
            .entry(query_id.to_string())
            .or_default()
            .push(EvalJudgment {
                path: format!("{DEFAULT_COLLECTION}/{document_id}.md"),
                relevance,
            });
        judgment_count += 1;
    }

    Ok((judgments_by_query, judgment_count))
}

fn build_eval_dataset(
    queries: Vec<BeirQueryRecord>,
    mut judgments_by_query: HashMap<String, Vec<EvalJudgment>>,
) -> Result<EvalDataset> {
    let mut cases = Vec::new();
    for query in queries {
        let Some(mut judgments) = judgments_by_query.remove(&query.id) else {
            continue;
        };
        judgments.sort_by(|left, right| {
            right
                .relevance
                .cmp(&left.relevance)
                .then_with(|| left.path.cmp(&right.path))
        });
        cases.push(EvalCase {
            query: query.text.trim().to_string(),
            space: Some(DEFAULT_SPACE.to_string()),
            collections: vec![DEFAULT_COLLECTION.to_string()],
            judgments,
        });
    }

    if cases.is_empty() {
        return Err(KboltError::InvalidInput(
            "scifact qrels did not produce any positive judged queries".to_string(),
        )
        .into());
    }

    Ok(EvalDataset { cases })
}

fn read_jsonl<T>(path: &Path, label: &str) -> Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record = serde_json::from_str(trimmed).map_err(|err| {
            KboltError::InvalidInput(format!(
                "invalid {label} jsonl line {} in {}: {err}",
                line_number,
                path.display()
            ))
        })?;
        records.push(record);
    }
    Ok(records)
}

fn validate_record_id(kind: &str, value: &str) -> Result<()> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(KboltError::InvalidInput(format!("{kind} id must not be empty")).into());
    }
    if trimmed == "." || trimmed == ".." || trimmed.contains('/') || trimmed.contains('\\') {
        return Err(KboltError::InvalidInput(format!(
            "{kind} id '{}' is not a valid filesystem-safe identifier",
            value
        ))
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use tempfile::tempdir;

    use crate::eval_store::load_eval_dataset_with_file;

    use super::import_scifact;

    #[test]
    fn import_scifact_materializes_corpus_and_manifest() {
        let tmp = tempdir().expect("create tempdir");
        let source = tmp.path().join("source");
        let output = tmp.path().join("output");
        write_scifact_fixture(&source);

        let report = import_scifact(&source, &output).expect("import scifact");

        assert_eq!(report.dataset, "scifact");
        assert_eq!(report.default_space, "bench");
        assert_eq!(report.collection, "scifact");
        assert_eq!(report.document_count, 2);
        assert_eq!(report.query_count, 2);
        assert_eq!(report.judgment_count, 3);
        assert_eq!(
            fs::read_to_string(output.join("corpus/10.md")).expect("read corpus doc"),
            "# Alpha Evidence\n\nAlpha evidence text.\n"
        );

        let dataset = load_eval_dataset_with_file(tmp.path(), Some(&output.join("eval.toml")))
            .expect("load imported manifest");
        assert_eq!(dataset.cases.len(), 2);
        assert_eq!(dataset.cases[0].space.as_deref(), Some("bench"));
        assert_eq!(dataset.cases[0].collections, vec!["scifact".to_string()]);
        assert_eq!(dataset.cases[0].judgments[0].path, "scifact/10.md");
        assert_eq!(dataset.cases[0].judgments[0].relevance, 2);
    }

    #[test]
    fn import_scifact_rejects_nonempty_output_directory() {
        let tmp = tempdir().expect("create tempdir");
        let source = tmp.path().join("source");
        let output = tmp.path().join("output");
        write_scifact_fixture(&source);
        fs::create_dir_all(&output).expect("create output");
        fs::write(output.join("keep.txt"), "existing").expect("write sentinel");

        let err = import_scifact(&source, &output).expect_err("nonempty output should fail");

        assert!(
            err.to_string()
                .contains("eval import output directory must be empty"),
            "unexpected error: {err}"
        );
    }

    fn write_scifact_fixture(root: &Path) {
        fs::create_dir_all(root.join("qrels")).expect("create qrels dir");
        fs::write(
            root.join("corpus.jsonl"),
            concat!(
                "{\"_id\":\"10\",\"title\":\"Alpha Evidence\",\"text\":\"Alpha evidence text.\"}\n",
                "{\"_id\":\"20\",\"title\":\"Beta Study\",\"text\":\"Beta study text.\"}\n"
            ),
        )
        .expect("write corpus");
        fs::write(
            root.join("queries.jsonl"),
            concat!(
                "{\"_id\":\"q1\",\"text\":\"alpha evidence\"}\n",
                "{\"_id\":\"q2\",\"text\":\"beta study\"}\n"
            ),
        )
        .expect("write queries");
        fs::write(
            root.join("qrels").join("test.tsv"),
            concat!(
                "query-id\tcorpus-id\tscore\n",
                "q1\t10\t2\n",
                "q1\t20\t1\n",
                "q2\t20\t1\n",
                "q2\t10\t0\n"
            ),
        )
        .expect("write qrels");
    }
}
