use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use kbolt_types::{EvalCase, EvalDataset, EvalImportReport, EvalJudgment, KboltError};
use serde::Deserialize;

use crate::Result;

const DEFAULT_SPACE: &str = "bench";
const CORPUS_DIRNAME: &str = "corpus";
const MANIFEST_FILENAME: &str = "eval.toml";

#[derive(Debug, Deserialize)]
struct BeirCorpusRecord {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: Option<String>,
    text: String,
}

#[derive(Debug, Deserialize)]
struct BeirQueryRecord {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

pub fn import_beir(
    dataset: &str,
    source: &Path,
    output: &Path,
    collection: Option<&str>,
) -> Result<EvalImportReport> {
    let dataset = normalize_import_name("dataset", dataset)?;
    let collection = normalize_import_name("collection", collection.unwrap_or(dataset))?;
    let layout = validate_beir_layout(dataset, source)?;
    prepare_output_dir(output)?;

    let corpus_dir = output.join(CORPUS_DIRNAME);
    fs::create_dir_all(&corpus_dir)?;

    let document_ids = materialize_beir_corpus(dataset, &layout.corpus, &corpus_dir)?;
    let queries = load_beir_queries(dataset, &layout.queries)?;
    let (judgments_by_query, judgment_count) =
        load_beir_qrels(dataset, collection, &layout.qrels, &document_ids, &queries)?;
    let eval_dataset = build_eval_dataset(dataset, collection, queries, judgments_by_query)?;
    let query_count = eval_dataset.cases.len();
    let manifest_path = output.join(MANIFEST_FILENAME);
    fs::write(&manifest_path, toml::to_string_pretty(&eval_dataset)?)?;

    Ok(EvalImportReport {
        dataset: dataset.to_string(),
        source: source.display().to_string(),
        output_dir: output.display().to_string(),
        corpus_dir: corpus_dir.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
        default_space: DEFAULT_SPACE.to_string(),
        collection: collection.to_string(),
        document_count: document_ids.len(),
        query_count,
        judgment_count,
    })
}

struct BeirLayout {
    corpus: PathBuf,
    queries: PathBuf,
    qrels: PathBuf,
}

fn validate_beir_layout(dataset: &str, source: &Path) -> Result<BeirLayout> {
    if !source.is_dir() {
        return Err(KboltError::InvalidInput(format!(
            "{dataset} source must be a directory: {}",
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
                "invalid {dataset} source {}: missing {label}",
                source.display()
            ))
            .into());
        }
    }

    Ok(BeirLayout {
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

fn materialize_beir_corpus(
    dataset: &str,
    corpus_file: &Path,
    corpus_dir: &Path,
) -> Result<HashSet<String>> {
    let records = read_jsonl::<BeirCorpusRecord>(corpus_file, "corpus")?;
    if records.is_empty() {
        return Err(KboltError::InvalidInput(format!(
            "{dataset} corpus is empty: {}",
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
    let title = record.title.as_deref().map(str::trim).unwrap_or_default();
    let text = record.text.trim();
    if title.is_empty() {
        format!("{text}\n")
    } else {
        format!("# {title}\n\n{text}\n")
    }
}

fn load_beir_queries(dataset: &str, queries_file: &Path) -> Result<Vec<BeirQueryRecord>> {
    let queries = read_jsonl::<BeirQueryRecord>(queries_file, "queries")?;
    if queries.is_empty() {
        return Err(KboltError::InvalidInput(format!(
            "{dataset} queries are empty: {}",
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

fn load_beir_qrels(
    dataset: &str,
    collection: &str,
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
                "invalid {dataset} qrels line {} in {}: expected 3 tab-separated fields",
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
                path: format!("{collection}/{document_id}.md"),
                relevance,
            });
        judgment_count += 1;
    }

    Ok((judgments_by_query, judgment_count))
}

fn build_eval_dataset(
    dataset: &str,
    collection: &str,
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
            collections: vec![collection.to_string()],
            judgments,
        });
    }

    if cases.is_empty() {
        return Err(KboltError::InvalidInput(format!(
            "{dataset} qrels did not produce any positive judged queries"
        ))
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

fn normalize_import_name<'a>(kind: &str, value: &'a str) -> Result<&'a str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(KboltError::InvalidInput(format!("{kind} name must not be empty")).into());
    }
    validate_record_id(kind, trimmed)?;
    Ok(trimmed)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use tempfile::tempdir;

    use crate::eval_store::load_eval_dataset_with_file;

    use super::import_beir;

    #[test]
    fn import_beir_materializes_corpus_and_manifest() {
        let tmp = tempdir().expect("create tempdir");
        let source = tmp.path().join("source");
        let output = tmp.path().join("output");
        write_beir_fixture(&source);

        let report = import_beir("scifact", &source, &output, None).expect("import benchmark");

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
    fn import_beir_honors_collection_override() {
        let tmp = tempdir().expect("create tempdir");
        let source = tmp.path().join("source");
        let output = tmp.path().join("output");
        write_beir_fixture(&source);

        let report =
            import_beir("fiqa", &source, &output, Some("finance")).expect("import benchmark");

        assert_eq!(report.dataset, "fiqa");
        assert_eq!(report.collection, "finance");

        let dataset = load_eval_dataset_with_file(tmp.path(), Some(&output.join("eval.toml")))
            .expect("load imported manifest");
        assert_eq!(dataset.cases[0].collections, vec!["finance".to_string()]);
        assert_eq!(dataset.cases[0].judgments[0].path, "finance/10.md");
    }

    #[test]
    fn import_beir_accepts_missing_titles() {
        let tmp = tempdir().expect("create tempdir");
        let source = tmp.path().join("source");
        let output = tmp.path().join("output");
        write_beir_fixture_with_missing_title(&source);

        import_beir("fiqa", &source, &output, None).expect("import benchmark");

        assert_eq!(
            fs::read_to_string(output.join("corpus/10.md")).expect("read corpus doc"),
            "Alpha evidence text.\n"
        );
    }

    #[test]
    fn import_beir_rejects_nonempty_output_directory() {
        let tmp = tempdir().expect("create tempdir");
        let source = tmp.path().join("source");
        let output = tmp.path().join("output");
        write_beir_fixture(&source);
        fs::create_dir_all(&output).expect("create output");
        fs::write(output.join("keep.txt"), "existing").expect("write sentinel");

        let err =
            import_beir("fiqa", &source, &output, None).expect_err("nonempty output should fail");

        assert!(
            err.to_string()
                .contains("eval import output directory must be empty"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn import_beir_rejects_missing_test_split() {
        let tmp = tempdir().expect("create tempdir");
        let source = tmp.path().join("source");
        fs::create_dir_all(source.join("qrels")).expect("create qrels dir");
        fs::write(
            source.join("corpus.jsonl"),
            "{\"_id\":\"10\",\"title\":\"Alpha Evidence\",\"text\":\"Alpha evidence text.\"}\n",
        )
        .expect("write corpus");
        fs::write(
            source.join("queries.jsonl"),
            "{\"_id\":\"q1\",\"text\":\"alpha evidence\"}\n",
        )
        .expect("write queries");

        let err = import_beir("fiqa", &source, &tmp.path().join("output"), None)
            .expect_err("missing qrels/test.tsv should fail");

        assert!(
            err.to_string().contains("missing qrels/test.tsv"),
            "unexpected error: {err}"
        );
    }

    fn write_beir_fixture(root: &Path) {
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

    fn write_beir_fixture_with_missing_title(root: &Path) {
        fs::create_dir_all(root.join("qrels")).expect("create qrels dir");
        fs::write(
            root.join("corpus.jsonl"),
            "{\"_id\":\"10\",\"text\":\"Alpha evidence text.\"}\n",
        )
        .expect("write corpus");
        fs::write(
            root.join("queries.jsonl"),
            "{\"_id\":\"q1\",\"text\":\"alpha evidence\"}\n",
        )
        .expect("write queries");
        fs::write(
            root.join("qrels").join("test.tsv"),
            concat!("query-id\tcorpus-id\tscore\n", "q1\t10\t2\n"),
        )
        .expect("write qrels");
    }
}
