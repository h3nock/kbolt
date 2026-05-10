use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use kbolt_types::{
    DoctorCheck, DoctorCheckStatus, DoctorReport, DoctorSetupStatus, KboltError, ModelInfo,
    ModelStatus,
};
use rusqlite::{Connection, OpenFlags};
use tantivy::Index;
use usearch::{IndexOptions, MetricKind, ScalarKind};

use crate::config;
use crate::config::RoleBindingsConfig;
use crate::models;
use crate::models::EmbeddingInputKind;

const SQLITE_FILE_NAME: &str = "meta.sqlite";
const SPACES_DIR_NAME: &str = "spaces";
const TANTIVY_DIR_NAME: &str = "tantivy";
const USEARCH_FILE_NAME: &str = "vectors.usearch";
const WRITE_CHECK_FILE_NAME: &str = ".kbolt-doctor-write-check";
const SETUP_FIX: &str =
    "Create an index.toml with provider profiles and role bindings, then rerun `kbolt doctor`.";

pub fn run(config_path: Option<&Path>) -> DoctorReport {
    let mut checks = Vec::new();
    let mut config_file = None;
    let mut config_dir = None;
    let mut cache_dir = None;

    let resolved_config_file = match timed_result_check(
        "config.path_resolves",
        "config",
        Some("Use the platform default config location or pass a valid kbolt config directory."),
        || config::resolve_config_file_path(config_path),
    ) {
        (check, Ok(path)) => {
            checks.push(check);
            config_dir = path.parent().map(Path::to_path_buf);
            config_file = Some(path.clone());
            path
        }
        (check, Err(_)) => {
            checks.push(check);
            return finalize_report(
                DoctorSetupStatus::ConfigInvalid,
                config_file,
                config_dir,
                cache_dir,
                checks,
            );
        }
    };

    if !resolved_config_file.exists() {
        checks.push(timed_static_check(
            "config.file_exists",
            "config",
            DoctorCheckStatus::Fail,
            format!(
                "kbolt is not set up yet: config file does not exist at {}",
                resolved_config_file.display()
            ),
            Some(SETUP_FIX.to_string()),
        ));
        return finalize_report(
            DoctorSetupStatus::ConfigMissing,
            config_file,
            config_dir,
            cache_dir,
            checks,
        );
    }

    let config = match timed_result_check(
        "config.file_parses",
        "config",
        Some("Fix the TOML config file and rerun `kbolt doctor`."),
        || config::load_existing(Some(&resolved_config_file)),
    ) {
        (check, Ok(config)) => {
            cache_dir = Some(config.cache_dir.clone());
            checks.push(check);
            config
        }
        (check, Err(_)) => {
            checks.push(check);
            return finalize_report(
                DoctorSetupStatus::ConfigInvalid,
                config_file,
                config_dir,
                cache_dir,
                checks,
            );
        }
    };

    if is_not_configured(&config.roles) {
        checks.push(timed_static_check(
            "config.roles_bound",
            "config",
            DoctorCheckStatus::Fail,
            "kbolt is not set up yet: no inference roles are configured".to_string(),
            Some(SETUP_FIX.to_string()),
        ));
        return finalize_report(
            DoctorSetupStatus::NotConfigured,
            config_file,
            config_dir,
            cache_dir,
            checks,
        );
    }

    checks.push(timed_static_check(
        "config.roles_bound",
        "config",
        DoctorCheckStatus::Pass,
        "at least one inference role is configured".to_string(),
        None,
    ));

    if let Some(path) = config_dir.as_deref() {
        checks.push(check_directory_writable(
            "config.dir_writable",
            "config",
            path,
        ));
    }
    checks.push(check_cache_dir(&config.cache_dir));
    checks.push(check_sqlite_readable(&config.cache_dir));
    checks.push(check_search_indexes_readable(&config.cache_dir));

    let model_status = match models::status(&config) {
        Ok(status) => status,
        Err(err) => {
            checks.push(timed_static_check(
                "models.status",
                "models",
                DoctorCheckStatus::Fail,
                format!("failed to resolve model status: {err}"),
                Some("Fix provider profiles and role bindings in index.toml.".to_string()),
            ));
            return finalize_report(
                DoctorSetupStatus::Configured,
                config_file,
                config_dir,
                cache_dir,
                checks,
            );
        }
    };

    checks.extend(role_binding_checks(&config.roles, &model_status));
    checks.extend(role_readiness_checks(&model_status));

    let built_models = match timed_result_check(
        "models.build_clients",
        "models",
        Some("Fix provider profile settings that are incompatible with their bound roles."),
        || models::build_inference_clients(&config),
    ) {
        (check, Ok(models)) => {
            checks.push(check);
            models
        }
        (check, Err(_)) => {
            checks.push(check);
            return finalize_report(
                DoctorSetupStatus::Configured,
                config_file,
                config_dir,
                cache_dir,
                checks,
            );
        }
    };

    if model_status.embedder.ready {
        if let Some(embedder) = built_models.embedder.as_ref() {
            checks.push(timed_unit_check(
                "roles.embedder.embedding_smoke",
                "roles.embedder",
                Some("Check the embedder model, endpoint path, and server logs."),
                || {
                    let input = vec!["kbolt doctor embedding smoke".to_string()];
                    let vectors = embedder.embed_batch(EmbeddingInputKind::Query, &input)?;
                    if vectors.len() != 1 || vectors[0].is_empty() {
                        return Err(KboltError::Inference(
                            "embedding smoke returned an empty response".to_string(),
                        )
                        .into());
                    }
                    Ok(())
                },
            ));
        }
        if let Some(tokenizer) = built_models.embedding_tokenizer.as_ref() {
            checks.push(timed_unit_check(
                "roles.embedder.tokenize_smoke",
                "roles.embedder",
                Some("Verify the embedder tokenizer runtime can count tokens."),
                || {
                    let counts =
                        tokenizer.count_embedding_tokens_batch(&["kbolt doctor tokenize smoke"])?;
                    let count = counts.into_iter().next().ok_or_else(|| {
                        KboltError::Inference("tokenize smoke returned no token count".to_string())
                    })?;
                    if count == 0 {
                        return Err(KboltError::Inference(
                            "tokenize smoke returned zero tokens".to_string(),
                        )
                        .into());
                    }
                    Ok(())
                },
            ));
        }
    }

    if model_status.reranker.ready {
        if let Some(reranker) = built_models.reranker.as_ref() {
            checks.push(timed_unit_check(
                "roles.reranker.rerank_smoke",
                "roles.reranker",
                Some("Check the reranker model, endpoint path, and server logs."),
                || {
                    let scores = reranker.rerank(
                        "kbolt doctor rerank smoke",
                        &["kbolt doctor rerank document".to_string()],
                    )?;
                    if scores.len() != 1 || !scores[0].is_finite() {
                        return Err(KboltError::Inference(
                            "rerank smoke returned an invalid response".to_string(),
                        )
                        .into());
                    }
                    Ok(())
                },
            ));
        }
    }

    if model_status.expander.ready {
        if let Some(expander) = built_models.expander.as_ref() {
            checks.push(timed_unit_check(
                "roles.expander.chat_smoke",
                "roles.expander",
                Some("Check the expander model, endpoint path, and server logs."),
                || {
                    let variants = expander.expand("kbolt doctor smoke", 2)?;
                    if variants.is_empty() {
                        return Err(KboltError::Inference(
                            "expander smoke returned no variants".to_string(),
                        )
                        .into());
                    }
                    Ok(())
                },
            ));
        }
    }

    finalize_report(
        DoctorSetupStatus::Configured,
        config_file,
        config_dir,
        cache_dir,
        checks,
    )
}

fn role_binding_checks(roles: &RoleBindingsConfig, model_status: &ModelStatus) -> Vec<DoctorCheck> {
    let mut checks = Vec::new();

    checks.push(role_binding_check(
        "roles.embedder.bound",
        "roles.embedder",
        roles.embedder.is_some(),
        &model_status.embedder,
        "configure `[roles.embedder]` to enable semantic search and embeddings",
    ));
    checks.push(role_binding_check(
        "roles.reranker.bound",
        "roles.reranker",
        roles.reranker.is_some(),
        &model_status.reranker,
        "configure `[roles.reranker]` to enable reranking",
    ));
    checks.push(role_binding_check(
        "roles.expander.bound",
        "roles.expander",
        roles.expander.is_some(),
        &model_status.expander,
        "configure `[roles.expander]` to enable deep query expansion",
    ));
    checks
}

fn role_binding_check(
    id: &str,
    scope: &str,
    configured: bool,
    info: &ModelInfo,
    fix: &str,
) -> DoctorCheck {
    if configured {
        timed_static_check(
            id,
            scope,
            DoctorCheckStatus::Pass,
            format!(
                "bound to profile={} kind={} operation={} model={} endpoint={}",
                info.profile.as_deref().unwrap_or("unknown"),
                info.kind.as_deref().unwrap_or("unknown"),
                info.operation.as_deref().unwrap_or("unknown"),
                info.model.as_deref().unwrap_or("unknown"),
                info.endpoint.as_deref().unwrap_or("unknown"),
            ),
            None,
        )
    } else {
        timed_static_check(
            id,
            scope,
            DoctorCheckStatus::Warn,
            "role is not configured".to_string(),
            Some(fix.to_string()),
        )
    }
}

fn role_readiness_checks(model_status: &ModelStatus) -> Vec<DoctorCheck> {
    vec![
        role_readiness_check(
            "roles.embedder.reachable",
            "roles.embedder",
            &model_status.embedder,
        ),
        role_readiness_check(
            "roles.reranker.reachable",
            "roles.reranker",
            &model_status.reranker,
        ),
        role_readiness_check(
            "roles.expander.reachable",
            "roles.expander",
            &model_status.expander,
        ),
    ]
}

fn role_readiness_check(id: &str, scope: &str, info: &ModelInfo) -> DoctorCheck {
    if !info.configured {
        return timed_static_check(
            id,
            scope,
            DoctorCheckStatus::Warn,
            "skipped because role is not configured".to_string(),
            None,
        );
    }

    if info.ready {
        return timed_static_check(
            id,
            scope,
            DoctorCheckStatus::Pass,
            "provider endpoint is reachable".to_string(),
            None,
        );
    }

    timed_static_check(
        id,
        scope,
        DoctorCheckStatus::Fail,
        info.issue
            .clone()
            .unwrap_or_else(|| "provider endpoint is not ready".to_string()),
        Some(
            "Start the provider server or fix the provider endpoint/auth configuration."
                .to_string(),
        ),
    )
}

fn check_cache_dir(cache_dir: &Path) -> DoctorCheck {
    if !cache_dir.exists() {
        return timed_static_check(
            "cache.dir_exists",
            "storage",
            DoctorCheckStatus::Warn,
            format!(
                "cache directory does not exist yet: {}",
                cache_dir.display()
            ),
            Some("This is expected before the first successful indexing run.".to_string()),
        );
    }

    check_directory_writable("cache.dir_writable", "storage", cache_dir)
}

fn check_sqlite_readable(cache_dir: &Path) -> DoctorCheck {
    let db_path = cache_dir.join(SQLITE_FILE_NAME);
    if !db_path.exists() {
        return timed_static_check(
            "storage.sqlite_readable",
            "storage",
            DoctorCheckStatus::Warn,
            format!("index database does not exist yet: {}", db_path.display()),
            Some("Run `kbolt update` after adding a collection to build the index.".to_string()),
        );
    }

    timed_unit_check(
        "storage.sqlite_readable",
        "storage",
        Some("Check filesystem permissions or back up and rebuild the index if the DB is corrupt."),
        || {
            let conn = Connection::open_with_flags(&db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            let _: i64 = conn.query_row("SELECT 1", [], |row| row.get(0))?;
            Ok(())
        },
    )
}

fn check_search_indexes_readable(cache_dir: &Path) -> DoctorCheck {
    let spaces_dir = cache_dir.join(SPACES_DIR_NAME);
    if !spaces_dir.exists() {
        return timed_static_check(
            "storage.search_indexes_readable",
            "storage",
            DoctorCheckStatus::Warn,
            format!(
                "search index directory does not exist yet: {}",
                spaces_dir.display()
            ),
            Some(
                "Run `kbolt update` after adding a collection to build search indexes.".to_string(),
            ),
        );
    }

    let has_space_dirs = match fs::read_dir(&spaces_dir) {
        Ok(entries) => entries
            .filter_map(std::result::Result::ok)
            .any(|entry| entry.path().is_dir()),
        Err(err) => {
            return timed_static_check(
                "storage.search_indexes_readable",
                "storage",
                DoctorCheckStatus::Fail,
                err.to_string(),
                Some("Check filesystem permissions on the cache spaces directory.".to_string()),
            )
        }
    };

    if !has_space_dirs {
        return timed_static_check(
            "storage.search_indexes_readable",
            "storage",
            DoctorCheckStatus::Warn,
            format!("search index directory is empty: {}", spaces_dir.display()),
            Some(
                "Run `kbolt update` after adding a collection to build search indexes.".to_string(),
            ),
        );
    }

    timed_unit_check(
        "storage.search_indexes_readable",
        "storage",
        Some("Back up the cache directory and rebuild the affected space indexes if they are corrupt."),
        || {
            for entry in fs::read_dir(&spaces_dir)? {
                let space_dir = entry?.path();
                if !space_dir.is_dir() {
                    continue;
                }

                check_tantivy_space_index(&space_dir.join(TANTIVY_DIR_NAME))?;
                check_usearch_space_index(&space_dir.join(USEARCH_FILE_NAME))?;
            }

            Ok(())
        },
    )
}

fn check_tantivy_space_index(index_dir: &Path) -> crate::Result<()> {
    if !index_dir.exists() {
        return Ok(());
    }
    Index::open_in_dir(index_dir)?;
    Ok(())
}

fn check_usearch_space_index(index_path: &Path) -> crate::Result<()> {
    let file_size = fs::metadata(index_path).map(|meta| meta.len()).unwrap_or(0);
    if file_size == 0 {
        return Ok(());
    }

    let path = index_path
        .to_str()
        .ok_or_else(|| KboltError::Internal("invalid usearch path encoding".to_string()))?;
    let index = usearch::Index::new(&IndexOptions {
        dimensions: 256,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: 16,
        expansion_add: 200,
        expansion_search: 100,
        ..IndexOptions::default()
    })
    .map_err(|err| KboltError::Internal(format!("usearch init failed: {err}")))?;
    index
        .load(path)
        .map_err(|err| KboltError::Internal(format!("usearch load failed: {err}")))?;
    if index.size() == 0 {
        return Err(KboltError::Internal(format!(
            "usearch index file is non-empty but loaded zero vectors: {}",
            index_path.display()
        ))
        .into());
    }
    Ok(())
}

fn check_directory_writable(id: &str, scope: &str, dir: &Path) -> DoctorCheck {
    if !dir.exists() {
        return timed_static_check(
            id,
            scope,
            DoctorCheckStatus::Warn,
            format!("directory does not exist yet: {}", dir.display()),
            Some(
                "Create the directory or run the setup/indexing command that owns it.".to_string(),
            ),
        );
    }

    timed_unit_check(
        id,
        scope,
        Some("Fix directory permissions so kbolt can write state files."),
        || {
            let probe = writable_probe_path(dir);
            fs::write(&probe, b"doctor")?;
            fs::remove_file(&probe)?;
            Ok(())
        },
    )
}

fn writable_probe_path(dir: &Path) -> PathBuf {
    dir.join(format!(
        "{WRITE_CHECK_FILE_NAME}-{}.tmp",
        std::process::id()
    ))
}

fn timed_unit_check<F>(id: &str, scope: &str, fix: Option<&str>, check: F) -> DoctorCheck
where
    F: FnOnce() -> crate::Result<()>,
{
    let started = Instant::now();
    let outcome = check();
    let elapsed_ms = elapsed_ms(started);

    match outcome {
        Ok(()) => DoctorCheck {
            id: id.to_string(),
            scope: scope.to_string(),
            status: DoctorCheckStatus::Pass,
            elapsed_ms,
            message: "ok".to_string(),
            fix: None,
        },
        Err(err) => DoctorCheck {
            id: id.to_string(),
            scope: scope.to_string(),
            status: DoctorCheckStatus::Fail,
            elapsed_ms,
            message: err.to_string(),
            fix: fix.map(ToString::to_string),
        },
    }
}

fn timed_result_check<T, F>(
    id: &str,
    scope: &str,
    fix: Option<&str>,
    check: F,
) -> (DoctorCheck, crate::Result<T>)
where
    F: FnOnce() -> crate::Result<T>,
{
    let started = Instant::now();
    let outcome = check();
    let elapsed_ms = elapsed_ms(started);

    let report_check = match &outcome {
        Ok(_) => DoctorCheck {
            id: id.to_string(),
            scope: scope.to_string(),
            status: DoctorCheckStatus::Pass,
            elapsed_ms,
            message: "ok".to_string(),
            fix: None,
        },
        Err(err) => DoctorCheck {
            id: id.to_string(),
            scope: scope.to_string(),
            status: DoctorCheckStatus::Fail,
            elapsed_ms,
            message: err.to_string(),
            fix: fix.map(ToString::to_string),
        },
    };

    (report_check, outcome)
}

fn timed_static_check(
    id: &str,
    scope: &str,
    status: DoctorCheckStatus,
    message: String,
    fix: Option<String>,
) -> DoctorCheck {
    let started = Instant::now();
    DoctorCheck {
        id: id.to_string(),
        scope: scope.to_string(),
        status,
        elapsed_ms: elapsed_ms(started),
        message,
        fix,
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn is_not_configured(roles: &RoleBindingsConfig) -> bool {
    roles.embedder.is_none() && roles.reranker.is_none() && roles.expander.is_none()
}

fn finalize_report(
    setup_status: DoctorSetupStatus,
    config_file: Option<PathBuf>,
    config_dir: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    checks: Vec<DoctorCheck>,
) -> DoctorReport {
    let ready = setup_status == DoctorSetupStatus::Configured
        && checks
            .iter()
            .all(|check| check.status != DoctorCheckStatus::Fail);

    DoctorReport {
        setup_status,
        config_file,
        config_dir,
        cache_dir,
        ready,
        checks,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Mutex, OnceLock};
    use std::thread;

    use tempfile::tempdir;

    use super::*;
    use crate::config::{
        ChunkingConfig, Config, EmbedderRoleConfig, ProviderOperation, ProviderProfileConfig,
        RankingConfig, ReapingConfig,
    };

    struct EnvRestore {
        home: Option<OsString>,
        config_home: Option<OsString>,
        cache_home: Option<OsString>,
    }

    impl EnvRestore {
        fn capture() -> Self {
            Self {
                home: std::env::var_os("HOME"),
                config_home: std::env::var_os("XDG_CONFIG_HOME"),
                cache_home: std::env::var_os("XDG_CACHE_HOME"),
            }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match &self.home {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            match &self.config_home {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
            match &self.cache_home {
                Some(value) => std::env::set_var("XDG_CACHE_HOME", value),
                None => std::env::remove_var("XDG_CACHE_HOME"),
            }
        }
    }

    fn with_isolated_xdg_dirs(run: impl FnOnce()) {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let _restore = EnvRestore::capture();
        let tmp = tempdir().expect("create env tempdir");
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("XDG_CONFIG_HOME", tmp.path().join("config-home"));
        std::env::set_var("XDG_CACHE_HOME", tmp.path().join("cache-home"));

        run();
    }

    #[test]
    fn run_reports_missing_config_without_creating_files() {
        with_isolated_xdg_dirs(|| {
            let tmp = tempdir().expect("create tempdir");
            let config_dir = tmp.path().join("config");

            let report = run(Some(&config_dir));

            assert_eq!(report.setup_status, DoctorSetupStatus::ConfigMissing);
            assert!(!report.ready);
            assert!(!config_dir.exists());
            assert!(report
                .checks
                .iter()
                .any(|check| check.id == "config.file_exists"
                    && check.status == DoctorCheckStatus::Fail));
        });
    }

    #[test]
    fn run_reports_invalid_config_file() {
        with_isolated_xdg_dirs(|| {
            let tmp = tempdir().expect("create tempdir");
            let config_dir = tmp.path().join("config");
            fs::create_dir_all(&config_dir).expect("create config dir");
            fs::write(config_dir.join("index.toml"), "providers = [").expect("write config");

            let report = run(Some(&config_dir));

            assert_eq!(report.setup_status, DoctorSetupStatus::ConfigInvalid);
            assert!(!report.ready);
            assert!(report
                .checks
                .iter()
                .any(|check| check.id == "config.file_parses"
                    && check.status == DoctorCheckStatus::Fail));
        });
    }

    #[test]
    fn run_reports_not_configured_when_roles_are_empty() {
        with_isolated_xdg_dirs(|| {
            let tmp = tempdir().expect("create tempdir");
            let config_dir = tmp.path().join("config");
            let cache_dir = tmp.path().join("cache");
            let config = Config {
                config_dir: config_dir.clone(),
                cache_dir,
                default_space: None,
                providers: HashMap::new(),
                roles: RoleBindingsConfig::default(),
                reaping: ReapingConfig { days: 7 },
                chunking: ChunkingConfig::default(),
                ranking: RankingConfig::default(),
            };
            config::save(&config).expect("save config");

            let report = run(Some(&config_dir));

            assert_eq!(report.setup_status, DoctorSetupStatus::NotConfigured);
            assert!(!report.ready);
            assert!(report
                .checks
                .iter()
                .any(|check| check.id == "config.roles_bound"
                    && check.status == DoctorCheckStatus::Fail));
        });
    }

    #[test]
    fn run_smoke_checks_llama_embedder_and_tokenizer() {
        with_isolated_xdg_dirs(|| {
            let endpoint = serve_llama_embedder_and_tokenize();
            let tmp = tempdir().expect("create tempdir");
            let config_dir = tmp.path().join("config");
            let cache_dir = tmp.path().join("cache");
            let config = Config {
                config_dir: config_dir.clone(),
                cache_dir,
                default_space: None,
                providers: HashMap::from([(
                    "local_embed".to_string(),
                    ProviderProfileConfig::LlamaCppServer {
                        operation: ProviderOperation::Embedding,
                        base_url: endpoint,
                        model: "embeddinggemma".to_string(),
                        timeout_ms: 5_000,
                        max_retries: 0,
                    },
                )]),
                roles: RoleBindingsConfig {
                    embedder: Some(EmbedderRoleConfig {
                        provider: "local_embed".to_string(),
                        batch_size: 16,
                    }),
                    reranker: None,
                    expander: None,
                },
                reaping: ReapingConfig { days: 7 },
                chunking: ChunkingConfig::default(),
                ranking: RankingConfig::default(),
            };
            config::save(&config).expect("save config");

            let report = run(Some(&config_dir));

            assert_eq!(report.setup_status, DoctorSetupStatus::Configured);
            assert!(report.ready, "unexpected report: {report:#?}");
            assert!(report
                .checks
                .iter()
                .any(|check| check.id == "roles.embedder.embedding_smoke"
                    && check.status == DoctorCheckStatus::Pass));
            assert!(report
                .checks
                .iter()
                .any(|check| check.id == "roles.embedder.tokenize_smoke"
                    && check.status == DoctorCheckStatus::Pass));
        });
    }

    #[test]
    fn run_reports_corrupt_search_indexes_as_not_ready() {
        with_isolated_xdg_dirs(|| {
            let endpoint = serve_llama_embedder_and_tokenize();
            let tmp = tempdir().expect("create tempdir");
            let config_dir = tmp.path().join("config");
            let cache_dir = tmp.path().join("cache");
            let config = Config {
                config_dir: config_dir.clone(),
                cache_dir: cache_dir.clone(),
                default_space: None,
                providers: HashMap::from([(
                    "local_embed".to_string(),
                    ProviderProfileConfig::LlamaCppServer {
                        operation: ProviderOperation::Embedding,
                        base_url: endpoint,
                        model: "embeddinggemma".to_string(),
                        timeout_ms: 5_000,
                        max_retries: 0,
                    },
                )]),
                roles: RoleBindingsConfig {
                    embedder: Some(EmbedderRoleConfig {
                        provider: "local_embed".to_string(),
                        batch_size: 16,
                    }),
                    reranker: None,
                    expander: None,
                },
                reaping: ReapingConfig { days: 7 },
                chunking: ChunkingConfig::default(),
                ranking: RankingConfig::default(),
            };
            config::save(&config).expect("save config");

            let effective_cache_dir = config::load_existing(Some(&config_dir))
                .expect("load saved config")
                .cache_dir;
            let bad_space_dir = effective_cache_dir.join("spaces").join("default");
            fs::create_dir_all(&bad_space_dir).expect("create bad space dir");
            fs::write(bad_space_dir.join("tantivy"), "not an index")
                .expect("write bad tantivy artifact");

            let report = run(Some(&config_dir));

            assert_eq!(report.setup_status, DoctorSetupStatus::Configured);
            assert!(!report.ready, "unexpected report: {report:#?}");
            assert!(report.checks.iter().any(|check| {
                check.id == "storage.search_indexes_readable"
                    && check.status == DoctorCheckStatus::Fail
            }));
        });
    }

    fn serve_llama_embedder_and_tokenize() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("read server addr");
        thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().expect("accept request");
                let request_path = read_request_path(&mut stream);
                let body = if request_path == "/" {
                    r#"{"status":"ok"}"#
                } else if request_path == "/v1/embeddings" {
                    r#"{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}"#
                } else if request_path == "/tokenize" {
                    r#"{"tokens":[1,2,3]}"#
                } else {
                    r#"{"error":"unexpected path"}"#
                };
                let status_line = if body.contains("unexpected path") {
                    "HTTP/1.1 404 Not Found"
                } else {
                    "HTTP/1.1 200 OK"
                };
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write response");
            }
        });
        format!("http://{addr}")
    }

    fn read_request_path(stream: &mut std::net::TcpStream) -> String {
        let mut raw = [0_u8; 8192];
        let mut received = Vec::new();
        let mut expected_len = None;

        while expected_len.map_or(true, |len| received.len() < len) {
            let read_len = stream.read(&mut raw).expect("read request");
            if read_len == 0 {
                break;
            }
            received.extend_from_slice(&raw[..read_len]);

            if expected_len.is_none() {
                if let Some(header_end) = received
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|offset| offset + 4)
                {
                    let headers =
                        String::from_utf8_lossy(&received[..header_end]).to_ascii_lowercase();
                    let mut content_length = 0usize;
                    for line in headers.lines() {
                        if let Some(value) = line.strip_prefix("content-length:") {
                            content_length = value.trim().parse::<usize>().unwrap_or(0);
                            break;
                        }
                    }
                    expected_len = Some(header_end.saturating_add(content_length));
                }
            }
        }

        let request_line_end = received
            .windows(2)
            .position(|window| window == b"\r\n")
            .unwrap_or(received.len());
        let request_line = String::from_utf8_lossy(&received[..request_line_end]);
        request_line
            .split_whitespace()
            .nth(1)
            .unwrap_or("")
            .to_string()
    }
}
