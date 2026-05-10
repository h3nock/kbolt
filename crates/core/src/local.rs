use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use kbolt_types::{KboltError, LocalAction, LocalReport, LocalServiceReport};

use crate::config::{
    self, Config, EmbedderRoleConfig, ExpanderRoleConfig, ProviderOperation, ProviderProfileConfig,
    RerankerRoleConfig,
};
use crate::models::{self, build_inference_clients_without_managed_recovery, EmbeddingInputKind};
use crate::Result;

const APP_NAME: &str = "kbolt";
const LOCALHOST: &str = "127.0.0.1";
const LLAMA_SERVER_BREW_HINT: &str = "brew install llama.cpp";
const LLAMA_SERVER_LOG_VERBOSITY: &str = "1";
const MODEL_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(15);
const STOP_WAIT_TIMEOUT: Duration = Duration::from_secs(3);

const MANAGED_EMBED_PROVIDER: &str = "kbolt_local_embed";
const MANAGED_RERANK_PROVIDER: &str = "kbolt_local_rerank";
const MANAGED_EXPAND_PROVIDER: &str = "kbolt_local_expand";

const EMBEDDER_MODEL_LABEL: &str = "embeddinggemma";
const RERANKER_MODEL_LABEL: &str = "qwen3-reranker";
const EXPANDER_MODEL_LABEL: &str = "qwen3-1.7b";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManagedRole {
    Embedder,
    Reranker,
    Expander,
}

#[derive(Debug, Clone, Copy)]
struct ManagedServiceSpec {
    role: ManagedRole,
    name: &'static str,
    provider_name: &'static str,
    model_label: &'static str,
    model_repo: &'static str,
    model_file: &'static str,
    preferred_port: u16,
}

const EMBEDDER_SPEC: ManagedServiceSpec = ManagedServiceSpec {
    role: ManagedRole::Embedder,
    name: "embedder",
    provider_name: MANAGED_EMBED_PROVIDER,
    model_label: EMBEDDER_MODEL_LABEL,
    model_repo: "ggml-org/embeddinggemma-300M-GGUF",
    model_file: "embeddinggemma-300M-Q8_0.gguf",
    preferred_port: 8101,
};

const RERANKER_SPEC: ManagedServiceSpec = ManagedServiceSpec {
    role: ManagedRole::Reranker,
    name: "reranker",
    provider_name: MANAGED_RERANK_PROVIDER,
    model_label: RERANKER_MODEL_LABEL,
    model_repo: "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
    model_file: "qwen3-reranker-0.6b-q8_0.gguf",
    preferred_port: 8102,
};

const EXPANDER_SPEC: ManagedServiceSpec = ManagedServiceSpec {
    role: ManagedRole::Expander,
    name: "expander",
    provider_name: MANAGED_EXPAND_PROVIDER,
    model_label: EXPANDER_MODEL_LABEL,
    model_repo: "Qwen/Qwen3-1.7B-GGUF",
    model_file: "Qwen3-1.7B-Q8_0.gguf",
    preferred_port: 8103,
};

static EMBEDDER_RECOVERY_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static RERANKER_RECOVERY_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static EXPANDER_RECOVERY_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn is_managed_provider_name(provider_name: &str) -> bool {
    managed_service_spec(provider_name).is_some()
}

pub(crate) fn managed_provider_label(provider_name: &str) -> Option<&'static str> {
    managed_service_spec(provider_name).map(|spec| spec.name)
}

pub(crate) fn managed_embedder_model_path(
    cache_dir: &Path,
    provider_name: &str,
) -> Option<PathBuf> {
    managed_service_spec(provider_name)
        .filter(|spec| spec.role == ManagedRole::Embedder)
        .map(|spec| managed_model_path(cache_dir, spec))
}

pub(crate) fn restart_managed_service(config: &Config, provider_name: &str) -> Result<()> {
    let spec = managed_service_spec(provider_name).ok_or_else(|| {
        KboltError::Config(format!(
            "automatic recovery is only supported for managed local providers; got '{provider_name}'"
        ))
    })?;
    let _guard = managed_service_recovery_lock(spec).lock().map_err(|_| {
        KboltError::Internal(format!("managed {} recovery lock was poisoned", spec.name))
    })?;

    prepare_runtime_dirs(&config.cache_dir)?;
    let llama_server_path = find_llama_server()?;
    let port = provider_port(config, spec.provider_name)?;
    let service = start_or_reuse_service(config, &llama_server_path, spec, port)?;
    if service.ready {
        return Ok(());
    }

    Err(KboltError::Inference(format!(
        "managed {} on {} is running but not ready; run `kbolt local status`",
        spec.name, service.endpoint
    ))
    .into())
}

pub fn setup_local(config_path: Option<&Path>) -> Result<LocalReport> {
    let llama_server_path = find_llama_server()?;
    let (mut config, mut notes) = load_setup_config(config_path)?;
    let mut reserved_ports = HashSet::new();
    let mut started = Vec::new();

    prepare_runtime_dirs(&config.cache_dir)?;
    let embedder = match prepare_service(
        &config,
        &llama_server_path,
        &EMBEDDER_SPEC,
        &mut reserved_ports,
        &mut notes,
        &mut started,
    ) {
        Ok(service) => service,
        Err(err) => {
            stop_started_children(&started);
            return Err(err);
        }
    };
    let reranker = match prepare_service(
        &config,
        &llama_server_path,
        &RERANKER_SPEC,
        &mut reserved_ports,
        &mut notes,
        &mut started,
    ) {
        Ok(service) => service,
        Err(err) => {
            stop_started_children(&started);
            return Err(err);
        }
    };

    apply_managed_service_config(&mut config, &EMBEDDER_SPEC, embedder.port);
    apply_managed_service_config(&mut config, &RERANKER_SPEC, reranker.port);
    if config.default_space.is_none() {
        config.default_space = Some("default".to_string());
    }

    if let Err(err) = config::save(&config) {
        stop_started_children(&started);
        return Err(err);
    }

    build_report(
        &config,
        Some(llama_server_path),
        LocalAction::Setup,
        notes,
        &[EMBEDDER_SPEC, RERANKER_SPEC, EXPANDER_SPEC],
    )
}

fn load_setup_config(config_path: Option<&Path>) -> Result<(Config, Vec<String>)> {
    let config_file = config::resolve_config_file_path(config_path)?;
    match config::load(config_path) {
        Ok(config) => Ok((config, Vec::new())),
        Err(err @ crate::error::CoreError::Domain(KboltError::Config(_)))
            if config_file.exists() =>
        {
            if !looks_like_legacy_config(&config_file)? {
                return Err(err);
            }
            let backup_path = backup_invalid_config(&config_file)?;
            let config = config::load(config_path)?;
            Ok((
                config,
                vec![format!(
                    "moved incompatible legacy config to {} and created a fresh current config",
                    backup_path.display()
                )],
            ))
        }
        Err(err) => Err(err),
    }
}

fn looks_like_legacy_config(config_file: &Path) -> Result<bool> {
    let raw = fs::read_to_string(config_file)?;
    Ok(raw
        .lines()
        .map(str::trim)
        .any(|line| matches!(line, "[embeddings]" | "[models]")))
}

fn backup_invalid_config(config_file: &Path) -> Result<PathBuf> {
    let Some(file_name) = config_file.file_name().and_then(|name| name.to_str()) else {
        return Err(KboltError::Internal(format!(
            "invalid config path: {}",
            config_file.display()
        ))
        .into());
    };
    let parent = config_file.parent().ok_or_else(|| {
        KboltError::Internal(format!(
            "config file has no parent: {}",
            config_file.display()
        ))
    })?;

    for suffix in std::iter::once(".invalid.bak".to_string())
        .chain((1usize..).map(|index| format!(".invalid.{index}.bak")))
    {
        let candidate = parent.join(format!("{file_name}{suffix}"));
        if candidate.exists() {
            continue;
        }

        fs::rename(config_file, &candidate)?;
        return Ok(candidate);
    }

    unreachable!("backup suffix iterator is infinite");
}

pub fn enable_deep(config_path: Option<&Path>) -> Result<LocalReport> {
    let llama_server_path = find_llama_server()?;
    let mut config = config::load(config_path)?;
    ensure_managed_local_base_configured(&config)?;

    let mut reserved_ports = reserved_ports_from_config(&config);
    let mut notes = Vec::new();
    let mut started = Vec::new();
    let expander = match prepare_service(
        &config,
        &llama_server_path,
        &EXPANDER_SPEC,
        &mut reserved_ports,
        &mut notes,
        &mut started,
    ) {
        Ok(service) => service,
        Err(err) => {
            stop_started_children(&started);
            return Err(err);
        }
    };
    apply_managed_service_config(&mut config, &EXPANDER_SPEC, expander.port);

    if let Err(err) = config::save(&config) {
        stop_started_children(&started);
        return Err(err);
    }

    build_report(
        &config,
        Some(llama_server_path),
        LocalAction::EnableDeep,
        notes,
        &[EMBEDDER_SPEC, RERANKER_SPEC, EXPANDER_SPEC],
    )
}

pub fn start_local(config_path: Option<&Path>) -> Result<LocalReport> {
    let llama_server_path = find_llama_server()?;
    let config = config::load_existing(config_path)?;
    ensure_managed_local_base_configured(&config)?;
    prepare_runtime_dirs(&config.cache_dir)?;

    let mut notes = Vec::new();
    let specs = configured_specs(&config);
    for spec in &specs {
        let port = provider_port(&config, spec.provider_name)?;
        let service = start_or_reuse_service(&config, &llama_server_path, spec, port)?;
        if let Some(pid) = service.started_pid {
            let _ = pid;
            notes.push(format!("started {} on {}", spec.name, service.endpoint));
        } else if service.ready {
            notes.push(format!(
                "{} already ready on {}",
                spec.name, service.endpoint
            ));
        } else if service.running {
            notes.push(format!(
                "{} is already running on {} but is not managed by kbolt",
                spec.name, service.endpoint
            ));
        }
    }

    build_report(
        &config,
        Some(llama_server_path),
        LocalAction::Start,
        notes,
        &[EMBEDDER_SPEC, RERANKER_SPEC, EXPANDER_SPEC],
    )
}

pub fn stop_local(config_path: Option<&Path>) -> Result<LocalReport> {
    let config = config::load_existing(config_path)?;
    let mut notes = Vec::new();

    for spec in configured_specs(&config) {
        let pid_file = pid_file_path(&config.cache_dir, spec);
        let Some(pid) = read_pid(&pid_file)? else {
            notes.push(format!("{} is not managed by a kbolt pid file", spec.name));
            continue;
        };

        if !pid_is_alive(pid) {
            remove_pid_file(&pid_file)?;
            notes.push(format!("removed stale pid file for {}", spec.name));
            continue;
        }

        terminate_pid(pid)?;
        remove_pid_file(&pid_file)?;
        notes.push(format!("stopped {}", spec.name));
    }

    build_report(
        &config,
        find_llama_server_optional(),
        LocalAction::Stop,
        notes,
        &[EMBEDDER_SPEC, RERANKER_SPEC, EXPANDER_SPEC],
    )
}

pub fn local_status(config_path: Option<&Path>) -> Result<LocalReport> {
    let config_file = config::resolve_config_file_path(config_path)?;
    let cache_dir = default_cache_dir()?;
    if !config_file.exists() {
        return Ok(LocalReport {
            action: LocalAction::Status,
            config_file,
            cache_dir: cache_dir.clone(),
            llama_server_path: find_llama_server_optional(),
            ready: false,
            notes: vec!["kbolt is not set up yet; run `kbolt setup local`.".to_string()],
            services: vec![
                missing_service_report(&cache_dir, &EMBEDDER_SPEC),
                missing_service_report(&cache_dir, &RERANKER_SPEC),
                missing_service_report(&cache_dir, &EXPANDER_SPEC),
            ],
        });
    }

    let config = config::load_existing(config_path)?;
    build_report(
        &config,
        find_llama_server_optional(),
        LocalAction::Status,
        Vec::new(),
        &[EMBEDDER_SPEC, RERANKER_SPEC, EXPANDER_SPEC],
    )
}

#[derive(Debug)]
struct PreparedService {
    port: u16,
}

#[derive(Debug)]
struct RunningService {
    endpoint: String,
    running: bool,
    ready: bool,
    started_pid: Option<u32>,
}

fn prepare_service(
    config: &Config,
    llama_server_path: &Path,
    spec: &ManagedServiceSpec,
    reserved_ports: &mut HashSet<u16>,
    notes: &mut Vec<String>,
    started: &mut Vec<u32>,
) -> Result<PreparedService> {
    ensure_model_file(&config.cache_dir, spec)?;
    let port = select_port(config, spec, reserved_ports)?;
    reserved_ports.insert(port);
    let service = start_or_reuse_service(config, llama_server_path, spec, port)?;
    if let Some(pid) = service.started_pid {
        started.push(pid);
        notes.push(started_service_note(spec.name, &service.endpoint));
    } else if port != spec.preferred_port {
        notes.push(format!(
            "{} default port {} was unavailable; using {}",
            spec.name, spec.preferred_port, port
        ));
    }
    Ok(PreparedService { port })
}

fn started_service_note(name: &str, endpoint: &str) -> String {
    format!("started {name} on {endpoint}")
}

fn start_or_reuse_service(
    config: &Config,
    llama_server_path: &Path,
    spec: &ManagedServiceSpec,
    port: u16,
) -> Result<RunningService> {
    let endpoint = endpoint_for_port(port);
    let pid_file = pid_file_path(&config.cache_dir, spec);
    let model_path = ensure_model_file(&config.cache_dir, spec)?;

    if let Some(pid) = read_pid(&pid_file)? {
        if pid_is_alive(pid) {
            if probe_service(config, spec, port).is_ok() {
                return Ok(RunningService {
                    endpoint,
                    running: true,
                    ready: true,
                    started_pid: None,
                });
            }
        } else {
            remove_pid_file(&pid_file)?;
        }
    }

    if is_port_bound(port) {
        let ready = probe_service(config, spec, port).is_ok();
        return Ok(RunningService {
            endpoint,
            running: true,
            ready,
            started_pid: None,
        });
    }

    let log_file = log_file_path(&config.cache_dir, spec);
    let child = spawn_llama_server(llama_server_path, spec, &model_path, port, &log_file)?;
    write_pid(&pid_file, child.id())?;
    let pid = child.id();
    drop(child);

    let start = Instant::now();
    loop {
        if probe_service(config, spec, port).is_ok() {
            return Ok(RunningService {
                endpoint,
                running: true,
                ready: true,
                started_pid: Some(pid),
            });
        }

        if start.elapsed() >= MODEL_DOWNLOAD_TIMEOUT {
            let _ = terminate_pid(pid);
            let _ = remove_pid_file(&pid_file);
            return Err(KboltError::Inference(format!(
                "{} did not become ready on {} within {}s; check {}",
                spec.name,
                endpoint,
                MODEL_DOWNLOAD_TIMEOUT.as_secs(),
                log_file.display()
            ))
            .into());
        }

        thread::sleep(Duration::from_millis(250));
    }
}

fn build_report(
    config: &Config,
    llama_server_path: Option<PathBuf>,
    action: LocalAction,
    notes: Vec<String>,
    specs: &[ManagedServiceSpec],
) -> Result<LocalReport> {
    let mut services = Vec::new();
    let mut ready = true;

    for spec in specs {
        let provider = config.providers.get(spec.provider_name);
        let configured = role_uses_provider(config, spec.provider_name);
        let enabled = provider.is_some();
        let model_path = managed_model_path(&config.cache_dir, spec);
        let port = provider
            .map(|profile| provider_profile_port(profile))
            .transpose()?
            .unwrap_or(spec.preferred_port);
        let endpoint = endpoint_for_port(port);
        let pid_file = pid_file_path(&config.cache_dir, spec);
        let log_file = log_file_path(&config.cache_dir, spec);
        let pid = read_pid(&pid_file)?;
        let managed = pid.map(pid_is_alive).unwrap_or(false);
        let running = is_port_bound(port);
        let probe = if enabled {
            probe_service(config, spec, port).err()
        } else {
            None
        };
        let service_ready = enabled && probe.is_none();
        if enabled && !service_ready {
            ready = false;
        }

        services.push(LocalServiceReport {
            name: spec.name.to_string(),
            provider: spec.provider_name.to_string(),
            enabled,
            configured,
            managed,
            running,
            ready: service_ready,
            model: spec.model_label.to_string(),
            model_path,
            endpoint,
            port,
            pid: pid.filter(|value| pid_is_alive(*value)),
            pid_file,
            log_file,
            issue: if !enabled {
                Some("service is not configured".to_string())
            } else if service_ready {
                if running && !managed {
                    Some("service is reachable but not managed by kbolt".to_string())
                } else {
                    None
                }
            } else {
                Some(
                    probe
                        .map(|err| err.to_string())
                        .unwrap_or_else(|| "service is not ready".to_string()),
                )
            },
        });
    }

    Ok(LocalReport {
        action,
        config_file: config.config_dir.join("index.toml"),
        cache_dir: config.cache_dir.clone(),
        llama_server_path,
        ready,
        notes,
        services,
    })
}

fn missing_service_report(cache_dir: &Path, spec: &ManagedServiceSpec) -> LocalServiceReport {
    LocalServiceReport {
        name: spec.name.to_string(),
        provider: spec.provider_name.to_string(),
        enabled: false,
        configured: false,
        managed: false,
        running: false,
        ready: false,
        model: spec.model_label.to_string(),
        model_path: managed_model_path(cache_dir, spec),
        endpoint: endpoint_for_port(spec.preferred_port),
        port: spec.preferred_port,
        pid: None,
        pid_file: pid_file_path(cache_dir, spec),
        log_file: log_file_path(cache_dir, spec),
        issue: Some("service is not configured".to_string()),
    }
}

fn prepare_runtime_dirs(cache_dir: &Path) -> Result<()> {
    fs::create_dir_all(cache_dir.join("models"))?;
    fs::create_dir_all(cache_dir.join("run"))?;
    fs::create_dir_all(cache_dir.join("logs"))?;
    Ok(())
}

fn ensure_managed_local_base_configured(config: &Config) -> Result<()> {
    if !config.providers.contains_key(MANAGED_EMBED_PROVIDER)
        || !config.providers.contains_key(MANAGED_RERANK_PROVIDER)
    {
        return Err(KboltError::Config(
            "managed local setup is not configured; run `kbolt setup local` first".to_string(),
        )
        .into());
    }
    Ok(())
}

fn managed_service_spec(provider_name: &str) -> Option<&'static ManagedServiceSpec> {
    match provider_name {
        MANAGED_EMBED_PROVIDER => Some(&EMBEDDER_SPEC),
        MANAGED_RERANK_PROVIDER => Some(&RERANKER_SPEC),
        MANAGED_EXPAND_PROVIDER => Some(&EXPANDER_SPEC),
        _ => None,
    }
}

fn managed_service_recovery_lock(spec: &ManagedServiceSpec) -> &'static Mutex<()> {
    match spec.role {
        ManagedRole::Embedder => EMBEDDER_RECOVERY_LOCK.get_or_init(|| Mutex::new(())),
        ManagedRole::Reranker => RERANKER_RECOVERY_LOCK.get_or_init(|| Mutex::new(())),
        ManagedRole::Expander => EXPANDER_RECOVERY_LOCK.get_or_init(|| Mutex::new(())),
    }
}

fn configured_specs(config: &Config) -> Vec<&'static ManagedServiceSpec> {
    let mut specs = Vec::new();
    if config.providers.contains_key(MANAGED_EMBED_PROVIDER) {
        specs.push(&EMBEDDER_SPEC);
    }
    if config.providers.contains_key(MANAGED_RERANK_PROVIDER) {
        specs.push(&RERANKER_SPEC);
    }
    if config.providers.contains_key(MANAGED_EXPAND_PROVIDER) {
        specs.push(&EXPANDER_SPEC);
    }
    specs
}

fn role_uses_provider(config: &Config, provider_name: &str) -> bool {
    config
        .roles
        .embedder
        .as_ref()
        .map(|role| role.provider == provider_name)
        .unwrap_or(false)
        || config
            .roles
            .reranker
            .as_ref()
            .map(|role| role.provider == provider_name)
            .unwrap_or(false)
        || config
            .roles
            .expander
            .as_ref()
            .map(|role| role.provider == provider_name)
            .unwrap_or(false)
}

fn apply_managed_service_config(config: &mut Config, spec: &ManagedServiceSpec, port: u16) {
    config.providers.insert(
        spec.provider_name.to_string(),
        ProviderProfileConfig::LlamaCppServer {
            operation: match spec.role {
                ManagedRole::Embedder => ProviderOperation::Embedding,
                ManagedRole::Reranker => ProviderOperation::Reranking,
                ManagedRole::Expander => ProviderOperation::ChatCompletion,
            },
            base_url: endpoint_for_port(port),
            model: spec.model_label.to_string(),
            timeout_ms: 30_000,
            max_retries: 2,
        },
    );

    match spec.role {
        ManagedRole::Embedder => {
            config.roles.embedder = Some(EmbedderRoleConfig {
                provider: spec.provider_name.to_string(),
                batch_size: 32,
            });
        }
        ManagedRole::Reranker => {
            config.roles.reranker = Some(RerankerRoleConfig {
                provider: spec.provider_name.to_string(),
            });
        }
        ManagedRole::Expander => {
            config.roles.expander = Some(ExpanderRoleConfig {
                provider: spec.provider_name.to_string(),
                max_tokens: 600,
                sampling: crate::config::ExpanderSamplingConfig::default(),
            });
        }
    }
}

fn ensure_model_file(cache_dir: &Path, spec: &ManagedServiceSpec) -> Result<PathBuf> {
    let model_path = managed_model_path(cache_dir, spec);
    if model_path.is_file() {
        return Ok(model_path);
    }

    let download_dir = cache_dir.join("models").join(spec.name);
    fs::create_dir_all(&download_dir)?;

    let api = ApiBuilder::new()
        .with_cache_dir(download_dir.clone())
        .build()
        .map_err(|err| KboltError::ModelDownload(format!("{}: {err}", spec.model_repo)))?;
    let repo = api.repo(Repo::new(spec.model_repo.to_string(), RepoType::Model));
    let downloaded_path = repo
        .get(spec.model_file)
        .map_err(|err| KboltError::ModelDownload(format!("{}: {err}", spec.model_repo)))?;

    if downloaded_path != model_path {
        fs::copy(&downloaded_path, &model_path)?;
    }

    Ok(model_path)
}

fn managed_model_path(cache_dir: &Path, spec: &ManagedServiceSpec) -> PathBuf {
    cache_dir
        .join("models")
        .join(spec.name)
        .join(spec.model_file)
}

fn pid_file_path(cache_dir: &Path, spec: &ManagedServiceSpec) -> PathBuf {
    cache_dir.join("run").join(format!("{}.pid", spec.name))
}

fn log_file_path(cache_dir: &Path, spec: &ManagedServiceSpec) -> PathBuf {
    cache_dir.join("logs").join(format!("{}.log", spec.name))
}

fn endpoint_for_port(port: u16) -> String {
    format!("http://{LOCALHOST}:{port}")
}

fn default_cache_dir() -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .ok_or_else(|| KboltError::Config("unable to determine user cache directory".into()))?;
    Ok(base.join(APP_NAME))
}

fn select_port(
    config: &Config,
    spec: &ManagedServiceSpec,
    reserved_ports: &HashSet<u16>,
) -> Result<u16> {
    let preferred = config
        .providers
        .get(spec.provider_name)
        .map(provider_profile_port)
        .transpose()?
        .unwrap_or(spec.preferred_port);

    if port_candidate_usable(config, spec, preferred, reserved_ports)? {
        return Ok(preferred);
    }

    for port in preferred..(preferred + 20) {
        if port_candidate_usable(config, spec, port, reserved_ports)? {
            return Ok(port);
        }
    }

    Err(KboltError::Config(format!(
        "no free local port found for {} near {}",
        spec.name, preferred
    ))
    .into())
}

fn port_candidate_usable(
    config: &Config,
    spec: &ManagedServiceSpec,
    port: u16,
    reserved_ports: &HashSet<u16>,
) -> Result<bool> {
    if reserved_ports.contains(&port) {
        return Ok(false);
    }

    let pid_file = pid_file_path(&config.cache_dir, spec);
    if let Some(pid) = read_pid(&pid_file)? {
        if pid_is_alive(pid) && provider_port_matches(config, spec.provider_name, port)? {
            return Ok(true);
        }
    }

    Ok(!is_port_bound(port))
}

fn provider_port_matches(config: &Config, provider_name: &str, port: u16) -> Result<bool> {
    let Some(profile) = config.providers.get(provider_name) else {
        return Ok(false);
    };
    Ok(provider_profile_port(profile)? == port)
}

fn reserved_ports_from_config(config: &Config) -> HashSet<u16> {
    let mut ports = HashSet::new();
    for provider_name in [
        MANAGED_EMBED_PROVIDER,
        MANAGED_RERANK_PROVIDER,
        MANAGED_EXPAND_PROVIDER,
    ] {
        if let Some(profile) = config.providers.get(provider_name) {
            if let Ok(port) = provider_profile_port(profile) {
                ports.insert(port);
            }
        }
    }
    ports
}

fn provider_port(config: &Config, provider_name: &str) -> Result<u16> {
    let profile = config.providers.get(provider_name).ok_or_else(|| {
        KboltError::Config(format!("missing managed provider config: {provider_name}"))
    })?;
    provider_profile_port(profile)
}

fn provider_profile_port(profile: &ProviderProfileConfig) -> Result<u16> {
    let base_url = profile.base_url();
    let port = base_url
        .rsplit(':')
        .next()
        .and_then(|raw| raw.parse::<u16>().ok())
        .ok_or_else(|| KboltError::Config(format!("invalid managed local base_url: {base_url}")))?;
    Ok(port)
}

fn open_managed_service_log(log_file: &Path) -> Result<File> {
    if let Some(parent) = log_file.parent() {
        fs::create_dir_all(parent)?;
    }
    File::options()
        .create(true)
        .write(true)
        .truncate(true)
        .open(log_file)
        .map_err(Into::into)
}

fn configure_llama_server_command(
    command: &mut Command,
    llama_server_path: &Path,
    spec: &ManagedServiceSpec,
    model_path: &Path,
    port: u16,
) {
    command
        .arg(llama_server_path)
        .arg("-m")
        .arg(model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--log-verbosity")
        .arg(LLAMA_SERVER_LOG_VERBOSITY);

    match spec.role {
        ManagedRole::Embedder => {
            command
                .arg("--embedding")
                .arg("--pooling")
                .arg("mean")
                .arg("-ngl")
                .arg("99")
                .arg("-c")
                .arg("2048")
                .arg("-ub")
                .arg("2048");
        }
        ManagedRole::Reranker => {
            command
                .arg("--reranking")
                .arg("--pooling")
                .arg("rank")
                .arg("-ngl")
                .arg("99")
                .arg("-np")
                .arg("4")
                .arg("-c")
                .arg("8192")
                .arg("-ub")
                .arg("2048");
        }
        ManagedRole::Expander => {
            command.arg("-ngl").arg("99").arg("-c").arg("2048");
        }
    }
}

fn spawn_llama_server(
    llama_server_path: &Path,
    spec: &ManagedServiceSpec,
    model_path: &Path,
    port: u16,
    log_file: &Path,
) -> Result<std::process::Child> {
    let stdout = open_managed_service_log(log_file)?;
    let stderr = stdout.try_clone()?;

    let mut command = if Path::new("/usr/bin/nohup").is_file() {
        Command::new("/usr/bin/nohup")
    } else {
        Command::new("nohup")
    };
    configure_llama_server_command(&mut command, llama_server_path, spec, model_path, port);

    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|err| {
            KboltError::Inference(format!(
                "failed to start {} via {}: {err}",
                spec.name,
                llama_server_path.display()
            ))
        })
        .map_err(Into::into)
}

fn probe_service(config: &Config, spec: &ManagedServiceSpec, port: u16) -> Result<()> {
    let mut probe_config = config.clone();
    probe_config.providers = HashMap::new();
    probe_config.roles.embedder = None;
    probe_config.roles.reranker = None;
    probe_config.roles.expander = None;
    apply_managed_service_config(&mut probe_config, spec, port);

    let status = models::status(&probe_config)?;
    let info = match spec.role {
        ManagedRole::Embedder => &status.embedder,
        ManagedRole::Reranker => &status.reranker,
        ManagedRole::Expander => &status.expander,
    };
    if !info.ready {
        return Err(KboltError::Inference(
            info.issue
                .clone()
                .unwrap_or_else(|| format!("{} is not ready", spec.name)),
        )
        .into());
    }

    let clients = build_inference_clients_without_managed_recovery(&probe_config)?;
    match spec.role {
        ManagedRole::Embedder => {
            let embedder = clients.embedder.as_ref().ok_or_else(|| {
                KboltError::Inference("managed embedder client was not built".to_string())
            })?;
            let vectors = embedder.embed_batch(
                EmbeddingInputKind::Query,
                &["kbolt local probe".to_string()],
            )?;
            if vectors.len() != 1 || vectors[0].is_empty() {
                return Err(KboltError::Inference(
                    "managed embedder smoke returned an invalid embedding".to_string(),
                )
                .into());
            }
            let tokenizer = clients.embedding_tokenizer.as_ref().ok_or_else(|| {
                KboltError::Inference(
                    "managed embedder tokenizer runtime was not built".to_string(),
                )
            })?;
            let counts = tokenizer.count_embedding_tokens_batch(&["kbolt local probe"])?;
            let tokens = counts.into_iter().next().ok_or_else(|| {
                KboltError::Inference(
                    "managed embedder tokenize smoke returned no token count".to_string(),
                )
            })?;
            if tokens == 0 {
                return Err(KboltError::Inference(
                    "managed embedder tokenize smoke returned zero tokens".to_string(),
                )
                .into());
            }
        }
        ManagedRole::Reranker => {
            let reranker = clients.reranker.as_ref().ok_or_else(|| {
                KboltError::Inference("managed reranker client was not built".to_string())
            })?;
            let scores = reranker.rerank(
                "kbolt local probe",
                &["kbolt local rerank document".to_string()],
            )?;
            if scores.len() != 1 || !scores[0].is_finite() {
                return Err(KboltError::Inference(
                    "managed reranker smoke returned an invalid score".to_string(),
                )
                .into());
            }
        }
        ManagedRole::Expander => {
            let expander = clients.expander.as_ref().ok_or_else(|| {
                KboltError::Inference("managed expander client was not built".to_string())
            })?;
            let variants = expander.expand("kbolt local probe", 2)?;
            if variants.is_empty() {
                return Err(KboltError::Inference(
                    "managed expander smoke returned no variants".to_string(),
                )
                .into());
            }
        }
    }

    Ok(())
}

fn find_llama_server() -> Result<PathBuf> {
    find_llama_server_optional().ok_or_else(|| {
        KboltError::Config(format!(
            "llama-server was not found on PATH. Install llama.cpp and rerun setup. macOS hint: `{LLAMA_SERVER_BREW_HINT}`"
        ))
        .into()
    })
}

fn find_llama_server_optional() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            candidates.push(dir.join("llama-server"));
        }
    }
    candidates.push(PathBuf::from("/opt/homebrew/bin/llama-server"));
    candidates.push(PathBuf::from("/usr/local/bin/llama-server"));

    candidates.into_iter().find(|path| path.is_file())
}

fn write_pid(path: &Path, pid: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{pid}\n"))?;
    Ok(())
}

fn read_pid(path: &Path) -> Result<Option<u32>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)?;
    let pid = raw.trim().parse::<u32>().map_err(|err| {
        KboltError::Internal(format!("invalid pid file {}: {err}", path.display()))
    })?;
    Ok(Some(pid))
}

fn remove_pid_file(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

fn pid_is_alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn terminate_pid(pid: u32) -> Result<()> {
    let term = Command::new("kill")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|err| KboltError::Internal(format!("failed to send SIGTERM to {pid}: {err}")))?;
    if !term.success() && pid_is_alive(pid) {
        return Err(KboltError::Internal(format!("failed to stop process {pid}")).into());
    }

    let start = Instant::now();
    while pid_is_alive(pid) && start.elapsed() < STOP_WAIT_TIMEOUT {
        thread::sleep(Duration::from_millis(100));
    }

    if pid_is_alive(pid) {
        let kill = Command::new("kill")
            .arg("-9")
            .arg(pid.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|err| {
                KboltError::Internal(format!("failed to send SIGKILL to {pid}: {err}"))
            })?;
        if !kill.success() && pid_is_alive(pid) {
            return Err(KboltError::Internal(format!("failed to kill process {pid}")).into());
        }
    }

    Ok(())
}

fn is_port_bound(port: u16) -> bool {
    TcpListener::bind((LOCALHOST, port)).is_err()
}

fn stop_started_children(started: &[u32]) {
    for pid in started {
        let _ = terminate_pid(*pid);
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::net::TcpListener;
    use std::path::Path;
    use std::process::Command;

    use tempfile::tempdir;

    use super::{
        apply_managed_service_config, configure_llama_server_command, endpoint_for_port,
        load_setup_config, managed_embedder_model_path, managed_model_path, missing_service_report,
        open_managed_service_log, select_port, started_service_note, EMBEDDER_SPEC, EXPANDER_SPEC,
        LLAMA_SERVER_LOG_VERBOSITY, MANAGED_EMBED_PROVIDER, MANAGED_EXPAND_PROVIDER,
        MANAGED_RERANK_PROVIDER, RERANKER_SPEC,
    };
    use crate::config::{self, Config};

    fn temp_config() -> Config {
        let tmp = tempdir().expect("tempdir");
        let root = tmp.keep();
        let config_dir = root.join("config");
        fs::create_dir_all(&config_dir).expect("config dir");
        let mut config = config::load(Some(&config_dir)).expect("load config");
        config.cache_dir = root.join("cache");
        fs::create_dir_all(&config.cache_dir).expect("cache dir");
        config
    }

    #[test]
    fn setup_config_binds_managed_local_roles() {
        let mut config = temp_config();
        apply_managed_service_config(&mut config, &EMBEDDER_SPEC, 8101);
        apply_managed_service_config(&mut config, &RERANKER_SPEC, 8102);

        assert_eq!(
            config
                .roles
                .embedder
                .as_ref()
                .map(|role| role.provider.as_str()),
            Some("kbolt_local_embed")
        );
        assert_eq!(
            config
                .roles
                .reranker
                .as_ref()
                .map(|role| role.provider.as_str()),
            Some("kbolt_local_rerank")
        );
        assert!(config.providers.contains_key("kbolt_local_embed"));
        assert!(config.providers.contains_key("kbolt_local_rerank"));
    }

    #[test]
    fn load_setup_config_moves_incompatible_index_toml_aside() {
        let tmp = tempdir().expect("tempdir");
        let config_dir = tmp.path().join("config");
        fs::create_dir_all(&config_dir).expect("config dir");
        let config_file = config_dir.join("index.toml");
        fs::write(
            &config_file,
            r#"
[embeddings]
provider = "legacy"
"#,
        )
        .expect("write invalid config");

        let (config, notes) =
            load_setup_config(Some(&config_dir)).expect("setup config should recover");

        let backup_path = config_dir.join("index.toml.invalid.bak");
        assert!(backup_path.is_file(), "backup should exist");
        assert!(
            notes
                .iter()
                .any(|note| note.contains("moved incompatible legacy config")),
            "expected recovery note, got {notes:?}"
        );
        assert!(
            config.providers.is_empty(),
            "fresh config should be created"
        );
        assert!(config.roles.embedder.is_none());
        assert!(config.roles.reranker.is_none());
    }

    #[test]
    fn load_setup_config_keeps_non_legacy_invalid_config_as_error() {
        let tmp = tempdir().expect("tempdir");
        let config_dir = tmp.path().join("config");
        fs::create_dir_all(&config_dir).expect("config dir");
        let config_file = config_dir.join("index.toml");
        fs::write(
            &config_file,
            r#"
[providers.bad]
operation = "embedding"
"#,
        )
        .expect("write invalid config");

        let err = load_setup_config(Some(&config_dir)).expect_err("invalid config should fail");
        assert!(
            err.to_string().contains("invalid config file"),
            "unexpected error: {err}"
        );
        assert!(config_file.is_file(), "invalid file should remain in place");
        assert!(
            !config_dir.join("index.toml.invalid.bak").exists(),
            "non-legacy config should not be backed up automatically"
        );
    }

    #[test]
    fn enable_deep_binds_managed_expander_role() {
        let mut config = temp_config();
        apply_managed_service_config(&mut config, &EXPANDER_SPEC, 8103);

        assert_eq!(
            config
                .roles
                .expander
                .as_ref()
                .map(|role| role.provider.as_str()),
            Some("kbolt_local_expand")
        );
        assert!(config.providers.contains_key("kbolt_local_expand"));
    }

    #[test]
    fn select_port_skips_bound_ports() {
        let mut config = temp_config();
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind test port");
        let occupied_port = listener.local_addr().expect("local addr").port();
        apply_managed_service_config(&mut config, &EMBEDDER_SPEC, occupied_port);

        let port = select_port(&config, &EMBEDDER_SPEC, &Default::default()).expect("select port");
        assert_ne!(port, occupied_port);
    }

    #[test]
    fn missing_service_report_marks_unconfigured() {
        let cache = tempdir().expect("tempdir");
        let report = missing_service_report(cache.path(), &EMBEDDER_SPEC);
        assert!(!report.enabled);
        assert!(!report.ready);
        assert_eq!(report.endpoint, endpoint_for_port(8101));
    }

    #[test]
    fn managed_model_paths_are_stable() {
        let cache = tempdir().expect("tempdir");
        assert_eq!(
            managed_model_path(cache.path(), &EMBEDDER_SPEC),
            cache
                .path()
                .join("models")
                .join("embedder")
                .join("embeddinggemma-300M-Q8_0.gguf")
        );
        assert_eq!(
            managed_model_path(cache.path(), &RERANKER_SPEC),
            cache
                .path()
                .join("models")
                .join("reranker")
                .join("qwen3-reranker-0.6b-q8_0.gguf")
        );
        assert_eq!(
            managed_model_path(cache.path(), &EXPANDER_SPEC),
            cache
                .path()
                .join("models")
                .join("expander")
                .join("Qwen3-1.7B-Q8_0.gguf")
        );
    }

    #[test]
    fn managed_embedder_model_path_only_resolves_embedder_provider() {
        let cache = tempdir().expect("tempdir");
        assert_eq!(
            managed_embedder_model_path(cache.path(), MANAGED_EMBED_PROVIDER),
            Some(
                cache
                    .path()
                    .join("models")
                    .join("embedder")
                    .join("embeddinggemma-300M-Q8_0.gguf")
            )
        );
        assert_eq!(
            managed_embedder_model_path(cache.path(), MANAGED_RERANK_PROVIDER),
            None
        );
        assert_eq!(
            managed_embedder_model_path(cache.path(), MANAGED_EXPAND_PROVIDER),
            None
        );
    }

    #[test]
    fn managed_service_log_is_truncated_on_spawn_open() {
        let tmp = tempdir().expect("tempdir");
        let log_file = tmp.path().join("logs").join("embedder.log");
        fs::create_dir_all(log_file.parent().expect("log parent")).expect("create log dir");
        fs::write(&log_file, "old output\n").expect("write old log");

        {
            let mut log = open_managed_service_log(&log_file).expect("open managed log");
            log.write_all(b"new output\n").expect("write new log");
        }

        let content = fs::read_to_string(&log_file).expect("read log");
        assert_eq!(content, "new output\n");
    }

    #[test]
    fn llama_server_command_limits_managed_log_verbosity() {
        for spec in [&EMBEDDER_SPEC, &RERANKER_SPEC, &EXPANDER_SPEC] {
            let mut command = Command::new("nohup");
            configure_llama_server_command(
                &mut command,
                Path::new("/usr/local/bin/llama-server"),
                spec,
                Path::new("/tmp/model.gguf"),
                spec.preferred_port,
            );
            let args = command
                .get_args()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect::<Vec<_>>();

            assert!(
                args.windows(2).any(|pair| {
                    pair[0] == "--log-verbosity" && pair[1] == LLAMA_SERVER_LOG_VERBOSITY
                }),
                "expected log verbosity in args for {}: {args:?}",
                spec.name
            );
        }
    }

    #[test]
    fn started_service_note_omits_model_path() {
        let note = started_service_note("embedder", "http://127.0.0.1:8101");
        assert_eq!(note, "started embedder on http://127.0.0.1:8101");
        assert!(!note.contains(".gguf"));
    }
}
