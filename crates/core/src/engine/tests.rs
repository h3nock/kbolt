use fs2::FileExt;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tempfile::tempdir;

use crate::config::{
    ChunkPolicy, ChunkingConfig, Config, EmbedderRoleConfig, ProviderOperation,
    ProviderProfileConfig, RankingConfig, ReapingConfig, RoleBindingsConfig,
};
use crate::engine::{retrieval_text_with_prefix, Engine, TargetScope};
use crate::ingest::chunk::FinalChunkKind;
use crate::storage::Storage;
use kbolt_types::{
    ActiveSpaceSource, AddCollectionRequest, AddScheduleRequest, GetRequest, InitialIndexingBlock,
    InitialIndexingOutcome, KboltError, Locator, MultiGetRequest, OmitReason,
    RemoveScheduleRequest, RemoveScheduleSelector, ScheduleBackend, ScheduleInterval,
    ScheduleIntervalUnit, ScheduleRunResult, ScheduleScope, ScheduleState, ScheduleTrigger,
    ScheduleWeekday, SearchMode, SearchRequest, UpdateDecisionKind, UpdateOptions,
};

#[derive(Default)]
struct DeterministicEmbedder;

impl crate::models::Embedder for DeterministicEmbedder {
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
struct RecordingEmbedder {
    calls: Mutex<Vec<Vec<String>>>,
}

impl RecordingEmbedder {
    fn texts(&self) -> Vec<String> {
        self.calls
            .lock()
            .expect("lock recording embedder")
            .iter()
            .flatten()
            .cloned()
            .collect()
    }
}

impl crate::models::Embedder for RecordingEmbedder {
    fn embed_batch(
        &self,
        _kind: crate::models::EmbeddingInputKind,
        texts: &[String],
    ) -> crate::Result<Vec<Vec<f32>>> {
        self.calls
            .lock()
            .expect("lock recording embedder")
            .push(texts.to_vec());
        Ok(texts
            .iter()
            .map(|text| {
                vec![
                    text.split_whitespace().count().max(1) as f32,
                    text.len() as f32,
                ]
            })
            .collect())
    }
}

#[derive(Default)]
struct SelectiveFailureEmbedder;

impl crate::models::Embedder for SelectiveFailureEmbedder {
    fn embed_batch(
        &self,
        _kind: crate::models::EmbeddingInputKind,
        texts: &[String],
    ) -> crate::Result<Vec<Vec<f32>>> {
        if texts.iter().any(|text| text.contains("EMBED_FAIL")) {
            return Err(KboltError::Inference("simulated embed failure".to_string()).into());
        }

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
struct CharCountDocumentSizer;

impl crate::models::EmbeddingDocumentSizer for CharCountDocumentSizer {
    fn count_document_tokens(&self, text: &str) -> crate::Result<usize> {
        Ok(text.chars().count())
    }
}

#[derive(Default)]
struct CountingCharDocumentSizer {
    calls: AtomicUsize,
}

impl CountingCharDocumentSizer {
    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn reset(&self) {
        self.calls.store(0, Ordering::SeqCst);
    }
}

impl crate::models::EmbeddingDocumentSizer for CountingCharDocumentSizer {
    fn count_document_tokens(&self, text: &str) -> crate::Result<usize> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(text.chars().count())
    }
}

#[derive(Default)]
struct SecondCallOversizeDocumentSizer {
    calls: AtomicUsize,
    reject_after_first: std::sync::atomic::AtomicBool,
}

impl SecondCallOversizeDocumentSizer {
    fn reject_after_first_call(&self) {
        self.calls.store(0, Ordering::SeqCst);
        self.reject_after_first.store(true, Ordering::SeqCst);
    }
}

impl crate::models::EmbeddingDocumentSizer for SecondCallOversizeDocumentSizer {
    fn count_document_tokens(&self, text: &str) -> crate::Result<usize> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.reject_after_first.load(Ordering::SeqCst) && call > 1 {
            return Ok(10_000);
        }

        Ok(text.chars().count())
    }
}

#[derive(Default)]
struct SelectiveFailureDocumentSizer;

impl crate::models::EmbeddingDocumentSizer for SelectiveFailureDocumentSizer {
    fn count_document_tokens(&self, text: &str) -> crate::Result<usize> {
        if text.contains("TOKENIZE_FAIL") {
            return Err(KboltError::Inference("simulated tokenize failure".to_string()).into());
        }

        Ok(text.chars().count())
    }
}

#[derive(Default)]
struct DeterministicReranker;

impl crate::models::Reranker for DeterministicReranker {
    fn rerank(&self, query: &str, docs: &[String]) -> crate::Result<Vec<f32>> {
        let query = query.to_ascii_lowercase();
        Ok(docs
            .iter()
            .map(|doc| {
                if doc.to_ascii_lowercase().contains(&query) {
                    1.0
                } else {
                    0.5
                }
            })
            .collect())
    }
}

struct ConstantReranker(f32);

impl crate::models::Reranker for ConstantReranker {
    fn rerank(&self, _query: &str, docs: &[String]) -> crate::Result<Vec<f32>> {
        Ok(vec![self.0; docs.len()])
    }
}

#[derive(Default)]
struct DeterministicExpander;

impl crate::models::Expander for DeterministicExpander {
    fn expand(&self, query: &str, _max_variants: usize) -> crate::Result<Vec<String>> {
        Ok(vec![format!("explain {query}")])
    }
}

struct StaticExpander {
    items: Vec<String>,
}

impl crate::models::Expander for StaticExpander {
    fn expand(&self, _query: &str, _max_variants: usize) -> crate::Result<Vec<String>> {
        Ok(self.items.clone())
    }
}

fn base_test_config(config_dir: std::path::PathBuf, cache_dir: std::path::PathBuf) -> Config {
    Config {
        config_dir,
        cache_dir,
        default_space: None,
        providers: HashMap::new(),
        roles: RoleBindingsConfig::default(),
        reaping: ReapingConfig { days: 7 },
        chunking: ChunkingConfig::default(),
        ranking: RankingConfig::default(),
    }
}

fn test_engine() -> Engine {
    let root = tempdir().expect("create temp root");
    let root_path = root.path().to_path_buf();
    std::mem::forget(root);
    let config_dir = root_path.join("config");
    let cache_dir = root_path.join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let config = base_test_config(config_dir, cache_dir);
    Engine::from_parts(storage, config)
}

fn test_engine_with_embedder(embedder: Arc<dyn crate::models::Embedder>) -> Engine {
    let root = tempdir().expect("create temp root");
    let root_path = root.path().to_path_buf();
    std::mem::forget(root);
    let config_dir = root_path.join("config");
    let cache_dir = root_path.join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let config = base_test_config(config_dir, cache_dir);
    Engine::from_parts_with_embedder(storage, config, Some(embedder))
}

fn test_engine_with_embedder_and_chunking(
    embedder: Arc<dyn crate::models::Embedder>,
    chunking: ChunkingConfig,
) -> Engine {
    let root = tempdir().expect("create temp root");
    let root_path = root.path().to_path_buf();
    std::mem::forget(root);
    let config_dir = root_path.join("config");
    let cache_dir = root_path.join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let mut config = base_test_config(config_dir, cache_dir);
    config.chunking = chunking;
    Engine::from_parts_with_embedder(storage, config, Some(embedder))
}

fn test_engine_with_embedding_runtime(
    embedder: Arc<dyn crate::models::Embedder>,
    document_sizer: Arc<dyn crate::models::EmbeddingDocumentSizer>,
    chunking: ChunkingConfig,
) -> Engine {
    let root = tempdir().expect("create temp root");
    let root_path = root.path().to_path_buf();
    std::mem::forget(root);
    let config_dir = root_path.join("config");
    let cache_dir = root_path.join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let mut config = base_test_config(config_dir, cache_dir);
    config.chunking = chunking;
    Engine::from_parts_with_embedding_runtime(storage, config, Some(embedder), Some(document_sizer))
}

fn test_engine_with_search_models(
    embedder: Option<Arc<dyn crate::models::Embedder>>,
    reranker: Option<Arc<dyn crate::models::Reranker>>,
    expander: Option<Arc<dyn crate::models::Expander>>,
) -> Engine {
    test_engine_with_search_models_and_ranking(
        embedder,
        reranker,
        expander,
        RankingConfig::default(),
    )
}

fn test_engine_with_search_models_and_ranking(
    embedder: Option<Arc<dyn crate::models::Embedder>>,
    reranker: Option<Arc<dyn crate::models::Reranker>>,
    expander: Option<Arc<dyn crate::models::Expander>>,
    ranking: RankingConfig,
) -> Engine {
    let root = tempdir().expect("create temp root");
    let root_path = root.path().to_path_buf();
    std::mem::forget(root);
    let config_dir = root_path.join("config");
    let cache_dir = root_path.join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let mut config = base_test_config(config_dir, cache_dir);
    config.ranking = ranking;
    Engine::from_parts_with_models(storage, config, embedder, reranker, expander)
}

fn test_engine_with_embedder_and_embedding_model(
    embedder: Arc<dyn crate::models::Embedder>,
    model: &str,
) -> Engine {
    let root = tempdir().expect("create temp root");
    let root_path = root.path().to_path_buf();
    std::mem::forget(root);
    let config_dir = root_path.join("config");
    let cache_dir = root_path.join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let mut config = base_test_config(config_dir, cache_dir);
    config.providers.insert(
        "remote_embed".to_string(),
        ProviderProfileConfig::OpenAiCompatible {
            operation: ProviderOperation::Embedding,
            base_url: "https://example.test/v1".to_string(),
            model: model.to_string(),
            api_key_env: None,
            timeout_ms: 30_000,
            max_retries: 0,
        },
    );
    config.roles.embedder = Some(EmbedderRoleConfig {
        provider: "remote_embed".to_string(),
        batch_size: 32,
    });
    Engine::from_parts_with_embedder(storage, config, Some(embedder))
}

fn test_engine_with_default_space(default_space: Option<&str>) -> Engine {
    let root = tempdir().expect("create temp root");
    let root_path = root.path().to_path_buf();
    std::mem::forget(root);
    let config_dir = root_path.join("config");
    let cache_dir = root_path.join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let mut config = base_test_config(config_dir, cache_dir);
    config.default_space = default_space.map(ToString::to_string);
    Engine::from_parts(storage, config)
}

fn table_split_chunking_config() -> ChunkingConfig {
    let policy = ChunkPolicy {
        target_tokens: 12,
        soft_max_tokens: 12,
        hard_max_tokens: 12,
        boundary_overlap_tokens: 0,
        neighbor_window: 1,
        contextual_prefix: true,
    };
    ChunkingConfig {
        defaults: policy.clone(),
        profiles: HashMap::from([("md".to_string(), policy)]),
    }
}

fn test_engine_with_reaping_days(days: u32) -> Engine {
    let root = tempdir().expect("create temp root");
    let root_path = root.path().to_path_buf();
    std::mem::forget(root);
    let config_dir = root_path.join("config");
    let cache_dir = root_path.join("cache");
    let storage = Storage::new(&cache_dir).expect("create storage");
    let mut config = base_test_config(config_dir, cache_dir);
    config.reaping = ReapingConfig { days };
    Engine::from_parts(storage, config)
}

fn with_kbolt_space_env<T>(value: Option<&str>, run: impl FnOnce() -> T) -> T {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("lock env mutex");

    let old_value: Option<OsString> = std::env::var_os("KBOLT_SPACE");
    match value {
        Some(v) => std::env::set_var("KBOLT_SPACE", v),
        None => std::env::remove_var("KBOLT_SPACE"),
    }

    let result = run();
    match old_value {
        Some(v) => std::env::set_var("KBOLT_SPACE", v),
        None => std::env::remove_var("KBOLT_SPACE"),
    }
    result
}

fn update_options(space: Option<&str>, collections: &[&str]) -> UpdateOptions {
    UpdateOptions {
        space: space.map(ToString::to_string),
        collections: collections.iter().map(|item| item.to_string()).collect(),
        no_embed: false,
        dry_run: false,
        verbose: false,
    }
}

fn verbose_update_options(space: Option<&str>, collections: &[&str]) -> UpdateOptions {
    let mut options = update_options(space, collections);
    options.verbose = true;
    options
}

fn add_collection_fixture(engine: &Engine, space: &str, name: &str, path: std::path::PathBuf) {
    engine
        .add_collection(AddCollectionRequest {
            path,
            space: Some(space.to_string()),
            name: Some(name.to_string()),
            description: None,
            extensions: None,
            no_index: true,
        })
        .expect("add collection fixture");
}

fn write_text_file(path: &std::path::Path, text: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent directories");
    }
    std::fs::write(path, text).expect("write file");
}

fn expected_schedule_backend() -> ScheduleBackend {
    #[cfg(target_os = "macos")]
    {
        ScheduleBackend::Launchd
    }

    #[cfg(target_os = "linux")]
    {
        ScheduleBackend::SystemdUser
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        panic!("schedule backend is unsupported on this platform")
    }
}

fn schedule_backend_artifact_paths(engine: &Engine, schedule_id: &str) -> Vec<std::path::PathBuf> {
    #[cfg(target_os = "macos")]
    {
        return vec![engine
            .config()
            .config_dir
            .join("launchd/LaunchAgents")
            .join(format!("com.kbolt.schedule.{schedule_id}.plist"))];
    }

    #[cfg(target_os = "linux")]
    {
        return vec![
            engine
                .config()
                .config_dir
                .join("systemd/user")
                .join(format!("kbolt-schedule-{schedule_id}.service")),
            engine
                .config()
                .config_dir
                .join("systemd/user")
                .join(format!("kbolt-schedule-{schedule_id}.timer")),
        ];
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (engine, schedule_id);
        panic!("schedule backend is unsupported on this platform");
    }
}

#[test]
fn retrieval_text_with_prefix_adds_title_and_heading_context() {
    let text =
        retrieval_text_with_prefix("body text", Some("Guide"), Some("Setup > Install"), true);
    assert_eq!(text, "title: Guide\nheading: Setup > Install\n\nbody text");
}

#[test]
fn retrieval_text_with_prefix_respects_disabled_flag() {
    let text = retrieval_text_with_prefix("body text", Some("Guide"), Some("Setup"), false);
    assert_eq!(text, "body text");
}

#[test]
fn retrieval_text_with_prefix_omits_fallback_title_when_absent() {
    let text = retrieval_text_with_prefix("body text", None, Some("Setup"), true);
    assert_eq!(text, "heading: Setup\n\nbody text");
}

#[test]
fn add_space_and_space_info_include_description_and_zero_counts() {
    let engine = test_engine();

    let added = engine
        .add_space("work", Some("work docs"))
        .expect("add space");
    assert_eq!(added.name, "work");
    assert_eq!(added.description.as_deref(), Some("work docs"));
    assert_eq!(added.collection_count, 0);
    assert_eq!(added.document_count, 0);
    assert_eq!(added.chunk_count, 0);

    let fetched = engine.space_info("work").expect("fetch space info");
    assert_eq!(fetched.name, "work");
    assert_eq!(fetched.description.as_deref(), Some("work docs"));
}

#[test]
fn list_spaces_returns_default_and_added_spaces() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    engine.add_space("notes", None).expect("add notes");

    let spaces = engine.list_spaces().expect("list spaces");
    let names: Vec<String> = spaces.into_iter().map(|space| space.name).collect();
    assert_eq!(
        names,
        vec![
            "default".to_string(),
            "notes".to_string(),
            "work".to_string()
        ]
    );
}

#[test]
fn describe_rename_and_remove_space_delegate_to_storage() {
    let mut engine = test_engine();
    engine.add_space("work", None).expect("add work");

    engine
        .describe_space("work", "new description")
        .expect("describe space");
    let described = engine.space_info("work").expect("space info");
    assert_eq!(described.description.as_deref(), Some("new description"));

    engine
        .rename_space("work", "team")
        .expect("rename work to team");
    let renamed = engine.space_info("team").expect("team should exist");
    assert_eq!(renamed.name, "team");
    let missing_old = engine
        .space_info("work")
        .expect_err("work should be missing");
    match KboltError::from(missing_old) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "work"),
        other => panic!("unexpected error: {other}"),
    }

    engine.remove_space("team").expect("remove team");
    let missing_team = engine
        .space_info("team")
        .expect_err("team should be missing");
    match KboltError::from(missing_team) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "team"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn remove_space_clears_matching_default_space() {
    let mut engine = test_engine();
    engine.add_space("work", None).expect("add work");
    engine
        .set_default_space(Some("work"))
        .expect("set default space");

    engine.remove_space("work").expect("remove work");

    assert_eq!(engine.config().default_space, None);
    let loaded = crate::config::load(Some(engine.config().config_dir.as_path()))
        .expect("reload config from disk");
    assert_eq!(loaded.default_space, None);
}

#[test]
fn rename_space_updates_matching_default_space() {
    let mut engine = test_engine();
    engine.add_space("work", None).expect("add work");
    engine
        .set_default_space(Some("work"))
        .expect("set default space");

    engine
        .rename_space("work", "team")
        .expect("rename default space");

    assert_eq!(engine.config().default_space.as_deref(), Some("team"));
    let loaded = crate::config::load(Some(engine.config().config_dir.as_path()))
        .expect("reload config from disk");
    assert_eq!(loaded.default_space.as_deref(), Some("team"));
}

#[test]
fn config_and_storage_accessors_expose_engine_components() {
    let engine = test_engine();
    assert!(engine.config().providers.is_empty());
    assert_eq!(engine.config().roles, RoleBindingsConfig::default());

    let default_space = engine
        .storage()
        .get_space("default")
        .expect("default space should exist");
    assert_eq!(default_space.name, "default");
    assert_eq!(engine.config().default_space, None::<String>);
    assert!(!engine.config().config_dir.as_os_str().is_empty());
}

#[test]
fn resolve_space_returns_explicit_space_when_provided() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let resolved = engine
        .resolve_space(Some("work"))
        .expect("resolve explicit space");
    assert_eq!(resolved, "work");
}

#[test]
fn resolve_space_uses_configured_default_when_no_explicit_space() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");

        let resolved = engine.resolve_space(None).expect("resolve default space");
        assert_eq!(resolved, "work");
    });
}

#[test]
fn set_default_space_persists_config_and_can_clear_it() {
    let mut engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let set = engine
        .set_default_space(Some("work"))
        .expect("set default space");
    assert_eq!(set.as_deref(), Some("work"));
    assert_eq!(engine.config().default_space.as_deref(), Some("work"));

    let loaded = crate::config::load(Some(engine.config().config_dir.as_path()))
        .expect("reload config from disk");
    assert_eq!(loaded.default_space.as_deref(), Some("work"));

    let cleared = engine.set_default_space(None).expect("clear default space");
    assert_eq!(cleared, None);
    assert_eq!(engine.config().default_space, None);
}

#[test]
fn set_default_space_requires_existing_space() {
    let mut engine = test_engine();

    let err = engine
        .set_default_space(Some("missing"))
        .expect_err("missing space should fail");
    match KboltError::from(err) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn resolve_space_prefers_env_over_config_default() {
    with_kbolt_space_env(Some("notes"), || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let resolved = engine.resolve_space(None).expect("resolve space");
        assert_eq!(resolved, "notes");
    });
}

#[test]
fn resolve_space_returns_no_active_space_when_no_sources_exist() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        let err = engine
            .resolve_space(None)
            .expect_err("expected no active space");
        match KboltError::from(err) {
            KboltError::NoActiveSpace => {}
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn current_space_prefers_flag_over_env_and_default() {
    with_kbolt_space_env(Some("notes"), || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");
        engine.add_space("ops", None).expect("add ops");

        let current = engine
            .current_space(Some("ops"))
            .expect("resolve current space")
            .expect("expected active space");
        assert_eq!(current.name, "ops");
        assert_eq!(current.source, ActiveSpaceSource::Flag);
    });
}

#[test]
fn current_space_reports_env_source() {
    with_kbolt_space_env(Some("notes"), || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let current = engine
            .current_space(None)
            .expect("resolve current space")
            .expect("expected active space");
        assert_eq!(current.name, "notes");
        assert_eq!(current.source, ActiveSpaceSource::EnvVar);
    });
}

#[test]
fn current_space_reports_default_source() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");

        let current = engine
            .current_space(None)
            .expect("resolve current space")
            .expect("expected active space");
        assert_eq!(current.name, "work");
        assert_eq!(current.source, ActiveSpaceSource::ConfigDefault);
    });
}

#[test]
fn current_space_returns_none_when_no_space_is_active() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        let current = engine.current_space(None).expect("resolve current space");
        assert_eq!(current, None);
    });
}

#[test]
fn collection_info_without_space_uses_unique_collection_lookup() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        engine
            .add_collection(AddCollectionRequest {
                path: collection_path,
                space: Some("work".to_string()),
                name: Some("api".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add collection");

        let info = engine
            .collection_info(None, "api")
            .expect("resolve unique collection");
        assert_eq!(info.space, "work");
        assert_eq!(info.name, "api");
    });
}

#[test]
fn collection_info_without_space_reports_ambiguous_collection_lookup() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");
        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        engine
            .add_collection(AddCollectionRequest {
                path: work_path,
                space: Some("work".to_string()),
                name: Some("api".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add work collection");
        engine
            .add_collection(AddCollectionRequest {
                path: notes_path,
                space: Some("notes".to_string()),
                name: Some("api".to_string()),
                description: None,
                extensions: None,
                no_index: true,
            })
            .expect("add notes collection");

        let err = engine
            .collection_info(None, "api")
            .expect_err("expected ambiguous collection");
        match KboltError::from(err) {
            KboltError::AmbiguousSpace { collection, spaces } => {
                assert_eq!(collection, "api");
                assert_eq!(spaces, vec!["notes".to_string(), "work".to_string()]);
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn add_collection_and_collection_info_with_explicit_space() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    let root = tempdir().expect("create temp root");
    let collection_path = root.path().join("api");
    std::fs::create_dir_all(&collection_path).expect("create collection dir");

    let added = engine
        .add_collection(AddCollectionRequest {
            path: collection_path.clone(),
            space: Some("work".to_string()),
            name: Some("api".to_string()),
            description: Some("API docs".to_string()),
            extensions: Some(vec!["rs".to_string(), "md".to_string()]),
            no_index: true,
        })
        .expect("add collection");
    assert!(matches!(
        added.initial_indexing,
        InitialIndexingOutcome::Skipped
    ));
    assert_eq!(added.collection.name, "api");
    assert_eq!(added.collection.space, "work");
    assert_eq!(added.collection.path, collection_path);
    assert_eq!(added.collection.description.as_deref(), Some("API docs"));
    assert_eq!(
        added.collection.extensions,
        Some(vec!["rs".to_string(), "md".to_string()])
    );
    assert_eq!(added.collection.document_count, 0);
    assert_eq!(added.collection.active_document_count, 0);
    assert_eq!(added.collection.chunk_count, 0);
    assert_eq!(added.collection.embedded_chunk_count, 0);

    let info = engine
        .collection_info(Some("work"), "api")
        .expect("fetch collection info");
    assert_eq!(info.name, "api");
    assert_eq!(info.space, "work");
}

#[test]
fn add_collection_implicitly_creates_explicit_space_when_missing() {
    let engine = test_engine();
    let root = tempdir().expect("create temp root");
    let collection_path = root.path().join("api");
    std::fs::create_dir_all(&collection_path).expect("create collection dir");

    let added = engine
        .add_collection(AddCollectionRequest {
            path: collection_path.clone(),
            space: Some("work".to_string()),
            name: Some("api".to_string()),
            description: None,
            extensions: None,
            no_index: true,
        })
        .expect("add collection with implicit space");

    assert!(matches!(
        added.initial_indexing,
        InitialIndexingOutcome::Skipped
    ));
    assert_eq!(added.collection.space, "work");
    assert_eq!(added.collection.name, "api");
    assert_eq!(added.collection.path, collection_path);

    let space = engine.space_info("work").expect("fetch implicit space");
    assert_eq!(space.name, "work");

    let info = engine
        .collection_info(Some("work"), "api")
        .expect("fetch collection info");
    assert_eq!(info.name, "api");
    assert_eq!(info.space, "work");
}

#[test]
fn add_collection_without_no_index_triggers_initial_index_update() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    let root = tempdir().expect("create temp root");
    let collection_path = root.path().join("api");
    std::fs::create_dir_all(&collection_path).expect("create collection dir");
    write_text_file(&collection_path.join("src/lib.rs"), "fn alpha() {}\n");

    let added = engine
        .add_collection(AddCollectionRequest {
            path: collection_path.clone(),
            space: Some("work".to_string()),
            name: Some("api".to_string()),
            description: None,
            extensions: None,
            no_index: false,
        })
        .expect("collection add should index by default");
    let report = match &added.initial_indexing {
        InitialIndexingOutcome::Indexed(report) => report,
        other => panic!("expected indexed outcome, got: {other:?}"),
    };
    assert_eq!(added.collection.space, "work");
    assert_eq!(added.collection.name, "api");
    assert_eq!(added.collection.path, collection_path);
    assert_eq!(added.collection.document_count, 1);
    assert_eq!(added.collection.active_document_count, 1);
    assert_eq!(added.collection.chunk_count, 1);
    assert_eq!(report.added_docs, 1);
    assert_eq!(report.failed_docs, 0);
}

#[test]
fn add_collection_returns_blocked_outcome_when_space_dense_repair_is_required() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let first_path = root.path().join("work-api");
        std::fs::create_dir_all(&first_path).expect("create first collection dir");
        add_collection_fixture(&engine, "work", "api", first_path.clone());
        write_text_file(&first_path.join("src/lib.rs"), "fn alpha() {}\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("index first collection");

        let work_space = engine.storage().get_space("work").expect("get work space");
        let api = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(api.id, "src/lib.rs")
            .expect("query document")
            .expect("document should exist");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");
        assert!(!chunks.is_empty(), "expected indexed chunks");
        engine
            .storage()
            .insert_embeddings(&[(chunks[0].id, "stale-model")])
            .expect("insert stale embedding row");

        let second_path = root.path().join("work-docs");
        std::fs::create_dir_all(&second_path).expect("create second collection dir");
        write_text_file(&second_path.join("guide.md"), "guide text\n");

        let added = engine
            .add_collection(AddCollectionRequest {
                path: second_path.clone(),
                space: Some("work".to_string()),
                name: Some("docs".to_string()),
                description: None,
                extensions: None,
                no_index: false,
            })
            .expect("collection registration should still succeed");

        assert_eq!(added.collection.space, "work");
        assert_eq!(added.collection.name, "docs");
        assert_eq!(added.collection.path, second_path);
        match added.initial_indexing {
            InitialIndexingOutcome::Blocked(InitialIndexingBlock::SpaceDenseRepairRequired {
                space,
                reason,
            }) => {
                assert_eq!(space, "work");
                assert!(
                    reason.contains("stale-model"),
                    "expected model drift detail in reason: {reason}"
                );
            }
            other => panic!("expected dense-repair block, got: {other:?}"),
        }

        let info = engine
            .collection_info(Some("work"), "docs")
            .expect("collection should remain registered");
        assert_eq!(info.name, "docs");
        assert_eq!(info.space, "work");
        assert_eq!(info.path, root.path().join("work-docs"));
        assert_eq!(info.document_count, 0);
    });
}

#[test]
fn collection_mutation_wrappers_delegate_to_storage_with_explicit_space() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    let root = tempdir().expect("create temp root");
    let collection_path = root.path().join("api");
    std::fs::create_dir_all(&collection_path).expect("create collection dir");
    engine
        .add_collection(AddCollectionRequest {
            path: collection_path,
            space: Some("work".to_string()),
            name: Some("api".to_string()),
            description: None,
            extensions: None,
            no_index: true,
        })
        .expect("add collection");
    let ignore_dir = engine.config().config_dir.join("ignores").join("work");
    std::fs::create_dir_all(&ignore_dir).expect("create ignore dir");
    let old_ignore_path = ignore_dir.join("api.ignore");
    write_text_file(&old_ignore_path, "dist/\n");
    assert!(old_ignore_path.exists(), "ignore file should exist");

    engine
        .describe_collection(Some("work"), "api", "updated desc")
        .expect("describe collection");
    let described = engine
        .collection_info(Some("work"), "api")
        .expect("collection info");
    assert_eq!(described.description.as_deref(), Some("updated desc"));

    engine
        .rename_collection(Some("work"), "api", "backend")
        .expect("rename collection");
    let renamed = engine
        .collection_info(Some("work"), "backend")
        .expect("backend info");
    assert_eq!(renamed.name, "backend");
    let renamed_ignore_path = ignore_dir.join("backend.ignore");
    assert!(
        !old_ignore_path.exists(),
        "old ignore file should be renamed"
    );
    assert!(
        renamed_ignore_path.exists(),
        "renamed ignore file should exist"
    );

    engine
        .remove_collection(Some("work"), "backend")
        .expect("remove collection");
    assert!(
        !renamed_ignore_path.exists(),
        "ignore file should be deleted with collection"
    );
    let missing = engine
        .collection_info(Some("work"), "backend")
        .expect_err("backend should be removed");
    match KboltError::from(missing) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "backend"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn remove_collection_purges_search_indexes() {
    let engine = test_engine_with_default_space(None);
    engine.add_space("work", None).expect("add work");

    let root = tempdir().expect("create temp root");
    let alpha_path = root.path().join("alpha");
    let beta_path = root.path().join("beta");
    std::fs::create_dir_all(&alpha_path).expect("create alpha dir");
    std::fs::create_dir_all(&beta_path).expect("create beta dir");
    add_collection_fixture(&engine, "work", "alpha", alpha_path.clone());
    add_collection_fixture(&engine, "work", "beta", beta_path.clone());

    write_text_file(
        &alpha_path.join("strong.md"),
        "token token token token token\n",
    );
    write_text_file(&beta_path.join("weak.md"), "token\n");
    engine
        .update(update_options(Some("work"), &["alpha", "beta"]))
        .expect("initial update");

    engine
        .remove_collection(Some("work"), "alpha")
        .expect("remove alpha collection");

    let response = engine
        .search(SearchRequest {
            query: "token".to_string(),
            mode: SearchMode::Keyword,
            space: Some("work".to_string()),
            collections: vec!["beta".to_string()],
            limit: 1,
            min_score: 0.0,
            no_rerank: false,
            debug: false,
        })
        .expect("run keyword search");

    assert_eq!(response.results.len(), 1);
    assert_eq!(response.results[0].path, "beta/weak.md");
    assert_eq!(response.results[0].collection, "beta");
}

#[test]
fn list_collections_returns_all_or_space_scoped_collections() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");
    engine.add_space("notes", None).expect("add notes");

    let root = tempdir().expect("create temp root");
    let work_path = root.path().join("work-api");
    let notes_path = root.path().join("notes-wiki");
    std::fs::create_dir_all(&work_path).expect("create work dir");
    std::fs::create_dir_all(&notes_path).expect("create notes dir");

    engine
        .add_collection(AddCollectionRequest {
            path: work_path,
            space: Some("work".to_string()),
            name: Some("api".to_string()),
            description: None,
            extensions: None,
            no_index: true,
        })
        .expect("add work collection");
    engine
        .add_collection(AddCollectionRequest {
            path: notes_path,
            space: Some("notes".to_string()),
            name: Some("wiki".to_string()),
            description: None,
            extensions: None,
            no_index: true,
        })
        .expect("add notes collection");

    let all = engine.list_collections(None).expect("list all");
    assert_eq!(all.len(), 2);
    assert!(all
        .iter()
        .any(|collection| collection.space == "work" && collection.name == "api"));
    assert!(all
        .iter()
        .any(|collection| collection.space == "notes" && collection.name == "wiki"));

    let work_only = engine
        .list_collections(Some("work"))
        .expect("list work only");
    assert_eq!(work_only.len(), 1);
    assert_eq!(work_only[0].space, "work");
    assert_eq!(work_only[0].name, "api");
}

#[test]
fn read_collection_ignore_returns_none_when_file_missing() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let (space, content) = engine
            .read_collection_ignore(Some("work"), "api")
            .expect("read ignore file");
        assert_eq!(space, "work");
        assert_eq!(content, None);
    });
}

#[test]
fn read_collection_ignore_returns_file_contents() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        write_text_file(&ignore_path, "dist/\n*.tmp\n");

        let (space, content) = engine
            .read_collection_ignore(None, "api")
            .expect("read ignore file");
        assert_eq!(space, "work");
        assert_eq!(content.as_deref(), Some("dist/\n*.tmp"));
    });
}

#[test]
fn add_collection_ignore_pattern_creates_file_and_appends_patterns() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let (space, first) = engine
            .add_collection_ignore_pattern(Some("work"), "api", "dist/")
            .expect("add first pattern");
        assert_eq!(space, "work");
        assert_eq!(first, "dist/");

        let (space, second) = engine
            .add_collection_ignore_pattern(None, "api", "*.tmp")
            .expect("add second pattern");
        assert_eq!(space, "work");
        assert_eq!(second, "*.tmp");

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        let saved = std::fs::read_to_string(ignore_path).expect("read ignore file");
        assert_eq!(saved, "dist/\n*.tmp\n");
    });
}

#[test]
fn add_collection_ignore_pattern_rejects_empty_or_multiline_input() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let empty = engine
            .add_collection_ignore_pattern(Some("work"), "api", "   ")
            .expect_err("empty pattern should fail");
        match KboltError::from(empty) {
            KboltError::InvalidInput(message) => {
                assert!(
                    message.contains("cannot be empty"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }

        let multiline = engine
            .add_collection_ignore_pattern(Some("work"), "api", "dist/\n*.tmp")
            .expect_err("multiline pattern should fail");
        match KboltError::from(multiline) {
            KboltError::InvalidInput(message) => {
                assert!(
                    message.contains("single line"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn remove_collection_ignore_pattern_removes_matches_and_deletes_empty_file() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        write_text_file(&ignore_path, "dist/\n*.tmp\ndist/\n");

        let (space, removed) = engine
            .remove_collection_ignore_pattern(Some("work"), "api", "dist/")
            .expect("remove dist pattern");
        assert_eq!(space, "work");
        assert_eq!(removed, 2);
        let saved = std::fs::read_to_string(&ignore_path).expect("read updated ignore file");
        assert_eq!(saved, "*.tmp\n");

        let (space, removed) = engine
            .remove_collection_ignore_pattern(None, "api", "*.tmp")
            .expect("remove tmp pattern");
        assert_eq!(space, "work");
        assert_eq!(removed, 1);
        assert!(!ignore_path.exists(), "ignore file should be deleted");
    });
}

#[test]
fn remove_collection_ignore_pattern_returns_zero_when_pattern_or_file_is_missing() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let (space, removed) = engine
            .remove_collection_ignore_pattern(Some("work"), "api", "dist/")
            .expect("remove from missing file");
        assert_eq!(space, "work");
        assert_eq!(removed, 0);

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        write_text_file(&ignore_path, "*.tmp\n");

        let (space, removed) = engine
            .remove_collection_ignore_pattern(None, "api", "dist/")
            .expect("remove missing pattern");
        assert_eq!(space, "work");
        assert_eq!(removed, 0);
        let saved = std::fs::read_to_string(ignore_path).expect("read untouched ignore file");
        assert_eq!(saved, "*.tmp\n");
    });
}

#[test]
fn list_collection_ignores_returns_entries_with_pattern_counts_and_space_scope() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "wiki", notes_path);

        write_text_file(
            &engine
                .config()
                .config_dir
                .join("ignores")
                .join("work")
                .join("api.ignore"),
            "dist/\n*.tmp\n",
        );
        write_text_file(
            &engine
                .config()
                .config_dir
                .join("ignores")
                .join("notes")
                .join("wiki.ignore"),
            "# comment\n\nbuild/\n",
        );

        let all = engine
            .list_collection_ignores(None)
            .expect("list all ignores");
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].space, "notes");
        assert_eq!(all[0].collection, "wiki");
        assert_eq!(all[0].pattern_count, 1);
        assert_eq!(all[1].space, "work");
        assert_eq!(all[1].collection, "api");
        assert_eq!(all[1].pattern_count, 2);

        let scoped = engine
            .list_collection_ignores(Some("work"))
            .expect("list scoped ignores");
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].space, "work");
        assert_eq!(scoped[0].collection, "api");
        assert_eq!(scoped[0].pattern_count, 2);
    });
}

#[test]
fn prepare_collection_ignore_edit_creates_missing_ignore_file() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let (space, path) = engine
            .prepare_collection_ignore_edit(None, "api")
            .expect("prepare ignore file");
        assert_eq!(space, "work");
        assert!(
            path.ends_with(std::path::Path::new("ignores/work/api.ignore")),
            "unexpected ignore path: {}",
            path.display()
        );
        assert!(path.exists(), "ignore file should be created");
        let content = std::fs::read_to_string(path).expect("read ignore file");
        assert_eq!(content, "");
    });
}

#[test]
fn list_files_returns_entries_and_applies_prefix_filter() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&work_path.join("docs/guide.md"), "guide text\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let work_space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        let to_deactivate = engine
            .storage()
            .get_document_by_path(collection.id, "docs/guide.md")
            .expect("get docs/guide.md")
            .expect("docs/guide.md should exist");
        engine
            .storage()
            .deactivate_document(to_deactivate.id)
            .expect("deactivate docs file");

        let all = engine
            .list_files(Some("work"), "api", None)
            .expect("list all files");
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].path, "docs/guide.md");
        assert_eq!(all[1].path, "src/lib.rs");
        assert!(!all[0].active);
        assert!(all[1].active);
        assert!(all.iter().all(|file| file.docid.starts_with('#')));
        assert!(all.iter().all(|file| file.chunk_count > 0));
        assert!(all.iter().all(|file| !file.embedded));

        let src_only = engine
            .list_files(Some("work"), "api", Some("src"))
            .expect("list src files");
        assert_eq!(src_only.len(), 1);
        assert_eq!(src_only[0].path, "src/lib.rs");
    });
}

#[test]
fn list_files_without_space_uses_unique_collection_lookup() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");
        engine
            .update(update_options(None, &["api"]))
            .expect("initial update");

        let files = engine
            .list_files(None, "api", None)
            .expect("list files with unique lookup");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/lib.rs");
    });
}

#[test]
fn list_files_errors_for_ambiguous_collection_and_invalid_prefix() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "api", notes_path);

        let err = engine
            .list_files(None, "api", None)
            .expect_err("expected ambiguous collection");
        match KboltError::from(err) {
            KboltError::AmbiguousSpace { collection, spaces } => {
                assert_eq!(collection, "api");
                assert_eq!(spaces, vec!["notes".to_string(), "work".to_string()]);
            }
            other => panic!("unexpected error: {other}"),
        }

        let err = engine
            .list_files(Some("work"), "api", Some("../src"))
            .expect_err("expected invalid prefix");
        match KboltError::from(err) {
            KboltError::InvalidInput(message) => {
                assert!(message.contains("prefix"), "unexpected message: {message}");
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn get_document_by_path_supports_offsets_and_stale_detection() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let file_path = work_path.join("src/lib.rs");
        write_text_file(&file_path, "line-a\nline-b\nline-c\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let sliced = engine
            .get_document(GetRequest {
                locator: Locator::Path("api/src/lib.rs".to_string()),
                space: Some("work".to_string()),
                offset: Some(1),
                limit: Some(1),
            })
            .expect("get sliced document");
        assert_eq!(sliced.path, "api/src/lib.rs");
        assert_eq!(sliced.space, "work");
        assert_eq!(sliced.collection, "api");
        assert_eq!(sliced.content, "line-b");
        assert_eq!(sliced.total_lines, 3);
        assert_eq!(sliced.returned_lines, 1);
        assert!(!sliced.stale);

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "line-a\nline-b\nline-c\nline-d\n");
        let stale = engine
            .get_document(GetRequest {
                locator: Locator::Path("api/src/lib.rs".to_string()),
                space: Some("work".to_string()),
                offset: None,
                limit: None,
            })
            .expect("get stale document");
        assert!(stale.stale);
        assert_eq!(stale.content, "line-a\nline-b\nline-c");
        assert!(!stale.content.contains("line-d"));
        assert_eq!(stale.total_lines, 3);
        assert_eq!(stale.returned_lines, stale.total_lines);
    });
}

#[test]
fn get_document_by_docid_resolves_uniquely_and_honors_optional_space_scope() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "wiki", notes_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&notes_path.join("guide.md"), "notes guide\n");
        engine
            .update(update_options(None, &[]))
            .expect("initial update");

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list files");
        let docid = files[0].docid.clone();

        let doc = engine
            .get_document(GetRequest {
                locator: Locator::DocId(docid.clone()),
                space: None,
                offset: None,
                limit: None,
            })
            .expect("get document by docid");
        assert_eq!(doc.space, "work");
        assert_eq!(doc.collection, "api");
        assert_eq!(doc.path, "api/src/lib.rs");

        let wrong_scope = engine
            .get_document(GetRequest {
                locator: Locator::DocId(docid),
                space: Some("notes".to_string()),
                offset: None,
                limit: None,
            })
            .expect_err("wrong space scope should not resolve docid");
        match KboltError::from(wrong_scope) {
            KboltError::DocumentNotFound { path } => assert!(path.starts_with('#')),
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn get_document_handles_deleted_unreadable_and_ambiguous_docid_cases() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let file_path = work_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        std::fs::remove_file(&file_path).expect("remove file");

        let deleted = engine
            .get_document(GetRequest {
                locator: Locator::Path("api/src/lib.rs".to_string()),
                space: Some("work".to_string()),
                offset: None,
                limit: None,
            })
            .expect("deleted source should still return indexed text");
        assert_eq!(deleted.content, "fn alpha() {}");
        assert!(deleted.stale);

        std::fs::create_dir(&file_path).expect("replace source file with directory");
        let unreadable = engine
            .get_document(GetRequest {
                locator: Locator::Path("api/src/lib.rs".to_string()),
                space: Some("work".to_string()),
                offset: None,
                limit: None,
            })
            .expect_err("unreadable source should surface an io error");
        match KboltError::from(unreadable) {
            KboltError::Io(err) => {
                let message = err.to_string();
                assert!(
                    message.contains("failed to read indexed source"),
                    "unexpected io error: {message}"
                );
                assert!(
                    message.contains("src/lib.rs"),
                    "expected source path in io error: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }

        let work_space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        engine
            .storage()
            .upsert_document(
                collection.id,
                "a.rs",
                "a.rs",
                crate::storage::DocumentTitleSource::Extracted,
                "abc123000000",
                "2026-03-01T10:00:00Z",
            )
            .expect("insert first synthetic hash");
        engine
            .storage()
            .upsert_document(
                collection.id,
                "b.rs",
                "b.rs",
                crate::storage::DocumentTitleSource::Extracted,
                "abc123999999",
                "2026-03-01T10:01:00Z",
            )
            .expect("insert second synthetic hash");

        let ambiguous = engine
            .get_document(GetRequest {
                locator: Locator::DocId("#abc123".to_string()),
                space: Some("work".to_string()),
                offset: None,
                limit: None,
            })
            .expect_err("ambiguous docid should fail");
        match KboltError::from(ambiguous) {
            KboltError::InvalidInput(message) => {
                assert!(
                    message.contains("ambiguous"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn get_document_errors_when_canonical_text_is_missing() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("guide.md"), "alpha canonical body\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let document = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query document")
            .expect("document exists");
        {
            let conn = rusqlite::Connection::open(engine.config().cache_dir.join("meta.sqlite"))
                .expect("open metadata db");
            conn.execute(
                "DELETE FROM document_texts WHERE doc_id = ?1",
                [document.id],
            )
            .expect("delete canonical text");
        }

        let err = engine
            .get_document(GetRequest {
                locator: Locator::Path("api/guide.md".to_string()),
                space: Some("work".to_string()),
                offset: None,
                limit: None,
            })
            .expect_err("missing canonical text should fail explicitly");
        assert!(
            err.to_string().contains("missing persisted canonical text"),
            "unexpected error: {err}"
        );
    });
}

#[test]
fn multi_get_respects_max_files_and_preserves_locator_order() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("a.md"), "a\n");
        write_text_file(&work_path.join("b.md"), "bb\n");
        write_text_file(&work_path.join("c.md"), "ccc\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let result = engine
            .multi_get(MultiGetRequest {
                locators: vec![
                    Locator::Path("api/a.md".to_string()),
                    Locator::Path("api/b.md".to_string()),
                    Locator::Path("api/c.md".to_string()),
                ],
                space: Some("work".to_string()),
                max_files: 2,
                max_bytes: 1024,
            })
            .expect("run multi_get");

        assert_eq!(result.resolved_count, 3);
        assert_eq!(result.documents.len(), 2);
        assert_eq!(result.documents[0].path, "api/a.md");
        assert_eq!(result.documents[1].path, "api/b.md");
        assert_eq!(result.omitted.len(), 1);
        assert_eq!(result.omitted[0].path, "api/c.md");
        assert_eq!(result.omitted[0].reason, OmitReason::MaxFiles);
    });
}

#[test]
fn multi_get_respects_max_bytes_and_supports_mixed_locators() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("a.md"), "alpha\n");
        write_text_file(&work_path.join("b.md"), "beta\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list files");
        let docid = files
            .iter()
            .find(|entry| entry.path == "a.md")
            .expect("a.md entry should exist")
            .docid
            .clone();

        let result = engine
            .multi_get(MultiGetRequest {
                locators: vec![Locator::DocId(docid), Locator::Path("api/b.md".to_string())],
                space: Some("work".to_string()),
                max_files: 10,
                max_bytes: 7,
            })
            .expect("run multi_get");

        assert_eq!(result.resolved_count, 2);
        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.documents[0].path, "api/a.md");
        assert_eq!(result.omitted.len(), 1);
        assert_eq!(result.omitted[0].path, "api/b.md");
        assert_eq!(result.omitted[0].reason, OmitReason::MaxBytes);
        assert!(result.warnings.is_empty());
    });
}

#[test]
fn multi_get_returns_deleted_sources_as_stale_and_surfaces_unreadable_sources() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let existing = work_path.join("a.md");
        let deleted = work_path.join("b.md");
        let unreadable = work_path.join("c.md");
        write_text_file(&existing, "alpha\n");
        write_text_file(&deleted, "beta\n");
        write_text_file(&unreadable, "gamma\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        std::fs::remove_file(&deleted).expect("remove b.md");
        std::fs::remove_file(&unreadable).expect("remove c.md");
        std::fs::create_dir(&unreadable).expect("replace c.md with directory");

        let result = engine
            .multi_get(MultiGetRequest {
                locators: vec![
                    Locator::Path("api/a.md".to_string()),
                    Locator::Path("api/b.md".to_string()),
                ],
                space: Some("work".to_string()),
                max_files: 10,
                max_bytes: 51_200,
            })
            .expect("run multi_get");

        assert_eq!(result.resolved_count, 2);
        assert_eq!(result.documents.len(), 2);
        assert_eq!(result.documents[0].path, "api/a.md");
        assert_eq!(result.documents[1].path, "api/b.md");
        assert_eq!(result.documents[1].content, "beta");
        assert!(result.documents[1].stale);
        assert!(result.omitted.is_empty());
        assert!(result.warnings.is_empty());

        let err = engine
            .multi_get(MultiGetRequest {
                locators: vec![Locator::Path("api/c.md".to_string())],
                space: Some("work".to_string()),
                max_files: 10,
                max_bytes: 51_200,
            })
            .expect_err("unreadable source should surface an io error");
        match KboltError::from(err) {
            KboltError::Io(err) => {
                let message = err.to_string();
                assert!(
                    message.contains("failed to read indexed source"),
                    "unexpected io error: {message}"
                );
                assert!(
                    message.contains("c.md"),
                    "expected source path in io error: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn multi_get_skips_missing_and_invalid_locators_with_warnings() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("a.md"), "alpha\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let result = engine
            .multi_get(MultiGetRequest {
                locators: vec![
                    Locator::Path("api/a.md".to_string()),
                    Locator::Path("api/missing.md".to_string()),
                    Locator::Path("missing-slash".to_string()),
                    Locator::DocId("#invalid".to_string()),
                ],
                space: Some("work".to_string()),
                max_files: 10,
                max_bytes: 51_200,
            })
            .expect("run multi_get");

        assert_eq!(result.resolved_count, 1);
        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.documents[0].path, "api/a.md");
        assert!(result.omitted.is_empty());
        assert_eq!(result.warnings.len(), 3);
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.contains("api/missing.md")));
        assert!(result.warnings.iter().any(|warning| warning
            .contains("invalid locator: path locator must be '<collection>/<path>'")));
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.contains("#invalid")));
    });
}

#[test]
fn search_keyword_returns_ranked_results_for_targeted_collection() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let noise_path = root.path().join("work-noise");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        std::fs::create_dir_all(&noise_path).expect("create noise dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "work", "noise", noise_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "fn alpha_search_term() {}\n");
        write_text_file(
            &noise_path.join("strong.md"),
            &std::iter::repeat_n("alpha_search_term", 100)
                .collect::<Vec<_>>()
                .join(" "),
        );
        engine
            .update(update_options(Some("work"), &["api", "noise"]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "alpha_search_term".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 1,
                min_score: 0.0,
                no_rerank: false,
                debug: true,
            })
            .expect("run keyword search");

        assert_eq!(response.effective_mode, SearchMode::Keyword);
        assert_eq!(response.query, "alpha_search_term");
        assert!(!response.results.is_empty(), "expected at least one result");
        let first = &response.results[0];
        assert_eq!(first.space, "work");
        assert_eq!(first.collection, "api");
        assert!(first.path.starts_with("api/"));
        assert!(first.docid.starts_with('#'));
        assert!(first.text.contains("alpha_search_term"));
        assert!(first.score >= 0.0 && first.score <= 1.0);
        assert!(response.staleness_hint.is_some());
        let signals = first.signals.as_ref().expect("debug signals");
        assert!(signals.bm25.is_some());
        assert!(signals.dense.is_none());
        assert!(signals.reranker.is_none());
    });
}

#[test]
fn search_keyword_includes_neighbor_chunks_for_context() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let left = std::iter::repeat_n("leftctx", 300)
            .collect::<Vec<_>>()
            .join(" ");
        let middle = std::iter::repeat_n("targetonly", 300)
            .collect::<Vec<_>>()
            .join(" ");
        let right = std::iter::repeat_n("rightctx", 300)
            .collect::<Vec<_>>()
            .join(" ");
        let markdown = format!("# Title\n\n{left}\n\n{middle}\n\n{right}\n");
        let file_path = work_path.join("docs/guide.md");
        write_text_file(&file_path, &markdown);
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        write_text_file(&file_path, "mutated source should not be used\n");

        let response = engine
            .search(SearchRequest {
                query: "targetonly".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 5,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect("run keyword search");

        assert!(!response.results.is_empty(), "expected at least one result");
        let first = &response.results[0];
        assert!(first.text.contains("targetonly"));
        assert!(
            first.text.contains("leftctx"),
            "neighbor window should include previous chunk"
        );
        assert!(
            first.text.contains("rightctx"),
            "neighbor window should include next chunk"
        );
        assert!(
            !first.text.contains("mutated source"),
            "neighbor snippet should be hydrated from canonical text"
        );
    });
}

#[test]
fn search_keyword_uses_profile_neighbor_window_for_context() {
    with_kbolt_space_env(None, || {
        let root = tempdir().expect("create temp root");
        let root_path = root.path().to_path_buf();
        std::mem::forget(root);
        let config_dir = root_path.join("config");
        let cache_dir = root_path.join("cache");
        let storage = Storage::new(&cache_dir).expect("create storage");
        let mut config = base_test_config(config_dir, cache_dir);
        config.chunking.defaults.neighbor_window = 1;
        let mut md_policy = config.chunking.defaults.clone();
        md_policy.target_tokens = 128;
        md_policy.soft_max_tokens = 128;
        md_policy.hard_max_tokens = 128;
        md_policy.boundary_overlap_tokens = 0;
        md_policy.neighbor_window = 0;
        config.chunking.profiles.insert("md".to_string(), md_policy);
        let engine = Engine::from_parts(storage, config);
        engine.add_space("work", None).expect("add work");

        let work_path = root_path.join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let left = std::iter::repeat_n("leftprofilectx", 100)
            .collect::<Vec<_>>()
            .join(" ");
        let middle = std::iter::repeat_n("profiletargetonly", 100)
            .collect::<Vec<_>>()
            .join(" ");
        let right = std::iter::repeat_n("rightprofilectx", 100)
            .collect::<Vec<_>>()
            .join(" ");
        write_text_file(
            &work_path.join("docs/guide.md"),
            &format!("# Title\n\n{left}\n\n{middle}\n\n{right}\n"),
        );
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "profiletargetonly".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 5,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect("run keyword search");

        assert!(!response.results.is_empty(), "expected at least one result");
        let first = &response.results[0];
        assert!(first.text.contains("profiletargetonly"));
        assert!(
            !first.text.contains("leftprofilectx"),
            "md profile neighbor_window=0 should suppress previous chunk"
        );
        assert!(
            !first.text.contains("rightprofilectx"),
            "md profile neighbor_window=0 should suppress next chunk"
        );
    });
}

#[test]
fn search_keyword_hydrates_result_text_when_source_file_is_missing() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let file_path = work_path.join("guide.md");
        write_text_file(&file_path, "alpha canonical result\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        std::fs::remove_file(&file_path).expect("remove source file after indexing");

        let response = engine
            .search(SearchRequest {
                query: "alpha".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 5,
                min_score: 0.0,
                no_rerank: true,
                debug: false,
            })
            .expect("search should hydrate from canonical text");

        assert_eq!(response.results.len(), 1);
        assert!(response.results[0].text.contains("alpha canonical result"));
    });
}

#[test]
fn update_get_and_search_use_extracted_html_text() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let file_path = work_path.join("docs/page.html");
        write_text_file(
            &file_path,
            r#"<!doctype html>
<html>
  <head>
    <title>Guide Title</title>
    <script>ignored_script_token</script>
    <style>.noise { color: red; }</style>
  </head>
  <body>
    <h1>Visible Guide</h1>
    <p>alpha <strong>htmltarget</strong> canonical body.</p>
    <table><tr><td>tabletarget visible cell</td></tr></table>
    <div hidden>secret hiddenword</div>
  </body>
</html>"#,
        );

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update html");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let document = engine
            .storage()
            .get_document_by_path(collection.id, "docs/page.html")
            .expect("query document")
            .expect("document exists");
        assert_eq!(document.title, "Guide Title");

        let indexed = engine
            .get_document(GetRequest {
                locator: Locator::Path("api/docs/page.html".to_string()),
                space: Some("work".to_string()),
                offset: None,
                limit: None,
            })
            .expect("get indexed html text");
        assert!(indexed.content.contains("Visible Guide"));
        assert!(indexed.content.contains("alpha htmltarget canonical body."));
        assert!(indexed.content.contains("tabletarget visible cell"));
        assert!(!indexed.content.contains("<p>"));
        assert!(!indexed.content.contains("ignored_script_token"));
        assert!(!indexed.content.contains("hiddenword"));

        let response = engine
            .search(SearchRequest {
                query: "htmltarget".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 5,
                min_score: 0.0,
                no_rerank: true,
                debug: false,
            })
            .expect("search html");

        assert_eq!(response.results.len(), 1);
        assert!(response.results[0]
            .text
            .contains("alpha htmltarget canonical body."));
        assert!(!response.results[0].text.contains("<strong>"));
        assert!(!response.results[0].text.contains("ignored_script_token"));
        assert!(!response.results[0].text.contains("hiddenword"));

        let hidden_response = engine
            .search(SearchRequest {
                query: "hiddenword".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 5,
                min_score: 0.0,
                no_rerank: true,
                debug: false,
            })
            .expect("search hidden html");
        assert!(hidden_response.results.is_empty());
    });
}

#[test]
fn update_get_and_search_use_extracted_pdf_text() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let file_path = work_path.join("papers/guide.pdf");
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent).expect("create pdf parent");
        }
        std::fs::write(
            &file_path,
            crate::ingest::pdf::simple_pdf_fixture(
                "alpha pdftarget canonical body.\nSecond PDF line.",
            ),
        )
        .expect("write pdf fixture");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update pdf");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let indexed = engine
            .get_document(GetRequest {
                locator: Locator::Path("api/papers/guide.pdf".to_string()),
                space: Some("work".to_string()),
                offset: None,
                limit: None,
            })
            .expect("get indexed pdf text");
        assert!(indexed.content.contains("alpha pdftarget canonical body."));
        assert!(indexed.content.contains("Second PDF line."));
        assert!(!indexed.content.contains("%PDF-1.4"));

        let response = engine
            .search(SearchRequest {
                query: "pdftarget".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 5,
                min_score: 0.0,
                no_rerank: true,
                debug: false,
            })
            .expect("search pdf");

        assert_eq!(response.results.len(), 1);
        assert!(response.results[0]
            .text
            .contains("alpha pdftarget canonical body."));
        assert!(!response.results[0].text.contains("%PDF-1.4"));
    });
}

#[test]
fn update_invalidates_stale_empty_pdf_generation_before_mtime_skip() {
    with_kbolt_space_env(None, || {
        let engine = test_engine();
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let file_path = work_path.join("papers/scan.pdf");
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent).expect("create pdf parent");
        }
        let pdf_bytes = crate::ingest::pdf::simple_pdf_fixture("");
        std::fs::write(&file_path, &pdf_bytes).expect("write empty-text pdf fixture");

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let modified =
            super::modified_token(&std::fs::metadata(&file_path).expect("read pdf metadata"))
                .expect("format pdf mtime");
        let source_hash = super::sha256_hex(&pdf_bytes);
        let text_hash = super::sha256_hex(b"");
        let policy = &engine.config().chunking.defaults;
        let stale_generation_key = format!(
            "extractor=pdf:v1;chunk=target:{}:soft:{}:hard:{}:overlap:{}:neighbors:{}:prefix:{}",
            policy.target_tokens,
            policy.soft_max_tokens,
            policy.hard_max_tokens,
            policy.boundary_overlap_tokens,
            policy.neighbor_window,
            policy.contextual_prefix
        );
        let doc_id = engine
            .storage()
            .upsert_document(
                collection.id,
                "papers/scan.pdf",
                "scan",
                crate::storage::DocumentTitleSource::FilenameFallback,
                &source_hash,
                &modified,
            )
            .expect("seed old pdf document");
        engine
            .storage()
            .put_document_text(
                doc_id,
                "pdf",
                &source_hash,
                &text_hash,
                &stale_generation_key,
                "",
            )
            .expect("seed stale empty pdf text");

        let report = engine
            .update(verbose_update_options(Some("work"), &["api"]))
            .expect("update stale empty pdf");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.skipped_mtime_docs, 0);
        assert_eq!(report.skipped_hash_docs, 0);
        assert_eq!(report.failed_docs, 1);
        assert!(
            report.decisions.iter().any(|decision| decision.kind
                == UpdateDecisionKind::ExtractFailed
                && decision
                    .detail
                    .as_deref()
                    .is_some_and(|detail| detail.contains("scanned or image-only PDFs"))),
            "expected stale pdf to be re-extracted and fail visibly: {:?}",
            report.decisions
        );
    });
}

#[test]
fn search_errors_when_canonical_text_is_missing() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("guide.md"), "alpha canonical body\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let document = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query document")
            .expect("document exists");
        {
            let conn = rusqlite::Connection::open(engine.config().cache_dir.join("meta.sqlite"))
                .expect("open metadata db");
            conn.execute(
                "DELETE FROM document_texts WHERE doc_id = ?1",
                [document.id],
            )
            .expect("delete canonical text");
        }

        let err = engine
            .search(SearchRequest {
                query: "alpha".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 5,
                min_score: 0.0,
                no_rerank: true,
                debug: false,
            })
            .expect_err("missing canonical text should fail explicitly");
        assert!(
            err.to_string().contains("missing persisted canonical text"),
            "unexpected error: {err}"
        );
    });
}

#[test]
fn search_semantic_returns_dense_ranked_results_when_embedder_is_configured() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_embedder(Arc::new(DeterministicEmbedder));
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let noise_path = root.path().join("work-noise");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        std::fs::create_dir_all(&noise_path).expect("create noise dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "work", "noise", noise_path.clone());

        write_text_file(
            &work_path.join("docs/guide.md"),
            "semantic anchor token appears here with extra words that make the vector farther\n",
        );
        write_text_file(&noise_path.join("exact.md"), "semantic anchor token\n");
        engine
            .update(update_options(Some("work"), &["api", "noise"]))
            .expect("initial update with embeddings");

        let response = engine
            .search(SearchRequest {
                query: "semantic anchor token".to_string(),
                mode: SearchMode::Semantic,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 1,
                min_score: 0.0,
                no_rerank: false,
                debug: true,
            })
            .expect("run semantic search");

        assert_eq!(response.effective_mode, SearchMode::Semantic);
        assert_eq!(response.results.len(), 1);
        let first = &response.results[0];
        assert_eq!(first.space, "work");
        assert_eq!(first.collection, "api");
        assert!(first.text.contains("semantic anchor token"));
        assert!(first.score > 0.0);
        let signals = first.signals.as_ref().expect("debug signals");
        assert!(signals.bm25.is_none());
        assert!(signals.dense.is_some());
        assert!(signals.reranker.is_none());
    });
}

#[test]
fn search_auto_mode_uses_keyword_path_and_scopes_space() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "api", notes_path.clone());

        write_text_file(&work_path.join("a.md"), "space scoped token\n");
        write_text_file(&notes_path.join("a.md"), "space scoped token\n");
        engine
            .update(update_options(None, &[]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "scoped".to_string(),
                mode: SearchMode::Auto,
                space: Some("work".to_string()),
                collections: vec![],
                limit: 10,
                min_score: 0.0,
                no_rerank: true,
                debug: false,
            })
            .expect("run auto search");

        assert_eq!(response.requested_mode, SearchMode::Auto);
        assert_eq!(response.effective_mode, SearchMode::Keyword);
        assert!(response.pipeline.keyword);
        assert!(!response.pipeline.dense);
        assert!(
            response
                .pipeline
                .notices
                .iter()
                .any(|notice| notice.step == kbolt_types::SearchPipelineStep::Dense),
            "expected dense-unavailable notice: {:?}",
            response.pipeline.notices
        );
        assert!(response.results.iter().all(|item| item.space == "work"));
    });
}

#[test]
fn search_auto_mode_uses_hybrid_signals_when_embedder_is_configured() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_embedder(Arc::new(DeterministicEmbedder));
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("guide.md"), "hybrid auto mode token\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "hybrid auto mode token".to_string(),
                mode: SearchMode::Auto,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 10,
                min_score: 0.0,
                no_rerank: false,
                debug: true,
            })
            .expect("run auto search");

        assert_eq!(response.effective_mode, SearchMode::Auto);
        assert!(!response.results.is_empty(), "expected at least one result");
        assert!(
            response.pipeline.notices.iter().any(|notice| {
                notice.step == kbolt_types::SearchPipelineStep::Rerank
                    && notice.reason == kbolt_types::SearchPipelineUnavailableReason::NotConfigured
            }),
            "expected rerank-not-configured notice: {:?}",
            response.pipeline.notices
        );
        let first = &response.results[0];
        let signals = first.signals.as_ref().expect("debug signals");
        assert!(signals.bm25.is_some());
        assert!(signals.dense.is_some());
        assert!(signals.reranker.is_none());
    });
}

#[test]
fn search_auto_without_reranker_does_not_hydrate_extra_rerank_candidates() {
    with_kbolt_space_env(None, || {
        let engine = test_engine();
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(
            &work_path.join("top.md"),
            "anchor anchor anchor anchor anchor top result\n",
        );
        write_text_file(&work_path.join("extra.md"), "anchor extra candidate\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let extra_doc = engine
            .storage()
            .get_document_by_path(collection.id, "extra.md")
            .expect("query extra document")
            .expect("extra document exists");
        {
            let conn = rusqlite::Connection::open(engine.config().cache_dir.join("meta.sqlite"))
                .expect("open metadata db");
            conn.execute(
                "DELETE FROM document_texts WHERE doc_id = ?1",
                [extra_doc.id],
            )
            .expect("delete extra canonical text");
        }

        let response = engine
            .search(SearchRequest {
                query: "anchor".to_string(),
                mode: SearchMode::Auto,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 1,
                min_score: 0.0,
                no_rerank: false,
                debug: true,
            })
            .expect("auto search should not hydrate non-result rerank candidates without reranker");

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].path, "api/top.md");
        assert!(
            response.pipeline.notices.iter().any(|notice| {
                notice.step == kbolt_types::SearchPipelineStep::Rerank
                    && notice.reason == kbolt_types::SearchPipelineUnavailableReason::NotConfigured
            }),
            "expected rerank-not-configured notice: {:?}",
            response.pipeline.notices
        );
    });
}

#[test]
fn search_auto_mode_honors_no_rerank_flag() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_embedder(Arc::new(DeterministicEmbedder));
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(&work_path.join("guide.md"), "hybrid auto mode token\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "hybrid auto mode token".to_string(),
                mode: SearchMode::Auto,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 10,
                min_score: 0.0,
                no_rerank: true,
                debug: true,
            })
            .expect("run auto search");

        assert!(!response.results.is_empty(), "expected at least one result");
        let first = &response.results[0];
        let signals = first.signals.as_ref().expect("debug signals");
        assert!(signals.bm25.is_some());
        assert!(signals.dense.is_some());
        assert!(signals.reranker.is_none());
    });
}

#[test]
fn search_deep_mode_returns_results_with_reranker_signal() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_search_models(
            Some(Arc::new(DeterministicEmbedder)),
            Some(Arc::new(DeterministicReranker)),
            Some(Arc::new(DeterministicExpander)),
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(
            &work_path.join("guide.md"),
            "# Setup\n\nThis document explains setup steps and install details.\n",
        );
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "setup install".to_string(),
                mode: SearchMode::Deep,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 10,
                min_score: 0.0,
                no_rerank: false,
                debug: true,
            })
            .expect("run deep search");

        assert_eq!(response.effective_mode, SearchMode::Deep);
        assert!(!response.results.is_empty(), "expected at least one result");
        let first = &response.results[0];
        let signals = first.signals.as_ref().expect("debug signals");
        assert!(signals.bm25.is_some() || signals.dense.is_some());
        assert!(signals.reranker.is_some());
    });
}

#[test]
fn search_deep_mode_filters_duplicate_and_original_query_expansions() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_search_models(
            None,
            None,
            Some(Arc::new(StaticExpander {
                items: vec![
                    "unrelated topic".to_string(),
                    "  setup install guide  ".to_string(),
                    "setup install guide".to_string(),
                ],
            })),
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        write_text_file(
            &work_path.join("guide.md"),
            "# Setup\n\nThis document explains setup install steps.\n",
        );
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "unrelated topic".to_string(),
                mode: SearchMode::Deep,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 10,
                min_score: 0.0,
                no_rerank: true,
                debug: false,
            })
            .expect("deep search should filter duplicate and original-query expansions");

        assert_eq!(response.effective_mode, SearchMode::Deep);
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].path, "api/guide.md");
        assert!(response.pipeline.expansion);
    });
}

#[test]
fn search_auto_mode_keeps_unreranked_tail_below_reranked_pool() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_search_models_and_ranking(
            Some(Arc::new(DeterministicEmbedder)),
            Some(Arc::new(ConstantReranker(0.05))),
            None,
            RankingConfig::default(),
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        for index in 0..31 {
            write_text_file(
                &work_path.join(format!("doc-{index:02}.md")),
                &format!("shared token document {index}\n"),
            );
        }
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "shared token".to_string(),
                mode: SearchMode::Auto,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 31,
                min_score: 0.0,
                no_rerank: false,
                debug: true,
            })
            .expect("run auto search");

        assert_eq!(response.results.len(), 31);

        let reranked_prefix_len = response
            .results
            .iter()
            .take_while(|result| {
                result
                    .signals
                    .as_ref()
                    .expect("debug signals")
                    .reranker
                    .is_some()
            })
            .count();

        assert_eq!(reranked_prefix_len, 30);
        assert!(
            response.results[reranked_prefix_len..]
                .iter()
                .all(|result| {
                    result
                        .signals
                        .as_ref()
                        .expect("debug signals")
                        .reranker
                        .is_none()
                }),
            "expected all non-reranked candidates after the reranked pool"
        );
        assert!(
            response.results[29].score > response.results[30].score,
            "expected untouched tail candidate to score below reranked pool"
        );
    });
}

#[test]
fn search_rerank_sends_one_representative_per_document() {
    use std::sync::Mutex;

    struct RecordingReranker {
        calls: Mutex<Vec<Vec<String>>>,
        score: f32,
    }

    impl crate::models::Reranker for RecordingReranker {
        fn rerank(&self, _query: &str, docs: &[String]) -> crate::Result<Vec<f32>> {
            self.calls
                .lock()
                .unwrap()
                .push(docs.iter().cloned().collect());
            Ok(vec![self.score; docs.len()])
        }
    }

    with_kbolt_space_env(None, || {
        let recording_reranker = Arc::new(RecordingReranker {
            calls: Mutex::new(Vec::new()),
            score: 0.5,
        });
        let engine = test_engine_with_search_models_and_ranking(
            Some(Arc::new(DeterministicEmbedder)),
            Some(recording_reranker.clone()),
            None,
            RankingConfig::default(),
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        // Write one large document that will produce multiple chunks, plus
        // a second small document. Both mention the query term.
        let mut big_body = String::new();
        for i in 0..120 {
            big_body.push_str(&format!(
                "Section {i}: This section discusses the search query topic in detail, covers retrieval scoring behavior, and repeats the search query topic so the document spans multiple chunks.\n\n"
            ));
        }
        write_text_file(&work_path.join("big.md"), &big_body);
        write_text_file(
            &work_path.join("small.md"),
            "This small document also discusses the search query topic.\n",
        );

        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let response = engine
            .search(SearchRequest {
                query: "search query topic".to_string(),
                mode: SearchMode::Auto,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 10,
                min_score: 0.0,
                no_rerank: false,
                debug: true,
            })
            .expect("run auto search");

        let calls = recording_reranker.calls.lock().unwrap();
        assert_eq!(calls.len(), 1, "expected exactly one rerank call");

        // The reranker should receive exactly 2 inputs: one per unique document.
        let rerank_inputs = &calls[0];
        assert_eq!(
            rerank_inputs.len(),
            2,
            "expected one representative per document, got {}",
            rerank_inputs.len()
        );

        // All returned results should have a reranker score since both docs
        // were in the rerank pool.
        for result in &response.results {
            let signals = result.signals.as_ref().expect("debug signals");
            assert!(
                signals.reranker.is_some(),
                "expected all chunks to inherit document-level reranker score"
            );
        }

        let big_results = response
            .results
            .iter()
            .filter(|result| result.path.ends_with("big.md"))
            .collect::<Vec<_>>();
        assert!(
            big_results.len() >= 2,
            "expected multiple chunks from the large document in the result set"
        );
        assert!(
            big_results[0].score > big_results[1].score,
            "expected within-document chunk ordering to keep retrieval differentiation after reranking"
        );
        assert_eq!(
            big_results[0]
                .signals
                .as_ref()
                .expect("debug signals")
                .reranker,
            big_results[1]
                .signals
                .as_ref()
                .expect("debug signals")
                .reranker
        );
    });
}

#[test]
fn search_rerank_uses_canonical_body_with_prefix_and_result_text_without_prefix() {
    use std::sync::Mutex;

    struct RecordingReranker {
        calls: Mutex<Vec<Vec<String>>>,
    }

    impl crate::models::Reranker for RecordingReranker {
        fn rerank(&self, _query: &str, docs: &[String]) -> crate::Result<Vec<f32>> {
            self.calls
                .lock()
                .unwrap()
                .push(docs.iter().cloned().collect());
            Ok(vec![1.0; docs.len()])
        }
    }

    with_kbolt_space_env(None, || {
        let recording_reranker = Arc::new(RecordingReranker {
            calls: Mutex::new(Vec::new()),
        });
        let engine_root = tempdir().expect("create engine temp root");
        let config_dir = engine_root.path().join("config");
        let cache_dir = engine_root.path().join("cache");
        let storage = Storage::new(&cache_dir).expect("create storage");
        let mut config = base_test_config(config_dir, cache_dir);
        config.chunking.defaults.contextual_prefix = false;
        let engine = Engine::from_parts_with_models(
            storage,
            config,
            Some(Arc::new(DeterministicEmbedder)),
            Some(recording_reranker.clone()),
            None,
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let file_path = work_path.join("guide.md");
        write_text_file(
            &file_path,
            "# Setup\n\nalpha canonical rerank body appears here\n",
        );
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        std::fs::remove_file(&file_path).expect("remove source file after indexing");

        let response = engine
            .search(SearchRequest {
                query: "alpha canonical rerank".to_string(),
                mode: SearchMode::Auto,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 5,
                min_score: 0.0,
                no_rerank: false,
                debug: true,
            })
            .expect("search with rerank should hydrate canonical text");

        let calls = recording_reranker.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), 1);
        let rerank_input = &calls[0][0];
        assert!(
            rerank_input.contains("title: Setup"),
            "rerank input should include explicit title prefix: {rerank_input:?}"
        );
        assert!(
            rerank_input.contains("heading: Setup"),
            "rerank input should include explicit heading prefix: {rerank_input:?}"
        );
        assert!(rerank_input.contains("alpha canonical rerank body"));

        assert_eq!(response.results.len(), 1);
        assert!(response.results[0]
            .text
            .contains("alpha canonical rerank body"));
        assert!(
            !response.results[0].text.contains("title:"),
            "result snippet should not include indexing/rerank prefix"
        );
        assert!(
            !response.results[0].text.contains("heading:"),
            "result snippet should not include indexing/rerank prefix"
        );
        assert!(response.results[0]
            .signals
            .as_ref()
            .expect("debug signals")
            .reranker
            .is_some());
    });
}

#[test]
fn initial_search_candidate_limit_uses_configured_rerank_expansion() {
    let engine = test_engine();

    assert_eq!(
        engine.initial_search_candidate_limit(&SearchMode::Auto, 10, true),
        40
    );
    assert_eq!(
        engine.initial_search_candidate_limit(&SearchMode::Deep, 10, true),
        40
    );
    assert_eq!(
        engine.initial_search_candidate_limit(&SearchMode::Auto, 10, false),
        10
    );
}

#[test]
fn search_keyword_refills_after_deactivated_result_is_filtered() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let strong = work_path.join("strong.md");
        let weak = work_path.join("weak.md");
        write_text_file(&strong, "token token token token token\n");
        write_text_file(&weak, "token\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        std::fs::remove_file(&strong).expect("remove strong file");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("deactivate removed file");

        let response = engine
            .search(SearchRequest {
                query: "token".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 1,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect("run keyword search");

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].path, "api/weak.md");
    });
}

#[test]
fn whole_space_search_scope_avoids_filters_until_inactive_docs_exist() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());

        let keep = work_path.join("keep.md");
        let stale = work_path.join("stale.md");
        write_text_file(&keep, "keep token\n");
        write_text_file(&stale, "stale token\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let targets = engine
            .resolve_targets(TargetScope {
                space: Some("work"),
                collections: &[],
            })
            .expect("resolve whole-space targets");
        let scopes = engine
            .search_target_scopes(&targets, false)
            .expect("build whole-space search scope");
        assert_eq!(scopes.len(), 1);
        assert!(!scopes[0].filtered);
        assert!(scopes[0].document_ids.is_empty());
        assert_eq!(scopes[0].chunk_count, 2);

        let collection_scopes = engine
            .search_target_scopes(&targets, true)
            .expect("build collection-filtered search scope");
        assert!(collection_scopes[0].filtered);
        assert_eq!(collection_scopes[0].document_ids.len(), 2);

        std::fs::remove_file(&stale).expect("remove stale file");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("deactivate stale document");

        let targets = engine
            .resolve_targets(TargetScope {
                space: Some("work"),
                collections: &[],
            })
            .expect("resolve whole-space targets after deactivate");
        let scopes = engine
            .search_target_scopes(&targets, false)
            .expect("build whole-space search scope after deactivate");
        assert!(scopes[0].filtered);
        assert_eq!(scopes[0].document_ids.len(), 1);
        assert_eq!(scopes[0].chunk_count, 1);
    });
}

#[test]
fn search_validates_semantic_and_collection_scope() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "api", notes_path);

        let semantic_err = engine
            .search(SearchRequest {
                query: "test".to_string(),
                mode: SearchMode::Semantic,
                space: None,
                collections: vec![],
                limit: 10,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect_err("semantic mode should require embedder");
        match KboltError::from(semantic_err) {
            KboltError::InvalidInput(message) => {
                assert!(
                    message.contains("semantic search requires a configured embedder role"),
                    "unexpected message: {message}"
                );
                assert!(
                    message.contains("[roles.embedder]"),
                    "expected role-binding guidance in message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }

        let deep = engine
            .search(SearchRequest {
                query: "test".to_string(),
                mode: SearchMode::Deep,
                space: None,
                collections: vec![],
                limit: 10,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect_err("deep mode should require expander configuration");
        match KboltError::from(deep) {
            KboltError::InvalidInput(message) => {
                assert!(
                    message.contains("deep search needs the optional expander"),
                    "unexpected message: {message}"
                );
                assert!(
                    message.contains("kbolt local enable deep"),
                    "expected managed expander guidance in message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }

        let ambiguous_err = engine
            .search(SearchRequest {
                query: "test".to_string(),
                mode: SearchMode::Keyword,
                space: None,
                collections: vec!["api".to_string()],
                limit: 10,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect_err("ambiguous collection should error");
        match KboltError::from(ambiguous_err) {
            KboltError::AmbiguousSpace { collection, spaces } => {
                assert_eq!(collection, "api");
                assert_eq!(spaces, vec!["notes".to_string(), "work".to_string()]);
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn resolve_update_targets_returns_all_collections_when_unscoped() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");

        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "wiki", notes_path);

        let targets = engine
            .resolve_update_targets(&update_options(None, &[]))
            .expect("resolve update targets");
        assert_eq!(targets.len(), 2);
        assert!(targets
            .iter()
            .any(|target| target.space == "work" && target.collection.name == "api"));
        assert!(targets
            .iter()
            .any(|target| target.space == "notes" && target.collection.name == "wiki"));
    });
}

#[test]
fn resolve_update_targets_scopes_to_requested_space() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");

        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "wiki", notes_path);

        let targets = engine
            .resolve_update_targets(&update_options(Some("work"), &[]))
            .expect("resolve update targets");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].space, "work");
        assert_eq!(targets[0].collection.name, "api");
    });
}

#[test]
fn resolve_update_targets_named_collection_uses_unique_lookup() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let targets = engine
            .resolve_update_targets(&update_options(None, &["api"]))
            .expect("resolve update targets");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].space, "work");
        assert_eq!(targets[0].collection.name, "api");
    });
}

#[test]
fn resolve_update_targets_named_collection_errors_on_ambiguity() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");

        add_collection_fixture(&engine, "work", "api", work_path);
        add_collection_fixture(&engine, "notes", "api", notes_path);

        let err = engine
            .resolve_update_targets(&update_options(None, &["api"]))
            .expect_err("expected ambiguous collection");
        match KboltError::from(err) {
            KboltError::AmbiguousSpace { collection, spaces } => {
                assert_eq!(collection, "api");
                assert_eq!(spaces, vec!["notes".to_string(), "work".to_string()]);
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn resolve_update_targets_named_collection_honors_default_space_precedence() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(Some("work"));
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let notes_path = root.path().join("notes-api");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "notes", "api", notes_path);

        let err = engine
            .resolve_update_targets(&update_options(None, &["api"]))
            .expect_err("default precedence should look in work first");
        match KboltError::from(err) {
            KboltError::CollectionNotFound { name } => assert_eq!(name, "api"),
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn resolve_update_targets_deduplicates_repeated_collection_names() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        add_collection_fixture(&engine, "work", "api", work_path);

        let targets = engine
            .resolve_update_targets(&update_options(Some("work"), &["api", "api"]))
            .expect("resolve update targets");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].space, "work");
        assert_eq!(targets[0].collection.name, "api");
    });
}

#[test]
fn resolve_update_targets_rejects_empty_collection_names() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        let err = engine
            .resolve_update_targets(&update_options(None, &[""]))
            .expect_err("empty collection names should be rejected");
        match KboltError::from(err) {
            KboltError::InvalidInput(message) => {
                assert!(
                    message.contains("cannot be empty"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn update_indexes_new_document_and_skips_unchanged_mtime() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("first update");
        assert_eq!(first.scanned_docs, 1);
        assert_eq!(first.added_docs, 1);
        assert_eq!(first.updated_docs, 0);
        assert_eq!(first.deactivated_docs, 0);
        assert!(
            first.errors.is_empty(),
            "unexpected errors: {:?}",
            first.errors
        );

        let hits = engine
            .storage()
            .query_bm25("work", "alpha", &[("body", 1.0)], 10)
            .expect("query bm25");
        assert_eq!(hits.len(), 1);

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("second update");
        assert_eq!(second.scanned_docs, 1);
        assert_eq!(second.skipped_mtime_docs, 1);
        assert_eq!(second.added_docs, 0);
        assert_eq!(second.updated_docs, 0);
        assert_eq!(second.deactivated_docs, 0);
    });
}

#[test]
fn update_rebuilds_matching_file_when_canonical_text_is_missing() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("first update");
        assert_eq!(first.added_docs, 1);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let document = engine
            .storage()
            .get_document_by_path(collection.id, "src/lib.rs")
            .expect("query document")
            .expect("document exists");
        {
            let conn = rusqlite::Connection::open(engine.config().cache_dir.join("meta.sqlite"))
                .expect("open metadata db");
            conn.execute(
                "DELETE FROM document_texts WHERE doc_id = ?1",
                [document.id],
            )
            .expect("delete canonical text");
        }
        assert!(!engine
            .storage()
            .has_document_text(document.id)
            .expect("canonical text should be absent"));

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("second update");
        assert_eq!(second.scanned_docs, 1);
        assert_eq!(second.skipped_mtime_docs, 0);
        assert_eq!(second.skipped_hash_docs, 0);
        assert_eq!(second.updated_docs, 1);
        assert!(
            second.errors.is_empty(),
            "unexpected errors: {:?}",
            second.errors
        );
        assert_eq!(
            engine
                .storage()
                .get_document_text(document.id)
                .expect("canonical text restored")
                .text,
            "fn alpha() {}\n"
        );
    });
}

#[test]
fn update_clears_fts_dirty_for_empty_canonical_document() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("empty.txt"), "");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update empty document");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let dirty = engine
            .storage()
            .get_fts_dirty_documents()
            .expect("load dirty docs");
        assert!(dirty.is_empty(), "empty document should not remain dirty");
    });
}

#[test]
fn update_replacing_nonempty_document_with_empty_text_removes_old_sidecars() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_embedder(Arc::new(DeterministicEmbedder));
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("guide.md");
        write_text_file(&file_path, "oldtoken body\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        assert_eq!(first.added_docs, 1);
        assert_eq!(first.embedded_chunks, 1);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let original_doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query original document")
            .expect("document exists");
        assert_eq!(
            engine
                .storage()
                .query_bm25("work", "oldtoken", &[("body", 1.0)], 10)
                .expect("query old projection")
                .len(),
            1
        );
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(space.id))
                .expect("count embedded chunks"),
            1
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors"),
            1
        );

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "");

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("replace with empty document");
        assert_eq!(second.scanned_docs, 1);
        assert_eq!(second.updated_docs, 1);
        assert!(
            second.errors.is_empty(),
            "unexpected errors: {:?}",
            second.errors
        );

        let updated_doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query updated document")
            .expect("document exists");
        assert_eq!(updated_doc.id, original_doc.id);
        assert_eq!(
            engine
                .storage()
                .get_document_text(updated_doc.id)
                .expect("load updated canonical text")
                .text,
            ""
        );
        assert!(engine
            .storage()
            .get_chunks_for_document(updated_doc.id)
            .expect("load updated chunks")
            .is_empty());
        assert!(engine
            .storage()
            .query_bm25("work", "oldtoken", &[("body", 1.0)], 10)
            .expect("query removed old projection")
            .is_empty());
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(space.id))
                .expect("count embedded chunks"),
            0
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors"),
            0
        );
        assert!(engine
            .storage()
            .get_fts_dirty_documents()
            .expect("load dirty docs")
            .is_empty());
    });
}

#[test]
fn update_replays_fts_dirty_documents_before_mtime_fast_path() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("first update");
        assert_eq!(first.scanned_docs, 1);
        assert_eq!(first.added_docs, 1);
        assert!(
            first.errors.is_empty(),
            "unexpected errors: {:?}",
            first.errors
        );

        let work_space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        let stored_doc = engine
            .storage()
            .get_document_by_path(collection.id, "src/lib.rs")
            .expect("query document")
            .expect("document should exist");
        let chunks = engine
            .storage()
            .get_chunks_for_document(stored_doc.id)
            .expect("load chunks");
        let chunk_ids = chunks.iter().map(|chunk| chunk.id).collect::<Vec<_>>();
        assert!(!chunk_ids.is_empty(), "expected indexed chunks");

        engine
            .storage()
            .delete_tantivy("work", &chunk_ids)
            .expect("delete tantivy entries");
        engine
            .storage()
            .commit_tantivy("work")
            .expect("commit tantivy deletes");
        let removed_hits = engine
            .storage()
            .query_bm25("work", "alpha", &[("body", 1.0)], 10)
            .expect("query bm25 after delete");
        assert!(
            removed_hits.is_empty(),
            "search should be empty before replay, got {} hits",
            removed_hits.len()
        );

        engine
            .storage()
            .upsert_document(
                collection.id,
                &stored_doc.path,
                &stored_doc.title,
                stored_doc.title_source,
                &stored_doc.hash,
                &stored_doc.modified,
            )
            .expect("mark document fts dirty");
        let dirty_before = engine
            .storage()
            .get_fts_dirty_documents()
            .expect("load dirty docs");
        assert_eq!(dirty_before.len(), 1);
        assert_eq!(dirty_before[0].doc_id, stored_doc.id);

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("second update");
        assert!(
            second.errors.is_empty(),
            "unexpected errors: {:?}",
            second.errors
        );

        let replayed_hits = engine
            .storage()
            .query_bm25("work", "alpha", &[("body", 1.0)], 10)
            .expect("query bm25 after replay");
        assert_eq!(replayed_hits.len(), 1);
        assert_eq!(replayed_hits[0].chunk_id, chunk_ids[0]);

        let dirty_after = engine
            .storage()
            .get_fts_dirty_documents()
            .expect("load dirty docs after replay");
        assert!(
            dirty_after.is_empty(),
            "expected replay to clear fts_dirty flags"
        );
    });
}

#[test]
fn update_does_not_replay_fts_dirty_documents_outside_scoped_targets() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "wiki", notes_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "worktoken\n");
        write_text_file(&notes_path.join("docs/guide.md"), "oldtoken\n");
        engine
            .update(update_options(None, &[]))
            .expect("index initial fixtures");

        let notes_space = engine
            .storage()
            .get_space("notes")
            .expect("get notes space");
        let notes_collection = engine
            .storage()
            .get_collection(notes_space.id, "wiki")
            .expect("get notes collection");
        let notes_doc = engine
            .storage()
            .get_document_by_path(notes_collection.id, "docs/guide.md")
            .expect("query notes document")
            .expect("notes document should exist");
        let note_chunks = engine
            .storage()
            .get_chunks_for_document(notes_doc.id)
            .expect("load note chunks");
        let note_chunk_ids = note_chunks.iter().map(|chunk| chunk.id).collect::<Vec<_>>();
        assert!(!note_chunk_ids.is_empty(), "expected note chunks");

        engine
            .storage()
            .delete_tantivy("notes", &note_chunk_ids)
            .expect("delete notes tantivy entries");
        engine
            .storage()
            .commit_tantivy("notes")
            .expect("commit notes tantivy deletes");
        engine
            .storage()
            .upsert_document(
                notes_collection.id,
                &notes_doc.path,
                &notes_doc.title,
                notes_doc.title_source,
                &notes_doc.hash,
                &notes_doc.modified,
            )
            .expect("mark notes document fts dirty");

        let notes_hits_before = engine
            .storage()
            .query_bm25("notes", "oldtoken", &[("body", 1.0)], 10)
            .expect("query notes bm25 before scoped update");
        assert!(
            notes_hits_before.is_empty(),
            "notes Tantivy should be empty"
        );

        let dirty_before = engine
            .storage()
            .get_fts_dirty_documents()
            .expect("load dirty docs before scoped update");
        assert_eq!(dirty_before.len(), 1);
        assert_eq!(dirty_before[0].doc_id, notes_doc.id);

        let scoped = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("run scoped update");
        assert!(
            scoped.errors.is_empty(),
            "unexpected errors: {:?}",
            scoped.errors
        );

        let notes_hits_after = engine
            .storage()
            .query_bm25("notes", "oldtoken", &[("body", 1.0)], 10)
            .expect("query notes bm25 after scoped update");
        assert!(
            notes_hits_after.is_empty(),
            "out-of-scope dirty doc should not be replayed"
        );

        let dirty_after = engine
            .storage()
            .get_fts_dirty_documents()
            .expect("load dirty docs after scoped update");
        assert_eq!(dirty_after.len(), 1);
        assert_eq!(dirty_after[0].doc_id, notes_doc.id);
    });
}

#[test]
fn space_scoped_update_clears_mismatched_dense_state_before_scan() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("first update");
        assert_eq!(first.scanned_docs, 1);
        assert_eq!(first.added_docs, 1);
        assert!(
            first.errors.is_empty(),
            "unexpected errors: {:?}",
            first.errors
        );

        let work_space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "src/lib.rs")
            .expect("query document")
            .expect("document should exist");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");
        assert!(!chunks.is_empty(), "expected indexed chunks");

        engine
            .storage()
            .insert_embeddings(&[(chunks[0].id, "model-a")])
            .expect("insert synthetic embedding row");
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks"),
            1
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors"),
            0
        );

        let second = engine
            .update(update_options(Some("work"), &[]))
            .expect("second update");
        assert_eq!(second.scanned_docs, 1);
        assert_eq!(second.skipped_mtime_docs, 1);
        assert!(
            second.errors.is_empty(),
            "unexpected errors: {:?}",
            second.errors
        );

        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks after reconcile"),
            0
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors after reconcile"),
            0
        );
    });
}

#[test]
fn collection_targeted_update_errors_on_space_level_dense_integrity_issues() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("src/lib.rs"), "fn alpha() {}\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("first update");

        let work_space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "src/lib.rs")
            .expect("query document")
            .expect("document should exist");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");
        assert!(!chunks.is_empty(), "expected indexed chunks");

        engine
            .storage()
            .insert_embeddings(&[(chunks[0].id, "stale-model")])
            .expect("insert stale embedding row");

        let err = engine
            .update(update_options(Some("work"), &["api"]))
            .expect_err("collection-targeted update should reject dense repair");
        assert!(
            err.to_string()
                .contains("collection-targeted update cannot repair space-level vector state"),
            "unexpected error: {err}"
        );
        assert!(
            err.to_string().contains("stale-model"),
            "expected model drift detail in error: {err}"
        );
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks after error"),
            1
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors after error"),
            0
        );
    });
}

#[test]
fn space_scoped_update_clears_dense_state_when_embedding_model_drifts() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("first update");
        assert_eq!(first.scanned_docs, 1);
        assert_eq!(first.added_docs, 1);
        assert!(
            first.errors.is_empty(),
            "unexpected errors: {:?}",
            first.errors
        );

        let work_space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "src/lib.rs")
            .expect("query document")
            .expect("document should exist");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");
        assert!(!chunks.is_empty(), "expected indexed chunks");

        engine
            .storage()
            .insert_embeddings(&[(chunks[0].id, "stale-model")])
            .expect("insert stale embedding row");
        engine
            .storage()
            .batch_insert_usearch("work", &[(chunks[0].id, &[1.0, 0.0])])
            .expect("insert stale usearch vector");

        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks before drift reconcile"),
            1
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors before drift reconcile"),
            1
        );

        let second = engine
            .update(update_options(Some("work"), &[]))
            .expect("second update");
        assert_eq!(second.scanned_docs, 1);
        assert_eq!(second.skipped_mtime_docs, 1);
        assert!(
            second.errors.is_empty(),
            "unexpected errors: {:?}",
            second.errors
        );

        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks after drift reconcile"),
            0
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors after drift reconcile"),
            0
        );
    });
}

#[test]
fn update_embeds_chunks_when_embedder_is_configured() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_embedder(Arc::new(DeterministicEmbedder));
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update with embedder");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert_eq!(report.errors.len(), 0);
        assert!(
            report.embedded_chunks > 0,
            "expected embedding phase to process chunks"
        );

        let work_space = engine.storage().get_space("work").expect("get work space");
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks"),
            report.embedded_chunks
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors"),
            report.embedded_chunks
        );

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list files");
        assert_eq!(files.len(), 1);
        assert!(files[0].embedded, "file should be fully embedded");
    });
}

#[test]
fn update_reuses_fresh_chunk_token_bounds_before_embedding() {
    with_kbolt_space_env(None, || {
        let sizer = Arc::new(CountingCharDocumentSizer::default());
        let engine = test_engine_with_embedding_runtime(
            Arc::new(DeterministicEmbedder),
            sizer.clone(),
            ChunkingConfig::default(),
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());
        write_text_file(&collection_path.join("guide.md"), "alpha beta\n");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update with fresh embedding");

        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert_eq!(report.embedded_chunks, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );
        assert_eq!(
            sizer.call_count(),
            1,
            "fresh chunk should be counted by chunking without duplicate preinsert preflight"
        );
    });
}

#[test]
fn update_still_preflights_embedding_backlog_before_embedding() {
    with_kbolt_space_env(None, || {
        let sizer = Arc::new(CountingCharDocumentSizer::default());
        let engine = test_engine_with_embedding_runtime(
            Arc::new(DeterministicEmbedder),
            sizer.clone(),
            ChunkingConfig::default(),
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());
        write_text_file(&collection_path.join("guide.md"), "alpha beta\n");

        engine
            .update(UpdateOptions {
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                no_embed: true,
                dry_run: false,
                verbose: false,
            })
            .expect("index without embeddings");
        sizer.reset();

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("embed backlog");

        assert_eq!(report.skipped_mtime_docs, 1);
        assert_eq!(report.embedded_chunks, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );
        assert_eq!(
            sizer.call_count(),
            1,
            "backlog chunks should still be preflighted before embedding"
        );
    });
}

#[test]
fn update_isolates_buffered_embedding_failures() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_embedder(Arc::new(SelectiveFailureEmbedder));
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("good.md"), "helpful setup guide\n");
        write_text_file(&collection_path.join("bad.md"), "EMBED_FAIL trigger\n");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update with partial embed failure");

        assert_eq!(report.scanned_docs, 2);
        assert_eq!(report.added_docs, 2);
        assert_eq!(report.failed_docs, 1);
        assert_eq!(report.embedded_chunks, 1);
        assert_eq!(report.errors.len(), 1);
        assert!(
            report.errors[0].path.ends_with("bad.md"),
            "unexpected error path: {:?}",
            report.errors
        );
        assert!(
            report.errors[0].error.contains("simulated embed failure"),
            "unexpected error: {:?}",
            report.errors
        );

        let work_space = engine.storage().get_space("work").expect("get work space");
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks"),
            1
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors"),
            1
        );

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list files");
        assert_eq!(files.len(), 2);
        let good = files
            .iter()
            .find(|file| file.path == "good.md")
            .expect("good file entry");
        let bad = files
            .iter()
            .find(|file| file.path == "bad.md")
            .expect("bad file entry");
        assert!(good.embedded, "good file should be embedded");
        assert!(!bad.embedded, "bad file should remain pending");
    });
}

#[test]
fn update_embeds_backlog_from_canonical_text_when_source_bytes_change() {
    with_kbolt_space_env(None, || {
        let embedder = Arc::new(RecordingEmbedder::default());
        let engine = test_engine_with_embedder(embedder.clone());
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("guide.md");
        write_text_file(&file_path, "alpha stored\n");

        let mut no_embed = update_options(Some("work"), &["api"]);
        no_embed.no_embed = true;
        engine.update(no_embed).expect("index without embeddings");

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let document = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query document")
            .expect("document exists");

        write_text_file(&file_path, "omega source\n");
        let modified = super::modified_token(
            &std::fs::metadata(&file_path).expect("read mutated source metadata"),
        )
        .expect("format mutated mtime");
        engine
            .storage()
            .refresh_document_activity(document.id, &modified)
            .expect("force scan mtime fast path");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("embed backlog from canonical text");
        assert_eq!(report.skipped_mtime_docs, 1);
        assert_eq!(report.embedded_chunks, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let embedded_texts = embedder.texts();
        assert!(
            embedded_texts
                .iter()
                .any(|text| text.contains("alpha stored")),
            "expected stored canonical text in embedding input: {embedded_texts:?}"
        );
        assert!(
            embedded_texts
                .iter()
                .all(|text| !text.contains("omega source")),
            "embedding backlog should not read mutated source bytes: {embedded_texts:?}"
        );
    });
}

#[test]
fn update_replays_fts_dirty_from_canonical_text_when_source_bytes_change() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("guide.md");
        write_text_file(&file_path, "alpha replay\n");

        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let document = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query document")
            .expect("document exists");

        engine
            .storage()
            .delete_tantivy_by_doc("work", document.id)
            .expect("delete existing projection");
        engine
            .storage()
            .commit_tantivy("work")
            .expect("commit projection delete");
        assert!(engine
            .storage()
            .query_bm25("work", "alpha", &[("body", 1.0)], 10)
            .expect("query deleted projection")
            .is_empty());

        engine
            .storage()
            .upsert_document(
                collection.id,
                "guide.md",
                &document.title,
                document.title_source,
                &document.hash,
                &document.modified,
            )
            .expect("mark document fts dirty");
        write_text_file(&file_path, "omega source\n");
        let modified = super::modified_token(
            &std::fs::metadata(&file_path).expect("read mutated source metadata"),
        )
        .expect("format mutated mtime");
        engine
            .storage()
            .refresh_document_activity(document.id, &modified)
            .expect("force scan mtime fast path");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("replay fts dirty from canonical text");
        assert_eq!(report.skipped_mtime_docs, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let alpha_hits = engine
            .storage()
            .query_bm25("work", "alpha", &[("body", 1.0)], 10)
            .expect("query replayed canonical projection");
        assert_eq!(alpha_hits.len(), 1);
        assert!(
            engine
                .storage()
                .query_bm25("work", "omega", &[("body", 1.0)], 10)
                .expect("query mutated source text")
                .is_empty(),
            "fts replay should not index mutated source bytes"
        );
    });
}

#[test]
fn update_fts_replay_does_not_buffer_delete_when_canonical_span_is_invalid() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("guide.md"), "alpha retained\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let document = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query document")
            .expect("document exists");

        engine
            .storage()
            .upsert_document(
                collection.id,
                "guide.md",
                &document.title,
                document.title_source,
                &document.hash,
                &document.modified,
            )
            .expect("mark document fts dirty");
        {
            let conn = rusqlite::Connection::open(engine.config().cache_dir.join("meta.sqlite"))
                .expect("open metadata db");
            conn.execute(
                "UPDATE document_texts SET text = '' WHERE doc_id = ?1",
                [document.id],
            )
            .expect("corrupt canonical text span");
        }

        let err = engine
            .update(update_options(Some("work"), &["api"]))
            .expect_err("invalid canonical span should fail replay");
        assert!(
            err.to_string().contains("text span"),
            "unexpected error: {err}"
        );

        engine
            .storage()
            .commit_tantivy("work")
            .expect("commit any pending writer mutations");
        let hits = engine
            .storage()
            .query_bm25("work", "alpha", &[("body", 1.0)], 10)
            .expect("query retained projection");
        assert_eq!(
            hits.len(),
            1,
            "failed replay must not leave a pending delete for the old projection"
        );
    });
}

#[test]
fn update_embeds_canonical_payload_without_raw_spacing_false_rejection() {
    with_kbolt_space_env(None, || {
        let mut chunking = ChunkingConfig::default();
        chunking.defaults.target_tokens = 12;
        chunking.defaults.soft_max_tokens = 12;
        chunking.defaults.hard_max_tokens = 12;
        chunking.defaults.boundary_overlap_tokens = 0;
        let engine = test_engine_with_embedding_runtime(
            Arc::new(DeterministicEmbedder),
            Arc::new(CharCountDocumentSizer),
            chunking,
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("guide.md"), "alpha\n\n\n\n\nbeta\n");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update with canonical payload");

        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert_eq!(report.failed_docs, 0);
        assert_eq!(report.embedded_chunks, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected error: {:?}",
            report.errors
        );

        let work_space = engine.storage().get_space("work").expect("get work space");
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks"),
            1
        );
        let collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get api collection");
        let document = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query document")
            .expect("document exists");
        assert_eq!(
            engine
                .storage()
                .get_document_text(document.id)
                .expect("load canonical text")
                .text,
            "alpha\n\nbeta"
        );
        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list files");
        assert_eq!(files.len(), 1);
        assert!(files[0].embedded, "file should be embedded");
    });
}

#[test]
fn update_embeds_backlog_from_canonical_payload_without_raw_spacing_false_rejection() {
    with_kbolt_space_env(None, || {
        let mut chunking = ChunkingConfig::default();
        chunking.defaults.target_tokens = 12;
        chunking.defaults.soft_max_tokens = 12;
        chunking.defaults.hard_max_tokens = 12;
        chunking.defaults.boundary_overlap_tokens = 0;
        let engine = test_engine_with_embedding_runtime(
            Arc::new(DeterministicEmbedder),
            Arc::new(CharCountDocumentSizer),
            chunking,
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("guide.md"), "alpha\n\n\n\n\nbeta\n");

        engine
            .update(UpdateOptions {
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                no_embed: true,
                dry_run: false,
                verbose: false,
            })
            .expect("index without embeddings");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update backlog with canonical payload");

        assert_eq!(report.skipped_mtime_docs, 1);
        assert_eq!(report.failed_docs, 0);
        assert_eq!(report.embedded_chunks, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected error: {:?}",
            report.errors
        );

        let work_space = engine.storage().get_space("work").expect("get work space");
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks"),
            1
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors"),
            1
        );
    });
}

#[test]
fn update_preserves_existing_chunks_when_chunk_token_count_fails() {
    with_kbolt_space_env(None, || {
        let mut chunking = ChunkingConfig::default();
        chunking.defaults.target_tokens = 32;
        chunking.defaults.soft_max_tokens = 32;
        chunking.defaults.hard_max_tokens = 32;
        chunking.defaults.boundary_overlap_tokens = 0;
        let engine = test_engine_with_embedding_runtime(
            Arc::new(DeterministicEmbedder),
            Arc::new(SelectiveFailureDocumentSizer),
            chunking,
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("guide.md");
        write_text_file(&file_path, "oldtoken body\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        assert_eq!(first.added_docs, 1);
        assert_eq!(first.failed_docs, 0);
        assert!(
            first.errors.is_empty(),
            "unexpected errors: {:?}",
            first.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let original_doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query original document")
            .expect("document exists");
        let original_chunks = engine
            .storage()
            .get_chunks_for_document(original_doc.id)
            .expect("load original chunks");
        assert_eq!(original_chunks.len(), 1);
        let original_chunk_ids = original_chunks
            .iter()
            .map(|chunk| chunk.id)
            .collect::<Vec<_>>();

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "TOKENIZE_FAIL replacement\n");

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update with token-count failure");
        assert_eq!(second.scanned_docs, 1);
        assert_eq!(second.updated_docs, 0);
        assert_eq!(second.failed_docs, 1);
        assert_eq!(second.errors.len(), 1);
        assert!(
            second.errors[0].error.contains("chunking failed"),
            "unexpected errors: {:?}",
            second.errors
        );

        let preserved_doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query preserved document")
            .expect("document should still exist");
        assert_eq!(preserved_doc.id, original_doc.id);
        assert_eq!(preserved_doc.hash, original_doc.hash);
        assert_eq!(preserved_doc.modified, original_doc.modified);

        let preserved_chunks = engine
            .storage()
            .get_chunks_for_document(preserved_doc.id)
            .expect("load preserved chunks");
        let preserved_chunk_ids = preserved_chunks
            .iter()
            .map(|chunk| chunk.id)
            .collect::<Vec<_>>();
        assert_eq!(preserved_chunk_ids, original_chunk_ids);

        let hits = engine
            .storage()
            .query_bm25("work", "oldtoken", &[("body", 1.0)], 10)
            .expect("query preserved bm25");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, original_chunk_ids[0]);
    });
}

#[test]
fn update_preserves_existing_chunks_when_preinsert_preflight_rejects_replacement() {
    with_kbolt_space_env(None, || {
        let mut chunking = ChunkingConfig::default();
        chunking.defaults.target_tokens = 32;
        chunking.defaults.soft_max_tokens = 32;
        chunking.defaults.hard_max_tokens = 32;
        chunking.defaults.boundary_overlap_tokens = 0;
        let sizer = Arc::new(SecondCallOversizeDocumentSizer::default());
        let engine = test_engine_with_embedding_runtime(
            Arc::new(DeterministicEmbedder),
            sizer.clone(),
            chunking,
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("guide.md");
        write_text_file(&file_path, "oldtoken body\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        assert_eq!(first.added_docs, 1);
        assert_eq!(first.failed_docs, 0);
        assert!(
            first.errors.is_empty(),
            "unexpected errors: {:?}",
            first.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let original_doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query original document")
            .expect("document exists");
        let original_text = engine
            .storage()
            .get_document_text(original_doc.id)
            .expect("load original canonical text");
        let original_chunks = engine
            .storage()
            .get_chunks_for_document(original_doc.id)
            .expect("load original chunks");
        let original_chunk_ids = original_chunks
            .iter()
            .map(|chunk| chunk.id)
            .collect::<Vec<_>>();

        sizer.reject_after_first_call();
        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "newtoken body\n");

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update with preflight rejection");
        assert_eq!(second.scanned_docs, 1);
        assert_eq!(second.updated_docs, 0);
        assert_eq!(second.failed_docs, 1);
        assert_eq!(second.errors.len(), 1);
        assert!(
            second.errors[0].error.contains("embed preflight failed"),
            "unexpected errors: {:?}",
            second.errors
        );

        let preserved_doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query preserved document")
            .expect("document should still exist");
        assert_eq!(preserved_doc.id, original_doc.id);
        assert_eq!(preserved_doc.hash, original_doc.hash);
        assert_eq!(preserved_doc.modified, original_doc.modified);
        assert_eq!(
            engine
                .storage()
                .get_document_text(preserved_doc.id)
                .expect("load preserved canonical text")
                .text,
            original_text.text
        );

        let preserved_chunk_ids = engine
            .storage()
            .get_chunks_for_document(preserved_doc.id)
            .expect("load preserved chunks")
            .iter()
            .map(|chunk| chunk.id)
            .collect::<Vec<_>>();
        assert_eq!(preserved_chunk_ids, original_chunk_ids);

        let hits = engine
            .storage()
            .query_bm25("work", "oldtoken", &[("body", 1.0)], 10)
            .expect("query preserved bm25");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, original_chunk_ids[0]);
    });
}

#[test]
fn update_replaces_existing_chunks_from_canonical_text_when_raw_spacing_changes() {
    with_kbolt_space_env(None, || {
        let mut chunking = ChunkingConfig::default();
        chunking.defaults.target_tokens = 12;
        chunking.defaults.soft_max_tokens = 12;
        chunking.defaults.hard_max_tokens = 12;
        chunking.defaults.boundary_overlap_tokens = 0;
        let engine = test_engine_with_embedding_runtime(
            Arc::new(DeterministicEmbedder),
            Arc::new(CharCountDocumentSizer),
            chunking,
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("guide.md");
        write_text_file(&file_path, "alpha beta\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        assert_eq!(first.added_docs, 1);
        assert_eq!(first.failed_docs, 0);
        assert!(
            first.errors.is_empty(),
            "unexpected errors: {:?}",
            first.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let original_doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query original document")
            .expect("document exists");
        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "alpha\n\n\n\n\nbeta\n");

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update with canonical replacement");
        assert_eq!(second.scanned_docs, 1);
        assert_eq!(second.updated_docs, 1);
        assert_eq!(second.failed_docs, 0);
        assert!(
            second.errors.is_empty(),
            "unexpected errors: {:?}",
            second.errors
        );

        let updated_doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query updated document")
            .expect("document should still exist");
        assert_eq!(updated_doc.id, original_doc.id);
        assert_ne!(updated_doc.hash, original_doc.hash);
        assert_ne!(updated_doc.modified, original_doc.modified);

        let document_text = engine
            .storage()
            .get_document_text(updated_doc.id)
            .expect("load canonical text");
        assert_eq!(document_text.text, "alpha\n\nbeta");

        let updated_chunks = engine
            .storage()
            .get_chunks_for_document(updated_doc.id)
            .expect("load updated chunks");
        let updated_chunk_ids = updated_chunks
            .iter()
            .map(|chunk| chunk.id)
            .collect::<Vec<_>>();
        assert!(!updated_chunk_ids.is_empty());

        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(space.id))
                .expect("count embedded chunks after canonical replacement"),
            updated_chunks.len()
        );
    });
}

#[test]
fn update_rebuilds_unchanged_file_when_chunking_generation_changes() {
    with_kbolt_space_env(None, || {
        let engine = test_engine();
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("guide.txt");
        write_text_file(
            &file_path,
            "one two three four five six seven eight nine ten\n",
        );

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        assert_eq!(first.added_docs, 1);
        assert_eq!(first.failed_docs, 0);
        assert!(
            first.errors.is_empty(),
            "unexpected errors: {:?}",
            first.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.txt")
            .expect("query document")
            .expect("document exists");
        let document_text = engine
            .storage()
            .get_document_text(doc.id)
            .expect("load document text");
        assert!(
            document_text
                .generation_key
                .starts_with("canonical=v1;chunker=v2;extractor=txt:v1;"),
            "generation key should include canonical and chunker versions: {}",
            document_text.generation_key
        );

        let config_dir = engine.config().config_dir.clone();
        let cache_dir = engine.config().cache_dir.clone();
        drop(engine);

        let storage = Storage::new(&cache_dir).expect("reopen storage");
        let mut config = base_test_config(config_dir, cache_dir);
        config.chunking.defaults.target_tokens = 2;
        config.chunking.defaults.soft_max_tokens = 2;
        config.chunking.defaults.hard_max_tokens = 2;
        config.chunking.defaults.boundary_overlap_tokens = 0;
        let engine = Engine::from_parts(storage, config);

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update after chunking generation change");
        assert_eq!(second.scanned_docs, 1);
        assert_eq!(second.skipped_mtime_docs, 0);
        assert_eq!(second.skipped_hash_docs, 0);
        assert_eq!(second.updated_docs, 1);
        assert_eq!(second.failed_docs, 0);
        assert!(
            second.errors.is_empty(),
            "unexpected errors: {:?}",
            second.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.txt")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load rebuilt chunks");
        assert!(
            chunks.len() > 1,
            "expected tighter chunking policy to rebuild into multiple chunks, got {chunks:?}"
        );
    });
}

#[test]
fn update_skips_unchanged_file_when_only_neighbor_window_changes() {
    with_kbolt_space_env(None, || {
        let engine = test_engine();
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("guide.txt");
        write_text_file(&file_path, "one two three\n\nfour five six\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        assert_eq!(first.added_docs, 1);
        assert_eq!(first.failed_docs, 0);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.txt")
            .expect("query document")
            .expect("document exists");
        let document_text = engine
            .storage()
            .get_document_text(doc.id)
            .expect("load document text");
        assert!(
            !document_text.generation_key.contains("neighbors:"),
            "neighbor window is search-time context, not canonical/chunk generation: {}",
            document_text.generation_key
        );

        let config_dir = engine.config().config_dir.clone();
        let cache_dir = engine.config().cache_dir.clone();
        drop(engine);

        let storage = Storage::new(&cache_dir).expect("reopen storage");
        let mut config = base_test_config(config_dir, cache_dir);
        config.chunking.defaults.neighbor_window = 0;
        let engine = Engine::from_parts(storage, config);

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update after neighbor window change");
        assert_eq!(second.scanned_docs, 1);
        assert_eq!(second.skipped_mtime_docs, 1);
        assert_eq!(second.skipped_hash_docs, 0);
        assert_eq!(second.updated_docs, 0);
        assert_eq!(second.failed_docs, 0);
        assert!(
            second.errors.is_empty(),
            "unexpected errors: {:?}",
            second.errors
        );
    });
}

#[test]
fn update_replaces_existing_chunks_after_successful_preinsert_preflight() {
    with_kbolt_space_env(None, || {
        let mut chunking = ChunkingConfig::default();
        chunking.defaults.target_tokens = 32;
        chunking.defaults.soft_max_tokens = 32;
        chunking.defaults.hard_max_tokens = 32;
        chunking.defaults.boundary_overlap_tokens = 0;
        let engine = test_engine_with_embedding_runtime(
            Arc::new(DeterministicEmbedder),
            Arc::new(CharCountDocumentSizer),
            chunking,
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("guide.md");
        write_text_file(&file_path, "oldtoken body\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        assert_eq!(first.added_docs, 1);
        assert_eq!(first.failed_docs, 0);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let original_doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query original document")
            .expect("document exists");
        let original_chunks = engine
            .storage()
            .get_chunks_for_document(original_doc.id)
            .expect("load original chunks");

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "newtoken body\n");

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("replacement update");
        assert_eq!(second.updated_docs, 1);
        assert_eq!(second.failed_docs, 0);
        assert!(
            second.errors.is_empty(),
            "unexpected errors: {:?}",
            second.errors
        );

        let updated_doc = engine
            .storage()
            .get_document_by_path(collection.id, "guide.md")
            .expect("query updated document")
            .expect("document should still exist");
        assert_eq!(updated_doc.id, original_doc.id);
        assert_ne!(updated_doc.hash, original_doc.hash);

        let updated_chunks = engine
            .storage()
            .get_chunks_for_document(updated_doc.id)
            .expect("load updated chunks");
        let updated_chunk_ids = updated_chunks
            .iter()
            .map(|chunk| chunk.id)
            .collect::<Vec<_>>();
        assert_eq!(updated_chunks.len(), original_chunks.len());

        let old_hits = engine
            .storage()
            .query_bm25("work", "oldtoken", &[("body", 1.0)], 10)
            .expect("query old bm25 token");
        assert!(old_hits.is_empty(), "stale oldtoken hit should be replaced");

        let new_hits = engine
            .storage()
            .query_bm25("work", "newtoken", &[("body", 1.0)], 10)
            .expect("query new bm25 token");
        assert_eq!(new_hits.len(), 1);
        assert_eq!(new_hits[0].chunk_id, updated_chunk_ids[0]);
    });
}

#[test]
fn update_isolates_backlog_embedding_failures() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_embedder(Arc::new(SelectiveFailureEmbedder));
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("good.md"), "helpful setup guide\n");
        write_text_file(&collection_path.join("bad.md"), "EMBED_FAIL trigger\n");

        engine
            .update(UpdateOptions {
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                no_embed: true,
                dry_run: false,
                verbose: false,
            })
            .expect("index without embeddings");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("embed backlog with partial failure");

        assert_eq!(report.skipped_mtime_docs, 2);
        assert_eq!(report.failed_docs, 1);
        assert_eq!(report.embedded_chunks, 1);
        assert_eq!(report.errors.len(), 1);
        assert!(
            report.errors[0].path.ends_with("bad.md"),
            "unexpected error path: {:?}",
            report.errors
        );
        assert!(
            report.errors[0].error.contains("simulated embed failure"),
            "unexpected error: {:?}",
            report.errors
        );

        let work_space = engine.storage().get_space("work").expect("get work space");
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks"),
            1
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors"),
            1
        );
    });
}

#[test]
fn update_does_not_embed_backlog_outside_scoped_targets() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_embedder(Arc::new(DeterministicEmbedder));
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "wiki", notes_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "worktoken\n");
        write_text_file(&notes_path.join("docs/guide.md"), "notestoken\n");

        engine
            .update(UpdateOptions {
                space: None,
                collections: Vec::new(),
                no_embed: true,
                dry_run: false,
                verbose: false,
            })
            .expect("index without embeddings");

        let work_space = engine.storage().get_space("work").expect("get work space");
        let notes_space = engine
            .storage()
            .get_space("notes")
            .expect("get notes space");
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count work embedded chunks before scoped update"),
            0
        );
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(notes_space.id))
                .expect("count notes embedded chunks before scoped update"),
            0
        );

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("scoped update");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.skipped_mtime_docs, 1);
        assert!(
            report.embedded_chunks > 0,
            "expected scoped update to repair in-scope backlog"
        );

        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count work embedded chunks after scoped update"),
            report.embedded_chunks
        );
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(notes_space.id))
                .expect("count notes embedded chunks after scoped update"),
            0
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("notes")
                .expect("count notes usearch vectors after scoped update"),
            0
        );
    });
}

#[test]
fn update_backlog_embedding_advances_past_failed_prefix() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_embedder(Arc::new(SelectiveFailureEmbedder));
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        for index in 0..64 {
            write_text_file(
                &collection_path.join(format!("bad-{index:02}.md")),
                "EMBED_FAIL trigger\n",
            );
        }
        write_text_file(&collection_path.join("good.md"), "helpful setup guide\n");

        engine
            .update(UpdateOptions {
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                no_embed: true,
                dry_run: false,
                verbose: false,
            })
            .expect("index without embeddings");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("embed backlog after failed prefix");

        assert_eq!(report.skipped_mtime_docs, 65);
        assert_eq!(report.failed_docs, 64);
        assert_eq!(report.embedded_chunks, 1);
        assert!(
            report.errors.len() >= 64,
            "expected one error per failed prefix chunk, got {:?}",
            report.errors
        );

        let work_space = engine.storage().get_space("work").expect("get work space");
        assert_eq!(
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .expect("count embedded chunks"),
            1
        );
        assert_eq!(
            engine
                .storage()
                .count_usearch("work")
                .expect("count usearch vectors"),
            1
        );

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list files");
        let good = files
            .iter()
            .find(|file| file.path == "good.md")
            .expect("good file entry");
        assert!(good.embedded, "good file should be embedded");
    });
}

#[test]
fn update_records_embeddings_with_configured_embedding_model_key() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_embedder_and_embedding_model(
            Arc::new(DeterministicEmbedder),
            "configured-model",
        );
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update with embedder");
        assert!(
            report.embedded_chunks > 0,
            "expected embedding phase to process chunks"
        );

        let work_space = engine.storage().get_space("work").expect("get work space");
        let models = engine
            .storage()
            .list_embedding_models_in_space(work_space.id)
            .expect("list embedding models in space");
        assert_eq!(models, vec!["configured-model".to_string()]);
    });
}

#[test]
fn update_markdown_uses_structural_chunking_and_heading_metadata() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let repeated_words = std::iter::repeat_n("chunktoken", 1_300)
            .collect::<Vec<_>>()
            .join(" ");
        let markdown = format!("# Title\n\n{repeated_words}\n");
        let file_path = collection_path.join("docs/guide.md");
        write_text_file(&file_path, &markdown);

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update markdown");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "docs/guide.md")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");

        assert!(chunks.len() >= 2, "expected markdown hard-split chunks");
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.kind == FinalChunkKind::Paragraph),
            "expected paragraph chunk kind"
        );
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.heading.as_deref() == Some("Title")),
            "expected heading breadcrumb on narrative chunks"
        );
    });
}

#[test]
fn update_uses_table_retrieval_prefix_for_fresh_index_and_embeddings() {
    with_kbolt_space_env(None, || {
        let embedder = Arc::new(RecordingEmbedder::default());
        let engine =
            test_engine_with_embedder_and_chunking(embedder.clone(), table_split_chunking_config());
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let markdown = r#"| headerneedle | status |
| --- | --- |
| alpha beta gamma delta epsilon zeta eta theta iota tailneedle |
"#;
        write_text_file(&collection_path.join("docs/table.md"), markdown);

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update markdown table");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "docs/table.md")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");
        let tail_chunk = chunks
            .iter()
            .find(|chunk| {
                engine
                    .storage()
                    .get_chunk_text(chunk.id)
                    .expect("hydrate chunk")
                    .text
                    .contains("tailneedle")
            })
            .expect("expected tail chunk");
        assert!(
            tail_chunk
                .retrieval_prefix
                .as_deref()
                .is_some_and(|prefix| prefix.contains("headerneedle")),
            "tail chunk should carry table header retrieval prefix: {tail_chunk:?}"
        );

        let header_hits = engine
            .storage()
            .query_bm25("work", "headerneedle", &[("body", 1.0)], 100)
            .expect("query header term");
        assert!(
            header_hits.iter().any(|hit| hit.chunk_id == tail_chunk.id),
            "expected header term to index tail chunk via retrieval prefix"
        );

        let embedded_texts = embedder.texts();
        assert!(
            embedded_texts
                .iter()
                .any(|text| text.contains("headerneedle") && text.contains("tailneedle")),
            "expected embedding input to include table header and tail value: {embedded_texts:?}"
        );
    });
}

#[test]
fn update_replay_and_backlog_use_table_retrieval_prefix() {
    with_kbolt_space_env(None, || {
        let embedder = Arc::new(RecordingEmbedder::default());
        let engine =
            test_engine_with_embedder_and_chunking(embedder.clone(), table_split_chunking_config());
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let markdown = r#"| headerneedle | status |
| --- | --- |
| alpha beta gamma delta epsilon zeta eta theta iota tailneedle |
"#;
        write_text_file(&collection_path.join("docs/table.md"), markdown);

        let mut no_embed = update_options(Some("work"), &["api"]);
        no_embed.no_embed = true;
        engine.update(no_embed).expect("index without embeddings");

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "docs/table.md")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");
        let tail_chunk = chunks
            .iter()
            .find(|chunk| {
                engine
                    .storage()
                    .get_chunk_text(chunk.id)
                    .expect("hydrate chunk")
                    .text
                    .contains("tailneedle")
            })
            .expect("expected tail chunk")
            .clone();

        engine
            .storage()
            .delete_tantivy_by_doc("work", doc.id)
            .expect("delete existing projection");
        engine
            .storage()
            .commit_tantivy("work")
            .expect("commit projection delete");
        engine
            .storage()
            .upsert_document(
                collection.id,
                "docs/table.md",
                &doc.title,
                doc.title_source,
                &doc.hash,
                &doc.modified,
            )
            .expect("mark document fts dirty");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("replay fts and embed backlog");
        assert_eq!(report.skipped_mtime_docs, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let header_hits = engine
            .storage()
            .query_bm25("work", "headerneedle", &[("body", 1.0)], 100)
            .expect("query replayed header term");
        assert!(
            header_hits.iter().any(|hit| hit.chunk_id == tail_chunk.id),
            "fts replay should preserve table header retrieval prefix"
        );

        let embedded_texts = embedder.texts();
        assert!(
            embedded_texts
                .iter()
                .any(|text| text.contains("headerneedle") && text.contains("tailneedle")),
            "embedding backlog should preserve table header retrieval prefix: {embedded_texts:?}"
        );
    });
}

#[test]
fn update_skipped_hash_preserves_extracted_markdown_title() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("docs/guide.md");
        write_text_file(&file_path, "# Guide\n\nbody text\n");

        let first = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        assert_eq!(first.added_docs, 1);

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "# Guide\n\nbody text\n");

        let second = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("second update");
        assert_eq!(second.skipped_hash_docs, 1);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "docs/guide.md")
            .expect("query document")
            .expect("document exists");
        assert_eq!(doc.title, "Guide");
        assert_eq!(
            doc.title_source,
            crate::storage::DocumentTitleSource::Extracted
        );
    });
}

#[test]
fn update_code_files_use_code_chunking_profile() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let repeated_tokens = std::iter::repeat_n("ident", 700)
            .collect::<Vec<_>>()
            .join(" ");
        let source = format!("fn alpha() {{\n    {repeated_tokens}\n}}\n");
        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, &source);

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update code");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "src/lib.rs")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");

        assert!(
            chunks.len() >= 2,
            "expected hard split from code profile (560 hard max)"
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.kind == FinalChunkKind::Code),
            "expected code chunk kind for code extractor output"
        );
    });
}

#[test]
fn update_code_uses_blank_line_grouping_before_token_fallback() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let g1 = std::iter::repeat_n("g1token", 240)
            .collect::<Vec<_>>()
            .join(" ");
        let g2 = std::iter::repeat_n("g2token", 240)
            .collect::<Vec<_>>()
            .join(" ");
        let g3 = std::iter::repeat_n("g3token", 240)
            .collect::<Vec<_>>()
            .join(" ");
        let source = format!("{g1}\n\n{g2}\n\n{g3}\n");
        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, &source);

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update code");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "src/lib.rs")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");

        assert_eq!(chunks.len(), 2);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.kind == FinalChunkKind::Code));

        let bytes = std::fs::read(&file_path).expect("read source bytes");
        let first = {
            let start = chunks[0].offset.min(bytes.len());
            let end = chunks[0]
                .offset
                .saturating_add(chunks[0].length)
                .min(bytes.len());
            String::from_utf8_lossy(&bytes[start..end]).into_owned()
        };
        let second = {
            let start = chunks[1].offset.min(bytes.len());
            let end = chunks[1]
                .offset
                .saturating_add(chunks[1].length)
                .min(bytes.len());
            String::from_utf8_lossy(&bytes[start..end]).into_owned()
        };
        assert!(first.contains("g1token"));
        assert!(first.contains("g2token"));
        assert!(!first.contains("g3token"));
        assert!(second.contains("g3token"));
    });
}

#[test]
fn update_preserves_structural_boundaries_across_chunk_kinds() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let markdown = r#"# Intro

alpha beta

```rust
fn main() {}
```

gamma delta
"#;
        let file_path = collection_path.join("docs/guide.md");
        write_text_file(&file_path, markdown);

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("update markdown");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let doc = engine
            .storage()
            .get_document_by_path(collection.id, "docs/guide.md")
            .expect("query document")
            .expect("document exists");
        let chunks = engine
            .storage()
            .get_chunks_for_document(doc.id)
            .expect("load chunks");
        let kinds = chunks.iter().map(|chunk| chunk.kind).collect::<Vec<_>>();

        assert_eq!(
            kinds,
            vec![
                FinalChunkKind::Section,
                FinalChunkKind::Code,
                FinalChunkKind::Paragraph,
            ]
        );
        assert!(!kinds.contains(&FinalChunkKind::Mixed));
    });
}

#[test]
fn update_skips_hardcoded_ignored_paths() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&collection_path.join(".git/config"), "[core]\n");
        write_text_file(
            &collection_path.join("node_modules/pkg/index.js"),
            "module.exports = {};\n",
        );
        write_text_file(&collection_path.join(".DS_Store"), "ignored\n");
        write_text_file(&collection_path.join("Cargo.lock"), "ignored\n");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("run update");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list indexed files");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/lib.rs");
    });
}

#[test]
fn update_applies_collection_ignore_patterns() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&collection_path.join("docs/guide.md"), "guide\n");

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        write_text_file(&ignore_path, "docs/**\n");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("run update");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);

        let files = engine
            .list_files(Some("work"), "api", None)
            .expect("list indexed files");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/lib.rs");
    });
}

#[cfg(unix)]
#[test]
fn update_reports_pathful_walk_errors_and_counts_them_as_failures() {
    use std::os::unix::fs::PermissionsExt;

    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        write_text_file(&collection_path.join("src/lib.rs"), "fn alpha() {}\n");
        let blocked_dir = collection_path.join("blocked");
        std::fs::create_dir_all(&blocked_dir).expect("create blocked dir");
        write_text_file(&blocked_dir.join("keep.md"), "keep indexed\n");
        let initial = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");
        assert_eq!(initial.added_docs, 2);

        let mut permissions = std::fs::metadata(&blocked_dir)
            .expect("stat blocked dir")
            .permissions();
        permissions.set_mode(0o000);
        std::fs::set_permissions(&blocked_dir, permissions).expect("chmod blocked dir");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("run update");
        let mut restored_permissions = std::fs::metadata(&blocked_dir)
            .expect("restat blocked dir")
            .permissions();
        restored_permissions.set_mode(0o755);
        std::fs::set_permissions(&blocked_dir, restored_permissions)
            .expect("restore blocked dir permissions");
        assert_eq!(report.failed_docs, 1);
        assert_eq!(report.deactivated_docs, 0);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.path.ends_with("blocked") && error.error.contains("walk error")),
            "expected walk error for blocked directory, got {:?}",
            report.errors
        );
        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let blocked_doc = engine
            .storage()
            .get_document_by_path(collection.id, "blocked/keep.md")
            .expect("query blocked doc")
            .expect("blocked doc should remain indexed");
        assert!(
            blocked_doc.active,
            "walk errors must not deactivate docs under the failed prefix"
        );
    });
}

#[test]
fn update_verbose_records_new_ignored_unsupported_and_extract_failed_decisions() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        engine
            .add_collection(AddCollectionRequest {
                path: collection_path.clone(),
                space: Some("work".to_string()),
                name: Some("api".to_string()),
                description: None,
                extensions: Some(vec!["rs".to_string()]),
                no_index: true,
            })
            .expect("add filtered collection");

        write_text_file(&collection_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(
            &collection_path.join("docs/ignored.rs"),
            "fn ignored() {}\n",
        );
        write_text_file(
            &collection_path.join("notes/guide.md"),
            "# ignored by ext\n",
        );
        if let Some(parent) = collection_path.join("src/bad.rs").parent() {
            std::fs::create_dir_all(parent).expect("create bad.rs parent");
        }
        std::fs::write(collection_path.join("src/bad.rs"), [0xff, 0xfe, 0xfd])
            .expect("write invalid utf8 file");

        let ignore_path = engine
            .config()
            .config_dir
            .join("ignores")
            .join("work")
            .join("api.ignore");
        write_text_file(&ignore_path, "docs/**\n");

        let report = engine
            .update(verbose_update_options(Some("work"), &["api"]))
            .expect("run verbose update");

        assert_eq!(report.added_docs, 1);
        assert_eq!(report.failed_docs, 1);

        let new = report
            .decisions
            .iter()
            .find(|decision| decision.path == "src/lib.rs")
            .expect("expected new decision");
        assert_eq!(new.space, "work");
        assert_eq!(new.collection, "api");
        assert_eq!(new.kind, UpdateDecisionKind::New);
        assert_eq!(new.detail, None);

        let ignored = report
            .decisions
            .iter()
            .find(|decision| decision.path == "docs/ignored.rs")
            .expect("expected ignored decision");
        assert_eq!(ignored.kind, UpdateDecisionKind::Ignored);
        assert_eq!(ignored.detail.as_deref(), Some("matched ignore patterns"));

        let unsupported = report
            .decisions
            .iter()
            .find(|decision| decision.path == "notes/guide.md")
            .expect("expected unsupported decision");
        assert_eq!(unsupported.kind, UpdateDecisionKind::Unsupported);
        assert_eq!(unsupported.detail.as_deref(), Some("extension not allowed"));

        let extract_failed = report
            .decisions
            .iter()
            .find(|decision| decision.path == "src/bad.rs")
            .expect("expected extract failure decision");
        assert_eq!(extract_failed.kind, UpdateDecisionKind::ExtractFailed);
        assert!(
            extract_failed
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("non-utf8 code input")),
            "unexpected extract failure detail: {:?}",
            extract_failed.detail
        );
    });
}

#[test]
fn update_tracks_modified_and_deactivated_documents() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "fn beta() {}\n");
        let changed = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("changed update");
        assert_eq!(changed.updated_docs, 1);
        assert_eq!(changed.added_docs, 0);
        assert_eq!(changed.deactivated_docs, 0);

        std::fs::remove_file(&file_path).expect("remove file");
        let removed = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("deactivate removed file");
        assert_eq!(removed.deactivated_docs, 1);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let docs = engine
            .storage()
            .list_documents(collection.id, false)
            .expect("list all documents");
        assert_eq!(docs.len(), 1);
        assert!(!docs[0].active, "removed document should be inactive");
    });
}

#[test]
fn update_verbose_records_skip_change_deactivate_and_reactivate_decisions() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        let skipped_mtime = engine
            .update(verbose_update_options(Some("work"), &["api"]))
            .expect("second update");
        assert_eq!(skipped_mtime.decisions.len(), 1);
        assert_eq!(skipped_mtime.decisions[0].path, "src/lib.rs");
        assert_eq!(
            skipped_mtime.decisions[0].kind,
            UpdateDecisionKind::SkippedMtime
        );

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "fn alpha() {}\n");
        let skipped_hash = engine
            .update(verbose_update_options(Some("work"), &["api"]))
            .expect("hash-stable update");
        assert_eq!(skipped_hash.decisions.len(), 1);
        assert_eq!(
            skipped_hash.decisions[0].kind,
            UpdateDecisionKind::SkippedHash
        );

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "fn beta() {}\n");
        let changed = engine
            .update(verbose_update_options(Some("work"), &["api"]))
            .expect("changed update");
        assert_eq!(changed.decisions.len(), 1);
        assert_eq!(changed.decisions[0].kind, UpdateDecisionKind::Changed);
        assert_eq!(changed.decisions[0].detail, None);

        std::fs::remove_file(&file_path).expect("remove file");
        let deactivated = engine
            .update(verbose_update_options(Some("work"), &["api"]))
            .expect("deactivate removed file");
        assert_eq!(deactivated.decisions.len(), 1);
        assert_eq!(
            deactivated.decisions[0].kind,
            UpdateDecisionKind::Deactivated
        );

        std::thread::sleep(std::time::Duration::from_millis(2));
        write_text_file(&file_path, "fn beta() {}\n");
        let reactivated = engine
            .update(verbose_update_options(Some("work"), &["api"]))
            .expect("reactivate file");
        assert_eq!(reactivated.decisions.len(), 1);
        assert_eq!(
            reactivated.decisions[0].kind,
            UpdateDecisionKind::Reactivated
        );
        assert_eq!(reactivated.decisions[0].detail, None);
    });
}

#[test]
fn update_reap_purges_search_indexes_for_old_removed_files() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_reaping_days(0);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let strong = collection_path.join("strong.md");
        let weak = collection_path.join("weak.md");
        write_text_file(&strong, "token token token token token\n");
        write_text_file(&weak, "token\n");
        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("initial update");

        std::fs::remove_file(&strong).expect("remove strong file");
        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("reap removed file");
        assert_eq!(report.reaped_docs, 1);

        let response = engine
            .search(SearchRequest {
                query: "token".to_string(),
                mode: SearchMode::Keyword,
                space: Some("work".to_string()),
                collections: vec!["api".to_string()],
                limit: 1,
                min_score: 0.0,
                no_rerank: false,
                debug: false,
            })
            .expect("run keyword search");

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].path, "api/weak.md");

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let docs = engine
            .storage()
            .list_documents(collection.id, false)
            .expect("list documents");
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].path, "weak.md");
    });
}

#[test]
fn update_does_not_reap_documents_outside_scoped_targets() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_reaping_days(0);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work dir");
        std::fs::create_dir_all(&notes_path).expect("create notes dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "wiki", notes_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "worktoken\n");
        write_text_file(&notes_path.join("docs/guide.md"), "notestoken\n");
        engine
            .update(update_options(None, &[]))
            .expect("index initial fixtures");

        let notes_space = engine
            .storage()
            .get_space("notes")
            .expect("get notes space");
        let notes_collection = engine
            .storage()
            .get_collection(notes_space.id, "wiki")
            .expect("get notes collection");
        let notes_doc = engine
            .storage()
            .get_document_by_path(notes_collection.id, "docs/guide.md")
            .expect("query notes document")
            .expect("notes doc should exist");
        engine
            .storage()
            .deactivate_document(notes_doc.id)
            .expect("deactivate notes doc");

        let report = engine
            .update(update_options(Some("work"), &["api"]))
            .expect("run scoped update");
        assert_eq!(report.reaped_docs, 0);

        let notes_doc_after = engine
            .storage()
            .get_document_by_path(notes_collection.id, "docs/guide.md")
            .expect("query notes document after scoped update")
            .expect("notes doc should remain");
        assert!(!notes_doc_after.active);
    });
}

#[test]
fn update_dry_run_reports_changes_without_writing() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let collection_path = root.path().join("work-api");
        std::fs::create_dir_all(&collection_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", collection_path.clone());

        let file_path = collection_path.join("src/lib.rs");
        write_text_file(&file_path, "fn alpha() {}\n");

        let mut options = update_options(Some("work"), &["api"]);
        options.dry_run = true;
        let report = engine.update(options).expect("dry run update");
        assert_eq!(report.scanned_docs, 1);
        assert_eq!(report.added_docs, 1);
        assert_eq!(report.updated_docs, 0);
        assert_eq!(report.deactivated_docs, 0);

        let space = engine.storage().get_space("work").expect("get work space");
        let collection = engine
            .storage()
            .get_collection(space.id, "api")
            .expect("get api collection");
        let docs = engine
            .storage()
            .list_documents(collection.id, false)
            .expect("list all documents");
        assert!(docs.is_empty(), "dry run should not persist documents");
    });
}

#[test]
fn update_creates_global_lock_file() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");

        engine
            .update(update_options(Some("work"), &["api"]))
            .expect("run update");

        let lock_path = engine.config().cache_dir.join("kbolt.lock");
        assert!(
            lock_path.exists(),
            "expected lock file at {}",
            lock_path.display()
        );
    });
}

#[test]
fn update_fails_fast_when_global_lock_is_unavailable() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        std::fs::create_dir_all(&work_path).expect("create collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");

        let lock_path = engine.config().cache_dir.join("kbolt.lock");
        std::fs::create_dir_all(&engine.config().cache_dir).expect("create cache dir");
        let holder = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .expect("open lock file");
        FileExt::try_lock_exclusive(&holder).expect("acquire lock in test");

        let err = engine
            .update(update_options(Some("work"), &["api"]))
            .expect_err("update should fail while lock is held");
        match KboltError::from(err) {
            KboltError::Internal(message) => {
                assert!(
                    message.contains("Another kbolt process is active. Try again shortly."),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn add_space_fails_fast_when_global_lock_is_unavailable() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);

        let lock_path = engine.config().cache_dir.join("kbolt.lock");
        std::fs::create_dir_all(&engine.config().cache_dir).expect("create cache dir");
        let holder = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .expect("open lock file");
        FileExt::try_lock_exclusive(&holder).expect("acquire lock in test");

        let err = engine
            .add_space("work", None)
            .expect_err("add_space should fail while lock is held");
        match KboltError::from(err) {
            KboltError::Internal(message) => {
                assert!(
                    message.contains("Another kbolt process is active. Try again shortly."),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    });
}

#[test]
fn status_reports_space_collection_and_model_counts() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine
            .add_space("work", Some("work docs"))
            .expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work collection dir");
        std::fs::create_dir_all(&notes_path).expect("create notes collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "wiki", notes_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&work_path.join("README.md"), "# docs\n");
        write_text_file(&notes_path.join("notes.md"), "meeting notes\n");

        engine
            .update(update_options(None, &[]))
            .expect("initial update");

        let work_space = engine.storage().get_space("work").expect("get work space");
        let work_collection = engine
            .storage()
            .get_collection(work_space.id, "api")
            .expect("get work collection");
        let work_active_docs = engine
            .storage()
            .list_documents(work_collection.id, true)
            .expect("list active work docs");
        engine
            .storage()
            .deactivate_document(work_active_docs[0].id)
            .expect("deactivate one work doc");

        let status = engine.status(None).expect("get global status");

        assert_eq!(status.cache_dir, engine.config().cache_dir);
        assert_eq!(status.config_dir, engine.config().config_dir);
        assert_eq!(status.total_documents, 3);
        assert_eq!(
            status.total_documents,
            engine.storage().count_documents(None).unwrap()
        );
        assert_eq!(
            status.total_chunks,
            engine.storage().count_chunks(None).unwrap()
        );
        assert_eq!(
            status.total_embedded,
            engine.storage().count_embedded_chunks(None).unwrap()
        );

        assert!(!status.models.embedder.configured);
        assert!(!status.models.reranker.configured);
        assert!(!status.models.expander.configured);
        assert!(!status.models.embedder.ready);
        assert!(!status.models.reranker.ready);
        assert!(!status.models.expander.ready);

        let default_status = status
            .spaces
            .iter()
            .find(|space| space.name == "default")
            .expect("default space status should exist");
        assert!(default_status.collections.is_empty());
        assert_eq!(default_status.last_updated, None);

        let work_status = status
            .spaces
            .iter()
            .find(|space| space.name == "work")
            .expect("work status should exist");
        assert_eq!(work_status.description.as_deref(), Some("work docs"));
        assert_eq!(work_status.collections.len(), 1);
        assert!(work_status.last_updated.is_some());

        let work_collection_status = &work_status.collections[0];
        assert_eq!(work_collection_status.name, "api");
        assert_eq!(work_collection_status.path, work_path);
        assert_eq!(
            work_collection_status.documents,
            engine
                .storage()
                .count_documents_in_collection(work_collection.id, false)
                .unwrap()
        );
        assert_eq!(
            work_collection_status.active_documents,
            engine
                .storage()
                .count_documents_in_collection(work_collection.id, true)
                .unwrap()
        );
        assert_eq!(
            work_collection_status.chunks,
            engine
                .storage()
                .count_chunks_in_collection(work_collection.id)
                .unwrap()
        );
        assert_eq!(
            work_collection_status.embedded_chunks,
            engine
                .storage()
                .count_embedded_chunks_in_collection(work_collection.id)
                .unwrap()
        );
        assert_eq!(
            work_status.last_updated.as_deref(),
            Some(work_collection_status.last_updated.as_str())
        );

        let notes_space = engine
            .storage()
            .get_space("notes")
            .expect("get notes space");
        let notes_collection = engine
            .storage()
            .get_collection(notes_space.id, "wiki")
            .expect("get notes collection");
        let notes_status = status
            .spaces
            .iter()
            .find(|space| space.name == "notes")
            .expect("notes status should exist");
        assert_eq!(notes_status.collections.len(), 1);
        assert!(notes_status.last_updated.is_some());

        let notes_collection_status = &notes_status.collections[0];
        assert_eq!(notes_collection_status.name, "wiki");
        assert_eq!(notes_collection_status.path, notes_path);
        assert_eq!(
            notes_collection_status.documents,
            engine
                .storage()
                .count_documents_in_collection(notes_collection.id, false)
                .unwrap()
        );
        assert_eq!(
            notes_collection_status.active_documents,
            engine
                .storage()
                .count_documents_in_collection(notes_collection.id, true)
                .unwrap()
        );
        assert_eq!(
            notes_collection_status.chunks,
            engine
                .storage()
                .count_chunks_in_collection(notes_collection.id)
                .unwrap()
        );
        assert_eq!(
            notes_collection_status.embedded_chunks,
            engine
                .storage()
                .count_embedded_chunks_in_collection(notes_collection.id)
                .unwrap()
        );
        assert_eq!(
            notes_status.last_updated.as_deref(),
            Some(notes_collection_status.last_updated.as_str())
        );
    });
}

#[test]
fn status_scopes_to_requested_space() {
    with_kbolt_space_env(None, || {
        let engine = test_engine_with_default_space(None);
        engine.add_space("work", None).expect("add work");
        engine.add_space("notes", None).expect("add notes");

        let root = tempdir().expect("create temp root");
        let work_path = root.path().join("work-api");
        let notes_path = root.path().join("notes-wiki");
        std::fs::create_dir_all(&work_path).expect("create work collection dir");
        std::fs::create_dir_all(&notes_path).expect("create notes collection dir");
        add_collection_fixture(&engine, "work", "api", work_path.clone());
        add_collection_fixture(&engine, "notes", "wiki", notes_path.clone());

        write_text_file(&work_path.join("src/lib.rs"), "fn alpha() {}\n");
        write_text_file(&notes_path.join("notes.md"), "meeting notes\n");
        engine
            .update(update_options(None, &[]))
            .expect("initial update");

        let work_space = engine.storage().get_space("work").expect("get work space");
        let scoped = engine.status(Some("work")).expect("get scoped status");

        assert_eq!(scoped.spaces.len(), 1);
        assert_eq!(scoped.spaces[0].name, "work");
        assert_eq!(
            scoped.total_documents,
            engine
                .storage()
                .count_documents(Some(work_space.id))
                .unwrap()
        );
        assert_eq!(
            scoped.total_chunks,
            engine.storage().count_chunks(Some(work_space.id)).unwrap()
        );
        assert_eq!(
            scoped.total_embedded,
            engine
                .storage()
                .count_embedded_chunks(Some(work_space.id))
                .unwrap()
        );
    });
}

#[test]
fn status_errors_for_missing_space_scope() {
    let engine = test_engine_with_default_space(None);
    let err = engine
        .status(Some("missing"))
        .expect_err("missing status scope should error");
    match KboltError::from(err) {
        KboltError::SpaceNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn model_status_reflects_configured_model_names() {
    let engine = test_engine_with_default_space(None);
    let status = engine.model_status().expect("read model status");

    assert!(!status.embedder.configured);
    assert!(!status.reranker.configured);
    assert!(!status.expander.configured);
    assert_eq!(status.embedder.profile, None);
    assert_eq!(status.reranker.profile, None);
    assert_eq!(status.expander.profile, None);
}

#[test]
fn add_schedule_normalizes_trigger_scope_and_assigns_short_ids() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let root = tempdir().expect("create temp root");
    let api_path = root.path().join("api");
    let docs_path = root.path().join("docs");
    std::fs::create_dir_all(&api_path).expect("create api dir");
    std::fs::create_dir_all(&docs_path).expect("create docs dir");
    add_collection_fixture(&engine, "work", "api", api_path);
    add_collection_fixture(&engine, "work", "docs", docs_path);

    let first = engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Weekly {
                weekdays: vec![
                    ScheduleWeekday::Fri,
                    ScheduleWeekday::Mon,
                    ScheduleWeekday::Mon,
                ],
                time: "3:00 pm".to_string(),
            },
            scope: ScheduleScope::Collections {
                space: " work ".to_string(),
                collections: vec!["docs".to_string(), "api".to_string(), "docs".to_string()],
            },
        })
        .expect("add weekly schedule");
    assert_eq!(first.backend, expected_schedule_backend());
    assert_eq!(first.schedule.id, "s1");
    assert_eq!(
        first.schedule.trigger,
        ScheduleTrigger::Weekly {
            weekdays: vec![ScheduleWeekday::Mon, ScheduleWeekday::Fri],
            time: "15:00".to_string(),
        }
    );
    assert_eq!(
        first.schedule.scope,
        ScheduleScope::Collections {
            space: "work".to_string(),
            collections: vec!["api".to_string(), "docs".to_string()],
        }
    );

    let second = engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Every {
                interval: ScheduleInterval {
                    value: 2,
                    unit: ScheduleIntervalUnit::Hours,
                },
            },
            scope: ScheduleScope::All,
        })
        .expect("add interval schedule");
    assert_eq!(second.schedule.id, "s2");

    let schedules = engine.list_schedules().expect("list schedules");
    assert_eq!(schedules.len(), 2);
    assert_eq!(schedules[0], first.schedule);
    assert_eq!(schedules[1], second.schedule);
}

#[test]
fn add_schedule_rejects_duplicate_after_normalization() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let first = engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "3pm".to_string(),
            },
            scope: ScheduleScope::Space {
                space: "work".to_string(),
            },
        })
        .expect("add first schedule");
    assert_eq!(first.schedule.id, "s1");

    let err = engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "15:00".to_string(),
            },
            scope: ScheduleScope::Space {
                space: " work ".to_string(),
            },
        })
        .expect_err("duplicate schedule should fail");
    match KboltError::from(err) {
        KboltError::InvalidInput(message) => {
            assert!(
                message.contains("schedule already exists: s1"),
                "unexpected message: {message}"
            );
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn add_schedule_rejects_short_intervals_and_missing_targets() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let short_interval = engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Every {
                interval: ScheduleInterval {
                    value: 4,
                    unit: ScheduleIntervalUnit::Minutes,
                },
            },
            scope: ScheduleScope::All,
        })
        .expect_err("interval shorter than 5m should fail");
    match KboltError::from(short_interval) {
        KboltError::InvalidInput(message) => {
            assert!(
                message.contains("at least 5 minutes"),
                "unexpected message: {message}"
            );
        }
        other => panic!("unexpected error: {other}"),
    }

    let missing_collection = engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "09:00".to_string(),
            },
            scope: ScheduleScope::Collections {
                space: "work".to_string(),
                collections: vec!["missing".to_string()],
            },
        })
        .expect_err("missing collection should fail");
    match KboltError::from(missing_collection) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "missing"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn remove_schedule_by_id_and_unique_scope_updates_catalog() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let root = tempdir().expect("create temp root");
    let api_path = root.path().join("api");
    std::fs::create_dir_all(&api_path).expect("create api dir");
    add_collection_fixture(&engine, "work", "api", api_path);

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "08:00".to_string(),
            },
            scope: ScheduleScope::Space {
                space: "work".to_string(),
            },
        })
        .expect("add space schedule");
    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Every {
                interval: ScheduleInterval {
                    value: 30,
                    unit: ScheduleIntervalUnit::Minutes,
                },
            },
            scope: ScheduleScope::Collections {
                space: "work".to_string(),
                collections: vec!["api".to_string()],
            },
        })
        .expect("add collection schedule");

    let removed_by_id = engine
        .remove_schedule(RemoveScheduleRequest {
            selector: RemoveScheduleSelector::Id {
                id: "s1".to_string(),
            },
        })
        .expect("remove schedule by id");
    assert_eq!(removed_by_id.removed_ids, vec!["s1".to_string()]);

    let removed_by_scope = engine
        .remove_schedule(RemoveScheduleRequest {
            selector: RemoveScheduleSelector::Scope {
                scope: ScheduleScope::Collections {
                    space: " work ".to_string(),
                    collections: vec!["api".to_string(), "api".to_string()],
                },
            },
        })
        .expect("remove schedule by scope");
    assert_eq!(removed_by_scope.removed_ids, vec!["s2".to_string()]);

    let schedules = engine.list_schedules().expect("list schedules");
    assert!(schedules.is_empty(), "all schedules should be removed");
}

#[test]
fn remove_schedule_by_scope_errors_when_multiple_schedules_match() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "08:00".to_string(),
            },
            scope: ScheduleScope::Space {
                space: "work".to_string(),
            },
        })
        .expect("add first schedule");
    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "17:00".to_string(),
            },
            scope: ScheduleScope::Space {
                space: "work".to_string(),
            },
        })
        .expect("add second schedule");

    let err = engine
        .remove_schedule(RemoveScheduleRequest {
            selector: RemoveScheduleSelector::Scope {
                scope: ScheduleScope::Space {
                    space: "work".to_string(),
                },
            },
        })
        .expect_err("ambiguous scope removal should fail");
    match KboltError::from(err) {
        KboltError::InvalidInput(message) => {
            assert!(
                message.contains("multiple schedules"),
                "unexpected message: {message}"
            );
            assert!(message.contains("s1, s2"), "unexpected message: {message}");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn remove_schedule_by_scope_does_not_require_targets_to_still_exist() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let root = tempdir().expect("create temp root");
    let api_path = root.path().join("api");
    std::fs::create_dir_all(&api_path).expect("create api dir");
    add_collection_fixture(&engine, "work", "api", api_path);

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "09:00".to_string(),
            },
            scope: ScheduleScope::Collections {
                space: "work".to_string(),
                collections: vec!["api".to_string()],
            },
        })
        .expect("add collection schedule");

    engine
        .remove_collection(Some("work"), "api")
        .expect("remove collection target");

    let removed = engine
        .remove_schedule(RemoveScheduleRequest {
            selector: RemoveScheduleSelector::Scope {
                scope: ScheduleScope::Collections {
                    space: "work".to_string(),
                    collections: vec!["api".to_string()],
                },
            },
        })
        .expect("remove schedule after target deletion");
    assert_eq!(removed.removed_ids, vec!["s1".to_string()]);
    assert!(engine.list_schedules().expect("list schedules").is_empty());
}

#[test]
fn run_schedule_indexes_target_scope_and_records_success_state() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let root = tempdir().expect("create temp root");
    let api_path = root.path().join("api");
    let docs_path = root.path().join("docs");
    std::fs::create_dir_all(&api_path).expect("create api dir");
    std::fs::create_dir_all(&docs_path).expect("create docs dir");
    add_collection_fixture(&engine, "work", "api", api_path.clone());
    add_collection_fixture(&engine, "work", "docs", docs_path.clone());

    write_text_file(&api_path.join("src/lib.rs"), "fn alpha() {}\n");
    write_text_file(&docs_path.join("guide.md"), "guide text\n");

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "09:00".to_string(),
            },
            scope: ScheduleScope::Collections {
                space: "work".to_string(),
                collections: vec!["api".to_string()],
            },
        })
        .expect("add api schedule");

    let state = engine.run_schedule("s1").expect("run schedule");
    assert_eq!(state.last_result, Some(ScheduleRunResult::Success));
    assert!(state.last_started.is_some());
    assert!(state.last_finished.is_some());
    assert_eq!(state.last_error, None);

    let api_files = engine
        .list_files(Some("work"), "api", None)
        .expect("list api files");
    assert_eq!(api_files.len(), 1);
    assert_eq!(api_files[0].path, "src/lib.rs");

    let docs_files = engine
        .list_files(Some("work"), "docs", None)
        .expect("list docs files");
    assert!(
        docs_files.is_empty(),
        "docs should not be indexed by api schedule"
    );

    let loaded = engine
        .schedule_run_state("s1")
        .expect("load schedule state");
    assert_eq!(loaded, state);
}

#[test]
fn run_schedule_records_skipped_lock_when_global_lock_is_held() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Every {
                interval: ScheduleInterval {
                    value: 30,
                    unit: ScheduleIntervalUnit::Minutes,
                },
            },
            scope: ScheduleScope::Space {
                space: "work".to_string(),
            },
        })
        .expect("add schedule");

    let lock_path = engine.config().cache_dir.join("kbolt.lock");
    std::fs::create_dir_all(&engine.config().cache_dir).expect("create cache dir");
    let holder = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .expect("open lock file");
    FileExt::try_lock_exclusive(&holder).expect("acquire lock in test");

    let state = engine
        .run_schedule("s1")
        .expect("run schedule with held lock");
    assert_eq!(state.last_result, Some(ScheduleRunResult::SkippedLock));
    assert!(state.last_started.is_some());
    assert!(state.last_finished.is_some());
    assert_eq!(state.last_error, None);
}

#[test]
fn run_schedule_records_failed_state_when_target_is_missing() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let root = tempdir().expect("create temp root");
    let api_path = root.path().join("api");
    std::fs::create_dir_all(&api_path).expect("create api dir");
    add_collection_fixture(&engine, "work", "api", api_path);

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "09:00".to_string(),
            },
            scope: ScheduleScope::Collections {
                space: "work".to_string(),
                collections: vec!["api".to_string()],
            },
        })
        .expect("add schedule");

    engine
        .remove_collection(Some("work"), "api")
        .expect("remove collection target");

    let err = engine
        .run_schedule("s1")
        .expect_err("missing target should fail");
    match KboltError::from(err) {
        KboltError::CollectionNotFound { name } => assert_eq!(name, "api"),
        other => panic!("unexpected error: {other}"),
    }

    let state = engine
        .schedule_run_state("s1")
        .expect("load failed run state");
    assert_eq!(state.last_result, Some(ScheduleRunResult::Failed));
    assert!(state.last_started.is_some());
    assert!(state.last_finished.is_some());
    assert!(
        state
            .last_error
            .as_deref()
            .is_some_and(|message| message.contains("collection not found")),
        "unexpected error detail: {:?}",
        state.last_error
    );
}

#[test]
fn remove_schedule_deletes_saved_run_state() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let root = tempdir().expect("create temp root");
    let api_path = root.path().join("api");
    std::fs::create_dir_all(&api_path).expect("create api dir");
    add_collection_fixture(&engine, "work", "api", api_path.clone());
    write_text_file(&api_path.join("src/lib.rs"), "fn alpha() {}\n");

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "09:00".to_string(),
            },
            scope: ScheduleScope::Collections {
                space: "work".to_string(),
                collections: vec!["api".to_string()],
            },
        })
        .expect("add schedule");
    engine.run_schedule("s1").expect("run schedule");
    assert!(
        engine
            .schedule_run_state("s1")
            .expect("load saved state")
            .last_result
            .is_some(),
        "state should exist before removal"
    );

    engine
        .remove_schedule(RemoveScheduleRequest {
            selector: RemoveScheduleSelector::Id {
                id: "s1".to_string(),
            },
        })
        .expect("remove schedule");

    let err = engine
        .schedule_run_state("s1")
        .expect_err("removed schedule should not have addressable state");
    match KboltError::from(err) {
        KboltError::InvalidInput(message) => {
            assert!(message.contains("schedule not found: s1"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn add_and_remove_schedule_reconcile_backend_artifacts() {
    let engine = test_engine();

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Every {
                interval: ScheduleInterval {
                    value: 30,
                    unit: ScheduleIntervalUnit::Minutes,
                },
            },
            scope: ScheduleScope::All,
        })
        .expect("add schedule");

    let artifact_paths = schedule_backend_artifact_paths(&engine, "s1");
    assert!(
        artifact_paths.iter().all(|path| path.exists()),
        "expected backend artifacts to exist: {artifact_paths:?}"
    );

    engine
        .remove_schedule(RemoveScheduleRequest {
            selector: RemoveScheduleSelector::Id {
                id: "s1".to_string(),
            },
        })
        .expect("remove schedule");

    assert!(
        artifact_paths.iter().all(|path| !path.exists()),
        "expected backend artifacts to be removed: {artifact_paths:?}"
    );
}

#[test]
fn schedule_status_reports_installed_state_for_reconciled_schedule() {
    let engine = test_engine();

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "3pm".to_string(),
            },
            scope: ScheduleScope::All,
        })
        .expect("add schedule");

    let status = engine.schedule_status().expect("load schedule status");
    assert_eq!(status.orphans, Vec::new());
    assert_eq!(status.schedules.len(), 1);
    assert_eq!(status.schedules[0].schedule.id, "s1");
    assert_eq!(status.schedules[0].backend, expected_schedule_backend());
    assert_eq!(status.schedules[0].state, ScheduleState::Installed);
    assert_eq!(status.schedules[0].run_state, Default::default());
}

#[test]
fn schedule_status_reports_drifted_when_backend_artifact_is_missing() {
    let engine = test_engine();

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Every {
                interval: ScheduleInterval {
                    value: 2,
                    unit: ScheduleIntervalUnit::Hours,
                },
            },
            scope: ScheduleScope::All,
        })
        .expect("add schedule");

    let artifact_path = schedule_backend_artifact_paths(&engine, "s1")
        .into_iter()
        .next()
        .expect("artifact path");
    std::fs::remove_file(&artifact_path).expect("remove backend artifact");

    let status = engine.schedule_status().expect("load schedule status");
    assert_eq!(status.schedules.len(), 1);
    assert_eq!(status.schedules[0].state, ScheduleState::Drifted);
}

#[test]
fn schedule_status_reports_target_missing_when_collection_is_removed() {
    let engine = test_engine();
    engine.add_space("work", None).expect("add work");

    let root = tempdir().expect("create temp root");
    let api_path = root.path().join("api");
    std::fs::create_dir_all(&api_path).expect("create api dir");
    add_collection_fixture(&engine, "work", "api", api_path);

    engine
        .add_schedule(AddScheduleRequest {
            trigger: ScheduleTrigger::Daily {
                time: "09:00".to_string(),
            },
            scope: ScheduleScope::Collections {
                space: "work".to_string(),
                collections: vec!["api".to_string()],
            },
        })
        .expect("add schedule");

    engine
        .remove_collection(Some("work"), "api")
        .expect("remove collection");

    let status = engine.schedule_status().expect("load schedule status");
    assert_eq!(status.schedules.len(), 1);
    assert_eq!(status.schedules[0].state, ScheduleState::TargetMissing);
}

#[test]
fn schedule_status_reports_orphaned_backend_artifacts() {
    let engine = test_engine();
    let orphan_paths = schedule_backend_artifact_paths(&engine, "s9");
    for path in &orphan_paths {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create backend dir");
        }
        std::fs::write(path, "orphan backend artifact").expect("write orphan artifact");
    }

    let status = engine.schedule_status().expect("load schedule status");
    assert!(status.schedules.is_empty());
    assert_eq!(status.orphans.len(), 1);
    assert_eq!(status.orphans[0].id, "s9");
    assert_eq!(status.orphans[0].backend, expected_schedule_backend());
}

#[test]
fn remove_schedule_all_cleans_orphaned_backend_artifacts_when_catalog_is_empty() {
    let engine = test_engine();
    let orphan_paths = schedule_backend_artifact_paths(&engine, "s9");
    for path in &orphan_paths {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create backend dir");
        }
        std::fs::write(path, "orphan backend artifact").expect("write orphan artifact");
    }

    let removed = engine
        .remove_schedule(RemoveScheduleRequest {
            selector: RemoveScheduleSelector::All,
        })
        .expect("remove all schedules");
    assert!(
        removed.removed_ids.is_empty(),
        "catalog should already be empty"
    );
    assert!(
        orphan_paths.iter().all(|path| !path.exists()),
        "expected orphaned backend artifacts to be removed: {orphan_paths:?}"
    );

    let status = engine.schedule_status().expect("load schedule status");
    assert!(status.schedules.is_empty());
    assert!(status.orphans.is_empty(), "orphans should be cleaned up");
}
