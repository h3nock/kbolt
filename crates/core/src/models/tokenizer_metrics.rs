use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use kbolt_types::KboltError;
use serde::Serialize;

use super::tokenizer::TokenizerRuntimeKind;
use crate::Result;

const PROFILE_PATH_ENV: &str = "KBOLT_TOKENIZER_PROFILE_PATH";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum TokenizerCallScope {
    Chunking,
    EmbedPreflightPending,
    EmbedPreflightPrepared,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GgufTokenizerCountMetrics {
    pub(crate) special_scan_positions: u64,
    pub(crate) special_token_probes: u64,
    pub(crate) special_matches: u64,
    pub(crate) raw_segments: u64,
    pub(crate) raw_segment_bytes: u64,
    pub(crate) raw_segment_elapsed_ns: u64,
    pub(crate) prefixed_allocations: u64,
    pub(crate) spm_symbols: u64,
    pub(crate) bigram_probes: u64,
    pub(crate) bigram_hits: u64,
    pub(crate) merges: u64,
    pub(crate) byte_fallback_tokens: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct TokenizerProfileReport {
    schema_version: u32,
    update_elapsed_ms: u64,
    totals: TokenizerCountStats,
    by_scope: BTreeMap<&'static str, TokenizerCountStats>,
    by_runtime: BTreeMap<&'static str, TokenizerCountStats>,
    gguf: GgufTokenizerProfileStats,
}

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub(crate) struct TokenizerCountStats {
    calls: u64,
    errors: u64,
    input_bytes: u64,
    output_tokens: u64,
    elapsed_ns: u64,
    max_input_bytes: u64,
    max_elapsed_ns: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub(crate) struct GgufTokenizerProfileStats {
    runtime_loads: u64,
    token_score_keys: u64,
    special_tokens: u64,
    byte_tokens: u64,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    special_scan_positions: u64,
    special_token_probes: u64,
    special_matches: u64,
    raw_segments: u64,
    raw_segment_bytes: u64,
    raw_segment_elapsed_ns: u64,
    prefixed_allocations: u64,
    spm_symbols: u64,
    bigram_probes: u64,
    bigram_hits: u64,
    merges: u64,
    byte_fallback_tokens: u64,
}

#[derive(Debug, Default)]
struct TokenizerProfileState {
    totals: TokenizerCountStats,
    by_scope: BTreeMap<TokenizerCallScope, TokenizerCountStats>,
    by_runtime: BTreeMap<TokenizerRuntimeKind, TokenizerCountStats>,
    gguf: GgufTokenizerProfileStats,
}

static PROFILE_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
static PROFILE_STATE: OnceLock<Mutex<TokenizerProfileState>> = OnceLock::new();

pub(crate) fn tokenizer_profile_enabled() -> bool {
    profile_path().is_some()
}

pub(crate) fn reset_tokenizer_profile() {
    if !tokenizer_profile_enabled() {
        return;
    }
    let mut state = profile_state()
        .lock()
        .expect("tokenizer profile lock poisoned");
    let gguf_metadata = GgufTokenizerProfileStats {
        runtime_loads: state.gguf.runtime_loads,
        token_score_keys: state.gguf.token_score_keys,
        special_tokens: state.gguf.special_tokens,
        byte_tokens: state.gguf.byte_tokens,
        add_bos: state.gguf.add_bos,
        add_eos: state.gguf.add_eos,
        add_space_prefix: state.gguf.add_space_prefix,
        ..GgufTokenizerProfileStats::default()
    };
    *state = TokenizerProfileState::default();
    state.gguf = gguf_metadata;
}

pub(crate) fn record_tokenizer_count(
    scope: TokenizerCallScope,
    runtime: TokenizerRuntimeKind,
    input_bytes: usize,
    elapsed: Duration,
    result: Result<usize>,
) -> Result<usize> {
    if let Some(count) = result.as_ref().ok().copied() {
        record_count_stats(scope, runtime, input_bytes, elapsed, count, false);
    } else {
        record_count_stats(scope, runtime, input_bytes, elapsed, 0, true);
    }
    result
}

pub(crate) fn record_gguf_runtime_load(
    token_score_keys: usize,
    special_tokens: usize,
    byte_tokens: usize,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
) {
    if !tokenizer_profile_enabled() {
        return;
    }
    let mut state = profile_state()
        .lock()
        .expect("tokenizer profile lock poisoned");
    state.gguf.runtime_loads += 1;
    state.gguf.token_score_keys = token_score_keys as u64;
    state.gguf.special_tokens = special_tokens as u64;
    state.gguf.byte_tokens = byte_tokens as u64;
    state.gguf.add_bos = add_bos;
    state.gguf.add_eos = add_eos;
    state.gguf.add_space_prefix = add_space_prefix;
}

pub(crate) fn record_gguf_count(metrics: GgufTokenizerCountMetrics) {
    if !tokenizer_profile_enabled() {
        return;
    }
    let mut state = profile_state()
        .lock()
        .expect("tokenizer profile lock poisoned");
    state.gguf.special_scan_positions += metrics.special_scan_positions;
    state.gguf.special_token_probes += metrics.special_token_probes;
    state.gguf.special_matches += metrics.special_matches;
    state.gguf.raw_segments += metrics.raw_segments;
    state.gguf.raw_segment_bytes += metrics.raw_segment_bytes;
    state.gguf.raw_segment_elapsed_ns += metrics.raw_segment_elapsed_ns;
    state.gguf.prefixed_allocations += metrics.prefixed_allocations;
    state.gguf.spm_symbols += metrics.spm_symbols;
    state.gguf.bigram_probes += metrics.bigram_probes;
    state.gguf.bigram_hits += metrics.bigram_hits;
    state.gguf.merges += metrics.merges;
    state.gguf.byte_fallback_tokens += metrics.byte_fallback_tokens;
}

pub(crate) fn write_tokenizer_profile_if_requested(update_elapsed_ms: u64) -> Result<()> {
    let Some(path) = profile_path() else {
        return Ok(());
    };
    let report = tokenizer_profile_snapshot(update_elapsed_ms);
    write_profile(path, &report)
}

pub(crate) fn tokenizer_profile_snapshot(update_elapsed_ms: u64) -> TokenizerProfileReport {
    let state = profile_state()
        .lock()
        .expect("tokenizer profile lock poisoned");
    TokenizerProfileReport {
        schema_version: 1,
        update_elapsed_ms,
        totals: state.totals,
        by_scope: state
            .by_scope
            .iter()
            .map(|(scope, stats)| (scope.as_str(), *stats))
            .collect(),
        by_runtime: state
            .by_runtime
            .iter()
            .map(|(runtime, stats)| (runtime.as_str(), *stats))
            .collect(),
        gguf: state.gguf,
    }
}

fn record_count_stats(
    scope: TokenizerCallScope,
    runtime: TokenizerRuntimeKind,
    input_bytes: usize,
    elapsed: Duration,
    output_tokens: usize,
    errored: bool,
) {
    if !tokenizer_profile_enabled() {
        return;
    }
    let mut state = profile_state()
        .lock()
        .expect("tokenizer profile lock poisoned");
    let elapsed_ns = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
    let input_bytes = input_bytes as u64;
    let output_tokens = output_tokens as u64;
    state
        .totals
        .record(input_bytes, output_tokens, elapsed_ns, errored);
    state.by_scope.entry(scope).or_default().record(
        input_bytes,
        output_tokens,
        elapsed_ns,
        errored,
    );
    state.by_runtime.entry(runtime).or_default().record(
        input_bytes,
        output_tokens,
        elapsed_ns,
        errored,
    );
}

impl TokenizerCountStats {
    fn record(&mut self, input_bytes: u64, output_tokens: u64, elapsed_ns: u64, errored: bool) {
        self.calls += 1;
        self.input_bytes += input_bytes;
        self.output_tokens += output_tokens;
        self.elapsed_ns += elapsed_ns;
        self.max_input_bytes = self.max_input_bytes.max(input_bytes);
        self.max_elapsed_ns = self.max_elapsed_ns.max(elapsed_ns);
        if errored {
            self.errors += 1;
        }
    }
}

impl TokenizerCallScope {
    fn as_str(self) -> &'static str {
        match self {
            Self::Chunking => "chunking",
            Self::EmbedPreflightPending => "embed_preflight_pending",
            Self::EmbedPreflightPrepared => "embed_preflight_prepared",
        }
    }
}

impl TokenizerRuntimeKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::LlamaSpmGgufEmbedded => "llama_spm_gguf_embedded",
            Self::LlamaCppHttpTokenize => "llama_cpp_http_tokenize",
            Self::Tiktoken => "tiktoken",
            #[cfg(test)]
            Self::Test => "test",
        }
    }
}

fn profile_state() -> &'static Mutex<TokenizerProfileState> {
    PROFILE_STATE.get_or_init(|| Mutex::new(TokenizerProfileState::default()))
}

fn profile_path() -> Option<&'static Path> {
    PROFILE_PATH
        .get_or_init(|| std::env::var_os(PROFILE_PATH_ENV).map(PathBuf::from))
        .as_deref()
}

fn write_profile(path: &Path, report: &TokenizerProfileReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(report)?;
    fs::write(path, bytes).map_err(|err| {
        KboltError::Internal(format!(
            "failed to write tokenizer profile {}: {err}",
            path.display()
        ))
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_stats_tracks_totals_and_maxima() {
        let mut stats = TokenizerCountStats::default();
        stats.record(12, 3, 100, false);
        stats.record(8, 0, 250, true);

        assert_eq!(stats.calls, 2);
        assert_eq!(stats.errors, 1);
        assert_eq!(stats.input_bytes, 20);
        assert_eq!(stats.output_tokens, 3);
        assert_eq!(stats.elapsed_ns, 350);
        assert_eq!(stats.max_input_bytes, 12);
        assert_eq!(stats.max_elapsed_ns, 250);
    }
}
