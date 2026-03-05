use std::borrow::Cow;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use kbolt_types::KboltError;
use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use ort::session::Session;
use ort::value::Tensor;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};

use crate::config::{
    EmbeddingConfig, ModelConfig, TextInferenceConfig, TextInferenceOutputMode,
    TextInferenceProvider,
};
use crate::models::artifacts::{resolve_file_with_extension, resolve_tokenizer_file};
use crate::models::expander::HeuristicExpander;
use crate::models::reranker::HeuristicReranker;
use crate::models::text::strip_json_fences;
use crate::models::{resolve_model_artifact, Embedder, Expander, ModelRole, Reranker};
use crate::Result;

#[derive(Debug, Clone)]
struct HttpApiEmbedder {
    client: HttpJsonClient,
    model: String,
    batch_size: usize,
}

struct LocalOnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

#[derive(Clone)]
struct ChatBackedReranker {
    chat: Arc<dyn CompletionClient>,
}

#[derive(Clone)]
struct ChatBackedExpander {
    chat: Arc<dyn CompletionClient>,
}

trait CompletionClient: Send + Sync {
    fn complete(&self, system_prompt: &str, user_prompt: &str) -> Result<String>;
}

const MAX_RETRY_AFTER_SECONDS: u64 = 30;

#[derive(Debug, Clone, Copy)]
enum HttpOperation {
    Embedding,
    ChatCompletion,
}

impl HttpOperation {
    fn label(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::ChatCompletion => "chat completion",
        }
    }
}

#[derive(Debug, Clone)]
struct HttpJsonClient {
    agent: ureq::Agent,
    base_url: String,
    api_key_env: Option<String>,
    max_retries: u32,
    api_key_scope: &'static str,
    provider_name: &'static str,
}

impl HttpJsonClient {
    fn new(
        base_url: &str,
        api_key_env: Option<&str>,
        timeout_ms: u64,
        max_retries: u32,
        api_key_scope: &'static str,
        provider_name: &'static str,
    ) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_millis(timeout_ms))
                .build(),
            base_url: base_url.to_string(),
            api_key_env: api_key_env.map(ToString::to_string),
            max_retries,
            api_key_scope,
            provider_name,
        }
    }

    fn post_json<T>(&self, endpoint_suffix: &str, payload: &Value, operation: HttpOperation) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let endpoint = resolve_endpoint(&self.base_url, endpoint_suffix);
        let mut attempt = 0_u32;

        loop {
            let mut request = self
                .agent
                .post(&endpoint)
                .set("content-type", "application/json");

            if let Some(api_key_env) = self.api_key_env.as_deref() {
                let api_key = std::env::var(api_key_env).map_err(|_| {
                    KboltError::Inference(format!(
                        "{} API key env var is not set: {api_key_env}",
                        self.api_key_scope
                    ))
                })?;
                request = request.set("authorization", &format!("Bearer {api_key}"));
            }

            match request.send_json(payload.clone()) {
                Ok(response) => {
                    let decoded = response.into_json().map_err(|err| {
                        KboltError::Inference(format!(
                            "failed to decode {} {} response: {err}",
                            self.provider_name,
                            operation.label()
                        ))
                    })?;
                    return Ok(decoded);
                }
                Err(ureq::Error::Status(status, response)) => {
                    let retry_after_secs =
                        parse_retry_after_seconds(response.header("retry-after"));
                    let body = response
                        .into_string()
                        .unwrap_or_else(|_| "<unreadable body>".to_string());
                    let can_retry = should_retry_status(status) && attempt < self.max_retries;
                    if can_retry {
                        attempt = attempt.saturating_add(1);
                        if let Some(wait_seconds) = retry_after_secs {
                            thread::sleep(Duration::from_secs(
                                wait_seconds.min(MAX_RETRY_AFTER_SECONDS),
                            ));
                        }
                        continue;
                    }

                    return Err(KboltError::Inference(format!(
                        "{} {} request failed ({status}): {body}",
                        self.provider_name,
                        operation.label()
                    ))
                    .into());
                }
                Err(ureq::Error::Transport(err)) => {
                    if attempt < self.max_retries {
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                    return Err(KboltError::Inference(format!(
                        "{} {} transport error: {err}",
                        self.provider_name,
                        operation.label()
                    ))
                    .into());
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct HttpChatClient {
    http: HttpJsonClient,
    model: String,
    output_mode: TextInferenceOutputMode,
}

impl HttpChatClient {
    fn new(
        base_url: &str,
        api_key_env: Option<&str>,
        timeout_ms: u64,
        max_retries: u32,
        model: &str,
        output_mode: TextInferenceOutputMode,
        provider_name: &'static str,
    ) -> Self {
        Self {
            http: HttpJsonClient::new(
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
                "inference",
                provider_name,
            ),
            model: model.to_string(),
            output_mode,
        }
    }
}

impl CompletionClient for HttpChatClient {
    fn complete(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let payload = build_chat_payload(
            &self.model,
            system_prompt,
            user_prompt,
            self.output_mode.clone(),
        );

        let response: ChatCompletionResponse =
            self.http
                .post_json("chat/completions", &payload, HttpOperation::ChatCompletion)?;
        let content = response.into_text()?;
        let normalized = match self.output_mode {
            TextInferenceOutputMode::JsonObject => content.trim(),
            TextInferenceOutputMode::Text => strip_json_fences(&content),
        };
        Ok(normalized.to_string())
    }
}

struct LocalLlamaClient {
    model: Arc<LlamaModel>,
    max_tokens: usize,
    n_ctx: u32,
}

impl LocalLlamaClient {
    fn new(model_path: &Path, max_tokens: usize, n_ctx: u32, n_gpu_layers: u32) -> Result<Self> {
        let mut params = LlamaParams::default();
        params.n_gpu_layers = n_gpu_layers;
        let model = LlamaModel::load_from_file(model_path, params).map_err(|err| {
            KboltError::Inference(format!(
                "failed to load local llama model {}: {err}",
                model_path.display()
            ))
        })?;

        Ok(Self {
            model: Arc::new(model),
            max_tokens,
            n_ctx,
        })
    }
}

impl CompletionClient for LocalLlamaClient {
    fn complete(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let mut session_params = SessionParams::default();
        session_params.n_ctx = self.n_ctx;
        let mut session = self
            .model
            .create_session(session_params)
            .map_err(|err| KboltError::Inference(format!("failed to create llama session: {err}")))?;

        let prompt = format!(
            "System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
        );
        session
            .advance_context(prompt.as_bytes())
            .map_err(|err| KboltError::Inference(format!("llama prompt failed: {err}")))?;
        let completion = session
            .start_completing_with(StandardSampler::new_greedy(), self.max_tokens)
            .map_err(|err| KboltError::Inference(format!("llama completion failed: {err}")))?;
        let text = completion.into_string();
        Ok(strip_json_fences(&text).trim().to_string())
    }
}

fn build_chat_payload(
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
    output_mode: TextInferenceOutputMode,
) -> Value {
    let mut payload = json!({
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
    });
    if output_mode == TextInferenceOutputMode::JsonObject {
        payload["response_format"] = json!({ "type": "json_object" });
    }
    payload
}

#[cfg(test)]
pub(crate) fn build_embedder(
    config: Option<&EmbeddingConfig>,
) -> Result<Option<Arc<dyn Embedder>>> {
    build_embedder_inner(config, None)
}

pub(crate) fn build_embedder_with_local_runtime(
    config: Option<&EmbeddingConfig>,
    model_config: &ModelConfig,
    model_dir: &Path,
) -> Result<Option<Arc<dyn Embedder>>> {
    build_embedder_inner(
        config,
        Some(LocalRuntimeContext {
            model_config,
            model_dir,
        }),
    )
}

#[cfg(test)]
pub(crate) fn build_reranker(config: Option<&TextInferenceConfig>) -> Result<Arc<dyn Reranker>> {
    build_reranker_inner(config, None)
}

pub(crate) fn build_reranker_with_local_runtime(
    config: Option<&TextInferenceConfig>,
    model_config: &ModelConfig,
    model_dir: &Path,
) -> Result<Arc<dyn Reranker>> {
    build_reranker_inner(
        config,
        Some(LocalRuntimeContext {
            model_config,
            model_dir,
        }),
    )
}

#[cfg(test)]
pub(crate) fn build_expander(config: Option<&TextInferenceConfig>) -> Result<Arc<dyn Expander>> {
    build_expander_inner(config, None)
}

pub(crate) fn build_expander_with_local_runtime(
    config: Option<&TextInferenceConfig>,
    model_config: &ModelConfig,
    model_dir: &Path,
) -> Result<Arc<dyn Expander>> {
    build_expander_inner(
        config,
        Some(LocalRuntimeContext {
            model_config,
            model_dir,
        }),
    )
}

#[derive(Clone, Copy)]
struct LocalRuntimeContext<'a> {
    model_config: &'a ModelConfig,
    model_dir: &'a Path,
}

fn build_embedder_inner(
    config: Option<&EmbeddingConfig>,
    local_runtime: Option<LocalRuntimeContext<'_>>,
) -> Result<Option<Arc<dyn Embedder>>> {
    let Some(config) = config else {
        return Ok(None);
    };

    let (provider_name, model, base_url, api_key_env, timeout_ms, max_retries, batch_size) =
        match config {
            EmbeddingConfig::OpenAiCompatible {
                model,
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
                batch_size,
            } => (
                "openai_compatible",
                model.clone(),
                base_url.as_str(),
                api_key_env.as_deref(),
                *timeout_ms,
                *max_retries,
                *batch_size,
            ),
            EmbeddingConfig::Voyage {
                model,
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
                batch_size,
            } => (
                "voyage",
                model.clone(),
                base_url.as_str(),
                api_key_env.as_deref(),
                *timeout_ms,
                *max_retries,
                *batch_size,
            ),
            EmbeddingConfig::LocalOnnx { .. } => {
                let runtime = local_runtime.ok_or_else(|| {
                    KboltError::Inference(
                        "local_onnx embedder requires local runtime context".to_string(),
                    )
                })?;
                let embedder = build_local_onnx_embedder(config, runtime)?;
                return Ok(Some(Arc::new(embedder)));
            }
        };
    let client = HttpJsonClient::new(
        base_url,
        api_key_env,
        timeout_ms,
        max_retries,
        "embedding",
        provider_name,
    );
    let embedder: Arc<dyn Embedder> = Arc::new(HttpApiEmbedder {
        client,
        model,
        batch_size,
    });
    Ok(Some(embedder))
}

fn build_reranker_inner(
    config: Option<&TextInferenceConfig>,
    local_runtime: Option<LocalRuntimeContext<'_>>,
) -> Result<Arc<dyn Reranker>> {
    let reranker: Arc<dyn Reranker> = match config {
        Some(config) => match &config.provider {
            TextInferenceProvider::OpenAiCompatible {
                output_mode,
                model,
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
            } => Arc::new(ChatBackedReranker {
                chat: Arc::new(HttpChatClient::new(
                    base_url,
                    api_key_env.as_deref(),
                    *timeout_ms,
                    *max_retries,
                    model,
                    output_mode.clone(),
                    "openai_compatible",
                )),
            }),
            TextInferenceProvider::LocalLlama {
                model_file,
                max_tokens,
                n_ctx,
                n_gpu_layers,
            } => {
                let runtime = local_runtime.ok_or_else(|| {
                    KboltError::Inference(
                        "local_llama reranker requires local runtime context".to_string(),
                    )
                })?;
                let chat = build_local_llama_client(
                    runtime,
                    ModelRole::Reranker,
                    model_file.as_deref(),
                    *max_tokens,
                    *n_ctx,
                    *n_gpu_layers,
                )?;
                Arc::new(ChatBackedReranker { chat })
            }
        },
        None => Arc::new(HeuristicReranker),
    };
    Ok(reranker)
}

fn build_expander_inner(
    config: Option<&TextInferenceConfig>,
    local_runtime: Option<LocalRuntimeContext<'_>>,
) -> Result<Arc<dyn Expander>> {
    let expander: Arc<dyn Expander> = match config {
        Some(config) => match &config.provider {
            TextInferenceProvider::OpenAiCompatible {
                output_mode,
                model,
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
            } => Arc::new(ChatBackedExpander {
                chat: Arc::new(HttpChatClient::new(
                    base_url,
                    api_key_env.as_deref(),
                    *timeout_ms,
                    *max_retries,
                    model,
                    output_mode.clone(),
                    "openai_compatible",
                )),
            }),
            TextInferenceProvider::LocalLlama {
                model_file,
                max_tokens,
                n_ctx,
                n_gpu_layers,
            } => {
                let runtime = local_runtime.ok_or_else(|| {
                    KboltError::Inference(
                        "local_llama expander requires local runtime context".to_string(),
                    )
                })?;
                let chat = build_local_llama_client(
                    runtime,
                    ModelRole::Expander,
                    model_file.as_deref(),
                    *max_tokens,
                    *n_ctx,
                    *n_gpu_layers,
                )?;
                Arc::new(ChatBackedExpander { chat })
            }
        },
        None => Arc::new(HeuristicExpander),
    };
    Ok(expander)
}

impl Embedder for HttpApiEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        embed_with_http_api(&self.client, &self.model, self.batch_size, texts)
    }
}

impl Embedder for LocalOnnxEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        embed_with_local_onnx(self, texts)
    }
}

impl Reranker for ChatBackedReranker {
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        let system = "You are a retrieval reranker. Return JSON only as {\"scores\":[number,...]} with one score per document, each score between 0 and 1.";
        let mut user = format!("Query:\n{query}\n\nDocuments:\n");
        for (index, doc) in docs.iter().enumerate() {
            user.push_str(&format!("[{index}] {doc}\n"));
        }
        user.push_str("\nRespond with exactly one score per document, in order.");

        let content = self.chat.complete(system, &user)?;
        let parsed: RerankerResponse = parse_json_payload("reranker response", &content)?;
        let mut scores = parsed.into_scores();
        if scores.len() != docs.len() {
            return Err(KboltError::Inference(format!(
                "reranker response size mismatch: expected {}, got {}",
                docs.len(),
                scores.len()
            ))
            .into());
        }

        for score in &mut scores {
            *score = score.clamp(0.0, 1.0);
        }
        Ok(scores)
    }
}

impl Expander for ChatBackedExpander {
    fn expand(&self, query: &str) -> Result<Vec<String>> {
        let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        let system = "You generate search query rewrites. Return JSON only as {\"variants\":[string,...]} with 2 to 4 concise variants.";
        let user = format!(
            "Original query:\n{normalized}\n\nGenerate variants that improve lexical recall, semantic recall, and one HyDE-style descriptive query."
        );
        let content = self.chat.complete(system, &user)?;
        let parsed: ExpanderResponse = parse_json_payload("expander response", &content)?;

        let mut seen = std::collections::HashSet::new();
        let mut variants = Vec::new();
        let normalized_key = normalized.to_ascii_lowercase();
        seen.insert(normalized_key);
        variants.push(normalized);

        for variant in parsed.into_variants() {
            let trimmed = variant.trim();
            if trimmed.is_empty() {
                continue;
            }

            let key = trimmed.to_ascii_lowercase();
            if seen.insert(key) {
                variants.push(trimmed.to_string());
            }
        }

        Ok(variants)
    }
}

fn embed_with_http_api(
    client: &HttpJsonClient,
    model: &str,
    batch_size: usize,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let mut vectors = Vec::new();
    for batch in texts.chunks(batch_size) {
        let payload = json!({
            "model": model,
            "input": batch,
        });
        let parsed: EmbeddingResponseEnvelope =
            client.post_json("embeddings", &payload, HttpOperation::Embedding)?;
        let response_vectors = parsed.into_vectors(batch.len())?;
        vectors.extend(response_vectors);
    }
    Ok(vectors)
}

fn build_local_onnx_embedder(
    config: &EmbeddingConfig,
    runtime: LocalRuntimeContext<'_>,
) -> Result<LocalOnnxEmbedder> {
    let EmbeddingConfig::LocalOnnx {
        onnx_file,
        tokenizer_file,
        max_length,
    } = config
    else {
        return Err(KboltError::Inference("invalid local onnx config".to_string()).into());
    };

    let artifact = resolve_model_artifact(runtime.model_config, runtime.model_dir, ModelRole::Embedder)?;
    let onnx_path = resolve_file_with_extension(
        &artifact.path,
        onnx_file.as_deref(),
        "onnx",
        "embeddings.onnx_file",
    )?;
    let tokenizer_path =
        resolve_tokenizer_file(&artifact.path, tokenizer_file.as_deref(), "embeddings.tokenizer_file")?;

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
        KboltError::Inference(format!(
            "failed to load tokenizer {}: {err}",
            tokenizer_path.display()
        ))
    })?;
    let pad_id = tokenizer
        .token_to_id("[PAD]")
        .or_else(|| tokenizer.token_to_id("<pad>"))
        .unwrap_or(0);
    let pad_token = tokenizer
        .id_to_token(pad_id)
        .unwrap_or_else(|| "[PAD]".to_string());
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id,
        pad_token,
        ..PaddingParams::default()
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: *max_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: Default::default(),
        }))
        .map_err(|err| KboltError::Inference(format!("failed to configure tokenizer: {err}")))?;

    let session = Session::builder()
        .map_err(|err| KboltError::Inference(format!("failed to create ONNX session builder: {err}")))?
        .commit_from_file(&onnx_path)
        .map_err(|err| {
            KboltError::Inference(format!(
                "failed to load ONNX model {}: {err}",
                onnx_path.display()
            ))
        })?;

    Ok(LocalOnnxEmbedder {
        session: Mutex::new(session),
        tokenizer,
    })
}

fn build_local_llama_client(
    runtime: LocalRuntimeContext<'_>,
    role: ModelRole,
    model_file: Option<&str>,
    max_tokens: usize,
    n_ctx: u32,
    n_gpu_layers: u32,
) -> Result<Arc<dyn CompletionClient>> {
    let artifact = resolve_model_artifact(runtime.model_config, runtime.model_dir, role)?;
    let gguf_path = resolve_file_with_extension(&artifact.path, model_file, "gguf", "model_file")?;
    let client = LocalLlamaClient::new(&gguf_path, max_tokens, n_ctx, n_gpu_layers)?;
    Ok(Arc::new(client))
}

fn embed_with_local_onnx(embedder: &LocalOnnxEmbedder, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let inputs = texts.iter().map(|text| text.as_str()).collect::<Vec<_>>();
    let encodings = embedder
        .tokenizer
        .encode_batch(inputs, true)
        .map_err(|err| KboltError::Inference(format!("tokenization failed: {err}")))?;
    if encodings.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = encodings.len();
    let seq_len = encodings[0].get_ids().len();
    if seq_len == 0 {
        return Err(KboltError::Inference("tokenizer produced empty sequences".to_string()).into());
    }

    let mut input_ids = Vec::with_capacity(batch_size * seq_len);
    let mut attention_mask = Vec::with_capacity(batch_size * seq_len);
    let mut token_type_ids = Vec::with_capacity(batch_size * seq_len);
    for encoding in &encodings {
        if encoding.get_ids().len() != seq_len {
            return Err(
                KboltError::Inference("tokenizer produced uneven sequence lengths".to_string())
                    .into(),
            );
        }
        input_ids.extend(encoding.get_ids().iter().map(|value| *value as i64));
        attention_mask.extend(encoding.get_attention_mask().iter().map(|value| *value as i64));
        if encoding.get_type_ids().is_empty() {
            token_type_ids.extend(std::iter::repeat_n(0_i64, seq_len));
        } else {
            token_type_ids.extend(encoding.get_type_ids().iter().map(|value| *value as i64));
        }
    }

    let ids_tensor = Tensor::<i64>::from_array(([batch_size, seq_len], input_ids.clone())).map_err(
        |err| KboltError::Inference(format!("failed to build input_ids tensor: {err}")),
    )?;
    let mask_tensor =
        Tensor::<i64>::from_array(([batch_size, seq_len], attention_mask.clone())).map_err(
            |err| KboltError::Inference(format!("failed to build attention_mask tensor: {err}")),
        )?;
    let type_tensor =
        Tensor::<i64>::from_array(([batch_size, seq_len], token_type_ids)).map_err(|err| {
            KboltError::Inference(format!("failed to build token_type_ids tensor: {err}"))
        })?;

    let mut session = embedder.session.lock().map_err(|_| {
        KboltError::Inference("local onnx session mutex poisoned".to_string())
    })?;
    let mut session_inputs = Vec::new();
    let single_input = session.inputs().len() == 1;
    let mut mapped_ids = false;
    for input in session.inputs() {
        let name = input.name();
        let lower = name.to_ascii_lowercase();
        if lower.contains("input_ids") || (!mapped_ids && single_input) {
            session_inputs.push((Cow::Owned(name.to_string()), ids_tensor.clone().into_dyn()));
            mapped_ids = true;
            continue;
        }
        if lower.contains("attention_mask") || lower.contains("mask") {
            session_inputs.push((Cow::Owned(name.to_string()), mask_tensor.clone().into_dyn()));
            continue;
        }
        if lower.contains("token_type_ids") || lower.contains("segment_ids") {
            session_inputs.push((Cow::Owned(name.to_string()), type_tensor.clone().into_dyn()));
            continue;
        }

        return Err(KboltError::Inference(format!(
            "unsupported ONNX input '{}' for local embedder",
            name
        ))
        .into());
    }

    if !mapped_ids {
        return Err(KboltError::Inference(
            "ONNX embedder inputs do not include input_ids".to_string(),
        )
        .into());
    }

    let outputs = session
        .run(session_inputs)
        .map_err(|err| KboltError::Inference(format!("onnx inference failed: {err}")))?;
    extract_embedding_vectors(outputs, batch_size, seq_len, &attention_mask)
}

fn extract_embedding_vectors(
    outputs: ort::session::SessionOutputs<'_>,
    batch_size: usize,
    seq_len: usize,
    attention_mask: &[i64],
) -> Result<Vec<Vec<f32>>> {
    for (_, output) in outputs.iter() {
        let Ok((shape, values)) = output.try_extract_tensor::<f32>() else {
            continue;
        };
        if let Some(vectors) =
            parse_embedding_tensor(shape, values, batch_size, seq_len, attention_mask)?
        {
            return Ok(vectors);
        }
    }

    Err(KboltError::Inference(
        "onnx output did not contain a usable embedding tensor".to_string(),
    )
    .into())
}

fn parse_embedding_tensor(
    shape: &ort::value::Shape,
    values: &[f32],
    batch_size: usize,
    seq_len: usize,
    attention_mask: &[i64],
) -> Result<Option<Vec<Vec<f32>>>> {
    let dims = shape.iter().copied().collect::<Vec<_>>();
    if dims.len() == 2 {
        let batch = usize::try_from(dims[0]).ok();
        let hidden = usize::try_from(dims[1]).ok();
        let (Some(batch), Some(hidden)) = (batch, hidden) else {
            return Ok(None);
        };
        if batch != batch_size || hidden == 0 || values.len() != batch.saturating_mul(hidden) {
            return Ok(None);
        }

        let mut vectors = Vec::with_capacity(batch);
        for row in 0..batch {
            let start = row.saturating_mul(hidden);
            let end = start.saturating_add(hidden);
            vectors.push(values[start..end].to_vec());
        }
        return Ok(Some(vectors));
    }

    if dims.len() == 3 {
        let batch = usize::try_from(dims[0]).ok();
        let tokens = usize::try_from(dims[1]).ok();
        let hidden = usize::try_from(dims[2]).ok();
        let (Some(batch), Some(tokens), Some(hidden)) = (batch, tokens, hidden) else {
            return Ok(None);
        };
        if batch != batch_size
            || hidden == 0
            || values.len() != batch.saturating_mul(tokens).saturating_mul(hidden)
        {
            return Ok(None);
        }

        let mask_tokens = if tokens == seq_len {
            attention_mask
        } else {
            return Ok(None);
        };

        let mut vectors = vec![vec![0.0_f32; hidden]; batch];
        for batch_index in 0..batch {
            let mut weight_sum = 0.0_f32;
            for token_index in 0..tokens {
                let mask_index = batch_index
                    .saturating_mul(tokens)
                    .saturating_add(token_index);
                let weight = if mask_tokens.get(mask_index).copied().unwrap_or(1) > 0 {
                    1.0_f32
                } else {
                    0.0_f32
                };
                if weight == 0.0 {
                    continue;
                }
                weight_sum += weight;

                let value_offset = batch_index
                    .saturating_mul(tokens)
                    .saturating_add(token_index)
                    .saturating_mul(hidden);
                for hidden_index in 0..hidden {
                    vectors[batch_index][hidden_index] += values[value_offset + hidden_index];
                }
            }

            if weight_sum == 0.0 {
                return Err(KboltError::Inference(
                    "attention mask produced zero-weight embedding row".to_string(),
                )
                .into());
            }
            for value in &mut vectors[batch_index] {
                *value /= weight_sum;
            }
        }

        return Ok(Some(vectors));
    }

    Ok(None)
}

fn resolve_endpoint(base_url: &str, suffix: &str) -> String {
    let trimmed_base = base_url.trim_end_matches('/');
    let normalized_suffix = suffix.trim_start_matches('/');
    if trimmed_base.ends_with(normalized_suffix) {
        trimmed_base.to_string()
    } else {
        format!("{trimmed_base}/{normalized_suffix}")
    }
}

fn should_retry_status(status: u16) -> bool {
    status == 429 || status >= 500
}

fn parse_retry_after_seconds(header_value: Option<&str>) -> Option<u64> {
    let raw = header_value?.trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(seconds) = raw.parse::<u64>() {
        return Some(seconds);
    }

    let retry_at = httpdate::parse_http_date(raw).ok()?;
    let now = std::time::SystemTime::now();
    let seconds = match retry_at.duration_since(now) {
        Ok(duration) => duration.as_secs(),
        Err(_) => 0,
    };
    Some(seconds)
}

fn parse_json_payload<T>(label: &str, content: &str) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let trimmed = content.trim();
    serde_json::from_str(trimmed).map_err(|err| {
        KboltError::Inference(format!(
            "failed to parse {label} as JSON: {err}; payload={content}"
        ))
        .into()
    })
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingResponseEnvelope {
    OpenAiLike { data: Vec<EmbeddingItem> },
    VoyageLike { embeddings: Vec<Vec<f32>> },
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

impl ChatCompletionResponse {
    fn into_text(self) -> Result<String> {
        let Some(choice) = self.choices.into_iter().next() else {
            return Err(KboltError::Inference(
                "chat completion response is missing choices".to_string(),
            )
            .into());
        };

        extract_text(choice.message.content).ok_or_else(|| {
            KboltError::Inference(
                "chat completion response did not contain text content".to_string(),
            )
            .into()
        })
    }
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Value,
}

fn extract_text(value: Value) -> Option<String> {
    match value {
        Value::String(content) => Some(content),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    Value::String(segment) => text.push_str(&segment),
                    Value::Object(map) => {
                        if let Some(Value::String(segment)) = map.get("text") {
                            text.push_str(segment);
                        }
                    }
                    _ => {}
                }
            }

            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        Value::Object(map) => map
            .get("text")
            .and_then(|item| item.as_str())
            .map(ToString::to_string),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RerankerResponse {
    Scores(Vec<f32>),
    Wrapped { scores: Vec<f32> },
}

impl RerankerResponse {
    fn into_scores(self) -> Vec<f32> {
        match self {
            Self::Scores(scores) => scores,
            Self::Wrapped { scores } => scores,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ExpanderResponse {
    Variants(Vec<String>),
    Wrapped { variants: Vec<String> },
}

impl ExpanderResponse {
    fn into_variants(self) -> Vec<String> {
        match self {
            Self::Variants(variants) => variants,
            Self::Wrapped { variants } => variants,
        }
    }
}

impl EmbeddingResponseEnvelope {
    fn into_vectors(self, expected_len: usize) -> Result<Vec<Vec<f32>>> {
        let mut vectors = match self {
            Self::OpenAiLike { data } => {
                let all_indexed = data.iter().all(|item| item.index.is_some());
                let mut ordered = data
                    .into_iter()
                    .enumerate()
                    .map(|(position, item)| {
                        let order = item.index.unwrap_or(position);
                        (order, item.embedding.into_vec())
                    })
                    .collect::<Vec<_>>();
                if all_indexed {
                    ordered.sort_by_key(|(order, _)| *order);
                }
                ordered
                    .into_iter()
                    .map(|(_, embedding)| embedding)
                    .collect::<Vec<_>>()
            }
            Self::VoyageLike { embeddings } => embeddings,
        };

        if vectors.len() != expected_len {
            return Err(KboltError::Inference(format!(
                "embedding response size mismatch: expected {expected_len}, got {}",
                vectors.len()
            ))
            .into());
        }

        for (index, vector) in vectors.iter().enumerate() {
            if vector.is_empty() {
                return Err(KboltError::Inference(format!(
                    "embedding response contains empty vector at index {index}"
                ))
                .into());
            }
        }

        Ok(std::mem::take(&mut vectors))
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    #[serde(default)]
    index: Option<usize>,
    embedding: EmbeddingVector,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingVector {
    Direct(Vec<f32>),
    Wrapped { values: Vec<f32> },
}

impl EmbeddingVector {
    fn into_vec(self) -> Vec<f32> {
        match self {
            Self::Direct(values) => values,
            Self::Wrapped { values } => values,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;

    use super::*;

    fn base_openai_config(base_url: String) -> EmbeddingConfig {
        EmbeddingConfig::OpenAiCompatible {
            model: "embed-model".to_string(),
            base_url,
            api_key_env: None,
            timeout_ms: 5_000,
            batch_size: 64,
            max_retries: 0,
        }
    }

    fn base_voyage_config(base_url: String) -> EmbeddingConfig {
        EmbeddingConfig::Voyage {
            model: "embed-model".to_string(),
            base_url,
            api_key_env: None,
            timeout_ms: 5_000,
            batch_size: 64,
            max_retries: 0,
        }
    }

    fn base_text_config(
        base_url: String,
        output_mode: TextInferenceOutputMode,
    ) -> TextInferenceConfig {
        TextInferenceConfig {
            provider: TextInferenceProvider::OpenAiCompatible {
                output_mode,
                model: "text-model".to_string(),
                base_url,
                api_key_env: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        }
    }

    fn base_local_llama_text_config() -> TextInferenceConfig {
        TextInferenceConfig {
            provider: TextInferenceProvider::LocalLlama {
                model_file: None,
                max_tokens: 128,
                n_ctx: 2048,
                n_gpu_layers: 0,
            },
        }
    }

    fn serve_once(status_code: u16, body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        let payload = body.to_string();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept client");

            let _ = read_full_request(&mut stream);

            let status_line = match status_code {
                200 => "HTTP/1.1 200 OK",
                429 => "HTTP/1.1 429 Too Many Requests",
                400 => "HTTP/1.1 400 Bad Request",
                500 => "HTTP/1.1 500 Internal Server Error",
                other => panic!("unsupported status code in test server: {other}"),
            };
            let response = format!(
                "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.as_bytes().len(),
                payload
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });

        format!("http://{addr}")
    }

    struct TestResponse {
        status_code: u16,
        body: &'static str,
        retry_after: Option<&'static str>,
    }

    fn serve_sequence(responses: Vec<TestResponse>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        std::thread::spawn(move || {
            for response_spec in responses {
                let (mut stream, _) = listener.accept().expect("accept client");
                let _ = read_full_request(&mut stream);
                let status_line = match response_spec.status_code {
                    200 => "HTTP/1.1 200 OK",
                    429 => "HTTP/1.1 429 Too Many Requests",
                    400 => "HTTP/1.1 400 Bad Request",
                    500 => "HTTP/1.1 500 Internal Server Error",
                    other => panic!("unsupported status code in test server: {other}"),
                };
                let retry_after = response_spec
                    .retry_after
                    .map(|value| format!("Retry-After: {value}\r\n"))
                    .unwrap_or_default();
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\n{retry_after}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_spec.body.as_bytes().len(),
                    response_spec.body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write response");
            }
        });

        format!("http://{addr}")
    }

    fn read_full_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        let mut raw = Vec::new();
        let mut header_end = None;
        while header_end.is_none() {
            let mut chunk = [0_u8; 1024];
            let read = stream.read(&mut chunk).expect("read request bytes");
            if read == 0 {
                break;
            }
            raw.extend_from_slice(&chunk[..read]);
            header_end = raw.windows(4).position(|window| window == b"\r\n\r\n");
        }

        let Some(header_end) = header_end else {
            return Vec::new();
        };
        let header_end = header_end + 4;
        let headers = String::from_utf8_lossy(&raw[..header_end]).to_ascii_lowercase();
        let mut content_length = 0usize;
        for line in headers.lines() {
            if let Some(value) = line.strip_prefix("content-length:") {
                content_length = value.trim().parse::<usize>().unwrap_or(0);
                break;
            }
        }

        let already_read_body = raw.len().saturating_sub(header_end);
        let mut remaining = content_length.saturating_sub(already_read_body);
        while remaining > 0 {
            let mut chunk = [0_u8; 1024];
            let read = stream.read(&mut chunk).expect("read request body");
            if read == 0 {
                break;
            }
            remaining = remaining.saturating_sub(read);
        }

        let body_end = header_end.saturating_add(content_length).min(raw.len());
        raw.get(header_end..body_end).unwrap_or_default().to_vec()
    }

    #[test]
    fn build_embedder_returns_none_when_not_configured() {
        let embedder = build_embedder(None).expect("build embedder");
        assert!(embedder.is_none());
    }

    #[test]
    fn openai_compatible_embedder_parses_openai_style_response() {
        let body = r#"{
  "data": [
    {"index": 1, "embedding": [0.3, 0.4]},
    {"index": 0, "embedding": [0.1, 0.2]}
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_openai_config(base_url);
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder
            .embed_batch(&["a".to_string(), "b".to_string()])
            .expect("embed");
        assert_eq!(vectors, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    }

    #[test]
    fn openai_compatible_embedder_parses_openai_style_response_without_index() {
        let body = r#"{
  "data": [
    {"embedding": [0.51, 0.52]},
    {"embedding": [0.61, 0.62]}
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_openai_config(base_url);
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder
            .embed_batch(&["a".to_string(), "b".to_string()])
            .expect("embed");
        assert_eq!(vectors, vec![vec![0.51, 0.52], vec![0.61, 0.62]]);
    }

    #[test]
    fn openai_compatible_embedder_parses_wrapped_embedding_values() {
        let body = r#"{
  "data": [
    {"index": 0, "embedding": {"values": [0.15, 0.25]}}
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_openai_config(base_url);
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder.embed_batch(&["a".to_string()]).expect("embed");
        assert_eq!(vectors, vec![vec![0.15, 0.25]]);
    }

    #[test]
    fn voyage_embedder_parses_voyage_style_response() {
        let body = r#"{
  "embeddings": [
    [0.11, 0.22],
    [0.33, 0.44]
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_voyage_config(base_url);
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder
            .embed_batch(&["a".to_string(), "b".to_string()])
            .expect("embed");
        assert_eq!(vectors, vec![vec![0.11, 0.22], vec![0.33, 0.44]]);
    }

    #[test]
    fn embedder_errors_when_api_key_env_is_missing() {
        let body = r#"{"data":[{"index":0,"embedding":[0.1,0.2]}]}"#;
        let base_url = serve_once(200, body);
        let config = EmbeddingConfig::OpenAiCompatible {
            model: "embed-model".to_string(),
            base_url,
            api_key_env: Some("KBOLT_TEST_MISSING_API_KEY".to_string()),
            timeout_ms: 5_000,
            batch_size: 64,
            max_retries: 0,
        };
        let embedder: Arc<dyn Embedder> = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        std::env::remove_var("KBOLT_TEST_MISSING_API_KEY");
        let err = embedder
            .embed_batch(&["hello".to_string()])
            .expect_err("missing key should fail");
        assert!(err
            .to_string()
            .contains("embedding API key env var is not set"));
    }

    #[test]
    fn local_onnx_embedder_requires_local_runtime_context() {
        let config = EmbeddingConfig::LocalOnnx {
            onnx_file: None,
            tokenizer_file: None,
            max_length: 256,
        };
        let err = match build_embedder(Some(&config)) {
            Ok(_) => panic!("local runtime context should be required"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("local_onnx embedder requires local runtime context"));
    }

    #[test]
    fn build_reranker_uses_heuristic_when_inference_config_is_missing() {
        let reranker = build_reranker(None).expect("build reranker");
        let scores = reranker
            .rerank(
                "rust traits",
                &[
                    "Rust traits and impl examples".to_string(),
                    "Python decorators".to_string(),
                ],
            )
            .expect("rerank docs");
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn local_llama_reranker_requires_local_runtime_context() {
        let config = base_local_llama_text_config();
        let err = match build_reranker(Some(&config)) {
            Ok(_) => panic!("local runtime context should be required"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("local_llama reranker requires local runtime context"));
    }

    #[test]
    fn openai_compatible_reranker_parses_json_scores() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "{\"scores\":[0.2,0.9]}"
      }
    }
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_text_config(base_url, TextInferenceOutputMode::JsonObject);
        let reranker = build_reranker(Some(&config)).expect("build reranker");

        let scores = reranker
            .rerank("query", &["doc one".to_string(), "doc two".to_string()])
            .expect("rerank docs");
        assert_eq!(scores, vec![0.2, 0.9]);
    }

    #[test]
    fn build_expander_uses_heuristic_when_inference_config_is_missing() {
        let expander = build_expander(None).expect("build expander");
        let variants = expander.expand("rust traits").expect("expand query");
        assert!(!variants.is_empty());
        assert_eq!(variants[0], "rust traits");
    }

    #[test]
    fn openai_compatible_expander_parses_json_variants() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "{\"variants\":[\"trait object rust\",\"explain rust traits\"]}"
      }
    }
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_text_config(base_url, TextInferenceOutputMode::JsonObject);
        let expander = build_expander(Some(&config)).expect("build expander");

        let variants = expander.expand("rust traits").expect("expand query");
        assert_eq!(
            variants,
            vec![
                "rust traits".to_string(),
                "trait object rust".to_string(),
                "explain rust traits".to_string(),
            ]
        );
    }

    #[test]
    fn json_object_mode_sets_response_format_in_chat_payload() {
        let payload = build_chat_payload(
            "model",
            "system",
            "user",
            TextInferenceOutputMode::JsonObject,
        );
        assert_eq!(payload["response_format"]["type"], "json_object");
    }

    #[test]
    fn text_mode_omits_response_format_in_chat_payload() {
        let payload = build_chat_payload("model", "system", "user", TextInferenceOutputMode::Text);
        assert!(payload.get("response_format").is_none());
    }

    #[test]
    fn text_mode_parses_fenced_json() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "```json\n{\"scores\":[0.2,0.9]}\n```"
      }
    }
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_text_config(base_url, TextInferenceOutputMode::Text);
        let reranker = build_reranker(Some(&config)).expect("build reranker");

        let scores = reranker
            .rerank("query", &["doc one".to_string(), "doc two".to_string()])
            .expect("rerank docs");
        assert_eq!(scores, vec![0.2, 0.9]);
    }

    #[test]
    fn text_mode_fails_fast_when_content_is_not_json() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "this is not json"
      }
    }
  ]
}"#;
        let base_url = serve_once(200, body);
        let config = base_text_config(base_url, TextInferenceOutputMode::Text);
        let reranker = build_reranker(Some(&config)).expect("build reranker");

        let err = reranker
            .rerank("query", &["doc one".to_string(), "doc two".to_string()])
            .expect_err("non-json payload should fail");
        assert!(err
            .to_string()
            .contains("failed to parse reranker response as JSON"));
    }

    #[test]
    fn embedder_retries_on_rate_limit_status() {
        let base_url = serve_sequence(vec![
            TestResponse {
                status_code: 429,
                body: r#"{"error":"rate limit"}"#,
                retry_after: Some("0"),
            },
            TestResponse {
                status_code: 200,
                body: r#"{"data":[{"index":0,"embedding":[0.1,0.2]}]}"#,
                retry_after: None,
            },
        ]);
        let config = EmbeddingConfig::OpenAiCompatible {
            model: "embed-model".to_string(),
            base_url,
            api_key_env: None,
            timeout_ms: 5_000,
            batch_size: 64,
            max_retries: 1,
        };
        let embedder = build_embedder(Some(&config))
            .expect("build embedder")
            .expect("embedder should exist");

        let vectors = embedder
            .embed_batch(&["hello".to_string()])
            .expect("embed should retry then succeed");
        assert_eq!(vectors, vec![vec![0.1, 0.2]]);
    }

    #[test]
    fn parse_retry_after_seconds_accepts_delta_seconds() {
        assert_eq!(parse_retry_after_seconds(Some("7")), Some(7));
    }

    #[test]
    fn parse_retry_after_seconds_accepts_http_date() {
        let retry_at = std::time::SystemTime::now() + Duration::from_secs(60);
        let header = httpdate::fmt_http_date(retry_at);
        let parsed = parse_retry_after_seconds(Some(&header));
        assert!(parsed.is_some());
        let seconds = parsed.expect("parsed seconds");
        assert!(seconds <= 60);
    }
}
