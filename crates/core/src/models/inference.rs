use std::sync::Arc;

use kbolt_types::KboltError;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::{Config, ProviderOperation};
use crate::local;
use crate::models::chat::{
    ChatCompletionOutputMode, ChatCompletionRequestOptions, HttpChatClient,
    LlamaCppChatRequestOptions,
};
use crate::models::completion::CompletionClient;
use crate::models::gateway::{
    resolve_inference_gateway_bindings, EmbedderBinding, ExpanderBinding, GatewayProviderKind,
    InferenceGatewayBindings, ProviderDeployment, RerankerBinding,
};
use crate::models::http::{HttpJsonClient, HttpOperation, HttpTransportRecovery};
use crate::models::variants_expander::{ChatVariantsExpander, VARIANTS_GRAMMAR};
use crate::models::{Embedder, EmbeddingDocumentSizer, EmbeddingInputKind, Expander, Reranker};
use crate::{RecoveryNoticeSink, Result};

#[cfg(test)]
use crate::models::chat::build_chat_payload;
#[cfg(test)]
use crate::models::http::parse_retry_after_seconds;

#[derive(Debug, Clone)]
struct HttpApiEmbedder {
    client: HttpJsonClient,
    model: String,
    batch_size: usize,
    endpoint_suffix: &'static str,
}

#[derive(Debug, Clone)]
struct LlamaCppServerDocumentSizer {
    client: HttpJsonClient,
    endpoint_suffix: &'static str,
}

#[derive(Clone)]
struct ChatBackedReranker {
    chat: Arc<dyn CompletionClient>,
}

#[derive(Debug, Clone)]
struct LlamaCppEndpointReranker {
    client: HttpJsonClient,
    model: String,
    endpoint_suffix: &'static str,
    parallel_requests: usize,
}

#[derive(Debug, Clone)]
struct OpenAiCompatibleEndpointReranker {
    client: HttpJsonClient,
    model: String,
    endpoint_suffix: &'static str,
}

pub(crate) struct BuiltInferenceClients {
    pub embedder: Option<Arc<dyn Embedder>>,
    pub embedding_document_sizer: Option<Arc<dyn EmbeddingDocumentSizer>>,
    pub reranker: Option<Arc<dyn Reranker>>,
    pub expander: Option<Arc<dyn Expander>>,
}

#[derive(Clone)]
struct InferenceClientBuildOptions {
    enable_managed_recovery: bool,
    recovery_notice: Option<RecoveryNoticeSink>,
}

impl InferenceClientBuildOptions {
    fn with_managed_recovery(recovery_notice: Option<RecoveryNoticeSink>) -> Self {
        Self {
            enable_managed_recovery: true,
            recovery_notice,
        }
    }

    fn without_managed_recovery() -> Self {
        Self {
            enable_managed_recovery: false,
            recovery_notice: None,
        }
    }
}

impl std::fmt::Debug for InferenceClientBuildOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceClientBuildOptions")
            .field("enable_managed_recovery", &self.enable_managed_recovery)
            .field("recovery_notice", &self.recovery_notice.is_some())
            .finish()
    }
}

impl std::fmt::Debug for BuiltInferenceClients {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuiltInferenceClients")
            .field("embedder", &self.embedder.is_some())
            .field(
                "embedding_document_sizer",
                &self.embedding_document_sizer.is_some(),
            )
            .field("reranker", &self.reranker.is_some())
            .field("expander", &self.expander.is_some())
            .finish()
    }
}

pub(crate) fn build_inference_clients(config: &Config) -> Result<BuiltInferenceClients> {
    build_inference_clients_with_recovery_notice(config, None)
}

pub(crate) fn build_inference_clients_with_recovery_notice(
    config: &Config,
    recovery_notice: Option<RecoveryNoticeSink>,
) -> Result<BuiltInferenceClients> {
    build_inference_clients_with_options(
        config,
        InferenceClientBuildOptions::with_managed_recovery(recovery_notice),
    )
}

pub(crate) fn build_inference_clients_without_managed_recovery(
    config: &Config,
) -> Result<BuiltInferenceClients> {
    build_inference_clients_with_options(
        config,
        InferenceClientBuildOptions::without_managed_recovery(),
    )
}

fn build_inference_clients_with_options(
    config: &Config,
    options: InferenceClientBuildOptions,
) -> Result<BuiltInferenceClients> {
    let bindings = resolve_inference_gateway_bindings(config)?;
    let config = Arc::new(config.clone());
    build_provider_bound_clients(&bindings, &config, options)
}

fn build_provider_bound_clients(
    bindings: &InferenceGatewayBindings,
    config: &Arc<Config>,
    options: InferenceClientBuildOptions,
) -> Result<BuiltInferenceClients> {
    Ok(BuiltInferenceClients {
        embedder: bindings
            .embedder
            .as_ref()
            .map(|binding| build_embedder_from_binding(config, binding, options.clone()))
            .transpose()?,
        embedding_document_sizer: bindings
            .embedder
            .as_ref()
            .map(|binding| {
                build_embedding_document_sizer_from_binding(config, binding, options.clone())
            })
            .transpose()?
            .flatten(),
        reranker: bindings
            .reranker
            .as_ref()
            .map(|binding| build_reranker_from_binding(config, binding, options.clone()))
            .transpose()?,
        expander: bindings
            .expander
            .as_ref()
            .map(|binding| build_expander_from_binding(config, binding, options.clone()))
            .transpose()?,
    })
}

fn build_embedder_from_binding(
    config: &Arc<Config>,
    binding: &EmbedderBinding,
    options: InferenceClientBuildOptions,
) -> Result<Arc<dyn Embedder>> {
    if binding.deployment.operation != ProviderOperation::Embedding {
        return Err(KboltError::Inference(format!(
            "provider profile '{}' uses incompatible operation '{}' for embedder bindings",
            binding.provider_name,
            binding.deployment.operation.as_str()
        ))
        .into());
    }

    Ok(Arc::new(HttpApiEmbedder {
        client: build_http_client(
            config,
            &binding.provider_name,
            &binding.deployment,
            "embedding",
            options,
        ),
        model: binding.deployment.model.clone(),
        batch_size: binding.batch_size,
        endpoint_suffix: embedding_endpoint_suffix(binding.deployment.kind),
    }))
}

fn build_embedding_document_sizer_from_binding(
    config: &Arc<Config>,
    binding: &EmbedderBinding,
    options: InferenceClientBuildOptions,
) -> Result<Option<Arc<dyn EmbeddingDocumentSizer>>> {
    if binding.deployment.operation != ProviderOperation::Embedding {
        return Err(KboltError::Inference(format!(
            "provider profile '{}' uses incompatible operation '{}' for embedder bindings",
            binding.provider_name,
            binding.deployment.operation.as_str()
        ))
        .into());
    }

    match binding.deployment.kind {
        GatewayProviderKind::LlamaCppServer => Ok(Some(Arc::new(LlamaCppServerDocumentSizer {
            client: build_http_client(
                config,
                &binding.provider_name,
                &binding.deployment,
                "embedding",
                options,
            ),
            endpoint_suffix: llama_cpp_tokenize_endpoint_suffix(),
        }))),
        GatewayProviderKind::OpenAiCompatible => Ok(None),
    }
}

fn build_reranker_from_binding(
    config: &Arc<Config>,
    binding: &RerankerBinding,
    options: InferenceClientBuildOptions,
) -> Result<Arc<dyn Reranker>> {
    match binding.deployment.operation {
        ProviderOperation::Reranking => match binding.deployment.kind {
            GatewayProviderKind::LlamaCppServer => Ok(Arc::new(LlamaCppEndpointReranker {
                client: build_http_client(
                    config,
                    &binding.provider_name,
                    &binding.deployment,
                    "reranking",
                    options,
                ),
                model: binding.deployment.model.clone(),
                endpoint_suffix: llama_cpp_reranking_endpoint_suffix(),
                parallel_requests: binding.parallel_requests,
            })),
            GatewayProviderKind::OpenAiCompatible => {
                Ok(Arc::new(OpenAiCompatibleEndpointReranker {
                    client: build_http_client(
                        config,
                        &binding.provider_name,
                        &binding.deployment,
                        "reranking",
                        options,
                    ),
                    model: binding.deployment.model.clone(),
                    endpoint_suffix: openai_compatible_reranking_endpoint_suffix(),
                }))
            }
        },
        ProviderOperation::ChatCompletion => Ok(Arc::new(ChatBackedReranker {
            chat: Arc::new(build_chat_client(
                config,
                &binding.provider_name,
                &binding.deployment,
                match binding.deployment.kind {
                    GatewayProviderKind::LlamaCppServer => {
                        structured_llama_cpp_chat_options(ChatCompletionOutputMode::JsonObject)
                    }
                    GatewayProviderKind::OpenAiCompatible => {
                        ChatCompletionRequestOptions::json_object()
                    }
                },
                options,
            )),
        })),
        ProviderOperation::Embedding => Err(KboltError::Inference(format!(
            "provider profile '{}' uses incompatible operation 'embedding' for reranker bindings",
            binding.provider_name
        ))
        .into()),
    }
}

fn build_expander_from_binding(
    config: &Arc<Config>,
    binding: &ExpanderBinding,
    options: InferenceClientBuildOptions,
) -> Result<Arc<dyn Expander>> {
    if binding.deployment.operation != ProviderOperation::ChatCompletion {
        return Err(KboltError::Inference(format!(
            "provider profile '{}' uses incompatible operation '{}' for expander bindings",
            binding.provider_name,
            binding.deployment.operation.as_str()
        ))
        .into());
    }

    let chat_options = match binding.deployment.kind {
        GatewayProviderKind::OpenAiCompatible => {
            validate_openai_expander_sampling(binding)?;
            openai_expander_options(binding)
        }
        GatewayProviderKind::LlamaCppServer => llama_cpp_expander_options(binding),
    };

    Ok(Arc::new(ChatVariantsExpander::new(Arc::new(
        build_chat_client(
            config,
            &binding.provider_name,
            &binding.deployment,
            chat_options,
            options,
        ),
    ))))
}

fn build_chat_client(
    config: &Arc<Config>,
    provider_name: &str,
    deployment: &ProviderDeployment,
    options: ChatCompletionRequestOptions,
    build_options: InferenceClientBuildOptions,
) -> HttpChatClient {
    HttpChatClient::new(
        &deployment.base_url,
        deployment.api_key_env.as_deref(),
        deployment.timeout_ms,
        deployment.max_retries,
        &deployment.model,
        chat_completion_endpoint_suffix(deployment.kind),
        options,
        deployment.kind.as_str(),
        build_managed_transport_recovery(config, provider_name, deployment, build_options),
    )
}

fn build_http_client(
    config: &Arc<Config>,
    provider_name: &str,
    deployment: &ProviderDeployment,
    api_key_scope: &'static str,
    options: InferenceClientBuildOptions,
) -> HttpJsonClient {
    HttpJsonClient::new(
        &deployment.base_url,
        deployment.api_key_env.as_deref(),
        deployment.timeout_ms,
        deployment.max_retries,
        api_key_scope,
        deployment.kind.as_str(),
        build_managed_transport_recovery(config, provider_name, deployment, options),
    )
}

fn build_managed_transport_recovery(
    config: &Arc<Config>,
    provider_name: &str,
    deployment: &ProviderDeployment,
    options: InferenceClientBuildOptions,
) -> Option<HttpTransportRecovery> {
    if !options.enable_managed_recovery
        || deployment.kind != GatewayProviderKind::LlamaCppServer
        || !local::is_managed_provider_name(provider_name)
    {
        return None;
    }

    let config = Arc::clone(config);
    let provider_name = provider_name.to_string();
    let label = local::managed_provider_label(&provider_name).unwrap_or("managed local provider");
    Some(HttpTransportRecovery::new(
        label,
        Arc::new(move || local::restart_managed_service(config.as_ref(), &provider_name)),
        options.recovery_notice.clone(),
    ))
}

fn embedding_endpoint_suffix(kind: GatewayProviderKind) -> &'static str {
    match kind {
        GatewayProviderKind::LlamaCppServer => "v1/embeddings",
        GatewayProviderKind::OpenAiCompatible => "embeddings",
    }
}

fn chat_completion_endpoint_suffix(kind: GatewayProviderKind) -> &'static str {
    match kind {
        GatewayProviderKind::LlamaCppServer => "v1/chat/completions",
        GatewayProviderKind::OpenAiCompatible => "chat/completions",
    }
}

fn llama_cpp_reranking_endpoint_suffix() -> &'static str {
    "v1/reranking"
}

fn openai_compatible_reranking_endpoint_suffix() -> &'static str {
    "rerank"
}

fn llama_cpp_tokenize_endpoint_suffix() -> &'static str {
    "tokenize"
}

fn openai_expander_options(binding: &ExpanderBinding) -> ChatCompletionRequestOptions {
    ChatCompletionRequestOptions {
        output_mode: ChatCompletionOutputMode::Text,
        max_tokens: Some(binding.max_tokens),
        seed: Some(binding.sampling.seed),
        temperature: Some(binding.sampling.temperature),
        top_k: None,
        top_p: Some(binding.sampling.top_p),
        min_p: None,
        repeat_last_n: None,
        repeat_penalty: None,
        frequency_penalty: Some(binding.sampling.frequency_penalty),
        presence_penalty: Some(binding.sampling.presence_penalty),
        llama_cpp: None,
    }
}

fn llama_cpp_expander_options(binding: &ExpanderBinding) -> ChatCompletionRequestOptions {
    let mut llama_cpp = LlamaCppChatRequestOptions::non_thinking();
    llama_cpp.grammar = Some(VARIANTS_GRAMMAR.to_string());

    ChatCompletionRequestOptions {
        output_mode: ChatCompletionOutputMode::Text,
        max_tokens: Some(binding.max_tokens),
        seed: Some(binding.sampling.seed),
        temperature: Some(binding.sampling.temperature),
        top_k: Some(binding.sampling.top_k),
        top_p: Some(binding.sampling.top_p),
        min_p: Some(binding.sampling.min_p),
        repeat_last_n: Some(binding.sampling.repeat_last_n),
        repeat_penalty: Some(binding.sampling.repeat_penalty),
        frequency_penalty: Some(binding.sampling.frequency_penalty),
        presence_penalty: Some(binding.sampling.presence_penalty),
        llama_cpp: Some(llama_cpp),
    }
}

fn structured_llama_cpp_chat_options(
    output_mode: ChatCompletionOutputMode,
) -> ChatCompletionRequestOptions {
    let mut options = match output_mode {
        ChatCompletionOutputMode::JsonObject => ChatCompletionRequestOptions::json_object(),
        ChatCompletionOutputMode::Text => ChatCompletionRequestOptions {
            output_mode: ChatCompletionOutputMode::Text,
            max_tokens: None,
            seed: None,
            temperature: Some(0.0),
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_last_n: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            llama_cpp: None,
        },
    };
    options.llama_cpp = Some(LlamaCppChatRequestOptions::non_thinking());
    options
}

fn validate_openai_expander_sampling(binding: &ExpanderBinding) -> Result<()> {
    let defaults = crate::config::ExpanderSamplingConfig::default();
    if binding.sampling.top_k != defaults.top_k
        || binding.sampling.min_p != defaults.min_p
        || binding.sampling.repeat_last_n != defaults.repeat_last_n
        || binding.sampling.repeat_penalty != defaults.repeat_penalty
    {
        return Err(KboltError::Config(format!(
            "provider profile '{}' uses openai_compatible chat completion for expander bindings, which does not support top_k, min_p, repeat_last_n, or repeat_penalty overrides",
            binding.provider_name
        ))
        .into());
    }
    Ok(())
}

impl Embedder for HttpApiEmbedder {
    fn embed_batch(&self, _kind: EmbeddingInputKind, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        embed_with_http_api(
            &self.client,
            self.endpoint_suffix,
            &self.model,
            self.batch_size,
            texts,
        )
    }
}

impl EmbeddingDocumentSizer for LlamaCppServerDocumentSizer {
    fn count_document_tokens(&self, text: &str) -> Result<usize> {
        let payload = json!({
            "content": text,
            "add_special": true,
        });
        let response = self.client.post_json::<TokenizeResponseEnvelope>(
            self.endpoint_suffix,
            &payload,
            HttpOperation::Tokenize,
        )?;
        response.token_count()
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

impl Reranker for LlamaCppEndpointReranker {
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        if self.parallel_requests <= 1 || docs.len() <= 1 {
            return self.rerank_single_request(query, docs);
        }

        self.rerank_parallel_requests(query, docs)
    }
}

impl LlamaCppEndpointReranker {
    fn rerank_single_request(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let payload = json!({
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_n": docs.len(),
            "return_text": false,
        });
        let scored = self
            .client
            .post_json::<LlamaCppRerankResponseEnvelope>(
                self.endpoint_suffix,
                &payload,
                HttpOperation::Reranking,
            )?
            .into_items()?;
        finalize_rerank_scores(scored, docs.len())
    }

    fn rerank_parallel_requests(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let shard_count = self.parallel_requests.min(docs.len());
        let ranges = shard_ranges(docs.len(), shard_count);

        std::thread::scope(|scope| {
            let handles = ranges
                .into_iter()
                .map(|(start, end)| {
                    (
                        (start, end),
                        scope.spawn(move || self.rerank_single_request(query, &docs[start..end])),
                    )
                })
                .collect::<Vec<_>>();

            let mut scores = Vec::with_capacity(docs.len());
            let total_shards = handles.len();
            for (shard_index, ((start, end), handle)) in handles.into_iter().enumerate() {
                let shard_label = format!(
                    "llama.cpp rerank shard {}/{} docs {start}..{end}",
                    shard_index + 1,
                    total_shards
                );
                let shard_result = handle.join().map_err(|_| {
                    crate::CoreError::Domain(KboltError::Inference(format!(
                        "{shard_label} worker panicked"
                    )))
                })?;
                let mut shard_scores = shard_result.map_err(|err| {
                    crate::CoreError::Domain(KboltError::Inference(format!(
                        "{shard_label} failed: {err}"
                    )))
                })?;
                scores.append(&mut shard_scores);
            }
            Ok(scores)
        })
    }
}

fn shard_ranges(len: usize, shard_count: usize) -> Vec<(usize, usize)> {
    (0..shard_count)
        .filter_map(|shard| {
            let start = shard * len / shard_count;
            let end = (shard + 1) * len / shard_count;
            (start < end).then_some((start, end))
        })
        .collect()
}

impl Reranker for OpenAiCompatibleEndpointReranker {
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        let payload = json!({
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_n": docs.len(),
            "return_text": false,
        });
        let scored = self
            .client
            .post_json::<OpenAiCompatibleRerankResponseEnvelope>(
                self.endpoint_suffix,
                &payload,
                HttpOperation::Reranking,
            )?
            .into_items()?;
        finalize_rerank_scores(scored, docs.len())
    }
}

fn embed_with_http_api(
    client: &HttpJsonClient,
    endpoint_suffix: &str,
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
            client.post_json(endpoint_suffix, &payload, HttpOperation::Embedding)?;
        let response_vectors = parsed.into_vectors(batch.len())?;
        vectors.extend(response_vectors);
    }
    Ok(vectors)
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
#[serde(untagged)]
enum TokenizeResponseEnvelope {
    Wrapped { tokens: Vec<Value> },
    Direct(Vec<Value>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RerankerResponse {
    Scores(Vec<f32>),
    Wrapped { scores: Vec<f32> },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LlamaCppRerankResponseEnvelope {
    Items(Vec<LlamaCppRerankItem>),
    Wrapped { results: Vec<LlamaCppRerankItem> },
}

impl LlamaCppRerankResponseEnvelope {
    fn into_items(self) -> Result<Vec<RerankScoreItem>> {
        let items = match self {
            Self::Items(items) => items,
            Self::Wrapped { results } => results,
        }
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>();
        validate_rerank_scores(&items)?;
        Ok(items)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiCompatibleRerankResponseEnvelope {
    Items(Vec<OpenAiCompatibleRerankItem>),
    Wrapped {
        results: Vec<OpenAiCompatibleRerankItem>,
    },
}

impl OpenAiCompatibleRerankResponseEnvelope {
    fn into_items(self) -> Result<Vec<RerankScoreItem>> {
        let items = match self {
            Self::Items(items) => items,
            Self::Wrapped { results } => results,
        }
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>();
        validate_rerank_scores(&items)?;
        Ok(items)
    }
}

impl RerankerResponse {
    fn into_scores(self) -> Vec<f32> {
        match self {
            Self::Scores(scores) => scores,
            Self::Wrapped { scores } => scores,
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

impl TokenizeResponseEnvelope {
    fn token_count(self) -> Result<usize> {
        let tokens = match self {
            Self::Wrapped { tokens } => tokens,
            Self::Direct(tokens) => tokens,
        };
        Ok(tokens.len())
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    #[serde(default)]
    index: Option<usize>,
    embedding: EmbeddingVector,
}

#[derive(Debug, Deserialize)]
struct RerankScoreItem {
    index: usize,
    score: f32,
}

#[derive(Debug, Deserialize)]
struct LlamaCppRerankItem {
    index: usize,
    relevance_score: f32,
}

impl From<LlamaCppRerankItem> for RerankScoreItem {
    fn from(value: LlamaCppRerankItem) -> Self {
        Self {
            index: value.index,
            score: value.relevance_score,
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiCompatibleRerankItem {
    index: usize,
    score: f32,
}

impl From<OpenAiCompatibleRerankItem> for RerankScoreItem {
    fn from(value: OpenAiCompatibleRerankItem) -> Self {
        Self {
            index: value.index,
            score: value.score,
        }
    }
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

fn validate_rerank_scores(items: &[RerankScoreItem]) -> Result<()> {
    if items.iter().any(|item| !item.score.is_finite()) {
        return Err(
            KboltError::Inference("rerank response contains non-finite score".to_string()).into(),
        );
    }
    Ok(())
}

fn finalize_rerank_scores(
    mut scored: Vec<RerankScoreItem>,
    expected_len: usize,
) -> Result<Vec<f32>> {
    if scored.len() != expected_len {
        return Err(KboltError::Inference(format!(
            "rerank response size mismatch: expected {expected_len}, got {}",
            scored.len()
        ))
        .into());
    }

    scored.sort_by_key(|item| item.index);
    let mut scores = Vec::with_capacity(scored.len());
    for (expected, item) in scored.into_iter().enumerate() {
        if item.index != expected {
            return Err(KboltError::Inference(format!(
                "rerank response index mismatch: expected {expected}, got {}",
                item.index
            ))
            .into());
        }
        scores.push(item.score);
    }

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc, Condvar, Mutex};
    use std::time::Duration;

    use super::*;
    use crate::config::{
        ChunkingConfig, Config, EmbedderRoleConfig, ExpanderRoleConfig, ExpanderSamplingConfig,
        ProviderProfileConfig, RankingConfig, ReapingConfig, RerankerRoleConfig,
        RoleBindingsConfig,
    };

    fn base_config() -> Config {
        Config {
            config_dir: PathBuf::from("/tmp/kbolt-config"),
            cache_dir: PathBuf::from("/tmp/kbolt-cache"),
            default_space: None,
            providers: HashMap::new(),
            roles: RoleBindingsConfig::default(),
            reaping: ReapingConfig { days: 7 },
            chunking: ChunkingConfig::default(),
            ranking: RankingConfig::default(),
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
                401 => "HTTP/1.1 401 Unauthorized",
                404 => "HTTP/1.1 404 Not Found",
                429 => "HTTP/1.1 429 Too Many Requests",
                500 => "HTTP/1.1 500 Internal Server Error",
                other => panic!("unsupported status code in test server: {other}"),
            };
            let response = format!(
                "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.len(),
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
                    500 => "HTTP/1.1 500 Internal Server Error",
                    other => panic!("unsupported status code in test server: {other}"),
                };
                let retry_after = response_spec
                    .retry_after
                    .map(|value| format!("Retry-After: {value}\r\n"))
                    .unwrap_or_default();
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\n{retry_after}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_spec.body.len(),
                    response_spec.body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write response");
            }
        });

        format!("http://{addr}")
    }

    fn serve_once_capturing_request(
        status_code: u16,
        body: &str,
    ) -> (String, mpsc::Receiver<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        let payload = body.to_string();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept client");
            let request = read_raw_request(&mut stream);
            tx.send(request).expect("send captured request");

            let status_line = match status_code {
                200 => "HTTP/1.1 200 OK",
                401 => "HTTP/1.1 401 Unauthorized",
                404 => "HTTP/1.1 404 Not Found",
                429 => "HTTP/1.1 429 Too Many Requests",
                500 => "HTTP/1.1 500 Internal Server Error",
                other => panic!("unsupported status code in test server: {other}"),
            };
            let response = format!(
                "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.len(),
                payload
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });

        (format!("http://{addr}"), rx)
    }

    fn serve_llama_rerank_shards(
        expected_requests: usize,
    ) -> (String, mpsc::Receiver<String>, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        let (tx, rx) = mpsc::channel();
        let max_inflight = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new((Mutex::new(0usize), Condvar::new()));
        let max_inflight_for_server = Arc::clone(&max_inflight);
        std::thread::spawn(move || {
            for _ in 0..expected_requests {
                let (mut stream, _) = listener.accept().expect("accept client");
                let tx = tx.clone();
                let active = Arc::clone(&active);
                let max_inflight = Arc::clone(&max_inflight_for_server);
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || {
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    max_inflight.fetch_max(current, Ordering::SeqCst);

                    let body = read_full_request(&mut stream);
                    let payload: Value =
                        serde_json::from_slice(&body).expect("parse rerank payload");
                    let docs = payload
                        .get("documents")
                        .and_then(Value::as_array)
                        .expect("documents array")
                        .iter()
                        .map(|doc| doc.as_str().expect("document string").to_string())
                        .collect::<Vec<_>>();
                    for doc in &docs {
                        tx.send(doc.clone()).expect("send captured document");
                    }

                    let (lock, cvar) = &*gate;
                    let mut arrived = lock.lock().expect("lock shard gate");
                    *arrived += 1;
                    cvar.notify_all();
                    let _ = cvar
                        .wait_timeout_while(arrived, Duration::from_secs(1), |arrived| {
                            *arrived < expected_requests
                        })
                        .expect("wait for concurrent shard arrivals");

                    if docs.iter().any(|doc| doc == "fail") {
                        let payload = r#"{"error":"boom"}"#;
                        let response = format!(
                            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            payload.len(),
                            payload
                        );
                        stream
                            .write_all(response.as_bytes())
                            .expect("write failure response");
                        active.fetch_sub(1, Ordering::SeqCst);
                        return;
                    }

                    let mut results = docs
                        .iter()
                        .enumerate()
                        .map(|(index, doc)| {
                            let score = doc
                                .strip_prefix("doc ")
                                .expect("doc prefix")
                                .parse::<f32>()
                                .expect("doc score");
                            json!({"index": index, "relevance_score": score})
                        })
                        .collect::<Vec<_>>();
                    results.reverse();
                    let payload = json!({
                        "model": "qwen3-reranker",
                        "object": "list",
                        "results": results,
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        payload.len(),
                        payload
                    );
                    stream
                        .write_all(response.as_bytes())
                        .expect("write response");
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });

        (format!("http://{addr}"), rx, max_inflight)
    }

    fn read_full_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        let raw = read_raw_request(stream);
        let header_end = raw
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|offset| offset + 4);
        let Some(header_end) = header_end else {
            return Vec::new();
        };

        let headers = String::from_utf8_lossy(&raw[..header_end]).to_ascii_lowercase();
        let mut content_length = 0usize;
        for line in headers.lines() {
            if let Some(value) = line.strip_prefix("content-length:") {
                content_length = value.trim().parse::<usize>().unwrap_or(0);
                break;
            }
        }

        let body_end = header_end.saturating_add(content_length).min(raw.len());
        raw.get(header_end..body_end).unwrap_or_default().to_vec()
    }

    fn read_raw_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
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
            raw.extend_from_slice(&chunk[..read]);
            remaining = remaining.saturating_sub(read);
        }
        raw
    }

    #[test]
    fn provider_profile_embedder_builds_and_embeds() {
        let body = r#"{"data":[{"index":0,"embedding":[0.1,0.2]}]}"#;
        let mut config = base_config();
        config.providers.insert(
            "remote_embed".to_string(),
            ProviderProfileConfig::OpenAiCompatible {
                operation: ProviderOperation::Embedding,
                base_url: serve_once(200, body),
                model: "embed-model".to_string(),
                api_key_env: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.embedder = Some(EmbedderRoleConfig {
            provider: "remote_embed".to_string(),
            batch_size: 64,
        });

        let built = build_inference_clients(&config).expect("build clients");
        let embedder = built.embedder.expect("embedder should exist");
        let vectors = embedder
            .embed_batch(EmbeddingInputKind::Document, &["hello".to_string()])
            .expect("embed");
        assert_eq!(vectors, vec![vec![0.1, 0.2]]);
    }

    #[test]
    fn llama_cpp_embedder_builds_document_sizer_against_tokenize_endpoint() {
        let (base_url, requests) = serve_once_capturing_request(200, r#"{"tokens":[1,2,3]}"#);
        let mut config = base_config();
        config.providers.insert(
            "local_embed".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::Embedding,
                base_url,
                model: "embeddinggemma".to_string(),
                parallel_requests: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.embedder = Some(EmbedderRoleConfig {
            provider: "local_embed".to_string(),
            batch_size: 64,
        });

        let built = build_inference_clients(&config).expect("build clients");
        let sizer = built
            .embedding_document_sizer
            .expect("document sizer should exist");
        let token_count = sizer
            .count_document_tokens("hello world")
            .expect("count tokens");
        assert_eq!(token_count, 3);

        let raw_request = requests
            .recv_timeout(Duration::from_secs(1))
            .expect("captured request");
        let request = String::from_utf8(raw_request).expect("utf8 request");
        assert!(
            request.starts_with("POST /tokenize HTTP/1.1\r\n"),
            "unexpected request line: {request}"
        );
        assert!(
            request.contains("\"content\":\"hello world\""),
            "missing content payload: {request}"
        );
        assert!(
            request.contains("\"add_special\":true"),
            "missing add_special payload: {request}"
        );
    }

    #[test]
    fn openai_compatible_embedder_does_not_build_document_sizer() {
        let body = r#"{"data":[{"index":0,"embedding":[0.1,0.2]}]}"#;
        let mut config = base_config();
        config.providers.insert(
            "remote_embed".to_string(),
            ProviderProfileConfig::OpenAiCompatible {
                operation: ProviderOperation::Embedding,
                base_url: serve_once(200, body),
                model: "embed-model".to_string(),
                api_key_env: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.embedder = Some(EmbedderRoleConfig {
            provider: "remote_embed".to_string(),
            batch_size: 64,
        });

        let built = build_inference_clients(&config).expect("build clients");
        assert!(built.embedding_document_sizer.is_none());
    }

    #[test]
    fn provider_profile_reranker_supports_native_rerank_endpoint() {
        let body = r#"{"model":"qwen3-reranker","object":"list","usage":{"prompt_tokens":153,"total_tokens":153},"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.2}]}"#;
        let mut config = base_config();
        config.providers.insert(
            "local_rerank".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::Reranking,
                base_url: serve_once(200, body),
                model: "qwen3-reranker".to_string(),
                parallel_requests: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.reranker = Some(RerankerRoleConfig {
            provider: "local_rerank".to_string(),
        });

        let built = build_inference_clients(&config).expect("build clients");
        let reranker = built.reranker.expect("reranker should exist");
        let scores = reranker
            .rerank("query", &["doc one".to_string(), "doc two".to_string()])
            .expect("rerank");
        assert_eq!(scores, vec![0.2, 0.9]);
    }

    #[test]
    fn llama_cpp_native_reranker_shards_large_batches() {
        let (base_url, captured_docs, max_inflight) = serve_llama_rerank_shards(4);
        let mut config = base_config();
        config.providers.insert(
            "local_rerank".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::Reranking,
                base_url,
                model: "qwen3-reranker".to_string(),
                parallel_requests: Some(4),
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.reranker = Some(RerankerRoleConfig {
            provider: "local_rerank".to_string(),
        });

        let built = build_inference_clients(&config).expect("build clients");
        let reranker = built.reranker.expect("reranker should exist");
        let docs = (0..10)
            .map(|index| format!("doc {index}"))
            .collect::<Vec<_>>();
        let scores = reranker.rerank("query", &docs).expect("rerank");

        assert_eq!(
            scores,
            (0..10).map(|index| index as f32).collect::<Vec<_>>()
        );

        let mut captured = (0..10)
            .map(|_| captured_docs.recv_timeout(Duration::from_secs(1)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("captured shard docs");
        captured.sort_by_key(|doc| {
            doc.strip_prefix("doc ")
                .expect("doc prefix")
                .parse::<usize>()
                .expect("doc index")
        });
        assert_eq!(captured, docs);
        assert!(
            max_inflight.load(Ordering::SeqCst) > 1,
            "expected concurrent shard requests"
        );
    }

    #[test]
    fn llama_cpp_native_reranker_returns_shard_errors() {
        let (base_url, _captured_docs, _max_inflight) = serve_llama_rerank_shards(2);
        let mut config = base_config();
        config.providers.insert(
            "local_rerank".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::Reranking,
                base_url,
                model: "qwen3-reranker".to_string(),
                parallel_requests: Some(2),
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.reranker = Some(RerankerRoleConfig {
            provider: "local_rerank".to_string(),
        });

        let built = build_inference_clients(&config).expect("build clients");
        let reranker = built.reranker.expect("reranker should exist");
        let err = reranker
            .rerank(
                "query",
                &[
                    "doc 0".to_string(),
                    "doc 1".to_string(),
                    "fail".to_string(),
                    "doc 3".to_string(),
                ],
            )
            .expect_err("shard failure should fail rerank");

        let message = err.to_string();
        assert!(
            message.contains("llama.cpp rerank shard") && message.contains("failed"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn openai_native_reranker_uses_score_field() {
        let body = r#"{"results":[{"index":1,"score":0.9},{"index":0,"score":0.2}]}"#;
        let mut config = base_config();
        config.providers.insert(
            "remote_rerank".to_string(),
            ProviderProfileConfig::OpenAiCompatible {
                operation: ProviderOperation::Reranking,
                base_url: serve_once(200, body),
                model: "rerank-model".to_string(),
                api_key_env: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.reranker = Some(RerankerRoleConfig {
            provider: "remote_rerank".to_string(),
        });

        let built = build_inference_clients(&config).expect("build clients");
        let reranker = built.reranker.expect("reranker should exist");
        let scores = reranker
            .rerank("query", &["doc one".to_string(), "doc two".to_string()])
            .expect("rerank");
        assert_eq!(scores, vec![0.2, 0.9]);
    }

    #[test]
    fn provider_profile_reranker_supports_chat_backed_mode() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "{\"scores\":[0.2,0.9]}"
      }
    }
  ]
}"#;
        let mut config = base_config();
        config.providers.insert(
            "remote_rerank".to_string(),
            ProviderProfileConfig::OpenAiCompatible {
                operation: ProviderOperation::ChatCompletion,
                base_url: serve_once(200, body),
                model: "gpt-5-mini".to_string(),
                api_key_env: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.reranker = Some(RerankerRoleConfig {
            provider: "remote_rerank".to_string(),
        });

        let built = build_inference_clients(&config).expect("build clients");
        let reranker = built.reranker.expect("reranker should exist");
        let scores = reranker
            .rerank("query", &["doc one".to_string(), "doc two".to_string()])
            .expect("rerank");
        assert_eq!(scores, vec![0.2, 0.9]);
    }

    #[test]
    fn provider_profile_expander_uses_role_sampling() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "[\"trait object rust\",\"explain rust traits\"]"
      }
    }
  ]
}"#;
        let mut config = base_config();
        config.providers.insert(
            "local_expand".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::ChatCompletion,
                base_url: serve_once(200, body),
                model: "qwen3-1.7b".to_string(),
                parallel_requests: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.expander = Some(ExpanderRoleConfig {
            provider: "local_expand".to_string(),
            max_tokens: 256,
            sampling: ExpanderSamplingConfig::default(),
        });

        let built = build_inference_clients(&config).expect("build clients");
        let expander = built.expander.expect("expander should exist");
        let variants = expander.expand("rust traits", 4).expect("expand");
        assert_eq!(
            variants,
            vec![
                "trait object rust".to_string(),
                "explain rust traits".to_string(),
            ]
        );
    }

    #[test]
    fn llama_cpp_expander_posts_v1_chat_request_with_non_thinking_grammar() {
        let body = r#"{
  "choices": [
    {
      "message": {
        "content": "[\"trait object rust\",\"explain rust traits\"]"
      }
    }
  ]
}"#;
        let (base_url, requests) = serve_once_capturing_request(200, body);
        let mut config = base_config();
        config.providers.insert(
            "local_expand".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::ChatCompletion,
                base_url,
                model: "qwen3-1.7b".to_string(),
                parallel_requests: None,
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.expander = Some(ExpanderRoleConfig {
            provider: "local_expand".to_string(),
            max_tokens: 256,
            sampling: ExpanderSamplingConfig::default(),
        });

        let built = build_inference_clients(&config).expect("build clients");
        let expander = built.expander.expect("expander should exist");
        let _ = expander.expand("rust traits", 4).expect("expand");

        let raw_request = requests
            .recv_timeout(Duration::from_secs(1))
            .expect("captured request");
        let request = String::from_utf8(raw_request).expect("utf8 request");
        assert!(
            request.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"),
            "unexpected request line: {request}"
        );
        assert!(
            request.contains("\"chat_template_kwargs\":{\"enable_thinking\":false}"),
            "missing non-thinking kwargs in request: {request}"
        );
        assert!(
            request.contains("array ::= \\\"[\\\" ws elements? ws \\\"]\\\""),
            "missing variants grammar in request: {request}"
        );
    }

    #[test]
    fn openai_expander_rejects_llama_only_sampling_overrides() {
        let mut config = base_config();
        config.providers.insert(
            "remote_expand".to_string(),
            ProviderProfileConfig::OpenAiCompatible {
                operation: ProviderOperation::ChatCompletion,
                base_url: "https://api.openai.com/v1".to_string(),
                model: "gpt-5-mini".to_string(),
                api_key_env: Some("OPENAI_API_KEY".to_string()),
                timeout_ms: 5_000,
                max_retries: 0,
            },
        );
        config.roles.expander = Some(ExpanderRoleConfig {
            provider: "remote_expand".to_string(),
            max_tokens: 512,
            sampling: ExpanderSamplingConfig {
                top_k: 99,
                ..ExpanderSamplingConfig::default()
            },
        });

        let err = build_inference_clients(&config).expect_err("unsupported sampling should fail");
        assert!(err.to_string().contains("does not support top_k"));
    }

    #[test]
    fn build_chat_payload_uses_json_response_format_for_reranker() {
        let options = ChatCompletionRequestOptions::json_object();
        let payload = build_chat_payload("model", "system", "user", &options);
        assert_eq!(payload["response_format"]["type"], "json_object");
    }

    #[test]
    fn build_chat_payload_omits_response_format_for_text_generation() {
        let options = ChatCompletionRequestOptions::text();
        let payload = build_chat_payload("model", "system", "user", &options);
        assert!(payload.get("response_format").is_none());
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

        let mut config = base_config();
        config.providers.insert(
            "remote_embed".to_string(),
            ProviderProfileConfig::OpenAiCompatible {
                operation: ProviderOperation::Embedding,
                base_url,
                model: "embed-model".to_string(),
                api_key_env: None,
                timeout_ms: 5_000,
                max_retries: 1,
            },
        );
        config.roles.embedder = Some(EmbedderRoleConfig {
            provider: "remote_embed".to_string(),
            batch_size: 64,
        });

        let built = build_inference_clients(&config).expect("build clients");
        let embedder = built.embedder.expect("embedder should exist");
        let vectors = embedder
            .embed_batch(EmbeddingInputKind::Document, &["hello".to_string()])
            .expect("embed should retry then succeed");
        assert_eq!(vectors, vec![vec![0.1, 0.2]]);
    }

    #[test]
    fn managed_llama_provider_enables_transport_recovery() {
        let deployment = ProviderDeployment {
            kind: GatewayProviderKind::LlamaCppServer,
            operation: ProviderOperation::Embedding,
            base_url: "http://127.0.0.1:8101".to_string(),
            model: "embeddinggemma".to_string(),
            api_key_env: None,
            timeout_ms: 5_000,
            max_retries: 0,
        };
        let recovery = build_managed_transport_recovery(
            &Arc::new(base_config()),
            "kbolt_local_embed",
            &deployment,
            InferenceClientBuildOptions::with_managed_recovery(None),
        )
        .expect("managed recovery should exist");

        assert_eq!(recovery.label(), "embedder");
    }

    #[test]
    fn unmanaged_or_disabled_providers_do_not_enable_transport_recovery() {
        let llama_deployment = ProviderDeployment {
            kind: GatewayProviderKind::LlamaCppServer,
            operation: ProviderOperation::Embedding,
            base_url: "http://127.0.0.1:8101".to_string(),
            model: "embeddinggemma".to_string(),
            api_key_env: None,
            timeout_ms: 5_000,
            max_retries: 0,
        };
        let openai_deployment = ProviderDeployment {
            kind: GatewayProviderKind::OpenAiCompatible,
            operation: ProviderOperation::Embedding,
            base_url: "https://api.openai.com/v1".to_string(),
            model: "text-embedding-3-large".to_string(),
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            timeout_ms: 5_000,
            max_retries: 0,
        };
        let config = Arc::new(base_config());

        assert!(build_managed_transport_recovery(
            &config,
            "local_embed",
            &llama_deployment,
            InferenceClientBuildOptions::with_managed_recovery(None),
        )
        .is_none());
        assert!(build_managed_transport_recovery(
            &config,
            "kbolt_local_embed",
            &openai_deployment,
            InferenceClientBuildOptions::with_managed_recovery(None),
        )
        .is_none());
        assert!(build_managed_transport_recovery(
            &config,
            "kbolt_local_embed",
            &llama_deployment,
            InferenceClientBuildOptions::without_managed_recovery(),
        )
        .is_none());
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
