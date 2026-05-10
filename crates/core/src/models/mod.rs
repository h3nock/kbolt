use kbolt_types::{ModelInfo, ModelStatus};

use crate::config::Config;
use crate::Result;

mod chat;
mod completion;
mod embedder;
mod expander;
mod gateway;
mod gguf_tokenizer;
mod http;
mod inference;
mod reranker;
mod text;
mod tokenizer;
mod variants_expander;

pub(crate) use embedder::{Embedder, EmbeddingInputKind};
pub(crate) use expander::{normalize_query_text, Expander};
pub(crate) use inference::{
    build_inference_clients, build_inference_clients_with_recovery_notice,
    build_inference_clients_without_managed_recovery,
};
pub(crate) use reranker::Reranker;
pub(crate) use tokenizer::TokenizerRuntime;
#[cfg(test)]
pub(crate) use tokenizer::TokenizerRuntimeKind;

use gateway::{resolve_inference_gateway_bindings, ProviderDeployment};
use http::HttpJsonClient;

pub fn status(config: &Config) -> Result<ModelStatus> {
    let bindings = resolve_inference_gateway_bindings(config)?;
    Ok(ModelStatus {
        embedder: bindings
            .embedder
            .as_ref()
            .map(|binding| readiness_for_binding(&binding.provider_name, &binding.deployment))
            .unwrap_or_else(unconfigured_model_info),
        reranker: bindings
            .reranker
            .as_ref()
            .map(|binding| readiness_for_binding(&binding.provider_name, &binding.deployment))
            .unwrap_or_else(unconfigured_model_info),
        expander: bindings
            .expander
            .as_ref()
            .map(|binding| readiness_for_binding(&binding.provider_name, &binding.deployment))
            .unwrap_or_else(unconfigured_model_info),
    })
}

fn readiness_for_binding(profile: &str, deployment: &ProviderDeployment) -> ModelInfo {
    let client = HttpJsonClient::new(
        &deployment.base_url,
        deployment.api_key_env.as_deref(),
        deployment.timeout_ms,
        0,
        deployment.operation.as_str(),
        deployment.kind.as_str(),
        None,
    );
    let readiness = client.probe_readiness();
    ModelInfo {
        configured: true,
        ready: readiness.ready,
        profile: Some(profile.to_string()),
        kind: Some(deployment.kind.as_str().to_string()),
        operation: Some(deployment.operation.as_str().to_string()),
        model: Some(deployment.model.clone()),
        endpoint: Some(deployment.base_url.clone()),
        issue: readiness.issue,
    }
}

fn unconfigured_model_info() -> ModelInfo {
    ModelInfo {
        configured: false,
        ready: false,
        profile: None,
        kind: None,
        operation: None,
        model: None,
        endpoint: None,
        issue: None,
    }
}

#[cfg(test)]
mod tests;
