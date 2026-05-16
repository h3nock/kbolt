use std::collections::HashMap;

use kbolt_types::KboltError;

use crate::config::{
    Config, EmbedderRoleConfig, ExpanderRoleConfig, ExpanderSamplingConfig, ProviderOperation,
    ProviderProfileConfig, RerankerRoleConfig,
};
use crate::local::{MANAGED_RERANKER_PARALLEL_REQUESTS, MANAGED_RERANK_PROVIDER};
use crate::Result;

const DEFAULT_RERANK_PARALLEL_REQUESTS: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GatewayProviderKind {
    LlamaCppServer,
    OpenAiCompatible,
}

impl GatewayProviderKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::LlamaCppServer => "llama_cpp_server",
            Self::OpenAiCompatible => "openai_compatible",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProviderDeployment {
    pub kind: GatewayProviderKind,
    pub operation: ProviderOperation,
    pub base_url: String,
    pub model: String,
    pub api_key_env: Option<String>,
    pub timeout_ms: u64,
    pub max_retries: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbedderBinding {
    pub provider_name: String,
    pub deployment: ProviderDeployment,
    pub batch_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RerankerBinding {
    pub provider_name: String,
    pub deployment: ProviderDeployment,
    pub parallel_requests: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpanderBinding {
    pub provider_name: String,
    pub deployment: ProviderDeployment,
    pub max_tokens: usize,
    pub sampling: ExpanderSamplingConfig,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct InferenceGatewayBindings {
    pub embedder: Option<EmbedderBinding>,
    pub reranker: Option<RerankerBinding>,
    pub expander: Option<ExpanderBinding>,
}

pub(crate) fn resolve_inference_gateway_bindings(
    config: &Config,
) -> Result<InferenceGatewayBindings> {
    Ok(InferenceGatewayBindings {
        embedder: config
            .roles
            .embedder
            .as_ref()
            .map(|role| {
                resolve_embedder_binding(role, &config.providers, &[ProviderOperation::Embedding])
            })
            .transpose()?,
        reranker: config
            .roles
            .reranker
            .as_ref()
            .map(|role| {
                resolve_reranker_binding(
                    role,
                    &config.providers,
                    &[
                        ProviderOperation::Reranking,
                        ProviderOperation::ChatCompletion,
                    ],
                )
            })
            .transpose()?,
        expander: config
            .roles
            .expander
            .as_ref()
            .map(|role| {
                resolve_expander_binding(
                    role,
                    &config.providers,
                    &[ProviderOperation::ChatCompletion],
                )
            })
            .transpose()?,
    })
}

fn resolve_embedder_binding(
    role: &EmbedderRoleConfig,
    providers: &HashMap<String, ProviderProfileConfig>,
    allowed_operations: &[ProviderOperation],
) -> Result<EmbedderBinding> {
    let (provider_name, deployment) = resolve_provider_deployment(
        "roles.embedder",
        &role.provider,
        providers,
        allowed_operations,
    )?;
    Ok(EmbedderBinding {
        provider_name,
        deployment,
        batch_size: role.batch_size,
    })
}

fn resolve_reranker_binding(
    role: &RerankerRoleConfig,
    providers: &HashMap<String, ProviderProfileConfig>,
    allowed_operations: &[ProviderOperation],
) -> Result<RerankerBinding> {
    let (provider_name, deployment) = resolve_provider_deployment(
        "roles.reranker",
        &role.provider,
        providers,
        allowed_operations,
    )?;
    let parallel_requests = providers
        .get(&provider_name)
        .and_then(ProviderProfileConfig::parallel_requests)
        .unwrap_or_else(|| default_rerank_parallel_requests(&provider_name, &deployment));
    Ok(RerankerBinding {
        provider_name,
        deployment,
        parallel_requests,
    })
}

fn default_rerank_parallel_requests(provider_name: &str, deployment: &ProviderDeployment) -> usize {
    if provider_name == MANAGED_RERANK_PROVIDER
        && deployment.kind == GatewayProviderKind::LlamaCppServer
        && deployment.operation == ProviderOperation::Reranking
    {
        MANAGED_RERANKER_PARALLEL_REQUESTS
    } else {
        DEFAULT_RERANK_PARALLEL_REQUESTS
    }
}

fn resolve_expander_binding(
    role: &ExpanderRoleConfig,
    providers: &HashMap<String, ProviderProfileConfig>,
    allowed_operations: &[ProviderOperation],
) -> Result<ExpanderBinding> {
    let (provider_name, deployment) = resolve_provider_deployment(
        "roles.expander",
        &role.provider,
        providers,
        allowed_operations,
    )?;
    Ok(ExpanderBinding {
        provider_name,
        deployment,
        max_tokens: role.max_tokens,
        sampling: role.sampling.clone(),
    })
}

fn resolve_provider_deployment(
    scope: &str,
    provider_name: &str,
    providers: &HashMap<String, ProviderProfileConfig>,
    allowed_operations: &[ProviderOperation],
) -> Result<(String, ProviderDeployment)> {
    if provider_name.trim().is_empty() {
        return Err(KboltError::Config(format!("{scope}.provider must not be empty")).into());
    }

    let Some(profile) = providers.get(provider_name) else {
        return Err(KboltError::Config(format!(
            "{scope}.provider references undefined provider profile '{provider_name}'"
        ))
        .into());
    };

    let operation = profile.operation();
    if !allowed_operations.contains(&operation) {
        return Err(KboltError::Config(format!(
            "{scope}.provider '{provider_name}' uses incompatible operation '{}'",
            operation.as_str()
        ))
        .into());
    }

    Ok((
        provider_name.to_string(),
        ProviderDeployment {
            kind: match profile {
                ProviderProfileConfig::LlamaCppServer { .. } => GatewayProviderKind::LlamaCppServer,
                ProviderProfileConfig::OpenAiCompatible { .. } => {
                    GatewayProviderKind::OpenAiCompatible
                }
            },
            operation,
            base_url: profile.base_url().to_string(),
            model: profile.model().to_string(),
            api_key_env: profile.api_key_env().map(ToString::to_string),
            timeout_ms: profile.timeout_ms(),
            max_retries: profile.max_retries(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::*;
    use crate::config::{
        ChunkingConfig, Config, ExpanderRoleConfig, RankingConfig, ReapingConfig,
        RoleBindingsConfig,
    };

    fn base_config() -> Config {
        Config {
            config_dir: PathBuf::from("/tmp/config"),
            cache_dir: PathBuf::from("/tmp/cache"),
            default_space: None,
            providers: HashMap::new(),
            roles: RoleBindingsConfig::default(),
            reaping: ReapingConfig { days: 7 },
            chunking: ChunkingConfig::default(),
            ranking: RankingConfig::default(),
        }
    }

    #[test]
    fn resolve_gateway_bindings_maps_provider_profiles() {
        let mut config = base_config();
        config.providers.insert(
            "local_embed".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::Embedding,
                base_url: "http://127.0.0.1:8101".to_string(),
                model: "embeddinggemma".to_string(),
                parallel_requests: None,
                timeout_ms: 30_000,
                max_retries: 2,
            },
        );
        config.providers.insert(
            "remote_rerank".to_string(),
            ProviderProfileConfig::OpenAiCompatible {
                operation: ProviderOperation::ChatCompletion,
                base_url: "https://api.openai.com/v1".to_string(),
                model: "gpt-5-mini".to_string(),
                api_key_env: Some("OPENAI_API_KEY".to_string()),
                timeout_ms: 30_000,
                max_retries: 2,
            },
        );
        config.roles.embedder = Some(EmbedderRoleConfig {
            provider: "local_embed".to_string(),
            batch_size: 16,
        });
        config.roles.reranker = Some(RerankerRoleConfig {
            provider: "remote_rerank".to_string(),
        });
        config.roles.expander = Some(ExpanderRoleConfig {
            provider: "remote_rerank".to_string(),
            max_tokens: 512,
            sampling: ExpanderSamplingConfig::default(),
        });

        let bindings = resolve_inference_gateway_bindings(&config).expect("resolve bindings");

        assert_eq!(
            bindings.embedder,
            Some(EmbedderBinding {
                provider_name: "local_embed".to_string(),
                deployment: ProviderDeployment {
                    kind: GatewayProviderKind::LlamaCppServer,
                    operation: ProviderOperation::Embedding,
                    base_url: "http://127.0.0.1:8101".to_string(),
                    model: "embeddinggemma".to_string(),
                    api_key_env: None,
                    timeout_ms: 30_000,
                    max_retries: 2,
                },
                batch_size: 16,
            })
        );
        assert_eq!(
            bindings.reranker,
            Some(RerankerBinding {
                provider_name: "remote_rerank".to_string(),
                deployment: ProviderDeployment {
                    kind: GatewayProviderKind::OpenAiCompatible,
                    operation: ProviderOperation::ChatCompletion,
                    base_url: "https://api.openai.com/v1".to_string(),
                    model: "gpt-5-mini".to_string(),
                    api_key_env: Some("OPENAI_API_KEY".to_string()),
                    timeout_ms: 30_000,
                    max_retries: 2,
                },
                parallel_requests: 1,
            })
        );
        assert_eq!(
            bindings
                .expander
                .expect("expander binding should exist")
                .provider_name,
            "remote_rerank"
        );
    }

    #[test]
    fn resolve_gateway_bindings_carries_llama_rerank_parallel_requests() {
        let mut config = base_config();
        config.providers.insert(
            "local_rerank".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::Reranking,
                base_url: "http://127.0.0.1:8102".to_string(),
                model: "qwen3-reranker".to_string(),
                parallel_requests: Some(4),
                timeout_ms: 30_000,
                max_retries: 2,
            },
        );
        config.roles.reranker = Some(RerankerRoleConfig {
            provider: "local_rerank".to_string(),
        });

        let bindings = resolve_inference_gateway_bindings(&config).expect("resolve bindings");

        assert_eq!(
            bindings
                .reranker
                .expect("reranker binding should exist")
                .parallel_requests,
            4
        );
    }

    #[test]
    fn resolve_gateway_bindings_defaults_legacy_managed_rerank_to_managed_parallelism() {
        let mut config = base_config();
        config.providers.insert(
            MANAGED_RERANK_PROVIDER.to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::Reranking,
                base_url: "http://127.0.0.1:8102".to_string(),
                model: "qwen3-reranker".to_string(),
                parallel_requests: None,
                timeout_ms: 30_000,
                max_retries: 2,
            },
        );
        config.roles.reranker = Some(RerankerRoleConfig {
            provider: MANAGED_RERANK_PROVIDER.to_string(),
        });

        let bindings = resolve_inference_gateway_bindings(&config).expect("resolve bindings");

        assert_eq!(
            bindings
                .reranker
                .expect("reranker binding should exist")
                .parallel_requests,
            MANAGED_RERANKER_PARALLEL_REQUESTS
        );
    }

    #[test]
    fn resolve_gateway_bindings_rejects_missing_provider() {
        let mut config = base_config();
        config.roles.reranker = Some(RerankerRoleConfig {
            provider: "missing".to_string(),
        });

        let err = resolve_inference_gateway_bindings(&config).expect_err("missing provider");
        assert!(err
            .to_string()
            .contains("roles.reranker.provider references undefined provider profile 'missing'"));
    }

    #[test]
    fn resolve_gateway_bindings_rejects_incompatible_operation() {
        let mut config = base_config();
        config.providers.insert(
            "local_expand".to_string(),
            ProviderProfileConfig::LlamaCppServer {
                operation: ProviderOperation::ChatCompletion,
                base_url: "http://127.0.0.1:8103".to_string(),
                model: "qwen3".to_string(),
                parallel_requests: None,
                timeout_ms: 30_000,
                max_retries: 2,
            },
        );
        config.roles.embedder = Some(EmbedderRoleConfig {
            provider: "local_expand".to_string(),
            batch_size: 16,
        });

        let err = resolve_inference_gateway_bindings(&config).expect_err("incompatible binding");
        assert!(err.to_string().contains(
            "roles.embedder.provider 'local_expand' uses incompatible operation 'chat_completion'"
        ));
    }
}
