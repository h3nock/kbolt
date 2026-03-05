use std::path::Path;
use std::sync::Arc;

use kbolt_types::KboltError;
use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};

use crate::models::completion::CompletionClient;
use crate::models::text::strip_json_fences;
use crate::Result;

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

        let prompt = format!("System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n");
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

pub(super) fn build_local_llama_completion_client(
    model_path: &Path,
    max_tokens: usize,
    n_ctx: u32,
    n_gpu_layers: u32,
) -> Result<Arc<dyn CompletionClient>> {
    let client = LocalLlamaClient::new(model_path, max_tokens, n_ctx, n_gpu_layers)?;
    Ok(Arc::new(client))
}
