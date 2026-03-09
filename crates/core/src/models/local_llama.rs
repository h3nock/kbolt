use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use kbolt_types::KboltError;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::models::completion::CompletionClient;
use crate::models::text::strip_json_fences;
use crate::Result;

use super::llama_backend;

pub(super) struct LocalLlamaClient {
    model: Arc<LlamaModel>,
    chat_template: Option<LlamaChatTemplate>,
    max_tokens: usize,
    n_ctx: u32,
}

impl LocalLlamaClient {
    pub(super) fn new(
        model_path: &Path,
        max_tokens: usize,
        n_ctx: u32,
        n_gpu_layers: Option<u32>,
    ) -> Result<Self> {
        let (model, chat_template) = load_local_llama_model_and_template(model_path, n_gpu_layers)?;

        Ok(Self {
            model,
            chat_template,
            max_tokens,
            n_ctx,
        })
    }

    /// Format a system+user prompt using the model's embedded chat template if available,
    /// falling back to a generic format.
    fn format_prompt(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let Some(tmpl) = &self.chat_template else {
            return Ok(format!(
                "System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
            ));
        };

        let mut messages = Vec::new();
        if !system_prompt.is_empty() {
            messages.push(
                LlamaChatMessage::new("system".to_string(), system_prompt.to_string()).map_err(
                    |err| KboltError::Inference(format!("failed to build chat message: {err}")),
                )?,
            );
        }
        messages.push(
            LlamaChatMessage::new("user".to_string(), user_prompt.to_string()).map_err(|err| {
                KboltError::Inference(format!("failed to build chat message: {err}"))
            })?,
        );

        self.model
            .apply_chat_template(tmpl, &messages, true)
            .map_err(|err| {
                KboltError::Inference(format!("failed to apply chat template: {err}")).into()
            })
    }
}

impl CompletionClient for LocalLlamaClient {
    fn complete(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let backend = llama_backend();

        let prompt = self.format_prompt(system_prompt, user_prompt)?;
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|err| KboltError::Inference(format!("llama tokenization failed: {err}")))?;

        // Context size must fit the prompt + max generation tokens.
        let needed_ctx = (tokens.len() + self.max_tokens) as u32;
        if needed_ctx > self.n_ctx {
            return Err(KboltError::Inference(format!(
                "local llama request requires {needed_ctx} context tokens but n_ctx is configured as {}",
                self.n_ctx
            ))
            .into());
        }
        let n_ctx = needed_ctx;
        let n_batch = u32::try_from(tokens.len()).map_err(|_| {
            KboltError::Inference(
                "llama prompt token count exceeds supported batch size".to_string(),
            )
        })?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_batch);

        let mut ctx = self.model.new_context(backend, ctx_params).map_err(|err| {
            KboltError::Inference(format!("failed to create llama context: {err}"))
        })?;

        // Feed the prompt in chunks of n_batch to avoid exceeding batch limits.
        let batch_size = n_batch as usize;
        for (chunk_idx, chunk) in tokens.chunks(batch_size).enumerate() {
            let offset = chunk_idx * batch_size;
            let is_last_chunk = offset + chunk.len() >= tokens.len();

            let mut batch = LlamaBatch::new(chunk.len(), 1);
            for (i, &token) in chunk.iter().enumerate() {
                let pos = (offset + i) as i32;
                let logits = is_last_chunk && i == chunk.len() - 1;
                batch.add(token, pos, &[0], logits).map_err(|err| {
                    KboltError::Inference(format!("llama batch add failed: {err}"))
                })?;
            }

            ctx.decode(&mut batch).map_err(|err| {
                KboltError::Inference(format!("llama prompt decode failed: {err}"))
            })?;
        }

        let mut sampler = LlamaSampler::greedy();
        let mut output_tokens = Vec::new();
        let mut cur_pos = tokens.len() as i32;

        for _ in 0..self.max_tokens {
            let token = sampler.sample(&ctx, -1);

            if self.model.is_eog_token(token) {
                break;
            }

            output_tokens.push(token);

            let mut batch = LlamaBatch::new(1, 1);
            batch
                .add(token, cur_pos, &[0], true)
                .map_err(|err| KboltError::Inference(format!("llama batch add failed: {err}")))?;
            cur_pos += 1;

            ctx.decode(&mut batch)
                .map_err(|err| KboltError::Inference(format!("llama decode failed: {err}")))?;
        }

        let text = tokens_to_string(&self.model, &output_tokens)?;

        Ok(strip_json_fences(&text).trim().to_string())
    }
}

pub(super) fn load_local_llama_model_and_template(
    model_path: &Path,
    n_gpu_layers: Option<u32>,
) -> Result<(Arc<LlamaModel>, Option<LlamaChatTemplate>)> {
    let backend = llama_backend();
    let model_params = match n_gpu_layers_for_model_params(n_gpu_layers) {
        Some(n_gpu_layers) => LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers),
        None => LlamaModelParams::default(),
    };
    let model = LlamaModel::load_from_file(backend, model_path, &model_params).map_err(|err| {
        KboltError::Inference(format!(
            "failed to load local llama model {}: {err}",
            model_path.display()
        ))
    })?;
    let chat_template = model.chat_template(None).ok();
    Ok((Arc::new(model), chat_template))
}

/// Convert tokens to string using `token_to_bytes` which properly retries on buffer overflow.
#[allow(deprecated)]
fn tokens_to_string(
    model: &LlamaModel,
    tokens: &[llama_cpp_2::token::LlamaToken],
) -> Result<String> {
    use llama_cpp_2::model::Special;
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for &token in tokens {
        let piece = model
            .token_to_bytes(token, Special::Plaintext)
            .map_err(|err| KboltError::Inference(format!("llama token decode failed: {err}")))?;
        bytes.extend_from_slice(&piece);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

pub(super) fn build_local_llama_completion_client(
    model_path: &Path,
    max_tokens: usize,
    n_ctx: u32,
    n_gpu_layers: Option<u32>,
) -> Result<Arc<dyn CompletionClient>> {
    let client = LocalLlamaClient::new(model_path, max_tokens, n_ctx, n_gpu_layers)?;
    Ok(Arc::new(client))
}

pub(super) fn build_local_llama_client_shared(
    model_path: &Path,
    max_tokens: usize,
    n_ctx: u32,
    n_gpu_layers: Option<u32>,
) -> Result<Arc<LocalLlamaClient>> {
    let client = LocalLlamaClient::new(model_path, max_tokens, n_ctx, n_gpu_layers)?;
    Ok(Arc::new(client))
}

#[cfg(test)]
mod tests {
    #[test]
    fn zero_gpu_layers_now_resolve_to_auto() {
        assert_eq!(super::n_gpu_layers_for_model_params(None), None);
        assert_eq!(super::n_gpu_layers_for_model_params(Some(0)), None);
        assert_eq!(super::n_gpu_layers_for_model_params(Some(12)), Some(12));
    }
}

fn n_gpu_layers_for_model_params(n_gpu_layers: Option<u32>) -> Option<u32> {
    n_gpu_layers.filter(|layers| *layers > 0)
}
