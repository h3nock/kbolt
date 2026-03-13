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

pub(super) enum LocalLlamaPrompt<'a> {
    Raw(&'a str),
    Chat {
        system_prompt: &'a str,
        user_prompt: &'a str,
    },
}

#[derive(Debug, Clone, Default)]
pub(super) struct LocalLlamaGenerationOptions {
    pub max_tokens: Option<usize>,
    pub sampler: LocalLlamaSamplerConfig,
    pub stop_sequences: Vec<String>,
    pub grammar: Option<LocalLlamaGrammar>,
}

#[derive(Debug, Clone, Default)]
pub(super) enum LocalLlamaSamplerConfig {
    #[default]
    Greedy,
    Sample(LocalLlamaSamplingParams),
}

#[derive(Debug, Clone)]
pub(super) struct LocalLlamaSamplingParams {
    pub seed: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_last_n: i32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

#[derive(Debug, Clone)]
pub(super) struct LocalLlamaGrammar {
    pub grammar: String,
    pub root: String,
}

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

    pub(super) fn generate(
        &self,
        prompt: LocalLlamaPrompt<'_>,
        options: &LocalLlamaGenerationOptions,
    ) -> Result<String> {
        let backend = llama_backend();
        let rendered_prompt = self.render_prompt(prompt)?;
        let prompt_tokens = self
            .model
            .str_to_token(&rendered_prompt, AddBos::Always)
            .map_err(|err| KboltError::Inference(format!("llama tokenization failed: {err}")))?;
        let max_tokens = options.max_tokens.unwrap_or(self.max_tokens);

        let needed_ctx = u32::try_from(prompt_tokens.len() + max_tokens).map_err(|_| {
            KboltError::Inference(
                "local llama request exceeds supported context window size".to_string(),
            )
        })?;
        if needed_ctx > self.n_ctx {
            return Err(KboltError::Inference(format!(
                "local llama request requires {needed_ctx} context tokens but n_ctx is configured as {}",
                self.n_ctx
            ))
            .into());
        }

        let n_batch = u32::try_from(prompt_tokens.len()).map_err(|_| {
            KboltError::Inference(
                "llama prompt token count exceeds supported batch size".to_string(),
            )
        })?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(needed_ctx))
            .with_n_batch(n_batch);
        let mut ctx = self.model.new_context(backend, ctx_params).map_err(|err| {
            KboltError::Inference(format!("failed to create llama context: {err}"))
        })?;

        let batch_size = n_batch as usize;
        for (chunk_idx, chunk) in prompt_tokens.chunks(batch_size).enumerate() {
            let offset = chunk_idx * batch_size;
            let is_last_chunk = offset + chunk.len() >= prompt_tokens.len();

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

        let mut sampler = self.build_sampler(options)?;
        let stop_sequences = options
            .stop_sequences
            .iter()
            .filter(|sequence| !sequence.is_empty())
            .map(|sequence| sequence.as_bytes().to_vec())
            .collect::<Vec<_>>();

        let mut output_bytes = Vec::new();
        let mut cur_pos = prompt_tokens.len() as i32;

        for _ in 0..max_tokens {
            // The llama sampler API accepts sampled tokens internally; calling accept() again
            // corrupts grammar/repetition state.
            let token = sampler.sample(&ctx, -1);
            if self.model.is_eog_token(token) {
                break;
            }

            output_bytes.extend_from_slice(&decode_token_bytes(&self.model, token)?);
            if let Some(stop_len) = stop_sequences
                .iter()
                .find_map(|stop| output_bytes.ends_with(stop).then_some(stop.len()))
            {
                output_bytes.truncate(output_bytes.len().saturating_sub(stop_len));
                break;
            }

            let mut batch = LlamaBatch::new(1, 1);
            batch
                .add(token, cur_pos, &[0], true)
                .map_err(|err| KboltError::Inference(format!("llama batch add failed: {err}")))?;
            cur_pos += 1;

            ctx.decode(&mut batch)
                .map_err(|err| KboltError::Inference(format!("llama decode failed: {err}")))?;
        }

        Ok(String::from_utf8_lossy(&output_bytes).into_owned())
    }

    fn build_sampler(&self, options: &LocalLlamaGenerationOptions) -> Result<LlamaSampler> {
        let mut samplers = Vec::new();

        if let Some(grammar) = &options.grammar {
            samplers.push(
                LlamaSampler::grammar(&self.model, &grammar.grammar, &grammar.root).map_err(
                    |err| {
                        KboltError::Inference(format!(
                            "failed to initialize grammar sampler: {err}"
                        ))
                    },
                )?,
            );
        }

        match &options.sampler {
            LocalLlamaSamplerConfig::Greedy => samplers.push(LlamaSampler::greedy()),
            LocalLlamaSamplerConfig::Sample(params) => {
                validate_sampling_params(params)?;
                samplers.push(LlamaSampler::penalties(
                    params.repeat_last_n,
                    params.repeat_penalty,
                    params.frequency_penalty,
                    params.presence_penalty,
                ));
                samplers.push(LlamaSampler::top_k(params.top_k));
                samplers.push(LlamaSampler::top_p(params.top_p, 1));
                samplers.push(LlamaSampler::temp(params.temperature));
                samplers.push(LlamaSampler::dist(params.seed));
            }
        }

        Ok(if samplers.len() == 1 {
            samplers
                .into_iter()
                .next()
                .expect("sampler chain should contain at least one sampler")
        } else {
            LlamaSampler::chain_simple(samplers)
        })
    }

    fn render_prompt(&self, prompt: LocalLlamaPrompt<'_>) -> Result<String> {
        match prompt {
            LocalLlamaPrompt::Raw(prompt) => Ok(prompt.to_string()),
            LocalLlamaPrompt::Chat {
                system_prompt,
                user_prompt,
            } => self.format_chat_prompt(system_prompt, user_prompt),
        }
    }

    fn format_chat_prompt(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
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
        let text = self.generate(
            LocalLlamaPrompt::Chat {
                system_prompt,
                user_prompt,
            },
            &LocalLlamaGenerationOptions::default(),
        )?;

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

fn decode_token_bytes(
    model: &LlamaModel,
    token: llama_cpp_2::token::LlamaToken,
) -> Result<Vec<u8>> {
    match model.token_to_piece_bytes(token, 8, false, None) {
        Ok(bytes) => Ok(bytes),
        Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(size)) => model
            .token_to_piece_bytes(token, (-size) as usize, false, None)
            .map_err(|err| {
                KboltError::Inference(format!("llama token decode failed: {err}")).into()
            }),
        Err(err) => Err(KboltError::Inference(format!("llama token decode failed: {err}")).into()),
    }
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

fn validate_sampling_params(params: &LocalLlamaSamplingParams) -> Result<()> {
    if params.temperature <= 0.0 {
        return Err(KboltError::Inference(
            "local llama sampling temperature must be greater than zero".to_string(),
        )
        .into());
    }
    if params.top_k <= 0 {
        return Err(KboltError::Inference(
            "local llama sampling top_k must be greater than zero".to_string(),
        )
        .into());
    }
    if !(0.0 < params.top_p && params.top_p <= 1.0) {
        return Err(KboltError::Inference(
            "local llama sampling top_p must be in the range (0, 1]".to_string(),
        )
        .into());
    }
    if params.repeat_last_n < -1 {
        return Err(KboltError::Inference(
            "local llama repeat_last_n must be greater than or equal to -1".to_string(),
        )
        .into());
    }
    if params.repeat_penalty <= 0.0 {
        return Err(KboltError::Inference(
            "local llama repeat_penalty must be greater than zero".to_string(),
        )
        .into());
    }

    Ok(())
}
