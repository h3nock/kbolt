use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

use kbolt_types::KboltError;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::models::Reranker;
use crate::Result;

use super::llama_backend;

pub(super) struct LocalCrossEncoderReranker {
    model: Arc<LlamaModel>,
    chat_template: LlamaChatTemplate,
    n_ctx: u32,
    inference_lock: Mutex<()>,
}

impl LocalCrossEncoderReranker {
    pub(super) fn new(
        model: Arc<LlamaModel>,
        chat_template: LlamaChatTemplate,
        n_ctx: u32,
    ) -> Self {
        Self {
            model,
            chat_template,
            n_ctx,
            inference_lock: Mutex::new(()),
        }
    }
}

impl Reranker for LocalCrossEncoderReranker {
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| KboltError::Inference("local reranker mutex poisoned".to_string()))?;

        let tokenized = docs
            .iter()
            .map(|doc| tokenize_rerank_prompt(&self.model, &self.chat_template, query, doc))
            .collect::<Result<Vec<_>>>()?;
        score_docs(&self.model, self.n_ctx, &tokenized)
    }
}

fn tokenize_rerank_prompt(
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    query: &str,
    document: &str,
) -> Result<Vec<LlamaToken>> {
    let system_content =
        "Judge whether the Document is relevant to the Query. Answer only \"yes\" or \"no\".";
    let user_content = format!("<Query>: {query}\n<Document>: {document}");

    let messages = vec![
        LlamaChatMessage::new("system".to_string(), system_content.to_string()).map_err(|err| {
            KboltError::Inference(format!("reranker chat message build failed: {err}"))
        })?,
        LlamaChatMessage::new("user".to_string(), user_content).map_err(|err| {
            KboltError::Inference(format!("reranker chat message build failed: {err}"))
        })?,
    ];

    let prompt = model
        .apply_chat_template(template, &messages, true)
        .map_err(|err| {
            KboltError::Inference(format!("reranker chat template apply failed: {err}"))
        })?;

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|err| KboltError::Inference(format!("reranker tokenization failed: {err}")))?;

    if tokens.is_empty() {
        return Err(
            KboltError::Inference("reranker tokenization returned 0 tokens".to_string()).into(),
        );
    }

    Ok(tokens)
}

fn score_docs(model: &LlamaModel, n_ctx: u32, tokenized: &[Vec<LlamaToken>]) -> Result<Vec<f32>> {
    if tokenized.is_empty() {
        return Ok(Vec::new());
    }

    let backend = llama_backend();
    let max_seq_tokens = tokenized.iter().map(Vec::len).max().unwrap_or(0) as u32;
    let effective_ctx = resolve_reranker_context_size(n_ctx, max_seq_tokens)?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(effective_ctx))
        .with_n_batch(effective_ctx)
        .with_n_ubatch(effective_ctx)
        .with_n_seq_max(1)
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Rank);

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|err| KboltError::Inference(format!("reranker context creation failed: {err}")))?;

    let mut scores = Vec::with_capacity(tokenized.len());
    for tokens in tokenized {
        ctx.clear_kv_cache();

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch.add_sequence(tokens, 0, false).map_err(|err| {
            KboltError::Inference(format!("reranker batch creation failed: {err}"))
        })?;

        ctx.decode(&mut batch)
            .map_err(|err| KboltError::Inference(format!("reranker decode failed: {err}")))?;

        let embeddings = ctx.embeddings_seq_ith(0).map_err(|err| {
            KboltError::Inference(format!("reranker embedding read failed: {err}"))
        })?;

        let score = extract_rank_pooling_score(&embeddings)?;
        scores.push(score);
    }

    Ok(scores)
}

fn extract_rank_pooling_score(embeddings: &[f32]) -> Result<f32> {
    match embeddings.len() {
        0 => Err(KboltError::Inference("reranker returned empty embeddings".to_string()).into()),
        1 => Ok(embeddings[0]),
        2 => Ok(embeddings[1]),
        len => Err(KboltError::Inference(format!(
            "reranker returned unsupported rank output shape with {len} values"
        ))
        .into()),
    }
}

fn resolve_reranker_context_size(configured_n_ctx: u32, required_tokens: u32) -> Result<u32> {
    if required_tokens > configured_n_ctx {
        return Err(KboltError::Inference(format!(
            "local reranker request requires {required_tokens} context tokens but n_ctx is configured as {configured_n_ctx}"
        ))
        .into());
    }

    Ok(required_tokens)
}

#[cfg(test)]
mod tests {
    use super::{extract_rank_pooling_score, resolve_reranker_context_size};

    #[test]
    fn reranker_context_size_respects_configured_ceiling() {
        assert_eq!(resolve_reranker_context_size(128, 96).expect("fits"), 96);

        let err = resolve_reranker_context_size(128, 129).expect_err("must reject overflow");
        assert!(err.to_string().contains("requires 129 context tokens"));
    }

    #[test]
    fn reranker_rank_pooling_score_uses_supported_shapes_only() {
        assert_eq!(extract_rank_pooling_score(&[0.25]).expect("single"), 0.25);
        assert_eq!(extract_rank_pooling_score(&[0.1, 0.9]).expect("pair"), 0.9);
        assert!(extract_rank_pooling_score(&[]).is_err());
        assert!(extract_rank_pooling_score(&[0.1, 0.2, 0.3]).is_err());
    }
}
