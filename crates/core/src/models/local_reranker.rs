use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

use kbolt_types::KboltError;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::models::Reranker;
use crate::Result;

use crate::config::LlamaFlashAttentionMode;

use super::{llama_backend, llama_flash_attention_policy};

const QWEN3_RERANK_SYSTEM_PROMPT: &str = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".";
const QWEN3_RERANK_INSTRUCT: &str =
    "Given a web search query, retrieve relevant passages that answer the query";
const QWEN3_RERANK_ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

// This adapter is intentionally Qwen3-specific. The local `provider =
// "local_llama"` reranker path currently supports Qwen3-style binary
// rank-pooled GGUF rerankers only.
pub(super) struct LocalQwen3Reranker {
    model: Arc<LlamaModel>,
    n_ctx: u32,
    flash_attention: LlamaFlashAttentionMode,
    inference_lock: Mutex<()>,
}

impl LocalQwen3Reranker {
    pub(super) fn new(
        model: Arc<LlamaModel>,
        n_ctx: u32,
        flash_attention: LlamaFlashAttentionMode,
    ) -> Self {
        Self {
            model,
            n_ctx,
            flash_attention,
            inference_lock: Mutex::new(()),
        }
    }
}

impl Reranker for LocalQwen3Reranker {
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        let _guard = self.inference_lock.lock().map_err(|_| {
            KboltError::Inference("local Qwen3 reranker mutex poisoned".to_string())
        })?;

        let tokenized = docs
            .iter()
            .map(|doc| tokenize_rerank_prompt(&self.model, self.n_ctx, query, doc))
            .collect::<Result<Vec<_>>>()?;
        score_docs(&self.model, self.n_ctx, self.flash_attention, &tokenized)
    }
}

fn tokenize_rerank_prompt(
    model: &LlamaModel,
    n_ctx: u32,
    query: &str,
    document: &str,
) -> Result<Vec<LlamaToken>> {
    let prompt = build_qwen3_rerank_prompt(query, document);
    let tokens = tokenize_prompt(model, &prompt)?;
    if tokens.len() as u32 <= n_ctx {
        return Ok(tokens);
    }

    let base_prompt = build_qwen3_rerank_prompt(query, "");
    let base_tokens = tokenize_prompt(model, &base_prompt)?;
    if base_tokens.len() as u32 > n_ctx {
        return Err(KboltError::Inference(format!(
            "local Qwen3 reranker prompt requires at least {} context tokens before any document text but n_ctx is configured as {n_ctx}",
            base_tokens.len()
        ))
        .into());
    }

    let fitted_document = fit_utf8_prefix_to_token_limit(document, n_ctx as usize, |candidate| {
        let prompt = build_qwen3_rerank_prompt(query, candidate);
        Ok(tokenize_prompt(model, &prompt)?.len())
    })?;
    tokenize_prompt(model, &build_qwen3_rerank_prompt(query, fitted_document))
}

fn build_qwen3_rerank_prompt(query: &str, document: &str) -> String {
    let user_content =
        format!("<Instruct>: {QWEN3_RERANK_INSTRUCT}\n<Query>: {query}\n<Document>: {document}");

    format!(
        "<|im_start|>system\n{QWEN3_RERANK_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n{QWEN3_RERANK_ASSISTANT_PREFIX}"
    )
}

fn tokenize_prompt(model: &LlamaModel, prompt: &str) -> Result<Vec<LlamaToken>> {
    let tokens = model.str_to_token(prompt, AddBos::Always).map_err(|err| {
        KboltError::Inference(format!("Qwen3 reranker tokenization failed: {err}"))
    })?;

    if tokens.is_empty() {
        return Err(KboltError::Inference(
            "Qwen3 reranker tokenization returned 0 tokens".to_string(),
        )
        .into());
    }

    Ok(tokens)
}

fn fit_utf8_prefix_to_token_limit<'a, F>(
    text: &'a str,
    max_tokens: usize,
    mut token_count: F,
) -> Result<&'a str>
where
    F: FnMut(&str) -> Result<usize>,
{
    let mut boundaries = text
        .char_indices()
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    boundaries.push(text.len());

    let mut low = 0usize;
    let mut high = boundaries.len() - 1;
    let mut best_end = 0usize;

    while low <= high {
        let mid = low + (high - low) / 2;
        let end = boundaries[mid];
        let token_len = token_count(&text[..end])?;
        if token_len <= max_tokens {
            best_end = end;
            low = mid.saturating_add(1);
        } else if mid == 0 {
            break;
        } else {
            high = mid - 1;
        }
    }

    Ok(&text[..best_end])
}

fn score_docs(
    model: &LlamaModel,
    n_ctx: u32,
    flash_attention: LlamaFlashAttentionMode,
    tokenized: &[Vec<LlamaToken>],
) -> Result<Vec<f32>> {
    if tokenized.is_empty() {
        return Ok(Vec::new());
    }

    let backend = llama_backend();
    let max_seq_tokens = tokenized.iter().map(Vec::len).max().unwrap_or(0) as u32;
    let effective_ctx = resolve_reranker_context_size(n_ctx, max_seq_tokens)?;

    let ctx_params = reranker_context_params(effective_ctx, flash_attention);

    let mut ctx = model.new_context(backend, ctx_params).map_err(|err| {
        KboltError::Inference(format!("Qwen3 reranker context creation failed: {err}"))
    })?;

    let mut scores = Vec::with_capacity(tokenized.len());
    for tokens in tokenized {
        ctx.clear_kv_cache();

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch.add_sequence(tokens, 0, false).map_err(|err| {
            KboltError::Inference(format!("Qwen3 reranker batch creation failed: {err}"))
        })?;

        ctx.decode(&mut batch)
            .map_err(|err| KboltError::Inference(format!("Qwen3 reranker decode failed: {err}")))?;

        let embeddings = ctx.embeddings_seq_ith(0).map_err(|err| {
            KboltError::Inference(format!("Qwen3 reranker embedding read failed: {err}"))
        })?;

        let score = extract_binary_rank_relevance_score(&embeddings)?;
        scores.push(score);
    }

    Ok(scores)
}

fn reranker_context_params(
    effective_ctx: u32,
    flash_attention: LlamaFlashAttentionMode,
) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(effective_ctx))
        .with_n_batch(effective_ctx)
        .with_n_ubatch(effective_ctx)
        .with_n_seq_max(1)
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Rank)
        .with_flash_attention_policy(llama_flash_attention_policy(flash_attention))
}

fn extract_binary_rank_relevance_score(embeddings: &[f32]) -> Result<f32> {
    match embeddings.len() {
        0 => Err(
            KboltError::Inference("Qwen3 reranker returned empty embeddings".to_string()).into(),
        ),
        // Single logit from rank pooling.
        1 => Ok(embeddings[0]),
        // Qwen3 rank pooling yields [p(yes), p(no)] after softmax in llama.cpp.
        // cls_out weight is built as [true_row, false_row] in convert_hf_to_gguf.py,
        // and llama.cpp server reads embd[0] as the relevance score.
        2 => Ok(embeddings[0]),
        n => Err(KboltError::Inference(format!(
            "Qwen3 reranker returned {n} embedding values, expected 1 or 2 (binary rank pooling)"
        ))
        .into()),
    }
}

fn resolve_reranker_context_size(configured_n_ctx: u32, required_tokens: u32) -> Result<u32> {
    if required_tokens > configured_n_ctx {
        return Err(KboltError::Inference(format!(
            "local Qwen3 reranker request requires {required_tokens} context tokens but n_ctx is configured as {configured_n_ctx}"
        ))
        .into());
    }

    Ok(required_tokens)
}

#[cfg(test)]
mod tests {
    const LLAMA_FLASH_ATTN_TYPE_AUTO: i32 = -1;
    const LLAMA_FLASH_ATTN_TYPE_DISABLED: i32 = 0;
    const LLAMA_FLASH_ATTN_TYPE_ENABLED: i32 = 1;

    use super::{
        build_qwen3_rerank_prompt, extract_binary_rank_relevance_score,
        fit_utf8_prefix_to_token_limit, reranker_context_params, resolve_reranker_context_size,
    };
    use crate::config::LlamaFlashAttentionMode;

    #[test]
    fn reranker_context_size_respects_configured_ceiling() {
        assert_eq!(resolve_reranker_context_size(128, 96).expect("fits"), 96);

        let err = resolve_reranker_context_size(128, 129).expect_err("must reject overflow");
        assert!(err.to_string().contains("requires 129 context tokens"));
    }

    #[test]
    fn reranker_context_params_disable_flash_attention_by_default() {
        let params = reranker_context_params(256, LlamaFlashAttentionMode::Disabled);

        assert_eq!(
            params.flash_attention_policy(),
            LLAMA_FLASH_ATTN_TYPE_DISABLED
        );
    }

    #[test]
    fn reranker_context_params_enable_flash_attention_when_requested() {
        let params = reranker_context_params(256, LlamaFlashAttentionMode::Enabled);

        assert_eq!(
            params.flash_attention_policy(),
            LLAMA_FLASH_ATTN_TYPE_ENABLED
        );
    }

    #[test]
    fn reranker_context_params_use_auto_flash_attention_when_requested() {
        let params = reranker_context_params(256, LlamaFlashAttentionMode::Auto);

        assert_eq!(params.flash_attention_policy(), LLAMA_FLASH_ATTN_TYPE_AUTO);
    }

    #[test]
    fn reranker_rank_pooling_score_uses_supported_shapes_only() {
        assert_eq!(
            extract_binary_rank_relevance_score(&[0.25]).expect("single"),
            0.25
        );
        // [p(yes), p(no)] — relevance score is index 0
        assert_eq!(
            extract_binary_rank_relevance_score(&[0.9, 0.1]).expect("pair"),
            0.9
        );
        // Unsupported shape: n_cls_out > 2 is not a valid Qwen3 binary rank output
        assert!(extract_binary_rank_relevance_score(&[0.7, 0.2, 0.1]).is_err());
        assert!(extract_binary_rank_relevance_score(&[]).is_err());
    }

    #[test]
    fn qwen3_rerank_prompt_matches_documented_contract() {
        let prompt = build_qwen3_rerank_prompt("where is auth configured?", "Look in settings.");

        assert_eq!(
            prompt,
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: where is auth configured?\n<Document>: Look in settings.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn utf8_prefix_fit_preserves_character_boundaries() {
        let fitted = fit_utf8_prefix_to_token_limit("aé😀b", 3, |candidate| {
            Ok(candidate.as_bytes().len())
        })
        .expect("fit utf8");

        assert_eq!(fitted, "aé");
    }
}
