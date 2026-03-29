use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex};

use kbolt_types::KboltError;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::config::LlamaFlashAttentionMode;
use crate::models::{Embedder, EmbeddingInputKind};
use crate::Result;

use super::{llama_backend, llama_flash_attention_policy};

pub(super) struct LocalGgufEmbedder {
    model: Arc<LlamaModel>,
    batch_size: usize,
    flash_attention: LlamaFlashAttentionMode,
    n_threads: Option<u32>,
    n_threads_batch: Option<u32>,
    inference_lock: Mutex<()>,
}

impl Embedder for LocalGgufEmbedder {
    fn embed_batch(&self, kind: EmbeddingInputKind, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        embed_with_local_gguf(self, kind, texts)
    }
}

pub(super) fn build_local_gguf_embedder(
    model_path: &Path,
    batch_size: usize,
    flash_attention: LlamaFlashAttentionMode,
    n_threads: Option<u32>,
    n_threads_batch: Option<u32>,
) -> Result<LocalGgufEmbedder> {
    let backend = llama_backend();
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(backend, model_path, &model_params).map_err(|err| {
        KboltError::Inference(format!(
            "failed to load local gguf embedder model {}: {err}",
            model_path.display()
        ))
    })?;

    Ok(LocalGgufEmbedder {
        model: Arc::new(model),
        batch_size,
        flash_attention,
        n_threads,
        n_threads_batch,
        inference_lock: Mutex::new(()),
    })
}

const EMBED_CTX_SIZE: u32 = 512;
const EMBEDDING_GEMMA_QUERY_PREFIX: &str = "task: search result | query: ";
const EMBEDDING_GEMMA_DOCUMENT_PREFIX: &str = "title: none | text: ";

fn embed_with_local_gguf(
    embedder: &LocalGgufEmbedder,
    kind: EmbeddingInputKind,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let _guard = embedder
        .inference_lock
        .lock()
        .map_err(|_| KboltError::Inference("local gguf embedder mutex poisoned".to_string()))?;

    let backend = llama_backend();

    let ctx_params = embed_context_params(
        embedder.n_threads,
        embedder.n_threads_batch,
        embedder.flash_attention,
    );

    let mut ctx = embedder
        .model
        .new_context(backend, ctx_params)
        .map_err(|err| {
            KboltError::Inference(format!(
                "failed to create local gguf embedding context: {err}"
            ))
        })?;

    let mut vectors = Vec::with_capacity(texts.len());
    for text in texts {
        let tokens = tokenize_embedding_text(&embedder.model, kind, text)?;
        ctx.clear_kv_cache();

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch.add_sequence(&tokens, 0, false).map_err(|err| {
            KboltError::Inference(format!("local gguf batch creation failed: {err}"))
        })?;

        ctx.decode(&mut batch).map_err(|err| {
            KboltError::Inference(format!("local gguf embedding decode failed: {err}"))
        })?;

        let emb = ctx.embeddings_seq_ith(0).map_err(|err| {
            KboltError::Inference(format!("failed to read local gguf embedding: {err}"))
        })?;
        if emb.is_empty() {
            return Err(KboltError::Inference(
                "local gguf embedder returned an empty vector".to_string(),
            )
            .into());
        }
        vectors.push(normalize_embedding_vector(emb.to_vec()));
    }

    Ok(vectors)
}

fn embed_context_params(
    n_threads: Option<u32>,
    n_threads_batch: Option<u32>,
    flash_attention: LlamaFlashAttentionMode,
) -> LlamaContextParams {
    let mut ctx_params = LlamaContextParams::default()
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Mean)
        .with_n_ctx(NonZeroU32::new(EMBED_CTX_SIZE))
        .with_n_batch(EMBED_CTX_SIZE)
        .with_n_ubatch(EMBED_CTX_SIZE)
        .with_n_seq_max(1)
        .with_flash_attention_policy(llama_flash_attention_policy(flash_attention));
    if let Some(n_threads) = n_threads {
        ctx_params = ctx_params.with_n_threads(n_threads as i32);
    }
    if let Some(n_threads_batch) = n_threads_batch {
        ctx_params = ctx_params.with_n_threads_batch(n_threads_batch as i32);
    }
    ctx_params
}

fn tokenize_embedding_text(
    model: &LlamaModel,
    kind: EmbeddingInputKind,
    text: &str,
) -> Result<Vec<LlamaToken>> {
    let input = format_embedding_text(kind, text);
    let mut tokens = model
        .str_to_token(&input, AddBos::Always)
        .map_err(|err| KboltError::Inference(format!("local gguf tokenization failed: {err}")))?;
    truncate_embedding_tokens(&mut tokens);

    if tokens.is_empty() {
        return Err(
            KboltError::Inference("local gguf tokenization returned 0 tokens".to_string()).into(),
        );
    }

    Ok(tokens)
}

fn format_embedding_text(kind: EmbeddingInputKind, text: &str) -> String {
    let text = if text.is_empty() { " " } else { text };
    match kind {
        EmbeddingInputKind::Query => format!("{EMBEDDING_GEMMA_QUERY_PREFIX}{text}"),
        EmbeddingInputKind::Document => format!("{EMBEDDING_GEMMA_DOCUMENT_PREFIX}{text}"),
    }
}

fn normalize_embedding_vector(mut vector: Vec<f32>) -> Vec<f32> {
    let mut sum = 0.0_f64;
    for value in &vector {
        let value = f64::from(*value);
        sum += value * value;
    }

    let norm = sum.sqrt();
    let scale = if norm > 0.0 {
        1.0_f32 / norm as f32
    } else {
        0.0_f32
    };
    for value in &mut vector {
        *value *= scale;
    }

    vector
}

fn truncate_embedding_tokens(tokens: &mut Vec<LlamaToken>) {
    if tokens.len() > EMBED_CTX_SIZE as usize {
        tokens.truncate(EMBED_CTX_SIZE as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        embed_context_params, format_embedding_text, normalize_embedding_vector,
        truncate_embedding_tokens, EMBEDDING_GEMMA_DOCUMENT_PREFIX, EMBEDDING_GEMMA_QUERY_PREFIX,
        EMBED_CTX_SIZE,
    };
    use crate::config::LlamaFlashAttentionMode;
    use crate::models::EmbeddingInputKind;
    use llama_cpp_2::context::params::LlamaPoolingType;
    use llama_cpp_2::token::LlamaToken;

    const LLAMA_FLASH_ATTN_TYPE_AUTO: i32 = -1;
    const LLAMA_FLASH_ATTN_TYPE_DISABLED: i32 = 0;
    const LLAMA_FLASH_ATTN_TYPE_ENABLED: i32 = 1;

    #[test]
    fn embedding_tokens_are_truncated_to_embed_context_size() {
        let mut tokens = (0..(EMBED_CTX_SIZE as usize + 25))
            .map(|token| LlamaToken::new(token as i32))
            .collect::<Vec<_>>();

        truncate_embedding_tokens(&mut tokens);

        assert_eq!(tokens.len(), EMBED_CTX_SIZE as usize);
    }

    #[test]
    fn formats_query_inputs_with_embedding_gemma_prefix() {
        let formatted = format_embedding_text(EmbeddingInputKind::Query, "find setup docs");

        assert_eq!(
            formatted,
            format!("{EMBEDDING_GEMMA_QUERY_PREFIX}find setup docs")
        );
    }

    #[test]
    fn formats_document_inputs_with_embedding_gemma_prefix() {
        let formatted = format_embedding_text(EmbeddingInputKind::Document, "setup guide");

        assert_eq!(
            formatted,
            format!("{EMBEDDING_GEMMA_DOCUMENT_PREFIX}setup guide")
        );
    }

    #[test]
    fn gguf_embedding_context_uses_mean_pooling() {
        let params = embed_context_params(Some(4), Some(2), LlamaFlashAttentionMode::Disabled);

        assert_eq!(params.pooling_type(), LlamaPoolingType::Mean);
    }

    #[test]
    fn gguf_embedding_context_params_disable_flash_attention_by_default() {
        let params = embed_context_params(Some(4), Some(2), LlamaFlashAttentionMode::Disabled);

        assert_eq!(
            params.flash_attention_policy(),
            LLAMA_FLASH_ATTN_TYPE_DISABLED
        );
    }

    #[test]
    fn gguf_embedding_context_params_enable_flash_attention_when_requested() {
        let params = embed_context_params(Some(4), Some(2), LlamaFlashAttentionMode::Enabled);

        assert_eq!(
            params.flash_attention_policy(),
            LLAMA_FLASH_ATTN_TYPE_ENABLED
        );
    }

    #[test]
    fn gguf_embedding_context_params_use_auto_flash_attention_when_requested() {
        let params = embed_context_params(Some(4), Some(2), LlamaFlashAttentionMode::Auto);

        assert_eq!(params.flash_attention_policy(), LLAMA_FLASH_ATTN_TYPE_AUTO);
    }

    #[test]
    fn normalizes_embedding_vectors_with_l2_norm() {
        let vector = normalize_embedding_vector(vec![3.0, 4.0]);

        assert!((vector[0] - 0.6).abs() < 1.0e-6);
        assert!((vector[1] - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn leaves_zero_vectors_at_zero_when_normalizing() {
        let vector = normalize_embedding_vector(vec![0.0, 0.0, 0.0]);

        assert_eq!(vector, vec![0.0, 0.0, 0.0]);
    }
}
