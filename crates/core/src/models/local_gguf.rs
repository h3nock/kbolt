use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex};

use kbolt_types::KboltError;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::models::{Embedder, EmbeddingInputKind};
use crate::Result;

use super::llama_backend;

pub(super) struct LocalGgufEmbedder {
    model: Arc<LlamaModel>,
    batch_size: usize,
    n_threads: Option<u32>,
    n_threads_batch: Option<u32>,
    inference_lock: Mutex<()>,
}

impl Embedder for LocalGgufEmbedder {
    fn embed_batch(&self, _kind: EmbeddingInputKind, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        embed_with_local_gguf(self, texts)
    }
}

pub(super) fn build_local_gguf_embedder(
    model_path: &Path,
    batch_size: usize,
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
        n_threads,
        n_threads_batch,
        inference_lock: Mutex::new(()),
    })
}

const EMBED_CTX_SIZE: u32 = 512;

fn embed_with_local_gguf(embedder: &LocalGgufEmbedder, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let _guard = embedder
        .inference_lock
        .lock()
        .map_err(|_| KboltError::Inference("local gguf embedder mutex poisoned".to_string()))?;

    let backend = llama_backend();

    let mut ctx_params = LlamaContextParams::default()
        .with_embeddings(true)
        .with_n_ctx(NonZeroU32::new(EMBED_CTX_SIZE))
        .with_n_batch(EMBED_CTX_SIZE)
        .with_n_ubatch(EMBED_CTX_SIZE)
        .with_n_seq_max(1);
    if let Some(n_threads) = embedder.n_threads {
        ctx_params = ctx_params.with_n_threads(n_threads as i32);
    }
    if let Some(n_threads_batch) = embedder.n_threads_batch {
        ctx_params = ctx_params.with_n_threads_batch(n_threads_batch as i32);
    }

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
        let tokens = tokenize_embedding_text(&embedder.model, text)?;
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
        vectors.push(emb.to_vec());
    }

    Ok(vectors)
}

fn tokenize_embedding_text(model: &LlamaModel, text: &str) -> Result<Vec<LlamaToken>> {
    let input = if text.is_empty() { " " } else { text };
    let mut tokens = model
        .str_to_token(input, AddBos::Always)
        .map_err(|err| KboltError::Inference(format!("local gguf tokenization failed: {err}")))?;
    truncate_embedding_tokens(&mut tokens);

    if tokens.is_empty() {
        return Err(
            KboltError::Inference("local gguf tokenization returned 0 tokens".to_string()).into(),
        );
    }

    Ok(tokens)
}

fn truncate_embedding_tokens(tokens: &mut Vec<LlamaToken>) {
    if tokens.len() > EMBED_CTX_SIZE as usize {
        tokens.truncate(EMBED_CTX_SIZE as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::{truncate_embedding_tokens, EMBED_CTX_SIZE};
    use llama_cpp_2::token::LlamaToken;

    #[test]
    fn embedding_tokens_are_truncated_to_embed_context_size() {
        let mut tokens = (0..(EMBED_CTX_SIZE as usize + 25))
            .map(|token| LlamaToken::new(token as i32))
            .collect::<Vec<_>>();

        truncate_embedding_tokens(&mut tokens);

        assert_eq!(tokens.len(), EMBED_CTX_SIZE as usize);
    }
}
