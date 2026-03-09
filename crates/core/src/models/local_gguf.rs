use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex};

use kbolt_types::KboltError;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::models::Embedder;
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
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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

fn embed_with_local_gguf(embedder: &LocalGgufEmbedder, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let _guard = embedder
        .inference_lock
        .lock()
        .map_err(|_| KboltError::Inference("local gguf embedder mutex poisoned".to_string()))?;

    let backend = llama_backend();
    let mut vectors = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(embedder.batch_size.max(1)) {
        let tokenized = chunk
            .iter()
            .map(|text| tokenize_embedding_text(&embedder.model, text))
            .collect::<Result<Vec<_>>>()?;
        let mut chunk_vectors = embed_chunk(embedder, backend, &tokenized)?;
        vectors.append(&mut chunk_vectors);
    }

    if vectors.len() != texts.len() {
        return Err(KboltError::Inference(format!(
            "local gguf embedder returned {} vectors for {} texts",
            vectors.len(),
            texts.len()
        ))
        .into());
    }
    if let Some((index, _)) = vectors
        .iter()
        .enumerate()
        .find(|(_, vector)| vector.is_empty())
    {
        return Err(KboltError::Inference(format!(
            "local gguf embedder returned an empty vector at index {index}"
        ))
        .into());
    }

    Ok(vectors)
}

fn tokenize_embedding_text(model: &LlamaModel, text: &str) -> Result<Vec<LlamaToken>> {
    let input = if text.is_empty() { " " } else { text };
    let tokens = model
        .str_to_token(input, AddBos::Always)
        .map_err(|err| KboltError::Inference(format!("local gguf tokenization failed: {err}")))?;

    if tokens.is_empty() {
        return Err(
            KboltError::Inference("local gguf tokenization returned 0 tokens".to_string()).into(),
        );
    }

    Ok(tokens)
}

fn embed_chunk(
    embedder: &LocalGgufEmbedder,
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    tokenized: &[Vec<LlamaToken>],
) -> Result<Vec<Vec<f32>>> {
    let max_seq_tokens = tokenized.iter().map(Vec::len).max().unwrap_or(0) as u32;
    let effective_ctx = max_seq_tokens.max(512);

    let mut ctx_params = LlamaContextParams::default()
        .with_embeddings(true)
        .with_n_ctx(NonZeroU32::new(effective_ctx))
        .with_n_batch(effective_ctx)
        .with_n_ubatch(effective_ctx)
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

    let mut vectors = Vec::with_capacity(tokenized.len());
    for tokens in tokenized {
        ctx.clear_kv_cache();

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch.add_sequence(tokens, 0, false).map_err(|err| {
            KboltError::Inference(format!("local gguf batch creation failed: {err}"))
        })?;

        ctx.decode(&mut batch).map_err(|err| {
            KboltError::Inference(format!("local gguf embedding decode failed: {err}"))
        })?;

        let emb = ctx.embeddings_seq_ith(0).map_err(|err| {
            KboltError::Inference(format!("failed to read local gguf embedding: {err}"))
        })?;
        vectors.push(emb.to_vec());
    }

    Ok(vectors)
}
