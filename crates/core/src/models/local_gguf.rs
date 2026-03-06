use std::path::Path;
use std::sync::{Arc, Mutex};

use kbolt_types::KboltError;
use llama_cpp::{EmbeddingsParams, LlamaModel, LlamaParams};

use crate::models::Embedder;
use crate::Result;

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
    let model = LlamaModel::load_from_file(model_path, LlamaParams::default()).map_err(|err| {
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

    let mut vectors = Vec::with_capacity(texts.len());
    for batch in texts.chunks(embedder.batch_size.max(1)) {
        let inputs = batch.iter().map(|text| text.as_bytes()).collect::<Vec<_>>();
        let mut params = EmbeddingsParams::default();
        if let Some(n_threads) = embedder.n_threads {
            params.n_threads = n_threads;
        }
        if let Some(n_threads_batch) = embedder.n_threads_batch {
            params.n_threads_batch = n_threads_batch;
        }

        let mut batch_vectors = embedder
            .model
            .embeddings(&inputs, params)
            .map_err(|err| KboltError::Inference(format!("local gguf embedding failed: {err}")))?;
        vectors.append(&mut batch_vectors);
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
