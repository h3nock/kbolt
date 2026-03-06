use std::borrow::Cow;
use std::path::Path;
use std::sync::Mutex;

use kbolt_types::KboltError;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};

use crate::models::artifacts::{resolve_file_with_extension, resolve_tokenizer_file};
use crate::models::Embedder;
use crate::Result;

pub(super) struct LocalOnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl Embedder for LocalOnnxEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        embed_with_local_onnx(self, texts)
    }
}

pub(super) fn build_local_onnx_embedder(
    artifact_dir: &Path,
    onnx_file: Option<&str>,
    tokenizer_file: Option<&str>,
    max_length: usize,
) -> Result<LocalOnnxEmbedder> {
    let onnx_path =
        resolve_file_with_extension(artifact_dir, onnx_file, "onnx", "embeddings.onnx_file")?;
    let tokenizer_path =
        resolve_tokenizer_file(artifact_dir, tokenizer_file, "embeddings.tokenizer_file")?;

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
        KboltError::Inference(format!(
            "failed to load tokenizer {}: {err}",
            tokenizer_path.display()
        ))
    })?;
    let pad_id = tokenizer
        .token_to_id("[PAD]")
        .or_else(|| tokenizer.token_to_id("<pad>"))
        .unwrap_or(0);
    let pad_token = tokenizer
        .id_to_token(pad_id)
        .unwrap_or_else(|| "[PAD]".to_string());
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id,
        pad_token,
        ..PaddingParams::default()
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: Default::default(),
        }))
        .map_err(|err| KboltError::Inference(format!("failed to configure tokenizer: {err}")))?;

    let session = Session::builder()
        .map_err(|err| {
            KboltError::Inference(format!("failed to create ONNX session builder: {err}"))
        })?
        .commit_from_file(&onnx_path)
        .map_err(|err| {
            KboltError::Inference(format!(
                "failed to load ONNX model {}: {err}",
                onnx_path.display()
            ))
        })?;

    Ok(LocalOnnxEmbedder {
        session: Mutex::new(session),
        tokenizer,
    })
}

fn embed_with_local_onnx(embedder: &LocalOnnxEmbedder, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let inputs = texts.iter().map(|text| text.as_str()).collect::<Vec<_>>();
    let encodings = embedder
        .tokenizer
        .encode_batch(inputs, true)
        .map_err(|err| KboltError::Inference(format!("tokenization failed: {err}")))?;
    if encodings.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = encodings.len();
    let seq_len = encodings[0].get_ids().len();
    if seq_len == 0 {
        return Err(KboltError::Inference("tokenizer produced empty sequences".to_string()).into());
    }

    let mut input_ids = Vec::with_capacity(batch_size * seq_len);
    let mut attention_mask = Vec::with_capacity(batch_size * seq_len);
    let mut token_type_ids = Vec::with_capacity(batch_size * seq_len);
    for encoding in &encodings {
        if encoding.get_ids().len() != seq_len {
            return Err(KboltError::Inference(
                "tokenizer produced uneven sequence lengths".to_string(),
            )
            .into());
        }
        input_ids.extend(encoding.get_ids().iter().map(|value| *value as i64));
        attention_mask.extend(
            encoding
                .get_attention_mask()
                .iter()
                .map(|value| *value as i64),
        );
        if encoding.get_type_ids().is_empty() {
            token_type_ids.extend(std::iter::repeat_n(0_i64, seq_len));
        } else {
            token_type_ids.extend(encoding.get_type_ids().iter().map(|value| *value as i64));
        }
    }

    let ids_tensor = Tensor::<i64>::from_array(([batch_size, seq_len], input_ids.clone()))
        .map_err(|err| KboltError::Inference(format!("failed to build input_ids tensor: {err}")))?;
    let mask_tensor = Tensor::<i64>::from_array(([batch_size, seq_len], attention_mask.clone()))
        .map_err(|err| {
            KboltError::Inference(format!("failed to build attention_mask tensor: {err}"))
        })?;
    let type_tensor =
        Tensor::<i64>::from_array(([batch_size, seq_len], token_type_ids)).map_err(|err| {
            KboltError::Inference(format!("failed to build token_type_ids tensor: {err}"))
        })?;

    let mut session = embedder
        .session
        .lock()
        .map_err(|_| KboltError::Inference("local onnx session mutex poisoned".to_string()))?;
    let mut session_inputs = Vec::new();
    let single_input = session.inputs().len() == 1;
    let mut mapped_ids = false;
    for input in session.inputs() {
        let name = input.name();
        let lower = name.to_ascii_lowercase();
        if lower.contains("input_ids") || (!mapped_ids && single_input) {
            session_inputs.push((Cow::Owned(name.to_string()), ids_tensor.clone().into_dyn()));
            mapped_ids = true;
            continue;
        }
        if lower.contains("attention_mask") || lower.contains("mask") {
            session_inputs.push((Cow::Owned(name.to_string()), mask_tensor.clone().into_dyn()));
            continue;
        }
        if lower.contains("token_type_ids") || lower.contains("segment_ids") {
            session_inputs.push((Cow::Owned(name.to_string()), type_tensor.clone().into_dyn()));
            continue;
        }

        return Err(KboltError::Inference(format!(
            "unsupported ONNX input '{}' for local embedder",
            name
        ))
        .into());
    }

    if !mapped_ids {
        return Err(KboltError::Inference(
            "ONNX embedder inputs do not include input_ids".to_string(),
        )
        .into());
    }

    let outputs = session
        .run(session_inputs)
        .map_err(|err| KboltError::Inference(format!("onnx inference failed: {err}")))?;
    extract_embedding_vectors(outputs, batch_size, seq_len, &attention_mask)
}

fn extract_embedding_vectors(
    outputs: ort::session::SessionOutputs<'_>,
    batch_size: usize,
    seq_len: usize,
    attention_mask: &[i64],
) -> Result<Vec<Vec<f32>>> {
    for (_, output) in outputs.iter() {
        let Ok((shape, values)) = output.try_extract_tensor::<f32>() else {
            continue;
        };
        if let Some(vectors) =
            parse_embedding_tensor(shape, values, batch_size, seq_len, attention_mask)?
        {
            return Ok(vectors);
        }
    }

    Err(
        KboltError::Inference("onnx output did not contain a usable embedding tensor".to_string())
            .into(),
    )
}

fn parse_embedding_tensor(
    shape: &ort::value::Shape,
    values: &[f32],
    batch_size: usize,
    seq_len: usize,
    attention_mask: &[i64],
) -> Result<Option<Vec<Vec<f32>>>> {
    let dims = shape.iter().copied().collect::<Vec<_>>();
    if dims.len() == 2 {
        let batch = usize::try_from(dims[0]).ok();
        let hidden = usize::try_from(dims[1]).ok();
        let (Some(batch), Some(hidden)) = (batch, hidden) else {
            return Ok(None);
        };
        if batch != batch_size || hidden == 0 || values.len() != batch.saturating_mul(hidden) {
            return Ok(None);
        }

        let mut vectors = Vec::with_capacity(batch);
        for row in 0..batch {
            let start = row.saturating_mul(hidden);
            let end = start.saturating_add(hidden);
            vectors.push(values[start..end].to_vec());
        }
        return Ok(Some(vectors));
    }

    if dims.len() == 3 {
        let batch = usize::try_from(dims[0]).ok();
        let tokens = usize::try_from(dims[1]).ok();
        let hidden = usize::try_from(dims[2]).ok();
        let (Some(batch), Some(tokens), Some(hidden)) = (batch, tokens, hidden) else {
            return Ok(None);
        };
        if batch != batch_size
            || hidden == 0
            || values.len() != batch.saturating_mul(tokens).saturating_mul(hidden)
        {
            return Ok(None);
        }

        let mask_tokens = if tokens == seq_len {
            attention_mask
        } else {
            return Ok(None);
        };

        let mut vectors = vec![vec![0.0_f32; hidden]; batch];
        for (batch_index, vector) in vectors.iter_mut().enumerate() {
            let mut weight_sum = 0.0_f32;
            for token_index in 0..tokens {
                let mask_index = batch_index
                    .saturating_mul(tokens)
                    .saturating_add(token_index);
                let weight = if mask_tokens.get(mask_index).copied().unwrap_or(1) > 0 {
                    1.0_f32
                } else {
                    0.0_f32
                };
                if weight == 0.0 {
                    continue;
                }
                weight_sum += weight;

                let value_offset = batch_index
                    .saturating_mul(tokens)
                    .saturating_add(token_index)
                    .saturating_mul(hidden);
                for hidden_index in 0..hidden {
                    vector[hidden_index] += values[value_offset + hidden_index];
                }
            }

            if weight_sum == 0.0 {
                return Err(KboltError::Inference(
                    "attention mask produced zero-weight embedding row".to_string(),
                )
                .into());
            }
            for value in vector {
                *value /= weight_sum;
            }
        }

        return Ok(Some(vectors));
    }

    Ok(None)
}
