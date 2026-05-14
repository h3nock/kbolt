use std::collections::HashMap;

use crate::storage::ChunkRow;
use crate::Result;

pub(super) fn search_text_with_loaded_canonical_neighbors(
    primary: &ChunkRow,
    doc_chunks: Option<&Vec<ChunkRow>>,
    neighbor_window: usize,
    text_by_chunk: &HashMap<i64, String>,
) -> Result<String> {
    if neighbor_window == 0 {
        return loaded_chunk_text(text_by_chunk, primary).map(ToString::to_string);
    }

    let Some(chunks) = doc_chunks else {
        return loaded_chunk_text(text_by_chunk, primary).map(ToString::to_string);
    };

    let window = neighbor_window.min(i32::MAX as usize) as i32;
    let min_seq = primary.seq.saturating_sub(window);
    let max_seq = primary.seq.saturating_add(window);
    let mut snippets = Vec::new();
    for chunk in chunks {
        if chunk.seq < min_seq || chunk.seq > max_seq {
            continue;
        }

        let snippet = loaded_chunk_text(text_by_chunk, chunk)?;
        if !snippet.is_empty() {
            snippets.push(snippet.to_string());
        }
    }

    if snippets.is_empty() {
        loaded_chunk_text(text_by_chunk, primary).map(ToString::to_string)
    } else {
        Ok(snippets.join("\n\n"))
    }
}

fn loaded_chunk_text<'a>(
    text_by_chunk: &'a HashMap<i64, String>,
    chunk: &ChunkRow,
) -> Result<&'a str> {
    text_by_chunk
        .get(&chunk.id)
        .map(String::as_str)
        .ok_or_else(|| {
            crate::error::CoreError::Internal(format!(
                "canonical text cache missing for chunk {}",
                chunk.id
            ))
            .into()
        })
}

pub(crate) fn retrieval_text_with_prefix(
    source_text: &str,
    title: Option<&str>,
    heading: Option<&str>,
    contextual_prefix: bool,
) -> String {
    if !contextual_prefix {
        return source_text.to_string();
    }

    let mut lines = Vec::new();
    if let Some(title) = title {
        let normalized_title = title.trim();
        if !normalized_title.is_empty() {
            lines.push(format!("title: {normalized_title}"));
        }
    }

    if let Some(raw_heading) = heading {
        let normalized_heading = raw_heading.trim();
        if !normalized_heading.is_empty() {
            lines.push(format!("heading: {normalized_heading}"));
        }
    }

    if lines.is_empty() {
        source_text.to_string()
    } else {
        format!("{}\n\n{}", lines.join("\n"), source_text)
    }
}
