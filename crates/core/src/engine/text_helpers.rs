use crate::storage::ChunkRow;

pub(super) fn chunk_text_from_bytes(bytes: &[u8], offset: usize, length: usize) -> String {
    let start = offset.min(bytes.len());
    let end = offset.saturating_add(length).min(bytes.len());
    String::from_utf8_lossy(&bytes[start..end]).into_owned()
}

pub(super) fn search_text_with_neighbors(
    bytes: &[u8],
    primary: &ChunkRow,
    doc_chunks: Option<&Vec<ChunkRow>>,
    neighbor_window: usize,
) -> String {
    if neighbor_window == 0 {
        return chunk_text_from_bytes(bytes, primary.offset, primary.length);
    }

    let Some(chunks) = doc_chunks else {
        return chunk_text_from_bytes(bytes, primary.offset, primary.length);
    };

    let window = neighbor_window.min(i32::MAX as usize) as i32;
    let min_seq = primary.seq.saturating_sub(window);
    let max_seq = primary.seq.saturating_add(window);
    let mut snippets = Vec::new();
    for chunk in chunks {
        if chunk.seq < min_seq || chunk.seq > max_seq {
            continue;
        }

        let snippet = chunk_text_from_bytes(bytes, chunk.offset, chunk.length);
        if !snippet.is_empty() {
            snippets.push(snippet);
        }
    }

    if snippets.is_empty() {
        chunk_text_from_bytes(bytes, primary.offset, primary.length)
    } else {
        snippets.join("\n\n")
    }
}

pub(crate) fn retrieval_text_with_prefix(
    source_text: &str,
    title: &str,
    heading: Option<&str>,
    contextual_prefix: bool,
) -> String {
    if !contextual_prefix {
        return source_text.to_string();
    }

    let mut lines = Vec::new();
    let normalized_title = title.trim();
    if !normalized_title.is_empty() {
        lines.push(format!("title: {normalized_title}"));
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
