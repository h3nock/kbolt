use crate::storage::ChunkRow;
use crate::Result;

pub(super) fn search_text_with_canonical_neighbors(
    document_text: &str,
    primary: &ChunkRow,
    doc_chunks: Option<&Vec<ChunkRow>>,
    neighbor_window: usize,
) -> Result<String> {
    if neighbor_window == 0 {
        return crate::storage::chunk_text_from_canonical(document_text, primary);
    }

    let Some(chunks) = doc_chunks else {
        return crate::storage::chunk_text_from_canonical(document_text, primary);
    };

    let window = neighbor_window.min(i32::MAX as usize) as i32;
    let min_seq = primary.seq.saturating_sub(window);
    let max_seq = primary.seq.saturating_add(window);
    let mut snippets = Vec::new();
    for chunk in chunks {
        if chunk.seq < min_seq || chunk.seq > max_seq {
            continue;
        }

        let snippet = crate::storage::chunk_text_from_canonical(document_text, chunk)?;
        if !snippet.is_empty() {
            snippets.push(snippet);
        }
    }

    if snippets.is_empty() {
        crate::storage::chunk_text_from_canonical(document_text, primary)
    } else {
        Ok(snippets.join("\n\n"))
    }
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
