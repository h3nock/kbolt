use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalDocument {
    pub text: String,
    pub document: ExtractedDocument,
}

pub fn build_canonical_document(document: &ExtractedDocument) -> CanonicalDocument {
    let mut text = String::new();
    let mut blocks = Vec::with_capacity(document.blocks.len());
    let mut active_table_header: Option<String> = None;

    for block in &document.blocks {
        let canonical_block_text = canonical_block_text(block, active_table_header.as_deref());

        match block.kind {
            BlockKind::TableHeader => {
                active_table_header = Some(block.text.clone());
            }
            BlockKind::TableRow => {}
            _ => {
                active_table_header = None;
            }
        }

        if !text.is_empty() {
            text.push_str("\n\n");
        }

        let offset = text.len();
        text.push_str(&canonical_block_text);

        let mut canonical_block = block.clone();
        canonical_block.text = canonical_block_text;
        canonical_block.offset = offset;
        canonical_block.length = canonical_block.text.len();
        blocks.push(canonical_block);
    }

    CanonicalDocument {
        text,
        document: ExtractedDocument {
            blocks,
            metadata: document.metadata.clone(),
            title: document.title.clone(),
        },
    }
}

fn canonical_block_text(block: &ExtractedBlock, active_table_header: Option<&str>) -> String {
    if block.kind != BlockKind::TableRow {
        return block.text.clone();
    }

    let Some(header) = active_table_header.filter(|header| !header.trim().is_empty()) else {
        return block.text.clone();
    };

    format!("{header}\n{}", block.text)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::config::ChunkPolicy;
    use crate::ingest::canonical::build_canonical_document;
    use crate::ingest::chunk::{chunk_canonical_document, FinalChunkKind};
    use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument};

    fn block(kind: BlockKind, text: &str, offset: usize) -> ExtractedBlock {
        ExtractedBlock {
            text: text.to_string(),
            offset,
            length: text.len(),
            kind,
            heading_path: Vec::new(),
            attrs: HashMap::new(),
        }
    }

    fn chunk_policy() -> ChunkPolicy {
        ChunkPolicy {
            target_tokens: 8,
            soft_max_tokens: 8,
            hard_max_tokens: 8,
            boundary_overlap_tokens: 0,
            neighbor_window: 1,
            contextual_prefix: true,
        }
    }

    #[test]
    fn builds_canonical_text_with_block_spans() {
        let document = ExtractedDocument {
            blocks: vec![
                block(BlockKind::Heading, "# Intro", 10),
                block(BlockKind::Paragraph, "alpha beta", 25),
            ],
            metadata: HashMap::new(),
            title: Some("Intro".to_string()),
        };

        let canonical = build_canonical_document(&document);

        assert_eq!(canonical.text, "# Intro\n\nalpha beta");
        assert_eq!(canonical.document.blocks[0].offset, 0);
        assert_eq!(canonical.document.blocks[0].length, 7);
        assert_eq!(canonical.document.blocks[1].offset, 9);
        assert_eq!(canonical.document.blocks[1].length, 10);
        assert_eq!(canonical.document.title.as_deref(), Some("Intro"));
    }

    #[test]
    fn materializes_table_header_into_canonical_rows() {
        let document = ExtractedDocument {
            blocks: vec![
                block(BlockKind::TableHeader, "h1 h2 h3 h4", 0),
                block(BlockKind::TableRow, "r1a r1b r1c r1d", 20),
                block(BlockKind::TableRow, "r2a r2b r2c r2d", 40),
            ],
            metadata: HashMap::new(),
            title: None,
        };

        let canonical = build_canonical_document(&document);

        assert_eq!(
            canonical.text,
            "h1 h2 h3 h4\n\nh1 h2 h3 h4\nr1a r1b r1c r1d\n\nh1 h2 h3 h4\nr2a r2b r2c r2d"
        );
        assert_eq!(
            &canonical.text[canonical.document.blocks[1].offset
                ..canonical.document.blocks[1].offset + canonical.document.blocks[1].length],
            "h1 h2 h3 h4\nr1a r1b r1c r1d"
        );
    }

    #[test]
    fn canonical_chunk_spans_slice_chunk_text() {
        let document = ExtractedDocument {
            blocks: vec![
                block(BlockKind::TableHeader, "h1 h2 h3 h4", 0),
                block(BlockKind::TableRow, "r1a r1b r1c r1d", 20),
                block(BlockKind::TableRow, "r2a r2b r2c r2d", 40),
            ],
            metadata: HashMap::new(),
            title: None,
        };
        let canonical = build_canonical_document(&document);

        let chunks = chunk_canonical_document(&canonical.document, &chunk_policy());

        assert_eq!(chunks.len(), 3);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.kind == FinalChunkKind::Table));
        for chunk in chunks {
            let slice = &canonical.text[chunk.offset..chunk.offset + chunk.length];
            assert_eq!(chunk.text, slice);
        }
    }

    #[test]
    fn canonical_spans_stay_on_utf8_boundaries() {
        let document = ExtractedDocument {
            blocks: vec![
                block(BlockKind::Paragraph, "alpha café", 0),
                block(BlockKind::Paragraph, "東京 beta", 20),
            ],
            metadata: HashMap::new(),
            title: None,
        };

        let canonical = build_canonical_document(&document);

        for block in &canonical.document.blocks {
            assert!(canonical.text.is_char_boundary(block.offset));
            assert!(canonical.text.is_char_boundary(block.offset + block.length));
            assert_eq!(
                &canonical.text[block.offset..block.offset + block.length],
                block.text
            );
        }
    }
}
