use std::collections::HashMap;
use std::path::Path;

use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument, Extractor};
use crate::Result;

pub struct PlaintextExtractor;

impl Extractor for PlaintextExtractor {
    fn supports(&self) -> &[&str] {
        &["txt", "text", "log"]
    }

    fn profile_key(&self) -> &'static str {
        "txt"
    }

    fn supports_path(&self, _path: &Path) -> bool {
        true
    }

    fn extract(&self, _path: &Path, bytes: &[u8]) -> Result<ExtractedDocument> {
        if let Err(err) = std::str::from_utf8(bytes) {
            return Err(
                kbolt_types::KboltError::InvalidInput(format!("non-utf8 plaintext input: {err}"))
                    .into(),
            );
        }

        let mut blocks = Vec::new();
        for (offset, end) in paragraph_ranges(bytes) {
            let text = String::from_utf8_lossy(&bytes[offset..end]).to_string();
            blocks.push(ExtractedBlock {
                text,
                offset,
                length: end.saturating_sub(offset),
                kind: BlockKind::Paragraph,
                heading_path: Vec::new(),
                attrs: HashMap::new(),
            });
        }

        Ok(ExtractedDocument {
            blocks,
            metadata: HashMap::new(),
            title: None,
        })
    }
}

fn paragraph_ranges(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut paragraph_start: Option<usize> = None;
    let mut line_start = 0usize;

    while line_start < bytes.len() {
        let line_end = next_line_end(bytes, line_start);
        let content_end = trim_line_ending(bytes, line_start, line_end);
        let is_blank = is_blank_line(bytes, line_start, content_end);

        match (paragraph_start, is_blank) {
            (None, false) => {
                paragraph_start = Some(line_start);
            }
            (Some(start), true) => {
                let end = trim_trailing_newlines(bytes, line_start);
                if end > start {
                    ranges.push((start, end));
                }
                paragraph_start = None;
            }
            _ => {}
        }

        line_start = line_end;
    }

    if let Some(start) = paragraph_start {
        let end = trim_trailing_newlines(bytes, bytes.len());
        if end > start {
            ranges.push((start, end));
        }
    }

    ranges
}

fn next_line_end(bytes: &[u8], start: usize) -> usize {
    let mut index = start;
    while index < bytes.len() {
        if bytes[index] == b'\n' {
            return index + 1;
        }
        index += 1;
    }
    bytes.len()
}

fn trim_line_ending(bytes: &[u8], start: usize, end: usize) -> usize {
    let mut content_end = end;
    while content_end > start && matches!(bytes[content_end - 1], b'\n' | b'\r') {
        content_end -= 1;
    }
    content_end
}

fn is_blank_line(bytes: &[u8], start: usize, end: usize) -> bool {
    bytes[start..end]
        .iter()
        .all(|byte| matches!(byte, b' ' | b'\t'))
}

fn trim_trailing_newlines(bytes: &[u8], end: usize) -> usize {
    let mut result = end;
    while result > 0 && matches!(bytes[result - 1], b'\n' | b'\r') {
        result -= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::ingest::extract::Extractor;
    use crate::ingest::plaintext::PlaintextExtractor;

    #[test]
    fn extracts_single_paragraph_with_exact_span() {
        let extractor = PlaintextExtractor;
        let doc = extractor
            .extract(Path::new("notes/readme.txt"), b"alpha beta")
            .expect("extract plaintext");

        assert_eq!(doc.blocks.len(), 1);
        assert_eq!(doc.blocks[0].offset, 0);
        assert_eq!(doc.blocks[0].length, 10);
        assert_eq!(doc.blocks[0].text, "alpha beta");
    }

    #[test]
    fn splits_paragraphs_on_blank_lines_with_spans() {
        let extractor = PlaintextExtractor;
        let input = b"first line\nsecond line\n\nthird line\n\n  \nlast line\n";
        let doc = extractor
            .extract(Path::new("notes/readme.txt"), input)
            .expect("extract plaintext");

        assert_eq!(doc.blocks.len(), 3);
        assert_eq!(doc.blocks[0].text, "first line\nsecond line");
        assert_eq!(doc.blocks[0].offset, 0);
        assert_eq!(doc.blocks[0].length, 22);

        assert_eq!(doc.blocks[1].text, "third line");
        assert_eq!(doc.blocks[1].offset, 24);
        assert_eq!(doc.blocks[1].length, 10);

        assert_eq!(doc.blocks[2].text, "last line");
        assert_eq!(doc.blocks[2].offset, 39);
        assert_eq!(doc.blocks[2].length, 9);
    }

    #[test]
    fn supports_path_acts_as_generic_text_fallback() {
        let extractor = PlaintextExtractor;
        assert_eq!(extractor.profile_key(), "txt");
        assert!(extractor.supports_path(Path::new("docs/readme.md")));
        assert!(extractor.supports_path(Path::new("src/main.rs")));
    }

    #[test]
    fn rejects_non_utf8_bytes() {
        let extractor = PlaintextExtractor;
        let err = extractor
            .extract(Path::new("notes/data.bin"), &[0xff, 0xfe, 0xfd])
            .expect_err("invalid utf8 should fail");
        assert!(err.to_string().contains("non-utf8 plaintext input"));
    }
}
