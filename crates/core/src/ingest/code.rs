use std::collections::HashMap;
use std::path::Path;

use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument, Extractor};
use crate::Result;

pub struct CodeExtractor;

impl Extractor for CodeExtractor {
    fn supports(&self) -> &[&str] {
        &[
            "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "kt", "c", "cpp", "cc", "h",
            "hpp", "cs", "rb", "php", "swift",
        ]
    }

    fn profile_key(&self) -> &'static str {
        "code"
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> Result<ExtractedDocument> {
        if let Err(err) = std::str::from_utf8(bytes) {
            return Err(
                kbolt_types::KboltError::InvalidInput(format!("non-utf8 code input: {err}")).into(),
            );
        }

        let language = path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.trim().trim_start_matches('.').to_ascii_lowercase())
            .unwrap_or_default();

        let mut attrs = HashMap::new();
        if !language.is_empty() {
            attrs.insert("language".to_string(), language);
        }

        let block = ExtractedBlock {
            text: String::from_utf8_lossy(bytes).to_string(),
            offset: 0,
            length: bytes.len(),
            kind: BlockKind::CodeFence,
            heading_path: Vec::new(),
            attrs,
        };

        Ok(ExtractedDocument {
            blocks: vec![block],
            metadata: HashMap::new(),
            title: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::ingest::code::CodeExtractor;
    use crate::ingest::extract::{BlockKind, Extractor};

    #[test]
    fn extracts_code_block_with_full_span_and_language_attr() {
        let extractor = CodeExtractor;
        let source = b"fn alpha() {}\nfn beta() {}\n";
        let doc = extractor
            .extract(Path::new("src/lib.rs"), source)
            .expect("extract code");

        assert_eq!(extractor.profile_key(), "code");
        assert_eq!(doc.blocks.len(), 1);
        assert_eq!(doc.blocks[0].kind, BlockKind::CodeFence);
        assert_eq!(doc.blocks[0].offset, 0);
        assert_eq!(doc.blocks[0].length, source.len());
        assert_eq!(doc.blocks[0].attrs.get("language").map(String::as_str), Some("rs"));
    }

    #[test]
    fn rejects_non_utf8_code_bytes() {
        let extractor = CodeExtractor;
        let err = extractor
            .extract(Path::new("src/lib.rs"), &[0xff, 0xfe, 0xfd])
            .expect_err("invalid utf8 should fail");
        assert!(err.to_string().contains("non-utf8 code input"));
    }
}
