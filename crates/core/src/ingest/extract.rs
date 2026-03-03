use std::collections::HashMap;
use std::path::Path;

use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedDocument {
    pub blocks: Vec<ExtractedBlock>,
    pub metadata: HashMap<String, String>,
    pub title: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedBlock {
    pub text: String,
    pub offset: usize,
    pub length: usize,
    pub kind: BlockKind,
    pub heading_path: Vec<String>,
    pub attrs: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockKind {
    Heading,
    Paragraph,
    ListItem,
    BlockQuote,
    CodeFence,
    TableHeader,
    TableRow,
    HtmlBlock,
}

pub trait Extractor: Send + Sync {
    fn supports(&self) -> &[&str];

    fn supports_path(&self, _path: &Path) -> bool {
        false
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> Result<ExtractedDocument>;
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;

    use super::{BlockKind, ExtractedBlock, ExtractedDocument, Extractor};
    use crate::Result;

    struct DummyExtractor;

    impl Extractor for DummyExtractor {
        fn supports(&self) -> &[&str] {
            &["txt"]
        }

        fn extract(&self, _path: &Path, bytes: &[u8]) -> Result<ExtractedDocument> {
            Ok(ExtractedDocument {
                blocks: vec![ExtractedBlock {
                    text: String::from_utf8_lossy(bytes).to_string(),
                    offset: 0,
                    length: bytes.len(),
                    kind: BlockKind::Paragraph,
                    heading_path: vec![],
                    attrs: HashMap::new(),
                }],
                metadata: HashMap::new(),
                title: None,
            })
        }
    }

    #[test]
    fn extractor_default_supports_path_is_false() {
        let extractor = DummyExtractor;
        assert!(!extractor.supports_path(Path::new("notes/readme.txt")));
    }

    #[test]
    fn extracted_document_tracks_blocks_and_spans() {
        let extractor = DummyExtractor;
        let document = extractor
            .extract(Path::new("notes/readme.txt"), b"hello world")
            .expect("extract document");
        assert_eq!(document.blocks.len(), 1);
        assert_eq!(document.blocks[0].offset, 0);
        assert_eq!(document.blocks[0].length, 11);
        assert_eq!(document.blocks[0].kind, BlockKind::Paragraph);
    }
}
