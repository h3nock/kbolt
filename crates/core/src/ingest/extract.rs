use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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

    fn profile_key(&self) -> &'static str {
        "txt"
    }

    fn supports_path(&self, _path: &Path) -> bool {
        false
    }

    fn extract(&self, path: &Path, bytes: &[u8]) -> Result<ExtractedDocument>;
}

#[derive(Default)]
pub struct ExtractorRegistry {
    by_extension: HashMap<String, Arc<dyn Extractor>>,
    fallback_extractors: Vec<Arc<dyn Extractor>>,
}

impl ExtractorRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, extractor: Arc<dyn Extractor>) {
        for extension in extractor.supports() {
            self.by_extension
                .insert(normalize_extension_key(extension), Arc::clone(&extractor));
        }
        self.fallback_extractors.push(extractor);
    }

    pub fn resolve_for_path(&self, path: &Path) -> Option<Arc<dyn Extractor>> {
        if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
            let key = normalize_extension_key(extension);
            if let Some(extractor) = self.by_extension.get(&key) {
                return Some(Arc::clone(extractor));
            }
        }

        for extractor in &self.fallback_extractors {
            if extractor.supports_path(path) {
                return Some(Arc::clone(extractor));
            }
        }

        None
    }
}

pub fn default_registry() -> ExtractorRegistry {
    let mut registry = ExtractorRegistry::new();
    registry.register(Arc::new(crate::ingest::markdown::MarkdownExtractor));
    registry.register(Arc::new(crate::ingest::code::CodeExtractor));
    registry.register(Arc::new(crate::ingest::plaintext::PlaintextExtractor));
    registry
}

fn normalize_extension_key(raw: &str) -> String {
    raw.trim().trim_start_matches('.').to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;

    use super::{
        default_registry, normalize_extension_key, BlockKind, ExtractedBlock, ExtractedDocument,
        Extractor, ExtractorRegistry,
    };
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

    struct PathFallbackExtractor;

    impl Extractor for PathFallbackExtractor {
        fn supports(&self) -> &[&str] {
            &[]
        }

        fn supports_path(&self, path: &Path) -> bool {
            path.file_name().and_then(|value| value.to_str()) == Some("LICENSE")
        }

        fn extract(&self, _path: &Path, _bytes: &[u8]) -> Result<ExtractedDocument> {
            Ok(ExtractedDocument {
                blocks: vec![],
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
    fn extractor_default_profile_key_is_txt() {
        let extractor = DummyExtractor;
        assert_eq!(extractor.profile_key(), "txt");
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

    #[test]
    fn extension_key_normalization_trims_prefix_and_case() {
        assert_eq!(normalize_extension_key("MD"), "md");
        assert_eq!(normalize_extension_key(".Markdown"), "markdown");
        assert_eq!(normalize_extension_key(" rs "), "rs");
    }

    #[test]
    fn registry_resolves_by_extension_before_fallbacks() {
        let mut registry = ExtractorRegistry::new();
        registry.register(Arc::new(DummyExtractor));
        registry.register(Arc::new(PathFallbackExtractor));

        let resolved = registry
            .resolve_for_path(Path::new("notes/readme.TXT"))
            .expect("resolve txt extractor");
        assert_eq!(resolved.supports(), ["txt"]);
    }

    #[test]
    fn registry_uses_supports_path_as_fallback() {
        let mut registry = ExtractorRegistry::new();
        registry.register(Arc::new(PathFallbackExtractor));

        let resolved = registry
            .resolve_for_path(Path::new("docs/LICENSE"))
            .expect("resolve fallback extractor");
        assert!(resolved.supports().is_empty());
    }

    #[test]
    fn default_registry_resolves_plaintext_extensions() {
        let registry = default_registry();

        let txt = registry.resolve_for_path(Path::new("notes/readme.txt"));
        let md = registry.resolve_for_path(Path::new("notes/readme.md"));
        let code = registry.resolve_for_path(Path::new("src/lib.rs"));
        let unknown = registry.resolve_for_path(Path::new("notes/readme.rst"));

        assert!(txt.is_some());
        assert!(md
            .as_ref()
            .is_some_and(|extractor| extractor.supports().contains(&"md")));
        assert!(code
            .as_ref()
            .is_some_and(|extractor| extractor.profile_key() == "code"));
        assert!(unknown.is_none());
    }
}
