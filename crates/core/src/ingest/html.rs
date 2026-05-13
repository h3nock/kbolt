use std::collections::HashMap;
use std::path::Path;

use scraper::{ElementRef, Html, Node, Selector};

use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument, Extractor};
use crate::Result;

pub struct HtmlExtractor;

impl Extractor for HtmlExtractor {
    fn supports(&self) -> &[&str] {
        &["html", "htm"]
    }

    fn profile_key(&self) -> &'static str {
        "html"
    }

    fn extract(&self, _path: &Path, bytes: &[u8]) -> Result<ExtractedDocument> {
        let source = std::str::from_utf8(bytes).map_err(|err| {
            kbolt_types::KboltError::InvalidInput(format!("non-utf8 html input: {err}"))
        })?;
        let document = Html::parse_document(source);
        let mut state = ExtractionState::new(document_title(&document));

        let body_selector = Selector::parse("body").expect("valid body selector");
        let bodies = document.select(&body_selector).collect::<Vec<_>>();
        if bodies.is_empty() {
            state.walk_element(document.root_element());
        } else {
            for body in bodies {
                state.walk_element(body);
            }
        }

        Ok(ExtractedDocument {
            blocks: state.blocks,
            metadata: HashMap::new(),
            title: state.title.or(state.first_h1),
        })
    }
}

struct ExtractionState {
    blocks: Vec<ExtractedBlock>,
    heading_stack: Vec<String>,
    next_offset: usize,
    title: Option<String>,
    first_h1: Option<String>,
}

impl ExtractionState {
    fn new(title: Option<String>) -> Self {
        Self {
            blocks: Vec::new(),
            heading_stack: Vec::new(),
            next_offset: 0,
            title,
            first_h1: None,
        }
    }

    fn walk_element(&mut self, element: ElementRef<'_>) -> bool {
        let name = element_name(element);
        if should_skip_element(name) {
            return false;
        }

        if let Some(kind) = block_kind_for(name) {
            let text = match kind {
                BlockKind::CodeFence => collect_preserved_text(element),
                _ => collect_normal_text(element),
            };
            self.push_block(kind, name, text);
            return true;
        }

        let mut emitted_child = false;
        for child in element.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                emitted_child |= self.walk_element(child_element);
            }
        }

        if !emitted_child && is_text_container(name) {
            let text = collect_normal_text(element);
            if !text.is_empty() {
                self.push_block(BlockKind::Paragraph, name, text);
                return true;
            }
        }

        emitted_child
    }

    fn push_block(&mut self, kind: BlockKind, element_name: &str, text: String) {
        if text.is_empty() {
            return;
        }

        let heading_path = self.heading_stack.clone();
        let offset = self.next_offset;
        let length = text.len();
        self.next_offset = self.next_offset.saturating_add(length).saturating_add(2);

        if kind == BlockKind::Heading {
            let heading = text.clone();
            if let Some(level) = heading_level(element_name) {
                if level == 1 && self.first_h1.is_none() {
                    self.first_h1 = Some(heading.clone());
                }
                apply_heading(&mut self.heading_stack, level, heading);
            }
        }

        self.blocks.push(ExtractedBlock {
            text,
            offset,
            length,
            kind,
            heading_path,
            attrs: HashMap::new(),
        });
    }
}

fn document_title(document: &Html) -> Option<String> {
    let selector = Selector::parse("title").expect("valid title selector");
    document
        .select(&selector)
        .next()
        .map(collect_normal_text)
        .filter(|title| !title.is_empty())
}

fn element_name(element: ElementRef<'_>) -> &str {
    element.value().name()
}

fn should_skip_element(name: &str) -> bool {
    matches!(
        name,
        "head" | "script" | "style" | "template" | "noscript" | "svg" | "canvas" | "math"
    )
}

fn block_kind_for(name: &str) -> Option<BlockKind> {
    if heading_level(name).is_some() {
        return Some(BlockKind::Heading);
    }

    match name {
        "p" => Some(BlockKind::Paragraph),
        "li" => Some(BlockKind::ListItem),
        "blockquote" => Some(BlockKind::BlockQuote),
        "pre" => Some(BlockKind::CodeFence),
        _ => None,
    }
}

fn is_text_container(name: &str) -> bool {
    matches!(name, "body" | "main" | "article" | "section" | "div")
}

fn heading_level(name: &str) -> Option<usize> {
    let bytes = name.as_bytes();
    if bytes.len() == 2 && bytes[0] == b'h' && (b'1'..=b'6').contains(&bytes[1]) {
        return Some((bytes[1] - b'0') as usize);
    }
    None
}

fn apply_heading(stack: &mut Vec<String>, level: usize, heading: String) {
    while stack.len() >= level {
        stack.pop();
    }
    stack.push(heading);
}

fn collect_normal_text(element: ElementRef<'_>) -> String {
    let mut collector = TextCollector::normal();
    collect_text_from_element(element, &mut collector);
    collector.finish()
}

fn collect_preserved_text(element: ElementRef<'_>) -> String {
    let mut collector = TextCollector::preserve();
    collect_text_from_element(element, &mut collector);
    trim_preserved_text(collector.finish().as_str())
}

fn collect_text_from_element(element: ElementRef<'_>, collector: &mut TextCollector) {
    if should_skip_element(element_name(element)) {
        return;
    }

    for child in element.children() {
        match child.value() {
            Node::Text(text) => collector.push(text),
            Node::Element(_) => {
                if let Some(child_element) = ElementRef::wrap(child) {
                    collect_text_from_element(child_element, collector);
                }
            }
            _ => {}
        }
    }
}

enum TextMode {
    Normal,
    Preserve,
}

struct TextCollector {
    text: String,
    mode: TextMode,
    last_was_space: bool,
}

impl TextCollector {
    fn normal() -> Self {
        Self {
            text: String::new(),
            mode: TextMode::Normal,
            last_was_space: false,
        }
    }

    fn preserve() -> Self {
        Self {
            text: String::new(),
            mode: TextMode::Preserve,
            last_was_space: false,
        }
    }

    fn push(&mut self, raw: &str) {
        match self.mode {
            TextMode::Normal => self.push_normal(raw),
            TextMode::Preserve => self.text.push_str(raw),
        }
    }

    fn push_normal(&mut self, raw: &str) {
        for ch in raw.chars() {
            if ch.is_whitespace() {
                if !self.text.is_empty() && !self.last_was_space {
                    self.text.push(' ');
                    self.last_was_space = true;
                }
            } else {
                self.text.push(ch);
                self.last_was_space = false;
            }
        }
    }

    fn finish(self) -> String {
        match self.mode {
            TextMode::Normal => self.text.trim().to_string(),
            TextMode::Preserve => self.text,
        }
    }
}

fn trim_preserved_text(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines = normalized.lines().collect::<Vec<_>>();
    let start = lines
        .iter()
        .position(|line| !line.trim().is_empty())
        .unwrap_or(lines.len());
    let end = lines
        .iter()
        .rposition(|line| !line.trim().is_empty())
        .map(|index| index + 1)
        .unwrap_or(start);

    lines[start..end].join("\n")
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::ingest::extract::{BlockKind, Extractor};
    use crate::ingest::html::HtmlExtractor;

    #[test]
    fn extracts_structural_html_blocks() {
        let extractor = HtmlExtractor;
        assert_eq!(extractor.profile_key(), "html");

        let source = br#"<!doctype html>
<html>
  <head>
    <title>Guide Title</title>
    <style>.hidden { display: none; }</style>
    <script>ignored_script()</script>
  </head>
  <body>
    <h1>Guide</h1>
    <p>Alpha <strong>HTML</strong> &amp; canonical text.</p>
    <ul><li>First item</li></ul>
    <blockquote>Quoted text</blockquote>
    <pre><code>
fn main() {}
    </code></pre>
  </body>
</html>"#;

        let doc = extractor
            .extract(Path::new("docs/guide.html"), source)
            .expect("extract html");

        assert_eq!(doc.title.as_deref(), Some("Guide Title"));
        assert!(doc
            .blocks
            .iter()
            .any(|block| block.kind == BlockKind::Heading && block.text == "Guide"));
        assert!(doc.blocks.iter().any(|block| {
            block.kind == BlockKind::Paragraph
                && block.text == "Alpha HTML & canonical text."
                && block.heading_path == vec!["Guide".to_string()]
        }));
        assert!(doc
            .blocks
            .iter()
            .any(|block| block.kind == BlockKind::ListItem && block.text == "First item"));
        assert!(doc
            .blocks
            .iter()
            .any(|block| block.kind == BlockKind::BlockQuote && block.text == "Quoted text"));
        assert!(doc.blocks.iter().any(|block| {
            block.kind == BlockKind::CodeFence && block.text.contains("fn main() {}")
        }));
        assert!(!doc
            .blocks
            .iter()
            .any(|block| block.text.contains("ignored_script")));
    }

    #[test]
    fn uses_first_h1_as_title_when_title_is_missing() {
        let extractor = HtmlExtractor;
        let doc = extractor
            .extract(
                Path::new("docs/guide.html"),
                b"<main><h2>Section</h2><p>before</p><h1>Guide</h1><p>body</p></main>",
            )
            .expect("extract html");

        assert_eq!(doc.title.as_deref(), Some("Guide"));
    }

    #[test]
    fn rejects_non_utf8_html_bytes() {
        let extractor = HtmlExtractor;
        let err = extractor
            .extract(Path::new("docs/page.html"), &[0xff, 0xfe, 0xfd])
            .expect_err("invalid utf8 should fail");
        assert!(err.to_string().contains("non-utf8 html input"));
    }
}
