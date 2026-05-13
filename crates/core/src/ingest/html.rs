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

    fn version(&self) -> u32 {
        4
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
        if should_skip_element(element) {
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
        let mut residual = TextCollector::normal();
        for child in element.children() {
            match child.value() {
                Node::Text(text) => residual.push(text),
                Node::Element(_) => {
                    let Some(child_element) = ElementRef::wrap(child) else {
                        continue;
                    };
                    let child_name = element_name(child_element);
                    if should_skip_element(child_element) {
                        continue;
                    }
                    if is_text_boundary_element(child_name) {
                        residual.push_boundary();
                        continue;
                    }

                    if block_kind_for(child_name).is_some() || is_structural_container(child_name) {
                        emitted_child |= self.push_residual_paragraph(&mut residual);
                        emitted_child |= self.walk_element(child_element);
                    } else {
                        collect_text_from_element(child_element, &mut residual);
                    }
                }
                _ => {}
            }
        }

        emitted_child |= self.push_residual_paragraph(&mut residual);
        emitted_child
    }

    fn push_residual_paragraph(&mut self, residual: &mut TextCollector) -> bool {
        let text = residual.take();
        if text.is_empty() {
            return false;
        }

        self.push_block(BlockKind::Paragraph, "p", text);
        true
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

fn should_skip_element(element: ElementRef<'_>) -> bool {
    if should_skip_element_name(element_name(element)) {
        return true;
    }

    element.value().attr("hidden").is_some()
        || element
            .value()
            .attr("aria-hidden")
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("true"))
        || element
            .value()
            .attr("style")
            .is_some_and(style_declares_hidden)
}

fn should_skip_element_name(name: &str) -> bool {
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
        "dt" | "dd" => Some(BlockKind::Paragraph),
        "blockquote" => Some(BlockKind::BlockQuote),
        "pre" => Some(BlockKind::CodeFence),
        _ => None,
    }
}

fn is_text_boundary_element(name: &str) -> bool {
    matches!(name, "br" | "hr")
}

fn is_structural_container(name: &str) -> bool {
    matches!(
        name,
        "html"
            | "body"
            | "main"
            | "article"
            | "section"
            | "div"
            | "header"
            | "footer"
            | "nav"
            | "aside"
            | "ul"
            | "ol"
            | "menu"
            | "dl"
            | "table"
            | "thead"
            | "tbody"
            | "tfoot"
            | "tr"
            | "td"
            | "th"
            | "caption"
            | "figure"
            | "figcaption"
    )
}

fn style_declares_hidden(style: &str) -> bool {
    let mut display: Option<StyleDeclarationState> = None;
    let mut visibility: Option<StyleDeclarationState> = None;

    for declaration in style.split(';') {
        let Some((raw_name, raw_value)) = declaration.split_once(':') else {
            continue;
        };
        let name = raw_name.trim().to_ascii_lowercase();
        let important = raw_value
            .split('!')
            .skip(1)
            .any(|suffix| suffix.trim().eq_ignore_ascii_case("important"));
        let value = raw_value.split('!').next().unwrap_or(raw_value).trim();

        match name.as_str() {
            "display" => apply_style_state(
                &mut display,
                StyleDeclarationState {
                    important,
                    hidden: value.eq_ignore_ascii_case("none"),
                },
            ),
            "visibility" => apply_style_state(
                &mut visibility,
                StyleDeclarationState {
                    important,
                    hidden: value.eq_ignore_ascii_case("hidden")
                        || value.eq_ignore_ascii_case("collapse"),
                },
            ),
            _ => {}
        }
    }

    display.is_some_and(|state| state.hidden) || visibility.is_some_and(|state| state.hidden)
}

#[derive(Clone, Copy)]
struct StyleDeclarationState {
    important: bool,
    hidden: bool,
}

fn apply_style_state(current: &mut Option<StyleDeclarationState>, next: StyleDeclarationState) {
    if current.is_none_or(|state| next.important || !state.important) {
        *current = Some(next);
    }
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
    if should_skip_element(element) {
        return;
    }

    for child in element.children() {
        match child.value() {
            Node::Text(text) => collector.push(text),
            Node::Element(_) => {
                if let Some(child_element) = ElementRef::wrap(child) {
                    let child_name = element_name(child_element);
                    if should_skip_element(child_element) {
                        continue;
                    }

                    if is_text_boundary_element(child_name) {
                        collector.push_boundary();
                    } else if block_kind_for(child_name).is_some()
                        || is_structural_container(child_name)
                    {
                        collector.push_boundary();
                        collect_text_from_element(child_element, collector);
                        collector.push_boundary();
                    } else {
                        collect_text_from_element(child_element, collector);
                    }
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

    fn push_boundary(&mut self) {
        match self.mode {
            TextMode::Normal => {
                if !self.text.is_empty() && !self.last_was_space {
                    self.text.push(' ');
                    self.last_was_space = true;
                }
            }
            TextMode::Preserve => {
                if !self.text.ends_with('\n') {
                    self.text.push('\n');
                }
                self.last_was_space = false;
            }
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

    fn take(&mut self) -> String {
        let text = std::mem::take(&mut self.text);
        self.last_was_space = false;
        match self.mode {
            TextMode::Normal => text.trim().to_string(),
            TextMode::Preserve => text,
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
    fn preserves_visible_text_from_mixed_unrecognized_containers() {
        let extractor = HtmlExtractor;
        let doc = extractor
            .extract(
                Path::new("docs/prices.html"),
                br#"<body>
Lead text.
<p>Intro paragraph.</p>
<table><tr><td>Price tabletarget</td></tr></table>
<span>Tail text.</span>
</body>"#,
            )
            .expect("extract html");

        let texts = doc
            .blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>();
        assert!(texts.iter().any(|text| *text == "Lead text."));
        assert!(texts.iter().any(|text| *text == "Intro paragraph."));
        assert!(texts.iter().any(|text| *text == "Price tabletarget"));
        assert!(texts.iter().any(|text| *text == "Tail text."));
        assert_eq!(
            texts
                .iter()
                .filter(|text| text.contains("Intro paragraph."))
                .count(),
            1
        );
    }

    #[test]
    fn preserves_boundaries_for_html_separator_elements() {
        let extractor = HtmlExtractor;
        let doc = extractor
            .extract(
                Path::new("docs/separators.html"),
                br#"<body>
<p>alpha<br>beta brtarget</p>
<dl><dt>Term</dt><dd>Definition ddtarget</dd></dl>
<div><span>left</span><hr><span>right hrtarget</span></div>
</body>"#,
            )
            .expect("extract html");

        let canonical = doc
            .blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        assert!(canonical.contains("alpha beta brtarget"));
        assert!(canonical.contains("Term\n\nDefinition ddtarget"));
        assert!(canonical.contains("left right hrtarget"));
        assert!(!canonical.contains("alphabeta"));
        assert!(!canonical.contains("TermDefinition"));
        assert!(!canonical.contains("leftright"));
    }

    #[test]
    fn preserves_boundaries_for_nested_block_children() {
        let extractor = HtmlExtractor;
        let doc = extractor
            .extract(
                Path::new("docs/nested.html"),
                br#"<body>
<blockquote><p>Alpha quote</p><p>Beta quotetarget</p></blockquote>
<ul><li><p>Parent item</p><p>Child paragraph listtarget</p></li></ul>
</body>"#,
            )
            .expect("extract html");

        let canonical = doc
            .blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        assert!(canonical.contains("Alpha quote Beta quotetarget"));
        assert!(canonical.contains("Parent item Child paragraph listtarget"));
        assert!(!canonical.contains("quoteBeta"));
        assert!(!canonical.contains("itemChild"));
    }

    #[test]
    fn skips_hidden_html_elements() {
        let extractor = HtmlExtractor;
        let doc = extractor
            .extract(
                Path::new("docs/hidden.html"),
                br#"<body>
<p>Visible target</p>
<div hidden>secret hiddenword</div>
<section aria-hidden=" true "><p>aria hiddenword</p></section>
<div style="display: none">style hiddenword</div>
<div style="visibility:hidden !important">visibility hiddenword</div>
<div style="display:none !important; display:block">important hiddenword</div>
<div style="display:none; display:block">Actually visible visibletarget</div>
<div style="visibility:hidden; visibility:visible !important">Visible important importanttarget</div>
</body>"#,
            )
            .expect("extract html");

        let canonical = doc
            .blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        assert!(canonical.contains("Visible target"));
        assert!(canonical.contains("Actually visible visibletarget"));
        assert!(canonical.contains("Visible important importanttarget"));
        assert!(!canonical.contains("hiddenword"));
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
