use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument, Extractor};
use crate::Result;

pub struct MarkdownExtractor;

impl Extractor for MarkdownExtractor {
    fn supports(&self) -> &[&str] {
        &["md", "markdown", "mdown", "mkd"]
    }

    fn profile_key(&self) -> &'static str {
        "md"
    }

    fn version(&self) -> u32 {
        2
    }

    fn extract(&self, _path: &Path, bytes: &[u8]) -> Result<ExtractedDocument> {
        let source = std::str::from_utf8(bytes).map_err(|err| {
            kbolt_types::KboltError::InvalidInput(format!("non-utf8 markdown input: {err}"))
        })?;

        let mut blocks = Vec::new();
        let mut heading_stack: Vec<String> = Vec::new();
        let mut open_blocks: Vec<OpenBlock> = Vec::new();
        let mut title: Option<String> = None;
        let parser = Parser::new_ext(source, Options::all());

        for (event, range) in parser.into_offset_iter() {
            match event {
                Event::Start(tag) => {
                    if let Some(open) =
                        open_block_for_tag(&tag, range.start, &heading_stack, &open_blocks)
                    {
                        open_blocks.push(open);
                    }
                }
                Event::End(tag_end) => {
                    let Some(index) = open_blocks
                        .iter()
                        .rposition(|open| open.matches_end(&tag_end))
                    else {
                        continue;
                    };
                    let open = open_blocks.remove(index);
                    let exclude_end = range.end.min(source.len());
                    for parent in &mut open_blocks {
                        if parent.start <= open.start {
                            parent.excluded_ranges.push(open.start..exclude_end);
                        }
                    }

                    let span_end = trim_trailing_newlines(source, open.start, range.end);
                    if span_end <= open.start {
                        continue;
                    }

                    let text = block_text(source, open.start, span_end, &open.excluded_ranges);
                    if text.trim().is_empty() {
                        continue;
                    }

                    if let OpenKind::Heading(level) = open.kind {
                        let heading = extract_heading_label(text.as_str());
                        if !heading.is_empty() {
                            apply_heading(&mut heading_stack, level, heading.clone());
                            if title.is_none() {
                                title = Some(heading);
                            }
                        }
                    }

                    let length = text.len();
                    blocks.push(ExtractedBlock {
                        text,
                        offset: open.start,
                        length,
                        kind: open.block_kind,
                        heading_path: open.heading_path,
                        attrs: open.attrs,
                    });
                }
                _ => {}
            }
        }

        blocks.sort_by_key(|block| block.offset);

        Ok(ExtractedDocument {
            blocks,
            metadata: HashMap::new(),
            title,
        })
    }
}

#[derive(Debug, Clone)]
struct OpenBlock {
    kind: OpenKind,
    block_kind: BlockKind,
    start: usize,
    heading_path: Vec<String>,
    attrs: HashMap<String, String>,
    excluded_ranges: Vec<Range<usize>>,
}

impl OpenBlock {
    fn matches_end(&self, end: &TagEnd) -> bool {
        match (&self.kind, end) {
            (OpenKind::Heading(level), TagEnd::Heading(end_level)) => {
                *level == heading_level(end_level)
            }
            (OpenKind::Paragraph, TagEnd::Paragraph) => true,
            (OpenKind::ListItem, TagEnd::Item) => true,
            (OpenKind::BlockQuote, TagEnd::BlockQuote(_)) => true,
            (OpenKind::CodeFence, TagEnd::CodeBlock) => true,
            (OpenKind::TableHeader, TagEnd::TableHead) => true,
            (OpenKind::TableRow, TagEnd::TableRow) => true,
            (OpenKind::HtmlBlock, TagEnd::HtmlBlock) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum OpenKind {
    Heading(usize),
    Paragraph,
    ListItem,
    BlockQuote,
    CodeFence,
    TableHeader,
    TableRow,
    HtmlBlock,
}

fn open_block_for_tag(
    tag: &Tag<'_>,
    start: usize,
    heading_path: &[String],
    open_blocks: &[OpenBlock],
) -> Option<OpenBlock> {
    let (kind, block_kind, attrs) = match tag {
        Tag::Heading { level, .. } => (
            OpenKind::Heading(heading_level(level)),
            BlockKind::Heading,
            HashMap::new(),
        ),
        Tag::Paragraph if inside_list_or_quote(open_blocks) => return None,
        Tag::Paragraph => (OpenKind::Paragraph, BlockKind::Paragraph, HashMap::new()),
        Tag::Item => (OpenKind::ListItem, BlockKind::ListItem, HashMap::new()),
        Tag::BlockQuote(_) => (OpenKind::BlockQuote, BlockKind::BlockQuote, HashMap::new()),
        Tag::CodeBlock(kind) => {
            let mut attrs = HashMap::new();
            if let CodeBlockKind::Fenced(info) = kind {
                if let Some(language) = info.split_whitespace().next() {
                    if !language.is_empty() {
                        attrs.insert("language".to_string(), language.to_string());
                    }
                }
            }
            (OpenKind::CodeFence, BlockKind::CodeFence, attrs)
        }
        Tag::TableHead => (
            OpenKind::TableHeader,
            BlockKind::TableHeader,
            HashMap::new(),
        ),
        Tag::TableRow => (OpenKind::TableRow, BlockKind::TableRow, HashMap::new()),
        Tag::HtmlBlock => (OpenKind::HtmlBlock, BlockKind::HtmlBlock, HashMap::new()),
        _ => return None,
    };

    Some(OpenBlock {
        kind,
        block_kind,
        start,
        heading_path: heading_path.to_vec(),
        attrs,
        excluded_ranges: Vec::new(),
    })
}

fn inside_list_or_quote(open_blocks: &[OpenBlock]) -> bool {
    open_blocks
        .iter()
        .any(|open| matches!(open.kind, OpenKind::ListItem | OpenKind::BlockQuote))
}

fn heading_level(level: &HeadingLevel) -> usize {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

fn apply_heading(stack: &mut Vec<String>, level: usize, heading: String) {
    while stack.len() >= level {
        stack.pop();
    }
    stack.push(heading);
}

fn extract_heading_label(raw_markdown: &str) -> String {
    let line = raw_markdown.lines().next().unwrap_or("").trim();
    let stripped = line
        .trim_start_matches('#')
        .trim()
        .trim_end_matches('#')
        .trim();

    if stripped.is_empty() {
        line.to_string()
    } else {
        stripped.to_string()
    }
}

fn trim_trailing_newlines(source: &str, start: usize, end: usize) -> usize {
    let bytes = source.as_bytes();
    let mut cursor = end.min(bytes.len());
    while cursor > start && matches!(bytes[cursor - 1], b'\n' | b'\r') {
        cursor -= 1;
    }
    cursor
}

fn block_text(source: &str, start: usize, end: usize, excluded_ranges: &[Range<usize>]) -> String {
    if excluded_ranges.is_empty() {
        return source[start..end].to_string();
    }

    let mut ranges = excluded_ranges
        .iter()
        .filter_map(|range| {
            let range_start = range.start.max(start).min(end);
            let range_end = range.end.max(start).min(end);
            (range_start < range_end).then_some(range_start..range_end)
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.start);

    let mut text = String::new();
    let mut cursor = start;
    for range in ranges {
        if range.start > cursor {
            text.push_str(&source[cursor..range.start]);
        }
        cursor = cursor.max(range.end);
    }
    if cursor < end {
        text.push_str(&source[cursor..end]);
    }

    text.trim_end_matches([' ', '\t', '\n', '\r']).to_string()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::ingest::extract::{BlockKind, Extractor};
    use crate::ingest::markdown::MarkdownExtractor;

    #[test]
    fn extracts_heading_paths_for_nested_sections() {
        let extractor = MarkdownExtractor;
        assert_eq!(extractor.profile_key(), "md");
        let markdown = br#"# Title
Intro paragraph.

## Details
More text.
"#;

        let doc = extractor
            .extract(Path::new("docs/readme.md"), markdown)
            .expect("extract markdown");

        assert_eq!(doc.title.as_deref(), Some("Title"));
        assert!(
            doc.blocks
                .iter()
                .any(|block| block.kind == BlockKind::Heading),
            "expected heading blocks"
        );
        assert!(
            doc.blocks.iter().any(|block| {
                block.kind == BlockKind::Paragraph
                    && block.heading_path == vec!["Title".to_string(), "Details".to_string()]
            }),
            "expected paragraph to carry nested heading path"
        );
    }

    #[test]
    fn emits_list_quote_and_code_blocks_with_attrs() {
        let extractor = MarkdownExtractor;
        let markdown = br#"# Guide
- first item

> quoted text

```rust
fn main() {}
```
"#;

        let doc = extractor
            .extract(Path::new("docs/guide.md"), markdown)
            .expect("extract markdown");

        assert!(doc
            .blocks
            .iter()
            .any(|block| block.kind == BlockKind::ListItem));
        assert!(doc
            .blocks
            .iter()
            .any(|block| block.kind == BlockKind::BlockQuote));
        let code = doc
            .blocks
            .iter()
            .find(|block| block.kind == BlockKind::CodeFence)
            .expect("code fence block");
        assert_eq!(code.attrs.get("language").map(String::as_str), Some("rust"));
    }

    #[test]
    fn nested_list_items_do_not_duplicate_child_text() {
        let extractor = MarkdownExtractor;
        let doc = extractor
            .extract(
                Path::new("docs/list.md"),
                br#"- parent listtarget
  - child nestedtarget
"#,
            )
            .expect("extract markdown");

        let list_items = doc
            .blocks
            .iter()
            .filter(|block| block.kind == BlockKind::ListItem)
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            list_items,
            vec!["- parent listtarget", "- child nestedtarget"]
        );

        let canonical = list_items.join("\n\n");
        assert_eq!(canonical.matches("nestedtarget").count(), 1);
        assert!(!canonical.contains("listtarget\n  - child"));
        assert!(doc
            .blocks
            .iter()
            .all(|block| block.length == block.text.len()));
    }

    #[test]
    fn rejects_non_utf8_markdown_bytes() {
        let extractor = MarkdownExtractor;
        let err = extractor
            .extract(Path::new("docs/readme.md"), &[0xff, 0xfe, 0xfd])
            .expect_err("invalid utf8 should fail");
        assert!(err.to_string().contains("non-utf8 markdown input"));
    }
}
