use std::collections::HashMap;
use std::path::Path;

use crate::ingest::extract::{BlockKind, ExtractedBlock, ExtractedDocument, Extractor};
use crate::Result;

pub struct PdfExtractor;

impl Extractor for PdfExtractor {
    fn supports(&self) -> &[&str] {
        &["pdf"]
    }

    fn profile_key(&self) -> &'static str {
        "pdf"
    }

    fn extract(&self, _path: &Path, bytes: &[u8]) -> Result<ExtractedDocument> {
        let text = extract_pdf_text(bytes)?;

        Ok(ExtractedDocument {
            blocks: paragraph_blocks(text.as_str()),
            metadata: HashMap::new(),
            title: None,
        })
    }
}

fn extract_pdf_text(bytes: &[u8]) -> Result<String> {
    match std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem(bytes)) {
        Ok(Ok(text)) => Ok(text),
        Ok(Err(err)) => Err(kbolt_types::KboltError::InvalidInput(format!(
            "pdf text extraction failed: {err}"
        ))
        .into()),
        Err(_) => Err(kbolt_types::KboltError::InvalidInput(
            "pdf text extraction failed: parser panicked".to_string(),
        )
        .into()),
    }
}

fn paragraph_blocks(text: &str) -> Vec<ExtractedBlock> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    let mut next_offset = 0usize;

    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.trim().is_empty() {
            push_paragraph(&mut blocks, &mut current, &mut next_offset);
            continue;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(trimmed);
    }

    push_paragraph(&mut blocks, &mut current, &mut next_offset);
    blocks
}

fn push_paragraph(blocks: &mut Vec<ExtractedBlock>, current: &mut String, next_offset: &mut usize) {
    let text = current.trim().to_string();
    current.clear();
    if text.is_empty() {
        return;
    }

    let offset = *next_offset;
    let length = text.len();
    *next_offset = next_offset.saturating_add(length).saturating_add(2);
    blocks.push(ExtractedBlock {
        text,
        offset,
        length,
        kind: BlockKind::Paragraph,
        heading_path: Vec::new(),
        attrs: HashMap::new(),
    });
}

#[cfg(test)]
pub(crate) fn simple_pdf_fixture(text: &str) -> Vec<u8> {
    let escaped = text
        .replace('\\', "\\\\")
        .replace('(', "\\(")
        .replace(')', "\\)")
        .replace('\n', ") Tj T* (");
    let stream = format!("BT /F1 12 Tf 72 720 Td 14 TL ({escaped}) Tj ET");
    let objects = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".to_string(),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        format!("<< /Length {} >>\nstream\n{}\nendstream", stream.len(), stream),
    ];

    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (index, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", index + 1, object).as_bytes());
    }

    let xref_offset = pdf.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len() + 1).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n",
            objects.len() + 1
        )
        .as_bytes(),
    );
    pdf
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::ingest::extract::Extractor;
    use crate::ingest::pdf::{simple_pdf_fixture, PdfExtractor};

    #[test]
    fn extracts_digital_pdf_text_into_paragraphs() {
        let extractor = PdfExtractor;
        assert_eq!(extractor.profile_key(), "pdf");

        let doc = extractor
            .extract(
                Path::new("papers/guide.pdf"),
                &simple_pdf_fixture("Alpha pdf target.\nSecond line."),
            )
            .expect("extract pdf");

        assert_eq!(doc.blocks.len(), 1);
        assert!(doc.blocks[0].text.contains("Alpha pdf target."));
        assert!(doc.blocks[0].text.contains("Second line."));
    }

    #[test]
    fn rejects_invalid_pdf_bytes() {
        let extractor = PdfExtractor;
        let err = extractor
            .extract(Path::new("papers/bad.pdf"), b"not a pdf")
            .expect_err("invalid pdf should fail");
        assert!(err.to_string().contains("pdf text extraction failed"));
    }
}
