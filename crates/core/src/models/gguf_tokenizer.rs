use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use kbolt_types::KboltError;

use crate::models::tokenizer::{TokenizerRuntime, TokenizerRuntimeKind};
use crate::Result;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_SUPPORTED_VERSION: u32 = 3;

const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;
const TOKEN_TYPE_BYTE: i32 = 6;

const SPACE_MARKER: &str = "▁";

#[derive(Debug)]
pub(crate) struct LlamaSpmGgufTokenizerRuntime {
    model_path: PathBuf,
    tokenizer: GgufTokenizer,
}

impl LlamaSpmGgufTokenizerRuntime {
    pub(crate) fn from_path(path: &Path) -> Result<Self> {
        let metadata = GgufTokenizerMetadata::read_from_path(path)?;
        let tokenizer = GgufTokenizer::from_metadata(metadata)?;
        Ok(Self {
            model_path: path.to_path_buf(),
            tokenizer,
        })
    }
}

impl TokenizerRuntime for LlamaSpmGgufTokenizerRuntime {
    fn kind(&self) -> TokenizerRuntimeKind {
        TokenizerRuntimeKind::LlamaSpmGgufEmbedded
    }

    fn count_embedding_tokens(&self, text: &str) -> Result<usize> {
        self.tokenizer.count_embedding_tokens(text).map_err(|err| {
            KboltError::Inference(format!(
                "failed to count tokens with GGUF tokenizer {}: {err}",
                self.model_path.display()
            ))
            .into()
        })
    }
}

#[derive(Debug)]
struct GgufTokenizer {
    token_scores: HashMap<Vec<u8>, f64>,
    user_defined_tokens: Vec<Vec<u8>>,
    byte_tokens: [bool; 256],
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
}

impl GgufTokenizer {
    fn from_metadata(metadata: GgufTokenizerMetadata) -> Result<Self> {
        if metadata.model != "llama" {
            return Err(KboltError::Inference(format!(
                "unsupported GGUF tokenizer model '{}'",
                metadata.model
            ))
            .into());
        }
        if metadata.pre.as_deref().unwrap_or("default") != "default" {
            return Err(KboltError::Inference(format!(
                "unsupported GGUF tokenizer preprocessor '{}'",
                metadata.pre.unwrap_or_default()
            ))
            .into());
        }
        if metadata.tokens.is_empty() {
            return Err(KboltError::Inference("GGUF tokenizer has no tokens".to_string()).into());
        }
        if metadata.scores.len() != metadata.tokens.len() {
            return Err(KboltError::Inference(format!(
                "GGUF tokenizer scores length {} does not match token length {}",
                metadata.scores.len(),
                metadata.tokens.len()
            ))
            .into());
        }
        if metadata.token_types.len() != metadata.tokens.len() {
            return Err(KboltError::Inference(format!(
                "GGUF tokenizer token_type length {} does not match token length {}",
                metadata.token_types.len(),
                metadata.tokens.len()
            ))
            .into());
        }

        let mut token_scores: HashMap<Vec<u8>, f64> = HashMap::with_capacity(metadata.tokens.len());
        let mut user_defined_tokens = Vec::new();
        let mut byte_tokens = [false; 256];
        for (index, token) in metadata.tokens.iter().enumerate() {
            let score = f64::from(metadata.scores[index]);
            let token_type = metadata.token_types[index];

            match token_type {
                TOKEN_TYPE_NORMAL => {
                    insert_token_score(&mut token_scores, token.as_bytes(), score);
                    if token.contains(SPACE_MARKER) {
                        let unescaped = token.replace(SPACE_MARKER, " ");
                        insert_token_score(&mut token_scores, unescaped.as_bytes(), score);
                    }
                }
                TOKEN_TYPE_USER_DEFINED => {
                    insert_token_score(&mut token_scores, token.as_bytes(), score);
                    user_defined_tokens.push(token.as_bytes().to_vec());
                    if token.contains(SPACE_MARKER) {
                        let unescaped = token.replace(SPACE_MARKER, " ");
                        insert_token_score(&mut token_scores, unescaped.as_bytes(), score);
                        user_defined_tokens.push(unescaped.into_bytes());
                    }
                }
                TOKEN_TYPE_BYTE => {
                    if let Some(byte) = parse_byte_token(token) {
                        byte_tokens[usize::from(byte)] = true;
                    }
                }
                TOKEN_TYPE_CONTROL => {}
                _ => {}
            }
        }

        Ok(Self {
            token_scores,
            user_defined_tokens: sort_user_defined_tokens(user_defined_tokens),
            byte_tokens,
            add_bos: metadata
                .bos_token_id
                .map(|_| metadata.add_bos_token)
                .unwrap_or(false),
            add_eos: metadata
                .eos_token_id
                .map(|_| metadata.add_eos_token)
                .unwrap_or(false),
            add_space_prefix: metadata.add_space_prefix,
        })
    }

    fn count_embedding_tokens(&self, text: &str) -> Result<usize> {
        let body = self.count_body_tokens(text.as_bytes())?;
        let specials = usize::from(self.add_bos) + usize::from(self.add_eos);
        Ok(body.saturating_add(specials))
    }

    fn count_body_tokens(&self, input: &[u8]) -> Result<usize> {
        let mut count = 0;
        let mut segment_start = 0;
        let mut cursor = 0;
        let mut previous_was_special = true;
        while cursor < input.len() {
            if let Some(token_len) = self.match_user_defined(input, cursor) {
                count +=
                    self.count_raw_segment(&input[segment_start..cursor], previous_was_special)?;
                count += 1;
                cursor += token_len;
                segment_start = cursor;
                previous_was_special = true;
            } else {
                cursor = next_utf8_boundary(input, cursor);
            }
        }
        count += self.count_raw_segment(&input[segment_start..], previous_was_special)?;
        Ok(count)
    }

    fn count_raw_segment(&self, segment: &[u8], previous_was_special: bool) -> Result<usize> {
        if segment.is_empty() {
            return Ok(0);
        }

        if self.add_space_prefix && previous_was_special {
            let mut prefixed = Vec::with_capacity(segment.len() + 1);
            prefixed.push(b' ');
            prefixed.extend_from_slice(segment);
            SpmTokenizationSession::new(&self.token_scores, &self.byte_tokens)
                .count_tokens(&prefixed)
        } else {
            SpmTokenizationSession::new(&self.token_scores, &self.byte_tokens).count_tokens(segment)
        }
    }

    fn match_user_defined(&self, input: &[u8], start: usize) -> Option<usize> {
        self.user_defined_tokens
            .iter()
            .find_map(|token| input[start..].starts_with(token).then_some(token.len()))
    }
}

#[derive(Debug)]
struct SpmTokenizationSession<'a> {
    token_scores: &'a HashMap<Vec<u8>, f64>,
    byte_tokens: &'a [bool; 256],
    symbols: Vec<SpmSymbol>,
    work_queue: BinaryHeap<SpmBigram>,
}

fn insert_token_score(token_scores: &mut HashMap<Vec<u8>, f64>, token: &[u8], score: f64) {
    token_scores
        .entry(token.to_vec())
        .and_modify(|existing| *existing = (*existing).max(score))
        .or_insert(score);
}

fn sort_user_defined_tokens(mut tokens: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    tokens.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    tokens.dedup();
    tokens
}

fn next_utf8_boundary(input: &[u8], start: usize) -> usize {
    let mut cursor = start + 1;
    while cursor < input.len() && !is_utf8_leading_byte(input[cursor]) {
        cursor += 1;
    }
    cursor
}

impl<'a> SpmTokenizationSession<'a> {
    fn new(token_scores: &'a HashMap<Vec<u8>, f64>, byte_tokens: &'a [bool; 256]) -> Self {
        Self {
            token_scores,
            byte_tokens,
            symbols: Vec::new(),
            work_queue: BinaryHeap::new(),
        }
    }

    fn count_tokens(mut self, input: &[u8]) -> Result<usize> {
        if input.is_empty() {
            return Ok(0);
        }

        let mut starts = input
            .iter()
            .enumerate()
            .filter_map(|(index, byte)| is_utf8_leading_byte(*byte).then_some(index))
            .peekable();
        let mut symbol_index: usize = 0;
        while let Some(start) = starts.next() {
            let end = starts.peek().copied().unwrap_or(input.len());
            self.symbols.push(SpmSymbol {
                prev: symbol_index.checked_sub(1),
                next: (end < input.len()).then_some(symbol_index + 1),
                start,
                len: end - start,
            });
            symbol_index += 1;
        }

        for index in 1..self.symbols.len() {
            self.try_add_bigram(index - 1, index, input);
        }

        while let Some(bigram) = self.work_queue.pop() {
            let left = self.symbols[bigram.left];
            let right = self.symbols[bigram.right];
            if left.len == 0 || right.len == 0 || left.len + right.len != bigram.size {
                continue;
            }

            self.symbols[bigram.left].len += right.len;
            self.symbols[bigram.right].len = 0;
            self.symbols[bigram.left].next = right.next;
            if let Some(next) = right.next {
                self.symbols[next].prev = Some(bigram.left);
            }

            if let Some(prev) = left.prev {
                self.try_add_bigram(prev, bigram.left, input);
            }
            if let Some(next) = self.symbols[bigram.left].next {
                self.try_add_bigram(bigram.left, next, input);
            }
        }

        let mut count = 0;
        let mut index = Some(0);
        while let Some(symbol_index) = index {
            let symbol = self.symbols[symbol_index];
            let text = &input[symbol.start..symbol.start + symbol.len];
            if self.token_scores.contains_key(text) {
                count += 1;
            } else {
                for byte in text {
                    if !self.byte_tokens[usize::from(*byte)] {
                        return Err(KboltError::Inference(format!(
                            "GGUF tokenizer has no token or byte fallback for byte 0x{byte:02X}"
                        ))
                        .into());
                    }
                }
                count += text.len();
            }
            index = symbol.next;
        }
        Ok(count)
    }

    fn try_add_bigram(&mut self, left: usize, right: usize, input: &[u8]) {
        let left_symbol = self.symbols[left];
        let right_symbol = self.symbols[right];
        let size = left_symbol.len + right_symbol.len;
        let text = &input[left_symbol.start..left_symbol.start + size];
        if let Some(score) = self.token_scores.get(text) {
            self.work_queue.push(SpmBigram {
                left,
                right,
                score: *score,
                size,
            });
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SpmSymbol {
    prev: Option<usize>,
    next: Option<usize>,
    start: usize,
    len: usize,
}

#[derive(Debug, Clone, Copy)]
struct SpmBigram {
    left: usize,
    right: usize,
    score: f64,
    size: usize,
}

impl Eq for SpmBigram {}

impl PartialEq for SpmBigram {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left && self.right == other.right && self.size == other.size
    }
}

impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn is_utf8_leading_byte(byte: u8) -> bool {
    (byte & 0b1100_0000) != 0b1000_0000
}

fn parse_byte_token(token: &str) -> Option<u8> {
    token
        .strip_prefix("<0x")
        .and_then(|rest| rest.strip_suffix('>'))
        .and_then(|hex| u8::from_str_radix(hex, 16).ok())
}

#[derive(Debug)]
struct GgufTokenizerMetadata {
    model: String,
    pre: Option<String>,
    tokens: Vec<String>,
    scores: Vec<f32>,
    token_types: Vec<i32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    add_bos_token: bool,
    add_eos_token: bool,
    add_space_prefix: bool,
}

impl GgufTokenizerMetadata {
    fn read_from_path(path: &Path) -> Result<Self> {
        let mut reader = GgufReader::open(path)?;
        let mut metadata = GgufTokenizerMetadataBuilder::default();
        for _ in 0..reader.metadata_count {
            let key = reader.read_string()?;
            let value_type = reader.read_u32()?;
            match key.as_str() {
                "tokenizer.ggml.model" => {
                    metadata.model = Some(reader.read_string_value(value_type, &key)?);
                }
                "tokenizer.ggml.pre" => {
                    metadata.pre = Some(reader.read_string_value(value_type, &key)?);
                }
                "tokenizer.ggml.tokens" => {
                    metadata.tokens = Some(reader.read_string_array(value_type, &key)?);
                }
                "tokenizer.ggml.scores" => {
                    metadata.scores = Some(reader.read_f32_array(value_type, &key)?);
                }
                "tokenizer.ggml.token_type" => {
                    metadata.token_types = Some(reader.read_i32_array(value_type, &key)?);
                }
                "tokenizer.ggml.bos_token_id" => {
                    metadata.bos_token_id = Some(reader.read_u32_value(value_type, &key)?);
                }
                "tokenizer.ggml.eos_token_id" => {
                    metadata.eos_token_id = Some(reader.read_u32_value(value_type, &key)?);
                }
                "tokenizer.ggml.unknown_token_id" => {
                    let _ = reader.read_u32_value(value_type, &key)?;
                }
                "tokenizer.ggml.add_bos_token" => {
                    metadata.add_bos_token = Some(reader.read_bool_value(value_type, &key)?);
                }
                "tokenizer.ggml.add_eos_token" => {
                    metadata.add_eos_token = Some(reader.read_bool_value(value_type, &key)?);
                }
                "tokenizer.ggml.add_space_prefix" => {
                    metadata.add_space_prefix = Some(reader.read_bool_value(value_type, &key)?);
                }
                _ => reader.skip_value(value_type)?,
            }
        }
        metadata.build(path)
    }
}

#[derive(Debug, Default)]
struct GgufTokenizerMetadataBuilder {
    model: Option<String>,
    pre: Option<String>,
    tokens: Option<Vec<String>>,
    scores: Option<Vec<f32>>,
    token_types: Option<Vec<i32>>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    add_space_prefix: Option<bool>,
}

impl GgufTokenizerMetadataBuilder {
    fn build(self, path: &Path) -> Result<GgufTokenizerMetadata> {
        let tokens = self.tokens.ok_or_else(|| {
            KboltError::Inference(format!(
                "GGUF tokenizer {} is missing tokenizer.ggml.tokens",
                path.display()
            ))
        })?;
        let model = self.model.ok_or_else(|| {
            KboltError::Inference(format!(
                "GGUF tokenizer {} is missing tokenizer.ggml.model",
                path.display()
            ))
        })?;
        let is_llama_spm = model == "llama";
        Ok(GgufTokenizerMetadata {
            model,
            pre: self.pre,
            scores: self.scores.ok_or_else(|| {
                KboltError::Inference(format!(
                    "GGUF tokenizer {} is missing tokenizer.ggml.scores",
                    path.display()
                ))
            })?,
            token_types: self.token_types.ok_or_else(|| {
                KboltError::Inference(format!(
                    "GGUF tokenizer {} is missing tokenizer.ggml.token_type",
                    path.display()
                ))
            })?,
            tokens,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            add_bos_token: self.add_bos_token.unwrap_or(is_llama_spm),
            add_eos_token: self.add_eos_token.unwrap_or(false),
            add_space_prefix: self.add_space_prefix.unwrap_or(is_llama_spm),
        })
    }
}

struct GgufReader {
    file: File,
    metadata_count: u64,
}

impl GgufReader {
    fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0_u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != GGUF_MAGIC {
            return Err(
                KboltError::Inference(format!("not a GGUF file: {}", path.display())).into(),
            );
        }
        let version = read_u32_from(&mut file)?;
        if version != GGUF_SUPPORTED_VERSION {
            return Err(KboltError::Inference(format!(
                "unsupported GGUF version {version} in {}",
                path.display()
            ))
            .into());
        }
        let _tensor_count = read_u64_from(&mut file)?;
        let metadata_count = read_u64_from(&mut file)?;
        Ok(Self {
            file,
            metadata_count,
        })
    }

    fn read_u32(&mut self) -> Result<u32> {
        read_u32_from(&mut self.file)
    }

    fn read_string(&mut self) -> Result<String> {
        read_string_from(&mut self.file)
    }

    fn read_string_value(&mut self, value_type: u32, key: &str) -> Result<String> {
        require_type(key, value_type, GGUF_TYPE_STRING)?;
        self.read_string()
    }

    fn read_u32_value(&mut self, value_type: u32, key: &str) -> Result<u32> {
        require_type(key, value_type, GGUF_TYPE_UINT32)?;
        self.read_u32()
    }

    fn read_bool_value(&mut self, value_type: u32, key: &str) -> Result<bool> {
        require_type(key, value_type, GGUF_TYPE_BOOL)?;
        let mut byte = [0_u8; 1];
        self.file.read_exact(&mut byte)?;
        match byte[0] {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(KboltError::Inference(format!(
                "invalid GGUF bool value {other} for {key}"
            ))
            .into()),
        }
    }

    fn read_string_array(&mut self, value_type: u32, key: &str) -> Result<Vec<String>> {
        let len = self.read_array_header(value_type, key, GGUF_TYPE_STRING)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_string()?);
        }
        Ok(values)
    }

    fn read_f32_array(&mut self, value_type: u32, key: &str) -> Result<Vec<f32>> {
        let len = self.read_array_header(value_type, key, GGUF_TYPE_FLOAT32)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(read_f32_from(&mut self.file)?);
        }
        Ok(values)
    }

    fn read_i32_array(&mut self, value_type: u32, key: &str) -> Result<Vec<i32>> {
        let len = self.read_array_header(value_type, key, GGUF_TYPE_INT32)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(read_i32_from(&mut self.file)?);
        }
        Ok(values)
    }

    fn read_array_header(
        &mut self,
        value_type: u32,
        key: &str,
        expected_element_type: u32,
    ) -> Result<usize> {
        require_type(key, value_type, GGUF_TYPE_ARRAY)?;
        let element_type = self.read_u32()?;
        if element_type != expected_element_type {
            return Err(KboltError::Inference(format!(
                "GGUF metadata key {key} has array element type {element_type}, expected {expected_element_type}"
            ))
            .into());
        }
        let len = read_u64_from(&mut self.file)?;
        len.try_into().map_err(|_| {
            KboltError::Inference(format!("GGUF metadata array {key} is too large")).into()
        })
    }

    fn skip_value(&mut self, value_type: u32) -> Result<()> {
        match value_type {
            0 | 1 => {
                self.file.seek(SeekFrom::Current(1))?;
            }
            2 | 3 => {
                self.file.seek(SeekFrom::Current(2))?;
            }
            GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => {
                self.file.seek(SeekFrom::Current(4))?;
            }
            GGUF_TYPE_BOOL => {
                self.file.seek(SeekFrom::Current(1))?;
            }
            GGUF_TYPE_STRING => {
                let len = read_u64_from(&mut self.file)?;
                self.file
                    .seek(SeekFrom::Current(len.try_into().map_err(|_| {
                        KboltError::Inference("GGUF string length is too large".to_string())
                    })?))?;
            }
            GGUF_TYPE_ARRAY => {
                let element_type = self.read_u32()?;
                let len = read_u64_from(&mut self.file)?;
                for _ in 0..len {
                    self.skip_value(element_type)?;
                }
            }
            GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => {
                self.file.seek(SeekFrom::Current(8))?;
            }
            other => {
                return Err(KboltError::Inference(format!(
                    "unsupported GGUF metadata value type {other}"
                ))
                .into());
            }
        }
        Ok(())
    }
}

fn require_type(key: &str, actual: u32, expected: u32) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(KboltError::Inference(format!(
            "GGUF metadata key {key} has type {actual}, expected {expected}"
        ))
        .into())
    }
}

fn read_string_from(reader: &mut impl Read) -> Result<String> {
    let len = read_u64_from(reader)?;
    let len: usize = len
        .try_into()
        .map_err(|_| KboltError::Inference("GGUF string length is too large".to_string()))?;
    let mut bytes = vec![0_u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|err| {
        KboltError::Inference(format!("GGUF string is not valid UTF-8: {err}")).into()
    })
}

fn read_u32_from(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32_from(reader: &mut impl Read) -> Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64_from(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32_from(reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::process::Command;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn gguf_embedded_tokenizer_counts_space_marker_tokens_with_specials() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("model.gguf");
        write_test_gguf(&path, default_test_tokens()).expect("write test gguf");

        let runtime = LlamaSpmGgufTokenizerRuntime::from_path(&path).expect("load tokenizer");

        assert_eq!(runtime.count_embedding_tokens("").unwrap(), 2);
        assert_eq!(runtime.count_embedding_tokens("hello world").unwrap(), 4);
        assert_eq!(runtime.count_embedding_tokens("a b").unwrap(), 4);
        assert_eq!(runtime.count_embedding_tokens("hello  world").unwrap(), 5);
        assert_eq!(runtime.count_embedding_tokens("a\nb").unwrap(), 5);
    }

    #[test]
    fn gguf_embedded_tokenizer_add_space_prefix_matches_llama_spm_fragments() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("model.gguf");
        let mut fixture = default_test_tokens();
        fixture.add_space_prefix = true;
        write_test_gguf(&path, fixture).expect("write test gguf");

        let runtime = LlamaSpmGgufTokenizerRuntime::from_path(&path).expect("load tokenizer");

        assert_eq!(runtime.count_embedding_tokens("").unwrap(), 2);
        assert_eq!(runtime.count_embedding_tokens("hello").unwrap(), 4);
        assert_eq!(runtime.count_embedding_tokens("  hello").unwrap(), 5);
    }

    #[test]
    fn gguf_embedded_tokenizer_rejects_unsupported_tokenizer_model() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("model.gguf");
        let mut fixture = default_test_tokens();
        fixture.model = "gpt2";
        write_test_gguf(&path, fixture).expect("write test gguf");

        let err = LlamaSpmGgufTokenizerRuntime::from_path(&path)
            .expect_err("unsupported tokenizer should fail");
        assert!(
            err.to_string()
                .contains("unsupported GGUF tokenizer model 'gpt2'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gguf_embedded_tokenizer_uses_llama_spm_defaults_when_special_flags_are_missing() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("model.gguf");
        let mut fixture = default_test_tokens();
        fixture.include_special_flags = false;
        write_test_gguf(&path, fixture).expect("write test gguf");

        let runtime = LlamaSpmGgufTokenizerRuntime::from_path(&path).expect("load tokenizer");

        assert_eq!(runtime.count_embedding_tokens("").unwrap(), 1);
        assert_eq!(runtime.count_embedding_tokens("hello").unwrap(), 3);
    }

    #[test]
    fn gguf_embedded_tokenizer_rejects_missing_scores() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("model.gguf");
        let mut fixture = default_test_tokens();
        fixture.include_scores = false;
        write_test_gguf(&path, fixture).expect("write test gguf");

        let err =
            LlamaSpmGgufTokenizerRuntime::from_path(&path).expect_err("missing scores should fail");
        assert!(
            err.to_string().contains("missing tokenizer.ggml.scores"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gguf_embedded_tokenizer_rejects_missing_token_types() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("model.gguf");
        let mut fixture = default_test_tokens();
        fixture.include_token_types = false;
        write_test_gguf(&path, fixture).expect("write test gguf");

        let err = LlamaSpmGgufTokenizerRuntime::from_path(&path)
            .expect_err("missing token types should fail");
        assert!(
            err.to_string()
                .contains("missing tokenizer.ggml.token_type"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_byte_token_accepts_hex_byte_tokens() {
        assert_eq!(parse_byte_token("<0x0A>"), Some(b'\n'));
        assert_eq!(parse_byte_token("<0xff>"), Some(255));
        assert_eq!(parse_byte_token("<unk>"), None);
    }

    #[test]
    fn managed_embeddinggemma_counts_match_recorded_llama_tokenize_cases_when_requested() {
        let Ok(path) = std::env::var("KBOLT_GGUF_TOKENIZER_SMOKE_MODEL") else {
            return;
        };
        let path = Path::new(&path);

        let runtime = LlamaSpmGgufTokenizerRuntime::from_path(path).expect("load tokenizer");
        let cases = [
            ("", 2),
            ("hello world", 4),
            ("the quick brown fox jumps over the lazy dog", 11),
            ("kbolt doctor tokenize smoke", 7),
            ("fn main() { println!(\"hi\"); }", 11),
            ("こんにちは世界", 4),
            ("a\nb", 5),
            ("a  b", 5),
            (" hello", 3),
            ("  hello", 4),
            ("https://example.com/foo?bar=baz", 13),
            ("https://example.com/foo?bar=baz&x=1#frag", 19),
            ("src/models/tokenizer.rs", 9),
            ("foo/bar_baz-qux.test", 12),
            ("hello, world!", 6),
            ("emoji 😀 test", 5),
            ("中文 mixed English 123", 9),
            ("naïve café résumé", 7),
            ("spaces   three", 5),
            ("multiple   spaces   here", 7),
            ("line1\nline2\nline3", 10),
            ("line1\r\nline2", 8),
            ("tab\tindent", 5),
            ("snake_case::path_name", 9),
            ("HTTPRequest::new(url)?;", 8),
            ("\"quoted\" `code` [link](url)", 13),
            ("0xdeadbeef 1234567890 3.14159", 25),
            ("<bos> literal", 6),
            ("<start_of_turn>user\nhi<end_of_turn>", 19),
        ];
        for (text, expected) in cases {
            assert_eq!(
                runtime.count_embedding_tokens(text).unwrap(),
                expected,
                "unexpected token count for {text:?}"
            );
        }
    }

    #[test]
    fn managed_embeddinggemma_counts_match_llama_tokenize_binary_when_requested() {
        if std::env::var("KBOLT_GGUF_TOKENIZER_COMPARE_LLAMA_TOKENIZE").is_err() {
            return;
        }
        let Ok(path) = std::env::var("KBOLT_GGUF_TOKENIZER_SMOKE_MODEL") else {
            return;
        };
        let path = Path::new(&path);

        let runtime = LlamaSpmGgufTokenizerRuntime::from_path(path).expect("load tokenizer");
        for text in generated_llama_tokenize_compare_cases() {
            let expected = llama_tokenize_count(path, &text);
            let actual = runtime.count_embedding_tokens(&text).unwrap();
            assert_eq!(actual, expected, "unexpected token count for {text:?}");
        }
    }

    #[derive(Clone)]
    struct TestGgufTokens {
        model: &'static str,
        tokens: Vec<(&'static str, f32, i32)>,
        bos_id: u32,
        eos_id: u32,
        unk_id: u32,
        add_bos: bool,
        add_eos: bool,
        add_space_prefix: bool,
        include_scores: bool,
        include_token_types: bool,
        include_special_flags: bool,
    }

    fn default_test_tokens() -> TestGgufTokens {
        TestGgufTokens {
            model: "llama",
            tokens: vec![
                ("<eos>", -1000.0, TOKEN_TYPE_CONTROL),
                ("<bos>", -1000.0, TOKEN_TYPE_CONTROL),
                ("<unk>", -1000.0, TOKEN_TYPE_CONTROL),
                ("▁", 10.0, TOKEN_TYPE_NORMAL),
                ("h", 10.0, TOKEN_TYPE_NORMAL),
                ("e", 10.0, TOKEN_TYPE_NORMAL),
                ("l", 10.0, TOKEN_TYPE_NORMAL),
                ("o", 10.0, TOKEN_TYPE_NORMAL),
                ("he", 10.0, TOKEN_TYPE_NORMAL),
                ("hel", 10.0, TOKEN_TYPE_NORMAL),
                ("hell", 10.0, TOKEN_TYPE_NORMAL),
                ("hello", 10.0, TOKEN_TYPE_NORMAL),
                ("w", 10.0, TOKEN_TYPE_NORMAL),
                ("r", 10.0, TOKEN_TYPE_NORMAL),
                ("d", 10.0, TOKEN_TYPE_NORMAL),
                ("wo", 10.0, TOKEN_TYPE_NORMAL),
                ("wor", 10.0, TOKEN_TYPE_NORMAL),
                ("worl", 10.0, TOKEN_TYPE_NORMAL),
                ("▁world", 10.0, TOKEN_TYPE_NORMAL),
                ("world", 8.0, TOKEN_TYPE_NORMAL),
                ("▁w", 10.0, TOKEN_TYPE_NORMAL),
                ("▁wo", 10.0, TOKEN_TYPE_NORMAL),
                ("▁wor", 10.0, TOKEN_TYPE_NORMAL),
                ("▁worl", 10.0, TOKEN_TYPE_NORMAL),
                ("  ", 10.0, TOKEN_TYPE_USER_DEFINED),
                ("a", 10.0, TOKEN_TYPE_NORMAL),
                ("▁b", 10.0, TOKEN_TYPE_NORMAL),
                ("b", 8.0, TOKEN_TYPE_NORMAL),
                ("\n", 10.0, TOKEN_TYPE_USER_DEFINED),
            ],
            bos_id: 1,
            eos_id: 0,
            unk_id: 2,
            add_bos: true,
            add_eos: true,
            add_space_prefix: false,
            include_scores: true,
            include_token_types: true,
            include_special_flags: true,
        }
    }

    fn write_test_gguf(path: &Path, fixture: TestGgufTokens) -> Result<()> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        write_u32(&mut bytes, GGUF_SUPPORTED_VERSION)?;
        write_u64(&mut bytes, 0)?;
        let metadata_count = 6
            + u64::from(fixture.include_scores)
            + u64::from(fixture.include_token_types)
            + if fixture.include_special_flags { 3 } else { 0 };
        write_u64(&mut bytes, metadata_count)?;

        write_kv_string(&mut bytes, "tokenizer.ggml.model", fixture.model)?;
        write_kv_string(&mut bytes, "tokenizer.ggml.pre", "default")?;
        write_kv_string_array(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &fixture
                .tokens
                .iter()
                .map(|(token, _, _)| *token)
                .collect::<Vec<_>>(),
        )?;
        if fixture.include_scores {
            write_kv_f32_array(
                &mut bytes,
                "tokenizer.ggml.scores",
                &fixture
                    .tokens
                    .iter()
                    .map(|(_, score, _)| *score)
                    .collect::<Vec<_>>(),
            )?;
        }
        if fixture.include_token_types {
            write_kv_i32_array(
                &mut bytes,
                "tokenizer.ggml.token_type",
                &fixture
                    .tokens
                    .iter()
                    .map(|(_, _, token_type)| *token_type)
                    .collect::<Vec<_>>(),
            )?;
        }
        write_kv_u32(&mut bytes, "tokenizer.ggml.bos_token_id", fixture.bos_id)?;
        write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", fixture.eos_id)?;
        write_kv_u32(
            &mut bytes,
            "tokenizer.ggml.unknown_token_id",
            fixture.unk_id,
        )?;
        if fixture.include_special_flags {
            write_kv_bool(&mut bytes, "tokenizer.ggml.add_bos_token", fixture.add_bos)?;
            write_kv_bool(&mut bytes, "tokenizer.ggml.add_eos_token", fixture.add_eos)?;
            write_kv_bool(
                &mut bytes,
                "tokenizer.ggml.add_space_prefix",
                fixture.add_space_prefix,
            )?;
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    fn write_kv_string(out: &mut Vec<u8>, key: &str, value: &str) -> Result<()> {
        write_string(out, key)?;
        write_u32(out, GGUF_TYPE_STRING)?;
        write_string(out, value)
    }

    fn write_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) -> Result<()> {
        write_string(out, key)?;
        write_u32(out, GGUF_TYPE_UINT32)?;
        write_u32(out, value)
    }

    fn write_kv_bool(out: &mut Vec<u8>, key: &str, value: bool) -> Result<()> {
        write_string(out, key)?;
        write_u32(out, GGUF_TYPE_BOOL)?;
        out.push(u8::from(value));
        Ok(())
    }

    fn write_kv_string_array(out: &mut Vec<u8>, key: &str, values: &[&str]) -> Result<()> {
        write_string(out, key)?;
        write_u32(out, GGUF_TYPE_ARRAY)?;
        write_u32(out, GGUF_TYPE_STRING)?;
        write_u64(out, values.len() as u64)?;
        for value in values {
            write_string(out, value)?;
        }
        Ok(())
    }

    fn write_kv_f32_array(out: &mut Vec<u8>, key: &str, values: &[f32]) -> Result<()> {
        write_string(out, key)?;
        write_u32(out, GGUF_TYPE_ARRAY)?;
        write_u32(out, GGUF_TYPE_FLOAT32)?;
        write_u64(out, values.len() as u64)?;
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    fn write_kv_i32_array(out: &mut Vec<u8>, key: &str, values: &[i32]) -> Result<()> {
        write_string(out, key)?;
        write_u32(out, GGUF_TYPE_ARRAY)?;
        write_u32(out, GGUF_TYPE_INT32)?;
        write_u64(out, values.len() as u64)?;
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    fn write_string(out: &mut Vec<u8>, value: &str) -> Result<()> {
        write_u64(out, value.len() as u64)?;
        out.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn write_u32(out: &mut Vec<u8>, value: u32) -> Result<()> {
        out.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn write_u64(out: &mut Vec<u8>, value: u64) -> Result<()> {
        out.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn generated_llama_tokenize_compare_cases() -> Vec<String> {
        let pieces = [
            "",
            "a",
            "hello",
            "world",
            "the quick brown fox",
            "kbolt",
            "tokenizer",
            "src/models/tokenizer.rs",
            "https://example.com/foo?bar=baz",
            "fn main() { println!(\"hi\"); }",
            "こんにちは世界",
            "中文 mixed English 123",
            "emoji 😀 test",
            "naïve café résumé",
            "<bos> literal",
            "<start_of_turn>user\nhi<end_of_turn>",
            "line1\nline2",
            "tab\tindent",
            "multiple   spaces",
            "0xdeadbeef 1234567890",
        ];
        let separators = ["", " ", "  ", "\n", "\t", "/", "::", "-", "_", "."];
        let mut cases = Vec::new();
        for piece in pieces {
            cases.push(piece.to_string());
        }
        for left in pieces {
            for right in pieces {
                if cases.len() >= 200 {
                    return cases;
                }
                let separator = separators[(left.len() + right.len()) % separators.len()];
                cases.push(format!("{left}{separator}{right}"));
            }
        }
        cases
    }

    fn llama_tokenize_count(model_path: &Path, text: &str) -> usize {
        let output = Command::new("llama-tokenize")
            .arg("--model")
            .arg(model_path)
            .arg("--prompt")
            .arg(text)
            .arg("--show-count")
            .arg("--ids")
            .arg("--no-parse-special")
            .arg("--log-disable")
            .output()
            .expect("run llama-tokenize");
        assert!(
            output.status.success(),
            "llama-tokenize failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("llama-tokenize stdout utf8");
        stdout
            .lines()
            .find_map(|line| {
                line.strip_prefix("Total number of tokens: ")
                    .and_then(|count| count.parse::<usize>().ok())
            })
            .unwrap_or_else(|| panic!("missing token count in llama-tokenize output: {stdout}"))
    }
}
