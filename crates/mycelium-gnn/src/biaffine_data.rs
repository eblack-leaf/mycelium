// =============================================================================
// biaffine_data.rs — Dataset types + subword alignment for the biaffine head
//
// Handles:
//   - BIO tag definition (9 classes)
//   - Subword-to-word alignment (WordPiece offsets → word indices)
//   - BIO tag assignment from ground-truth spans
//   - Serializable dataset for training
// =============================================================================

use serde::{Serialize, Deserialize};
use crate::nlp::SpanType;

// =============================================================================
// BIO tags
// =============================================================================

/// BIO tag set for sequence labeling. 9 classes total.
/// Order matches classifier output: O=0, B-Intent=1, ..., I-Quant=8
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum BioTag {
    O       = 0,
    BIntent = 1,
    IIntent = 2,
    BNP     = 3,
    INP     = 4,
    BComp   = 5,
    IComp   = 6,
    BQuant  = 7,
    IQuant  = 8,
}

impl BioTag {
    pub const COUNT: usize = 9;

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::O,
            1 => Self::BIntent,
            2 => Self::IIntent,
            3 => Self::BNP,
            4 => Self::INP,
            5 => Self::BComp,
            6 => Self::IComp,
            7 => Self::BQuant,
            8 => Self::IQuant,
            _ => Self::O,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    /// Get the B-tag for a span type.
    pub fn b_tag(span_type: SpanType) -> Self {
        match span_type {
            SpanType::Intent     => Self::BIntent,
            SpanType::NounPhrase => Self::BNP,
            SpanType::Comparator => Self::BComp,
            SpanType::Quantifier => Self::BQuant,
        }
    }

    /// Get the I-tag for a span type.
    pub fn i_tag(span_type: SpanType) -> Self {
        match span_type {
            SpanType::Intent     => Self::IIntent,
            SpanType::NounPhrase => Self::INP,
            SpanType::Comparator => Self::IComp,
            SpanType::Quantifier => Self::IQuant,
        }
    }

    /// Decode a BIO tag back to its span type (if not O).
    pub fn span_type(self) -> Option<SpanType> {
        match self {
            Self::BIntent | Self::IIntent => Some(SpanType::Intent),
            Self::BNP     | Self::INP     => Some(SpanType::NounPhrase),
            Self::BComp   | Self::IComp   => Some(SpanType::Comparator),
            Self::BQuant  | Self::IQuant  => Some(SpanType::Quantifier),
            Self::O => None,
        }
    }

    /// Is this a B- (begin) tag?
    pub fn is_begin(self) -> bool {
        matches!(self, Self::BIntent | Self::BNP | Self::BComp | Self::BQuant)
    }
}

// =============================================================================
// Subword-to-word alignment
// =============================================================================

/// Map subword token indices to whitespace word indices.
///
/// `offsets` are `(char_start, char_end)` from the tokenizer's `get_offsets()`.
/// Special tokens (CLS/SEP) have offset `(0, 0)` and are skipped.
/// Returns a vec of length `n_content_tokens` (excluding CLS/SEP) where each
/// entry is the word index that subword belongs to.
pub fn build_subword_to_word(offsets: &[(usize, usize)], text: &str) -> Vec<usize> {
    // Precompute word boundaries from whitespace splitting
    let mut word_starts: Vec<usize> = Vec::new();
    let mut word_ends: Vec<usize> = Vec::new();
    let mut in_word = false;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if in_word {
                word_ends.push(i);
                in_word = false;
            }
        } else if !in_word {
            word_starts.push(i);
            in_word = true;
        }
    }
    if in_word {
        word_ends.push(text.len());
    }

    let mut mapping = Vec::new();
    // Skip CLS (index 0) and SEP (last index)
    for &(char_start, char_end) in offsets.iter().skip(1) {
        if char_start == 0 && char_end == 0 {
            // SEP or padding — skip
            continue;
        }
        // Find which word this subword's char_start falls into
        let word_idx = word_starts.iter()
            .zip(word_ends.iter())
            .position(|(&ws, &we)| char_start >= ws && char_start < we)
            .unwrap_or(0);
        mapping.push(word_idx);
    }

    mapping
}

/// Assign BIO tags to subword tokens given ground-truth word-level spans.
///
/// `subword_to_word`: mapping from subword index to word index (from `build_subword_to_word`)
/// `spans`: (start_word, end_word_exclusive, SpanType) — word-level span boundaries
/// `seq_len`: number of content subword tokens (excluding CLS/SEP)
///
/// Returns a vec of BioTag indices (as usize), one per subword token.
pub fn assign_bio_tags(
    subword_to_word: &[usize],
    spans: &[(usize, usize, SpanType)],
    seq_len: usize,
) -> Vec<usize> {
    let mut tags = vec![BioTag::O.index(); seq_len];

    for &(start_word, end_word, span_type) in spans {
        let b_tag = BioTag::b_tag(span_type).index();
        let i_tag = BioTag::i_tag(span_type).index();
        let mut first = true;

        for (sw_idx, &word_idx) in subword_to_word.iter().enumerate() {
            if sw_idx >= seq_len { break; }
            if word_idx >= start_word && word_idx < end_word {
                if first {
                    tags[sw_idx] = b_tag;
                    first = false;
                } else {
                    tags[sw_idx] = i_tag;
                }
            }
        }
    }

    tags
}

// =============================================================================
// Dataset types
// =============================================================================

/// A single training sample for the biaffine head.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiaffineSample {
    /// Original NL query text.
    pub query: String,
    /// BIO tag indices per subword token (excluding CLS/SEP).
    pub bio_tags: Vec<usize>,
    /// Subword-to-word mapping (excluding CLS/SEP).
    pub subword_to_word: Vec<usize>,
    /// Decoded span boundaries: (start_word, end_word_exclusive, span_type_index).
    /// span_type_index: 0=NP, 1=Quant, 2=Comp, 3=Intent (SpanType order).
    pub span_boundaries: Vec<(usize, usize, usize)>,
    /// Dependency arcs: (src_span_idx, dst_span_idx, relation_index).
    /// relation_index: 0=Possessive, 1=Quantifies, 2=Comparison, 3=IntentTarget.
    pub arcs: Vec<(usize, usize, usize)>,
}

/// Full dataset of biaffine training samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiaffineDataset {
    pub samples: Vec<BiaffineSample>,
}

impl BiaffineDataset {
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let dataset: Self = serde_json::from_str(&data)?;
        Ok(dataset)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let data = serde_json::to_string(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}

// =============================================================================
// BIO → span decoding
// =============================================================================

/// Decoded span from BIO tags: word-level boundaries + type.
#[derive(Debug, Clone)]
pub struct DecodedSpan {
    pub start_word: usize,
    pub end_word: usize,  // exclusive
    pub span_type: SpanType,
}

/// Decode BIO tag sequence into spans, mapping through subword-to-word alignment.
///
/// Greedy left-to-right: a B-tag starts a new span, I-tags of the same type extend it,
/// anything else closes the current span.
pub fn decode_bio_spans(bio_tags: &[usize], subword_to_word: &[usize]) -> Vec<DecodedSpan> {
    let mut spans = Vec::new();
    let mut current: Option<(usize, usize, SpanType)> = None; // (start_word, end_word, type)

    for (i, &tag_idx) in bio_tags.iter().enumerate() {
        let tag = BioTag::from_index(tag_idx);
        let word_idx = subword_to_word.get(i).copied().unwrap_or(0);

        if tag.is_begin() {
            // Close previous span
            if let Some((s, e, t)) = current.take() {
                spans.push(DecodedSpan { start_word: s, end_word: e, span_type: t });
            }
            // Start new span
            let st = tag.span_type().unwrap();
            current = Some((word_idx, word_idx + 1, st));
        } else if tag == BioTag::O {
            // Close current span
            if let Some((s, e, t)) = current.take() {
                spans.push(DecodedSpan { start_word: s, end_word: e, span_type: t });
            }
        } else {
            // I-tag: extend if matching type, otherwise close and start new
            let tag_st = tag.span_type().unwrap();
            if let Some((_, ref mut end, ref cur_type)) = current {
                if *cur_type == tag_st {
                    *end = word_idx + 1;
                } else {
                    let (s, e, t) = current.take().unwrap();
                    spans.push(DecodedSpan { start_word: s, end_word: e, span_type: t });
                    // Orphan I-tag — treat as B
                    current = Some((word_idx, word_idx + 1, tag_st));
                }
            } else {
                // Orphan I-tag — treat as B
                current = Some((word_idx, word_idx + 1, tag_st));
            }
        }
    }

    // Close last span
    if let Some((s, e, t)) = current {
        spans.push(DecodedSpan { start_word: s, end_word: e, span_type: t });
    }

    spans
}

/// Convert SpanType to index used in span_boundaries serialization.
pub fn span_type_to_index(st: SpanType) -> usize {
    match st {
        SpanType::NounPhrase => 0,
        SpanType::Quantifier => 1,
        SpanType::Comparator => 2,
        SpanType::Intent     => 3,
    }
}

/// Convert index back to SpanType.
pub fn index_to_span_type(i: usize) -> SpanType {
    match i {
        0 => SpanType::NounPhrase,
        1 => SpanType::Quantifier,
        2 => SpanType::Comparator,
        3 => SpanType::Intent,
        _ => SpanType::NounPhrase,
    }
}
