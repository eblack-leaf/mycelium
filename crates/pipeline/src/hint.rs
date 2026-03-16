use serde::{Deserialize, Serialize};
use std::ops::Range;

use crate::condition::Op;

/// A span in the original input text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub range: Range<usize>,
    pub text: String,
}

/// Everything the spores produce for the orchestrator model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnnotatedInput {
    pub original: String,
    pub hints: Vec<Hint>,
}

/// A single annotation from a spore — a suggestion, not a decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hint {
    pub kind: HintKind,
    pub span: Span,
    /// Activation strength from the spore's conv filters.
    /// Lower confidence on typos comes naturally from partial n-gram overlap.
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HintKind {
    /// Compound phrase matched: expands to a known condition.
    Phrase { field: String, op: Op, value: String },

    /// Field name or alias detected at this span.
    Field { field: String },

    /// Operator phrase detected at this span.
    Op { op: Op },

    /// Temporal expression candidate at this span.
    Temporal { marker: String },

    /// Numeric literal found at this span.
    Numeric { value: f64 },
}
