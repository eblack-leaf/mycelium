// =============================================================================
// intent.rs — Model-agnostic extraction types consumed by QueryGraph
//
// Any NL model (token classifier, encoder-decoder, etc.) produces an
// Extraction. QueryGraph consumes it without knowing how it was built.
// =============================================================================

use serde::{Serialize, Deserialize};

/// Tentative link between a candidate and a schema element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMatch {
    pub schema_node_type: String,
    pub schema_node_id: usize,
    pub score: f32,
}

/// A candidate that matched a schema entity (collection, field, or traversal target).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateMatch {
    pub surface_form: String,
    pub confidence: f32,
    pub schema_matches: Vec<SchemaMatch>,
    pub operation_matches: Vec<OperationMatch>,
}

/// Scored guess linking a filter phrase to an operation node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMatch {
    pub operation_id: usize,   // index into all_operations()
    pub score: f32,            // Grounding model's confidence
}

/// A candidate filter: field reference + operator + value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterMatch {
    pub field: CandidateMatch,
    pub operator: String,
    pub value: String,
    pub confidence: f32,
    pub operation_matches: Vec<OperationMatch>,
}

/// A query-level modifier (LIMIT, OFFSET, etc.) — not applied to a field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifierMatch {
    pub surface_form: String,
    pub value: String,
    pub confidence: f32,
    pub operation_matches: Vec<OperationMatch>,
}

/// Model-agnostic output from NL intent extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extraction {
    pub collections: Vec<CandidateMatch>,
    pub fields: Vec<CandidateMatch>,
    pub filters: Vec<FilterMatch>,
    pub traversals: Vec<CandidateMatch>,
    pub modifiers: Vec<ModifierMatch>,
}
