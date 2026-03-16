// =============================================================================
// intent.rs — Model-agnostic extraction types consumed by QueryGraph
//
// Any NL model (token classifier, encoder-decoder, etc.) produces an
// Extraction. QueryGraph consumes it without knowing how it was built.
// =============================================================================

/// Tentative link between a candidate and a schema element.
#[derive(Debug, Clone)]
pub struct SchemaMatch {
    pub schema_node_type: String,
    pub schema_node_id: usize,
    pub score: f32,
}

/// A candidate that matched a schema entity (collection, field, or traversal target).
#[derive(Debug, Clone)]
pub struct CandidateMatch {
    pub surface_form: String,
    pub confidence: f32,
    pub schema_matches: Vec<SchemaMatch>,
}

/// A candidate filter: field reference + operator + value.
#[derive(Debug, Clone)]
pub struct FilterMatch {
    pub field: CandidateMatch,
    pub operator: String,
    pub value: String,
    pub confidence: f32,
}

/// Model-agnostic output from NL intent extraction.
#[derive(Debug, Clone)]
pub struct Extraction {
    pub collections: Vec<CandidateMatch>,
    pub fields: Vec<CandidateMatch>,
    pub filters: Vec<FilterMatch>,
    pub traversals: Vec<CandidateMatch>,
}
