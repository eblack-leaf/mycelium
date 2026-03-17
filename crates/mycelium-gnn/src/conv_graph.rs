// =============================================================================
// conv_graph.rs — Shared types for message-passing topology
// =============================================================================

/// A single relation type: source node type → destination node type.
#[derive(Debug, Clone)]
pub struct ConvRelation {
    pub src_type: String,
    pub edge_type: String,
    pub dst_type: String,
    pub src_indices: Vec<usize>,
    pub dst_indices: Vec<usize>,
}
