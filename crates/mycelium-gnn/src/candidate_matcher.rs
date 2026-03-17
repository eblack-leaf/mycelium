// =============================================================================
// candidate_matcher.rs — Semantic similarity scoring (phrase ↔ schema nodes)
//
// Takes phrase embeddings from NlpModel + precomputed schema node embeddings.
// Produces scored candidate edges: each linguistic node → top-k schema nodes.
//
// No role assignment — a noun phrase gets scored against ALL schema node types
// (tables, fields, operations). The GNN decides what role each phrase plays.
// =============================================================================

use crate::graph::SchemaGraph;
use crate::nlp::LinguisticGraph;
use crate::operations::OpNode;

/// Precomputed schema embeddings for fast similarity lookup.
pub struct SchemaEmbeddings {
    /// Table name embeddings [n_tables, embed_dim]
    pub table_embeds: Vec<Vec<f32>>,
    /// Field name embeddings [n_fields, embed_dim]
    /// Embeds "table_name.field_name" or just "field_name" — TBD
    pub field_embeds: Vec<Vec<f32>>,
    /// Operation name embeddings [n_ops, embed_dim]
    pub operation_embeds: Vec<Vec<f32>>,
    pub embed_dim: usize,
}

/// A scored candidate link from a linguistic node to a schema node.
#[derive(Debug, Clone)]
pub struct CandidateEdge {
    /// Index into LinguisticGraph.nodes
    pub linguistic_node: usize,
    /// "table", "field", or "operation"
    pub schema_node_type: String,
    /// Index into that node type's list
    pub schema_node_id: usize,
    /// Cosine similarity score
    pub score: f32,
}

/// All candidate edges for a query — input to the GNN.
#[derive(Debug, Clone)]
pub struct CandidateSet {
    pub edges: Vec<CandidateEdge>,
}

pub struct CandidateMatcherConfig {
    /// Max candidates per linguistic node per schema type
    pub top_k: usize,
    /// Minimum similarity threshold (skip very low matches)
    pub min_score: f32,
}

impl Default for CandidateMatcherConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: 0.1,
        }
    }
}

pub struct CandidateMatcher {
    schema_embeds: SchemaEmbeddings,
    config: CandidateMatcherConfig,
}

impl CandidateMatcher {
    /// Build from schema + NLP model. Precomputes schema name embeddings.
    pub fn new(
        _schema_graph: &SchemaGraph,
        _operations: &[OpNode],
        _schema_embeds: SchemaEmbeddings,
        config: CandidateMatcherConfig,
    ) -> Self {
        todo!("store schema embeddings for lookup")
    }

    /// Precompute schema node name embeddings using the NLP model.
    /// Call once at startup.
    pub fn embed_schema(
        _schema_graph: &SchemaGraph,
        _operations: &[OpNode],
        // nlp_model: &NlpModel,
    ) -> SchemaEmbeddings {
        todo!("embed all schema node names via transformer")
        // For tables: embed table name ("users", "products", ...)
        // For fields: embed "table.field" or just field name — needs experimentation
        // For operations: embed operation name + description
    }

    /// Score all linguistic nodes against all schema nodes.
    /// Returns candidate edges above threshold, limited to top-k per type.
    pub fn match_candidates(
        &self,
        _ling_graph: &LinguisticGraph,
        _phrase_embeddings: &[Vec<f32>],
    ) -> CandidateSet {
        todo!("cosine similarity + top-k filtering")
        // For each linguistic node:
        //   1. Compute cosine sim against all table embeddings → take top-k
        //   2. Compute cosine sim against all field embeddings → take top-k
        //   3. Compute cosine sim against all operation embeddings → take top-k
        //   4. Filter by min_score
        //   5. Collect into CandidateEdge list
    }
}

/// Cosine similarity between two vectors.
fn _cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
