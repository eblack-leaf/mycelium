// =============================================================================
// candidate_matcher.rs — Stage 2: Cross-encoder candidate matching
//
// Uses MiniLM as cross-encoder to score (phrase, schema_name) pairs.
// No role assignment — every linguistic node scored against all schema nodes.
// =============================================================================

use serde::{Serialize, Deserialize};
use crate::graph::SchemaGraph;
use crate::nlp::{NlpModel, LinguisticGraph};
use crate::operations::OpNode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateEdge {
    pub linguistic_node: usize,
    pub schema_node_type: String,
    pub schema_node_id: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateSet {
    pub edges: Vec<CandidateEdge>,
}

#[derive(Debug, Clone)]
pub struct CandidateMatcherConfig {
    pub top_k: usize,
    pub min_score: f32,
}

impl Default for CandidateMatcherConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: 0.3,
        }
    }
}

/// Schema node names for cross-encoder pairing.
pub struct SchemaNames {
    pub table_names: Vec<String>,
    pub field_names: Vec<String>,
    pub operation_names: Vec<String>,
}

pub struct CandidateMatcher {
    schema_names: SchemaNames,
    config: CandidateMatcherConfig,
}

impl CandidateMatcher {
    pub fn new(
        schema_graph: &SchemaGraph,
        operations: &[OpNode],
        config: CandidateMatcherConfig,
    ) -> Self {
        let table_names: Vec<String> = schema_graph.table_nodes.iter()
            .map(|n| n.name.clone())
            .collect();

        let field_names: Vec<String> = schema_graph.field_nodes.iter()
            .map(|n| n.name.clone())
            .collect();

        let operation_names: Vec<String> = operations.iter()
            .map(|op| op.name.clone())
            .collect();

        Self {
            schema_names: SchemaNames { table_names, field_names, operation_names },
            config,
        }
    }

    /// Score all linguistic nodes against all schema nodes.
    pub fn match_candidates(
        &self,
        nlp: &NlpModel,
        ling_graph: &LinguisticGraph,
    ) -> CandidateSet {
        let mut edges = Vec::new();

        for node in &ling_graph.nodes {
            // Score against tables
            let table_scores: Vec<f32> = self.schema_names.table_names.iter()
                .map(|name| nlp.cross_encode(&node.text, name))
                .collect();
            self.collect_top_k(
                &mut edges, node.id, "table", &table_scores,
            );

            // Score against fields
            let field_scores: Vec<f32> = self.schema_names.field_names.iter()
                .map(|name| nlp.cross_encode(&node.text, name))
                .collect();
            self.collect_top_k(
                &mut edges, node.id, "field", &field_scores,
            );

            // Score against operations
            let op_scores: Vec<f32> = self.schema_names.operation_names.iter()
                .map(|name| nlp.cross_encode(&node.text, name))
                .collect();
            self.collect_top_k(
                &mut edges, node.id, "operation", &op_scores,
            );
        }

        CandidateSet { edges }
    }

    fn collect_top_k(
        &self,
        edges: &mut Vec<CandidateEdge>,
        ling_node: usize,
        schema_type: &str,
        scores: &[f32],
    ) {
        // Sort by score descending, take top-k above threshold
        let mut indexed: Vec<(usize, f32)> = scores.iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (schema_id, score) in indexed.into_iter().take(self.config.top_k) {
            if score < self.config.min_score { break; }
            edges.push(CandidateEdge {
                linguistic_node: ling_node,
                schema_node_type: schema_type.to_string(),
                schema_node_id: schema_id,
                score,
            });
        }
    }
}
