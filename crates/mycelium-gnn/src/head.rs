// =============================================================================
// head.rs — Output head: resolves candidates to schema entities + operations
//
// Reads final embeddings from the encoder, scores candidates against schema
// nodes and operations, produces a ResolvedGraph.
// =============================================================================

use std::collections::HashMap;
use burn::tensor::{backend::Backend, Tensor};
use crate::query_graph::QueryGraph;
use crate::operations::OpNode;

/// A resolved candidate — the output head's decision.
#[derive(Debug, Clone)]
pub struct Resolution {
    pub candidate_type: String,
    pub candidate_id: usize,
    pub resolved_type: String,
    pub resolved_id: usize,
    pub score: f32,
}

/// Fully resolved query graph — ready for the orchestrator.
#[derive(Debug, Clone)]
pub struct ResolvedGraph {
    pub collection: Option<Resolution>,
    pub fields: Vec<Resolution>,
    pub filters: Vec<ResolvedFilter>,
    pub traversals: Vec<Resolution>,
    pub modifiers: Vec<ResolvedModifier>,
}

#[derive(Debug, Clone)]
pub struct ResolvedFilter {
    pub field: Resolution,
    pub operation: Resolution,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedModifier {
    pub operation: Resolution,
    pub value: String,
}

pub struct OutputHead {
    // TODO: learned scoring layers
}

impl OutputHead {
    pub fn new<B: Backend>(_hidden_dim: usize, _device: &B::Device) -> Self {
        todo!()
    }

    pub fn resolve<B: Backend>(
        &self,
        _embeddings: &HashMap<String, Tensor<B, 2>>,
        _query_graph: &QueryGraph,
        _operations: &[OpNode],
    ) -> ResolvedGraph {
        todo!()
    }
}
