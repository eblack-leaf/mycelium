// =============================================================================
// head.rs — Output head: bilinear scoring of query candidates against targets
//
// Each query candidate type gets scored against its target node type:
//   q_collection → table, q_field → field, q_filter → operation,
//   q_traversal → table, q_modifier → operation
//
// score(q, t) = proj(q) · tᵀ   (learned bilinear via a Linear projection)
//
// Two modes:
//   score_logits() — returns raw tensors for training (differentiable)
//   resolve()      — argmax into discrete Resolutions for inference
// =============================================================================

use std::collections::HashMap;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
    tensor::activation,
};
use crate::query_graph::QueryGraph;
use crate::operations::OpNode;

// =============================================================================
// Inference output types
// =============================================================================

#[derive(Debug, Clone)]
pub struct Resolution {
    pub candidate_type: String,
    pub candidate_id: usize,
    pub resolved_type: String,
    pub resolved_id: usize,
    pub score: f32,
}

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

// =============================================================================
// Training output — raw logit tensors (differentiable)
// =============================================================================

pub struct ScoreLogits<B: Backend> {
    /// [n_q_collections, n_tables]
    pub collection: Option<Tensor<B, 2>>,
    /// [n_q_fields, n_schema_fields]
    pub field: Option<Tensor<B, 2>>,
    /// [n_q_filters, n_operations]
    pub filter_op: Option<Tensor<B, 2>>,
    /// [n_q_traversals, n_tables]
    pub traversal: Option<Tensor<B, 2>>,
    /// [n_q_modifiers, n_operations]
    pub modifier_op: Option<Tensor<B, 2>>,
}

// =============================================================================
// Output head
// =============================================================================

/// One learned projection per candidate→target scoring pair.
#[derive(Module, Debug)]
pub struct OutputHead<B: Backend> {
    collection_proj: Linear<B>,
    field_proj: Linear<B>,
    filter_proj: Linear<B>,
    traversal_proj: Linear<B>,
    modifier_proj: Linear<B>,
}

impl<B: Backend> OutputHead<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            collection_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            field_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            filter_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            traversal_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            modifier_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
        }
    }

    /// Raw logit scores for each candidate type. Used by training loss.
    pub fn score_logits(
        &self,
        embeddings: &HashMap<String, Tensor<B, 2>>,
    ) -> ScoreLogits<B> {
        ScoreLogits {
            collection: bilinear_score(embeddings, "q_collection", "table", &self.collection_proj),
            field: bilinear_score(embeddings, "q_field", "field", &self.field_proj),
            filter_op: bilinear_score(embeddings, "q_filter", "operation", &self.filter_proj),
            traversal: bilinear_score(embeddings, "q_traversal", "table", &self.traversal_proj),
            modifier_op: bilinear_score(embeddings, "q_modifier", "operation", &self.modifier_proj),
        }
    }

    /// Inference: argmax logits into discrete resolutions.
    pub fn resolve(
        &self,
        embeddings: &HashMap<String, Tensor<B, 2>>,
        query_graph: &QueryGraph,
        _operations: &[OpNode],
    ) -> ResolvedGraph {
        let logits = self.score_logits(embeddings);

        let collection = logits.collection.map(|l| {
            let (indices, scores) = argmax_with_scores(&l);
            // First collection candidate → best table
            Resolution {
                candidate_type: "collection".into(),
                candidate_id: 0,
                resolved_type: "table".into(),
                resolved_id: indices[0],
                score: scores[0],
            }
        });

        let fields: Vec<Resolution> = logits.field.map(|l| {
            let (indices, scores) = argmax_with_scores(&l);
            indices.iter().enumerate().map(|(i, &idx)| Resolution {
                candidate_type: "field".into(),
                candidate_id: i,
                resolved_type: "field".into(),
                resolved_id: idx,
                score: scores[i],
            }).collect()
        }).unwrap_or_default();

        let filters: Vec<ResolvedFilter> = logits.filter_op.map(|l| {
            let (op_indices, op_scores) = argmax_with_scores(&l);
            query_graph.filters.iter().enumerate().map(|(i, fc)| {
                let field_res = fields.iter()
                    .find(|f| f.candidate_id == fc.field_candidate_id)
                    .cloned()
                    .unwrap_or(Resolution {
                        candidate_type: "field".into(),
                        candidate_id: fc.field_candidate_id,
                        resolved_type: "field".into(),
                        resolved_id: 0,
                        score: 0.0,
                    });
                ResolvedFilter {
                    field: field_res,
                    operation: Resolution {
                        candidate_type: "filter".into(),
                        candidate_id: i,
                        resolved_type: "operation".into(),
                        resolved_id: op_indices[i],
                        score: op_scores[i],
                    },
                    value: fc.value.clone(),
                }
            }).collect()
        }).unwrap_or_default();

        let traversals: Vec<Resolution> = logits.traversal.map(|l| {
            let (indices, scores) = argmax_with_scores(&l);
            indices.iter().enumerate().map(|(i, &idx)| Resolution {
                candidate_type: "traversal".into(),
                candidate_id: i,
                resolved_type: "table".into(),
                resolved_id: idx,
                score: scores[i],
            }).collect()
        }).unwrap_or_default();

        let modifiers: Vec<ResolvedModifier> = logits.modifier_op.map(|l| {
            let (op_indices, op_scores) = argmax_with_scores(&l);
            query_graph.modifiers.iter().enumerate().map(|(i, mc)| {
                ResolvedModifier {
                    operation: Resolution {
                        candidate_type: "modifier".into(),
                        candidate_id: i,
                        resolved_type: "operation".into(),
                        resolved_id: op_indices[i],
                        score: op_scores[i],
                    },
                    value: mc.value.clone(),
                }
            }).collect()
        }).unwrap_or_default();

        ResolvedGraph { collection, fields, filters, traversals, modifiers }
    }
}

// =============================================================================
// Scoring helpers
// =============================================================================

/// proj(query_embs) @ target_embs.T → [n_query, n_targets]
fn bilinear_score<B: Backend>(
    embeddings: &HashMap<String, Tensor<B, 2>>,
    query_type: &str,
    target_type: &str,
    proj: &Linear<B>,
) -> Option<Tensor<B, 2>> {
    let q = embeddings.get(query_type)?;
    let t = embeddings.get(target_type)?;
    // Skip empty tensors — fusion optimizer can't handle [0, dim] matmuls
    if q.dims()[0] == 0 || t.dims()[0] == 0 {
        return None;
    }
    let projected = proj.forward(q.clone());
    Some(projected.matmul(t.clone().transpose()))
}

/// Argmax per row, returning indices and softmax scores.
fn argmax_with_scores<B: Backend>(logits: &Tensor<B, 2>) -> (Vec<usize>, Vec<f32>) {
    let probs = activation::softmax(logits.clone(), 1);

    let idx_data = logits.clone().argmax(1).into_data();
    let indices: Vec<usize> = idx_data
        .to_vec::<i32>().expect("argmax to i32")
        .iter().map(|&x| x as usize).collect();

    let max_data = probs.max_dim(1).into_data();
    let scores: Vec<f32> = max_data
        .to_vec::<f32>().expect("max scores to f32");

    (indices, scores)
}
