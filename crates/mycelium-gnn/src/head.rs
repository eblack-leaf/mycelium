// =============================================================================
// head.rs — Output head: joint role + target prediction for linguistic nodes
//
// For each linguistic node, predicts:
//   1. Schema role (Collection, Field, FilterField, Modifier, Traversal, None)
//   2. Target schema node — scored against all candidates of the resolved type
//
// score(ling_node, schema_node) = proj(ling_emb) · schema_embᵀ
// role(ling_node) = role_classifier(ling_emb)
// =============================================================================

use std::collections::HashMap;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use crate::training::SchemaRole;

// =============================================================================
// Output types
// =============================================================================

/// Resolution of a single linguistic node.
#[derive(Debug, Clone)]
pub struct NodeResolution {
    pub linguistic_node_id: usize,
    pub role: SchemaRole,
    pub role_confidence: f32,
    pub target_type: String,
    pub target_id: usize,
    pub target_score: f32,
}

/// Full pipeline output.
#[derive(Debug, Clone)]
pub struct ResolvedGraph {
    pub resolutions: Vec<NodeResolution>,
}

// =============================================================================
// Training output — raw logit tensors
// =============================================================================

pub struct HeadLogits<B: Backend> {
    /// Role classification: [n_linguistic_nodes, n_roles]
    pub role_logits: Tensor<B, 2>,
    /// Target scores per linguistic node type against each schema type:
    /// [n_np, n_tables], [n_np, n_fields], [n_np, n_operations]
    pub target_table: Option<Tensor<B, 2>>,
    pub target_field: Option<Tensor<B, 2>>,
    pub target_op: Option<Tensor<B, 2>>,
}

// =============================================================================
// Output head
// =============================================================================

#[derive(Module, Debug)]
pub struct OutputHead<B: Backend> {
    /// Role classifier: hidden_dim → n_roles
    role_classifier: Linear<B>,
    /// Target projections: one per schema node type
    table_proj: Linear<B>,
    field_proj: Linear<B>,
    op_proj: Linear<B>,
}

impl<B: Backend> OutputHead<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            role_classifier: LinearConfig::new(hidden_dim, SchemaRole::COUNT).init(device),
            table_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            field_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            op_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
        }
    }

    /// Compute logits for training.
    pub fn forward(
        &self,
        embeddings: &HashMap<String, Tensor<B, 2>>,
    ) -> HeadLogits<B> {
        // Concatenate all linguistic node embeddings for role classification
        let ling_types = ["np", "quantifier", "comparator", "intent"];
        let mut ling_parts: Vec<Tensor<B, 2>> = Vec::new();
        for t in &ling_types {
            if let Some(emb) = embeddings.get(*t) {
                if emb.dims()[0] > 0 {
                    ling_parts.push(emb.clone());
                }
            }
        }

        // If no linguistic nodes, return empty logits
        if ling_parts.is_empty() {
            let device = embeddings.values().next().unwrap().device();
            return HeadLogits {
                role_logits: Tensor::zeros([0, SchemaRole::COUNT], &device),
                target_table: None,
                target_field: None,
                target_op: None,
            };
        }

        let ling_emb = Tensor::cat(ling_parts, 0); // [n_ling, hidden]
        let role_logits = self.role_classifier.forward(ling_emb.clone());

        // Target scoring: project linguistic embeddings, dot with schema embeddings
        let target_table = bilinear_score(&ling_emb, embeddings.get("table"), &self.table_proj);
        let target_field = bilinear_score(&ling_emb, embeddings.get("field"), &self.field_proj);
        let target_op = bilinear_score(&ling_emb, embeddings.get("operation"), &self.op_proj);

        HeadLogits { role_logits, target_table, target_field, target_op }
    }
}

fn bilinear_score<B: Backend>(
    query: &Tensor<B, 2>,
    target: Option<&Tensor<B, 2>>,
    proj: &Linear<B>,
) -> Option<Tensor<B, 2>> {
    let t = target?;
    if query.dims()[0] == 0 || t.dims()[0] == 0 {
        return None;
    }
    let projected = proj.forward(query.clone());
    Some(projected.matmul(t.clone().transpose()))
}
