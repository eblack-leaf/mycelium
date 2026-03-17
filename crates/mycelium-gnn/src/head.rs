// =============================================================================
// head.rs — Output head: joint role + target prediction for linguistic nodes
//
// For each linguistic node, predicts:
//   1. Schema role (Collection, Field, FilterField, Modifier, Traversal, None)
//   2. Target schema node — bilinear score masked by candidate edges so the
//      head only picks among schema nodes the cross-encoder surfaced.
//
// score(ling_node, schema_node) = proj(ling_emb) · schema_embᵀ + mask
// role(ling_node) = role_classifier(ling_emb)
// =============================================================================

use std::collections::HashMap;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor, TensorData},
};
use crate::training::SchemaRole;
use crate::nlp::{LinguisticGraph, SpanType};
use crate::candidate_matcher::CandidateSet;

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
// Candidate mask — constrains target predictions to cross-encoder candidates
// =============================================================================

/// Binary masks [n_ling, n_schema_nodes] per schema type.
/// 1.0 where a candidate edge exists, large negative elsewhere.
/// Built from the CandidateSet so the head doesn't redo cross-encoder work.
pub struct CandidateMask {
    /// [n_ling, n_tables] — 0.0 for candidate, -1e9 for non-candidate
    pub table_mask: Vec<f32>,
    pub n_tables: usize,
    /// [n_ling, n_fields]
    pub field_mask: Vec<f32>,
    pub n_fields: usize,
    /// [n_ling, n_ops]
    pub op_mask: Vec<f32>,
    pub n_ops: usize,
    pub n_ling: usize,
}

impl CandidateMask {
    /// Build masks from a CandidateSet.
    /// n_ling is the total linguistic node count in head concatenation order
    /// (all np, then quantifier, comparator, intent).
    pub fn from_candidates(
        ling_graph: &LinguisticGraph,
        candidates: &CandidateSet,
        n_tables: usize,
        n_fields: usize,
        n_ops: usize,
    ) -> Self {
        // Build the same ordering the head uses when concatenating
        let mut ordered_ids: Vec<usize> = Vec::new();
        for st in &[SpanType::NounPhrase, SpanType::Quantifier, SpanType::Comparator, SpanType::Intent] {
            for node in &ling_graph.nodes {
                if node.span_type == *st {
                    ordered_ids.push(node.id);
                }
            }
        }
        let n_ling = ordered_ids.len();

        // Map from original node.id → position in concatenated order
        let mut id_to_pos: HashMap<usize, usize> = HashMap::new();
        for (pos, &nid) in ordered_ids.iter().enumerate() {
            id_to_pos.insert(nid, pos);
        }

        // Initialize with large negative (mask out everything)
        let neg = -1e9_f32;
        let mut table_mask = vec![neg; n_ling * n_tables];
        let mut field_mask = vec![neg; n_ling * n_fields];
        let mut op_mask = vec![neg; n_ling * n_ops];

        // Unmask candidate positions (set to 0.0)
        for edge in &candidates.edges {
            if let Some(&pos) = id_to_pos.get(&edge.linguistic_node) {
                match edge.schema_node_type.as_str() {
                    "table" if edge.schema_node_id < n_tables => {
                        table_mask[pos * n_tables + edge.schema_node_id] = 0.0;
                    }
                    "field" if edge.schema_node_id < n_fields => {
                        field_mask[pos * n_fields + edge.schema_node_id] = 0.0;
                    }
                    "operation" if edge.schema_node_id < n_ops => {
                        op_mask[pos * n_ops + edge.schema_node_id] = 0.0;
                    }
                    _ => {}
                }
            }
        }

        Self { table_mask, n_tables, field_mask, n_fields, op_mask, n_ops, n_ling }
    }

    fn table_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(self.table_mask.clone(), [self.n_ling, self.n_tables]),
            device,
        )
    }

    fn field_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(self.field_mask.clone(), [self.n_ling, self.n_fields]),
            device,
        )
    }

    fn op_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(self.op_mask.clone(), [self.n_ling, self.n_ops]),
            device,
        )
    }
}

// =============================================================================
// Training output — raw logit tensors
// =============================================================================

pub struct HeadLogits<B: Backend> {
    /// Role classification: [n_linguistic_nodes, n_roles]
    pub role_logits: Tensor<B, 2>,
    /// Target scores per linguistic node against each schema type:
    /// [n_ling, n_tables], [n_ling, n_fields], [n_ling, n_operations]
    /// Masked: non-candidate positions have -1e9 added.
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

    /// Compute logits for training/inference.
    /// mask: constrains target scores to candidate edges (None = no masking).
    pub fn forward(
        &self,
        embeddings: &HashMap<String, Tensor<B, 2>>,
        mask: Option<&CandidateMask>,
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
        let device = ling_emb.device();
        let role_logits = self.role_classifier.forward(ling_emb.clone());

        // Target scoring: bilinear + candidate mask
        let target_table = bilinear_score_masked(
            &ling_emb, embeddings.get("table"), &self.table_proj,
            mask.map(|m| m.table_tensor::<B>(&device)),
        );
        let target_field = bilinear_score_masked(
            &ling_emb, embeddings.get("field"), &self.field_proj,
            mask.map(|m| m.field_tensor::<B>(&device)),
        );
        let target_op = bilinear_score_masked(
            &ling_emb, embeddings.get("operation"), &self.op_proj,
            mask.map(|m| m.op_tensor::<B>(&device)),
        );

        HeadLogits { role_logits, target_table, target_field, target_op }
    }
}

fn bilinear_score_masked<B: Backend>(
    query: &Tensor<B, 2>,
    target: Option<&Tensor<B, 2>>,
    proj: &Linear<B>,
    mask: Option<Tensor<B, 2>>,
) -> Option<Tensor<B, 2>> {
    let t = target?;
    if query.dims()[0] == 0 || t.dims()[0] == 0 {
        return None;
    }
    let projected = proj.forward(query.clone());
    let scores = projected.matmul(t.clone().transpose());
    match mask {
        Some(m) => Some(scores + m),
        None => Some(scores),
    }
}
