// sage.rs — R-GCN style heterogeneous SageConv
//
// One Linear per edge type (HeteroConv pattern).
// Covers: schema possibility graph edges + cross edges (slot→schema) + inter-span edges.

use std::collections::HashMap;
use burn::{
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

/// All edge types in the combined graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeType {
    // --- Schema structure (fixed per schema) ---
    HasField,               // Table → Field
    FieldOf,                // Field → Table
    LinksTo,                // Table → Table (record link)
    LinkedFrom,             // Table → Table (reverse record link)

    // --- Possibility graph (valid ops/modifiers on schema nodes) ---
    OperationToTable,       // Operation → Table (all ops valid on any table)
    OperationToModifier,    // Operation → Modifier (which modifiers each op supports)
    FieldHasComparator,     // Field → Comparator (type-derived: int→>,<,=; string→=,CONTAINS)
    ModifierToField,        // Modifier → Field (WHERE/ORDER BY apply to fields)

    // --- Cross edges: slot span nodes → schema possibility nodes ---
    Intent,                 // intent slot     → Operation node
    Entity,                 // entities slot   → Table or Field candidate
    Projection,             // projections slot → Field candidate
    Condition,              // conditions slot → Field + Comparator candidates
    Assignment,             // assignments slot → Field candidate (write, no comparator)
    Modifier,               // modifiers slot  → Modifier node

    // --- Inter-span edges: subordinate slots → governing entity ---
    ConditionToEntity,      // Condition span → Entity span (condition applies to this entity)
    ProjectionToEntity,     // Projection span → Entity span (fields are from this entity)
    AssignmentToEntity,     // Assignment span → Entity span (writes target this entity)
    ModifierToEntity,       // Modifier span → Entity span (ordering/limit on this entity)
}

impl EdgeType {
    pub fn all() -> &'static [EdgeType] {
        &[
            EdgeType::HasField,
            EdgeType::FieldOf,
            EdgeType::LinksTo,
            EdgeType::LinkedFrom,
            EdgeType::OperationToTable,
            EdgeType::OperationToModifier,
            EdgeType::FieldHasComparator,
            EdgeType::ModifierToField,
            EdgeType::Intent,
            EdgeType::Entity,
            EdgeType::Projection,
            EdgeType::Condition,
            EdgeType::Assignment,
            EdgeType::Modifier,
            EdgeType::ConditionToEntity,
            EdgeType::ProjectionToEntity,
            EdgeType::AssignmentToEntity,
            EdgeType::ModifierToEntity,
        ]
    }
}

/// A directed edge between node indices.
#[derive(Debug, Clone)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
}

pub type TypedEdges = HashMap<EdgeType, Vec<Edge>>;

/// Single HeteroConv layer — one Linear per edge type.
/// HashMap<EdgeType, Linear> cannot derive Module — needs custom impl.
pub struct SageConvLayer<B: Backend> {
    pub self_proj:  Linear<B>,
    pub edge_projs: HashMap<EdgeType, Linear<B>>,
}

impl<B: Backend> SageConvLayer<B> {
    pub fn new(feat_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let edge_projs = EdgeType::all().iter()
            .map(|et| (et.clone(), LinearConfig::new(feat_dim, hidden_dim).init(device)))
            .collect();

        Self {
            self_proj:  LinearConfig::new(feat_dim, hidden_dim).init(device),
            edge_projs,
        }
    }

    /// features: [num_nodes, feat_dim] → [num_nodes, hidden_dim]
    pub fn forward(
        &self,
        features:  Tensor<B, 2>,
        edges:     &TypedEdges,
        num_nodes: usize,
    ) -> Tensor<B, 2> {
        todo!()
    }
}

/// Multi-layer SageConv stack.
pub struct SageConv<B: Backend> {
    pub layers: Vec<SageConvLayer<B>>,
}

impl<B: Backend> SageConv<B> {
    pub fn new(feat_dim: usize, hidden_dim: usize, num_layers: usize, device: &B::Device) -> Self {
        let layers = (0..num_layers)
            .map(|i| {
                let in_dim = if i == 0 { feat_dim } else { hidden_dim };
                SageConvLayer::new(in_dim, hidden_dim, device)
            })
            .collect();
        Self { layers }
    }

    pub fn forward(
        &self,
        features:  Tensor<B, 2>,
        edges:     &TypedEdges,
        num_nodes: usize,
    ) -> Tensor<B, 2> {
        todo!()
    }
}
