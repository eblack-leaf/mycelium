// sage.rs — R-GCN style heterogeneous SageConv
//
// One Linear per edge type (HeteroConv pattern).
// 13 edge types: schema structure, modifier routing, cross edges, inter-span.

use crate::ops;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation, backend::Backend, Tensor},
};
use std::collections::HashMap;

/// All edge types in the combined graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeType {
    // --- Schema structure (fixed per schema) ---
    HasField,   // Table → Field
    FieldOf,    // Field → Table
    LinksTo,    // Record-link Field → linked Table
    LinkedFrom, // Linked Table → record-link Field (reverse)

    // --- Span routing (added per query in inject()) ---
    EntityToSpan,      // entity span → field-resolving spans (table context broadcast)
    SpanToTable,       // field-resolving spans → all table nodes (bridge to field subgraph)
    ProjectionToFetch, // ProjectionSpan → ModifierSpan (co-reference)
}

impl EdgeType {
    pub fn all() -> &'static [EdgeType] {
        &[
            EdgeType::HasField,
            EdgeType::FieldOf,
            EdgeType::LinksTo,
            EdgeType::LinkedFrom,
            EdgeType::EntityToSpan,
            EdgeType::SpanToTable,
            EdgeType::ProjectionToFetch,
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
/// Uses parallel Vec instead of HashMap so burn can derive Module.
/// Index into edge_projs matches index into EdgeType::all().
#[derive(Module, Debug)]
pub struct SageConvLayer<B: Backend> {
    pub self_proj: Linear<B>,
    pub edge_projs: Vec<Linear<B>>,
}

impl<B: Backend> SageConvLayer<B> {
    pub fn new(feat_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let edge_projs = EdgeType::all()
            .iter()
            .map(|_| LinearConfig::new(feat_dim, hidden_dim).init(device))
            .collect();

        Self {
            self_proj: LinearConfig::new(feat_dim, hidden_dim).init(device),
            edge_projs,
        }
    }

    /// features: [num_nodes, feat_dim] → [num_nodes, hidden_dim]
    pub fn forward(
        &self,
        features: Tensor<B, 2>,
        edges: &TypedEdges,
        num_nodes: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let mut out = self.self_proj.forward(features.clone());

        for (edge_type, edge_list) in edges {
            if edge_list.is_empty() {
                continue;
            }
            let proj = match EdgeType::all().iter().position(|e| e == edge_type) {
                Some(i) => &self.edge_projs[i],
                None => continue,
            };

            let src_idx: Vec<usize> = edge_list.iter().map(|e| e.src).collect();
            let dst_idx: Vec<usize> = edge_list.iter().map(|e| e.dst).collect();

            let gathered = ops::gather(features.clone(), &src_idx, device);
            let projected = proj.forward(gathered);
            out = out + ops::scatter_add(projected, &dst_idx, num_nodes, device);
        }

        ops::l2_normalize(activation::relu(out))
    }
}

/// Multi-layer SageConv stack.
#[derive(Module, Debug)]
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
        features: Tensor<B, 2>,
        edges: &TypedEdges,
        num_nodes: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let mut h = features;
        for layer in &self.layers {
            h = layer.forward(h, edges, num_nodes, device);
        }
        h
    }
}
