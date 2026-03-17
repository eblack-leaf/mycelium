// =============================================================================
// sage.rs — GraphSAGE message passing over a ResolverConv
//
// SAGEConv    — one relation type
// HeteroConv  — one SAGEConv per relation, merges results per dst type
// Encoder     — stacked HeteroConv layers
// =============================================================================

use std::collections::HashMap;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use burn::tensor::activation;
use crate::linguistic_graph::LinguisticConv;
use crate::tensor_ops as ops;

// =============================================================================
// SAGEConv — single relation message passing
// =============================================================================

#[derive(Module, Debug)]
pub struct SAGEConv<B: Backend> {
    pub neighbor_proj: Linear<B>,
    pub self_proj: Linear<B>,
}

impl<B: Backend> SAGEConv<B> {
    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self {
            neighbor_proj: LinearConfig::new(in_dim, out_dim).with_bias(true).init(device),
            self_proj: LinearConfig::new(in_dim, out_dim).with_bias(false).init(device),
        }
    }

    /// src_features: [n_src, in_dim], dst_features: [n_dst, in_dim]
    /// weights: per-edge weights (empty = uniform). Cross-encoder scores for candidate edges.
    /// returns: [n_dst, out_dim]
    pub fn forward(
        &self,
        src_features: Tensor<B, 2>,
        dst_features: Tensor<B, 2>,
        src_indices: &[usize],
        dst_indices: &[usize],
        weights: &[f32],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let n_dst = dst_features.dims()[0];

        let gathered = ops::gather(src_features, src_indices, device);
        let agg = if weights.is_empty() {
            ops::scatter_mean(gathered, dst_indices, n_dst, device)
        } else {
            ops::scatter_weighted_mean(gathered, dst_indices, weights, n_dst, device)
        };

        let neighbor_out = self.neighbor_proj.forward(agg);
        let self_out = self.self_proj.forward(dst_features);

        ops::l2_normalize(activation::relu(neighbor_out + self_out))
    }
}

// =============================================================================
// HeteroConv — one SAGEConv per relation type
// =============================================================================

#[derive(Module, Debug)]
pub struct HeteroConv<B: Backend> {
    pub relation_keys: Vec<String>,
    pub convs: Vec<SAGEConv<B>>,
}

impl<B: Backend> HeteroConv<B> {
    pub fn new(
        conv: &LinguisticConv,
        in_dims: &HashMap<String, usize>,
        out_dim: usize,
        device: &B::Device,
    ) -> Self {
        let mut relation_keys = Vec::new();
        let mut convs = Vec::new();

        let mut sorted: Vec<_> = conv.relations.iter().collect();
        sorted.sort_by_key(|r| (&r.src_type, &r.edge_type, &r.dst_type));

        for rel in sorted {
            let key = format!("{}__{}__{}", rel.src_type, rel.edge_type, rel.dst_type);
            let in_dim = in_dims[&rel.src_type];
            relation_keys.push(key);
            convs.push(SAGEConv::new(in_dim, out_dim, device));
        }

        Self { relation_keys, convs }
    }

    pub fn forward(
        &self,
        conv: &LinguisticConv,
        embeddings: &HashMap<String, Tensor<B, 2>>,
        device: &B::Device,
    ) -> HashMap<String, Tensor<B, 2>> {
        let mut dst_acc: HashMap<String, Vec<Tensor<B, 2>>> = HashMap::new();

        for rel in &conv.relations {
            if rel.src_indices.is_empty() { continue; }

            let key = format!("{}__{}__{}", rel.src_type, rel.edge_type, rel.dst_type);
            let sage = match self.relation_keys.iter().position(|k| *k == key) {
                Some(idx) => &self.convs[idx],
                None => continue,
            };

            let src_emb = match embeddings.get(&rel.src_type) {
                Some(e) => e.clone(),
                None => continue,
            };
            let dst_emb = match embeddings.get(&rel.dst_type) {
                Some(e) => e.clone(),
                None => continue,
            };

            let updated = sage.forward(
                src_emb, dst_emb,
                &rel.src_indices, &rel.dst_indices,
                &rel.weights,
                device,
            );

            dst_acc.entry(rel.dst_type.clone()).or_default().push(updated);
        }

        let mut output = embeddings.clone();

        for (dst_type, contributions) in dst_acc {
            if contributions.is_empty() {
                continue;
            }
            let n = contributions.len() as f32;
            let summed = contributions.into_iter().reduce(|a, b| a + b).unwrap();
            output.insert(dst_type, summed / n);
        }

        output
    }
}

// =============================================================================
// Encoder — stacked HeteroConv layers
// =============================================================================

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pub layers: Vec<HeteroConv<B>>,
    pub hidden_dim: usize,
}

impl<B: Backend> Encoder<B> {
    pub fn new(
        conv: &LinguisticConv,
        input_dims: &HashMap<String, usize>,
        hidden_dim: usize,
        n_layers: usize,
        device: &B::Device,
    ) -> Self {
        let mut layers = Vec::new();

        for i in 0..n_layers {
            let in_dims: HashMap<String, usize> = if i == 0 {
                input_dims.clone()
            } else {
                conv.node_counts
                    .iter()
                    .map(|(name, _)| (name.clone(), hidden_dim))
                    .collect()
            };

            layers.push(HeteroConv::new(conv, &in_dims, hidden_dim, device));
        }

        Self { layers, hidden_dim }
    }

    /// Run all layers of message passing.
    /// returns: node_type → [n_nodes, hidden_dim]
    pub fn forward(
        &self,
        conv: &LinguisticConv,
        initial: HashMap<String, Tensor<B, 2>>,
        device: &B::Device,
    ) -> HashMap<String, Tensor<B, 2>> {
        let mut embeddings = initial;
        for layer in &self.layers {
            embeddings = layer.forward(conv, &embeddings, device);
        }
        embeddings
    }
}
