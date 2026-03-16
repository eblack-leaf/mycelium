// =============================================================================
// hetero_conv.rs — Heterogeneous Convolution Layer
//
// Wraps multiple SAGEConv layers, one per relation type in the graph.
// A relation type is (src_node_type, edge_type, dst_node_type).
//
// Why one conv per relation?
//   Different relations carry different semantic meaning.
//   HAS_FIELD edges should transform features differently than LINKS_TO edges.
//   Sharing weights across relation types would conflate these signals.
//
// What HeteroConv does:
//   For each relation key (src_type, edge_type, dst_type):
//     1. Look up the SAGEConv for this relation
//     2. Run forward with src embeddings, dst embeddings, edge indices
//     3. Collect resulting [n_dst, out_dim] tensor
//   After all relations are processed:
//     For each dst_type that received messages from multiple relations,
//     average the contributions together.
//     This is the "sum" aggregation in the HeteroConv literature
//     (mean here for stability).
//
// Input/output:
//   Input  HashMap<node_type, Tensor[n_nodes, in_dim]>
//   Output HashMap<node_type, Tensor[n_nodes, out_dim]>
// =============================================================================

use std::collections::HashMap;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};
use crate::{
    graph::{HeteroGraph, RelationKey},
    sage_conv::SAGEConv,
};

// -----------------------------------------------------------------------------
// HeteroConvConfig — used to construct a HeteroConv before having a device
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct HeteroConvConfig {
    /// Map from relation key string "src__edge__dst" to (in_dim, out_dim)
    pub relation_dims: HashMap<String, (usize, usize)>,
    /// Output dim (all relations output the same dim for easy aggregation)
    pub out_dim: usize,
}

impl HeteroConvConfig {
    /// Build config from a graph and uniform in_dims per node type
    pub fn from_graph(
        graph: &HeteroGraph,
        node_in_dims: &HashMap<String, usize>,
        out_dim: usize,
    ) -> Self {
        let mut relation_dims = HashMap::new();

        for key in graph.relation_keys() {
            let rel_str = relation_key_to_str(key);
            let in_dim = *node_in_dims
                .get(&key.src_type)
                .expect("node_in_dims must contain all node types in graph");
            relation_dims.insert(rel_str, (in_dim, out_dim));
        }

        Self { relation_dims, out_dim }
    }
}

// -----------------------------------------------------------------------------
// HeteroConv
// -----------------------------------------------------------------------------
#[derive(Module, Debug)]
pub struct HeteroConv<B: Backend> {
    /// One SAGEConv per relation, keyed by "src__edge__dst"
    /// Burn's Module derive handles serialization of this Vec automatically.
    /// We store as parallel Vecs because HashMap isn't Module-derivable.
    pub relation_keys: Vec<String>,
    pub convs: Vec<SAGEConv<B>>,
    pub out_dim: usize,
}

impl<B: Backend> HeteroConv<B> {
    pub fn new(config: &HeteroConvConfig, device: &B::Device) -> Self {
        let mut relation_keys = Vec::new();
        let mut convs = Vec::new();

        // Sort for deterministic ordering (important for Module serialization)
        let mut sorted_relations: Vec<_> = config.relation_dims.iter().collect();
        sorted_relations.sort_by_key(|(k, _)| k.clone());

        for (key, (in_dim, out_dim)) in sorted_relations {
            relation_keys.push(key.clone());
            convs.push(SAGEConv::new(*in_dim, *out_dim, device));
        }

        Self { relation_keys, convs, out_dim: config.out_dim }
    }

    /// One full heterogeneous message passing step.
    ///
    /// node_embeddings: current embedding for each node type
    /// graph:           graph topology (edge indices)
    ///
    /// Returns updated embeddings for all node types.
    /// Node types that received no messages keep their input embeddings.
    pub fn forward(
        &self,
        graph: &HeteroGraph,
        node_embeddings: &HashMap<String, Tensor<B, 2>>,
        device: &B::Device,
    ) -> HashMap<String, Tensor<B, 2>> {
        // Accumulate updated embeddings per dst_type
        // Each dst_type may receive messages from multiple relation types
        let mut dst_accumulator: HashMap<String, Vec<Tensor<B, 2>>> = HashMap::new();

        for (rel_str, conv) in self.relation_keys.iter().zip(self.convs.iter()) {
            let (src_type, edge_type, dst_type) = parse_relation_key(rel_str);
            let key = RelationKey::new(&src_type, &edge_type, &dst_type);

            let edge_store = match graph.edge_stores.get(&key) {
                Some(e) => e,
                None => continue, // this relation not in graph, skip
            };

            let src_emb = match node_embeddings.get(&src_type) {
                Some(e) => e.clone(),
                None => continue,
            };
            let dst_emb = match node_embeddings.get(&dst_type) {
                Some(e) => e.clone(),
                None => continue,
            };

            // Run SAGEConv for this relation
            let updated = conv.forward(
                src_emb,
                dst_emb,
                &edge_store.src_indices,
                &edge_store.dst_indices,
                device,
            );

            dst_accumulator
                .entry(dst_type.to_string())
                .or_default()
                .push(updated);
        }

        // Start with current embeddings (pass-through for untouched node types)
        let mut output = node_embeddings.clone();

        // For each dst_type that got messages, average across contributing relations
        for (dst_type, contributions) in dst_accumulator {
            if contributions.is_empty() {
                continue;
            }

            let n_contribs = contributions.len() as f32;
            let summed = contributions
                .into_iter()
                .reduce(|acc, t| acc + t)
                .unwrap();

            // Divide by number of contributing relations
            output.insert(dst_type, summed / n_contribs);
        }

        output
    }

    /// Look up the SAGEConv for a given relation key string
    fn get_conv(&self, rel_str: &str) -> Option<&SAGEConv<B>> {
        self.relation_keys
            .iter()
            .position(|k| k == rel_str)
            .map(|i| &self.convs[i])
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

pub fn relation_key_to_str(key: &RelationKey) -> String {
    format!("{}__{}__{}", key.src_type, key.edge_type, key.dst_type)
}

pub fn parse_relation_key(s: &str) -> (String, String, String) {
    let parts: Vec<&str> = s.splitn(3, "__").collect();
    assert_eq!(parts.len(), 3, "relation key must have format src__edge__dst");
    (parts[0].to_string(), parts[1].to_string(), parts[2].to_string())
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    use crate::graph::make_example_graph;

    type B = NdArray;

    #[test]
    fn test_hetero_conv_output_shapes() {
        let device = Default::default();
        let feat_dim = 6;
        let out_dim = 8;

        let graph = make_example_graph(feat_dim);

        let node_in_dims: HashMap<String, usize> = [
            ("collection".to_string(), feat_dim),
            ("field".to_string(), feat_dim),
        ]
        .into();

        let config = HeteroConvConfig::from_graph(&graph, &node_in_dims, out_dim);
        let conv = HeteroConv::<B>::new(&config, &device);

        // Build initial embeddings from graph node features
        let mut embeddings: HashMap<String, Tensor<B, 2>> = HashMap::new();
        for (node_type, store) in &graph.node_stores {
            let t = Tensor::<B, 2>::from_data(
                TensorData::new(store.features.clone(), [store.n_nodes, store.feat_dim]),
                &device,
            );
            embeddings.insert(node_type.clone(), t);
        }

        let updated = conv.forward(&graph, &embeddings, &device);

        // collection nodes: 2 nodes, out_dim features
        let coll = updated.get("collection").expect("collection embeddings missing");
        assert_eq!(coll.dims(), [2, out_dim]);

        // field nodes: 3 nodes, out_dim features
        let field = updated.get("field").expect("field embeddings missing");
        assert_eq!(field.dims(), [3, out_dim]);
    }
}
