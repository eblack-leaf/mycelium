// =============================================================================
// encoder.rs — Stacked HeteroConv Layers = Full GNN Encoder
//
// The encoder is a stack of HeteroConv layers. Each layer does one round of
// message passing across all relation types. Stacking k layers means each
// node can "see" k hops away in the graph.
//
// For query building:
//   - 2 layers is usually enough (field → collection → field is 2 hops)
//   - 3 layers if you want graph traversal patterns (person → order → product)
//
// Architecture:
//   Layer 0: [node_in_dims]   → [hidden_dim]   (maps heterogeneous input dims)
//   Layer 1: [hidden_dim]     → [hidden_dim]   (uniform from here)
//   ...
//   Layer k: [hidden_dim]     → [hidden_dim]
//
// Output:
//   HashMap<node_type, Tensor[n_nodes, hidden_dim]>
//   These are the final node embeddings used by the constraint layer
//   and emitter downstream.
// =============================================================================

use std::collections::HashMap;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor, TensorData},
};
use crate::{
    graph::HeteroGraph,
    hetero_conv::{HeteroConv, HeteroConvConfig},
};

// -----------------------------------------------------------------------------
// GnnEncoderConfig
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct GnnEncoderConfig {
    pub n_layers: usize,
    pub hidden_dim: usize,
    /// Input feature dim per node type (only needed for layer 0)
    pub node_in_dims: HashMap<String, usize>,
}

impl GnnEncoderConfig {
    pub fn new(
        n_layers: usize,
        hidden_dim: usize,
        node_in_dims: HashMap<String, usize>,
    ) -> Self {
        Self { n_layers, hidden_dim, node_in_dims }
    }
}

// -----------------------------------------------------------------------------
// GnnEncoder
// -----------------------------------------------------------------------------
#[derive(Module, Debug)]
pub struct GnnEncoder<B: Backend> {
    /// Stacked HeteroConv layers
    pub layers: Vec<HeteroConv<B>>,
    pub n_layers: usize,
    pub hidden_dim: usize,
}

impl<B: Backend> GnnEncoder<B> {
    pub fn new(
        config: &GnnEncoderConfig,
        graph: &HeteroGraph,
        device: &B::Device,
    ) -> Self {
        let mut layers = Vec::new();

        for i in 0..config.n_layers {
            // Layer 0 maps from input dims; subsequent layers map hidden→hidden
            let in_dims: HashMap<String, usize> = if i == 0 {
                config.node_in_dims.clone()
            } else {
                // After layer 0 all node types have hidden_dim
                graph.node_types()
                    .iter()
                    .map(|&t| (t.to_string(), config.hidden_dim))
                    .collect()
            };

            let conv_config = HeteroConvConfig::from_graph(graph, &in_dims, config.hidden_dim);
            layers.push(HeteroConv::new(&conv_config, device));
        }

        Self {
            layers,
            n_layers: config.n_layers,
            hidden_dim: config.hidden_dim,
        }
    }

    /// Run the full encoder: k rounds of message passing.
    ///
    /// initial_embeddings: raw node features (from graph.node_stores)
    ///                     shape per type: [n_nodes, feat_dim]
    ///
    /// returns: final node embeddings, shape per type: [n_nodes, hidden_dim]
    pub fn forward(
        &self,
        graph: &HeteroGraph,
        initial_embeddings: HashMap<String, Tensor<B, 2>>,
        device: &B::Device,
    ) -> HashMap<String, Tensor<B, 2>> {
        let mut embeddings = initial_embeddings;

        for layer in &self.layers {
            embeddings = layer.forward(graph, &embeddings, device);
        }

        embeddings
    }

    /// Convenience: build initial embeddings from a HeteroGraph's NodeStores
    pub fn embeddings_from_graph(
        graph: &HeteroGraph,
        device: &B::Device,
    ) -> HashMap<String, Tensor<B, 2>> {
        graph
            .node_stores
            .iter()
            .map(|(node_type, store)| {
                let t = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        store.features.clone(),
                        [store.n_nodes, store.feat_dim],
                    ),
                    device,
                );
                (node_type.clone(), t)
            })
            .collect()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use crate::graph::make_example_graph;

    type B = NdArray;

    #[test]
    fn test_encoder_two_layers() {
        let device = Default::default();
        let feat_dim = 6;
        let hidden_dim = 16;
        let n_layers = 2;

        let graph = make_example_graph(feat_dim);

        let node_in_dims: HashMap<String, usize> = graph
            .node_stores
            .iter()
            .map(|(k, v)| (k.clone(), v.feat_dim))
            .collect();

        let config = GnnEncoderConfig::new(n_layers, hidden_dim, node_in_dims);
        let encoder = GnnEncoder::<B>::new(&config, &graph, &device);

        let initial = GnnEncoder::<B>::embeddings_from_graph(&graph, &device);
        let final_emb = encoder.forward(&graph, initial, &device);

        // All node types should now have [n_nodes, hidden_dim] embeddings
        let coll = final_emb.get("collection").unwrap();
        assert_eq!(coll.dims()[1], hidden_dim, "collection embedding dim wrong");

        let field = final_emb.get("field").unwrap();
        assert_eq!(field.dims()[1], hidden_dim, "field embedding dim wrong");

        println!("collection embeddings: {:?}", coll.dims());
        println!("field embeddings:      {:?}", field.dims());
    }

    #[test]
    fn test_encoder_no_nan() {
        let device = Default::default();
        let feat_dim = 4;
        let graph = make_example_graph(feat_dim);

        let node_in_dims: HashMap<String, usize> = graph
            .node_stores
            .iter()
            .map(|(k, v)| (k.clone(), v.feat_dim))
            .collect();

        let config = GnnEncoderConfig::new(3, 8, node_in_dims);
        let encoder = GnnEncoder::<B>::new(&config, &graph, &device);

        let initial = GnnEncoder::<B>::embeddings_from_graph(&graph, &device);
        let output = encoder.forward(&graph, initial, &device);

        for (node_type, emb) in &output {
            let has_nan = emb.clone().is_nan().any().into_scalar();
            assert!(!has_nan, "{} embeddings contain NaN", node_type);
        }
    }
}
