// model.rs — SageConv + bilinear heads GNN architecture

use crate::sage::SageConv;
use burn::{config::Config, module::Module, tensor::backend::Backend};
use crate::graph::GroundedGraph;
use crate::query::QueryIr;

#[derive(Debug, Config)]
pub struct HyphaeConfig {
    #[config(default = 256)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub num_layers: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Input node feature dimensionality.
    #[config(default = 64)]
    pub node_feat_dim: usize,
    /// Token embedding dim used to initialise span node features.
    #[config(default = 128)]
    pub embed_dim: usize,
}

/// R-GCN / GraphSAGE GNN with bilinear resolution heads. HyphaeConfig holds all hyperparameters.
#[derive(Module, Debug)]
pub struct Hyphae<B: Backend> {
    pub sage: SageConv<B>,
}

impl<B: Backend> Hyphae<B> {
    pub fn new(config: &HyphaeConfig, device: &B::Device) -> Self {
        Self {
            sage: SageConv::new(
                config.node_feat_dim,
                config.hidden_dim,
                config.num_layers,
                device,
            ),
        }
    }

    pub fn forward(&self, _graph: &GroundedGraph, _device: &B::Device) -> QueryIr {
        todo!()
    }
}
