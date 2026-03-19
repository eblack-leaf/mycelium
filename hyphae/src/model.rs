// model.rs — SageConv + bilinear head GNN architecture

use crate::Predictions;
use std::path::Path;
use burn::config::Config;

#[derive(Debug, Config)]
pub struct GnnConfig {
    #[config(default = 256)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub num_layers: usize,
    #[config(default = 0.3)]
    pub dropout: f32,
    #[config(default = 64)]
    pub node_feat_dim: usize, // input node feature dimensionality
}

pub struct GnnModel {
    config: GnnConfig,
}

impl GnnModel {
    pub fn new(config: GnnConfig) -> Self {
        Self { config }
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        todo!()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        todo!()
    }

    pub fn forward(&self, graph: &crate::GroundedGraph) -> Predictions {
        todo!()
    }
}
