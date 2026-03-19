// model.rs — SageConv + bilinear head GNN architecture

use std::path::Path;
use crate::Predictions;

pub struct GnnConfig {
    pub hidden_dim:  usize,
    pub num_layers:  usize,
    pub dropout:     f32,
    pub node_feat_dim: usize,  // input node feature dimensionality
}

impl Default for GnnConfig {
    fn default() -> Self {
        Self {
            hidden_dim:    256,
            num_layers:    3,
            dropout:       0.3,
            node_feat_dim: 64,
        }
    }
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
