// model.rs — BiLSTM-CRF architecture for slot extraction

use std::path::Path;
use crate::Slots;

pub struct ModelConfig {
    pub hidden_dim:  usize,
    pub num_layers:  usize,
    pub dropout:     f32,
    pub vocab_size:  usize,
    pub embed_dim:   usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            num_layers: 2,
            dropout:    0.3,
            vocab_size: 10_000,
            embed_dim:  128,
        }
    }
}

pub struct Model {
    config: ModelConfig,
}

impl Model {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        todo!()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        todo!()
    }

    pub fn forward(&self, tokens: &[&str]) -> Slots {
        todo!()
    }
}
