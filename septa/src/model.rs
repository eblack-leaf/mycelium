// model.rs — BiLSTM-CRF architecture for slot extraction

use crate::Semantics;
use burn::config::Config;
use burn::module::Module;
use std::path::Path;

#[derive(Debug, Config)]
pub struct ModelConfig {
    #[config(default = 256)]
    pub hidden_dim: usize,
    #[config(default = 2)]
    pub num_layers: usize,
    #[config(default = 0.3)]
    pub dropout: f32,
    #[config(default = 10_000)]
    pub vocab_size: usize,
    #[config(default = 128)]
    pub embed_dim: usize,
}

pub struct Model {
    config: ModelConfig,
}
#[derive(Module, Debug, Clone)]
pub struct Septa {}

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

    pub fn forward(&self, tokens: &[&str]) -> Semantics {
        todo!()
    }
}
