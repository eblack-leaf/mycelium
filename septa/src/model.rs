// model.rs — BiLSTM-CRF architecture for semantic span extraction

use crate::Semantics;
use burn::{config::Config, module::Module, tensor::backend::Backend};

#[derive(Debug, Config)]
pub struct SeptaConfig {
    #[config(default = 256)]
    pub hidden_dim: usize,
    #[config(default = 2)]
    pub num_lstm_layers: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = 10_000)]
    pub vocab_size: usize,
    #[config(default = 128)]
    pub embed_dim: usize,
    /// Number of BIO tag classes — set from the tag set size, no default.
    pub num_tags: usize,
}

/// BiLSTM-CRF span extractor. SeptaConfig holds all hyperparameters.
#[derive(Module, Debug)]
pub struct Septa<B: Backend> {
    _phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> Septa<B> {
    pub fn new(_config: &SeptaConfig, _device: &B::Device) -> Self {
        Self { _phantom: core::marker::PhantomData }
    }

    /// tokens: preprocessed token strings (slots and temporals already marked).
    pub fn forward(&self, _tokens: &[&str]) -> Semantics {
        todo!()
    }
}
