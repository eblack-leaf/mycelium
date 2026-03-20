// model.rs — BiLSTM-CRF architecture for semantic span extraction

use crate::Semantics;
use burn::{config::Config, module::Module, tensor::{backend::Backend, Tensor}};

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

/// BiLSTM hidden states for every span in one query, shape [2 * hidden_dim] per entry.
/// Produced by Septa::forward; consumed by Hyphae::forward to initialise span node features.
pub struct SpanHiddens<B: Backend> {
    pub intent:      Tensor<B, 1>,
    pub entity:      Tensor<B, 1>,
    pub projections: Vec<Tensor<B, 1>>,
    pub conditions:  Vec<Tensor<B, 1>>,
    pub assignments: Vec<Tensor<B, 1>>,
    pub modifiers:   Vec<Tensor<B, 1>>,
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
    /// Returns span boundaries/text in Semantics and mean-pooled BiLSTM hiddens in SpanHiddens.
    pub fn forward(&self, _tokens: &[&str], _device: &B::Device) -> (Semantics, SpanHiddens<B>) {
        todo!()
    }
}
