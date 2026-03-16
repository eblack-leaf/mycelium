// =============================================================================
// training.rs — Training loop, loss, dataset
// =============================================================================

use std::path::Path;
use burn::tensor::backend::Backend;
use crate::head::ResolvedGraph;
use crate::intent::Extraction;

/// One training example: NL query + expected resolved output.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub nl_query: String,
    pub schema_path: String,
    pub expected: ResolvedGraph,
}

/// Training dataset.
pub struct Dataset {
    pub samples: Vec<TrainingSample>,
}

impl Dataset {
    pub fn load(_path: &Path) -> Self {
        todo!()
    }
}

/// Training configuration.
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
}

/// Compute loss between predicted and expected resolutions.
pub fn compute_loss<B: Backend>(
    _predicted: &ResolvedGraph,
    _expected: &ResolvedGraph,
) -> f32 {
    todo!()
}

/// Run training loop.
pub fn train<B: Backend>(
    _config: &TrainingConfig,
    _dataset: &Dataset,
    _device: &B::Device,
) {
    todo!()
}
