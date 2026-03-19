// trainer.rs — unified training loop for septa and hyphae models

use std::path::PathBuf;
use crate::Datum;

pub struct TrainerConfig {
    pub output_dir:    PathBuf,
    pub epochs:        usize,
    pub batch_size:    usize,
    pub learning_rate: f32,
    pub patience:      usize,
    pub scheduler:     Scheduler,
    pub optimizer:     Optimizer,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            output_dir:    PathBuf::from("weights"),
            epochs:        100,
            batch_size:    32,
            learning_rate: 1e-3,
            patience:      10,
            scheduler:     Scheduler::ReduceOnPlateau { factor: 0.5, patience: 5 },
            optimizer:     Optimizer::Adam { beta1: 0.9, beta2: 0.999 },
        }
    }
}

#[derive(Debug, Clone)]
pub enum Scheduler {
    ReduceOnPlateau { factor: f32, patience: usize },
    CosineAnnealing { t_max: usize },
    StepLr { step_size: usize, gamma: f32 },
}

#[derive(Debug, Clone)]
pub enum Optimizer {
    Adam  { beta1: f32, beta2: f32 },
    AdamW { beta1: f32, beta2: f32, weight_decay: f32 },
    Sgd   { momentum: f32 },
}

pub struct Metrics {
    pub train_loss: f32,
    pub val_loss:   f32,
    pub train_acc:  f32,
    pub val_acc:    f32,
    pub f1:         f32,  // slot F1 for septa, node F1 for hyphae
}

pub trait Trainable {
    fn step(&mut self, batch: &[Datum]) -> f32;     // returns train loss
    fn evaluate(&self, batch: &[Datum]) -> Metrics;
    fn save(&self, path: &PathBuf) -> std::io::Result<()>;
}

pub struct Trainer<M: Trainable> {
    config: TrainerConfig,
    model:  M,
}

impl<M: Trainable> Trainer<M> {
    pub fn new(config: TrainerConfig, model: M) -> Self {
        Self { config, model }
    }

    pub fn train(&mut self, data: &[Datum]) -> TrainResult {
        let split        = (data.len() as f32 * 0.9) as usize;
        let (train, val) = data.split_at(split);
        todo!()
    }
}

pub struct TrainResult {
    pub best_epoch:   usize,
    pub best_metrics: Metrics,
    pub weights_path: PathBuf,
}
