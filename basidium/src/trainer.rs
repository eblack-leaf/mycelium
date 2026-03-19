// trainer.rs — unified training loop for septa and hyphae models

use crate::Datum;
use burn::config::Config;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::optim::{AdamWConfig, LearningRate};
use std::path::PathBuf;

#[derive(Config, Debug)]
pub struct TrainerConfig {
    #[config(default = 20)]
    pub epochs: usize,
    #[config(default = 1)]
    pub batch_size: usize,
    #[config(default = 1e-3)]
    pub learning_rate: LearningRate,
    #[config(default = 5)]
    pub patience: usize,
}

pub struct Metrics {
    pub train_loss: f32,
    pub val_loss: f32,
    pub train_acc: f32,
    pub val_acc: f32,
    pub f1: f32, // slot F1 for septa, node F1 for hyphae
}

pub trait Trainable {
    fn step(&mut self, batch: &[Datum]) -> f32; // returns train loss
    fn evaluate(&self, batch: &[Datum]) -> Metrics;
    fn save(&self, path: &PathBuf) -> std::io::Result<()>;
}

pub struct Trainer<M: Trainable> {
    output_dir: PathBuf,
    optimizer: AdamWConfig,
    scheduler: CosineAnnealingLrSchedulerConfig,
    config: TrainerConfig,
    model: M,
}

impl<M: Trainable> Trainer<M> {
    pub fn new<P: Into<PathBuf>>(config: TrainerConfig, model: M, output_dir: P) -> Self {
        Self {
            output_dir: output_dir.into(),
            optimizer: AdamWConfig::new(),
            scheduler: CosineAnnealingLrSchedulerConfig::new(config.learning_rate, config.epochs),
            config,
            model,
        }
    }

    pub fn train(&mut self, data: &[Datum]) -> TrainResult {
        let split = (data.len() as f32 * 0.9) as usize;
        let (train, val) = data.split_at(split);
        todo!()
    }
}

pub struct TrainResult {
    pub best_epoch: usize,
    pub best_metrics: Metrics,
    pub weights_path: PathBuf,
}
