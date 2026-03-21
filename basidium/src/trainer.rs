// trainer.rs — unified training loop for septa and hyphae models

use crate::Datum;
use burn::config::Config;
use burn::optim::LearningRate;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

#[derive(Config, Debug)]
pub struct TrainerConfig {
    #[config(default = 50)]
    pub epochs: usize,
    #[config(default = 1e-3)]
    pub learning_rate: LearningRate,
    #[config(default = 10)]
    pub patience: usize,
    #[config(default = 64)]
    pub batch_size: usize,
}

pub struct Metrics {
    pub train_loss: f32,
    pub val_loss: f32,
    pub train_acc: f32,
    pub val_acc: f32,
    pub f1: f32,
    pub head_acc: HeadAcc,
}

/// Per-head accuracy: (correct, total) for each bilinear resolution head.
#[derive(Default)]
pub struct HeadAcc {
    pub intent:     (usize, usize),
    pub entity:     (usize, usize),
    pub proj:       (usize, usize),
    pub cond_field: (usize, usize),
    pub cond_cmp:   (usize, usize),
    pub assign:     (usize, usize),
    pub mod_type:   (usize, usize),
    pub mod_field:  (usize, usize),
}

impl HeadAcc {
    pub fn acc(&self, (c, t): (usize, usize)) -> f32 {
        if t > 0 { c as f32 / t as f32 } else { 0.0 }
    }

    pub fn display(&self) -> String {
        let a = |pair: (usize, usize)| -> String {
            if pair.1 > 0 { format!("{:.0}%", pair.0 as f32 / pair.1 as f32 * 100.0) }
            else { "-".into() }
        };
        format!(
            "int={} ent={} proj={} cf={} cc={} asgn={} mt={} mf={}",
            a(self.intent), a(self.entity), a(self.proj),
            a(self.cond_field), a(self.cond_cmp), a(self.assign),
            a(self.mod_type), a(self.mod_field),
        )
    }
}

pub trait Trainable {
    /// Process a mini-batch. Returns mean loss for the batch.
    fn step_batch(&mut self, batch: &[&Datum]) -> f32;
    fn evaluate(&self, batch: &[&Datum], bar: &ProgressBar) -> Metrics;
    fn save(&self, path: &PathBuf) -> std::io::Result<()>;
}

pub struct Trainer<M: Trainable> {
    output_dir: PathBuf,
    config: TrainerConfig,
    model: M,
}

impl<M: Trainable> Trainer<M> {
    pub fn new<P: Into<PathBuf>>(config: TrainerConfig, model: M, output_dir: P) -> Self {
        Self {
            output_dir: output_dir.into(),
            config,
            model,
        }
    }

    pub fn train(&mut self, train_data: &[Datum], val_data: &[Datum]) -> TrainResult {
        let mut train_indices: Vec<usize> = (0..train_data.len()).collect();

        let bar = ProgressBar::new(self.config.epochs as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] epoch {pos}/{len} (eta {eta}) {msg}"
            ).unwrap()
            .progress_chars("=> "),
        );

        let mut best_val_loss = f32::MAX;
        let mut best_epoch = 0;
        let mut patience_counter = 0;
        let mut best_metrics = Metrics {
            train_loss: 0.0, val_loss: f32::MAX, train_acc: 0.0, val_acc: 0.0, f1: 0.0,
            head_acc: HeadAcc::default(),
        };

        let bs = self.config.batch_size;
        let num_batches = (train_indices.len() + bs - 1) / bs;

        for epoch in 0..self.config.epochs {
            shuffle(&mut train_indices, epoch as u64);

            let mut epoch_loss = 0.0f32;
            let mut epoch_datums = 0usize;
            let batch_bar = ProgressBar::new(num_batches as u64);
            batch_bar.set_style(
                ProgressStyle::with_template(
                    "  train [{bar:30.yellow/red}] {pos}/{len} batch  {per_sec}  eta {eta}"
                ).unwrap().progress_chars("=> "),
            );

            for chunk in train_indices.chunks(bs) {
                let batch: Vec<&Datum> = chunk.iter().map(|&i| &train_data[i]).collect();
                let batch_loss = self.model.step_batch(&batch);
                epoch_loss += batch_loss * batch.len() as f32;
                epoch_datums += batch.len();
                batch_bar.inc(1);
            }
            batch_bar.finish_and_clear();

            let train_loss = epoch_loss / epoch_datums as f32;
            let val_refs: Vec<&Datum> = val_data.iter().collect();
            let eval_bar = ProgressBar::new(((val_refs.len() + 31) / 32) as u64);
            eval_bar.set_style(
                ProgressStyle::with_template(
                    "  eval  [{bar:30.green/blue}] {pos}/{len} batch  {per_sec}  eta {eta}"
                ).unwrap().progress_chars("=> "),
            );
            let mut metrics = self.model.evaluate(&val_refs, &eval_bar);
            eval_bar.finish_and_clear();
            metrics.train_loss = train_loss;

            bar.set_message(format!(
                "loss={:.4} val={:.4} acc={:.1}%",
                train_loss, metrics.val_loss, metrics.val_acc * 100.0,
            ));
            bar.inc(1);
            bar.println(format!(
                "  epoch {:>2}  loss={:.4}  val={:.4}  acc={:.1}%  | {}",
                epoch + 1, train_loss, metrics.val_loss, metrics.val_acc * 100.0,
                metrics.head_acc.display(),
            ));

            if metrics.val_loss < best_val_loss {
                best_val_loss = metrics.val_loss;
                best_epoch = epoch;
                best_metrics = metrics;
                patience_counter = 0;

                let _ = std::fs::create_dir_all(&self.output_dir);
                let _ = self.model.save(&self.output_dir.join("best.bin"));
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    bar.finish_with_message(format!(
                        "early stop at epoch {} (best={})", epoch, best_epoch
                    ));
                    return TrainResult { best_epoch, best_metrics, weights_path: self.output_dir.join("best.bin") };
                }
            }
        }

        bar.finish_with_message(format!("done (best epoch={})", best_epoch));

        TrainResult {
            best_epoch,
            best_metrics,
            weights_path: self.output_dir.join("best.bin"),
        }
    }
}

/// Fisher-Yates shuffle with a simple LCG PRNG (no external dependency).
fn shuffle(v: &mut [usize], seed: u64) {
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    for i in (1..v.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
}

pub struct TrainResult {
    pub best_epoch: usize,
    pub best_metrics: Metrics,
    pub weights_path: PathBuf,
}
