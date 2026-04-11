use std::path::PathBuf;

use crate::namer::{
    model::{NamerConfig, NamerModel},
    profiles::{NamerExample, Profile},
    vocab::{encode_value, WordVocab, CHAR_MAX_LEN},
};
use burn::{
    backend::{ndarray::NdArray, Autodiff},
    module::Module,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{ElementConversion, Int, Tensor},
};

type TrainBackend = Autodiff<NdArray>;

pub struct TrainConfig {
    pub epochs:     usize,
    pub batch_size: usize,
    pub lr:         f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self { epochs: 100, batch_size: 32, lr: 1e-3 }
    }
}

fn load_examples(profile: &Profile, base: &PathBuf) -> Vec<NamerExample> {
    let path = profile.training_data_path(base);
    std::fs::read_to_string(&path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Build vocab, train model, save both to the profile checkpoint directory.
pub fn train(profile: &Profile, base: &PathBuf, cfg: TrainConfig) {
    let examples = load_examples(profile, base);
    if examples.is_empty() {
        eprintln!("[hyphae] No training data for '{}', skipping.", profile.name());
        return;
    }

    // Build word vocabulary from all training names
    let vocab = WordVocab::build(examples.iter().map(|e| e.name.as_str()));
    eprintln!(
        "[hyphae] Training '{}' on {} examples, vocab size {}.",
        profile.name(), examples.len(), vocab.size()
    );

    let device = Default::default();
    let model_cfg = NamerConfig::new(vocab.size());
    let mut model: NamerModel<TrainBackend> = model_cfg.build(&device);
    let mut optim = AdamWConfig::new()
        .init::<TrainBackend, NamerModel<TrainBackend>>();

    // Pre-encode all examples into tensors
    let char_data: Vec<Vec<i32>> = examples.iter()
        .map(|e| encode_value(&e.value).iter().map(|&x| x as i32).collect())
        .collect();
    let targets: Vec<[usize; 2]> = examples.iter()
        .map(|e| vocab.encode_name(&e.name))
        .collect();

    let n = examples.len();

    for epoch in 0..cfg.epochs {
        let mut total_loss = 0f32;
        let mut steps = 0;

        for batch_start in (0..n).step_by(cfg.batch_size) {
            let batch_end = (batch_start + cfg.batch_size).min(n);
            let bs = batch_end - batch_start;

            let mut chars_flat = Vec::with_capacity(bs * CHAR_MAX_LEN);
            let mut t1_flat = Vec::with_capacity(bs);
            let mut t2_flat = Vec::with_capacity(bs);

            for i in batch_start..batch_end {
                chars_flat.extend_from_slice(&char_data[i]);
                t1_flat.push(targets[i][0] as i32);
                t2_flat.push(targets[i][1] as i32);
            }

            let chars = Tensor::<TrainBackend, 2, Int>::from_ints(chars_flat.as_slice(), &device)
                .reshape([bs, CHAR_MAX_LEN]);
            let t1 = Tensor::<TrainBackend, 1, Int>::from_ints(t1_flat.as_slice(), &device);
            let t2 = Tensor::<TrainBackend, 1, Int>::from_ints(t2_flat.as_slice(), &device);

            let loss = model.forward_loss(chars, t1, t2);
            total_loss += loss.clone().into_scalar().elem::<f32>();
            steps += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(cfg.lr, model, grads);
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            eprintln!(
                "[hyphae] epoch {}/{} loss={:.4}",
                epoch + 1, cfg.epochs,
                total_loss / steps as f32
            );
        }
    }

    // Save vocab + model checkpoint
    let out_dir = profile.checkpoint_dir(base);
    std::fs::create_dir_all(&out_dir).ok();

    vocab.save(&out_dir.join("vocab.txt"));

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(model.into_record(), profile.model_path(base).with_extension(""))
        .expect("failed to save model");

    eprintln!("[hyphae] Saved to {:?}", out_dir);
}
