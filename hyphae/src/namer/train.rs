use std::path::PathBuf;

use crate::namer::{
    model::{NamerConfig, NamerModel},
    profiles::{NamerExample, Profile},
    vocab::{encode_input, encode_output, IN_MAX_LEN, OUT_MAX_LEN},
};
use burn::{
    backend::{Autodiff, ndarray::NdArray},
    module::Module,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{backend::Backend, ElementConversion, Int, Tensor},
};

type TrainBackend = Autodiff<NdArray>;

pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self { epochs: 50, batch_size: 16, lr: 1e-3 }
    }
}

/// Load examples from the profile's JSONL training file.
fn load_examples(profile: &Profile, base: &PathBuf) -> Vec<NamerExample> {
    let path = profile.training_data_path(base);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_default();
    text.lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn examples_to_tensors(
    examples: &[NamerExample],
    device: &<TrainBackend as Backend>::Device,
) -> (Tensor<TrainBackend, 2, Int>, Tensor<TrainBackend, 2, Int>) {
    let mut inputs = Vec::with_capacity(examples.len() * IN_MAX_LEN);
    let mut targets = Vec::with_capacity(examples.len() * OUT_MAX_LEN);

    for ex in examples {
        let combined = format!("{}|{}", ex.value, ex.context);
        inputs.extend(encode_input(&combined).iter().map(|&x| x as i32));
        targets.extend(encode_output(&ex.name).iter().map(|&x| x as i32));
    }

    let n = examples.len();
    let inp = Tensor::<TrainBackend, 2, Int>::from_ints(inputs.as_slice(), device)
        .reshape([n, IN_MAX_LEN]);
    let tgt = Tensor::<TrainBackend, 2, Int>::from_ints(targets.as_slice(), device)
        .reshape([n, OUT_MAX_LEN]);
    (inp, tgt)
}

/// Train a profile from scratch using its JSONL training data, save checkpoint.
pub fn train(profile: &Profile, base: &PathBuf, cfg: TrainConfig) {
    let device = Default::default();
    let examples = load_examples(profile, base);
    if examples.is_empty() {
        eprintln!("[hyphae] No training data for profile '{}', skipping.", profile.name());
        return;
    }
    eprintln!("[hyphae] Training '{}' on {} examples.", profile.name(), examples.len());

    let model_cfg = NamerConfig::new();
    let mut model: NamerModel<TrainBackend> = model_cfg.build(&device);
    let mut optim = AdamWConfig::new().init::<TrainBackend, NamerModel<TrainBackend>>();

    let (all_inputs, all_targets) = examples_to_tensors(&examples, &device);

    for epoch in 0..cfg.epochs {
        let mut total_loss = 0f32;
        let mut steps = 0usize;
        let n = examples.len();

        for batch_start in (0..n).step_by(cfg.batch_size) {
            let batch_end = (batch_start + cfg.batch_size).min(n);
            let inp = all_inputs.clone().slice([batch_start..batch_end, 0..IN_MAX_LEN]);
            let tgt = all_targets.clone().slice([batch_start..batch_end, 0..OUT_MAX_LEN]);

            let loss = model.forward_loss(inp, tgt);
            total_loss += loss.clone().into_scalar().elem::<f32>();
            steps += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(cfg.lr, model, grads);
        }

        eprintln!("[hyphae] epoch {}/{} loss={:.4}", epoch + 1, cfg.epochs, total_loss / steps as f32);
    }

    // Save checkpoint
    let out_dir = profile.checkpoint_dir(base);
    std::fs::create_dir_all(&out_dir).ok();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(model.into_record(), profile.model_path(base).with_extension(""))
        .expect("failed to save model");
    eprintln!("[hyphae] Saved checkpoint to {:?}", profile.model_path(base));
}
