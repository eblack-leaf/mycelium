//! Train the schema re-ranker.
//!
//! Small MLP that replaces the pretrained cross-encoder for candidate scoring.
//! Binary classification: does (phrase, schema_name) match?
//!
//! Usage:
//!   cargo run --release --example gen_reranker_dataset -p gnn-burn
//!   cargo run --release --example train_reranker -p gnn-burn

use std::path::Path;
use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Tensor, TensorData},
    optim::{AdamConfig, GradientsParams, Optimizer},
    module::{Module, AutodiffModule},
    record::{CompactRecorder, Recorder},
};
use gnn_burn::reranker::Reranker;
use gnn_burn::reranker_data::RerankerDataset;

type B = Autodiff<NdArray>;

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");
    let device = Default::default();

    // --- Load dataset ---
    let ds = RerankerDataset::load(&demo_dir.join("reranker_dataset.json"))
        .expect("load reranker dataset — run gen_reranker_dataset first");
    println!("loaded {} pairs", ds.pairs.len());

    // --- Train/val split ---
    let n_val = ds.pairs.len() / 5;
    let n_train = ds.pairs.len() - n_val;
    let train_pairs = &ds.pairs[..n_train];
    let val_pairs = &ds.pairs[n_train..];
    println!("split: {} train, {} val", train_pairs.len(), val_pairs.len());

    // --- Init model + optimizer ---
    let model: Reranker<B> = Reranker::new(&device);
    let mut model = model;
    let mut optim = AdamConfig::new().init();
    let batch_size = 64;
    let epochs = 30;
    let lr = 0.001;
    let patience = 6;

    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0usize;
    let output_path = demo_dir.join("reranker_model");

    for epoch in 0..epochs {
        // --- Train ---
        let mut train_loss_sum = 0.0f32;
        let mut train_correct = 0usize;
        let mut train_total = 0usize;
        let n_batches = (train_pairs.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(train_pairs.len());
            let batch = &train_pairs[start..end];
            let bs = batch.len();

            // Build tensors
            let phrase_data: Vec<f32> = batch.iter().flat_map(|p| p.phrase_emb.iter().copied()).collect();
            let schema_data: Vec<f32> = batch.iter().flat_map(|p| p.schema_emb.iter().copied()).collect();
            let label_data: Vec<f32> = batch.iter().map(|p| p.label).collect();

            let phrase_t: Tensor<B, 2> = Tensor::from_data(
                TensorData::new(phrase_data, [bs, 384]), &device
            );
            let schema_t: Tensor<B, 2> = Tensor::from_data(
                TensorData::new(schema_data, [bs, 384]), &device
            );
            let labels: Tensor<B, 1> = Tensor::from_data(
                TensorData::new(label_data.clone(), [bs]), &device
            );

            // Forward
            let logits = model.forward(phrase_t, schema_t); // [bs]

            // Binary cross-entropy with logits
            // BCE = max(logit, 0) - logit * label + log(1 + exp(-|logit|))
            let zeros: Tensor<B, 1> = Tensor::zeros([bs], &device);
            let max_logit = logits.clone().max_pair(zeros);
            let neg_abs = logits.clone().abs().mul_scalar(-1.0);
            let log_term = neg_abs.exp().add_scalar(1.0).log();
            let bce = max_logit - logits.clone() * labels.clone() + log_term;
            let loss = bce.mean();

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            train_loss_sum += loss_val * bs as f32;

            // Accuracy
            let preds: Vec<f32> = logits.clone().into_data().to_vec().unwrap();
            for (pred, &lab) in preds.iter().zip(label_data.iter()) {
                let pred_label = if *pred > 0.0 { 1.0 } else { 0.0 };
                if (pred_label - lab).abs() < 0.5 { train_correct += 1; }
                train_total += 1;
            }

            // Backward
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);
        }

        // --- Validate ---
        let valid_model = model.valid();
        let mut val_loss_sum = 0.0f32;
        let mut val_correct = 0usize;
        let mut val_total = 0usize;
        let n_val_batches = (val_pairs.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..n_val_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(val_pairs.len());
            let batch = &val_pairs[start..end];
            let bs = batch.len();

            let phrase_data: Vec<f32> = batch.iter().flat_map(|p| p.phrase_emb.iter().copied()).collect();
            let schema_data: Vec<f32> = batch.iter().flat_map(|p| p.schema_emb.iter().copied()).collect();
            let label_data: Vec<f32> = batch.iter().map(|p| p.label).collect();

            let phrase_t: Tensor<NdArray, 2> = Tensor::from_data(
                TensorData::new(phrase_data, [bs, 384]), &device
            );
            let schema_t: Tensor<NdArray, 2> = Tensor::from_data(
                TensorData::new(schema_data, [bs, 384]), &device
            );
            let labels: Tensor<NdArray, 1> = Tensor::from_data(
                TensorData::new(label_data.clone(), [bs]), &device
            );

            let logits = valid_model.forward(phrase_t, schema_t);

            let zeros: Tensor<NdArray, 1> = Tensor::zeros([bs], &device);
            let max_logit = logits.clone().max_pair(zeros);
            let neg_abs = logits.clone().abs().mul_scalar(-1.0);
            let log_term = neg_abs.exp().add_scalar(1.0).log();
            let bce = max_logit - logits.clone() * labels + log_term;
            let loss = bce.mean();

            val_loss_sum += loss.into_data().to_vec::<f32>().unwrap()[0] * bs as f32;

            let preds: Vec<f32> = logits.into_data().to_vec().unwrap();
            for (pred, &lab) in preds.iter().zip(label_data.iter()) {
                let pred_label: f32 = if *pred > 0.0 { 1.0 } else { 0.0 };
                if (pred_label - lab).abs() < 0.5 { val_correct += 1; }
                val_total += 1;
            }
        }

        let train_avg = train_loss_sum / train_total as f32;
        let val_avg = val_loss_sum / val_total as f32;
        let train_acc = train_correct as f64 / train_total as f64;
        let val_acc = val_correct as f64 / val_total as f64;

        println!(
            "epoch {:>2}: loss={:.4}/{:.4} acc={:.1}%/{:.1}%",
            epoch, train_avg, val_avg,
            train_acc * 100.0, val_acc * 100.0,
        );

        // Early stopping
        if val_avg < best_val_loss {
            best_val_loss = val_avg;
            epochs_without_improvement = 0;
            CompactRecorder::new()
                .record(model.clone().into_record(), output_path.clone())
                .expect("save model");
        } else {
            epochs_without_improvement += 1;
            if epochs_without_improvement >= patience {
                println!("early stopping at epoch {}", epoch);
                break;
            }
        }
    }

    println!("best val_loss={:.4}, saved to {:?}", best_val_loss, output_path);
}
