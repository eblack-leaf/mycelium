//! Train the biaffine head for BIO tagging + dependency parsing.
//!
//! Loads BiaffineDataset + MiniLM ONNX session, trains BiaffineHead on:
//!   - BIO loss (cross-entropy per token)
//!   - Arc loss (binary cross-entropy on biaffine arc scores)
//!   - Relation loss (cross-entropy at GT arc positions)
//!
//! Span embeddings use teacher forcing: mean-pool ground-truth span boundaries.
//!
//! Usage:
//!   cargo run --release --example gen_biaffine_dataset -p gnn-burn
//!   cargo run --release --example train_biaffine -p gnn-burn

use std::path::Path;
use burn::{
    module::{Module, AutodiffModule},
    backend::{Autodiff, NdArray},
    optim::{AdamConfig, GradientsParams, Optimizer},
    lr_scheduler::{LrScheduler, cosine::CosineAnnealingLrSchedulerConfig},
    record::CompactRecorder,
    tensor::{Tensor, TensorData},
};
use indicatif::{ProgressBar, ProgressStyle};
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use tokenizers::Tokenizer;

use gnn_burn::biaffine::{BiaffineHead, mean_pool_spans, HIDDEN_DIM};
use gnn_burn::biaffine_data::{BiaffineDataset, BioTag};
use gnn_burn::nlp::DepRelation;

type B = NdArray;
type AB = Autodiff<B>;

const LEARNING_RATE: f64 = 3e-4;
const EPOCHS: usize = 30;
const PATIENCE: usize = 6;

/// Run MiniLM ONNX to get token-level embeddings [seq_len, 384] (no CLS/SEP).
fn encode_tokens(
    session: &mut Session,
    tokenizer: &Tokenizer,
    text: &str,
) -> Option<Vec<f32>> {
    let encoding = tokenizer.encode(text, true).ok()?;
    let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
    let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
    let seq_len = ids.len();

    let ids_tensor = OrtTensor::from_array((vec![1i64, seq_len as i64], ids)).ok()?;
    let mask_tensor = OrtTensor::from_array((vec![1i64, seq_len as i64], mask)).ok()?;
    let type_tensor = OrtTensor::from_array((vec![1i64, seq_len as i64], type_ids)).ok()?;

    let outputs = session.run(ort::inputs![ids_tensor, mask_tensor, type_tensor]).ok()?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>().ok()?;
    let hidden_dim = shape[2] as usize;

    // Extract content tokens (skip CLS=0 and SEP=last)
    let n_content = seq_len.saturating_sub(2);
    let mut token_embs = Vec::with_capacity(n_content * hidden_dim);
    for i in 1..=n_content {
        let offset = i * hidden_dim;
        token_embs.extend_from_slice(&data[offset..offset + hidden_dim]);
    }

    Some(token_embs)
}

/// Compute BIO cross-entropy loss (one-hot targets).
fn bio_loss(logits: Tensor<AB, 2>, targets: &[usize]) -> Tensor<AB, 1> {
    let device = logits.device();
    let seq_len = targets.len();
    let n_classes = BioTag::COUNT;
    // Build one-hot target matrix [seq_len, n_classes]
    let mut target_data = vec![0.0f32; seq_len * n_classes];
    for (i, &t) in targets.iter().enumerate() {
        if t < n_classes {
            target_data[i * n_classes + t] = 1.0;
        }
    }
    let target_tensor = Tensor::<AB, 2>::from_data(
        TensorData::new(target_data, [seq_len, n_classes]),
        &device,
    );
    burn::tensor::loss::cross_entropy_with_logits(logits, target_tensor)
}

/// Compute arc binary cross-entropy loss.
/// arc_scores: [n, n] (sigmoid applied), gt_arcs: (src, dst) pairs
fn arc_loss(arc_scores: Tensor<AB, 2>, n_spans: usize, gt_arcs: &[(usize, usize, usize)]) -> Tensor<AB, 1> {
    if n_spans == 0 { return Tensor::zeros([1], &arc_scores.device()); }
    let device = arc_scores.device();

    // Build binary target matrix
    let mut target_data = vec![0.0f32; n_spans * n_spans];
    for &(src, dst, _) in gt_arcs {
        if src < n_spans && dst < n_spans {
            target_data[src * n_spans + dst] = 1.0;
        }
    }
    let targets = Tensor::<AB, 2>::from_data(
        TensorData::new(target_data, [n_spans, n_spans]),
        &device,
    );

    // Binary cross-entropy: -[t*log(p) + (1-t)*log(1-p)]
    let eps: f32 = 1e-7;
    let scores_clamped = arc_scores.clone().clamp(eps, 1.0 - eps);
    let bce = targets.clone().neg()
        * scores_clamped.clone().log()
        - (targets.neg() + 1.0)
        * (scores_clamped.neg() + 1.0).log();

    bce.mean()
}

/// Compute relation cross-entropy loss at ground-truth arc positions.
/// rel_logits: [n, n, 4], gt_arcs: (src, dst, rel_idx)
fn rel_loss(rel_logits: Tensor<AB, 3>, n_spans: usize, gt_arcs: &[(usize, usize, usize)]) -> Tensor<AB, 1> {
    let device = rel_logits.device();
    if gt_arcs.is_empty() || n_spans == 0 {
        return Tensor::zeros([1], &device);
    }

    // Gather logits at GT arc positions and compute CE
    let n_rels = DepRelation::COUNT;
    let mut logit_rows: Vec<Tensor<AB, 2>> = Vec::new();
    let mut target_indices: Vec<i32> = Vec::new();

    for &(src, dst, rel_idx) in gt_arcs {
        if src >= n_spans || dst >= n_spans { continue; }
        // Extract rel_logits[src, dst, :] → [1, n_rels]
        let row = rel_logits.clone()
            .slice([src..src+1, dst..dst+1, 0..n_rels])
            .reshape([1, n_rels]);
        logit_rows.push(row);
        target_indices.push(rel_idx as i32);
    }

    if logit_rows.is_empty() {
        return Tensor::zeros([1], &device);
    }

    let logits_cat = Tensor::cat(logit_rows, 0); // [k, 4]
    let k = target_indices.len();
    // Build one-hot targets [k, n_rels]
    let mut target_data = vec![0.0f32; k * n_rels];
    for (i, &rel_idx) in target_indices.iter().enumerate() {
        if (rel_idx as usize) < n_rels {
            target_data[i * n_rels + rel_idx as usize] = 1.0;
        }
    }
    let targets = Tensor::<AB, 2>::from_data(
        TensorData::new(target_data, [k, n_rels]),
        &device,
    );

    burn::tensor::loss::cross_entropy_with_logits(logits_cat, targets)
}

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");

    // Load dataset
    let dataset_path = demo_dir.join("biaffine_dataset.json");
    let dataset = BiaffineDataset::load(&dataset_path).expect("load biaffine dataset");
    println!("loaded {} samples", dataset.samples.len());

    // Load MiniLM ONNX + tokenizer
    let mut session = Session::builder().unwrap()
        .with_intra_threads(4).unwrap()
        .commit_from_file(model_dir.join("model.onnx")).unwrap();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .expect("load tokenizer");

    // Split train/val (90/10)
    let n_val = (dataset.samples.len() / 10).max(1);
    let val_samples = &dataset.samples[..n_val];
    let train_samples = &dataset.samples[n_val..];
    println!("train: {}, val: {}", train_samples.len(), val_samples.len());

    // Init model + optimizer
    let device = Default::default();
    let model = BiaffineHead::<AB>::new(&device);
    let mut optim = AdamConfig::new().init::<AB, BiaffineHead<AB>>();
    let total_steps = EPOCHS * train_samples.len();
    let mut scheduler = CosineAnnealingLrSchedulerConfig::new(LEARNING_RATE, total_steps)
        .init()
        .expect("init scheduler");

    let mut best_val_loss = f64::MAX;
    let mut patience_counter = 0usize;
    let mut model = model;

    let output_path = demo_dir.join("biaffine_model");

    for epoch in 0..EPOCHS {
        // --- Train ---
        let pb = ProgressBar::new(train_samples.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template(&format!("epoch {}/{} [{{bar:40}}] {{pos}}/{{len}} {{msg}}", epoch + 1, EPOCHS))
            .unwrap());

        let mut epoch_loss = 0.0f64;
        let mut epoch_bio_correct = 0usize;
        let mut epoch_bio_total = 0usize;

        for sample in train_samples {
            // Get token embeddings from MiniLM
            let token_data = match encode_tokens(&mut session, &tokenizer, &sample.query) {
                Some(d) => d,
                None => { pb.inc(1); continue; }
            };

            let seq_len = sample.bio_tags.len();
            let actual_tokens = token_data.len() / HIDDEN_DIM;
            let use_len = seq_len.min(actual_tokens);
            if use_len == 0 { pb.inc(1); continue; }

            // Truncate to matching length
            let token_embs = Tensor::<AB, 2>::from_data(
                TensorData::new(token_data[..use_len * HIDDEN_DIM].to_vec(), [use_len, HIDDEN_DIM]),
                &device,
            );

            // Task 1: BIO tagging
            let bio_logits = model.forward_bio(token_embs.clone());
            let loss_bio = bio_loss(bio_logits.clone(), &sample.bio_tags[..use_len]);

            // BIO accuracy
            let bio_preds: Vec<usize> = {
                let data = bio_logits.clone().into_data();
                let vals: Vec<f32> = data.to_vec().unwrap();
                (0..use_len).map(|i| {
                    let row = &vals[i * BioTag::COUNT..(i + 1) * BioTag::COUNT];
                    row.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap().0
                }).collect()
            };
            for (pred, &gt) in bio_preds.iter().zip(&sample.bio_tags[..use_len]) {
                if *pred == gt { epoch_bio_correct += 1; }
                epoch_bio_total += 1;
            }

            // Task 2: Dependency parsing (teacher forcing — use GT span boundaries)
            let n_spans = sample.span_boundaries.len();
            let (loss_arc, loss_rel) = if n_spans > 0 {
                let span_bounds: Vec<(usize, usize)> = sample.span_boundaries.iter()
                    .map(|&(s, e, _)| (s, e))
                    .collect();
                let span_embs = mean_pool_spans(&token_embs, &sample.subword_to_word[..use_len], &span_bounds);
                let (arc_scores, rel_logits) = model.forward_deps(span_embs);

                let la = arc_loss(arc_scores, n_spans, &sample.arcs);
                let lr = rel_loss(rel_logits, n_spans, &sample.arcs);
                (la, lr)
            } else {
                (Tensor::zeros([1], &device), Tensor::zeros([1], &device))
            };

            let total_loss = loss_bio + loss_arc + loss_rel;
            let loss_val: f32 = total_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_val as f64;

            // Backward + step
            let grads = total_loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &model);
            let lr = scheduler.step();
            model = optim.step(lr, model, grad_params);

            pb.inc(1);
        }

        let avg_loss = epoch_loss / train_samples.len() as f64;
        let bio_acc = if epoch_bio_total > 0 {
            epoch_bio_correct as f64 / epoch_bio_total as f64
        } else { 0.0 };
        pb.finish_with_message(format!("loss={:.4} bio_acc={:.4}", avg_loss, bio_acc));

        // --- Validation ---
        let model_valid = model.valid();
        let mut val_loss = 0.0f64;
        let mut val_bio_correct = 0usize;
        let mut val_bio_total = 0usize;

        for sample in val_samples {
            let token_data = match encode_tokens(&mut session, &tokenizer, &sample.query) {
                Some(d) => d,
                None => continue,
            };

            let seq_len = sample.bio_tags.len();
            let actual_tokens = token_data.len() / HIDDEN_DIM;
            let use_len = seq_len.min(actual_tokens);
            if use_len == 0 { continue; }

            let token_embs = Tensor::<B, 2>::from_data(
                TensorData::new(token_data[..use_len * HIDDEN_DIM].to_vec(), [use_len, HIDDEN_DIM]),
                &device,
            );

            let bio_logits = model_valid.forward_bio(token_embs.clone());

            // BIO accuracy
            let bio_data = bio_logits.clone().into_data();
            let bio_vals: Vec<f32> = bio_data.to_vec().unwrap();
            for i in 0..use_len {
                let row = &bio_vals[i * BioTag::COUNT..(i + 1) * BioTag::COUNT];
                let pred = row.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap().0;
                if pred == sample.bio_tags[i] { val_bio_correct += 1; }
                val_bio_total += 1;
            }

            // Dep loss
            let n_spans = sample.span_boundaries.len();
            if n_spans > 0 {
                let span_bounds: Vec<(usize, usize)> = sample.span_boundaries.iter()
                    .map(|&(s, e, _)| (s, e))
                    .collect();
                let span_embs = mean_pool_spans(&token_embs, &sample.subword_to_word[..use_len], &span_bounds);
                let (arc_scores, _rel_logits) = model_valid.forward_deps(span_embs);

                // Simple val loss estimate: arc BCE
                let mut target_data = vec![0.0f32; n_spans * n_spans];
                for &(src, dst, _) in &sample.arcs {
                    if src < n_spans && dst < n_spans {
                        target_data[src * n_spans + dst] = 1.0;
                    }
                }
                let arc_data = arc_scores.into_data();
                let arc_vals: Vec<f32> = arc_data.to_vec().unwrap();
                for (p, t) in arc_vals.iter().zip(target_data.iter()) {
                    let p_clamped = p.clamp(1e-7, 1.0 - 1e-7);
                    val_loss += -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln()) as f64;
                }
            }
        }

        let avg_val_loss = val_loss / val_samples.len().max(1) as f64;
        let val_bio_acc = if val_bio_total > 0 {
            val_bio_correct as f64 / val_bio_total as f64
        } else { 0.0 };
        println!("  val: loss={:.4} bio_acc={:.4}", avg_val_loss, val_bio_acc);

        // Early stopping
        if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            patience_counter = 0;
            // Save best model
            model.clone()
                .save_file(output_path.clone(), &CompactRecorder::new())
                .expect("save model");
            println!("  saved -> {:?}", output_path);
        } else {
            patience_counter += 1;
            if patience_counter >= PATIENCE {
                println!("early stopping at epoch {}", epoch + 1);
                break;
            }
        }
    }

    println!("done. best val loss: {:.4}", best_val_loss);
}
