//! Train the n-gram cross-attention model (multi-label).
//!
//! Loads MiniLM ONNX for token embeddings, trains NgramCrossAttn with:
//!   - Multi-label BCE: n-grams inherit ALL concept labels from GT spans they
//!     contain. "age over 25" (trigram) inherits {field:age, op:gt}.
//!     Uses sigmoid per concept with pos-weighted BCE.
//!   - Type loss: cross-entropy on span type classification
//!
//! Usage:
//!   cargo run --release --example gen_ngram_dataset -p gnn-burn
//!   cargo run --release --example train_ngram -p gnn-burn

use std::path::Path;
use std::io::Write;
use indicatif::{ProgressBar, ProgressStyle};
use burn::{
    module::{Module, AutodiffModule},
    backend::{Autodiff, NdArray},
    optim::{AdamConfig, GradientsParams, Optimizer},
    lr_scheduler::{LrScheduler, cosine::CosineAnnealingLrSchedulerConfig},
    record::CompactRecorder,
    tensor::{Tensor, Int, TensorData, activation},
};
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use tokenizers::Tokenizer;

use gnn_burn::schema::{Reader, Extractor};
use gnn_burn::graph::SchemaGraph;
use gnn_burn::operations::all_operations;
use gnn_burn::ngram_data::{NgramDataset, ConceptMap};
use gnn_burn::ngram_attn::{NgramCrossAttn, words_from_subwords, generate_ngrams, MINILM_DIM};
use gnn_burn::biaffine_data::build_subword_to_word;
use gnn_burn::nlp::SpanType;

type B = Autodiff<NdArray>;

/// Encode text with MiniLM, return content token embeddings (no CLS/SEP).
fn encode_tokens(
    session: &mut Session,
    tokenizer: &Tokenizer,
    text: &str,
) -> Option<(Vec<f32>, Vec<usize>)> {
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

    let n_content = seq_len.saturating_sub(2);
    let mut token_embs = Vec::with_capacity(n_content * hidden_dim);
    for i in 1..=n_content {
        let offset = i * hidden_dim;
        token_embs.extend_from_slice(&data[offset..offset + hidden_dim]);
    }

    let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
    let subword_to_word = build_subword_to_word(&offsets, text);

    Some((token_embs, subword_to_word))
}

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");

    // --- Load dataset ---
    let dataset = NgramDataset::load(&demo_dir.join("ngram_dataset.json"))
        .expect("load ngram dataset — run gen_ngram_dataset first");
    println!("loaded {} samples", dataset.samples.len());

    // --- Load schema for concept map ---
    let raw = Reader::read(&demo_dir.join("schema.surql")).expect("read schema");
    let (_, _) = Extractor::extract(&raw);
    let schema_graph = SchemaGraph::from_schema(&Extractor::extract(&Reader::read(&demo_dir.join("schema.surql")).unwrap()).0);
    let operations = all_operations();

    let table_names: Vec<String> = schema_graph.table_nodes.iter()
        .map(|n| n.name.clone()).collect();
    let field_names: Vec<String> = schema_graph.field_nodes.iter()
        .map(|n| n.name.splitn(2, '.').nth(1).unwrap_or(&n.name).to_string()).collect();
    let op_names: Vec<String> = operations.iter()
        .map(|op| op.name.clone()).collect();
    let concept_map = ConceptMap::new(&table_names, &field_names, &op_names);
    let n_concepts = concept_map.total();
    println!("concepts: {} ({} tables, {} fields, {} ops)",
        n_concepts, concept_map.n_tables, concept_map.n_fields, concept_map.n_ops);

    // --- Load MiniLM ---
    println!("Loading MiniLM...");
    let mut session = Session::builder().unwrap()
        .with_intra_threads(4).unwrap()
        .commit_from_file(model_dir.join("model.onnx")).unwrap();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .expect("load tokenizer");

    // --- Train/val split ---
    let n = dataset.samples.len();
    let n_train = (n as f64 * 0.8) as usize;
    let train_samples = &dataset.samples[..n_train];
    let val_samples = &dataset.samples[n_train..];
    println!("split: {} train, {} val", train_samples.len(), val_samples.len());

    // --- Init model ---
    let device = &Default::default();
    let mut model = NgramCrossAttn::<B>::new(n_concepts, device);
    let mut optim = AdamConfig::new().with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4))).init();

    let epochs = 50;
    let lr = 0.0005;
    let patience = 10;
    let mut scheduler = CosineAnnealingLrSchedulerConfig::new(lr, epochs)
        .init().expect("valid scheduler config");

    // --- Metrics CSV ---
    let metrics_path = demo_dir.join("ngram_metrics.csv");
    let mut metrics_file = std::fs::File::create(&metrics_path).expect("create metrics");
    writeln!(metrics_file, "epoch,train_loss,val_loss,train_concept_acc,val_concept_acc,lr").unwrap();

    let pb_style = ProgressStyle::default_bar()
        .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("##-");

    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0usize;
    let model_path = demo_dir.join("ngram_model");

    for epoch in 0..epochs {
        // --- Train ---
        let mut train_loss_sum = 0.0f32;
        let mut train_type_correct = 0usize;
        let mut train_type_total = 0usize;
        let mut train_count = 0usize;

        let pb = ProgressBar::new(train_samples.len() as u64);
        pb.set_style(pb_style.clone());
        pb.set_message(format!("epoch {:>3} train", epoch));

        let mut current_lr = 0.0;
        for sample in train_samples {
            let result = encode_tokens(&mut session, &tokenizer, &sample.query);
            let (token_embs, subword_to_word) = match result {
                Some(r) => r,
                None => { pb.inc(1); continue; }
            };

            let (word_embs, n_words) = words_from_subwords(&token_embs, &subword_to_word);
            if n_words == 0 { pb.inc(1); continue; }

            let (ngram_embs_flat, spans) = generate_ngrams(&word_embs, n_words);
            let n_ngrams = spans.len();
            if n_ngrams == 0 { pb.inc(1); continue; }

            // Forward
            let ngram_tensor = Tensor::<B, 2>::from_data(
                TensorData::new(ngram_embs_flat, [n_ngrams, MINILM_DIM]), device,
            );
            let (affinity, type_logits) = model.forward_scores(ngram_tensor);

            // Build targets: each GT span → exact-match n-gram gets its concept label
            let mut type_targets_vec: Vec<(usize, usize)> = Vec::new(); // (ngram_idx, span_type)

            let gt_ngram_indices: Vec<usize> = sample.spans.iter()
                .filter_map(|gs| spans.iter().position(|&(s, e)| s == gs.start_word && e == gs.end_word))
                .collect();

            if gt_ngram_indices.is_empty() {
                pb.inc(1);
                continue;
            }

            for gt_span in &sample.spans {
                if let Some(ng_idx) = spans.iter().position(|&(s, e)| {
                    s == gt_span.start_word && e == gt_span.end_word
                }) {
                    type_targets_vec.push((ng_idx, gt_span.span_type));
                }
            }

            // Affinity loss: cross-entropy over concepts (softmax)
            let mut affinity_loss = Tensor::<B, 1>::zeros([1], device);
            let mut n_aff_terms = 0;

            for (gt_i, gt_span) in sample.spans.iter().enumerate() {
                if gt_i >= gt_ngram_indices.len() { break; }
                let ng_idx = gt_ngram_indices[gt_i];

                let row = affinity.clone().slice([ng_idx..ng_idx + 1, 0..n_concepts]);
                let log_probs = activation::log_softmax(row, 1);
                let target_idx = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(vec![gt_span.concept_idx as i32], [1, 1]), device,
                );
                let nll = log_probs.gather(1, target_idx).neg();
                affinity_loss = affinity_loss + nll.reshape([1]);
                n_aff_terms += 1;
            }

            if n_aff_terms > 0 {
                affinity_loss = affinity_loss / n_aff_terms as f32;
            }

            // Type loss: cross-entropy on GT n-grams
            let mut type_loss = Tensor::<B, 1>::zeros([1], device);
            let mut has_type_loss = false;
            for &(ng_idx, span_type) in &type_targets_vec {
                let logit_row = type_logits.clone().slice([ng_idx..ng_idx + 1, 0..SpanType::COUNT]);
                let log_probs = activation::log_softmax(logit_row, 1);
                let target_idx = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(vec![span_type as i32], [1, 1]), device,
                );
                let nll = log_probs.gather(1, target_idx).neg();
                type_loss = type_loss + nll.reshape([1]);
                has_type_loss = true;

                // Track concept accuracy (argmax of affinity row)
                let aff_row: Vec<f32> = affinity.clone()
                    .slice([ng_idx..ng_idx + 1, 0..n_concepts])
                    .into_data().to_vec().unwrap();
                let pred_concept = aff_row.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap().0;
                let gt_concept = sample.spans.iter()
                    .find(|s| s.start_word == spans[ng_idx].0 && s.end_word == spans[ng_idx].1)
                    .map(|s| s.concept_idx).unwrap_or(usize::MAX);
                train_type_total += 1;
                if pred_concept == gt_concept { train_type_correct += 1; }
            }
            if has_type_loss {
                type_loss = type_loss / type_targets_vec.len() as f32;
            }

            let total_loss = affinity_loss + type_loss;
            let loss_val = total_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            train_loss_sum += loss_val;
            train_count += 1;

            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            current_lr = scheduler.step();
            model = optim.step(current_lr, model, grads);

            pb.inc(1);
        }
        pb.finish_and_clear();

        // --- Validate ---
        let mut val_loss_sum = 0.0f32;
        let mut val_type_correct = 0usize;
        let mut val_type_total = 0usize;
        let mut val_count = 0usize;

        let pb = ProgressBar::new(val_samples.len() as u64);
        pb.set_style(pb_style.clone());
        pb.set_message(format!("epoch {:>3} val  ", epoch));

        let valid_model = model.valid();
        for sample in val_samples {
            let result = encode_tokens(&mut session, &tokenizer, &sample.query);
            let (token_embs, subword_to_word) = match result {
                Some(r) => r,
                None => { pb.inc(1); continue; }
            };

            let (word_embs, n_words) = words_from_subwords(&token_embs, &subword_to_word);
            if n_words == 0 { pb.inc(1); continue; }

            let (ngram_embs_flat, spans) = generate_ngrams(&word_embs, n_words);
            let n_ngrams = spans.len();
            if n_ngrams == 0 { pb.inc(1); continue; }

            let ngram_tensor = Tensor::<NdArray, 2>::from_data(
                TensorData::new(ngram_embs_flat, [n_ngrams, MINILM_DIM]), device,
            );
            let (affinity, type_logits) = valid_model.forward_scores(ngram_tensor);

            let affinity_data: Vec<f32> = affinity.into_data().to_vec().unwrap();
            let _type_data: Vec<f32> = type_logits.into_data().to_vec().unwrap();

            // Val loss: cross-entropy over concepts (same as train)
            let mut sample_loss = 0.0f32;
            let mut n_loss_terms = 0usize;

            for gt_span in &sample.spans {
                let matched = spans.iter().position(|&(s, e)| {
                    s == gt_span.start_word && e == gt_span.end_word
                });
                if let Some(ng_idx) = matched {
                    let base = ng_idx * n_concepts;
                    let row = &affinity_data[base..base + n_concepts];
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = row.iter().map(|&v| (v - max_val).exp()).sum();
                    let log_softmax = (row[gt_span.concept_idx] - max_val) - exp_sum.ln();
                    sample_loss += -log_softmax;
                    n_loss_terms += 1;

                    // Concept accuracy: is argmax the correct concept?
                    let pred_concept = row.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap().0;
                    val_type_total += 1;
                    if pred_concept == gt_span.concept_idx { val_type_correct += 1; }
                }
            }

            if n_loss_terms > 0 {
                val_loss_sum += sample_loss / n_loss_terms as f32;
                val_count += 1;
            }

            pb.inc(1);
        }
        pb.finish_and_clear();

        let train_avg = if train_count > 0 { train_loss_sum / train_count as f32 } else { 0.0 };
        let val_avg = if val_count > 0 { val_loss_sum / val_count as f32 } else { 0.0 };
        let train_type_acc = if train_type_total > 0 { train_type_correct as f64 / train_type_total as f64 } else { 0.0 };
        let val_type_acc = if val_type_total > 0 { val_type_correct as f64 / val_type_total as f64 } else { 0.0 };

        println!("epoch {:>3}: loss={:.4}/{:.4} concept_acc={:.1}%/{:.1}% lr={:.6}",
            epoch, train_avg, val_avg,
            train_type_acc * 100.0, val_type_acc * 100.0,  // reusing var names
            current_lr);
        writeln!(metrics_file, "{},{:.6},{:.6},{:.6},{:.6},{:.8}",
            epoch, train_avg, val_avg, train_type_acc, val_type_acc, current_lr).unwrap();

        // Early stopping
        if val_avg < best_val_loss {
            best_val_loss = val_avg;
            epochs_without_improvement = 0;
            model.valid()
                .save_file(&model_path, &CompactRecorder::new())
                .expect("save model");
        } else {
            epochs_without_improvement += 1;
            if epochs_without_improvement >= patience {
                println!("early stopping at epoch {} (no improvement for {} epochs)", epoch, patience);
                break;
            }
        }
    }

    println!("best val_loss={:.4}, saved to {:?}", best_val_loss, model_path);
}
