// =============================================================================
// training.rs — Training loop for the GNN resolver
//
// Trains: Encoder (SAGEConv weights) + OutputHead (bilinear projections)
//         + type Embedding (learned type/op identity vectors)
// Frozen: GloVe vectors
//
// Each sample: Extraction (from grounding model) + GroundTruth (target indices)
// Forward: Embedder → Encoder → OutputHead::score_logits
// Loss: cross-entropy per candidate type, averaged
// =============================================================================

use std::collections::HashMap;
use std::path::Path;
use std::io::Write;
use serde::{Serialize, Deserialize};
use indicatif::{ProgressBar, ProgressStyle};
use burn::{
    module::{Module, AutodiffModule},
    backend::{Autodiff, NdArray},
    optim::{AdamConfig, GradientsParams, Optimizer},
    lr_scheduler::{LrScheduler, cosine::CosineAnnealingLrSchedulerConfig},
    tensor::{backend::Backend, Tensor, Int, TensorData},
    tensor::activation,
};
use burn::nn::Embedding;
use crate::embed::{Embedder, GloveVocab, create_type_embedding};
use crate::schema::{Reader, Extractor};
use crate::graph::SchemaGraph;
use crate::query_graph::QueryGraph;
use crate::conv_graph::ResolverConv;
use crate::operations::all_operations;
use crate::sage::Encoder;
use crate::head::{OutputHead, ScoreLogits};
use crate::intent::Extraction;

// =============================================================================
// Data types
// =============================================================================

/// Ground truth: for each query candidate, the index of its correct target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    /// For each q_collection → index into table nodes
    pub collection_targets: Vec<usize>,
    /// For each q_field → index into schema field nodes
    pub field_targets: Vec<usize>,
    /// For each q_filter → index into operation nodes
    pub filter_op_targets: Vec<usize>,
    /// For each q_traversal → index into table nodes
    pub traversal_targets: Vec<usize>,
    /// For each q_modifier → index into operation nodes
    pub modifier_op_targets: Vec<usize>,
}

/// One training example: pre-computed extraction + correct resolution indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub extraction: Extraction,
    pub ground_truth: GroundTruth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub samples: Vec<TrainingSample>,
}

impl Dataset {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }
}

pub struct TrainingConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub glove_path: String,
    pub schema_path: String,
    pub type_dim: usize,
    pub patience: usize,
}

// =============================================================================
// Combined trainable model
// =============================================================================

#[derive(Module, Debug)]
pub struct GnnModel<B: Backend> {
    pub type_embed: Embedding<B>,
    pub encoder: Encoder<B>,
    pub head: OutputHead<B>,
}

// =============================================================================
// Loss
// =============================================================================

/// Cross-entropy for one candidate type: logits [n, n_targets] vs target indices.
fn cross_entropy<B: Backend>(
    logits: Tensor<B, 2>,
    targets: &[usize],
    device: &B::Device,
) -> Tensor<B, 1> {
    let n = targets.len();
    let log_probs = activation::log_softmax(logits, 1);

    let target_ints: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
    let indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(target_ints, [n, 1]),
        device,
    );

    // Gather log-prob at the target index for each candidate
    let gathered = log_probs.gather(1, indices); // [n, 1]
    gathered.neg().mean().reshape([1])
}

/// Total loss across all active candidate types.
pub fn compute_loss<B: Backend>(
    logits: &ScoreLogits<B>,
    truth: &GroundTruth,
    device: &B::Device,
) -> Tensor<B, 1> {
    let mut losses: Vec<Tensor<B, 1>> = Vec::new();

    if let Some(ref l) = logits.collection {
        if !truth.collection_targets.is_empty() {
            losses.push(cross_entropy(l.clone(), &truth.collection_targets, device));
        }
    }
    if let Some(ref l) = logits.field {
        if !truth.field_targets.is_empty() {
            losses.push(cross_entropy(l.clone(), &truth.field_targets, device));
        }
    }
    if let Some(ref l) = logits.filter_op {
        if !truth.filter_op_targets.is_empty() {
            losses.push(cross_entropy(l.clone(), &truth.filter_op_targets, device));
        }
    }
    if let Some(ref l) = logits.traversal {
        if !truth.traversal_targets.is_empty() {
            losses.push(cross_entropy(l.clone(), &truth.traversal_targets, device));
        }
    }
    if let Some(ref l) = logits.modifier_op {
        if !truth.modifier_op_targets.is_empty() {
            losses.push(cross_entropy(l.clone(), &truth.modifier_op_targets, device));
        }
    }

    if losses.is_empty() {
        return Tensor::zeros([1], device);
    }

    let n = losses.len() as f32;
    let sum = losses.into_iter().reduce(|a, b| a + b).unwrap();
    sum / n
}

// =============================================================================
// Training loop
// =============================================================================

// NdArray is faster for per-sample training (small tensors, high kernel overhead on GPU).
// Switch to Autodiff<Wgpu> when graph batching is implemented.
type TrainBackend = Autodiff<NdArray>;

/// Compute per-candidate-type accuracy from logits vs ground truth.
/// Returns (correct, total) across all candidate types.
fn accuracy<B: Backend>(
    logits: &ScoreLogits<B>,
    truth: &GroundTruth,
) -> (usize, usize) {
    let mut correct = 0usize;
    let mut total = 0usize;

    fn check<B: Backend>(l: &Tensor<B, 2>, targets: &[usize]) -> (usize, usize) {
        let preds = l.clone().argmax(1).float(); // [n, 1] → float for portable extraction
        let pred_data: Vec<f32> = preds.into_data().to_vec().unwrap();
        let mut c = 0;
        for (p, &t) in pred_data.iter().zip(targets.iter()) {
            if (*p as usize) == t { c += 1; }
        }
        (c, targets.len())
    }

    if let Some(ref l) = logits.collection {
        if !truth.collection_targets.is_empty() {
            let (c, t) = check(l, &truth.collection_targets);
            correct += c; total += t;
        }
    }
    if let Some(ref l) = logits.field {
        if !truth.field_targets.is_empty() {
            let (c, t) = check(l, &truth.field_targets);
            correct += c; total += t;
        }
    }
    if let Some(ref l) = logits.filter_op {
        if !truth.filter_op_targets.is_empty() {
            let (c, t) = check(l, &truth.filter_op_targets);
            correct += c; total += t;
        }
    }
    if let Some(ref l) = logits.traversal {
        if !truth.traversal_targets.is_empty() {
            let (c, t) = check(l, &truth.traversal_targets);
            correct += c; total += t;
        }
    }
    if let Some(ref l) = logits.modifier_op {
        if !truth.modifier_op_targets.is_empty() {
            let (c, t) = check(l, &truth.modifier_op_targets);
            correct += c; total += t;
        }
    }

    (correct, total)
}

pub fn train(config: &TrainingConfig, dataset: &Dataset) {
    let device = &Default::default();

    // --- Train/val split (80/20, deterministic) ---
    let n = dataset.samples.len();
    let n_train = (n as f64 * 0.8) as usize;
    let train_samples = &dataset.samples[..n_train];
    let val_samples = &dataset.samples[n_train..];
    println!("split: {} train, {} val", train_samples.len(), val_samples.len());

    // --- Shared schema (same for all samples) ---
    let raw = Reader::read(Path::new(&config.schema_path)).expect("read schema");
    let (schema, _) = Extractor::extract(&raw);
    let schema_graph = SchemaGraph::from_schema(&schema);
    let operations = all_operations();

    // --- Embedder (GloVe frozen, type embedding trainable in model) ---
    let glove = GloveVocab::load(Path::new(&config.glove_path), 42).expect("load glove");
    let embedder = Embedder::new(glove, config.type_dim);
    let embed_dim = embedder.dim();

    // --- Build model from template conv (all relation types) ---
    let template = ResolverConv::template(&schema_graph);
    let input_dims: HashMap<String, usize> = template.node_counts.iter()
        .map(|(name, _)| (name.clone(), embed_dim))
        .collect();

    let type_embed: Embedding<TrainBackend> = create_type_embedding(config.type_dim, device);
    let encoder: Encoder<TrainBackend> = Encoder::new(
        &template, &input_dims, config.hidden_dim, config.n_layers, device,
    );
    let head: OutputHead<TrainBackend> = OutputHead::new(config.hidden_dim, device);
    let mut model = GnnModel { type_embed, encoder, head };
    let mut optim = AdamConfig::new().init();

    let total_steps = train_samples.len() * config.epochs;
    let mut scheduler = CosineAnnealingLrSchedulerConfig::new(
        config.learning_rate, total_steps,
    ).init().expect("valid scheduler config");

    // --- Early stopping state ---
    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0usize;

    // --- Metrics CSV ---
    let metrics_path = Path::new(&config.schema_path).parent()
        .unwrap_or(Path::new(".")).join("metrics.csv");
    let mut metrics_file = std::fs::File::create(&metrics_path).expect("create metrics.csv");
    writeln!(metrics_file, "epoch,train_loss,val_loss,train_acc,val_acc,lr").unwrap();

    // --- Training loop ---
    let pb_style = ProgressStyle::with_template(
        "{msg} [{bar:40.cyan/blue}] {pos}/{len} [{elapsed_precise} < {eta_precise}]"
    ).unwrap().progress_chars("=> ");

    for epoch in 0..config.epochs {
        // --- Train ---
        let mut train_loss = 0.0f32;
        let mut train_correct = 0usize;
        let mut train_total = 0usize;

        let pb = ProgressBar::new(train_samples.len() as u64);
        pb.set_style(pb_style.clone());
        pb.set_message(format!("epoch {:>3} train", epoch));

        let mut current_lr = 0.0;
        for sample in train_samples {
            let query_graph = QueryGraph::from_extraction(&sample.extraction);
            let conv = ResolverConv::new(&schema_graph, &query_graph);

            let initial = embedder.embed_all::<TrainBackend>(
                &model.type_embed, &schema, &schema_graph, &query_graph, &operations, device,
            );
            let encoded = model.encoder.forward(&conv, initial, device);
            let logits = model.head.score_logits(&encoded);

            let loss = compute_loss(&logits, &sample.ground_truth, device);
            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            train_loss += loss_val;

            let (c, t) = accuracy(&logits, &sample.ground_truth);
            train_correct += c;
            train_total += t;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            current_lr = scheduler.step();
            model = optim.step(current_lr, model, grads);

            pb.inc(1);
        }
        pb.finish_and_clear();

        // --- Validate ---
        let mut val_loss = 0.0;
        let mut val_correct = 0usize;
        let mut val_total = 0usize;

        let pb = ProgressBar::new(val_samples.len() as u64);
        pb.set_style(pb_style.clone());
        pb.set_message(format!("epoch {:>3} val  ", epoch));

        let valid_model = model.valid();
        for sample in val_samples {
            let query_graph = QueryGraph::from_extraction(&sample.extraction);
            let conv = ResolverConv::new(&schema_graph, &query_graph);

            let initial = embedder.embed_all(
                &valid_model.type_embed, &schema, &schema_graph, &query_graph, &operations, device,
            );
            let encoded = valid_model.encoder.forward(&conv, initial, device);
            let logits = valid_model.head.score_logits(&encoded);

            let loss = compute_loss(&logits, &sample.ground_truth, device);
            val_loss += loss.clone().into_data().to_vec::<f32>().unwrap()[0];

            let (c, t) = accuracy(&logits, &sample.ground_truth);
            val_correct += c;
            val_total += t;

            pb.inc(1);
        }
        pb.finish_and_clear();

        let train_avg = train_loss / train_samples.len() as f32;
        let val_avg = val_loss / val_samples.len() as f32;
        let train_acc = if train_total > 0 { train_correct as f64 / train_total as f64 } else { 0.0 };
        let val_acc = if val_total > 0 { val_correct as f64 / val_total as f64 } else { 0.0 };
        println!(
            "epoch {:>3}: train_loss={:.4} val_loss={:.4} train_acc={:.2}% val_acc={:.2}% lr={:.6}",
            epoch, train_avg, val_avg, train_acc * 100.0, val_acc * 100.0, current_lr,
        );
        writeln!(metrics_file, "{},{:.6},{:.6},{:.6},{:.6},{:.8}",
            epoch, train_avg, val_avg, train_acc, val_acc, current_lr).unwrap();
        metrics_file.flush().unwrap();

        // Early stopping
        if val_avg < best_val_loss {
            best_val_loss = val_avg;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement += 1;
            if epochs_without_improvement >= config.patience {
                println!("early stopping at epoch {} (no improvement for {} epochs)", epoch, config.patience);
                break;
            }
        }
    }
}
