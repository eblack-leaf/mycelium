// =============================================================================
// training.rs — Training loop for the GNN resolver
//
// Trains: Encoder (SAGEConv weights) + OutputHead (role + target prediction)
//         + type Embedding (learned type/op identity vectors)
// Frozen: GloVe vectors
//
// Each sample: LinguisticGraph + CandidateSet + GroundTruth
// Forward: Embed → LinguisticConv → Encoder → OutputHead
// Loss: cross-entropy on role classification + cross-entropy on target selection
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
    record::CompactRecorder,
    tensor::{backend::Backend, Tensor, Int, TensorData},
    tensor::activation,
};
use burn::nn::{Embedding, Linear, LinearConfig};
use crate::embed::{Embedder, GloveVocab, create_type_embedding};
use crate::schema::{Reader, Extractor};
use crate::graph::SchemaGraph;
use crate::nlp::{LinguisticGraph, SpanType};
use crate::candidate_matcher::CandidateSet;
use crate::linguistic_graph::LinguisticConv;
use crate::operations::all_operations;
use crate::sage::Encoder;
use crate::head::{OutputHead, HeadLogits, CandidateMask};

// =============================================================================
// Data types
// =============================================================================

/// What schema role a linguistic node plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaRole {
    /// Maps to a table (SELECT FROM ...)
    Collection,
    /// Maps to a field (SELECT field ...)
    Field,
    /// Maps to a field used in a filter (WHERE field ...)
    FilterField,
    /// Maps to an operation (ORDER_BY, LIMIT, COUNT, etc.)
    Modifier,
    /// Maps to a table for traversal (->relation->target)
    Traversal,
    /// Does not map to any schema node (intent verbs, noise)
    None,
}

impl SchemaRole {
    pub const COUNT: usize = 6;

    pub fn index(&self) -> usize {
        match self {
            SchemaRole::Collection => 0,
            SchemaRole::Field => 1,
            SchemaRole::FilterField => 2,
            SchemaRole::Modifier => 3,
            SchemaRole::Traversal => 4,
            SchemaRole::None => 5,
        }
    }
}

/// Ground truth for one linguistic node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTarget {
    /// Index into LinguisticGraph.nodes
    pub linguistic_node: usize,
    /// What role this node plays
    pub role: SchemaRole,
    /// Target schema node type ("table", "field", "operation", or "" for None)
    pub target_type: String,
    /// Target schema node index (within its type), or 0 for None
    pub target_id: usize,
}

/// Ground truth for one training sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    pub targets: Vec<NodeTarget>,
}

/// One training example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub linguistic_graph: LinguisticGraph,
    pub candidates: CandidateSet,
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
    /// Path to save best model (without extension — burn appends it).
    pub model_path: String,
}

// =============================================================================
// Combined trainable model
// =============================================================================

#[derive(Module, Debug)]
pub struct GnnModel<B: Backend> {
    pub type_embed: Embedding<B>,
    /// Projects 384-dim transformer embeddings → schema_dim for linguistic nodes.
    pub ling_proj: Linear<B>,
    pub encoder: Encoder<B>,
    pub head: OutputHead<B>,
}

const LING_TYPES: &[&str] = &["np", "quantifier", "comparator", "intent"];

/// Project linguistic node embeddings (384-dim transformer → schema_dim) using
/// a learned linear layer. Replaces the truncation that was here before.
pub fn project_linguistic<B: Backend>(
    embeddings: &mut HashMap<String, Tensor<B, 2>>,
    ling_proj: &Linear<B>,
) {
    for &lt in LING_TYPES {
        if let Some(emb) = embeddings.get(lt) {
            if emb.dims()[0] > 0 {
                embeddings.insert(lt.to_string(), ling_proj.forward(emb.clone()));
            }
        }
    }
}

// =============================================================================
// Loss
// =============================================================================

/// Cross-entropy: logits [n, n_classes] vs target indices [n].
fn cross_entropy<B: Backend>(
    logits: Tensor<B, 2>,
    targets: &[usize],
    device: &B::Device,
) -> Tensor<B, 1> {
    let n = targets.len();
    if n == 0 {
        return Tensor::zeros([1], device);
    }
    let log_probs = activation::log_softmax(logits, 1);

    let target_ints: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
    let indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(target_ints, [n, 1]),
        device,
    );

    let gathered = log_probs.gather(1, indices); // [n, 1]
    gathered.neg().mean().reshape([1])
}

/// Build the ordered list of linguistic node indices matching the head's concatenation order.
/// The head concatenates: all np nodes, then quantifier, comparator, intent.
fn ordered_ling_nodes(ling_graph: &LinguisticGraph) -> Vec<usize> {
    let mut ordered = Vec::new();
    for st in &[SpanType::NounPhrase, SpanType::Quantifier, SpanType::Comparator, SpanType::Intent] {
        for node in &ling_graph.nodes {
            if node.span_type == *st {
                ordered.push(node.id);
            }
        }
    }
    ordered
}

/// Total loss: role classification + target selection.
pub fn compute_loss<B: Backend>(
    logits: &HeadLogits<B>,
    truth: &GroundTruth,
    ling_graph: &LinguisticGraph,
    mask: Option<&CandidateMask>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let ordered = ordered_ling_nodes(ling_graph);
    let n_ling = ordered.len();

    if n_ling == 0 {
        return Tensor::zeros([1], device);
    }

    // Build target arrays aligned to head's concatenation order
    let mut role_targets = vec![SchemaRole::None.index(); n_ling];
    let mut table_targets: Vec<Option<usize>> = vec![Option::None; n_ling];
    let mut field_targets: Vec<Option<usize>> = vec![Option::None; n_ling];
    let mut op_targets: Vec<Option<usize>> = vec![Option::None; n_ling];

    // Map from node id → position in ordered list
    let mut id_to_pos: HashMap<usize, usize> = HashMap::new();
    for (pos, &node_id) in ordered.iter().enumerate() {
        id_to_pos.insert(node_id, pos);
    }

    for nt in &truth.targets {
        if let Some(&pos) = id_to_pos.get(&nt.linguistic_node) {
            role_targets[pos] = nt.role.index();
            match nt.target_type.as_str() {
                "table" => table_targets[pos] = Some(nt.target_id),
                "field" => field_targets[pos] = Some(nt.target_id),
                "operation" => op_targets[pos] = Some(nt.target_id),
                _ => {}
            }
        }
    }

    let mut losses: Vec<Tensor<B, 1>> = Vec::new();

    // Role classification loss
    losses.push(cross_entropy(logits.role_logits.clone(), &role_targets, device));

    // Target losses — only for nodes whose ground-truth target is in the
    // candidate set (unmasked). If the cross-encoder didn't surface the correct
    // target, the logit is -1e9 and the loss is meaningless — skip it.
    if let Some(ref t_logits) = logits.target_table {
        let active: Vec<(usize, usize)> = table_targets.iter().enumerate()
            .filter_map(|(i, t)| t.map(|tid| (i, tid)))
            .filter(|&(i, tid)| mask.map_or(true, |m| {
                tid < m.n_tables && m.table_mask[i * m.n_tables + tid] > -1e8
            }))
            .collect();
        if !active.is_empty() {
            let row_indices: Vec<usize> = active.iter().map(|&(i, _)| i).collect();
            let targets: Vec<usize> = active.iter().map(|&(_, t)| t).collect();
            let selected = gather_rows(t_logits, &row_indices, device);
            losses.push(cross_entropy(selected, &targets, device));
        }
    }

    if let Some(ref f_logits) = logits.target_field {
        let active: Vec<(usize, usize)> = field_targets.iter().enumerate()
            .filter_map(|(i, t)| t.map(|tid| (i, tid)))
            .filter(|&(i, tid)| mask.map_or(true, |m| {
                tid < m.n_fields && m.field_mask[i * m.n_fields + tid] > -1e8
            }))
            .collect();
        if !active.is_empty() {
            let row_indices: Vec<usize> = active.iter().map(|&(i, _)| i).collect();
            let targets: Vec<usize> = active.iter().map(|&(_, t)| t).collect();
            let selected = gather_rows(f_logits, &row_indices, device);
            losses.push(cross_entropy(selected, &targets, device));
        }
    }

    if let Some(ref o_logits) = logits.target_op {
        let active: Vec<(usize, usize)> = op_targets.iter().enumerate()
            .filter_map(|(i, t)| t.map(|tid| (i, tid)))
            .filter(|&(i, tid)| mask.map_or(true, |m| {
                tid < m.n_ops && m.op_mask[i * m.n_ops + tid] > -1e8
            }))
            .collect();
        if !active.is_empty() {
            let row_indices: Vec<usize> = active.iter().map(|&(i, _)| i).collect();
            let targets: Vec<usize> = active.iter().map(|&(_, t)| t).collect();
            let selected = gather_rows(o_logits, &row_indices, device);
            losses.push(cross_entropy(selected, &targets, device));
        }
    }

    if losses.is_empty() {
        return Tensor::zeros([1], device);
    }

    let n = losses.len() as f32;
    let sum = losses.into_iter().reduce(|a, b| a + b).unwrap();
    sum / n
}

/// Select specific rows from a 2D tensor.
fn gather_rows<B: Backend>(
    tensor: &Tensor<B, 2>,
    row_indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let [_n, cols] = tensor.dims();
    let n_select = row_indices.len();

    let flat: Vec<i32> = row_indices.iter()
        .flat_map(|&idx| std::iter::repeat(idx as i32).take(cols))
        .collect();
    let indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(flat, [n_select, cols]),
        device,
    );
    tensor.clone().gather(0, indices)
}

// =============================================================================
// Accuracy
// =============================================================================

/// Compute role accuracy + target accuracy.
/// Returns (role_correct, role_total, target_correct, target_total, reachable_correct, reachable_total)
fn accuracy<B: Backend>(
    logits: &HeadLogits<B>,
    truth: &GroundTruth,
    ling_graph: &LinguisticGraph,
    mask: Option<&CandidateMask>,
) -> (usize, usize, usize, usize, usize, usize) {
    let ordered = ordered_ling_nodes(ling_graph);
    let n_ling = ordered.len();
    if n_ling == 0 {
        return (0, 0, 0, 0, 0, 0);
    }

    let mut id_to_pos: HashMap<usize, usize> = HashMap::new();
    for (pos, &node_id) in ordered.iter().enumerate() {
        id_to_pos.insert(node_id, pos);
    }

    // Role accuracy
    let role_preds = logits.role_logits.clone().argmax(1);
    let role_data: Vec<i64> = role_preds.into_data().to_vec().unwrap();

    let mut role_correct = 0usize;
    let mut role_total = 0usize;
    let mut target_correct = 0usize;
    let mut target_total = 0usize;
    let mut reachable_correct = 0usize;
    let mut reachable_total = 0usize;

    // Extract target predictions once
    let table_preds = logits.target_table.as_ref().map(|t| {
        let data: Vec<i64> = t.clone().argmax(1).into_data().to_vec().unwrap();
        data
    });
    let field_preds = logits.target_field.as_ref().map(|t| {
        let data: Vec<i64> = t.clone().argmax(1).into_data().to_vec().unwrap();
        data
    });
    let op_preds = logits.target_op.as_ref().map(|t| {
        let data: Vec<i64> = t.clone().argmax(1).into_data().to_vec().unwrap();
        data
    });

    for nt in &truth.targets {
        if let Some(&pos) = id_to_pos.get(&nt.linguistic_node) {
            if pos < role_data.len() {
                role_total += 1;
                if role_data[pos] as usize == nt.role.index() {
                    role_correct += 1;
                }

                // Check if GT target is reachable in candidate mask
                let reachable = mask.map_or(true, |m| {
                    let mask_val = match nt.target_type.as_str() {
                        "table" => m.table_mask.get(pos * m.n_tables + nt.target_id).copied(),
                        "field" => m.field_mask.get(pos * m.n_fields + nt.target_id).copied(),
                        "operation" => m.op_mask.get(pos * m.n_ops + nt.target_id).copied(),
                        _ => None,
                    };
                    mask_val.map_or(false, |v| v > -1e8)
                });

                // Target accuracy
                let pred_target = match nt.target_type.as_str() {
                    "table" => table_preds.as_ref().and_then(|p| p.get(pos).map(|&v| v as usize)),
                    "field" => field_preds.as_ref().and_then(|p| p.get(pos).map(|&v| v as usize)),
                    "operation" => op_preds.as_ref().and_then(|p| p.get(pos).map(|&v| v as usize)),
                    _ => Option::None,
                };

                if let Some(pred) = pred_target {
                    target_total += 1;
                    if pred == nt.target_id {
                        target_correct += 1;
                    }
                    if reachable {
                        reachable_total += 1;
                        if pred == nt.target_id {
                            reachable_correct += 1;
                        }
                    }
                }
            }
        }
    }

    (role_correct, role_total, target_correct, target_total, reachable_correct, reachable_total)
}

// =============================================================================
// Training loop
// =============================================================================

type TrainBackend = Autodiff<NdArray>;

pub fn train(config: &TrainingConfig, dataset: &Dataset) {
    let device = &Default::default();

    // --- Train/val split (80/20, deterministic) ---
    let n = dataset.samples.len();
    let n_train = (n as f64 * 0.8) as usize;
    let train_samples = &dataset.samples[..n_train];
    let val_samples = &dataset.samples[n_train..];
    println!("split: {} train, {} val", train_samples.len(), val_samples.len());

    // --- Shared schema ---
    let raw = Reader::read(Path::new(&config.schema_path)).expect("read schema");
    let (schema, _) = Extractor::extract(&raw);
    let schema_graph = SchemaGraph::from_schema(&schema);
    let operations = all_operations();

    // --- Embedder (GloVe frozen, type embedding trainable in model) ---
    let glove = GloveVocab::load(Path::new(&config.glove_path), 42).expect("load glove");
    let embedder = Embedder::new(glove, config.type_dim, 384);
    let embed_dim = embedder.schema_dim();

    // --- Build model from template conv (all relation types) ---
    let template = LinguisticConv::template(&schema_graph);
    let input_dims: HashMap<String, usize> = template.node_counts.iter()
        .map(|(name, _)| (name.clone(), embed_dim))
        .collect();

    let type_embed: Embedding<TrainBackend> = create_type_embedding(config.type_dim, device);
    let ling_proj: Linear<TrainBackend> = LinearConfig::new(384, embed_dim).init(device);
    let encoder: Encoder<TrainBackend> = Encoder::new(
        &template, &input_dims, config.hidden_dim, config.n_layers, device,
    );
    let head: OutputHead<TrainBackend> = OutputHead::new(config.hidden_dim, device);
    let mut model = GnnModel { type_embed, ling_proj, encoder, head };
    let mut optim = AdamConfig::new().init();
    let mut scheduler = CosineAnnealingLrSchedulerConfig::new(
        config.learning_rate, config.epochs,
    ).init().expect("valid scheduler config");

    // --- Metrics CSV ---
    let metrics_path = Path::new(&config.schema_path).parent().unwrap().join("metrics.csv");
    let mut metrics_file = std::fs::File::create(&metrics_path).expect("create metrics.csv");
    writeln!(metrics_file, "epoch,train_loss,val_loss,train_role_acc,val_role_acc,train_target_acc,val_target_acc,train_reachable_acc,val_reachable_acc,lr").unwrap();

    let pb_style = ProgressStyle::default_bar()
        .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("##-");

    // --- Early stopping state ---
    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0usize;

    let n_tables = schema_graph.table_nodes.len();
    let n_fields = schema_graph.field_nodes.len();
    let n_ops = operations.len();

    // --- Training loop ---
    for epoch in 0..config.epochs {
        // --- Train ---
        let mut train_loss = 0.0f32;
        let mut train_role_correct = 0usize;
        let mut train_role_total = 0usize;
        let mut train_target_correct = 0usize;
        let mut train_target_total = 0usize;
        let mut train_reachable_correct = 0usize;
        let mut train_reachable_total = 0usize;

        let pb = ProgressBar::new(train_samples.len() as u64);
        pb.set_style(pb_style.clone());
        pb.set_message(format!("epoch {:>3} train", epoch));

        let mut current_lr = 0.0;
        for sample in train_samples {
            let conv = LinguisticConv::new(&schema_graph, &sample.linguistic_graph, &sample.candidates);

            let mut initial = embedder.embed_all::<TrainBackend>(
                &model.type_embed, &schema, &schema_graph,
                &sample.linguistic_graph, &operations, device,
            );
            project_linguistic(&mut initial, &model.ling_proj);
            let encoded = model.encoder.forward(&conv, initial, device);
            let mask = CandidateMask::from_candidates(
                &sample.linguistic_graph, &sample.candidates,
                n_tables, n_fields, n_ops,
            );
            let logits = model.head.forward(&encoded, Some(&mask));

            let loss = compute_loss(&logits, &sample.ground_truth, &sample.linguistic_graph, Some(&mask), device);

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            train_loss += loss_val;

            let (rc, rt, tc, tt, rrc, rrt) = accuracy(&logits, &sample.ground_truth, &sample.linguistic_graph, Some(&mask));
            train_role_correct += rc;
            train_role_total += rt;
            train_target_correct += tc;
            train_target_total += tt;
            train_reachable_correct += rrc;
            train_reachable_total += rrt;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            current_lr = scheduler.step();
            model = optim.step(current_lr, model, grads);

            pb.inc(1);
        }
        pb.finish_and_clear();

        // --- Validate ---
        let mut val_loss = 0.0f32;
        let mut val_role_correct = 0usize;
        let mut val_role_total = 0usize;
        let mut val_target_correct = 0usize;
        let mut val_target_total = 0usize;
        let mut val_reachable_correct = 0usize;
        let mut val_reachable_total = 0usize;

        let pb = ProgressBar::new(val_samples.len() as u64);
        pb.set_style(pb_style.clone());
        pb.set_message(format!("epoch {:>3} val  ", epoch));

        let valid_model = model.valid();
        for sample in val_samples {
            let conv = LinguisticConv::new(&schema_graph, &sample.linguistic_graph, &sample.candidates);

            let mut initial = embedder.embed_all(
                &valid_model.type_embed, &schema, &schema_graph,
                &sample.linguistic_graph, &operations, device,
            );
            project_linguistic(&mut initial, &valid_model.ling_proj);
            let encoded = valid_model.encoder.forward(&conv, initial, device);
            let mask = CandidateMask::from_candidates(
                &sample.linguistic_graph, &sample.candidates,
                n_tables, n_fields, n_ops,
            );
            let logits = valid_model.head.forward(&encoded, Some(&mask));

            let loss = compute_loss(&logits, &sample.ground_truth, &sample.linguistic_graph, Some(&mask), device);
            val_loss += loss.clone().into_data().to_vec::<f32>().unwrap()[0];

            let (rc, rt, tc, tt, rrc, rrt) = accuracy(&logits, &sample.ground_truth, &sample.linguistic_graph, Some(&mask));
            val_role_correct += rc;
            val_role_total += rt;
            val_target_correct += tc;
            val_target_total += tt;
            val_reachable_correct += rrc;
            val_reachable_total += rrt;

            pb.inc(1);
        }
        pb.finish_and_clear();

        let train_avg = train_loss / train_samples.len() as f32;
        let val_avg = val_loss / val_samples.len() as f32;
        let train_role_acc = if train_role_total > 0 { train_role_correct as f64 / train_role_total as f64 } else { 0.0 };
        let val_role_acc = if val_role_total > 0 { val_role_correct as f64 / val_role_total as f64 } else { 0.0 };
        let train_target_acc = if train_target_total > 0 { train_target_correct as f64 / train_target_total as f64 } else { 0.0 };
        let val_target_acc = if val_target_total > 0 { val_target_correct as f64 / val_target_total as f64 } else { 0.0 };
        let train_reach_acc = if train_reachable_total > 0 { train_reachable_correct as f64 / train_reachable_total as f64 } else { 0.0 };
        let val_reach_acc = if val_reachable_total > 0 { val_reachable_correct as f64 / val_reachable_total as f64 } else { 0.0 };

        println!(
            "epoch {:>3}: loss={:.4}/{:.4} role={:.1}%/{:.1}% target={:.1}%/{:.1}% reachable={:.1}%/{:.1}% lr={:.6}",
            epoch, train_avg, val_avg,
            train_role_acc * 100.0, val_role_acc * 100.0,
            train_target_acc * 100.0, val_target_acc * 100.0,
            train_reach_acc * 100.0, val_reach_acc * 100.0,
            current_lr,
        );
        writeln!(metrics_file, "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.8}",
            epoch, train_avg, val_avg, train_role_acc, val_role_acc,
            train_target_acc, val_target_acc, train_reach_acc, val_reach_acc, current_lr).unwrap();

        // Early stopping + save best
        if val_avg < best_val_loss {
            best_val_loss = val_avg;
            epochs_without_improvement = 0;
            model.valid()
                .save_file(&config.model_path, &CompactRecorder::new())
                .expect("save model");
        } else {
            epochs_without_improvement += 1;
            if epochs_without_improvement >= config.patience {
                println!("early stopping at epoch {} (no improvement for {} epochs)", epoch, config.patience);
                break;
            }
        }
    }

    println!("best val_loss={:.4}, saved to {}", best_val_loss, config.model_path);
}

/// Load a trained GnnModel from disk.
pub fn load_model(
    model_path: &str,
    schema_path: &str,
    glove_path: &str,
    hidden_dim: usize,
    n_layers: usize,
    type_dim: usize,
) -> (GnnModel<NdArray>, Embedder, SchemaGraph) {
    let device = &Default::default();

    let raw = Reader::read(Path::new(schema_path)).expect("read schema");
    let (schema, _) = Extractor::extract(&raw);
    let schema_graph = SchemaGraph::from_schema(&schema);

    let glove = GloveVocab::load(Path::new(glove_path), 42).expect("load glove");
    let embedder = Embedder::new(glove, type_dim, 384);
    let embed_dim = embedder.schema_dim();

    let template = LinguisticConv::template(&schema_graph);
    let input_dims: HashMap<String, usize> = template.node_counts.iter()
        .map(|(name, _)| (name.clone(), embed_dim))
        .collect();

    let type_embed: Embedding<NdArray> = create_type_embedding(type_dim, device);
    let ling_proj: Linear<NdArray> = LinearConfig::new(384, embed_dim).init(device);
    let encoder: Encoder<NdArray> = Encoder::new(
        &template, &input_dims, hidden_dim, n_layers, device,
    );
    let head: OutputHead<NdArray> = OutputHead::new(hidden_dim, device);

    let model = GnnModel { type_embed, ling_proj, encoder, head }
        .load_file(model_path, &CompactRecorder::new(), device)
        .expect("load model");

    (model, embedder, schema_graph)
}
