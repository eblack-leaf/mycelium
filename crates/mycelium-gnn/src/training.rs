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
use serde::{Serialize, Deserialize};
use burn::{
    module::Module,
    backend::{Autodiff, NdArray},
    optim::{AdamConfig, GradientsParams, Optimizer},
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

type TrainBackend = Autodiff<NdArray>;

pub fn train(config: &TrainingConfig, dataset: &Dataset) {
    let device = &Default::default();

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

    // --- Training loop ---
    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;

        for sample in &dataset.samples {
            let query_graph = QueryGraph::from_extraction(&sample.extraction);
            let conv = ResolverConv::new(&schema_graph, &query_graph);

            // Forward pass (type_embed gradients flow through here)
            let initial = embedder.embed_all::<TrainBackend>(
                &model.type_embed, &schema, &schema_graph, &query_graph, &operations, device,
            );
            let encoded = model.encoder.forward(&conv, initial, device);
            let logits = model.head.score_logits(&encoded);

            // Loss
            let loss = compute_loss(&logits, &sample.ground_truth, device);
            epoch_loss += loss.clone().into_data().to_vec::<f32>().unwrap()[0];

            // Backward + optimizer step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);
        }

        let avg = epoch_loss / dataset.samples.len() as f32;
        println!("epoch {}: loss = {:.4}", epoch, avg);
    }
}
