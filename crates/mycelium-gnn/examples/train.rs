//! Demo: load schema + dataset, run GNN training loop, print loss per epoch.
//!
//! Usage:
//!   cargo run --example train
//!
//! Expects demo files at crates/mycelium-gnn/demo/:
//!   schema.surql   — SurrealDB schema
//!   dataset.json   — training samples (Extraction + GroundTruth)
//!   glove.6B.50d.txt — GloVe pretrained word embeddings (50d)

use std::path::Path;
use gnn_burn::training::{Dataset, TrainingConfig, train};

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");

    let dataset = Dataset::load(&demo_dir.join("dataset.json"))
        .expect("failed to load dataset");

    println!("loaded {} training samples", dataset.samples.len());

    let config = TrainingConfig {
        learning_rate: 0.001,
        epochs: 50,
        hidden_dim: 32,
        n_layers: 2,
        glove_path: demo_dir.join("glove.6B.50d.txt").to_string_lossy().into(),
        schema_path: demo_dir.join("schema.surql").to_string_lossy().into(),
        type_dim: 16,
        patience: 5,
    };

    train(&config, &dataset);
}
