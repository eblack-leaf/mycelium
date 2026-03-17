//! Train the GNN resolver.
//!
//! Usage:
//!   cargo run --release --example gen_dataset -p gnn-burn
//!   cargo run --release --example train -p gnn-burn

use std::path::Path;
use gnn_burn::training::{Dataset, TrainingConfig, train};

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");

    let dataset = Dataset::load(&demo_dir.join("dataset.json")).expect("load dataset");
    println!("loaded {} samples", dataset.samples.len());

    let config = TrainingConfig {
        learning_rate: 0.001,
        epochs: 50,
        hidden_dim: 64,
        n_layers: 2,
        glove_path: demo_dir.join("glove.6B.50d.txt").to_string_lossy().into(),
        schema_path: demo_dir.join("schema.surql").to_string_lossy().into(),
        type_dim: 16,
        patience: 8,
        model_path: demo_dir.join("gnn_model").to_string_lossy().into(),
    };

    train(&config, &dataset);
}
