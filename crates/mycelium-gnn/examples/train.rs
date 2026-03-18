//! Train the GNN resolver.
//!
//! Usage:
//!   cargo run --release --example gen_dataset -p gnn-burn
//!   cargo run --release --example train -p gnn-burn

use std::path::Path;
use gnn_burn::training::{Dataset, TrainingConfig, train};

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");

    // Prefer NLP dataset (real biaffine + cross-encoder), fall back to synthetic
    let nlp_path = demo_dir.join("dataset_nlp.json");
    let syn_path = demo_dir.join("dataset.json");
    let dataset_path = if nlp_path.exists() { &nlp_path } else { &syn_path };
    let dataset = Dataset::load(dataset_path).expect("load dataset");
    println!("using {:?}", dataset_path.file_name().unwrap());
    println!("loaded {} samples", dataset.samples.len());

    let config = TrainingConfig {
        learning_rate: 0.001,
        epochs: 50,
        hidden_dim: 64,
        n_layers: 2,
        glove_path: demo_dir.join("glove.6B.300d.txt").to_string_lossy().into(),
        schema_path: demo_dir.join("schema.surql").to_string_lossy().into(),
        type_dim: 16,
        patience: 8,
        model_path: demo_dir.join("gnn_model").to_string_lossy().into(),
    };

    train(&config, &dataset);
}
