use basidium::{trainable::PipelineTrainCtx, trainer::{Trainer, TrainerConfig}, Datum};
use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
use hyphae::{graph::SchemaGraph, model::HyphaeConfig, schema::Schema};
use septa::model::SeptaConfig;
use std::path::Path;

type B = Autodiff<Wgpu>;

const SCHEMA_DIR: &str = "stipe/fixtures/schema/";
const DATA_PATH: &str = "data/train.json";

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: basidium <generate|stats|train> [data_path]");
        std::process::exit(1);
    });

    match cmd.as_str() {
        "generate" | "gen" => cmd_generate(),
        "stats" => cmd_stats(),
        "train" => cmd_train(),
        other => {
            eprintln!("unknown command: {other}");
            eprintln!("usage: basidium <generate|stats|train>");
            std::process::exit(1);
        }
    }
}

fn data_path() -> String {
    std::env::args().nth(2).unwrap_or_else(|| DATA_PATH.to_string())
}

fn cmd_generate() {
    let schema = Schema::from_dir(Path::new(SCHEMA_DIR)).unwrap();
    let data = Datum::generate(&schema);
    println!("Generated {} datums", data.len());

    let path = data_path();
    let parent = Path::new(&path).parent().unwrap();
    std::fs::create_dir_all(parent).unwrap();
    let json = serde_json::to_string(&data).unwrap();
    std::fs::write(&path, &json).unwrap();
    println!("Wrote {}", path);
}

fn cmd_stats() {
    let path = data_path();
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        eprintln!("run `basidium generate` first");
        std::process::exit(1);
    });
    let data: Vec<Datum> = serde_json::from_str(&json).unwrap();
    Datum::print_stats(&data);
}

fn cmd_train() {
    let path = data_path();
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        eprintln!("run `basidium generate` first");
        std::process::exit(1);
    });
    let data: Vec<Datum> = serde_json::from_str(&json).unwrap();
    println!("Loaded {} datums from {}", data.len(), path);

    let device = WgpuDevice::default();
    let schema = Schema::from_dir(Path::new(SCHEMA_DIR)).unwrap();

    let hyphae_config = HyphaeConfig::new();
    let septa_config = SeptaConfig::new(12); // 12 BIO tags: B-/I- × 6 span types
    let schema_graph = SchemaGraph::new(schema, hyphae_config.ngram_buckets);

    let ctx: PipelineTrainCtx<B> = PipelineTrainCtx::new(
        hyphae_config, septa_config, schema_graph, 1e-3, &device,
    );
    let trainer_config = TrainerConfig::new();
    let mut trainer = Trainer::new(trainer_config, ctx, "weights/pipeline");

    let result = trainer.train(&data);
    println!(
        "Best epoch: {} — val_loss={:.4} val_acc={:.3}",
        result.best_epoch, result.best_metrics.val_loss, result.best_metrics.val_acc
    );
    println!("  {}", result.best_metrics.head_acc.display());
}
