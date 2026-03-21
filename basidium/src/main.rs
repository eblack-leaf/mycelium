use basidium::{trainable::BasidiumTrainCtx, trainer::{Trainer, TrainerConfig}, Datum};
use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
use hyphae::{graph::SchemaGraph, model::HyphaeConfig, schema::Schema};
use septa::model::SeptaConfig;
use std::path::Path;

type B = Autodiff<Wgpu>;

const SCHEMA_DIR: &str = "stipe/fixtures/schema/";
const DATA_DIR: &str = "data";

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: basidium <generate|stats|train>");
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

fn data_dir() -> String {
    std::env::args().nth(2).unwrap_or_else(|| DATA_DIR.to_string())
}

fn load_json(path: &str) -> Vec<Datum> {
    let json = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        eprintln!("run `basidium generate` first");
        std::process::exit(1);
    });
    serde_json::from_str(&json).unwrap()
}

fn cmd_generate() {
    let schema = Schema::from_dir(Path::new(SCHEMA_DIR)).unwrap();
    let data = Datum::generate(&schema);
    println!("Generated {} datums", data.len());

    // Shuffle deterministically then split 90/10
    let split = (data.len() as f32 * 0.9) as usize;
    let mut indices: Vec<usize> = (0..data.len()).collect();
    shuffle_indices(&mut indices, 42);
    let train: Vec<Datum> = indices[..split].iter().map(|&i| data[i].clone()).collect();
    let val: Vec<Datum> = indices[split..].iter().map(|&i| data[i].clone()).collect();

    let dir = data_dir();
    std::fs::create_dir_all(&dir).unwrap();

    let train_path = format!("{dir}/train.json");
    std::fs::write(&train_path, serde_json::to_string(&train).unwrap()).unwrap();
    println!("Wrote {} train datums → {}", train.len(), train_path);

    let val_path = format!("{dir}/val.json");
    std::fs::write(&val_path, serde_json::to_string(&val).unwrap()).unwrap();
    println!("Wrote {} val datums → {}", val.len(), val_path);
}

fn cmd_stats() {
    let dir = data_dir();
    let train_path = format!("{dir}/train.json");
    let val_path = format!("{dir}/val.json");

    let train = load_json(&train_path);
    let val = load_json(&val_path);

    println!("=== Train Set ({}) ===", train_path);
    Datum::print_stats(&train);
    println!("\n=== Val Set ({}) ===", val_path);
    Datum::print_stats(&val);
}

fn cmd_train() {
    let dir = data_dir();
    let train_data = load_json(&format!("{dir}/train.json"));
    let val_data = load_json(&format!("{dir}/val.json"));
    println!("Loaded {} train + {} val datums", train_data.len(), val_data.len());

    let device = WgpuDevice::default();
    let schema = Schema::from_dir(Path::new(SCHEMA_DIR)).unwrap();

    let hyphae_config = HyphaeConfig::new();
    let septa_config = SeptaConfig::new(12); // 12 BIO tags: B-/I- × 6 span types
    let schema_graph = SchemaGraph::new(schema, hyphae_config.ngram_buckets);

    let trainer_config = TrainerConfig::new();
    let bs = trainer_config.batch_size;
    let batches_per_epoch = (train_data.len() + bs - 1) / bs;
    let num_iters = trainer_config.epochs * batches_per_epoch;

    let ctx: BasidiumTrainCtx<B> = BasidiumTrainCtx::new(
        hyphae_config, septa_config, schema_graph, 1e-3, num_iters,
        trainer_config.micro_batch_size, &device,
    );
    let mut trainer = Trainer::new(trainer_config, ctx, "weights/pipeline");

    let result = trainer.train(&train_data, &val_data);
    println!(
        "Best epoch: {} — val_loss={:.4} val_acc={:.3}",
        result.best_epoch, result.best_metrics.val_loss, result.best_metrics.val_acc
    );
    println!("  {}", result.best_metrics.head_acc.display());
}

/// Fisher-Yates shuffle with LCG PRNG (same as trainer)
fn shuffle_indices(v: &mut [usize], seed: u64) {
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    for i in (1..v.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
}

