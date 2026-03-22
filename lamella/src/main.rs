#![recursion_limit = "256"]
use lamella::LamellaDatum;
use lamella::catalog::SchemaCatalog;
use lamella::embed::tokenize;
use lamella::load_dataset;
use lamella::model::{LamellaConfig, SlotCounts, ResolveValues};
use lamella::schema::Schema;
use lamella::train::{LamellaTrainCtx, TrainConfig, train_loop};

use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
use burn::tensor::ElementConversion;
use std::path::Path;

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

const SCHEMA_DIR: &str = "stipe/fixtures/schema/";
const EVAL_SCHEMA_DIR: &str = "stipe/fixtures/eval-schema/";
const WEIGHTS_FILE: &str = "weights/lamella.bin";
const DATASET_DIR: &str = "data/lamella/";
const EVAL_DATASET_DIR: &str = "data/lamella-eval/";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "train" => cmd_train(),
        "eval" => cmd_eval(),
        "infer" => cmd_infer(&args[2..]),
        _ => {
            eprintln!("Usage: lamella <train|eval|infer>");
            eprintln!("  train                 — train the model");
            eprintln!("  eval                  — evaluate on held-out schema");
            eprintln!("  infer --schema DIR NL — infer SurrealQL from NL");
        }
    }
}

fn cmd_train() {
    let schema = Schema::from_dir(Path::new(SCHEMA_DIR)).expect("Failed to load schema");
    let model_config = LamellaConfig::new();
    let train_config = TrainConfig::new();
    let catalog = SchemaCatalog::from_schema(&schema, model_config.schema_buckets);

    let (train_data, val_data) = load_dataset(DATASET_DIR, &catalog, 0.1);
    println!("Train: {}, Val: {}", train_data.len(), val_data.len());

    let device = WgpuDevice::default();
    let num_iters = (train_data.len() / train_config.batch_size) * train_config.epochs;

    let mut ctx = LamellaTrainCtx::<MyAutodiffBackend>::new(
        model_config,
        catalog,
        train_config.learning_rate,
        num_iters,
        &device,
    );

    train_loop(&mut ctx, &train_data, &val_data, &train_config, WEIGHTS_FILE);
}

fn cmd_eval() {
    let eval_schema = Schema::from_dir(Path::new(EVAL_SCHEMA_DIR)).expect("Failed to load eval schema");
    let config = LamellaConfig::new();
    let eval_catalog = SchemaCatalog::from_schema(&eval_schema, config.schema_buckets);

    let (eval_data, _) = load_dataset(EVAL_DATASET_DIR, &eval_catalog, 0.0);
    println!("Eval datums: {}", eval_data.len());

    let device = WgpuDevice::default();
    let mut ctx = LamellaTrainCtx::<MyAutodiffBackend>::new(
        config,
        eval_catalog,
        1e-3,
        1,
        &device,
    );

    let weights = std::path::PathBuf::from(WEIGHTS_FILE);
    ctx.load(&weights).expect("Failed to load weights");

    let eval_refs: Vec<&LamellaDatum> = eval_data.iter().collect();
    let bar = indicatif::ProgressBar::new(((eval_refs.len() + 31) / 32) as u64);
    let metrics = ctx.evaluate(&eval_refs, &bar);
    bar.finish_and_clear();

    println!("Eval: loss={:.4} acc={:.1}%", metrics.val_loss, metrics.val_acc * 100.0);
    println!("  {}", metrics.head_acc.display());
}

fn cmd_infer(args: &[String]) {
    let mut schema_dir = SCHEMA_DIR;
    let mut nl_parts: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--schema" && i + 1 < args.len() {
            schema_dir = &args[i + 1];
            i += 2;
        } else {
            nl_parts.push(&args[i]);
            i += 1;
        }
    }

    let nl = nl_parts.join(" ");
    if nl.is_empty() {
        eprintln!("Usage: lamella infer --schema DIR \"natural language query\"");
        return;
    }

    let schema = Schema::from_dir(Path::new(schema_dir)).expect("Failed to load schema");
    let config = LamellaConfig::new();
    let catalog = SchemaCatalog::from_schema(&schema, config.schema_buckets);

    let device = WgpuDevice::default();
    let model: lamella::model::Lamella<MyBackend> = config.init(&device);

    // Try to load weights
    let weights = std::path::PathBuf::from(WEIGHTS_FILE);
    let model = if weights.exists() {
        use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        match recorder.load(weights.clone(), &device) {
            Ok(record) => {
                use burn::module::Module;
                model.load_record(record)
            }
            Err(e) => {
                eprintln!("Warning: could not load weights: {e}");
                model
            }
        }
    } else {
        eprintln!("Warning: no weights found at {}", weights.display());
        model
    };

    let (tokens, _) = tokenize(&nl);

    let slots = SlotCounts {
        projections: 0,
        conditions: 0,
        assignments: 0,
        mod_types: 0,
        mod_fields: 0,
    };

    let logits = model.forward(&tokens, config.token_buckets, &slots, &catalog, 0, &device);
    let entity_idx: usize = logits.entity.clone().argmax(0).into_scalar().elem::<i32>() as usize;

    println!("Intent: {:?}", catalog.ops[logits.intent.clone().argmax(0).into_scalar().elem::<i32>() as usize]);
    println!("Entity: {}", catalog.tables[entity_idx]);

    let resolve_vals = ResolveValues {
        record_id: None,
        cond_values: vec![],
        asgn_values: vec![],
        mod_values: vec![],
        mod_descending: vec![],
    };

    let ir = model.resolve(&logits, &catalog, &resolve_vals);
    let q = ir.render(&[]);
    println!("SurrealQL: {}", q.surql);
}
