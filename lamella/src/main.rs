#![recursion_limit = "256"]
use lamella::LamellaDatum;
use lamella::catalog::SchemaCatalog;
use lamella::embed::tokenize;
use lamella::load_dataset;
use lamella::load_raw_dataset;
use lamella::model::{LamellaConfig, SlotCounts, ResolveValues};
use lamella::temporal::extract_temporal_values;
use lamella::schema::Schema;
use lamella::train::{LamellaTrainCtx, TrainConfig, train_loop};

use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
use burn::tensor::ElementConversion;
use std::path::Path;

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

const SCHEMA_DIR: &str = "data/fixtures/schema/";
const EVAL_SCHEMA_DIR: &str = "data/fixtures/eval-schema/";
const WEIGHTS_FILE: &str = "weights/lamella.bin";
const DATASET_DIR: &str = "data/lamella/";
const EVAL_DATASET_DIR: &str = "data/lamella-eval/";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "train"    => cmd_train(),
        "eval"     => cmd_eval(),
        "diagnose" => cmd_diagnose(),
        "stats"    => cmd_stats(),
        "infer"    => cmd_infer(&args[2..]),
        _ => {
            eprintln!("Usage: lamella <train|eval|diagnose|stats|infer>");
            eprintln!("  train                 — train the model");
            eprintln!("  eval                  — evaluate on held-out schema");
            eprintln!("  diagnose              — per-table asgn accuracy on val split");
            eprintln!("  stats                 — field distribution in training data");
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

fn cmd_diagnose() {
    let schema = Schema::from_dir(Path::new(SCHEMA_DIR)).expect("Failed to load schema");
    let config = LamellaConfig::new();
    let catalog = SchemaCatalog::from_schema(&schema, config.schema_buckets);

    let (_, val_data) = load_dataset(DATASET_DIR, &catalog, 0.1);
    println!("Val datums: {}", val_data.len());

    let device = WgpuDevice::default();
    let mut ctx = LamellaTrainCtx::<MyAutodiffBackend>::new(config, catalog, 1e-3, 1, &device);

    let weights = std::path::PathBuf::from(WEIGHTS_FILE);
    ctx.load(&weights).expect("Failed to load weights");

    let val_refs: Vec<&LamellaDatum> = val_data.iter().collect();
    let bar = indicatif::ProgressBar::new(((val_refs.len() + 31) / 32) as u64);
    let metrics = ctx.evaluate(&val_refs, &bar);
    bar.finish_and_clear();

    println!("Val: loss={:.4} acc={:.1}%", metrics.val_loss, metrics.val_acc * 100.0);
    println!("  {}", metrics.head_acc.display());

    ctx.diagnose_asgn(&val_refs);
}

fn cmd_stats() {
    use std::collections::HashMap;

    let datums = load_raw_dataset(DATASET_DIR);
    println!("Total datums: {}", datums.len());

    // Per table: count how many times each field appears in asgn / cond / proj
    let mut asgn:  HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut cond:  HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut proj:  HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut total: HashMap<String, usize> = HashMap::new();

    for d in &datums {
        *total.entry(d.table.clone()).or_default() += 1;
        for a in &d.assignments {
            *asgn.entry(d.table.clone()).or_default().entry(a.field.clone()).or_default() += 1;
        }
        for c in &d.conditions {
            *cond.entry(d.table.clone()).or_default().entry(c.field.clone()).or_default() += 1;
        }
        for f in &d.projections {
            *proj.entry(d.table.clone()).or_default().entry(f.clone()).or_default() += 1;
        }
    }

    let mut tables: Vec<String> = total.keys().cloned().collect();
    tables.sort();

    let print_dist = |label: &str, map: &HashMap<String, HashMap<String, usize>>, table: &str| {
        if let Some(fields) = map.get(table) {
            let total: usize = fields.values().sum();
            if total == 0 { return; }
            let mut pairs: Vec<_> = fields.iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(a.1));
            let parts: Vec<String> = pairs.iter()
                .map(|(f, n)| format!("{}:{} ({:.0}%)", f, n, **n as f64 / total as f64 * 100.0))
                .collect();
            println!("    {label}: {}", parts.join("  "));
        }
    };

    for table in &tables {
        println!("\n{} ({} datums)", table, total[table]);
        print_dist("asgn", &asgn, table);
        print_dist("cond", &cond, table);
        print_dist("proj", &proj, table);
    }
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

    let logits = model.forward(&tokens, config.token_buckets, &slots, &catalog, None, &device);
    let entity_idx: usize = logits.entity.clone().argmax(0).into_scalar().elem::<i32>() as usize;

    println!("Intent: {:?}", catalog.ops[logits.intent.clone().argmax(0).into_scalar().elem::<i32>() as usize]);
    println!("Entity: {}", catalog.tables[entity_idx]);

    let temporal_vals = extract_temporal_values(&nl);
    let resolve_vals = ResolveValues {
        record_id: None,
        cond_values: temporal_vals.clone(),
        asgn_values: temporal_vals,
        mod_values: vec![],
        mod_descending: vec![],
    };

    let ir = model.resolve(&logits, &catalog, &resolve_vals);
    let q = ir.render(&[]);
    println!("SurrealQL: {}", q.surql);
}
