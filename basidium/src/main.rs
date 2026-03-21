use basidium::{trainable::{Basidium, BasidiumTrainCtx}, trainer::{Trainer, TrainerConfig}, Datum};
use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
use burn::module::Module;
use hyphae::{graph::SchemaGraph, model::HyphaeConfig, schema::Schema};
use septa::model::SeptaConfig;
use std::path::Path;

type B = Autodiff<Wgpu>;
type InferB = Wgpu;

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
        "infer" => cmd_infer(),
        "eval" => cmd_eval(),
        other => {
            eprintln!("unknown command: {other}");
            eprintln!("usage: basidium <generate|stats|train|infer|eval>");
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
        hyphae_config, septa_config, schema_graph, 1e-3, num_iters, &device,
    );
    let mut trainer = Trainer::new(trainer_config, ctx, "weights/basidium");

    let result = trainer.train(&train_data, &val_data);
    println!(
        "Best epoch: {} — val_loss={:.4} val_acc={:.3}",
        result.best_epoch, result.best_metrics.val_loss, result.best_metrics.val_acc
    );
    println!("  {}", result.best_metrics.head_acc.display());
}

fn cmd_infer() {
    let dir = data_dir();
    let val_data = load_json(&format!("{dir}/val.json"));
    println!("Loaded {} val datums", val_data.len());

    let device = WgpuDevice::default();
    let schema = Schema::from_dir(Path::new(SCHEMA_DIR)).unwrap();
    let hyphae_config = HyphaeConfig::new();
    let septa_config = SeptaConfig::new(12);
    let schema_graph = SchemaGraph::new(schema, hyphae_config.ngram_buckets);

    // Build model and load pretrained weights
    let hyphae = hyphae::model::Hyphae::<InferB>::new(&hyphae_config, &device);
    let septa = septa::model::Septa::<InferB>::new(&septa_config, &device);
    let mut model = Basidium { septa, hyphae };

    let weights_path = Path::new("weights/basidium/best.bin");
    if !weights_path.exists() {
        eprintln!("No weights found at {}", weights_path.display());
        eprintln!("Run `basidium train` first");
        std::process::exit(1);
    }

    {
        use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(weights_path.to_path_buf(), &device).unwrap();
        model = model.load_record(record);
    }
    println!("Loaded weights from {}", weights_path.display());

    // Pick a sample of datums to test (first N, or all if small)
    let n = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let sample = &val_data[..n.min(val_data.len())];

    let mut correct = 0usize;
    let mut total = 0usize;

    for datum in sample {
        let hiddens = model.septa.forward_with_spans(
            &datum.nl, &datum.semantics, septa_config.vocab_size, &device,
        );
        let grounded = schema_graph.inject(&datum.semantics);
        let logits = model.hyphae.forward(&grounded, &hiddens, &device);
        let ir = hyphae::model::Hyphae::<InferB>::resolve(&logits, &grounded, &datum.semantics);
        let query = ir.render(&[]);

        // Compare predicted vs ground truth labels
        let matches = check_predictions(&ir, &datum.labels, &grounded.nodes);

        println!("---");
        println!("  NL:    {}", datum.nl);
        println!("  SurQL: {}", query.surql);
        if matches {
            println!("  ✓ all heads correct");
            correct += 1;
        } else {
            println!("  ✗ mismatch (see labels)");
            print_mismatches(&ir, &datum.labels, &grounded.nodes);
        }
        total += 1;
    }

    println!("\n=== Inference Results ===");
    println!("{correct}/{total} datums fully correct ({:.1}%)", correct as f32 / total as f32 * 100.0);
}

fn cmd_eval() {
    let eval_data = Datum::generate_eval();

    let device = WgpuDevice::default();
    let schema = Schema::from_dir(Path::new(SCHEMA_DIR)).unwrap();
    let hyphae_config = HyphaeConfig::new();
    let septa_config = SeptaConfig::new(12);
    let schema_graph = SchemaGraph::new(schema, hyphae_config.ngram_buckets);

    let hyphae = hyphae::model::Hyphae::<InferB>::new(&hyphae_config, &device);
    let septa = septa::model::Septa::<InferB>::new(&septa_config, &device);
    let mut model = Basidium { septa, hyphae };

    let weights_path = Path::new("weights/basidium/best.bin");
    if !weights_path.exists() {
        eprintln!("No weights found at {}", weights_path.display());
        eprintln!("Run `basidium train` first");
        std::process::exit(1);
    }

    {
        use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(weights_path.to_path_buf(), &device).unwrap();
        model = model.load_record(record);
    }
    println!("Loaded weights from {}\n", weights_path.display());

    let mut correct = 0usize;
    let mut total = 0usize;

    for datum in &eval_data {
        let hiddens = model.septa.forward_with_spans(
            &datum.nl, &datum.semantics, septa_config.vocab_size, &device,
        );
        let grounded = schema_graph.inject(&datum.semantics);
        let logits = model.hyphae.forward(&grounded, &hiddens, &device);
        let ir = hyphae::model::Hyphae::<InferB>::resolve(&logits, &grounded, &datum.semantics);
        let query = ir.render(&[]);

        let matches = check_predictions(&ir, &datum.labels, &grounded.nodes);

        println!("---");
        println!("  NL:    {}", datum.nl);
        println!("  SurQL: {}", query.surql);
        if matches {
            println!("  ✓ correct");
            correct += 1;
        } else {
            println!("  ✗ MISMATCH");
            print_mismatches(&ir, &datum.labels, &grounded.nodes);
        }
        total += 1;
    }

    println!("\n=== Eval Results ===");
    println!("{correct}/{total} datums correct ({:.1}%)", correct as f32 / total as f32 * 100.0);
}

/// Check if all label predictions match ground truth.
fn check_predictions(
    ir: &hyphae::query::QueryIr,
    labels: &[basidium::SpanLabel],
    _nodes: &[hyphae::query::QueryNode],
) -> bool {
    use basidium::SpanType;
    use hyphae::query::QueryNode;

    for label in labels {
        let ok = match (&label.span_type, &label.target) {
            (SpanType::Intent, QueryNode::Operation(expected)) => ir.intent == *expected,
            (SpanType::Entity, QueryNode::Table(expected)) => ir.table == *expected,
            (SpanType::Projection, QueryNode::Field { table, name }) => {
                ir.projections.iter().any(|p| &p.table == table && &p.field == name)
            }
            (SpanType::Condition, QueryNode::Field { table, name }) => {
                ir.conditions.iter().any(|c| &c.table == table && &c.field == name)
            }
            (SpanType::Condition, QueryNode::Comparator(expected)) => {
                ir.conditions.iter().any(|c| &c.comparator == expected)
            }
            (SpanType::Assignment, QueryNode::Field { table, name }) => {
                ir.assignments.iter().any(|a| a.field.as_ref() == Some(name) && &a.table == table)
            }
            (SpanType::Modifier, QueryNode::Modifier(expected)) => {
                use hyphae::query::ModifierKind;
                ir.modifiers.iter().any(|m| match (expected, m) {
                    (ModifierKind::OrderBy, hyphae::query::ResolvedModifier::OrderBy { .. }) => true,
                    (ModifierKind::Limit, hyphae::query::ResolvedModifier::Limit { .. }) => true,
                    (ModifierKind::Fetch, hyphae::query::ResolvedModifier::Fetch { .. }) => true,
                    _ => false,
                })
            }
            (SpanType::Modifier, QueryNode::Field { name, .. }) => {
                ir.modifiers.iter().any(|m| match m {
                    hyphae::query::ResolvedModifier::OrderBy { field, .. } => field == name,
                    hyphae::query::ResolvedModifier::Fetch { field } => field == name,
                    _ => false,
                })
            }
            _ => true, // unknown combos — skip
        };
        if !ok { return false; }
    }
    true
}

fn print_mismatches(
    ir: &hyphae::query::QueryIr,
    labels: &[basidium::SpanLabel],
    _nodes: &[hyphae::query::QueryNode],
) {
    use basidium::SpanType;
    use hyphae::query::QueryNode;

    for label in labels {
        let (head, expected, got) = match (&label.span_type, &label.target) {
            (SpanType::Intent, QueryNode::Operation(e)) => {
                if ir.intent != *e { ("intent", format!("{e:?}"), format!("{:?}", ir.intent)) } else { continue }
            }
            (SpanType::Entity, QueryNode::Table(e)) => {
                if ir.table != *e { ("entity", e.clone(), ir.table.clone()) } else { continue }
            }
            (SpanType::Projection, QueryNode::Field { table, name }) => {
                if !ir.projections.iter().any(|p| &p.table == table && &p.field == name) {
                    let got_fields: Vec<String> = ir.projections.iter().map(|p| format!("{}.{}", p.table, p.field)).collect();
                    ("proj", format!("{table}.{name}"), got_fields.join(", "))
                } else { continue }
            }
            (SpanType::Condition, QueryNode::Field { table, name }) => {
                if !ir.conditions.iter().any(|c| &c.table == table && &c.field == name) {
                    let got: Vec<String> = ir.conditions.iter().map(|c| format!("{}.{}", c.table, c.field)).collect();
                    ("cond_field", format!("{table}.{name}"), got.join(", "))
                } else { continue }
            }
            (SpanType::Condition, QueryNode::Comparator(e)) => {
                if !ir.conditions.iter().any(|c| &c.comparator == e) {
                    let got: Vec<String> = ir.conditions.iter().map(|c| format!("{:?}", c.comparator)).collect();
                    ("cond_cmp", format!("{e:?}"), got.join(", "))
                } else { continue }
            }
            _ => continue,
        };
        println!("    {head}: expected={expected}  got={got}");
    }
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

