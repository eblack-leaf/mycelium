//! Generate training data for the schema re-ranker.
//!
//! Reads the NLP dataset (dataset_nlp.json) which has:
//!   - linguistic_graph.nodes[].text  — phrases extracted by biaffine
//!   - ground_truth.targets[]         — correct (phrase → schema node) mappings
//!
//! Groups all unique phrase wordings per schema node across the full dataset,
//! giving natural surface-form variation (e.g., "email", "email address",
//! "user's email" all map to field "email"). Each unique (phrase, schema_node)
//! pair is a positive. Hard negatives are sampled from same-type schema nodes
//! (other fields when GT is a field, other tables when GT is a table).
//!
//! Schema names are prefix-stripped for fields ("users.email" → "email").
//!
//! Usage:
//!   cargo run --release --example gen_reranker_dataset -p gnn-burn

use std::path::Path;
use std::collections::{HashMap, HashSet};
use gnn_burn::training::Dataset;
use gnn_burn::graph::SchemaGraph;
use gnn_burn::operations::all_operations;
use gnn_burn::nlp::{NlpModel, NlpConfig};
use gnn_burn::reranker_data::{RerankerPair, RerankerDataset};
use gnn_burn::schema::{Reader, Extractor};
use rand::seq::SliceRandom;
use rand::rng;

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");

    // --- Load NLP dataset ---
    let nlp_path = demo_dir.join("dataset_nlp.json");
    if !nlp_path.exists() {
        eprintln!("dataset_nlp.json not found — run gen_dataset_nlp first");
        std::process::exit(1);
    }
    let dataset = Dataset::load(&nlp_path).expect("load NLP dataset");
    println!("loaded {} NLP samples", dataset.samples.len());

    // --- Load schema ---
    let raw = Reader::read(&demo_dir.join("schema.surql")).expect("read schema");
    let (schema, _) = Extractor::extract(&raw);
    let schema_graph = SchemaGraph::from_schema(&schema);
    let operations = all_operations();

    // --- Build schema name lists (prefix-stripped for fields) ---
    let table_names: Vec<String> = schema_graph.table_nodes.iter()
        .map(|n| n.name.clone())
        .collect();
    let field_names: Vec<String> = schema_graph.field_nodes.iter()
        .map(|n| n.name.splitn(2, '.').nth(1).unwrap_or(&n.name).to_string())
        .collect();
    let op_names: Vec<String> = operations.iter()
        .map(|op| op.name.clone())
        .collect();

    println!("schema: {} tables, {} fields, {} ops",
        table_names.len(), field_names.len(), op_names.len());

    // --- Load MiniLM for encoding ---
    println!("Loading MiniLM...");
    let nlp = NlpModel::load(&NlpConfig {
        model_path: model_dir.join("model.onnx").to_string_lossy().into(),
        tokenizer_path: model_dir.join("tokenizer.json").to_string_lossy().into(),
        cross_model_path: model_dir.join("cross-encoder.onnx").to_string_lossy().into(),
        cross_tokenizer_path: model_dir.join("cross-tokenizer.json").to_string_lossy().into(),
        biaffine_model_path: None,
        ngram_model_path: None,
    }).expect("load NLP models");

    // --- Pre-compute schema name embeddings ---
    println!("Encoding schema names...");
    let mut schema_emb_cache: HashMap<String, Vec<f32>> = HashMap::new();
    for name in table_names.iter().chain(field_names.iter()).chain(op_names.iter()) {
        if !schema_emb_cache.contains_key(name) {
            let emb = nlp.encode_pooled(name).expect("encode schema name");
            schema_emb_cache.insert(name.clone(), emb);
        }
    }
    println!("cached {} unique schema name embeddings", schema_emb_cache.len());

    // --- Collect all unique (phrase_text, schema_key) positive pairs ---
    // schema_key = "type:id" e.g. "field:3"
    // Also collect all unique phrase texts per schema node for variation counting
    let mut positive_set: HashSet<(String, String, String, usize)> = HashSet::new(); // (phrase_text, type, name, id)
    let mut skipped = 0usize;

    for sample in &dataset.samples {
        let id_to_text: HashMap<usize, &str> = sample.linguistic_graph.nodes.iter()
            .map(|n| (n.id, n.text.as_str()))
            .collect();

        for target in &sample.ground_truth.targets {
            let phrase_text = match id_to_text.get(&target.linguistic_node) {
                Some(t) => *t,
                None => { skipped += 1; continue; }
            };

            let gt_schema_name = match target.target_type.as_str() {
                "table" => table_names.get(target.target_id),
                "field" => field_names.get(target.target_id),
                "operation" => op_names.get(target.target_id),
                _ => None,
            };
            let gt_schema_name = match gt_schema_name {
                Some(n) => n.clone(),
                None => { skipped += 1; continue; }
            };

            positive_set.insert((
                phrase_text.to_string(),
                target.target_type.clone(),
                gt_schema_name,
                target.target_id,
            ));
        }
    }

    println!("found {} unique positive (phrase, schema) pairs ({} skipped)", positive_set.len(), skipped);

    // --- Count variations per schema node ---
    let mut variations_per_node: HashMap<String, Vec<String>> = HashMap::new();
    for (phrase, typ, _name, id) in &positive_set {
        let key = format!("{}:{}", typ, id);
        variations_per_node.entry(key).or_default().push(phrase.clone());
    }
    let avg_variations = variations_per_node.values().map(|v| v.len()).sum::<usize>() as f64
        / variations_per_node.len().max(1) as f64;
    println!("avg {:.1} phrase variations per schema node", avg_variations);

    // --- Encode all unique phrase texts ---
    println!("Encoding phrases...");
    let unique_phrases: HashSet<&str> = positive_set.iter().map(|(p, _, _, _)| p.as_str()).collect();
    let mut phrase_emb_cache: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, phrase) in unique_phrases.iter().enumerate() {
        if !phrase_emb_cache.contains_key(*phrase) {
            let emb = nlp.encode_pooled(phrase).expect("encode phrase");
            phrase_emb_cache.insert(phrase.to_string(), emb);
        }
        if (i + 1) % 500 == 0 {
            println!("  encoded {}/{} phrases", i + 1, unique_phrases.len());
        }
    }
    println!("encoded {} unique phrases", phrase_emb_cache.len());

    // --- Generate pairs ---
    let mut rng = rng();
    let mut pairs: Vec<RerankerPair> = Vec::new();
    let neg_per_positive = 3; // hard negatives from same type

    for (phrase, typ, name, id) in &positive_set {
        let phrase_emb = &phrase_emb_cache[phrase];
        let schema_emb = &schema_emb_cache[name];

        // Positive
        pairs.push(RerankerPair {
            phrase_emb: phrase_emb.clone(),
            schema_emb: schema_emb.clone(),
            label: 1.0,
        });

        // Hard negatives: same type, different node
        let same_type_names: Vec<(usize, &String)> = match typ.as_str() {
            "table" => table_names.iter().enumerate().collect(),
            "field" => field_names.iter().enumerate().collect(),
            "operation" => op_names.iter().enumerate().collect(),
            _ => vec![],
        };
        let mut neg_pool: Vec<&(usize, &String)> = same_type_names.iter()
            .filter(|(i, _)| *i != *id)
            .collect();
        neg_pool.shuffle(&mut rng);

        for (_, neg_name) in neg_pool.iter().take(neg_per_positive) {
            let neg_emb = &schema_emb_cache[neg_name.as_str()];
            pairs.push(RerankerPair {
                phrase_emb: phrase_emb.clone(),
                schema_emb: neg_emb.clone(),
                label: 0.0,
            });
        }
    }

    // Shuffle all pairs
    pairs.shuffle(&mut rng);

    let n_pos = pairs.iter().filter(|p| p.label > 0.5).count();
    let n_neg = pairs.len() - n_pos;
    println!("generated {} pairs ({} pos, {} neg)", pairs.len(), n_pos, n_neg);

    let ds = RerankerDataset { pairs };
    ds.save(&demo_dir.join("reranker_dataset.json")).expect("save dataset");
    println!("saved to demo/reranker_dataset.json");
}
