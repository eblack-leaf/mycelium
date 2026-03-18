//! End-to-end demo of the 3-stage NL→Schema pipeline.
//!
//! Stage 1: NL parse (rule-based + MiniLM embeddings) → LinguisticGraph
//! Stage 2: Cross-encoder candidate matching → CandidateSet
//! Stage 3: GNN (embed → SAGEConv → OutputHead) → role + target predictions
//!
//! Usage:
//!   cargo run --release --example pipeline_demo -p gnn-burn

use std::collections::HashMap;
use std::path::Path;
use burn::tensor::backend::Backend;
use burn::backend::NdArray;

use gnn_burn::nlp::{NlpModel, NlpConfig, LinguisticGraph, SpanType};
use gnn_burn::candidate_matcher::{CandidateMatcher, CandidateMatcherConfig, CandidateSet};
use gnn_burn::schema::{Reader, Extractor};
use gnn_burn::graph::SchemaGraph;
use gnn_burn::operations::{all_operations, OpNode};
use gnn_burn::embed::{GloveVocab, Embedder, create_type_embedding};
use gnn_burn::linguistic_graph::LinguisticConv;
use gnn_burn::sage::Encoder;
use gnn_burn::head::{OutputHead, HeadLogits, CandidateMask};
use gnn_burn::training::{load_model, project_linguistic};
use burn::nn::LinearConfig;

type B = NdArray;

const HIDDEN_DIM: usize = 64;
const TYPE_DIM: usize = 16;
const N_LAYERS: usize = 2;

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_dir = demo_dir.join("models");
    let schema_path = demo_dir.join("demo/schema.surql");
    let glove_path = demo_dir.join("demo/glove.6B.300d.txt");

    // --- Load NLP models ---
    println!("Loading NLP models...");
    // Use biaffine head if model exists
    let biaffine_path = demo_dir.join("demo/biaffine_model");
    let biaffine_model_path = if biaffine_path.with_extension("mpk").exists() {
        Some(biaffine_path.to_string_lossy().into_owned())
    } else {
        None
    };

    let nlp = NlpModel::load(&NlpConfig {
        model_path: model_dir.join("model.onnx").to_string_lossy().into(),
        tokenizer_path: model_dir.join("tokenizer.json").to_string_lossy().into(),
        cross_model_path: model_dir.join("cross-encoder.onnx").to_string_lossy().into(),
        cross_tokenizer_path: model_dir.join("cross-tokenizer.json").to_string_lossy().into(),
        biaffine_model_path,
    }).expect("load NLP models");

    // --- Load schema ---
    let raw = Reader::read(&schema_path).expect("read schema");
    let (schema, _) = Extractor::extract(&raw);
    let schema_graph = SchemaGraph::from_schema(&schema);
    let operations = all_operations();

    // --- Load GloVe ---
    println!("Loading GloVe...");
    let glove = GloveVocab::load(&glove_path, 42).expect("load GloVe");
    let embedder = Embedder::new(glove, TYPE_DIM, 384);

    // --- Build candidate matcher ---
    let matcher = CandidateMatcher::new(
        &schema_graph,
        &operations,
        CandidateMatcherConfig { top_k: 5, min_score: 0.0 },
    );

    // --- Load or initialize GNN ---
    let device = Default::default();
    let gnn_model_path = demo_dir.join("demo/gnn_model");

    let embed_dim = embedder.schema_dim();
    // CompactRecorder appends .mpk — check for the actual file
    let (type_embed, ling_proj, encoder, output_head) = if gnn_model_path.with_extension("mpk").exists() {
        println!("Loading trained GNN model...");
        let (model, _, _) = load_model(
            &gnn_model_path.to_string_lossy(),
            &schema_path.to_string_lossy(),
            &glove_path.to_string_lossy(),
            HIDDEN_DIM, N_LAYERS, TYPE_DIM,
        );
        (model.type_embed, model.ling_proj, model.encoder, model.head)
    } else {
        println!("No trained model at {:?}, using random weights", gnn_model_path);
        let template_conv = LinguisticConv::template(&schema_graph);
        let input_dims: HashMap<String, usize> = template_conv.node_counts.iter()
            .map(|(name, _)| (name.clone(), embed_dim))
            .collect();
        (
            create_type_embedding::<B>(TYPE_DIM, &device),
            LinearConfig::new(384, embed_dim).init::<B>(&device),
            Encoder::<B>::new(&template_conv, &input_dims, HIDDEN_DIM, N_LAYERS, &device),
            OutputHead::<B>::new(HIDDEN_DIM, &device),
        )
    };

    println!("Schema: {} tables, {} fields, {} operations",
        schema_graph.table_nodes.len(),
        schema_graph.field_nodes.len(),
        operations.len());
    println!("Embed dim: {}, Hidden dim: {}, Layers: {}", embed_dim, HIDDEN_DIM, N_LAYERS);

    // --- Run queries ---
    let queries = [
        "show me the goods' timestamp, first 49",
        "find users where age is over 25",
        "get all posts by rating",
        "list the messages' body",
        "count products where cost is above 100",
    ];

    for query in &queries {
        println!("\n{}", "=".repeat(70));
        println!("Query: \"{}\"", query);
        println!("{}", "=".repeat(70));

        // Stage 1: NL parse
        let ling_graph = nlp.parse(query);
        print_stage1(&ling_graph);

        // Stage 2: Cross-encoder candidate matching
        let candidates = matcher.match_candidates(&nlp, &ling_graph);
        print_stage2(&ling_graph, &candidates, &schema_graph, &operations);

        // Stage 3: GNN
        let conv = LinguisticConv::new(&schema_graph, &ling_graph, &candidates);

        // Embed all nodes
        let mut initial = embedder.embed_all::<B>(
            &type_embed, &schema, &schema_graph, &ling_graph, &operations, &device,
        );
        project_linguistic(&mut initial, &ling_proj);

        // Run SAGEConv encoder
        let encoded = encoder.forward(&conv, initial, &device);

        // Output head: role + target prediction (masked by candidate edges)
        let mask = CandidateMask::from_candidates(
            &ling_graph, &candidates,
            schema_graph.table_nodes.len(), schema_graph.field_nodes.len(), operations.len(),
        );
        let logits = output_head.forward(&encoded, Some(&mask));
        print_stage3(&ling_graph, &logits, &schema_graph, &operations, &device);
    }
}

fn print_stage1(ling: &LinguisticGraph) {
    println!("\n  Stage 1 — Linguistic Graph:");
    for node in &ling.nodes {
        println!("    [{:?}] \"{}\" (tokens {}..{})",
            node.span_type, node.text, node.token_span.0, node.token_span.1);
    }
    for edge in &ling.edges {
        let src = &ling.nodes[edge.src];
        let dst = &ling.nodes[edge.dst];
        println!("    {} \"{}\" --{:?}--> {} \"{}\"",
            src.id, src.text, edge.relation, dst.id, dst.text);
    }
}

fn print_stage2(
    ling: &LinguisticGraph,
    candidates: &CandidateSet,
    sg: &SchemaGraph,
    ops: &[OpNode],
) {
    println!("\n  Stage 2 — Candidate Matches:");
    for node in &ling.nodes {
        let mut node_edges: Vec<_> = candidates.edges.iter()
            .filter(|e| e.linguistic_node == node.id)
            .collect();
        node_edges.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        println!("    \"{}\" ({:?}):", node.text, node.span_type);
        for edge in node_edges.iter().take(3) {
            let name = match edge.schema_node_type.as_str() {
                "table" => sg.table_nodes[edge.schema_node_id].name.as_str(),
                "field" => sg.field_nodes[edge.schema_node_id].name.as_str(),
                "operation" => ops[edge.schema_node_id].name.as_str(),
                _ => "?",
            };
            println!("      {} {:>25} = {:.4}",
                edge.schema_node_type, name, edge.score);
        }
    }
}

fn print_stage3(
    ling: &LinguisticGraph,
    logits: &HeadLogits<B>,
    sg: &SchemaGraph,
    ops: &[OpNode],
    _device: &<B as Backend>::Device,
) {
    println!("\n  Stage 3 — GNN Resolution:");

    let role_logits = &logits.role_logits;
    let [n_ling, _n_roles] = role_logits.dims();

    if n_ling == 0 {
        println!("    (no linguistic nodes)");
        return;
    }

    // Collect all linguistic nodes in order (same order as head concatenates them)
    let ling_types = [SpanType::NounPhrase, SpanType::Quantifier, SpanType::Comparator, SpanType::Intent];
    let ling_type_names = ["np", "quantifier", "comparator", "intent"];
    let mut ordered_nodes: Vec<(usize, &str)> = Vec::new();
    for (st, name) in ling_types.iter().zip(ling_type_names.iter()) {
        for node in &ling.nodes {
            if node.span_type == *st {
                ordered_nodes.push((node.id, name));
            }
        }
    }

    let role_names = ["Collection", "Field", "FilterField", "Modifier", "Traversal", "None"];

    let role_data = role_logits.clone().into_data();
    let role_vals: Vec<f32> = role_data.to_vec().unwrap();

    for (i, (node_id, _type_name)) in ordered_nodes.iter().enumerate() {
        if i >= n_ling { break; }
        let node = &ling.nodes[*node_id];

        // Find predicted role (argmax)
        let row_start = i * role_names.len();
        let row = &role_vals[row_start..row_start + role_names.len()];
        let (best_role_idx, best_role_score) = row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // Find best target
        let target_str = find_best_target(i, logits, sg, ops);

        println!("    \"{}\" ({:?})", node.text, node.span_type);
        println!("      Role: {} ({:.4})", role_names[best_role_idx], best_role_score);
        println!("      Target: {}", target_str);
    }
}

fn find_best_target(
    ling_idx: usize,
    logits: &HeadLogits<B>,
    sg: &SchemaGraph,
    ops: &[OpNode],
) -> String {
    let mut best: Option<(f32, String)> = None;

    if let Some(ref t) = logits.target_table {
        let dims = t.dims();
        if ling_idx < dims[0] && dims[1] > 0 {
            let data = t.clone().into_data();
            let vals: Vec<f32> = data.to_vec().unwrap();
            let row_start = ling_idx * dims[1];
            for j in 0..dims[1] {
                let score = vals[row_start + j];
                if best.is_none() || score > best.as_ref().unwrap().0 {
                    let name = &sg.table_nodes[j].name;
                    best = Some((score, format!("table:{} ({:.4})", name, score)));
                }
            }
        }
    }

    if let Some(ref t) = logits.target_field {
        let dims = t.dims();
        if ling_idx < dims[0] && dims[1] > 0 {
            let data = t.clone().into_data();
            let vals: Vec<f32> = data.to_vec().unwrap();
            let row_start = ling_idx * dims[1];
            for j in 0..dims[1] {
                let score = vals[row_start + j];
                if best.is_none() || score > best.as_ref().unwrap().0 {
                    let name = &sg.field_nodes[j].name;
                    best = Some((score, format!("field:{} ({:.4})", name, score)));
                }
            }
        }
    }

    if let Some(ref t) = logits.target_op {
        let dims = t.dims();
        if ling_idx < dims[0] && dims[1] > 0 {
            let data = t.clone().into_data();
            let vals: Vec<f32> = data.to_vec().unwrap();
            let row_start = ling_idx * dims[1];
            for j in 0..dims[1] {
                let score = vals[row_start + j];
                if best.is_none() || score > best.as_ref().unwrap().0 {
                    let name = &ops[j].name;
                    best = Some((score, format!("op:{} ({:.4})", name, score)));
                }
            }
        }
    }

    best.map(|(_, s)| s).unwrap_or_else(|| "none".to_string())
}
