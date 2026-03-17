//! Generate training dataset from the demo schema.
//!
//! Produces LinguisticGraph + CandidateSet + GroundTruth for each sample.
//! The LinguisticGraph is synthetic (no real NLP model needed).
//! The CandidateSet simulates cross-encoder output with noise.
//!
//! Usage:
//!   cargo run --release --example gen_dataset -p gnn-burn

use std::path::Path;
use rand::prelude::*;
use rand::rngs::StdRng;
use gnn_burn::schema::{Reader, Extractor, Schema, FieldType};
use gnn_burn::graph::SchemaGraph;
use gnn_burn::nlp::{LinguisticGraph, LinguisticNode, LinguisticEdge, SpanType, DepRelation};
use gnn_burn::candidate_matcher::{CandidateEdge, CandidateSet};
use gnn_burn::training::{TrainingSample, GroundTruth, NodeTarget, SchemaRole, Dataset};

// =============================================================================
// Schema metadata
// =============================================================================

struct TableMeta {
    id: usize,
    name: String,
    fields: Vec<FieldMeta>,
}

struct FieldMeta {
    global_id: usize,
    local_name: String,
    field_type: FieldType,
    record_target: Option<usize>,
}

struct SchemaMeta {
    tables: Vec<TableMeta>,
}

impl SchemaMeta {
    fn from_schema(schema: &Schema, schema_graph: &SchemaGraph) -> Self {
        let mut tables = Vec::new();
        let mut field_offset = 0;

        for (table_id, table) in schema.tables.iter().enumerate() {
            let mut fields = Vec::new();
            for (i, field) in table.fields.iter().enumerate() {
                let record_target = extract_record_target(&field.field_type, schema);
                fields.push(FieldMeta {
                    global_id: field_offset + i,
                    local_name: field.name.clone(),
                    field_type: field.field_type.clone(),
                    record_target,
                });
            }
            field_offset += table.fields.len();
            tables.push(TableMeta {
                id: table_id,
                name: table.name.clone(),
                fields,
            });
        }

        assert_eq!(field_offset, schema_graph.field_nodes.len());
        Self { tables }
    }

    fn filterable_fields<'a>(&self, table: &'a TableMeta) -> Vec<&'a FieldMeta> {
        table.fields.iter()
            .filter(|f| matches!(f.field_type,
                FieldType::String | FieldType::Int | FieldType::Float |
                FieldType::Decimal | FieldType::Number | FieldType::Bool |
                FieldType::Datetime | FieldType::Duration
            ))
            .collect()
    }

    fn orderable_fields<'a>(&self, table: &'a TableMeta) -> Vec<&'a FieldMeta> {
        table.fields.iter()
            .filter(|f| matches!(f.field_type,
                FieldType::String | FieldType::Int | FieldType::Float |
                FieldType::Decimal | FieldType::Number | FieldType::Datetime
            ))
            .collect()
    }

    fn non_record_fields<'a>(&self, table: &'a TableMeta) -> Vec<&'a FieldMeta> {
        table.fields.iter()
            .filter(|f| f.record_target.is_none())
            .collect()
    }

    fn record_fields<'a>(&self, table: &'a TableMeta) -> Vec<&'a FieldMeta> {
        table.fields.iter()
            .filter(|f| f.record_target.is_some())
            .collect()
    }
}

fn extract_record_target(ft: &FieldType, schema: &Schema) -> Option<usize> {
    match ft {
        FieldType::Record { tables } if !tables.is_empty() => {
            schema.tables.iter().position(|t| t.name == tables[0])
        }
        FieldType::Option { inner } => extract_record_target(inner, schema),
        _ => None,
    }
}

// =============================================================================
// Surface forms
// =============================================================================

fn collection_surface(name: &str, rng: &mut impl Rng) -> String {
    let forms = [
        name.to_string(),
        format!("all {}", name),
        format!("the {}", name),
        format!("every {}", name),
        name.trim_end_matches('s').to_string(),
    ];
    forms[rng.random_range(0..forms.len())].clone()
}

fn field_surface(name: &str, rng: &mut impl Rng) -> String {
    let forms = [
        name.to_string(),
        format!("the {}", name),
        format!("their {}", name),
        format!("{}s", name),
    ];
    forms[rng.random_range(0..forms.len())].clone()
}

fn filter_op_surface(op_id: usize, rng: &mut impl Rng) -> String {
    let forms: &[&str] = match op_id {
        11 => &["equals", "is", "equal to", "matching"],
        12 => &["not", "not equal to", "different from"],
        13 => &["greater than", "more than", "above", "over"],
        14 => &["less than", "under", "below", "fewer than"],
        15 => &["at least", "minimum", "no less than"],
        16 => &["at most", "maximum", "no more than"],
        17 => &["like", "matching pattern", "similar to"],
        18 => &["containing", "includes", "has"],
        19 => &["starting with", "begins with"],
        20 => &["ending with", "ends with"],
        _ => &["equals"],
    };
    forms[rng.random_range(0..forms.len())].to_string()
}

fn random_value(ft: &FieldType, rng: &mut impl Rng) -> String {
    match ft {
        FieldType::Int => rng.random_range(1..500).to_string(),
        FieldType::Float | FieldType::Decimal | FieldType::Number =>
            format!("{:.1}", rng.random_range(0.5..100.0)),
        FieldType::String => {
            let words = ["hello", "world", "test", "admin", "user", "active", "pending",
                "john", "alice", "bob", "news", "update", "review", "red", "blue"];
            words[rng.random_range(0..words.len())].to_string()
        }
        FieldType::Bool => if rng.random_bool(0.5) { "true" } else { "false" }.to_string(),
        FieldType::Datetime => format!("2024-{:02}-{:02}", rng.random_range(1..13), rng.random_range(1..29)),
        _ => "value".to_string(),
    }
}

fn compatible_filter_ops(ft: &FieldType) -> Vec<usize> {
    match ft {
        FieldType::Int | FieldType::Float | FieldType::Decimal | FieldType::Number =>
            vec![11, 12, 13, 14, 15, 16],
        FieldType::String => vec![11, 12, 17, 18, 19, 20],
        FieldType::Bool => vec![11, 12],
        FieldType::Datetime | FieldType::Duration => vec![11, 12, 13, 14, 15, 16],
        _ => vec![11, 12],
    }
}

// =============================================================================
// Candidate edge generation (simulates cross-encoder output)
// =============================================================================

/// Generate candidate edges for a linguistic node.
/// Puts the correct target first with a high score, adds distractor edges.
fn make_candidates(
    ling_node_id: usize,
    correct_type: &str,
    correct_id: usize,
    meta: &SchemaMeta,
    rng: &mut impl Rng,
) -> Vec<CandidateEdge> {
    let mut edges = Vec::new();

    // Correct target — high score
    edges.push(CandidateEdge {
        linguistic_node: ling_node_id,
        schema_node_type: correct_type.to_string(),
        schema_node_id: correct_id,
        score: rng.random_range(0.65..0.95),
    });

    // Distractor edges
    let n_distractors = rng.random_range(1..=4);
    for _ in 0..n_distractors {
        let (dtype, did) = match rng.random_range(0..3) {
            0 => {
                let tid = loop {
                    let t = rng.random_range(0..meta.tables.len());
                    if correct_type != "table" || t != correct_id { break t; }
                };
                ("table", tid)
            }
            1 => {
                let total_fields: usize = meta.tables.iter().map(|t| t.fields.len()).sum();
                if total_fields == 0 { continue; }
                let fid = loop {
                    let f = rng.random_range(0..total_fields);
                    if correct_type != "field" || f != correct_id { break f; }
                };
                ("field", fid)
            }
            _ => {
                let oid = rng.random_range(0..34); // 34 operations
                if correct_type == "operation" && oid == correct_id { continue; }
                ("operation", oid)
            }
        };
        edges.push(CandidateEdge {
            linguistic_node: ling_node_id,
            schema_node_type: dtype.to_string(),
            schema_node_id: did,
            score: rng.random_range(0.05..0.45),
        });
    }

    edges
}

/// Generate weak/noise candidates for nodes with no schema target (intent, noise).
fn make_noise_candidates(
    ling_node_id: usize,
    meta: &SchemaMeta,
    rng: &mut impl Rng,
) -> Vec<CandidateEdge> {
    let mut edges = Vec::new();
    let n = rng.random_range(1..=3);
    for _ in 0..n {
        let total_fields: usize = meta.tables.iter().map(|t| t.fields.len()).sum();
        let (dtype, did) = match rng.random_range(0..3) {
            0 => ("table", rng.random_range(0..meta.tables.len())),
            1 if total_fields > 0 => ("field", rng.random_range(0..total_fields)),
            _ => ("operation", rng.random_range(0..34)),
        };
        edges.push(CandidateEdge {
            linguistic_node: ling_node_id,
            schema_node_type: dtype.to_string(),
            schema_node_id: did,
            score: rng.random_range(0.01..0.20),
        });
    }
    edges
}

// =============================================================================
// Sample builders
// =============================================================================

/// Helper: create a LinguisticNode with synthetic embedding (zeros — GloVe handles real embed).
fn make_node(id: usize, text: &str, start: usize, end: usize, span_type: SpanType) -> LinguisticNode {
    LinguisticNode {
        id, text: text.to_string(),
        token_span: (start, end), span_type,
        embedding: vec![0.0; 384], // placeholder — embedder fills real values
    }
}

fn gen_collection_only(meta: &SchemaMeta, rng: &mut impl Rng) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];

    // Intent + collection NP
    let intent_text = ["show", "find", "get", "list"][rng.random_range(0..4)];
    let coll_text = collection_surface(&table.name, rng);

    let nodes = vec![
        make_node(0, intent_text, 0, 1, SpanType::Intent),
        make_node(1, &coll_text, 1, 2, SpanType::NounPhrase),
    ];
    let edges = vec![
        LinguisticEdge { src: 0, dst: 1, relation: DepRelation::IntentTarget },
    ];

    let mut candidate_edges = Vec::new();
    candidate_edges.extend(make_noise_candidates(0, meta, rng));
    candidate_edges.extend(make_candidates(1, "table", table.id, meta, rng));

    TrainingSample {
        linguistic_graph: LinguisticGraph { raw_query: format!("{} {}", intent_text, coll_text), nodes, edges },
        candidates: CandidateSet { edges: candidate_edges },
        ground_truth: GroundTruth {
            targets: vec![
                NodeTarget { linguistic_node: 0, role: SchemaRole::None, target_type: String::new(), target_id: 0 },
                NodeTarget { linguistic_node: 1, role: SchemaRole::Collection, target_type: "table".into(), target_id: table.id },
            ],
        },
    }
}

fn gen_collection_fields(meta: &SchemaMeta, rng: &mut impl Rng) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let available = meta.non_record_fields(table);
    if available.is_empty() { return gen_collection_only(meta, rng); }

    let n_fields = rng.random_range(1..=2).min(available.len());
    let chosen: Vec<&FieldMeta> = available.choose_multiple(rng, n_fields).copied().collect();

    let intent_text = ["show", "get", "list", "find"][rng.random_range(0..4)];
    let coll_text = collection_surface(&table.name, rng);

    let mut nodes = vec![
        make_node(0, intent_text, 0, 1, SpanType::Intent),
        make_node(1, &coll_text, 1, 2, SpanType::NounPhrase),
    ];
    let mut edges = vec![
        LinguisticEdge { src: 0, dst: 1, relation: DepRelation::IntentTarget },
    ];
    let mut targets = vec![
        NodeTarget { linguistic_node: 0, role: SchemaRole::None, target_type: String::new(), target_id: 0 },
        NodeTarget { linguistic_node: 1, role: SchemaRole::Collection, target_type: "table".into(), target_id: table.id },
    ];
    let mut candidate_edges = Vec::new();
    candidate_edges.extend(make_noise_candidates(0, meta, rng));
    candidate_edges.extend(make_candidates(1, "table", table.id, meta, rng));

    let mut token_pos = 2;
    for field in chosen.iter() {
        let nid = nodes.len();
        let field_text = field_surface(&field.local_name, rng);
        nodes.push(make_node(nid, &field_text, token_pos, token_pos + 1, SpanType::NounPhrase));
        // Possessive edge from collection to field
        edges.push(LinguisticEdge { src: 1, dst: nid, relation: DepRelation::Possessive });
        targets.push(NodeTarget {
            linguistic_node: nid, role: SchemaRole::Field,
            target_type: "field".into(), target_id: field.global_id,
        });
        candidate_edges.extend(make_candidates(nid, "field", field.global_id, meta, rng));
        token_pos += 1;
    }

    let raw_query = format!("{} {}", intent_text, coll_text);
    TrainingSample {
        linguistic_graph: LinguisticGraph { raw_query, nodes, edges },
        candidates: CandidateSet { edges: candidate_edges },
        ground_truth: GroundTruth { targets },
    }
}

fn gen_collection_filter(meta: &SchemaMeta, rng: &mut impl Rng) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let filterable = meta.filterable_fields(table);
    if filterable.is_empty() { return gen_collection_only(meta, rng); }

    let field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);

    let intent_text = ["find", "get", "show"][rng.random_range(0..3)];
    let coll_text = collection_surface(&table.name, rng);
    let field_text = field_surface(&field.local_name, rng);

    let nodes = vec![
        make_node(0, intent_text, 0, 1, SpanType::Intent),
        make_node(1, &coll_text, 1, 2, SpanType::NounPhrase),
        make_node(2, &field_text, 3, 4, SpanType::NounPhrase),
        make_node(3, &format!("{} {}", op_surface, value), 5, 7, SpanType::Comparator),
    ];
    let edges = vec![
        LinguisticEdge { src: 0, dst: 1, relation: DepRelation::IntentTarget },
        LinguisticEdge { src: 3, dst: 2, relation: DepRelation::Comparison },
    ];

    let mut candidate_edges = Vec::new();
    candidate_edges.extend(make_noise_candidates(0, meta, rng));
    candidate_edges.extend(make_candidates(1, "table", table.id, meta, rng));
    candidate_edges.extend(make_candidates(2, "field", field.global_id, meta, rng));
    candidate_edges.extend(make_candidates(3, "operation", op_id, meta, rng));

    TrainingSample {
        linguistic_graph: LinguisticGraph {
            raw_query: format!("{} {} where {} {} {}", intent_text, coll_text, field_text, op_surface, value),
            nodes, edges,
        },
        candidates: CandidateSet { edges: candidate_edges },
        ground_truth: GroundTruth {
            targets: vec![
                NodeTarget { linguistic_node: 0, role: SchemaRole::None, target_type: String::new(), target_id: 0 },
                NodeTarget { linguistic_node: 1, role: SchemaRole::Collection, target_type: "table".into(), target_id: table.id },
                NodeTarget { linguistic_node: 2, role: SchemaRole::FilterField, target_type: "field".into(), target_id: field.global_id },
                NodeTarget { linguistic_node: 3, role: SchemaRole::Modifier, target_type: "operation".into(), target_id: op_id },
            ],
        },
    }
}

fn gen_collection_fields_filter(meta: &SchemaMeta, rng: &mut impl Rng) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let available = meta.non_record_fields(table);
    let filterable = meta.filterable_fields(table);
    if available.is_empty() || filterable.is_empty() { return gen_collection_only(meta, rng); }

    let sel_field = available[rng.random_range(0..available.len())];
    let filt_field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&filt_field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&filt_field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);

    let intent_text = ["show", "get", "find"][rng.random_range(0..3)];
    let coll_text = collection_surface(&table.name, rng);

    let nodes = vec![
        make_node(0, intent_text, 0, 1, SpanType::Intent),
        make_node(1, &coll_text, 1, 2, SpanType::NounPhrase),
        make_node(2, &field_surface(&sel_field.local_name, rng), 2, 3, SpanType::NounPhrase),
        make_node(3, &field_surface(&filt_field.local_name, rng), 4, 5, SpanType::NounPhrase),
        make_node(4, &format!("{} {}", op_surface, value), 6, 8, SpanType::Comparator),
    ];
    let edges = vec![
        LinguisticEdge { src: 0, dst: 1, relation: DepRelation::IntentTarget },
        LinguisticEdge { src: 1, dst: 2, relation: DepRelation::Possessive },
        LinguisticEdge { src: 4, dst: 3, relation: DepRelation::Comparison },
    ];

    let mut candidate_edges = Vec::new();
    candidate_edges.extend(make_noise_candidates(0, meta, rng));
    candidate_edges.extend(make_candidates(1, "table", table.id, meta, rng));
    candidate_edges.extend(make_candidates(2, "field", sel_field.global_id, meta, rng));
    candidate_edges.extend(make_candidates(3, "field", filt_field.global_id, meta, rng));
    candidate_edges.extend(make_candidates(4, "operation", op_id, meta, rng));

    TrainingSample {
        linguistic_graph: LinguisticGraph {
            raw_query: format!("{} {}'s {} where {} {} {}", intent_text, coll_text,
                sel_field.local_name, filt_field.local_name, op_surface, value),
            nodes, edges,
        },
        candidates: CandidateSet { edges: candidate_edges },
        ground_truth: GroundTruth {
            targets: vec![
                NodeTarget { linguistic_node: 0, role: SchemaRole::None, target_type: String::new(), target_id: 0 },
                NodeTarget { linguistic_node: 1, role: SchemaRole::Collection, target_type: "table".into(), target_id: table.id },
                NodeTarget { linguistic_node: 2, role: SchemaRole::Field, target_type: "field".into(), target_id: sel_field.global_id },
                NodeTarget { linguistic_node: 3, role: SchemaRole::FilterField, target_type: "field".into(), target_id: filt_field.global_id },
                NodeTarget { linguistic_node: 4, role: SchemaRole::Modifier, target_type: "operation".into(), target_id: op_id },
            ],
        },
    }
}

fn gen_collection_modifier(meta: &SchemaMeta, rng: &mut impl Rng) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];

    let intent_text = ["show", "get", "find", "list"][rng.random_range(0..4)];
    let coll_text = collection_surface(&table.name, rng);
    let limit_val = rng.random_range(1..50);
    let quant_text = format!("first {}", limit_val);

    let mut nodes = vec![
        make_node(0, intent_text, 0, 1, SpanType::Intent),
        make_node(1, &coll_text, 1, 2, SpanType::NounPhrase),
        make_node(2, &quant_text, 3, 5, SpanType::Quantifier),
    ];
    let mut edges = vec![
        LinguisticEdge { src: 0, dst: 1, relation: DepRelation::IntentTarget },
        LinguisticEdge { src: 2, dst: 1, relation: DepRelation::Quantifies },
    ];
    let mut targets = vec![
        NodeTarget { linguistic_node: 0, role: SchemaRole::None, target_type: String::new(), target_id: 0 },
        NodeTarget { linguistic_node: 1, role: SchemaRole::Collection, target_type: "table".into(), target_id: table.id },
        NodeTarget { linguistic_node: 2, role: SchemaRole::Modifier, target_type: "operation".into(), target_id: 10 }, // LIMIT
    ];

    let mut candidate_edges = Vec::new();
    candidate_edges.extend(make_noise_candidates(0, meta, rng));
    candidate_edges.extend(make_candidates(1, "table", table.id, meta, rng));
    candidate_edges.extend(make_candidates(2, "operation", 10, meta, rng));

    // Optionally add ORDER_BY
    if rng.random_bool(0.5) {
        let orderable = meta.orderable_fields(table);
        if let Some(field) = orderable.choose(rng) {
            let nid = nodes.len();
            nodes.push(make_node(nid, &field_surface(&field.local_name, rng), 5, 6, SpanType::NounPhrase));
            edges.push(LinguisticEdge { src: 1, dst: nid, relation: DepRelation::Possessive });
            targets.push(NodeTarget {
                linguistic_node: nid, role: SchemaRole::Field,
                target_type: "field".into(), target_id: field.global_id,
            });
            candidate_edges.extend(make_candidates(nid, "field", field.global_id, meta, rng));
        }
    }

    TrainingSample {
        linguistic_graph: LinguisticGraph {
            raw_query: format!("{} {}, {}", intent_text, coll_text, quant_text),
            nodes, edges,
        },
        candidates: CandidateSet { edges: candidate_edges },
        ground_truth: GroundTruth { targets },
    }
}

fn gen_collection_filter_modifier(meta: &SchemaMeta, rng: &mut impl Rng) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let filterable = meta.filterable_fields(table);
    if filterable.is_empty() { return gen_collection_modifier(meta, rng); }

    let field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);
    let limit_val = rng.random_range(1..50);

    let intent_text = ["find", "get", "show"][rng.random_range(0..3)];
    let coll_text = collection_surface(&table.name, rng);

    let nodes = vec![
        make_node(0, intent_text, 0, 1, SpanType::Intent),
        make_node(1, &coll_text, 1, 2, SpanType::NounPhrase),
        make_node(2, &field_surface(&field.local_name, rng), 3, 4, SpanType::NounPhrase),
        make_node(3, &format!("{} {}", op_surface, value), 5, 7, SpanType::Comparator),
        make_node(4, &format!("first {}", limit_val), 8, 10, SpanType::Quantifier),
    ];
    let edges = vec![
        LinguisticEdge { src: 0, dst: 1, relation: DepRelation::IntentTarget },
        LinguisticEdge { src: 3, dst: 2, relation: DepRelation::Comparison },
        LinguisticEdge { src: 4, dst: 1, relation: DepRelation::Quantifies },
    ];

    let mut candidate_edges = Vec::new();
    candidate_edges.extend(make_noise_candidates(0, meta, rng));
    candidate_edges.extend(make_candidates(1, "table", table.id, meta, rng));
    candidate_edges.extend(make_candidates(2, "field", field.global_id, meta, rng));
    candidate_edges.extend(make_candidates(3, "operation", op_id, meta, rng));
    candidate_edges.extend(make_candidates(4, "operation", 10, meta, rng)); // LIMIT

    TrainingSample {
        linguistic_graph: LinguisticGraph {
            raw_query: format!("{} {} where {} {} {}, first {}", intent_text, coll_text,
                field.local_name, op_surface, value, limit_val),
            nodes, edges,
        },
        candidates: CandidateSet { edges: candidate_edges },
        ground_truth: GroundTruth {
            targets: vec![
                NodeTarget { linguistic_node: 0, role: SchemaRole::None, target_type: String::new(), target_id: 0 },
                NodeTarget { linguistic_node: 1, role: SchemaRole::Collection, target_type: "table".into(), target_id: table.id },
                NodeTarget { linguistic_node: 2, role: SchemaRole::FilterField, target_type: "field".into(), target_id: field.global_id },
                NodeTarget { linguistic_node: 3, role: SchemaRole::Modifier, target_type: "operation".into(), target_id: op_id },
                NodeTarget { linguistic_node: 4, role: SchemaRole::Modifier, target_type: "operation".into(), target_id: 10 },
            ],
        },
    }
}

fn gen_multi_filter(meta: &SchemaMeta, rng: &mut impl Rng) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let filterable = meta.filterable_fields(table);
    if filterable.len() < 2 { return gen_collection_filter(meta, rng); }

    let n_filters = rng.random_range(2..=3).min(filterable.len());
    let chosen: Vec<&FieldMeta> = filterable.choose_multiple(rng, n_filters).copied().collect();

    let intent_text = ["find", "get", "show"][rng.random_range(0..3)];
    let coll_text = collection_surface(&table.name, rng);

    let mut nodes = vec![
        make_node(0, intent_text, 0, 1, SpanType::Intent),
        make_node(1, &coll_text, 1, 2, SpanType::NounPhrase),
    ];
    let mut edges = vec![
        LinguisticEdge { src: 0, dst: 1, relation: DepRelation::IntentTarget },
    ];
    let mut targets = vec![
        NodeTarget { linguistic_node: 0, role: SchemaRole::None, target_type: String::new(), target_id: 0 },
        NodeTarget { linguistic_node: 1, role: SchemaRole::Collection, target_type: "table".into(), target_id: table.id },
    ];
    let mut candidate_edges = Vec::new();
    candidate_edges.extend(make_noise_candidates(0, meta, rng));
    candidate_edges.extend(make_candidates(1, "table", table.id, meta, rng));

    let mut token_pos = 3;
    for field in &chosen {
        let ops = compatible_filter_ops(&field.field_type);
        let op_id = ops[rng.random_range(0..ops.len())];
        let value = random_value(&field.field_type, rng);
        let op_surface = filter_op_surface(op_id, rng);

        let field_nid = nodes.len();
        nodes.push(make_node(field_nid, &field_surface(&field.local_name, rng), token_pos, token_pos + 1, SpanType::NounPhrase));
        token_pos += 1;

        let comp_nid = nodes.len();
        nodes.push(make_node(comp_nid, &format!("{} {}", op_surface, value), token_pos, token_pos + 2, SpanType::Comparator));
        edges.push(LinguisticEdge { src: comp_nid, dst: field_nid, relation: DepRelation::Comparison });
        token_pos += 3;

        targets.push(NodeTarget {
            linguistic_node: field_nid, role: SchemaRole::FilterField,
            target_type: "field".into(), target_id: field.global_id,
        });
        targets.push(NodeTarget {
            linguistic_node: comp_nid, role: SchemaRole::Modifier,
            target_type: "operation".into(), target_id: op_id,
        });
        candidate_edges.extend(make_candidates(field_nid, "field", field.global_id, meta, rng));
        candidate_edges.extend(make_candidates(comp_nid, "operation", op_id, meta, rng));
    }

    TrainingSample {
        linguistic_graph: LinguisticGraph { raw_query: format!("{} {} where ...", intent_text, coll_text), nodes, edges },
        candidates: CandidateSet { edges: candidate_edges },
        ground_truth: GroundTruth { targets },
    }
}

fn gen_traversal(meta: &SchemaMeta, rng: &mut impl Rng) -> TrainingSample {
    let tables_with_records: Vec<&TableMeta> = meta.tables.iter()
        .filter(|t| !meta.record_fields(t).is_empty())
        .collect();
    if tables_with_records.is_empty() { return gen_collection_only(meta, rng); }

    let table = tables_with_records[rng.random_range(0..tables_with_records.len())];
    let record_fields = meta.record_fields(table);
    let rec_field = record_fields[rng.random_range(0..record_fields.len())];
    let target_table_id = rec_field.record_target.unwrap();
    let target_name = &meta.tables[target_table_id].name;

    let intent_text = ["find", "get", "show"][rng.random_range(0..3)];
    let coll_text = collection_surface(&table.name, rng);

    let nodes = vec![
        make_node(0, intent_text, 0, 1, SpanType::Intent),
        make_node(1, &coll_text, 1, 2, SpanType::NounPhrase),
        make_node(2, target_name, 3, 4, SpanType::NounPhrase),
    ];
    let edges = vec![
        LinguisticEdge { src: 0, dst: 1, relation: DepRelation::IntentTarget },
        LinguisticEdge { src: 1, dst: 2, relation: DepRelation::Possessive },
    ];

    let mut candidate_edges = Vec::new();
    candidate_edges.extend(make_noise_candidates(0, meta, rng));
    candidate_edges.extend(make_candidates(1, "table", table.id, meta, rng));
    candidate_edges.extend(make_candidates(2, "table", target_table_id, meta, rng));

    TrainingSample {
        linguistic_graph: LinguisticGraph {
            raw_query: format!("{} {}'s {}", intent_text, coll_text, target_name),
            nodes, edges,
        },
        candidates: CandidateSet { edges: candidate_edges },
        ground_truth: GroundTruth {
            targets: vec![
                NodeTarget { linguistic_node: 0, role: SchemaRole::None, target_type: String::new(), target_id: 0 },
                NodeTarget { linguistic_node: 1, role: SchemaRole::Collection, target_type: "table".into(), target_id: table.id },
                NodeTarget { linguistic_node: 2, role: SchemaRole::Traversal, target_type: "table".into(), target_id: target_table_id },
            ],
        },
    }
}

fn gen_count(meta: &SchemaMeta, rng: &mut impl Rng) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let coll_text = collection_surface(&table.name, rng);

    let nodes = vec![
        make_node(0, "count", 0, 1, SpanType::Intent),
        make_node(1, &coll_text, 1, 2, SpanType::NounPhrase),
    ];
    let edges = vec![
        LinguisticEdge { src: 0, dst: 1, relation: DepRelation::IntentTarget },
    ];

    let mut candidate_edges = Vec::new();
    // "count" should match the count operation strongly
    candidate_edges.extend(make_candidates(0, "operation", 24, meta, rng)); // count op
    candidate_edges.extend(make_candidates(1, "table", table.id, meta, rng));

    TrainingSample {
        linguistic_graph: LinguisticGraph {
            raw_query: format!("count {}", coll_text),
            nodes, edges,
        },
        candidates: CandidateSet { edges: candidate_edges },
        ground_truth: GroundTruth {
            targets: vec![
                NodeTarget { linguistic_node: 0, role: SchemaRole::Modifier, target_type: "operation".into(), target_id: 24 },
                NodeTarget { linguistic_node: 1, role: SchemaRole::Collection, target_type: "table".into(), target_id: table.id },
            ],
        },
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");
    let raw = Reader::read(&demo_dir.join("schema.surql")).expect("read schema");
    let (schema, _) = Extractor::extract(&raw);
    let schema_graph = SchemaGraph::from_schema(&schema);

    let meta = SchemaMeta::from_schema(&schema, &schema_graph);
    let mut rng = StdRng::seed_from_u64(42);

    println!("schema: {} tables, {} fields",
        meta.tables.len(),
        meta.tables.iter().map(|t| t.fields.len()).sum::<usize>());

    let mut samples = Vec::new();

    for _ in 0..600  { samples.push(gen_collection_only(&meta, &mut rng)); }
    for _ in 0..1000 { samples.push(gen_collection_fields(&meta, &mut rng)); }
    for _ in 0..900  { samples.push(gen_collection_filter(&meta, &mut rng)); }
    for _ in 0..500  { samples.push(gen_collection_fields_filter(&meta, &mut rng)); }
    for _ in 0..500  { samples.push(gen_collection_modifier(&meta, &mut rng)); }
    for _ in 0..400  { samples.push(gen_collection_filter_modifier(&meta, &mut rng)); }
    for _ in 0..400  { samples.push(gen_multi_filter(&meta, &mut rng)); }
    for _ in 0..300  { samples.push(gen_traversal(&meta, &mut rng)); }
    for _ in 0..400  { samples.push(gen_count(&meta, &mut rng)); }

    samples.shuffle(&mut rng);

    let dataset = Dataset { samples };
    dataset.save(&demo_dir.join("dataset.json")).expect("save dataset");
    println!("generated {} samples -> demo/dataset.json", dataset.samples.len());
}
