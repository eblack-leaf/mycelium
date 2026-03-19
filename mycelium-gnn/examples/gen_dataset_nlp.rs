//! Generate training dataset using the real inference pipeline.
//!
//! Runs the full pipeline that the GNN will see at inference:
//!   - NlpModel::parse() with biaffine head → real LinguisticGraph
//!   - CandidateMatcher::match_candidates() → real cross-encoder scores
//!
//! Graph structure comes from the biaffine head (or rule-based fallback),
//! embeddings are real bi-encoder output, candidate scores are real
//! cross-encoder output. Ground truth is derived by matching parsed nodes
//! back to known template targets. Samples where the parser produces
//! unmatchable output are skipped — the GNN only trains on samples where
//! the upstream pipeline gave usable results.
//!
//! Usage:
//!   cargo run --release --example gen_dataset_nlp -p gnn-burn

use std::path::Path;
use rand::prelude::*;
use rand::rngs::StdRng;
use indicatif::{ProgressBar, ProgressStyle};
use gnn_burn::schema::{Reader, Extractor, Schema, FieldType};
use gnn_burn::graph::SchemaGraph;
use gnn_burn::nlp::{NlpModel, NlpConfig, LinguisticGraph, SpanType};
use gnn_burn::candidate_matcher::{CandidateMatcher, CandidateMatcherConfig};
use gnn_burn::reranker::Reranker;
use gnn_burn::operations::all_operations;
use gnn_burn::training::{TrainingSample, GroundTruth, NodeTarget, SchemaRole, Dataset};
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::CompactRecorder;

// =============================================================================
// Schema metadata (same as gen_dataset.rs)
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
// Surface forms (same as gen_dataset.rs)
// =============================================================================

fn collection_surface(name: &str, rng: &mut impl Rng) -> String {
    let forms = [
        name.to_string(),
        format!("all {}", name),
        format!("the {}", name),
        format!("every {}", name),
        name.trim_end_matches('s').to_string(),
        format!("each {}", name.trim_end_matches('s')),
        format!("any {}", name.trim_end_matches('s')),
    ];
    forms[rng.random_range(0..forms.len())].clone()
}

fn field_surface(name: &str, rng: &mut impl Rng) -> String {
    let forms = [
        name.to_string(),
        format!("the {}", name),
        format!("their {}", name),
        format!("{}s", name),
        format!("its {}", name),
    ];
    forms[rng.random_range(0..forms.len())].clone()
}

fn intent_surface(rng: &mut impl Rng) -> &'static str {
    let forms = [
        "show", "find", "get", "list", "display", "fetch", "retrieve",
        "give me", "what are", "return",
    ];
    forms[rng.random_range(0..forms.len())]
}

fn filter_op_surface(op_id: usize, rng: &mut impl Rng) -> String {
    let forms: &[&str] = match op_id {
        11 => &["equals", "is", "equal to", "matching", "=", "same as"],
        12 => &["not", "not equal to", "different from", "isn't", "is not"],
        13 => &["greater than", "more than", "above", "over", "exceeds", "higher than", "is over", "is above", "is greater than", "is more than"],
        14 => &["less than", "under", "below", "fewer than", "is under", "is below", "is less than"],
        15 => &["at least", "minimum", "no less than", "is at least", ">="],
        16 => &["at most", "maximum", "no more than", "is at most", "<="],
        17 => &["like", "matching pattern", "similar to", "matches"],
        18 => &["containing", "includes", "has", "contains", "with"],
        19 => &["starting with", "begins with", "starts with"],
        20 => &["ending with", "ends with"],
        _ => &["equals"],
    };
    forms[rng.random_range(0..forms.len())].to_string()
}

fn count_surface(rng: &mut impl Rng) -> &'static str {
    let forms = ["count", "how many", "total", "number of"];
    forms[rng.random_range(0..forms.len())]
}

fn order_surface(rng: &mut impl Rng) -> &'static str {
    let forms = ["sorted by", "ordered by", "by", "order by", "sort by", "arrange by"];
    forms[rng.random_range(0..forms.len())]
}

fn agg_surface(op_id: usize, rng: &mut impl Rng) -> &'static str {
    match op_id {
        26 => { let f = ["sum of", "total", "sum", "add up"]; f[rng.random_range(0..f.len())] }
        27 => { let f = ["average", "avg", "mean", "average of"]; f[rng.random_range(0..f.len())] }
        28 => { let f = ["minimum", "min", "lowest", "smallest"]; f[rng.random_range(0..f.len())] }
        29 => { let f = ["maximum", "max", "highest", "largest", "biggest"]; f[rng.random_range(0..f.len())] }
        _ => "count",
    }
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
// Intended target: what we expect the parser to find
// =============================================================================

/// An entity we put into the query and expect to find in the parsed graph.
struct IntendedTarget {
    /// The core name to match against parsed node text (lowercase).
    /// e.g. for table "goods" with surface "the goods", key is "goods".
    match_key: String,
    /// What span type we expect the parser to produce.
    expected_span: ExpectedSpan,
    role: SchemaRole,
    target_type: String,
    target_id: usize,
}

enum ExpectedSpan {
    Intent,
    NounPhrase,
    Comparator,
    Quantifier,
}

/// A query we generated along with the ground truth we expect.
struct QueryTemplate {
    raw_query: String,
    targets: Vec<IntendedTarget>,
}

// =============================================================================
// Node matching: map parsed LinguisticNodes to IntendedTargets
// =============================================================================

/// Try to match each IntendedTarget to a parsed node. Returns None if any
/// target can't be matched (ambiguous or missing).
fn match_targets(
    ling_graph: &LinguisticGraph,
    targets: &[IntendedTarget],
) -> Option<Vec<NodeTarget>> {
    let mut result = Vec::new();
    let mut claimed: Vec<bool> = vec![false; ling_graph.nodes.len()];

    for target in targets {
        let candidate = find_matching_node(ling_graph, target, &claimed);
        match candidate {
            Some(node_id) => {
                claimed[node_id] = true;
                result.push(NodeTarget {
                    linguistic_node: node_id,
                    role: target.role.clone(),
                    target_type: target.target_type.clone(),
                    target_id: target.target_id,
                });
            }
            None => return None, // couldn't match this target — skip sample
        }
    }

    // Any unclaimed nodes get SchemaRole::None
    for (i, node) in ling_graph.nodes.iter().enumerate() {
        if !claimed[i] {
            result.push(NodeTarget {
                linguistic_node: node.id,
                role: SchemaRole::None,
                target_type: String::new(),
                target_id: 0,
            });
        }
    }

    Some(result)
}

fn find_matching_node(
    ling_graph: &LinguisticGraph,
    target: &IntendedTarget,
    claimed: &[bool],
) -> Option<usize> {
    let key = target.match_key.to_lowercase();

    for node in &ling_graph.nodes {
        if claimed[node.id] { continue; }

        // Check span type compatibility
        let type_ok = match target.expected_span {
            ExpectedSpan::Intent => node.span_type == SpanType::Intent,
            ExpectedSpan::NounPhrase => node.span_type == SpanType::NounPhrase,
            ExpectedSpan::Comparator => node.span_type == SpanType::Comparator,
            ExpectedSpan::Quantifier => node.span_type == SpanType::Quantifier,
        };
        if !type_ok { continue; }

        let node_text = node.text.to_lowercase();

        // Exact match or containment
        if node_text == key
            || node_text.contains(&key)
            || key.contains(&node_text)
        {
            return Some(node.id);
        }

        // Handle plural stripping: "goods" matches "good", "timestamps" matches "timestamp"
        let key_stripped = key.trim_end_matches('s');
        let node_stripped = node_text.trim_end_matches('s');
        if node_stripped == key_stripped
            || node_stripped.contains(key_stripped)
            || key_stripped.contains(node_stripped)
        {
            return Some(node.id);
        }
    }

    None
}

// =============================================================================
// Query template builders
// =============================================================================

fn build_collection_only(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);

    QueryTemplate {
        raw_query: format!("{} {}", intent, coll),
        targets: vec![
            IntendedTarget {
                match_key: intent.split_whitespace().next().unwrap().to_string(),
                expected_span: ExpectedSpan::Intent,
                role: SchemaRole::None,
                target_type: String::new(),
                target_id: 0,
            },
            IntendedTarget {
                match_key: table.name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Collection,
                target_type: "table".into(),
                target_id: table.id,
            },
        ],
    }
}

fn build_collection_fields(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let available = meta.non_record_fields(table);
    if available.is_empty() { return build_collection_only(meta, rng); }

    let n_fields = rng.random_range(1..=2).min(available.len());
    let chosen: Vec<&FieldMeta> = available.choose_multiple(rng, n_fields).copied().collect();

    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);

    let mut query_parts = vec![intent.to_string(), format!("{}'s", coll)];
    let mut targets = vec![
        IntendedTarget {
            match_key: intent.split_whitespace().next().unwrap().to_string(),
            expected_span: ExpectedSpan::Intent,
            role: SchemaRole::None,
            target_type: String::new(),
            target_id: 0,
        },
        IntendedTarget {
            match_key: table.name.clone(),
            expected_span: ExpectedSpan::NounPhrase,
            role: SchemaRole::Collection,
            target_type: "table".into(),
            target_id: table.id,
        },
    ];

    for (i, field) in chosen.iter().enumerate() {
        let f_surface = field_surface(&field.local_name, rng);
        if i > 0 { query_parts.push("and".to_string()); }
        query_parts.push(f_surface);
        targets.push(IntendedTarget {
            match_key: field.local_name.clone(),
            expected_span: ExpectedSpan::NounPhrase,
            role: SchemaRole::Field,
            target_type: "field".into(),
            target_id: field.global_id,
        });
    }

    QueryTemplate {
        raw_query: query_parts.join(" "),
        targets,
    }
}

fn build_collection_filter(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let filterable = meta.filterable_fields(table);
    if filterable.is_empty() { return build_collection_only(meta, rng); }

    let field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);

    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);
    let f_surface = field_surface(&field.local_name, rng);

    QueryTemplate {
        raw_query: format!("{} {} where {} {} {}", intent, coll, f_surface, op_surface, value),
        targets: vec![
            IntendedTarget {
                match_key: intent.split_whitespace().next().unwrap().to_string(),
                expected_span: ExpectedSpan::Intent,
                role: SchemaRole::None,
                target_type: String::new(),
                target_id: 0,
            },
            IntendedTarget {
                match_key: table.name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Collection,
                target_type: "table".into(),
                target_id: table.id,
            },
            IntendedTarget {
                match_key: field.local_name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::FilterField,
                target_type: "field".into(),
                target_id: field.global_id,
            },
            IntendedTarget {
                match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
                expected_span: ExpectedSpan::Comparator,
                role: SchemaRole::Modifier,
                target_type: "operation".into(),
                target_id: op_id,
            },
        ],
    }
}

fn build_collection_fields_filter(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let available = meta.non_record_fields(table);
    let filterable = meta.filterable_fields(table);
    if available.is_empty() || filterable.is_empty() { return build_collection_only(meta, rng); }

    let sel_field = available[rng.random_range(0..available.len())];
    let filt_field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&filt_field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&filt_field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);

    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);

    QueryTemplate {
        raw_query: format!("{} {}'s {} where {} {} {}",
            intent, coll, field_surface(&sel_field.local_name, rng),
            field_surface(&filt_field.local_name, rng), op_surface, value),
        targets: vec![
            IntendedTarget {
                match_key: intent.split_whitespace().next().unwrap().to_string(),
                expected_span: ExpectedSpan::Intent,
                role: SchemaRole::None,
                target_type: String::new(),
                target_id: 0,
            },
            IntendedTarget {
                match_key: table.name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Collection,
                target_type: "table".into(),
                target_id: table.id,
            },
            IntendedTarget {
                match_key: sel_field.local_name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Field,
                target_type: "field".into(),
                target_id: sel_field.global_id,
            },
            IntendedTarget {
                match_key: filt_field.local_name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::FilterField,
                target_type: "field".into(),
                target_id: filt_field.global_id,
            },
            IntendedTarget {
                match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
                expected_span: ExpectedSpan::Comparator,
                role: SchemaRole::Modifier,
                target_type: "operation".into(),
                target_id: op_id,
            },
        ],
    }
}

fn build_collection_modifier(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);
    let limit_val = rng.random_range(1..50);

    let mut targets = vec![
        IntendedTarget {
            match_key: intent.split_whitespace().next().unwrap().to_string(),
            expected_span: ExpectedSpan::Intent,
            role: SchemaRole::None,
            target_type: String::new(),
            target_id: 0,
        },
        IntendedTarget {
            match_key: table.name.clone(),
            expected_span: ExpectedSpan::NounPhrase,
            role: SchemaRole::Collection,
            target_type: "table".into(),
            target_id: table.id,
        },
        IntendedTarget {
            match_key: "first".to_string(),
            expected_span: ExpectedSpan::Quantifier,
            role: SchemaRole::Modifier,
            target_type: "operation".into(),
            target_id: 10, // LIMIT
        },
    ];

    let mut query = format!("{} {}, first {}", intent, coll, limit_val);

    // Optionally add ORDER_BY field
    if rng.random_bool(0.5) {
        let orderable = meta.orderable_fields(table);
        if let Some(field) = orderable.choose(rng) {
            query = format!("{} {} by {}, first {}", intent, coll,
                field_surface(&field.local_name, rng), limit_val);
            targets.push(IntendedTarget {
                match_key: field.local_name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Field,
                target_type: "field".into(),
                target_id: field.global_id,
            });
        }
    }

    QueryTemplate { raw_query: query, targets }
}

fn build_collection_filter_modifier(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let filterable = meta.filterable_fields(table);
    if filterable.is_empty() { return build_collection_modifier(meta, rng); }

    let field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);
    let limit_val = rng.random_range(1..50);

    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);

    QueryTemplate {
        raw_query: format!("{} {} where {} {} {}, first {}",
            intent, coll, field_surface(&field.local_name, rng), op_surface, value, limit_val),
        targets: vec![
            IntendedTarget {
                match_key: intent.split_whitespace().next().unwrap().to_string(),
                expected_span: ExpectedSpan::Intent,
                role: SchemaRole::None,
                target_type: String::new(),
                target_id: 0,
            },
            IntendedTarget {
                match_key: table.name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Collection,
                target_type: "table".into(),
                target_id: table.id,
            },
            IntendedTarget {
                match_key: field.local_name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::FilterField,
                target_type: "field".into(),
                target_id: field.global_id,
            },
            IntendedTarget {
                match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
                expected_span: ExpectedSpan::Comparator,
                role: SchemaRole::Modifier,
                target_type: "operation".into(),
                target_id: op_id,
            },
            IntendedTarget {
                match_key: "first".to_string(),
                expected_span: ExpectedSpan::Quantifier,
                role: SchemaRole::Modifier,
                target_type: "operation".into(),
                target_id: 10, // LIMIT
            },
        ],
    }
}

fn build_multi_filter(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let filterable = meta.filterable_fields(table);
    if filterable.len() < 2 { return build_collection_filter(meta, rng); }

    let n_filters = rng.random_range(2..=3).min(filterable.len());
    let chosen: Vec<&FieldMeta> = filterable.choose_multiple(rng, n_filters).copied().collect();

    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);

    let mut targets = vec![
        IntendedTarget {
            match_key: intent.split_whitespace().next().unwrap().to_string(),
            expected_span: ExpectedSpan::Intent,
            role: SchemaRole::None,
            target_type: String::new(),
            target_id: 0,
        },
        IntendedTarget {
            match_key: table.name.clone(),
            expected_span: ExpectedSpan::NounPhrase,
            role: SchemaRole::Collection,
            target_type: "table".into(),
            target_id: table.id,
        },
    ];

    let mut where_parts = Vec::new();
    for field in &chosen {
        let ops = compatible_filter_ops(&field.field_type);
        let op_id = ops[rng.random_range(0..ops.len())];
        let value = random_value(&field.field_type, rng);
        let op_surface = filter_op_surface(op_id, rng);

        where_parts.push(format!("{} {} {}",
            field_surface(&field.local_name, rng), op_surface, value));

        targets.push(IntendedTarget {
            match_key: field.local_name.clone(),
            expected_span: ExpectedSpan::NounPhrase,
            role: SchemaRole::FilterField,
            target_type: "field".into(),
            target_id: field.global_id,
        });
        targets.push(IntendedTarget {
            match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
            expected_span: ExpectedSpan::Comparator,
            role: SchemaRole::Modifier,
            target_type: "operation".into(),
            target_id: op_id,
        });
    }

    QueryTemplate {
        raw_query: format!("{} {} where {}", intent, coll, where_parts.join(" and ")),
        targets,
    }
}

fn build_traversal(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let tables_with_records: Vec<&TableMeta> = meta.tables.iter()
        .filter(|t| !meta.record_fields(t).is_empty())
        .collect();
    if tables_with_records.is_empty() { return build_collection_only(meta, rng); }

    let table = tables_with_records[rng.random_range(0..tables_with_records.len())];
    let record_fields = meta.record_fields(table);
    let rec_field = record_fields[rng.random_range(0..record_fields.len())];
    let target_table_id = rec_field.record_target.unwrap();
    let target_name = &meta.tables[target_table_id].name;

    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);

    QueryTemplate {
        raw_query: format!("{} {}'s {}", intent, coll, target_name),
        targets: vec![
            IntendedTarget {
                match_key: intent.split_whitespace().next().unwrap().to_string(),
                expected_span: ExpectedSpan::Intent,
                role: SchemaRole::None,
                target_type: String::new(),
                target_id: 0,
            },
            IntendedTarget {
                match_key: table.name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Collection,
                target_type: "table".into(),
                target_id: table.id,
            },
            IntendedTarget {
                match_key: target_name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Traversal,
                target_type: "table".into(),
                target_id: target_table_id,
            },
        ],
    }
}

fn build_count(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let coll = collection_surface(&table.name, rng);
    let surface = count_surface(rng);

    QueryTemplate {
        raw_query: format!("{} {}", surface, coll),
        targets: vec![
            IntendedTarget {
                match_key: surface.split_whitespace().next().unwrap().to_string(),
                expected_span: ExpectedSpan::Intent,
                role: SchemaRole::Modifier,
                target_type: "operation".into(),
                target_id: 25, // count aggregate
            },
            IntendedTarget {
                match_key: table.name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Collection,
                target_type: "table".into(),
                target_id: table.id,
            },
        ],
    }
}

fn build_count_filter(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let filterable = meta.filterable_fields(table);
    if filterable.is_empty() { return build_count(meta, rng); }

    let field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);
    let coll = collection_surface(&table.name, rng);
    let surface = count_surface(rng);

    QueryTemplate {
        raw_query: format!("{} {} where {} {} {}", surface, coll,
            field_surface(&field.local_name, rng), op_surface, value),
        targets: vec![
            IntendedTarget {
                match_key: surface.split_whitespace().next().unwrap().to_string(),
                expected_span: ExpectedSpan::Intent,
                role: SchemaRole::Modifier,
                target_type: "operation".into(),
                target_id: 25,
            },
            IntendedTarget {
                match_key: table.name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Collection,
                target_type: "table".into(),
                target_id: table.id,
            },
            IntendedTarget {
                match_key: field.local_name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::FilterField,
                target_type: "field".into(),
                target_id: field.global_id,
            },
            IntendedTarget {
                match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
                expected_span: ExpectedSpan::Comparator,
                role: SchemaRole::Modifier,
                target_type: "operation".into(),
                target_id: op_id,
            },
        ],
    }
}

fn build_aggregate(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let numeric_fields: Vec<&FieldMeta> = table.fields.iter()
        .filter(|f| matches!(f.field_type,
            FieldType::String | FieldType::Int | FieldType::Float |
            FieldType::Decimal | FieldType::Number))
        .filter(|f| f.record_target.is_none())
        .collect();
    if numeric_fields.is_empty() { return build_count(meta, rng); }

    let field = numeric_fields[rng.random_range(0..numeric_fields.len())];
    let agg_ops = [26usize, 27, 28, 29];
    let op_id = agg_ops[rng.random_range(0..agg_ops.len())];
    let surface = agg_surface(op_id, rng);
    let coll = collection_surface(&table.name, rng);

    let connector = ["of", "for", "across"][rng.random_range(0..3)];
    let query = if rng.random_bool(0.5) {
        format!("{} {} {} {}", surface, field_surface(&field.local_name, rng), connector, coll)
    } else {
        format!("{} {}'s {}", surface, coll, field_surface(&field.local_name, rng))
    };

    QueryTemplate {
        raw_query: query,
        targets: vec![
            IntendedTarget {
                match_key: surface.split_whitespace().next().unwrap().to_string(),
                expected_span: ExpectedSpan::Intent,
                role: SchemaRole::Modifier,
                target_type: "operation".into(),
                target_id: op_id,
            },
            IntendedTarget {
                match_key: table.name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Collection,
                target_type: "table".into(),
                target_id: table.id,
            },
            IntendedTarget {
                match_key: field.local_name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Field,
                target_type: "field".into(),
                target_id: field.global_id,
            },
        ],
    }
}

fn build_order_by(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let orderable = meta.orderable_fields(table);
    if orderable.is_empty() { return build_collection_only(meta, rng); }

    let field = orderable[rng.random_range(0..orderable.len())];
    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);
    let order = ["sorted by", "ordered by", "by", "order by", "sort by"][rng.random_range(0..5)];

    let direction = if rng.random_bool(0.3) {
        [" ascending", " descending", " asc", " desc"][rng.random_range(0..4)]
    } else { "" };

    QueryTemplate {
        raw_query: format!("{} {} {} {}{}", intent, coll, order, field_surface(&field.local_name, rng), direction),
        targets: vec![
            IntendedTarget {
                match_key: intent.split_whitespace().next().unwrap().to_string(),
                expected_span: ExpectedSpan::Intent,
                role: SchemaRole::None,
                target_type: String::new(),
                target_id: 0,
            },
            IntendedTarget {
                match_key: table.name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Collection,
                target_type: "table".into(),
                target_id: table.id,
            },
            IntendedTarget {
                match_key: order.split_whitespace().next().unwrap().to_string(),
                expected_span: ExpectedSpan::Comparator,
                role: SchemaRole::Modifier,
                target_type: "operation".into(),
                target_id: 6, // ORDER_BY
            },
            IntendedTarget {
                match_key: field.local_name.clone(),
                expected_span: ExpectedSpan::NounPhrase,
                role: SchemaRole::Field,
                target_type: "field".into(),
                target_id: field.global_id,
            },
        ],
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");

    // --- Load schema ---
    let raw = Reader::read(&demo_dir.join("schema.surql")).expect("read schema");
    let (schema, _) = Extractor::extract(&raw);
    let schema_graph = SchemaGraph::from_schema(&schema);
    let operations = all_operations();
    let meta = SchemaMeta::from_schema(&schema, &schema_graph);

    println!("schema: {} tables, {} fields",
        meta.tables.len(),
        meta.tables.iter().map(|t| t.fields.len()).sum::<usize>());

    // --- Load NLP models ---
    println!("Loading NLP models...");
    // Use biaffine head if available — this is the real inference path
    let biaffine_path = demo_dir.join("biaffine_model");
    let biaffine_model_path = if biaffine_path.with_extension("mpk").exists() {
        println!("Using biaffine head for parsing");
        Some(biaffine_path.to_string_lossy().into_owned())
    } else {
        println!("No biaffine model found — falling back to rule-based parser");
        None
    };

    // Use n-gram model if available
    let ngram_path = demo_dir.join("ngram_model");
    let ngram_model_path = if ngram_path.with_extension("mpk").exists() {
        println!("Using n-gram cross-attention model");
        Some(ngram_path.to_string_lossy().into_owned())
    } else {
        None
    };

    let mut nlp = NlpModel::load(&NlpConfig {
        model_path: model_dir.join("model.onnx").to_string_lossy().into(),
        tokenizer_path: model_dir.join("tokenizer.json").to_string_lossy().into(),
        cross_model_path: model_dir.join("cross-encoder.onnx").to_string_lossy().into(),
        cross_tokenizer_path: model_dir.join("cross-tokenizer.json").to_string_lossy().into(),
        biaffine_model_path,
        ngram_model_path,
    }).expect("load NLP models — run fetch_models.sh first");

    // Initialize n-gram concept map from schema
    nlp.init_ngram(&schema_graph, &operations);

    // --- Build candidate matcher ---
    let mut matcher = CandidateMatcher::new(
        &schema_graph,
        &operations,
        CandidateMatcherConfig { top_k: 5, min_score: 0.0 },
    );

    // Load re-ranker if available
    let reranker_path = demo_dir.join("reranker_model");
    if reranker_path.with_extension("mpk").exists() {
        println!("Loading trained re-ranker...");
        let device: <NdArray as burn::tensor::backend::Backend>::Device = Default::default();
        let reranker = Reranker::<NdArray>::new(&device)
            .load_file(&reranker_path, &CompactRecorder::new(), &device)
            .expect("load reranker model");
        matcher.load_reranker(reranker, &nlp);
        println!("Using trained re-ranker for candidate matching");
    } else {
        println!("No re-ranker found — using pretrained cross-encoder");
    }

    let mut rng = StdRng::seed_from_u64(42);

    // --- Define sample counts per query type ---
    // Smaller than synthetic because cross-encoder is slow (~250 calls per query)
    let query_specs: Vec<(&str, usize, fn(&SchemaMeta, &mut StdRng) -> QueryTemplate)> = vec![
        ("collection_only",         200, build_collection_only),
        ("collection_fields",       300, build_collection_fields),
        ("collection_filter",       300, build_collection_filter),
        ("collection_fields_filter", 200, build_collection_fields_filter),
        ("collection_modifier",     200, build_collection_modifier),
        ("filter_modifier",         150, build_collection_filter_modifier),
        ("multi_filter",            150, build_multi_filter),
        ("traversal",               100, build_traversal),
        ("count",                   150, build_count),
        ("count_filter",            100, build_count_filter),
        ("aggregate",               150, build_aggregate),
        ("order_by",                150, build_order_by),
    ];

    let total_attempts: usize = query_specs.iter().map(|(_, n, _)| *n).sum();
    println!("Generating {} query templates...", total_attempts);

    let bar = ProgressBar::new(total_attempts as u64);
    bar.set_style(ProgressStyle::default_bar()
        .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("##-"));

    let mut samples = Vec::new();
    let mut skipped = 0usize;

    for (name, count, builder) in &query_specs {
        bar.set_message(format!("{:<25}", *name));
        for _ in 0..*count {
            let template = builder(&meta, &mut rng);

            // Stage 1: biaffine/rule-based parse
            // Stage 2: n-gram concept scoring if available, else cross-encoder/re-ranker
            let (ling_graph, ngram_candidates) = nlp.parse_with_candidates(&template.raw_query);
            let candidates = ngram_candidates.unwrap_or_else(|| {
                matcher.match_candidates(&nlp, &ling_graph)
            });

            // Match parsed nodes to intended targets
            match match_targets(&ling_graph, &template.targets) {
                Some(node_targets) => {
                    samples.push(TrainingSample {
                        linguistic_graph: ling_graph,
                        candidates,
                        ground_truth: GroundTruth { targets: node_targets },
                    });
                }
                None => {
                    skipped += 1;
                }
            }

            bar.inc(1);
        }
    }

    bar.finish_with_message("done");

    samples.shuffle(&mut rng);

    let dataset = Dataset { samples };
    dataset.save(&demo_dir.join("dataset_nlp.json")).expect("save dataset");

    println!("generated {} samples ({} skipped) -> demo/dataset_nlp.json",
        dataset.samples.len(), skipped);
}
