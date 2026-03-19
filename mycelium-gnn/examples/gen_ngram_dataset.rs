//! Generate training dataset for the n-gram cross-attention model.
//!
//! Reuses the same template builders from gen_dataset_nlp.rs to generate
//! queries with known ground-truth. For each query:
//!   1. Tokenize → subword-to-word alignment
//!   2. Find word-level spans for each IntendedTarget
//!   3. Convert (schema_type, schema_id) → concept_idx via ConceptMap
//!   4. Record NgramSpanLabel per target
//!
//! Only needs the MiniLM tokenizer (no ONNX inference).
//!
//! Usage:
//!   cargo run --release --example gen_ngram_dataset -p gnn-burn

use std::path::Path;
use rand::prelude::*;
use rand::rngs::StdRng;
use indicatif::{ProgressBar, ProgressStyle};
use gnn_burn::schema::{Reader, Extractor, Schema, FieldType};
use gnn_burn::graph::SchemaGraph;
use gnn_burn::operations::all_operations;
use gnn_burn::ngram_data::{NgramSample, NgramSpanLabel, NgramDataset, ConceptMap};

// =============================================================================
// Schema metadata (same structures as gen_dataset_nlp.rs)
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
// Surface form generators (same as gen_dataset_nlp.rs)
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
        15 => &["at least", "minimum", "no less than", "is at least", ">=" ],
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

fn group_surface(rng: &mut impl Rng) -> &'static str {
    let forms = ["grouped by", "group by", "per", "for each", "by"];
    forms[rng.random_range(0..forms.len())]
}

fn agg_surface(op_id: usize, rng: &mut impl Rng) -> &'static str {
    match op_id {
        26 => { // sum
            let forms = ["sum of", "total", "sum", "add up"];
            forms[rng.random_range(0..forms.len())]
        }
        27 => { // avg
            let forms = ["average", "avg", "mean", "average of"];
            forms[rng.random_range(0..forms.len())]
        }
        28 => { // min
            let forms = ["minimum", "min", "lowest", "smallest"];
            forms[rng.random_range(0..forms.len())]
        }
        29 => { // max
            let forms = ["maximum", "max", "highest", "largest", "biggest"];
            forms[rng.random_range(0..forms.len())]
        }
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
// Intended target for n-gram labeling
// =============================================================================

struct IntendedTarget {
    /// The core word(s) to find in the query.
    match_key: String,
    span_type: usize,       // 0=NP, 1=Quant, 2=Comp, 3=Intent
    schema_type: String,    // "table", "field", "operation", ""
    schema_id: usize,
}

struct QueryTemplate {
    raw_query: String,
    targets: Vec<IntendedTarget>,
}

/// Find word-level span for a match_key in the whitespace-split query.
/// Returns (start_word, end_word_exclusive) or None.
fn find_word_span(words: &[&str], key: &str) -> Option<(usize, usize)> {
    let key_lower = key.to_lowercase();
    let key_words: Vec<&str> = key_lower.split_whitespace().collect();

    // Try exact multi-word match first
    if key_words.len() > 1 {
        'outer: for i in 0..=words.len().saturating_sub(key_words.len()) {
            for (j, kw) in key_words.iter().enumerate() {
                if !words[i + j].to_lowercase().contains(kw) {
                    continue 'outer;
                }
            }
            return Some((i, i + key_words.len()));
        }
    }

    // Single word: find first word containing the key
    let key_lower_str = key_lower.as_str();
    let key_single = key_words.last().unwrap_or(&key_lower_str);
    for (i, word) in words.iter().enumerate() {
        let w = word.to_lowercase();
        let w_clean = w.trim_end_matches("'s").trim_end_matches("'").trim_end_matches(',');
        if w_clean == *key_single
            || w_clean.contains(&**key_single)
            || key_single.contains(w_clean)
        {
            return Some((i, i + 1));
        }
        // Plural stripping
        let w_stripped = w_clean.trim_end_matches('s');
        let k_stripped = key_single.trim_end_matches('s');
        if w_stripped == k_stripped
            || w_stripped.contains(k_stripped)
            || k_stripped.contains(w_stripped)
        {
            return Some((i, i + 1));
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
                span_type: 3, // Intent
                schema_type: "operation".into(),
                schema_id: 0, // SELECT
            },
            IntendedTarget {
                match_key: table.name.clone(),
                span_type: 0, // NP
                schema_type: "table".into(),
                schema_id: table.id,
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
            span_type: 3,
            schema_type: "operation".into(),
            schema_id: 0, // SELECT
        },
        IntendedTarget {
            match_key: table.name.clone(),
            span_type: 0,
            schema_type: "table".into(),
            schema_id: table.id,
        },
    ];

    for (i, field) in chosen.iter().enumerate() {
        let f_surface = field_surface(&field.local_name, rng);
        if i > 0 { query_parts.push("and".to_string()); }
        query_parts.push(f_surface);
        targets.push(IntendedTarget {
            match_key: field.local_name.clone(),
            span_type: 0,
            schema_type: "field".into(),
            schema_id: field.global_id,
        });
    }

    QueryTemplate { raw_query: query_parts.join(" "), targets }
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
            IntendedTarget { match_key: intent.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 0 },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
            IntendedTarget { match_key: field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: field.global_id },
            IntendedTarget {
                match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
                span_type: 2, // Comparator
                schema_type: "operation".into(),
                schema_id: op_id,
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
            IntendedTarget { match_key: intent.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 0 },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
            IntendedTarget { match_key: sel_field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: sel_field.global_id },
            IntendedTarget { match_key: filt_field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: filt_field.global_id },
            IntendedTarget {
                match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
                span_type: 2,
                schema_type: "operation".into(),
                schema_id: op_id,
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
        IntendedTarget { match_key: intent.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 0 },
        IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
        IntendedTarget { match_key: "first".to_string(), span_type: 1, schema_type: "operation".into(), schema_id: 10 }, // LIMIT
    ];

    let mut query = format!("{} {}, first {}", intent, coll, limit_val);

    if rng.random_bool(0.5) {
        let orderable = meta.orderable_fields(table);
        if let Some(field) = orderable.choose(rng) {
            query = format!("{} {} by {}, first {}", intent, coll,
                field_surface(&field.local_name, rng), limit_val);
            targets.push(IntendedTarget {
                match_key: field.local_name.clone(),
                span_type: 0,
                schema_type: "field".into(),
                schema_id: field.global_id,
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
            IntendedTarget { match_key: intent.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 0 },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
            IntendedTarget { match_key: field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: field.global_id },
            IntendedTarget {
                match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
                span_type: 2,
                schema_type: "operation".into(),
                schema_id: op_id,
            },
            IntendedTarget { match_key: "first".to_string(), span_type: 1, schema_type: "operation".into(), schema_id: 10 },
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
        IntendedTarget { match_key: intent.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 0 },
        IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
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
            span_type: 0,
            schema_type: "field".into(),
            schema_id: field.global_id,
        });
        targets.push(IntendedTarget {
            match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
            span_type: 2,
            schema_type: "operation".into(),
            schema_id: op_id,
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
            IntendedTarget { match_key: intent.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 0 },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
            IntendedTarget { match_key: target_name.clone(), span_type: 0, schema_type: "table".into(), schema_id: target_table_id },
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
            IntendedTarget { match_key: surface.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 25 },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
        ],
    }
}

// --- NEW: count with filter ---
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
            IntendedTarget { match_key: surface.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 25 },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
            IntendedTarget { match_key: field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: field.global_id },
            IntendedTarget {
                match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
                span_type: 2,
                schema_type: "operation".into(),
                schema_id: op_id,
            },
        ],
    }
}

// --- NEW: aggregate (sum/avg/min/max) ---
fn build_aggregate(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let numeric_fields: Vec<&FieldMeta> = table.fields.iter()
        .filter(|f| matches!(f.field_type,
            FieldType::Int | FieldType::Float | FieldType::Decimal | FieldType::Number))
        .collect();
    if numeric_fields.is_empty() { return build_count(meta, rng); }

    let field = numeric_fields[rng.random_range(0..numeric_fields.len())];
    let agg_ops = [26usize, 27, 28, 29]; // sum, avg, min, max
    let op_id = agg_ops[rng.random_range(0..agg_ops.len())];
    let surface = agg_surface(op_id, rng);
    let coll = collection_surface(&table.name, rng);

    // "average rating of posts", "sum of users' age", "maximum price for products"
    let connector = ["of", "for", "across"][rng.random_range(0..3)];
    let query = if rng.random_bool(0.5) {
        format!("{} {} {} {}", surface, field_surface(&field.local_name, rng), connector, coll)
    } else {
        format!("{} {}'s {}", surface, coll, field_surface(&field.local_name, rng))
    };

    QueryTemplate {
        raw_query: query,
        targets: vec![
            IntendedTarget { match_key: surface.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: op_id },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
            IntendedTarget { match_key: field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: field.global_id },
        ],
    }
}

// --- NEW: ORDER_BY ---
fn build_order_by(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let orderable = meta.orderable_fields(table);
    if orderable.is_empty() { return build_collection_only(meta, rng); }

    let field = orderable[rng.random_range(0..orderable.len())];
    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);
    let order = order_surface(rng);

    // Optionally add direction
    let direction = if rng.random_bool(0.3) {
        [" ascending", " descending", " asc", " desc"][rng.random_range(0..4)]
    } else { "" };

    QueryTemplate {
        raw_query: format!("{} {} {} {}{}", intent, coll, order, field_surface(&field.local_name, rng), direction),
        targets: vec![
            IntendedTarget { match_key: intent.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 0 },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
            IntendedTarget { match_key: order.split_whitespace().next().unwrap().to_string(), span_type: 2, schema_type: "operation".into(), schema_id: 6 }, // ORDER_BY
            IntendedTarget { match_key: field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: field.global_id },
        ],
    }
}

// --- NEW: GROUP_BY ---
fn build_group_by(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let groupable: Vec<&FieldMeta> = table.fields.iter()
        .filter(|f| matches!(f.field_type,
            FieldType::String | FieldType::Int | FieldType::Bool))
        .collect();
    if groupable.is_empty() { return build_collection_only(meta, rng); }

    let field = groupable[rng.random_range(0..groupable.len())];
    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);
    let group = group_surface(rng);

    QueryTemplate {
        raw_query: format!("{} {} {} {}", intent, coll, group, field_surface(&field.local_name, rng)),
        targets: vec![
            IntendedTarget { match_key: intent.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 0 },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
            IntendedTarget { match_key: group.split_whitespace().next().unwrap().to_string(), span_type: 2, schema_type: "operation".into(), schema_id: 7 }, // GROUP_BY
            IntendedTarget { match_key: field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: field.global_id },
        ],
    }
}

// --- NEW: aggregate with filter ---
fn build_aggregate_filter(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let numeric_fields: Vec<&FieldMeta> = table.fields.iter()
        .filter(|f| matches!(f.field_type,
            FieldType::Int | FieldType::Float | FieldType::Decimal | FieldType::Number))
        .collect();
    let filterable = meta.filterable_fields(table);
    if numeric_fields.is_empty() || filterable.is_empty() { return build_aggregate(meta, rng); }

    let agg_field = numeric_fields[rng.random_range(0..numeric_fields.len())];
    let filt_field = filterable[rng.random_range(0..filterable.len())];
    let agg_ops = [26usize, 27, 28, 29];
    let agg_id = agg_ops[rng.random_range(0..agg_ops.len())];
    let agg = agg_surface(agg_id, rng);
    let ops = compatible_filter_ops(&filt_field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&filt_field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);
    let coll = collection_surface(&table.name, rng);

    QueryTemplate {
        raw_query: format!("{} {}'s {} where {} {} {}", agg, coll,
            field_surface(&agg_field.local_name, rng),
            field_surface(&filt_field.local_name, rng), op_surface, value),
        targets: vec![
            IntendedTarget { match_key: agg.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: agg_id },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
            IntendedTarget { match_key: agg_field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: agg_field.global_id },
            IntendedTarget { match_key: filt_field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: filt_field.global_id },
            IntendedTarget {
                match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
                span_type: 2,
                schema_type: "operation".into(),
                schema_id: op_id,
            },
        ],
    }
}

// --- NEW: ORDER_BY with filter ---
fn build_order_filter(meta: &SchemaMeta, rng: &mut impl Rng) -> QueryTemplate {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let orderable = meta.orderable_fields(table);
    let filterable = meta.filterable_fields(table);
    if orderable.is_empty() || filterable.is_empty() { return build_order_by(meta, rng); }

    let order_field = orderable[rng.random_range(0..orderable.len())];
    let filt_field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&filt_field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&filt_field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);
    let intent = intent_surface(rng);
    let coll = collection_surface(&table.name, rng);
    let order = order_surface(rng);

    QueryTemplate {
        raw_query: format!("{} {} where {} {} {} {} {}",
            intent, coll,
            field_surface(&filt_field.local_name, rng), op_surface, value,
            order, field_surface(&order_field.local_name, rng)),
        targets: vec![
            IntendedTarget { match_key: intent.split_whitespace().next().unwrap().to_string(), span_type: 3, schema_type: "operation".into(), schema_id: 0 },
            IntendedTarget { match_key: table.name.clone(), span_type: 0, schema_type: "table".into(), schema_id: table.id },
            IntendedTarget { match_key: filt_field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: filt_field.global_id },
            IntendedTarget {
                match_key: op_surface.split_whitespace().next().unwrap_or(&op_surface).to_string(),
                span_type: 2,
                schema_type: "operation".into(),
                schema_id: op_id,
            },
            IntendedTarget { match_key: order.split_whitespace().next().unwrap().to_string(), span_type: 2, schema_type: "operation".into(), schema_id: 6 },
            IntendedTarget { match_key: order_field.local_name.clone(), span_type: 0, schema_type: "field".into(), schema_id: order_field.global_id },
        ],
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");

    // --- Load schema ---
    let raw = Reader::read(&demo_dir.join("schema.surql")).expect("read schema");
    let (schema, _) = Extractor::extract(&raw);
    let schema_graph = SchemaGraph::from_schema(&schema);
    let operations = all_operations();
    let meta = SchemaMeta::from_schema(&schema, &schema_graph);

    println!("schema: {} tables, {} fields, {} operations",
        meta.tables.len(),
        meta.tables.iter().map(|t| t.fields.len()).sum::<usize>(),
        operations.len());

    // --- Build concept map ---
    let table_names: Vec<String> = schema_graph.table_nodes.iter()
        .map(|n| n.name.clone()).collect();
    let field_names: Vec<String> = schema_graph.field_nodes.iter()
        .map(|n| n.name.splitn(2, '.').nth(1).unwrap_or(&n.name).to_string()).collect();
    let op_names: Vec<String> = operations.iter()
        .map(|op| op.name.clone()).collect();
    let concept_map = ConceptMap::new(&table_names, &field_names, &op_names);
    println!("concept map: {} total ({} tables, {} fields, {} ops)",
        concept_map.total(), concept_map.n_tables, concept_map.n_fields, concept_map.n_ops);

    // No ONNX inference needed — word spans found by whitespace splitting

    let mut rng = StdRng::seed_from_u64(42);

    let query_specs: Vec<(&str, usize, fn(&SchemaMeta, &mut StdRng) -> QueryTemplate)> = vec![
        ("collection_only",         500, build_collection_only),
        ("collection_fields",       600, build_collection_fields),
        ("collection_filter",       700, build_collection_filter),
        ("collection_fields_filter", 500, build_collection_fields_filter),
        ("collection_modifier",     500, build_collection_modifier),
        ("filter_modifier",         400, build_collection_filter_modifier),
        ("multi_filter",            400, build_multi_filter),
        ("traversal",               300, build_traversal),
        ("count",                   400, build_count),
        ("count_filter",            300, build_count_filter),
        ("aggregate",               500, build_aggregate),
        ("aggregate_filter",        300, build_aggregate_filter),
        ("order_by",                400, build_order_by),
        ("order_filter",            300, build_order_filter),
        ("group_by",                300, build_group_by),
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
            let words: Vec<&str> = template.raw_query.split_whitespace().collect();

            // Find word spans for each target
            let mut spans = Vec::new();
            let mut ok = true;
            let mut claimed_words = vec![false; words.len()];

            for target in &template.targets {
                if target.schema_type.is_empty() { continue; } // skip None targets

                match find_word_span(&words, &target.match_key) {
                    Some((start, end)) => {
                        // Check no overlap with already-claimed words
                        let overlap = (start..end).any(|w| claimed_words[w]);
                        if overlap {
                            // Try to find an alternative occurrence
                            let alt = find_alt_word_span(&words, &target.match_key, &claimed_words);
                            if let Some((s2, e2)) = alt {
                                for w in s2..e2 { claimed_words[w] = true; }
                                spans.push(NgramSpanLabel {
                                    start_word: s2,
                                    end_word: e2,
                                    span_type: target.span_type,
                                    concept_idx: concept_map.to_idx(&target.schema_type, target.schema_id),
                                });
                            } else {
                                ok = false;
                                break;
                            }
                        } else {
                            for w in start..end { claimed_words[w] = true; }
                            spans.push(NgramSpanLabel {
                                start_word: start,
                                end_word: end,
                                span_type: target.span_type,
                                concept_idx: concept_map.to_idx(&target.schema_type, target.schema_id),
                            });
                        }
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }

            if ok && !spans.is_empty() {
                samples.push(NgramSample {
                    query: template.raw_query,
                    spans,
                });
            } else {
                skipped += 1;
            }

            bar.inc(1);
        }
    }

    bar.finish_with_message("done");

    samples.shuffle(&mut rng);

    let dataset = NgramDataset { samples };
    dataset.save(&demo_dir.join("ngram_dataset.json")).expect("save dataset");

    println!("generated {} samples ({} skipped) -> demo/ngram_dataset.json",
        dataset.samples.len(), skipped);
}

/// Find an alternative occurrence of match_key that doesn't overlap claimed words.
fn find_alt_word_span(words: &[&str], key: &str, claimed: &[bool]) -> Option<(usize, usize)> {
    let key_lower = key.to_lowercase();
    let key_stripped = key_lower.trim_end_matches('s');

    for (i, word) in words.iter().enumerate() {
        if claimed[i] { continue; }
        let w = word.to_lowercase();
        let w_clean = w.trim_end_matches("'s").trim_end_matches("'").trim_end_matches(',');
        let w_stripped = w_clean.trim_end_matches('s');

        if w_clean == key_lower || w_clean.contains(&key_lower) || key_lower.contains(w_clean)
            || w_stripped == key_stripped || w_stripped.contains(key_stripped) || key_stripped.contains(w_stripped)
        {
            return Some((i, i + 1));
        }
    }
    None
}
