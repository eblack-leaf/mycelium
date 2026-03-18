//! Generate training dataset for the biaffine head.
//!
//! Reuses template generation patterns from gen_dataset.rs (SchemaMeta, surface
//! forms, 9 query types). Only loads MiniLM **tokenizer** (no ONNX inference) —
//! we just need subword tokenization for alignment labels.
//!
//! For each template query with known ground-truth spans/edges:
//!   1. Tokenize → subword tokens + offsets
//!   2. build_subword_to_word() for alignment
//!   3. assign_bio_tags() for BIO labels
//!   4. Compute span boundaries + arc labels
//!
//! Output: demo/biaffine_dataset.json (~5000 samples)
//!
//! Usage:
//!   cargo run --release --example gen_biaffine_dataset -p gnn-burn

use std::path::Path;
use rand::prelude::*;
use rand::rngs::StdRng;
use tokenizers::Tokenizer;
use gnn_burn::schema::{Reader, Extractor, Schema, FieldType};
use gnn_burn::graph::SchemaGraph;
use gnn_burn::nlp::{SpanType, DepRelation};
use gnn_burn::biaffine_data::{
    BiaffineSample, BiaffineDataset,
    build_subword_to_word, assign_bio_tags, span_type_to_index,
};

// =============================================================================
// Schema metadata (same as gen_dataset.rs)
// =============================================================================

struct TableMeta {
    #[allow(dead_code)]
    id: usize,
    name: String,
    fields: Vec<FieldMeta>,
}

struct FieldMeta {
    #[allow(dead_code)]
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
            tables.push(TableMeta { id: table_id, name: table.name.clone(), fields });
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
        table.fields.iter().filter(|f| f.record_target.is_none()).collect()
    }

    fn record_fields<'a>(&self, table: &'a TableMeta) -> Vec<&'a FieldMeta> {
        table.fields.iter().filter(|f| f.record_target.is_some()).collect()
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
// Span descriptor — ground-truth span + text for building samples
// =============================================================================

struct SpanDesc {
    text: String,
    span_type: SpanType,
}

struct ArcDesc {
    src_span: usize,  // index into spans vec
    dst_span: usize,
    relation: DepRelation,
}

fn dep_relation_index(r: DepRelation) -> usize {
    match r {
        DepRelation::Possessive  => 0,
        DepRelation::Quantifies  => 1,
        DepRelation::Comparison  => 2,
        DepRelation::IntentTarget => 3,
    }
}

/// Build a BiaffineSample from a query string, ground-truth spans, and arcs.
///
/// The spans must appear in the query in order — we locate each span text in the
/// query to determine word-level boundaries, then use the tokenizer for subword
/// alignment.
fn build_sample(
    tokenizer: &Tokenizer,
    query: &str,
    spans: &[SpanDesc],
    arcs: &[ArcDesc],
) -> Option<BiaffineSample> {
    let lower = query.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    // Strip possessive 's and trailing punctuation for matching
    let clean_words: Vec<String> = words.iter()
        .map(|w| {
            let mut s = w.to_string();
            // Strip possessive suffix first
            if s.ends_with("'s") { s.truncate(s.len() - 2); }
            else if s.ends_with("'") { s.truncate(s.len() - 1); }
            // Then trailing commas/periods
            while s.ends_with(',') || s.ends_with('.') { s.pop(); }
            s
        })
        .collect();

    // Locate each span in the word sequence
    let mut word_spans: Vec<(usize, usize, SpanType)> = Vec::new();
    let mut search_from = 0;

    for span in spans {
        let span_lower = span.text.to_lowercase();
        let span_words: Vec<&str> = span_lower.split_whitespace().collect();
        if span_words.is_empty() { return None; }

        // Find span_words starting from search_from, matching against cleaned words
        let mut found = false;
        for start in search_from..words.len() {
            if start + span_words.len() > words.len() { break; }
            let matches = span_words.iter()
                .enumerate()
                .all(|(j, sw)| {
                    let cw = &clean_words[start + j];
                    cw == sw || words[start + j] == *sw
                });
            if matches {
                word_spans.push((start, start + span_words.len(), span.span_type));
                search_from = start + span_words.len();
                found = true;
                break;
            }
        }
        if !found {
            // Span not found in query — skip this sample
            return None;
        }
    }

    // Tokenize (with special tokens)
    let encoding = tokenizer.encode(query, true).ok()?;
    let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

    let subword_to_word = build_subword_to_word(&offsets, query);
    let seq_len = subword_to_word.len();
    if seq_len == 0 { return None; }

    let bio_tags = assign_bio_tags(&subword_to_word, &word_spans, seq_len);

    let span_boundaries: Vec<(usize, usize, usize)> = word_spans.iter()
        .map(|&(s, e, st)| (s, e, span_type_to_index(st)))
        .collect();

    let arc_labels: Vec<(usize, usize, usize)> = arcs.iter()
        .map(|a| (a.src_span, a.dst_span, dep_relation_index(a.relation)))
        .collect();

    Some(BiaffineSample {
        query: query.to_string(),
        bio_tags,
        subword_to_word,
        span_boundaries,
        arcs: arc_labels,
    })
}

// =============================================================================
// Sample generators (one per query pattern)
// =============================================================================

fn gen_collection_only(meta: &SchemaMeta, tokenizer: &Tokenizer, rng: &mut impl Rng) -> Option<BiaffineSample> {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let intent = ["show", "find", "get", "list"][rng.random_range(0..4)];
    let coll = collection_surface(&table.name, rng);
    let query = format!("{} {}", intent, coll);

    build_sample(tokenizer, &query, &[
        SpanDesc { text: intent.to_string(), span_type: SpanType::Intent },
        SpanDesc { text: coll, span_type: SpanType::NounPhrase },
    ], &[
        ArcDesc { src_span: 0, dst_span: 1, relation: DepRelation::IntentTarget },
    ])
}

fn gen_collection_fields(meta: &SchemaMeta, tokenizer: &Tokenizer, rng: &mut impl Rng) -> Option<BiaffineSample> {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let available = meta.non_record_fields(table);
    if available.is_empty() { return gen_collection_only(meta, tokenizer, rng); }

    let n_fields = rng.random_range(1..=2).min(available.len());
    let chosen: Vec<&FieldMeta> = available.choose_multiple(rng, n_fields).copied().collect();

    let intent = ["show", "get", "list", "find"][rng.random_range(0..4)];
    let coll = collection_surface(&table.name, rng);

    let field_texts: Vec<String> = chosen.iter().map(|f| field_surface(&f.local_name, rng)).collect();
    let fields_joined = field_texts.join(" and ");
    let query = format!("{} {}'s {}", intent, coll, fields_joined);

    let mut spans = vec![
        SpanDesc { text: intent.to_string(), span_type: SpanType::Intent },
        SpanDesc { text: coll, span_type: SpanType::NounPhrase },
    ];
    let mut arcs = vec![
        ArcDesc { src_span: 0, dst_span: 1, relation: DepRelation::IntentTarget },
    ];

    for ft in field_texts {
        let span_idx = spans.len();
        spans.push(SpanDesc { text: ft, span_type: SpanType::NounPhrase });
        arcs.push(ArcDesc { src_span: 1, dst_span: span_idx, relation: DepRelation::Possessive });
    }

    build_sample(tokenizer, &query, &spans, &arcs)
}

fn gen_collection_filter(meta: &SchemaMeta, tokenizer: &Tokenizer, rng: &mut impl Rng) -> Option<BiaffineSample> {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let filterable = meta.filterable_fields(table);
    if filterable.is_empty() { return gen_collection_only(meta, tokenizer, rng); }

    let field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);

    let intent = ["find", "get", "show"][rng.random_range(0..3)];
    let coll = collection_surface(&table.name, rng);
    let field_text = field_surface(&field.local_name, rng);
    let comp_text = format!("{} {}", op_surface, value);

    let query = format!("{} {} where {} {}", intent, coll, field_text, comp_text);

    build_sample(tokenizer, &query, &[
        SpanDesc { text: intent.to_string(), span_type: SpanType::Intent },
        SpanDesc { text: coll, span_type: SpanType::NounPhrase },
        SpanDesc { text: field_text, span_type: SpanType::NounPhrase },
        SpanDesc { text: comp_text, span_type: SpanType::Comparator },
    ], &[
        ArcDesc { src_span: 0, dst_span: 1, relation: DepRelation::IntentTarget },
        ArcDesc { src_span: 3, dst_span: 2, relation: DepRelation::Comparison },
    ])
}

fn gen_collection_fields_filter(meta: &SchemaMeta, tokenizer: &Tokenizer, rng: &mut impl Rng) -> Option<BiaffineSample> {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let available = meta.non_record_fields(table);
    let filterable = meta.filterable_fields(table);
    if available.is_empty() || filterable.is_empty() {
        return gen_collection_only(meta, tokenizer, rng);
    }

    let sel_field = available[rng.random_range(0..available.len())];
    let filt_field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&filt_field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&filt_field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);

    let intent = ["show", "get", "find"][rng.random_range(0..3)];
    let coll = collection_surface(&table.name, rng);
    let sel_text = field_surface(&sel_field.local_name, rng);
    let filt_text = field_surface(&filt_field.local_name, rng);
    let comp_text = format!("{} {}", op_surface, value);

    let query = format!("{} {}'s {} where {} {}", intent, coll, sel_text, filt_text, comp_text);

    build_sample(tokenizer, &query, &[
        SpanDesc { text: intent.to_string(), span_type: SpanType::Intent },
        SpanDesc { text: coll, span_type: SpanType::NounPhrase },
        SpanDesc { text: sel_text, span_type: SpanType::NounPhrase },
        SpanDesc { text: filt_text, span_type: SpanType::NounPhrase },
        SpanDesc { text: comp_text, span_type: SpanType::Comparator },
    ], &[
        ArcDesc { src_span: 0, dst_span: 1, relation: DepRelation::IntentTarget },
        ArcDesc { src_span: 1, dst_span: 2, relation: DepRelation::Possessive },
        ArcDesc { src_span: 4, dst_span: 3, relation: DepRelation::Comparison },
    ])
}

fn gen_collection_modifier(meta: &SchemaMeta, tokenizer: &Tokenizer, rng: &mut impl Rng) -> Option<BiaffineSample> {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let intent = ["show", "get", "find", "list"][rng.random_range(0..4)];
    let coll = collection_surface(&table.name, rng);
    let limit_val = rng.random_range(1..50);
    let quant_text = format!("first {}", limit_val);

    let mut spans = vec![
        SpanDesc { text: intent.to_string(), span_type: SpanType::Intent },
        SpanDesc { text: coll.clone(), span_type: SpanType::NounPhrase },
        SpanDesc { text: quant_text.clone(), span_type: SpanType::Quantifier },
    ];
    let mut arcs = vec![
        ArcDesc { src_span: 0, dst_span: 1, relation: DepRelation::IntentTarget },
        ArcDesc { src_span: 2, dst_span: 1, relation: DepRelation::Quantifies },
    ];

    let mut query = format!("{} {}, {}", intent, coll, quant_text);

    // Optionally add ORDER BY field
    if rng.random_bool(0.5) {
        let orderable = meta.orderable_fields(table);
        if let Some(field) = orderable.choose(rng) {
            let ft = field_surface(&field.local_name, rng);
            query = format!("{} by {}", query, ft);
            let span_idx = spans.len();
            spans.push(SpanDesc { text: ft, span_type: SpanType::NounPhrase });
            arcs.push(ArcDesc { src_span: 1, dst_span: span_idx, relation: DepRelation::Possessive });
        }
    }

    build_sample(tokenizer, &query, &spans, &arcs)
}

fn gen_collection_filter_modifier(meta: &SchemaMeta, tokenizer: &Tokenizer, rng: &mut impl Rng) -> Option<BiaffineSample> {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let filterable = meta.filterable_fields(table);
    if filterable.is_empty() { return gen_collection_modifier(meta, tokenizer, rng); }

    let field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];
    let value = random_value(&field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);
    let limit_val = rng.random_range(1..50);

    let intent = ["find", "get", "show"][rng.random_range(0..3)];
    let coll = collection_surface(&table.name, rng);
    let field_text = field_surface(&field.local_name, rng);
    let comp_text = format!("{} {}", op_surface, value);
    let quant_text = format!("first {}", limit_val);

    let query = format!("{} {} where {} {}, {}", intent, coll, field_text, comp_text, quant_text);

    build_sample(tokenizer, &query, &[
        SpanDesc { text: intent.to_string(), span_type: SpanType::Intent },
        SpanDesc { text: coll, span_type: SpanType::NounPhrase },
        SpanDesc { text: field_text, span_type: SpanType::NounPhrase },
        SpanDesc { text: comp_text, span_type: SpanType::Comparator },
        SpanDesc { text: quant_text, span_type: SpanType::Quantifier },
    ], &[
        ArcDesc { src_span: 0, dst_span: 1, relation: DepRelation::IntentTarget },
        ArcDesc { src_span: 3, dst_span: 2, relation: DepRelation::Comparison },
        ArcDesc { src_span: 4, dst_span: 1, relation: DepRelation::Quantifies },
    ])
}

fn gen_multi_filter(meta: &SchemaMeta, tokenizer: &Tokenizer, rng: &mut impl Rng) -> Option<BiaffineSample> {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let filterable = meta.filterable_fields(table);
    if filterable.len() < 2 { return gen_collection_filter(meta, tokenizer, rng); }

    let n_filters = rng.random_range(2..=3).min(filterable.len());
    let chosen: Vec<&FieldMeta> = filterable.choose_multiple(rng, n_filters).copied().collect();

    let intent = ["find", "get", "show"][rng.random_range(0..3)];
    let coll = collection_surface(&table.name, rng);

    let mut spans = vec![
        SpanDesc { text: intent.to_string(), span_type: SpanType::Intent },
        SpanDesc { text: coll.clone(), span_type: SpanType::NounPhrase },
    ];
    let mut arcs = vec![
        ArcDesc { src_span: 0, dst_span: 1, relation: DepRelation::IntentTarget },
    ];

    let mut filter_parts: Vec<String> = Vec::new();
    for field in &chosen {
        let ops = compatible_filter_ops(&field.field_type);
        let op_id = ops[rng.random_range(0..ops.len())];
        let value = random_value(&field.field_type, rng);
        let op_surface = filter_op_surface(op_id, rng);
        let ft = field_surface(&field.local_name, rng);
        let comp_text = format!("{} {}", op_surface, value);

        let field_span_idx = spans.len();
        spans.push(SpanDesc { text: ft.clone(), span_type: SpanType::NounPhrase });
        let comp_span_idx = spans.len();
        spans.push(SpanDesc { text: comp_text.clone(), span_type: SpanType::Comparator });
        arcs.push(ArcDesc { src_span: comp_span_idx, dst_span: field_span_idx, relation: DepRelation::Comparison });

        filter_parts.push(format!("{} {}", ft, comp_text));
    }

    let query = format!("{} {} where {}", intent, coll, filter_parts.join(" and "));
    build_sample(tokenizer, &query, &spans, &arcs)
}

fn gen_traversal(meta: &SchemaMeta, tokenizer: &Tokenizer, rng: &mut impl Rng) -> Option<BiaffineSample> {
    let tables_with_records: Vec<&TableMeta> = meta.tables.iter()
        .filter(|t| !meta.record_fields(t).is_empty())
        .collect();
    if tables_with_records.is_empty() { return gen_collection_only(meta, tokenizer, rng); }

    let table = tables_with_records[rng.random_range(0..tables_with_records.len())];
    let record_fields = meta.record_fields(table);
    let rec_field = record_fields[rng.random_range(0..record_fields.len())];
    let target_table_id = rec_field.record_target.unwrap();
    let target_name = &meta.tables[target_table_id].name;

    let intent = ["find", "get", "show"][rng.random_range(0..3)];
    let coll = collection_surface(&table.name, rng);

    let query = format!("{} {}'s {}", intent, coll, target_name);

    build_sample(tokenizer, &query, &[
        SpanDesc { text: intent.to_string(), span_type: SpanType::Intent },
        SpanDesc { text: coll, span_type: SpanType::NounPhrase },
        SpanDesc { text: target_name.clone(), span_type: SpanType::NounPhrase },
    ], &[
        ArcDesc { src_span: 0, dst_span: 1, relation: DepRelation::IntentTarget },
        ArcDesc { src_span: 1, dst_span: 2, relation: DepRelation::Possessive },
    ])
}

fn gen_count(meta: &SchemaMeta, tokenizer: &Tokenizer, rng: &mut impl Rng) -> Option<BiaffineSample> {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let coll = collection_surface(&table.name, rng);
    let query = format!("count {}", coll);

    build_sample(tokenizer, &query, &[
        SpanDesc { text: "count".to_string(), span_type: SpanType::Intent },
        SpanDesc { text: coll, span_type: SpanType::NounPhrase },
    ], &[
        ArcDesc { src_span: 0, dst_span: 1, relation: DepRelation::IntentTarget },
    ])
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");

    // Load schema
    let raw = Reader::read(&demo_dir.join("schema.surql")).expect("read schema");
    let (schema, _) = Extractor::extract(&raw);
    let schema_graph = SchemaGraph::from_schema(&schema);
    let meta = SchemaMeta::from_schema(&schema, &schema_graph);

    // Load tokenizer only (no ONNX model needed)
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .expect("load tokenizer");

    let mut rng = StdRng::seed_from_u64(42);
    let mut samples = Vec::new();
    let mut skipped = 0usize;

    println!("schema: {} tables, {} fields",
        meta.tables.len(),
        meta.tables.iter().map(|t| t.fields.len()).sum::<usize>());

    // Generate samples — same distribution as gen_dataset.rs
    let generators: Vec<(&str, usize, Box<dyn Fn(&SchemaMeta, &Tokenizer, &mut StdRng) -> Option<BiaffineSample>>)> = vec![
        ("collection_only",           600,  Box::new(|m, t, r| gen_collection_only(m, t, r))),
        ("collection_fields",         1000, Box::new(|m, t, r| gen_collection_fields(m, t, r))),
        ("collection_filter",         900,  Box::new(|m, t, r| gen_collection_filter(m, t, r))),
        ("collection_fields_filter",  500,  Box::new(|m, t, r| gen_collection_fields_filter(m, t, r))),
        ("collection_modifier",       500,  Box::new(|m, t, r| gen_collection_modifier(m, t, r))),
        ("collection_filter_modifier",400,  Box::new(|m, t, r| gen_collection_filter_modifier(m, t, r))),
        ("multi_filter",              400,  Box::new(|m, t, r| gen_multi_filter(m, t, r))),
        ("traversal",                 300,  Box::new(|m, t, r| gen_traversal(m, t, r))),
        ("count",                     400,  Box::new(|m, t, r| gen_count(m, t, r))),
    ];

    for (name, count, gen_fn) in &generators {
        let before = samples.len();
        for _ in 0..*count {
            match gen_fn(&meta, &tokenizer, &mut rng) {
                Some(s) => samples.push(s),
                None => skipped += 1,
            }
        }
        println!("  {}: {} samples", name, samples.len() - before);
    }

    samples.shuffle(&mut rng);

    let dataset = BiaffineDataset { samples };
    let output_path = demo_dir.join("biaffine_dataset.json");
    dataset.save(&output_path).expect("save dataset");
    println!("generated {} samples ({} skipped) -> {:?}",
        dataset.samples.len(), skipped, output_path);
}
