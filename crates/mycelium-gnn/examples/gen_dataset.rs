//! Generate a large training dataset from the demo schema.
//!
//! Usage:
//!   cargo run --example gen_dataset
//!
//! Parses demo/schema.surql, generates 5000 diverse training samples,
//! writes to demo/dataset.json.

use std::path::Path;
use rand::prelude::*;
use rand::rngs::StdRng;
use gnn_burn::schema::{Reader, Extractor, Schema, FieldType};
use gnn_burn::graph::SchemaGraph;
use gnn_burn::intent::*;
use gnn_burn::training::{TrainingSample, GroundTruth, Dataset};

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

        // Verify field count matches schema graph
        assert_eq!(field_offset, schema_graph.field_nodes.len(),
            "field count mismatch: {} vs {}", field_offset, schema_graph.field_nodes.len());

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

    fn random_distractor_tables(&self, exclude: usize, rng: &mut impl Rng) -> Vec<SchemaMatch> {
        let n = rng.random_range(1..=3).min(self.tables.len() - 1);
        let mut distractors = Vec::new();
        for _ in 0..n {
            let tid = loop {
                let t = rng.random_range(0..self.tables.len());
                if t != exclude { break t; }
            };
            // Some distractors are competitive (0.3-0.7), most are weak
            let score = if rng.random_bool(0.3) {
                rng.random_range(0.35..0.7)
            } else {
                rng.random_range(0.1..0.4)
            };
            distractors.push(SchemaMatch {
                schema_node_type: "table".into(),
                schema_node_id: tid,
                score,
            });
        }
        distractors
    }

    /// Find fields with the same local name across different tables (ambiguous matches)
    fn same_name_fields(&self, target_global_id: usize) -> Vec<usize> {
        // Find the local name of the target field
        let mut target_name = None;
        for table in &self.tables {
            for field in &table.fields {
                if field.global_id == target_global_id {
                    target_name = Some(field.local_name.clone());
                    break;
                }
            }
        }
        let target_name = match target_name {
            Some(n) => n,
            None => return vec![],
        };

        // Find all fields with the same name in other tables
        let mut matches = Vec::new();
        for table in &self.tables {
            for field in &table.fields {
                if field.local_name == target_name && field.global_id != target_global_id {
                    matches.push(field.global_id);
                }
            }
        }
        matches
    }

    fn random_distractor_fields(&self, exclude: usize, rng: &mut impl Rng) -> Vec<SchemaMatch> {
        let total_fields: usize = self.tables.iter().map(|t| t.fields.len()).sum();
        let mut distractors = Vec::new();

        // First: add same-name fields from other tables (hard distractors, high score)
        let ambiguous = self.same_name_fields(exclude);
        for fid in &ambiguous {
            distractors.push(SchemaMatch {
                schema_node_type: "field".into(),
                schema_node_id: *fid,
                score: rng.random_range(0.45..0.85), // competitive with correct match
            });
        }

        // Then: random distractors (easy, low score)
        let n = rng.random_range(0..=2).min(total_fields.saturating_sub(1));
        for _ in 0..n {
            let fid = loop {
                let f = rng.random_range(0..total_fields);
                if f != exclude && !ambiguous.contains(&f) { break f; }
            };
            distractors.push(SchemaMatch {
                schema_node_type: "field".into(),
                schema_node_id: fid,
                score: rng.random_range(0.1..0.35),
            });
        }
        distractors
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
// Surface form generators
// =============================================================================

/// Synonyms for table/collection names — more natural NL forms
fn collection_synonyms(name: &str) -> Vec<String> {
    let base = match name {
        "users" => vec!["users", "people", "accounts", "members", "staff", "customers", "folks"],
        "posts" => vec!["posts", "articles", "entries", "content", "publications", "stories", "writings"],
        "comments" => vec!["comments", "replies", "responses", "feedback", "remarks", "notes"],
        "tags" => vec!["tags", "labels", "categories", "topics", "markers"],
        "categories" => vec!["categories", "groups", "sections", "departments", "divisions", "types"],
        "likes" => vec!["likes", "favorites", "upvotes", "reactions", "endorsements"],
        "follows" => vec!["follows", "subscriptions", "connections", "followers"],
        "messages" => vec!["messages", "chats", "conversations", "mail", "inbox", "correspondence"],
        "products" => vec!["products", "items", "goods", "merchandise", "inventory", "listings"],
        "orders" => vec!["orders", "purchases", "transactions", "sales", "receipts"],
        "reviews" => vec!["reviews", "ratings", "evaluations", "assessments", "critiques", "feedback"],
        _ => vec![name],
    };
    base.into_iter().map(|s| s.to_string()).collect()
}

/// Synonyms for field names — natural ways people refer to fields
fn field_synonyms(field_name: &str) -> Vec<String> {
    let base = match field_name {
        "name" => vec!["name", "title", "label", "called", "named"],
        "email" => vec!["email", "mail", "email address", "contact"],
        "age" => vec!["age", "how old", "years old"],
        "bio" => vec!["bio", "biography", "about", "description", "profile"],
        "active" => vec!["active", "enabled", "alive", "online", "status"],
        "score" => vec!["score", "points", "rating", "rank"],
        "title" => vec!["title", "heading", "subject", "headline"],
        "content" => vec!["content", "body", "text", "details", "description"],
        "published" => vec!["published", "public", "visible", "live", "posted"],
        "views" => vec!["views", "view count", "seen", "impressions", "hits"],
        "rating" => vec!["rating", "stars", "score", "grade", "rank"],
        "text" => vec!["text", "content", "message", "body", "what they said"],
        "price" => vec!["price", "cost", "amount", "how much", "value"],
        "stock" => vec!["stock", "inventory", "available", "quantity", "in stock", "remaining"],
        "description" => vec!["description", "details", "about", "info", "summary"],
        "total" => vec!["total", "amount", "sum", "cost", "price"],
        "status" => vec!["status", "state", "condition", "progress"],
        "quantity" => vec!["quantity", "count", "number", "how many", "amount"],
        "body" => vec!["body", "content", "text", "message", "what they wrote"],
        "read" => vec!["read", "seen", "opened", "viewed"],
        "color" => vec!["color", "colour"],
        "created" => vec!["created", "date", "when", "created at", "timestamp", "time"],
        "likes" => vec!["likes", "upvotes", "reactions"],
        "parent" => vec!["parent", "above", "container", "belongs to"],
        _ => vec![field_name],
    };
    base.into_iter().map(|s| s.to_string()).collect()
}

fn collection_surface(name: &str, rng: &mut impl Rng) -> String {
    let synonyms = collection_synonyms(name);
    let word = &synonyms[rng.random_range(0..synonyms.len())];

    let templates: Vec<String> = vec![
        word.clone(),
        format!("all {}", word),
        format!("the {}", word),
        format!("every {}", word.trim_end_matches('s')),
        format!("show me {}", word),
        format!("get {}", word),
        format!("find {}", word),
        format!("list of {}", word),
        format!("all the {}", word),
        format!("which {}", word),
    ];
    templates[rng.random_range(0..templates.len())].clone()
}

fn field_surface(name: &str, rng: &mut impl Rng) -> String {
    let synonyms = field_synonyms(name);
    let word = &synonyms[rng.random_range(0..synonyms.len())];

    let templates: Vec<String> = vec![
        word.clone(),
        format!("the {}", word),
        format!("their {}", word),
        format!("show {}", word),
        format!("what {}", word),
    ];
    templates[rng.random_range(0..templates.len())].clone()
}

fn filter_op_surface(op_id: usize, rng: &mut impl Rng) -> String {
    let forms: &[&str] = match op_id {
        11 => &["equals", "is", "equal to", "=", "matching", "exactly", "same as", "where it's"],
        12 => &["not", "not equal to", "different from", "isn't", "!=", "other than", "excluding", "except"],
        13 => &["greater than", "more than", "above", "over", "exceeding", "higher than", ">", "bigger than"],
        14 => &["less than", "under", "below", "fewer than", "lower than", "<", "smaller than", "no more than"],
        15 => &["at least", "minimum", "no less than", ">=", "or more", "starting from"],
        16 => &["at most", "maximum", "no more than", "<=", "or less"],
        17 => &["like", "matching pattern", "similar to", "resembling"],
        18 => &["containing", "includes", "with", "has", "that contains"],
        19 => &["starting with", "begins with", "prefixed with", "starts with"],
        20 => &["ending with", "ends with", "suffixed with"],
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
            let words = [
                "hello", "world", "test", "admin", "user", "data", "example",
                "active", "pending", "draft", "published", "archived",
                "john", "alice", "bob", "carol", "dave", "eve", "frank",
                "gmail.com", "yahoo.com", "outlook.com",
                "news", "update", "review", "guide", "tutorial", "help",
                "red", "blue", "green", "yellow", "purple",
                "tech", "science", "art", "music", "sports",
            ];
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
        FieldType::String =>
            vec![11, 12, 17, 18, 19, 20],
        FieldType::Bool =>
            vec![11, 12],
        FieldType::Datetime | FieldType::Duration =>
            vec![11, 12, 13, 14, 15, 16],
        _ => vec![11, 12],
    }
}

fn modifier_surface(op_id: usize, value: &str, rng: &mut impl Rng) -> (String, String) {
    match op_id {
        10 => { // LIMIT
            let forms = [
                (format!("top {}", value), value.to_string()),
                (format!("first {}", value), value.to_string()),
                (format!("limit {}", value), value.to_string()),
                (format!("only {}", value), value.to_string()),
                (format!("{}", value), value.to_string()),
            ];
            let f = &forms[rng.random_range(0..forms.len())];
            (f.0.clone(), f.1.clone())
        }
        6 => { // ORDER_BY
            let forms = [
                ("sorted by".to_string(), value.to_string()),
                ("ordered by".to_string(), value.to_string()),
                ("by".to_string(), value.to_string()),
                ("sort by".to_string(), value.to_string()),
                ("newest".to_string(), String::new()),
                ("oldest".to_string(), String::new()),
                ("highest".to_string(), String::new()),
                ("lowest".to_string(), String::new()),
            ];
            let f = &forms[rng.random_range(0..forms.len())];
            (f.0.clone(), f.1.clone())
        }
        _ => ("modifier".to_string(), value.to_string()),
    }
}

fn traversal_surface(source: &str, target: &str, rng: &mut impl Rng) -> String {
    let src_syns = collection_synonyms(source);
    let tgt_syns = collection_synonyms(target);
    let s = &src_syns[rng.random_range(0..src_syns.len())];
    let t = &tgt_syns[rng.random_range(0..tgt_syns.len())];
    let forms = [
        format!("{} of {}", t, s),
        format!("{}'s {}", s, t),
        format!("{} from {}", t, s),
        format!("{} by {}", t, s),
        format!("{} related to {}", t, s),
        format!("linked {}", t),
        format!("{} who have {}", s, t),
        format!("{} with {}", s, t),
        format!("{} that belong to {}", t, s),
    ];
    forms[rng.random_range(0..forms.len())].clone()
}

// =============================================================================
// Sample generators
// =============================================================================

fn make_collection(
    meta: &SchemaMeta,
    table: &TableMeta,
    rng: &mut impl Rng,
) -> (CandidateMatch, Vec<usize>) {
    let conf = rng.random_range(0.55..0.95);
    let mut matches = vec![SchemaMatch {
        schema_node_type: "table".into(),
        schema_node_id: table.id,
        score: rng.random_range(0.5..0.9),
    }];
    // Distractors sometimes score close to correct
    for d in meta.random_distractor_tables(table.id, rng) {
        matches.push(d);
    }

    let cm = CandidateMatch {
        surface_form: collection_surface(&table.name, rng),
        confidence: conf,
        schema_matches: matches,
        operation_matches: vec![OperationMatch { operation_id: 0, score: rng.random_range(0.4..0.85) }],
    };
    (cm, vec![table.id])
}

fn make_fields(
    meta: &SchemaMeta,
    table: &TableMeta,
    n: usize,
    rng: &mut impl Rng,
) -> (Vec<CandidateMatch>, Vec<usize>) {
    let available = meta.non_record_fields(table);
    if available.is_empty() {
        return (vec![], vec![]);
    }
    let n = n.min(available.len());
    let chosen: Vec<&FieldMeta> = available.choose_multiple(rng, n).copied().collect();

    let mut cms = Vec::new();
    let mut targets = Vec::new();

    for field in chosen {
        let conf = rng.random_range(0.45..0.92);
        let correct_score = rng.random_range(0.45..0.88);
        let mut matches = vec![SchemaMatch {
            schema_node_type: "field".into(),
            schema_node_id: field.global_id,
            score: correct_score,
        }];
        // Ambiguous + random distractors
        matches.extend(meta.random_distractor_fields(field.global_id, rng));

        cms.push(CandidateMatch {
            surface_form: field_surface(&field.local_name, rng),
            confidence: conf,
            schema_matches: matches,
            operation_matches: vec![],
        });
        targets.push(field.global_id);
    }

    (cms, targets)
}

fn make_filter(
    meta: &SchemaMeta,
    table: &TableMeta,
    rng: &mut impl Rng,
) -> Option<(FilterMatch, usize, usize)> {
    let filterable = meta.filterable_fields(table);
    if filterable.is_empty() {
        return None;
    }
    let field = filterable[rng.random_range(0..filterable.len())];
    let ops = compatible_filter_ops(&field.field_type);
    let op_id = ops[rng.random_range(0..ops.len())];

    let conf = rng.random_range(0.5..0.92);
    let value = random_value(&field.field_type, rng);
    let op_surface = filter_op_surface(op_id, rng);

    let mut field_matches = vec![SchemaMatch {
        schema_node_type: "field".into(),
        schema_node_id: field.global_id,
        score: rng.random_range(0.45..0.88),
    }];
    field_matches.extend(meta.random_distractor_fields(field.global_id, rng));

    let fm = FilterMatch {
        field: CandidateMatch {
            surface_form: field_surface(&field.local_name, rng),
            confidence: rng.random_range(0.45..0.9),
            schema_matches: field_matches,
            operation_matches: vec![],
        },
        operator: op_surface,
        value,
        confidence: conf,
        operation_matches: vec![OperationMatch {
            operation_id: op_id,
            score: rng.random_range(0.5..0.88),
        }],
    };

    Some((fm, field.global_id, op_id))
}

fn gen_collection_only(
    meta: &SchemaMeta,
    rng: &mut impl Rng,
) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let (coll, coll_targets) = make_collection(meta, table, rng);

    TrainingSample {
        extraction: Extraction {
            collections: vec![coll],
            fields: vec![], filters: vec![], traversals: vec![], modifiers: vec![],
        },
        ground_truth: GroundTruth {
            collection_targets: coll_targets,
            field_targets: vec![], filter_op_targets: vec![],
            traversal_targets: vec![], modifier_op_targets: vec![],
        },
    }
}

fn gen_collection_fields(
    meta: &SchemaMeta,
    rng: &mut impl Rng,
) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let (coll, coll_targets) = make_collection(meta, table, rng);
    let n_fields = rng.random_range(1..=3);
    let (fields, field_targets) = make_fields(meta, table, n_fields, rng);

    TrainingSample {
        extraction: Extraction {
            collections: vec![coll],
            fields, filters: vec![], traversals: vec![], modifiers: vec![],
        },
        ground_truth: GroundTruth {
            collection_targets: coll_targets,
            field_targets, filter_op_targets: vec![],
            traversal_targets: vec![], modifier_op_targets: vec![],
        },
    }
}

fn gen_collection_filter(
    meta: &SchemaMeta,
    rng: &mut impl Rng,
) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let (coll, coll_targets) = make_collection(meta, table, rng);

    if let Some((fm, field_target, op_target)) = make_filter(meta, table, rng) {
        TrainingSample {
            extraction: Extraction {
                collections: vec![coll],
                fields: vec![], filters: vec![fm], traversals: vec![], modifiers: vec![],
            },
            ground_truth: GroundTruth {
                collection_targets: coll_targets,
                field_targets: vec![field_target],
                filter_op_targets: vec![op_target],
                traversal_targets: vec![], modifier_op_targets: vec![],
            },
        }
    } else {
        gen_collection_only(meta, rng)
    }
}

fn gen_collection_fields_filter(
    meta: &SchemaMeta,
    rng: &mut impl Rng,
) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let (coll, coll_targets) = make_collection(meta, table, rng);
    let n_fields = rng.random_range(1..=2);
    let (fields, mut field_targets) = make_fields(meta, table, n_fields, rng);

    if let Some((fm, filter_field_target, op_target)) = make_filter(meta, table, rng) {
        // Filter's field might duplicate a standalone field — QueryGraph deduplicates by surface_form.
        // If not already in field_targets, add it.
        let filter_field_name = &fm.field.surface_form;
        let already_exists = fields.iter().any(|f| f.surface_form == *filter_field_name);
        if !already_exists {
            field_targets.push(filter_field_target);
        }

        TrainingSample {
            extraction: Extraction {
                collections: vec![coll],
                fields, filters: vec![fm], traversals: vec![], modifiers: vec![],
            },
            ground_truth: GroundTruth {
                collection_targets: coll_targets,
                field_targets, filter_op_targets: vec![op_target],
                traversal_targets: vec![], modifier_op_targets: vec![],
            },
        }
    } else {
        gen_collection_fields(meta, rng)
    }
}

fn gen_collection_modifier(
    meta: &SchemaMeta,
    rng: &mut impl Rng,
) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let (coll, coll_targets) = make_collection(meta, table, rng);

    let mut modifiers = Vec::new();
    let mut modifier_targets = Vec::new();
    let mut fields = Vec::new();
    let mut field_targets = Vec::new();

    // LIMIT
    if rng.random_bool(0.7) {
        let limit_val = rng.random_range(1..50).to_string();
        let (surface, value) = modifier_surface(10, &limit_val, rng);
        modifiers.push(ModifierMatch {
            surface_form: surface, value,
            confidence: rng.random_range(0.75..0.95),
            operation_matches: vec![OperationMatch { operation_id: 10, score: rng.random_range(0.75..0.95) }],
        });
        modifier_targets.push(10);
    }

    // ORDER_BY
    if rng.random_bool(0.6) {
        let orderable = meta.orderable_fields(table);
        if let Some(field) = orderable.choose(rng) {
            let (surface, value) = modifier_surface(6, &field.local_name, rng);
            modifiers.push(ModifierMatch {
                surface_form: surface, value,
                confidence: rng.random_range(0.72..0.93),
                operation_matches: vec![OperationMatch { operation_id: 6, score: rng.random_range(0.7..0.92) }],
            });
            modifier_targets.push(6);

            // Add the sort field
            let mut matches = vec![SchemaMatch {
                schema_node_type: "field".into(),
                schema_node_id: field.global_id,
                score: rng.random_range(0.65..0.9),
            }];
            matches.extend(meta.random_distractor_fields(field.global_id, rng));
            fields.push(CandidateMatch {
                surface_form: field_surface(&field.local_name, rng),
                confidence: rng.random_range(0.65..0.88),
                schema_matches: matches,
                operation_matches: vec![],
            });
            field_targets.push(field.global_id);
        }
    }

    // Ensure at least one modifier
    if modifiers.is_empty() {
        let limit_val = rng.random_range(1..50).to_string();
        let (surface, value) = modifier_surface(10, &limit_val, rng);
        modifiers.push(ModifierMatch {
            surface_form: surface, value,
            confidence: rng.random_range(0.75..0.95),
            operation_matches: vec![OperationMatch { operation_id: 10, score: rng.random_range(0.75..0.95) }],
        });
        modifier_targets.push(10);
    }

    TrainingSample {
        extraction: Extraction {
            collections: vec![coll],
            fields, filters: vec![], traversals: vec![], modifiers,
        },
        ground_truth: GroundTruth {
            collection_targets: coll_targets,
            field_targets, filter_op_targets: vec![],
            traversal_targets: vec![], modifier_op_targets: modifier_targets,
        },
    }
}

fn gen_collection_filter_modifier(
    meta: &SchemaMeta,
    rng: &mut impl Rng,
) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let (coll, coll_targets) = make_collection(meta, table, rng);

    let mut field_targets = Vec::new();

    // Filter
    let (filters, filter_op_targets) = if let Some((fm, ft, ot)) = make_filter(meta, table, rng) {
        field_targets.push(ft);
        (vec![fm], vec![ot])
    } else {
        (vec![], vec![])
    };

    // Modifier (LIMIT)
    let limit_val = rng.random_range(1..50).to_string();
    let (surface, value) = modifier_surface(10, &limit_val, rng);
    let modifiers = vec![ModifierMatch {
        surface_form: surface, value,
        confidence: rng.random_range(0.75..0.95),
        operation_matches: vec![OperationMatch { operation_id: 10, score: rng.random_range(0.75..0.95) }],
    }];

    TrainingSample {
        extraction: Extraction {
            collections: vec![coll],
            fields: vec![], filters, traversals: vec![], modifiers,
        },
        ground_truth: GroundTruth {
            collection_targets: coll_targets,
            field_targets, filter_op_targets,
            traversal_targets: vec![], modifier_op_targets: vec![10],
        },
    }
}

fn gen_multi_filter(
    meta: &SchemaMeta,
    rng: &mut impl Rng,
) -> TrainingSample {
    let table = &meta.tables[rng.random_range(0..meta.tables.len())];
    let (coll, coll_targets) = make_collection(meta, table, rng);

    let n_filters = rng.random_range(2..=3);
    let mut filters = Vec::new();
    let mut field_targets = Vec::new();
    let mut filter_op_targets = Vec::new();
    let mut used_fields = std::collections::HashSet::new();

    for _ in 0..n_filters {
        if let Some((fm, ft, ot)) = make_filter(meta, table, rng) {
            if !used_fields.contains(&ft) {
                used_fields.insert(ft);
                field_targets.push(ft);
                filter_op_targets.push(ot);
                filters.push(fm);
            }
        }
    }

    if filters.is_empty() {
        return gen_collection_only(meta, rng);
    }

    TrainingSample {
        extraction: Extraction {
            collections: vec![coll],
            fields: vec![], filters, traversals: vec![], modifiers: vec![],
        },
        ground_truth: GroundTruth {
            collection_targets: coll_targets,
            field_targets, filter_op_targets,
            traversal_targets: vec![], modifier_op_targets: vec![],
        },
    }
}

fn gen_traversal(
    meta: &SchemaMeta,
    rng: &mut impl Rng,
) -> TrainingSample {
    // Find tables with record fields
    let tables_with_records: Vec<&TableMeta> = meta.tables.iter()
        .filter(|t| meta.record_fields(t).len() > 0)
        .collect();

    if tables_with_records.is_empty() {
        return gen_collection_only(meta, rng);
    }

    let table = tables_with_records[rng.random_range(0..tables_with_records.len())];
    let (coll, coll_targets) = make_collection(meta, table, rng);

    let record_fields = meta.record_fields(table);
    let rec_field = record_fields[rng.random_range(0..record_fields.len())];
    let target_table_id = rec_field.record_target.unwrap();
    let target_name = &meta.tables[target_table_id].name;

    let trav_ops = [31, 32, 33]; // arrow_right, arrow_left, arrow_both
    let trav_op = trav_ops[rng.random_range(0..trav_ops.len())];

    let mut matches = vec![SchemaMatch {
        schema_node_type: "table".into(),
        schema_node_id: target_table_id,
        score: rng.random_range(0.7..0.92),
    }];
    matches.extend(meta.random_distractor_tables(target_table_id, rng));

    let trav = CandidateMatch {
        surface_form: traversal_surface(&table.name, target_name, rng),
        confidence: rng.random_range(0.68..0.92),
        schema_matches: matches,
        operation_matches: vec![OperationMatch {
            operation_id: trav_op,
            score: rng.random_range(0.65..0.9),
        }],
    };

    TrainingSample {
        extraction: Extraction {
            collections: vec![coll],
            fields: vec![], filters: vec![],
            traversals: vec![trav], modifiers: vec![],
        },
        ground_truth: GroundTruth {
            collection_targets: coll_targets,
            field_targets: vec![], filter_op_targets: vec![],
            traversal_targets: vec![target_table_id], modifier_op_targets: vec![],
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

    // 5000 total, varied patterns
    for _ in 0..700  { samples.push(gen_collection_only(&meta, &mut rng)); }
    for _ in 0..1100 { samples.push(gen_collection_fields(&meta, &mut rng)); }
    for _ in 0..1000 { samples.push(gen_collection_filter(&meta, &mut rng)); }
    for _ in 0..500  { samples.push(gen_collection_fields_filter(&meta, &mut rng)); }
    for _ in 0..500  { samples.push(gen_collection_modifier(&meta, &mut rng)); }
    for _ in 0..500  { samples.push(gen_collection_filter_modifier(&meta, &mut rng)); }
    for _ in 0..400  { samples.push(gen_multi_filter(&meta, &mut rng)); }
    for _ in 0..300  { samples.push(gen_traversal(&meta, &mut rng)); }

    samples.shuffle(&mut rng);

    let dataset = Dataset { samples };
    dataset.save(&demo_dir.join("dataset.json")).expect("save dataset");
    println!("generated {} samples → demo/dataset.json", dataset.samples.len());
}
