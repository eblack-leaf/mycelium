// basidium — synthetic training data generation for the mycelium domain

pub mod trainable;
pub mod trainer;

use hyphae::query::{ModifierKind, QueryNode};
use hyphae::schema::{FieldType, Field, Schema, Table};
use serde::{Deserialize, Serialize};
use septa::*;

/// A single labelled training example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Datum {
    pub nl: String,
    pub surql: String,
    pub semantics: Semantics,
    /// GNN supervision: for each span node, the correct QueryNode resolution target.
    /// A single span can have multiple labels when it resolves more than one thing
    /// (e.g. a ConditionSpan has a Field label AND a Comparator label).
    pub labels: Vec<SpanLabel>,
    /// Ground truth QueryIr for end-to-end evaluation.
    pub ir: Option<hyphae::query::QueryIr>,
}

/// Which span vec in Semantics a label refers to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpanType {
    Intent,
    Entity,
    Projection,
    Condition,
    Assignment,
    Modifier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLabel {
    pub span_type: SpanType,
    pub span_index: usize,   // index within that vec (0 for Intent/Entity since they're singular)
    pub target: QueryNode, // correct resolution — variant tells the bilinear head which head to score
}

// =============================================================================
// Position-tracking string builder
// =============================================================================

struct Tk(String);

impl Tk {
    fn new() -> Self { Self(String::new()) }
    fn push(&mut self, s: &str) -> (usize, usize) {
        let start = self.0.len();
        self.0.push_str(s);
        (start, self.0.len())
    }
    fn lit(&mut self, s: &str) { self.0.push_str(s); }
    fn sp(&mut self) { self.0.push(' '); }
    fn done(self) -> String { self.0 }
}

// =============================================================================
// Phrasing + value helpers
// =============================================================================

fn comparator_texts() -> Vec<(Comparator, Vec<&'static str>)> {
    vec![
        (Comparator::Eq,       vec!["equals", "is", "is equal to", "matches"]),
        (Comparator::Neq,      vec!["is not", "does not equal", "differs from"]),
        (Comparator::Gt,       vec!["is greater than", "is more than", "exceeds", "is above"]),
        (Comparator::Gte,      vec!["is at least", "is no less than"]),
        (Comparator::Lt,       vec!["is less than", "is under", "is below"]),
        (Comparator::Lte,      vec!["is at most", "is no more than"]),
        (Comparator::Contains, vec!["contains", "includes", "has"]),
    ]
}

fn compatible_cmps(field: &Field) -> Vec<Comparator> {
    match &field.field_type {
        FieldType::Int | FieldType::Float | FieldType::Decimal | FieldType::Number =>
            vec![Comparator::Eq, Comparator::Neq, Comparator::Gt, Comparator::Gte, Comparator::Lt, Comparator::Lte],
        FieldType::String =>
            vec![Comparator::Eq, Comparator::Neq, Comparator::Contains],
        FieldType::Bool =>
            vec![Comparator::Eq, Comparator::Neq],
        FieldType::Datetime =>
            vec![Comparator::Eq, Comparator::Gt, Comparator::Lt],
        _ => vec![Comparator::Eq, Comparator::Neq],
    }
}

fn sample_value(field: &Field) -> (&'static str, ValueRef) {
    match &field.field_type {
        FieldType::String   => ("test",  ValueRef::Literal("test".into())),
        FieldType::Int      => ("42",    ValueRef::Literal("42".into())),
        FieldType::Float | FieldType::Decimal | FieldType::Number
                            => ("3.14",  ValueRef::Literal("3.14".into())),
        FieldType::Bool     => ("true",  ValueRef::Literal("true".into())),
        FieldType::Datetime => ("today", ValueRef::Temporal(TemporalExpr::Today)),
        _                   => ("test",  ValueRef::Literal("test".into())),
    }
}

fn is_record(field: &Field) -> bool {
    matches!(&field.field_type, FieldType::Record { .. })
}

fn non_record_fields(table: &Table) -> Vec<&Field> {
    table.fields.iter().filter(|f| !is_record(f)).collect()
}

fn record_fields(table: &Table) -> Vec<&Field> {
    table.fields.iter().filter(|f| is_record(f)).collect()
}

/// Pick the cmp_text for a given Comparator from the pool, cycling by index.
fn cmp_text_for(cmp: &Comparator, idx: usize) -> &'static str {
    let pool = comparator_texts();
    let entry = pool.iter().find(|(c, _)| c == cmp).unwrap();
    entry.1[idx % entry.1.len()]
}

// =============================================================================
// Pattern generators
// =============================================================================

fn gen_select_all(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;
    // (intent_text, entity_suffix)
    let patterns: &[(&str, &str)] = &[
        ("show me all", "s"),
        ("list every", ""),
        ("get all the", "s"),
        ("pull up", "s"),
        ("i need to see", "s"),
        ("gimme the", "s"),
        ("display all", "s"),
        ("grab every", ""),
        ("can you show", " records"),
        ("show", "s"),
    ];

    for &(intent_text, suffix) in patterns {
        let mut t = Tk::new();
        let ir = t.push(intent_text);
        t.sp();
        let es = t.0.len();
        t.lit(name);
        t.lit(suffix);
        let ee = t.0.len();

        out.push(Datum {
            nl: t.done(),
            surql: String::new(),
            semantics: Semantics {
                intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                projections: vec![], conditions: vec![], assignments: vec![], modifiers: vec![],
            },
            labels: vec![
                SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
            ],
            ir: None,
        });
    }

    // Entity-in-middle: "what {name}s are there"
    for &(prefix, suffix) in &[
        ("what", "s are there"),
        ("how about the", "s"),
        ("any", "s available"),
    ] {
        let mut t = Tk::new();
        let ir = t.push(prefix);
        t.sp();
        let es = t.0.len();
        t.lit(name);
        t.lit(suffix);
        let ee = es + name.len(); // entity is just the table name, not suffix

        out.push(Datum {
            nl: t.done(),
            surql: String::new(),
            semantics: Semantics {
                intent: IntentSpan { text: prefix.into(), start: ir.0, end: ir.1 },
                entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                projections: vec![], conditions: vec![], assignments: vec![], modifiers: vec![],
            },
            labels: vec![
                SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
            ],
            ir: None,
        });
    }
}

fn gen_select_projections(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;
    let fields = non_record_fields(table);
    if fields.is_empty() { return; }

    // Helper: build a projection datum from a field slice + phrasing
    let push_proj = |field_names: &[&str], intent_text: &str, sep: &str, out: &mut Vec<Datum>| {
        let mut t = Tk::new();
        let ir = t.push(intent_text);
        t.sp();
        let mut ranges = Vec::new();
        for (i, f) in field_names.iter().enumerate() {
            if i > 0 { t.lit(sep); }
            ranges.push(t.push(f));
        }
        t.lit(&format!(" from {name}s"));
        let es = t.0.rfind(name).unwrap();
        let ee = es + name.len();

        let projections: Vec<ProjectionSpan> = field_names.iter().zip(&ranges)
            .map(|(f, &(s, e))| ProjectionSpan { field_text: f.to_string(), start: s, end: e, fetch_index: None })
            .collect();
        let mut labels = vec![
            SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
            SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
        ];
        for (i, f) in field_names.iter().enumerate() {
            labels.push(SpanLabel {
                span_type: SpanType::Projection, span_index: i,
                target: QueryNode::Field { table: name.clone(), name: f.to_string() },
            });
        }

        out.push(Datum {
            nl: t.done(), surql: String::new(),
            semantics: Semantics {
                intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                projections, conditions: vec![], assignments: vec![], modifiers: vec![],
            },
            labels, ir: None,
        });
    };

    let intents = ["get the", "show me", "pull", "list", "i want the", "show"];
    let seps = [" and ", ", ", " ", ", "];

    // Singles: each field × each phrasing
    for field in &fields {
        for &intent in &intents[..3] {
            push_proj(&[&field.name], intent, "", out);
        }
    }

    // Pairs: all (i,j) combos × phrasing sample
    for i in 0..fields.len() {
        for j in (i + 1)..fields.len() {
            for (&intent, &sep) in intents.iter().zip(seps.iter().cycle()) {
                push_proj(&[&fields[i].name, &fields[j].name], intent, sep, out);
            }
        }
    }

    // Triples: sliding windows of 3
    if fields.len() >= 3 {
        for w in fields.windows(3) {
            for &intent in &intents[..3] {
                push_proj(&[&w[0].name, &w[1].name, &w[2].name], intent, ", ", out);
            }
        }
    }

    // Quads: first 4 fields if available
    if fields.len() >= 4 {
        let names: Vec<&str> = fields[..4].iter().map(|f| f.name.as_str()).collect();
        push_proj(&names, "get the", ", ", out);
        push_proj(&names, "show me", " and ", out);
    }
}

fn gen_select_conditions(table: &Table, cmp_pool: &[(Comparator, Vec<&str>)], out: &mut Vec<Datum>) {
    let name = &table.name;

    for field in non_record_fields(table) {
        let (val_text, val_ref) = sample_value(field);
        let cmps = compatible_cmps(field);

        for cmp in &cmps {
            let texts = &cmp_pool.iter().find(|(c, _)| c == cmp).unwrap().1;
            for &cmp_text in texts {
                // Vary the sentence structure
                let patterns: &[(&str, &str)] = &[
                    ("find", "s where"),
                    ("show", "s with"),
                    ("get", "s that have"),
                    ("which", "s have"),
                    ("look up", "s whose"),
                ];

                for &(intent_text, connector) in patterns {
                    let mut t = Tk::new();
                    let ir = t.push(intent_text);
                    t.sp();
                    let es = t.0.len();
                    t.lit(name);
                    t.lit(connector);
                    let ee = es + name.len();
                    t.sp();
                    let fr = t.push(&field.name);
                    t.sp();
                    let _cr = t.push(cmp_text);
                    t.sp();
                    t.lit(val_text);
                    let cond_start = fr.0;
                    let cond_end = t.0.len();

                    out.push(Datum {
                        nl: t.done(),
                        surql: String::new(),
                        semantics: Semantics {
                            intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                            entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                            projections: vec![],
                            conditions: vec![
                                ConditionSpan {
                                    field_text: field.name.clone(),
                                    comparator_text: cmp_text.into(),
                                    value: val_ref.clone(),
                                    start: cond_start,
                                    end: cond_end,
                                },
                            ],
                            assignments: vec![], modifiers: vec![],
                        },
                        labels: vec![
                            SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                            SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                            SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                            SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp.clone()) },
                        ],
                        ir: None,
                    });
                }
            }
        }
    }
}

fn gen_select_with_modifiers(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;
    let fields = non_record_fields(table);
    let rec_fields = record_fields(table);

    // OrderBy: "get {table}s {order_phrase} {field}"
    for field in &fields {
        for &(intent_text, order_phrase, desc) in &[
            ("get", "order by", None),
            ("show", "sorted by", None),
            ("list", "order by", Some(true)),
            ("get", "sort on", None),
        ] {
            let mut t = Tk::new();
            let ir = t.push(intent_text);
            t.sp();
            let es = t.0.len();
            t.lit(name);
            t.lit("s");
            let ee = es + name.len();
            t.sp();
            let mr = t.push(order_phrase);
            t.sp();
            let _fr = t.push(&field.name);
            if desc == Some(true) {
                t.lit(" desc");
            }

            out.push(Datum {
                nl: t.done(),
                surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                    projections: vec![],
                    conditions: vec![],
                    assignments: vec![],
                    modifiers: vec![
                        ModifierSpan {
                            text: order_phrase.into(),
                            argument: Some(field.name.clone()),
                            argument_value: None,
                            descending: desc,
                            start: mr.0,
                            end: mr.1,
                        },
                    ],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Modifier(ModifierKind::OrderBy) },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                ],
                ir: None,
            });
        }
    }

    // Limit: "get {table}s limit {n}"
    for &(intent_text, limit_phrase) in &[
        ("get", "limit 10"),
        ("show", "just the first 5"),
        ("list", "top 20"),
        ("get", "only 10"),
    ] {
        let mut t = Tk::new();
        let ir = t.push(intent_text);
        t.sp();
        let es = t.0.len();
        t.lit(name);
        t.lit("s");
        let ee = es + name.len();
        t.sp();
        let mr = t.push(limit_phrase);

        out.push(Datum {
            nl: t.done(),
            surql: String::new(),
            semantics: Semantics {
                intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                projections: vec![], conditions: vec![], assignments: vec![],
                modifiers: vec![
                    ModifierSpan {
                        text: limit_phrase.into(),
                        argument: None,
                        argument_value: Some(ValueRef::Literal("10".into())),
                        descending: None,
                        start: mr.0,
                        end: mr.1,
                    },
                ],
            },
            labels: vec![
                SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Modifier(ModifierKind::Limit) },
            ],
            ir: None,
        });
    }

    // Fetch: "get {table}s fetch {record_field}" — only for tables with record fields
    for field in &rec_fields {
        for &(intent_text, fetch_phrase) in &[
            ("get", "fetch"),
            ("show", "expand"),
            ("list", "include"),
        ] {
            let mut t = Tk::new();
            let ir = t.push(intent_text);
            t.sp();
            let es = t.0.len();
            t.lit(name);
            t.lit("s");
            let ee = es + name.len();
            t.sp();
            let mr = t.push(fetch_phrase);
            t.sp();
            t.lit(&field.name);

            out.push(Datum {
                nl: t.done(),
                surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                    projections: vec![], conditions: vec![], assignments: vec![],
                    modifiers: vec![
                        ModifierSpan {
                            text: fetch_phrase.into(),
                            argument: Some(field.name.clone()),
                            argument_value: None,
                            descending: None,
                            start: mr.0,
                            end: mr.1,
                        },
                    ],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Modifier(ModifierKind::Fetch) },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                ],
                ir: None,
            });
        }
    }

    // Combined: condition + modifier
    if let Some(field) = fields.first() {
        let (val_text, val_ref) = sample_value(field);
        let cmp = compatible_cmps(field)[0].clone();
        let cmp_text = cmp_text_for(&cmp, 0);

        if let Some(order_field) = fields.get(1) {
            // "get {table}s where {field} {cmp} {val} order by {order_field}"
            for &intent_text in &["get", "show", "find"] {
                let mut t = Tk::new();
                let ir = t.push(intent_text);
                t.sp();
                let es = t.0.len();
                t.lit(name);
                t.lit("s where");
                let ee = es + name.len();
                t.sp();
                let cond_start = t.0.len();
                let _fr = t.push(&field.name);
                t.sp();
                t.lit(cmp_text);
                t.sp();
                t.lit(val_text);
                let cond_end = t.0.len();
                t.sp();
                let mr = t.push("order by");
                t.sp();
                t.lit(&order_field.name);

                out.push(Datum {
                    nl: t.done(),
                    surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                        projections: vec![],
                        conditions: vec![
                            ConditionSpan {
                                field_text: field.name.clone(),
                                comparator_text: cmp_text.into(),
                                value: val_ref.clone(),
                                start: cond_start, end: cond_end,
                            },
                        ],
                        assignments: vec![],
                        modifiers: vec![
                            ModifierSpan {
                                text: "order by".into(),
                                argument: Some(order_field.name.clone()),
                                argument_value: None,
                                descending: None,
                                start: mr.0, end: mr.1,
                            },
                        ],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp.clone()) },
                        SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Modifier(ModifierKind::OrderBy) },
                        SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Field { table: name.clone(), name: order_field.name.clone() } },
                    ],
                    ir: None,
                });
            }

            // Reversed order: "get {table}s order by {order_field} where {field} {cmp} {val}"
            {
                let mut t = Tk::new();
                let ir = t.push("get");
                t.sp();
                let es = t.0.len();
                t.lit(name);
                t.lit("s");
                let ee = es + name.len();
                t.sp();
                let mr = t.push("order by");
                t.sp();
                t.lit(&order_field.name);
                t.lit(" where ");
                let cond_start = t.0.len();
                t.lit(&field.name);
                t.sp();
                t.lit(cmp_text);
                t.sp();
                t.lit(val_text);
                let cond_end = t.0.len();

                out.push(Datum {
                    nl: t.done(),
                    surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: "get".into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                        projections: vec![],
                        conditions: vec![
                            ConditionSpan {
                                field_text: field.name.clone(),
                                comparator_text: cmp_text.into(),
                                value: val_ref.clone(),
                                start: cond_start, end: cond_end,
                            },
                        ],
                        assignments: vec![],
                        modifiers: vec![
                            ModifierSpan {
                                text: "order by".into(),
                                argument: Some(order_field.name.clone()),
                                argument_value: None,
                                descending: None,
                                start: mr.0, end: mr.1,
                            },
                        ],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp.clone()) },
                        SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Modifier(ModifierKind::OrderBy) },
                        SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Field { table: name.clone(), name: order_field.name.clone() } },
                    ],
                    ir: None,
                });
            }
        }
    }
}

fn gen_create(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;
    let fields = non_record_fields(table);

    // Single-field explicit assignments: "create a {table} with {field} set to {value}"
    for field in &fields {
        let (val_text, val_ref) = sample_value(field);

        let patterns: &[(&str, &str, &str)] = &[
            ("create a", "with", "set to"),
            ("add a new", "where", "="),
            ("make a", "with", "as"),
            ("insert a", "with", "set to"),
            ("new", ":", ""),
        ];

        for &(intent_text, connector, setter) in patterns {
            let mut t = Tk::new();
            let ir = t.push(intent_text);
            t.sp();
            let es = t.0.len();
            t.lit(name);
            let ee = t.0.len();
            t.sp();
            t.lit(connector);
            t.sp();
            let assign_start = t.0.len();
            t.lit(&field.name);
            if !setter.is_empty() {
                t.sp();
                t.lit(setter);
            }
            t.sp();
            t.lit(val_text);
            let assign_end = t.0.len();

            out.push(Datum {
                nl: t.done(),
                surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                    projections: vec![], conditions: vec![],
                    assignments: vec![
                        AssignmentSpan {
                            field_text: Some(field.name.clone()),
                            value: val_ref.clone(),
                            start: assign_start, end: assign_end,
                        },
                    ],
                    modifiers: vec![],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Create) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                    SpanLabel { span_type: SpanType::Assignment, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                ],
                ir: None,
            });
        }
    }

    // Multi-field create: "create a {table} with {f1} set to {v1} and {f2} set to {v2}"
    if fields.len() >= 2 {
        let f1 = &fields[0];
        let f2 = &fields[1];
        let (v1_text, v1_ref) = sample_value(f1);
        let (v2_text, v2_ref) = sample_value(f2);

        for &intent_text in &["create a", "add a new", "make a"] {
            let mut t = Tk::new();
            let ir = t.push(intent_text);
            t.sp();
            let es = t.0.len();
            t.lit(name);
            let ee = t.0.len();
            t.lit(" with ");
            let a1s = t.0.len();
            t.lit(&f1.name);
            t.lit(" set to ");
            t.lit(v1_text);
            let a1e = t.0.len();
            t.lit(" and ");
            let a2s = t.0.len();
            t.lit(&f2.name);
            t.lit(" set to ");
            t.lit(v2_text);
            let a2e = t.0.len();

            out.push(Datum {
                nl: t.done(),
                surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                    projections: vec![], conditions: vec![],
                    assignments: vec![
                        AssignmentSpan { field_text: Some(f1.name.clone()), value: v1_ref.clone(), start: a1s, end: a1e },
                        AssignmentSpan { field_text: Some(f2.name.clone()), value: v2_ref.clone(), start: a2s, end: a2e },
                    ],
                    modifiers: vec![],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Create) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                    SpanLabel { span_type: SpanType::Assignment, span_index: 0, target: QueryNode::Field { table: name.clone(), name: f1.name.clone() } },
                    SpanLabel { span_type: SpanType::Assignment, span_index: 1, target: QueryNode::Field { table: name.clone(), name: f2.name.clone() } },
                ],
                ir: None,
            });
        }
    }

    // Object-expand create (field_text: None): "create a {table} with {slot}"
    for &(intent_text, phrasing) in &[
        ("create a", "with {1}"),
        ("add a new", "from {1}"),
        ("make a", "using {1}"),
    ] {
        let mut t = Tk::new();
        let ir = t.push(intent_text);
        t.sp();
        let es = t.0.len();
        t.lit(name);
        let ee = t.0.len();
        t.sp();
        let as_ = t.0.len();
        t.lit(phrasing);
        let ae = t.0.len();

        out.push(Datum {
            nl: t.done(),
            surql: String::new(),
            semantics: Semantics {
                intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                projections: vec![], conditions: vec![],
                assignments: vec![
                    AssignmentSpan { field_text: None, value: ValueRef::Slot(0), start: as_, end: ae },
                ],
                modifiers: vec![],
            },
            labels: vec![
                // No assignment field label — field_text is None, so no resolution
                SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Create) },
                SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
            ],
            ir: None,
        });
    }
}

fn gen_update(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;
    let fields = non_record_fields(table);
    if fields.len() < 2 { return; }

    // For each pair (assign_field, cond_field) where they differ
    for (i, assign_field) in fields.iter().enumerate() {
        for (j, cond_field) in fields.iter().enumerate() {
            if i == j { continue; }
            let (a_val_text, a_val_ref) = sample_value(assign_field);
            let (c_val_text, c_val_ref) = sample_value(cond_field);
            let cmps = compatible_cmps(cond_field);
            let cmp = &cmps[0];
            let cmp_text = cmp_text_for(cmp, 0);

            let patterns: &[(&str, bool)] = &[
                ("update", false),    // "update {table}s set {f} to {v} where {f2} {cmp} {v2}"
                ("change", false),    // "change {f} to {v} on {table}s where ..."
                ("modify", false),    // "modify {table}s set {f} to {v} where ..."
            ];

            for &(intent_text, _reversed) in patterns {
                let mut t = Tk::new();
                let ir = t.push(intent_text);
                t.sp();
                let es = t.0.len();
                t.lit(name);
                t.lit("s");
                let ee = es + name.len();
                t.lit(" set ");
                let as_ = t.0.len();
                t.lit(&assign_field.name);
                t.lit(" to ");
                t.lit(a_val_text);
                let ae = t.0.len();
                t.lit(" where ");
                let cs = t.0.len();
                t.lit(&cond_field.name);
                t.sp();
                t.lit(cmp_text);
                t.sp();
                t.lit(c_val_text);
                let ce = t.0.len();

                out.push(Datum {
                    nl: t.done(),
                    surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                        projections: vec![],
                        conditions: vec![
                            ConditionSpan {
                                field_text: cond_field.name.clone(),
                                comparator_text: cmp_text.into(),
                                value: c_val_ref.clone(),
                                start: cs, end: ce,
                            },
                        ],
                        assignments: vec![
                            AssignmentSpan {
                                field_text: Some(assign_field.name.clone()),
                                value: a_val_ref.clone(),
                                start: as_, end: ae,
                            },
                        ],
                        modifiers: vec![],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Update) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Assignment, span_index: 0, target: QueryNode::Field { table: name.clone(), name: assign_field.name.clone() } },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: name.clone(), name: cond_field.name.clone() } },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp.clone()) },
                    ],
                    ir: None,
                });
            }
        }
    }
}

fn gen_delete(table: &Table, cmp_pool: &[(Comparator, Vec<&str>)], out: &mut Vec<Datum>) {
    let name = &table.name;

    for field in non_record_fields(table) {
        let (val_text, val_ref) = sample_value(field);
        let cmps = compatible_cmps(field);

        for cmp in &cmps {
            let texts = &cmp_pool.iter().find(|(c, _)| c == cmp).unwrap().1;
            // Use first text variant for each comparator (conditions generator covers all variants)
            let cmp_text = texts[0];

            let patterns: &[(&str, &str)] = &[
                ("delete", "s where"),
                ("remove", "s that have"),
                ("get rid of", "s where"),
                ("drop any", " with"),
            ];

            for &(intent_text, connector) in patterns {
                let mut t = Tk::new();
                let ir = t.push(intent_text);
                t.sp();
                let es = t.0.len();
                t.lit(name);
                t.lit(connector);
                let ee = es + name.len();
                t.sp();
                let cs = t.0.len();
                t.lit(&field.name);
                t.sp();
                t.lit(cmp_text);
                t.sp();
                t.lit(val_text);
                let ce = t.0.len();

                out.push(Datum {
                    nl: t.done(),
                    surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                        projections: vec![], assignments: vec![], modifiers: vec![],
                        conditions: vec![
                            ConditionSpan {
                                field_text: field.name.clone(),
                                comparator_text: cmp_text.into(),
                                value: val_ref.clone(),
                                start: cs, end: ce,
                            },
                        ],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Delete) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp.clone()) },
                    ],
                    ir: None,
                });
            }
        }
    }
}

// =============================================================================
// Record ID patterns — "get user abc123", "show me post {1}", "update task xyz"
// =============================================================================

fn gen_record_id(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;

    // Sample record IDs: literal strings and slots
    let ids: Vec<(&str, ValueRef)> = vec![
        ("abc123",  ValueRef::Literal("abc123".into())),
        ("xyz789",  ValueRef::Literal("xyz789".into())),
        ("{1}",     ValueRef::Slot(0)),
        ("{2}",     ValueRef::Slot(1)),
    ];

    for (id_text, id_ref) in &ids {
        // SELECT with record ID
        // Natural phrasings: "get user abc123", "show me the post {1}", "look up task xyz789"
        for &(intent_text, mid) in &[
            ("get", " "),
            ("show me", " "),
            ("look up", " "),
            ("show", " the "),
            ("pull up", " "),
        ] {
            let mut t = Tk::new();
            let ir = t.push(intent_text);
            t.lit(mid);
            let es = t.0.len();
            t.lit(name);
            let ee = t.0.len();
            t.sp();
            t.lit(id_text);

            out.push(Datum {
                nl: t.done(), surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: Some(id_ref.clone()) },
                    projections: vec![], conditions: vec![], assignments: vec![], modifiers: vec![],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                ],
                ir: None,
            });
        }

        // SELECT with record ID + projections: "get the name and email of user abc123"
        let fields = non_record_fields(table);
        if fields.len() >= 2 {
            let f1 = &fields[0].name;
            let f2 = &fields[1].name;
            for &intent_text in &["get the", "show me the"] {
                let mut t = Tk::new();
                let ir = t.push(intent_text);
                t.sp();
                let f1r = t.push(f1);
                t.lit(" and ");
                let f2r = t.push(f2);
                t.lit(&format!(" of {name} "));
                let es = t.0.rfind(name).unwrap();
                let ee = es + name.len();
                t.lit(id_text);

                out.push(Datum {
                    nl: t.done(), surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: Some(id_ref.clone()) },
                        projections: vec![
                            ProjectionSpan { field_text: f1.clone(), start: f1r.0, end: f1r.1, fetch_index: None },
                            ProjectionSpan { field_text: f2.clone(), start: f2r.0, end: f2r.1, fetch_index: None },
                        ],
                        conditions: vec![], assignments: vec![], modifiers: vec![],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Projection, span_index: 0, target: QueryNode::Field { table: name.clone(), name: f1.clone() } },
                        SpanLabel { span_type: SpanType::Projection, span_index: 1, target: QueryNode::Field { table: name.clone(), name: f2.clone() } },
                    ],
                    ir: None,
                });
            }
        }

        // UPDATE with record ID: "update user abc123 set name to test"
        if let Some(field) = fields.first() {
            let (val_text, val_ref) = sample_value(field);
            for &intent_text in &["update", "change", "modify"] {
                let mut t = Tk::new();
                let ir = t.push(intent_text);
                t.sp();
                let es = t.0.len();
                t.lit(name);
                let ee = t.0.len();
                t.sp();
                t.lit(id_text);
                t.lit(" set ");
                let as_ = t.0.len();
                t.lit(&field.name);
                t.lit(" to ");
                t.lit(val_text);
                let ae = t.0.len();

                out.push(Datum {
                    nl: t.done(), surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: Some(id_ref.clone()) },
                        projections: vec![], conditions: vec![],
                        assignments: vec![
                            AssignmentSpan { field_text: Some(field.name.clone()), value: val_ref.clone(), start: as_, end: ae },
                        ],
                        modifiers: vec![],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Update) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Assignment, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                    ],
                    ir: None,
                });
            }
        }

        // DELETE with record ID: "delete user abc123", "remove post {1}"
        for &intent_text in &["delete", "remove", "drop"] {
            let mut t = Tk::new();
            let ir = t.push(intent_text);
            t.sp();
            let es = t.0.len();
            t.lit(name);
            let ee = t.0.len();
            t.sp();
            t.lit(id_text);

            out.push(Datum {
                nl: t.done(), surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: Some(id_ref.clone()) },
                    projections: vec![], conditions: vec![], assignments: vec![], modifiers: vec![],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Delete) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                ],
                ir: None,
            });
        }
    }
}

// =============================================================================
// Combined generators — SELECT + projections + conditions/modifiers
// =============================================================================

/// "get the {f1} and {f2} from {table}s where {cf} {cmp} {val}"
fn gen_select_proj_cond(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;
    let fields = non_record_fields(table);
    if fields.len() < 2 { return; }

    let intents = ["get the", "show me the", "find the", "pull the", "list the"];

    // For each projection field × condition field (different fields)
    for pf in &fields {
        for cf in &fields {
            if pf.name == cf.name { continue; }
            let cmps = compatible_cmps(cf);
            let cmp = &cmps[0];
            let cmp_text = cmp_text_for(cmp, 0);
            let (val_text, val_ref) = sample_value(cf);

            for &intent_text in &intents {
                let mut t = Tk::new();
                let ir = t.push(intent_text);
                t.sp();
                let pr = t.push(&pf.name);
                t.lit(&format!(" from {name}s where "));
                let es = t.0.find(name).unwrap();
                let ee = es + name.len();
                let cs = t.0.len();
                t.lit(&cf.name);
                t.sp();
                t.lit(cmp_text);
                t.sp();
                t.lit(val_text);
                let ce = t.0.len();

                out.push(Datum {
                    nl: t.done(), surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                        projections: vec![
                            ProjectionSpan { field_text: pf.name.clone(), start: pr.0, end: pr.1, fetch_index: None },
                        ],
                        conditions: vec![
                            ConditionSpan { field_text: cf.name.clone(), comparator_text: cmp_text.into(), value: val_ref.clone(), start: cs, end: ce },
                        ],
                        assignments: vec![], modifiers: vec![],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Projection, span_index: 0, target: QueryNode::Field { table: name.clone(), name: pf.name.clone() } },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: name.clone(), name: cf.name.clone() } },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp.clone()) },
                    ],
                    ir: None,
                });
            }
        }
    }

    // Two projections + condition: "get {f1} and {f2} from {table}s where {cf} {cmp} {val}"
    for i in 0..fields.len().min(5) {
        for j in (i+1)..fields.len().min(6) {
            for cf in &fields {
                if cf.name == fields[i].name || cf.name == fields[j].name { continue; }
                let cmp = &compatible_cmps(cf)[0];
                let cmp_text = cmp_text_for(cmp, 0);
                let (val_text, val_ref) = sample_value(cf);

                let mut t = Tk::new();
                let ir = t.push("get the");
                t.sp();
                let p1r = t.push(&fields[i].name);
                t.lit(" and ");
                let p2r = t.push(&fields[j].name);
                t.lit(&format!(" from {name}s where "));
                let es = t.0.find(name).unwrap();
                let ee = es + name.len();
                let cs = t.0.len();
                t.lit(&cf.name);
                t.sp();
                t.lit(cmp_text);
                t.sp();
                t.lit(val_text);
                let ce = t.0.len();

                out.push(Datum {
                    nl: t.done(), surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: "get the".into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                        projections: vec![
                            ProjectionSpan { field_text: fields[i].name.clone(), start: p1r.0, end: p1r.1, fetch_index: None },
                            ProjectionSpan { field_text: fields[j].name.clone(), start: p2r.0, end: p2r.1, fetch_index: None },
                        ],
                        conditions: vec![
                            ConditionSpan { field_text: cf.name.clone(), comparator_text: cmp_text.into(), value: val_ref.clone(), start: cs, end: ce },
                        ],
                        assignments: vec![], modifiers: vec![],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Projection, span_index: 0, target: QueryNode::Field { table: name.clone(), name: fields[i].name.clone() } },
                        SpanLabel { span_type: SpanType::Projection, span_index: 1, target: QueryNode::Field { table: name.clone(), name: fields[j].name.clone() } },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: name.clone(), name: cf.name.clone() } },
                        SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp.clone()) },
                    ],
                    ir: None,
                });
            }
        }
    }
}

/// "get {f1} from {table}s order by {f2}" — projections + modifiers
fn gen_select_proj_modifier(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;
    let fields = non_record_fields(table);
    if fields.len() < 2 { return; }

    let combos: &[(&str, &str)] = &[
        ("get the", "order by"),
        ("show me the", "sorted by"),
        ("list the", "order by"),
        ("find the", "sort on"),
    ];

    for pf in &fields {
        for of in &fields {
            if pf.name == of.name { continue; }

            for &(intent_text, order_phrase) in combos {
                let mut t = Tk::new();
                let ir = t.push(intent_text);
                t.sp();
                let pr = t.push(&pf.name);
                t.lit(&format!(" from {name}s "));
                let es = t.0.find(name).unwrap();
                let ee = es + name.len();
                let mr = t.push(order_phrase);
                t.sp();
                t.lit(&of.name);

                out.push(Datum {
                    nl: t.done(), surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                        projections: vec![
                            ProjectionSpan { field_text: pf.name.clone(), start: pr.0, end: pr.1, fetch_index: None },
                        ],
                        conditions: vec![], assignments: vec![],
                        modifiers: vec![
                            ModifierSpan { text: order_phrase.into(), argument: Some(of.name.clone()), argument_value: None, descending: None, start: mr.0, end: mr.1 },
                        ],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Projection, span_index: 0, target: QueryNode::Field { table: name.clone(), name: pf.name.clone() } },
                        SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Modifier(ModifierKind::OrderBy) },
                        SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Field { table: name.clone(), name: of.name.clone() } },
                    ],
                    ir: None,
                });
            }
        }
    }
}

/// Multi-field CREATE with all field pairs (not just first 2)
fn gen_create_multi(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;
    let fields = non_record_fields(table);
    if fields.len() < 2 { return; }

    let patterns: &[(&str, &str)] = &[
        ("create a", "set to"),
        ("add a new", "="),
        ("make a", "as"),
        ("insert a", "set to"),
    ];

    for i in 0..fields.len() {
        for j in (i+1)..fields.len() {
            let f1 = &fields[i];
            let f2 = &fields[j];
            let (v1_text, v1_ref) = sample_value(f1);
            let (v2_text, v2_ref) = sample_value(f2);

            for &(intent_text, setter) in patterns {
                let mut t = Tk::new();
                let ir = t.push(intent_text);
                t.sp();
                let es = t.0.len();
                t.lit(name);
                let ee = t.0.len();
                t.lit(" with ");
                let a1s = t.0.len();
                t.lit(&f1.name);
                t.sp();
                t.lit(setter);
                t.sp();
                t.lit(v1_text);
                let a1e = t.0.len();
                t.lit(" and ");
                let a2s = t.0.len();
                t.lit(&f2.name);
                t.sp();
                t.lit(setter);
                t.sp();
                t.lit(v2_text);
                let a2e = t.0.len();

                out.push(Datum {
                    nl: t.done(), surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                        projections: vec![], conditions: vec![],
                        assignments: vec![
                            AssignmentSpan { field_text: Some(f1.name.clone()), value: v1_ref.clone(), start: a1s, end: a1e },
                            AssignmentSpan { field_text: Some(f2.name.clone()), value: v2_ref.clone(), start: a2s, end: a2e },
                        ],
                        modifiers: vec![],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Create) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Assignment, span_index: 0, target: QueryNode::Field { table: name.clone(), name: f1.name.clone() } },
                        SpanLabel { span_type: SpanType::Assignment, span_index: 1, target: QueryNode::Field { table: name.clone(), name: f2.name.clone() } },
                    ],
                    ir: None,
                });
            }
        }
    }

    // Triple-field creates: sliding windows of 3
    if fields.len() >= 3 {
        for w in fields.windows(3) {
            let (v0_text, v0_ref) = sample_value(&w[0]);
            let (v1_text, v1_ref) = sample_value(&w[1]);
            let (v2_text, v2_ref) = sample_value(&w[2]);

            for &intent_text in &["create a", "add a new"] {
                let mut t = Tk::new();
                let ir = t.push(intent_text);
                t.sp();
                let es = t.0.len();
                t.lit(name);
                let ee = t.0.len();
                t.lit(" with ");
                let a0s = t.0.len();
                t.lit(&w[0].name); t.lit(" set to "); t.lit(v0_text);
                let a0e = t.0.len();
                t.lit(", ");
                let a1s = t.0.len();
                t.lit(&w[1].name); t.lit(" set to "); t.lit(v1_text);
                let a1e = t.0.len();
                t.lit(" and ");
                let a2s = t.0.len();
                t.lit(&w[2].name); t.lit(" set to "); t.lit(v2_text);
                let a2e = t.0.len();

                out.push(Datum {
                    nl: t.done(), surql: String::new(),
                    semantics: Semantics {
                        intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                        entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                        projections: vec![], conditions: vec![],
                        assignments: vec![
                            AssignmentSpan { field_text: Some(w[0].name.clone()), value: v0_ref.clone(), start: a0s, end: a0e },
                            AssignmentSpan { field_text: Some(w[1].name.clone()), value: v1_ref.clone(), start: a1s, end: a1e },
                            AssignmentSpan { field_text: Some(w[2].name.clone()), value: v2_ref.clone(), start: a2s, end: a2e },
                        ],
                        modifiers: vec![],
                    },
                    labels: vec![
                        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Create) },
                        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                        SpanLabel { span_type: SpanType::Assignment, span_index: 0, target: QueryNode::Field { table: name.clone(), name: w[0].name.clone() } },
                        SpanLabel { span_type: SpanType::Assignment, span_index: 1, target: QueryNode::Field { table: name.clone(), name: w[1].name.clone() } },
                        SpanLabel { span_type: SpanType::Assignment, span_index: 2, target: QueryNode::Field { table: name.clone(), name: w[2].name.clone() } },
                    ],
                    ir: None,
                });
            }
        }
    }
}

/// More modifier phrasings: OrderBy with all fields × more variety, dual modifiers
fn gen_more_modifiers(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;
    let fields = non_record_fields(table);

    // OrderBy with desc/asc variations for all fields
    let order_combos: &[(&str, &str, Option<bool>)] = &[
        ("get", "order by", None),
        ("show", "sorted by", None),
        ("list", "sort on", None),
        ("find", "order by", Some(true)),
        ("get", "sorted by", Some(true)),
        ("show", "order by", Some(false)),
        ("get", "in order of", None),
        ("list", "arranged by", None),
    ];

    for field in &fields {
        for &(intent_text, order_phrase, desc) in order_combos {
            let mut t = Tk::new();
            let ir = t.push(intent_text);
            t.sp();
            let es = t.0.len();
            t.lit(name); t.lit("s");
            let ee = es + name.len();
            t.sp();
            let mr = t.push(order_phrase);
            t.sp();
            t.lit(&field.name);
            if desc == Some(true) { t.lit(" desc"); }
            if desc == Some(false) { t.lit(" asc"); }

            out.push(Datum {
                nl: t.done(), surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                    projections: vec![], conditions: vec![], assignments: vec![],
                    modifiers: vec![
                        ModifierSpan { text: order_phrase.into(), argument: Some(field.name.clone()), argument_value: None, descending: desc, start: mr.0, end: mr.1 },
                    ],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Modifier(ModifierKind::OrderBy) },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                ],
                ir: None,
            });
        }
    }

    // OrderBy + Limit combined: "get {table}s order by {field} limit 10"
    for field in &fields {
        for &(intent_text, order_phrase, limit_text) in &[
            ("get", "order by", "limit 10"),
            ("show", "sorted by", "top 5"),
            ("list", "order by", "only 20"),
        ] {
            let mut t = Tk::new();
            let ir = t.push(intent_text);
            t.sp();
            let es = t.0.len();
            t.lit(name); t.lit("s ");
            let ee = es + name.len();
            let mr1 = t.push(order_phrase);
            t.sp();
            t.lit(&field.name);
            t.sp();
            let mr2 = t.push(limit_text);

            out.push(Datum {
                nl: t.done(), surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                    projections: vec![], conditions: vec![], assignments: vec![],
                    modifiers: vec![
                        ModifierSpan { text: order_phrase.into(), argument: Some(field.name.clone()), argument_value: None, descending: None, start: mr1.0, end: mr1.1 },
                        ModifierSpan { text: limit_text.into(), argument: None, argument_value: Some(ValueRef::Literal("10".into())), descending: None, start: mr2.0, end: mr2.1 },
                    ],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Modifier(ModifierKind::OrderBy) },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 1, target: QueryNode::Modifier(ModifierKind::Limit) },
                ],
                ir: None,
            });
        }
    }

    // Fetch with more phrasings
    let rec_fields = record_fields(table);
    for field in &rec_fields {
        for &(intent_text, fetch_phrase) in &[
            ("get", "and fetch"),
            ("show", "with"),
            ("list", "and include"),
            ("get", "along with"),
            ("show", "and expand"),
            ("find", "fetch"),
        ] {
            let mut t = Tk::new();
            let ir = t.push(intent_text);
            t.sp();
            let es = t.0.len();
            t.lit(name); t.lit("s ");
            let ee = es + name.len();
            let mr = t.push(fetch_phrase);
            t.sp();
            t.lit(&field.name);

            out.push(Datum {
                nl: t.done(), surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: intent_text.into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                    projections: vec![], conditions: vec![], assignments: vec![],
                    modifiers: vec![
                        ModifierSpan { text: fetch_phrase.into(), argument: Some(field.name.clone()), argument_value: None, descending: None, start: mr.0, end: mr.1 },
                    ],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Select) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Modifier(ModifierKind::Fetch) },
                    SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Field { table: name.clone(), name: field.name.clone() } },
                ],
                ir: None,
            });
        }
    }
}

/// UPDATE with more phrasing variety and multi-field updates
fn gen_update_expanded(table: &Table, out: &mut Vec<Datum>) {
    let name = &table.name;
    let fields = non_record_fields(table);
    if fields.len() < 2 { return; }

    // "change" pattern with reversed order: "change {f} to {v} on {table}s where ..."
    for (i, af) in fields.iter().enumerate() {
        for (j, cf) in fields.iter().enumerate() {
            if i == j { continue; }
            let (a_val, a_ref) = sample_value(af);
            let (c_val, c_ref) = sample_value(cf);
            let cmp = &compatible_cmps(cf)[0];
            let cmp_text = cmp_text_for(cmp, 0);

            let mut t = Tk::new();
            let ir = t.push("change");
            t.sp();
            let as_ = t.0.len();
            t.lit(&af.name); t.lit(" to "); t.lit(a_val);
            let ae = t.0.len();
            t.lit(" on ");
            let es = t.0.len();
            t.lit(name); t.lit("s");
            let ee = es + name.len();
            t.lit(" where ");
            let cs = t.0.len();
            t.lit(&cf.name); t.sp(); t.lit(cmp_text); t.sp(); t.lit(c_val);
            let ce = t.0.len();

            out.push(Datum {
                nl: t.done(), surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: "change".into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                    projections: vec![],
                    conditions: vec![
                        ConditionSpan { field_text: cf.name.clone(), comparator_text: cmp_text.into(), value: c_ref, start: cs, end: ce },
                    ],
                    assignments: vec![
                        AssignmentSpan { field_text: Some(af.name.clone()), value: a_ref, start: as_, end: ae },
                    ],
                    modifiers: vec![],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Update) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                    SpanLabel { span_type: SpanType::Assignment, span_index: 0, target: QueryNode::Field { table: name.clone(), name: af.name.clone() } },
                    SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: name.clone(), name: cf.name.clone() } },
                    SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp.clone()) },
                ],
                ir: None,
            });
        }
    }

    // "for {table}s with {cond}, set {f} to {v}" — yet another word order
    for af in &fields {
        if let Some(cf) = fields.iter().find(|f| f.name != af.name) {
            let (a_val, a_ref) = sample_value(af);
            let (c_val, c_ref) = sample_value(cf);
            let cmp = &compatible_cmps(cf)[0];
            let cmp_text = cmp_text_for(cmp, 0);

            let mut t = Tk::new();
            let ir = t.push("for");
            t.sp();
            let es = t.0.len();
            t.lit(name); t.lit("s");
            let ee = es + name.len();
            t.lit(" with ");
            let cs = t.0.len();
            t.lit(&cf.name); t.sp(); t.lit(cmp_text); t.sp(); t.lit(c_val);
            let ce = t.0.len();
            t.lit(", set ");
            let as_ = t.0.len();
            t.lit(&af.name); t.lit(" to "); t.lit(a_val);
            let ae = t.0.len();

            out.push(Datum {
                nl: t.done(), surql: String::new(),
                semantics: Semantics {
                    intent: IntentSpan { text: "for".into(), start: ir.0, end: ir.1 },
                    entity: EntitySpan { text: name.clone(), start: es, end: ee, record_id: None },
                    projections: vec![],
                    conditions: vec![
                        ConditionSpan { field_text: cf.name.clone(), comparator_text: cmp_text.into(), value: c_ref, start: cs, end: ce },
                    ],
                    assignments: vec![
                        AssignmentSpan { field_text: Some(af.name.clone()), value: a_ref, start: as_, end: ae },
                    ],
                    modifiers: vec![],
                },
                labels: vec![
                    SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(Intent::Update) },
                    SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(name.clone()) },
                    SpanLabel { span_type: SpanType::Assignment, span_index: 0, target: QueryNode::Field { table: name.clone(), name: af.name.clone() } },
                    SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: name.clone(), name: cf.name.clone() } },
                    SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp.clone()) },
                ],
                ir: None,
            });
        }
    }
}

// =============================================================================
// Top-level generation
// =============================================================================

impl Datum {
    pub fn generate(schema: &Schema) -> Vec<Datum> {
        let mut data = Vec::new();
        let cmp_pool = comparator_texts();

        for table in &schema.tables {
            gen_select_all(table, &mut data);
            gen_select_projections(table, &mut data);
            gen_select_conditions(table, &cmp_pool, &mut data);
            gen_select_with_modifiers(table, &mut data);
            gen_select_proj_cond(table, &mut data);
            gen_select_proj_modifier(table, &mut data);
            gen_create(table, &mut data);
            gen_create_multi(table, &mut data);
            gen_update(table, &mut data);
            gen_update_expanded(table, &mut data);
            gen_delete(table, &cmp_pool, &mut data);
            gen_record_id(table, &mut data);
            gen_more_modifiers(table, &mut data);
        }

        data
    }

    pub fn print_stats(data: &[Datum]) {
        println!("=== Dataset Statistics ===");
        println!("Total datums: {}", data.len());

        // Intent distribution — group by target Operation label
        let mut intents: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for d in data {
            let op = d.labels.iter()
                .find(|l| l.span_type == SpanType::Intent)
                .map(|l| format!("{:?}", l.target))
                .unwrap_or_else(|| "?".into());
            *intents.entry(op).or_default() += 1;
        }
        println!("\nIntent (by target):");
        let mut intents: Vec<_> = intents.into_iter().collect();
        intents.sort_by(|a, b| b.1.cmp(&a.1));
        for (k, v) in &intents {
            println!("  {:<30} {:>5} ({:.1}%)", k, v, *v as f32 / data.len() as f32 * 100.0);
        }

        // Span counts
        let mut n_proj = 0usize;
        let mut n_cond = 0usize;
        let mut n_assign = 0usize;
        let mut n_mod = 0usize;
        for d in data {
            n_proj += d.semantics.projections.len();
            n_cond += d.semantics.conditions.len();
            n_assign += d.semantics.assignments.len();
            n_mod += d.semantics.modifiers.len();
        }
        println!("\nSpan totals:");
        println!("  projections:  {} (avg {:.1}/datum)", n_proj, n_proj as f32 / data.len() as f32);
        println!("  conditions:   {} (avg {:.1}/datum)", n_cond, n_cond as f32 / data.len() as f32);
        println!("  assignments:  {} (avg {:.1}/datum)", n_assign, n_assign as f32 / data.len() as f32);
        println!("  modifiers:    {} (avg {:.1}/datum)", n_mod, n_mod as f32 / data.len() as f32);

        // Label distribution by head
        let mut head_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for d in data {
            for l in &d.labels {
                let key = match (&l.span_type, &l.target) {
                    (SpanType::Intent, _) => "intent",
                    (SpanType::Entity, _) => "entity",
                    (SpanType::Projection, _) => "projection",
                    (SpanType::Condition, hyphae::query::QueryNode::Field { .. }) => "cond_field",
                    (SpanType::Condition, hyphae::query::QueryNode::Comparator(_)) => "cond_cmp",
                    (SpanType::Assignment, _) => "assignment",
                    (SpanType::Modifier, hyphae::query::QueryNode::Modifier(_)) => "mod_type",
                    (SpanType::Modifier, hyphae::query::QueryNode::Field { .. }) => "mod_field",
                    _ => "other",
                };
                *head_counts.entry(key.to_string()).or_default() += 1;
            }
        }
        println!("\nLabels by head:");
        let mut head_counts: Vec<_> = head_counts.into_iter().collect();
        head_counts.sort_by(|a, b| b.1.cmp(&a.1));
        let total_labels: usize = head_counts.iter().map(|(_, v)| v).sum();
        for (k, v) in &head_counts {
            println!("  {:<15} {:>5} ({:.1}%)", k, v, *v as f32 / total_labels as f32 * 100.0);
        }
        println!("  total labels:  {}", total_labels);

        // Token length distribution (whitespace tokens)
        let lengths: Vec<usize> = data.iter().map(|d| d.nl.split_whitespace().count()).collect();
        let mut sorted = lengths.clone();
        sorted.sort();
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let median = sorted[sorted.len() / 2];
        let mean = sorted.iter().sum::<usize>() as f32 / sorted.len() as f32;
        println!("\nToken lengths (whitespace):");
        println!("  min={} max={} median={} mean={:.1}", min, max, median, mean);

        // Histogram buckets
        let buckets = [2, 4, 6, 8, 10, 12, 14, 16, 20, 30];
        println!("  distribution:");
        let mut prev = 0;
        for &b in &buckets {
            let count = sorted.iter().filter(|&&l| l > prev && l <= b).count();
            if count > 0 {
                println!("    {:>2}-{:<2} tokens: {:>5} ({:.1}%)", prev + 1, b, count, count as f32 / data.len() as f32 * 100.0);
            }
            prev = b;
        }
        let remainder = sorted.iter().filter(|&&l| l > prev).count();
        if remainder > 0 {
            println!("    {}>   tokens: {:>5} ({:.1}%)", prev, remainder, remainder as f32 / data.len() as f32 * 100.0);
        }
    }
}
