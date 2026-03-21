// eval.rs — Hand-crafted evaluation datums with natural phrasing.
//
// Each prompt is individually written to test generalization beyond training templates.
// Uses varied grammar, contractions, casual/formal mix, and unseen word orders.
// The Tk builder computes span byte positions automatically.

use crate::{Datum, SpanLabel, SpanType, Tk};
use hyphae::query::{ModifierKind, QueryNode};
use septa::{
    AssignmentSpan, Comparator, ConditionSpan, EntitySpan,
    IntentSpan, ModifierSpan, ProjectionSpan, Semantics,
    TemporalExpr, ValueRef,
};

// =============================================================================
// Datum builders — thin wrappers around Tk for each query shape
// =============================================================================

/// SELECT * FROM {table}
fn sel(parts: &[Part]) -> Datum {
    let (nl, spans) = layout(parts);
    let intent_text = spans.intent.unwrap();
    let table = spans.entity.unwrap();
    Datum {
        nl: nl.clone(), surql: String::new(),
        semantics: Semantics {
            intent: IntentSpan { text: intent_text.0.clone(), start: intent_text.1, end: intent_text.2 },
            entity: EntitySpan { text: table.0.clone(), start: table.1, end: table.2, record_id: None },
            projections: vec![], conditions: vec![], assignments: vec![], modifiers: vec![],
        },
        labels: vec![
            SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(septa::Intent::Select) },
            SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(table.0.clone()) },
        ],
        ir: None,
    }
}

/// SELECT {fields} FROM {table}
fn sel_proj(parts: &[Part]) -> Datum {
    let (nl, spans) = layout(parts);
    let intent_text = spans.intent.unwrap();
    let table = spans.entity.unwrap();
    let mut projs = Vec::new();
    let mut labels = vec![
        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(septa::Intent::Select) },
        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(table.0.clone()) },
    ];
    for (i, p) in spans.projections.iter().enumerate() {
        projs.push(ProjectionSpan { field_text: p.0.clone(), start: p.1, end: p.2, fetch_index: None });
        labels.push(SpanLabel { span_type: SpanType::Projection, span_index: i, target: QueryNode::Field { table: table.0.clone(), name: p.0.clone() } });
    }
    Datum {
        nl, surql: String::new(),
        semantics: Semantics {
            intent: IntentSpan { text: intent_text.0.clone(), start: intent_text.1, end: intent_text.2 },
            entity: EntitySpan { text: table.0.clone(), start: table.1, end: table.2, record_id: None },
            projections: projs, conditions: vec![], assignments: vec![], modifiers: vec![],
        },
        labels, ir: None,
    }
}

/// SELECT * FROM {table} WHERE {field} {cmp} {val}
fn sel_cond(parts: &[Part], cmp: Comparator, val: ValueRef) -> Datum {
    let (nl, spans) = layout(parts);
    let intent_text = spans.intent.unwrap();
    let table = spans.entity.unwrap();
    let cf = spans.cond_field.unwrap();
    let ct = spans.cond_cmp.unwrap();
    let cv = spans.cond_val.unwrap();
    Datum {
        nl, surql: String::new(),
        semantics: Semantics {
            intent: IntentSpan { text: intent_text.0.clone(), start: intent_text.1, end: intent_text.2 },
            entity: EntitySpan { text: table.0.clone(), start: table.1, end: table.2, record_id: None },
            projections: vec![],
            conditions: vec![ConditionSpan {
                field_text: cf.0.clone(), comparator_text: ct.0.clone(),
                value: val, start: cf.1, end: cv.2,
            }],
            assignments: vec![], modifiers: vec![],
        },
        labels: vec![
            SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(septa::Intent::Select) },
            SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(table.0.clone()) },
            SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: table.0.clone(), name: cf.0.clone() } },
            SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp) },
        ],
        ir: None,
    }
}

/// SELECT {fields} FROM {table} WHERE {field} {cmp} {val}
fn sel_proj_cond(parts: &[Part], cmp: Comparator, val: ValueRef) -> Datum {
    let (nl, spans) = layout(parts);
    let intent_text = spans.intent.unwrap();
    let table = spans.entity.unwrap();
    let cf = spans.cond_field.unwrap();
    let ct = spans.cond_cmp.unwrap();
    let cv = spans.cond_val.unwrap();
    let mut projs = Vec::new();
    let mut labels = vec![
        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(septa::Intent::Select) },
        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(table.0.clone()) },
    ];
    for (i, p) in spans.projections.iter().enumerate() {
        projs.push(ProjectionSpan { field_text: p.0.clone(), start: p.1, end: p.2, fetch_index: None });
        labels.push(SpanLabel { span_type: SpanType::Projection, span_index: i, target: QueryNode::Field { table: table.0.clone(), name: p.0.clone() } });
    }
    labels.push(SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: table.0.clone(), name: cf.0.clone() } });
    labels.push(SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp) });
    Datum {
        nl, surql: String::new(),
        semantics: Semantics {
            intent: IntentSpan { text: intent_text.0.clone(), start: intent_text.1, end: intent_text.2 },
            entity: EntitySpan { text: table.0.clone(), start: table.1, end: table.2, record_id: None },
            projections: projs,
            conditions: vec![ConditionSpan {
                field_text: cf.0.clone(), comparator_text: ct.0.clone(),
                value: val, start: cf.1, end: cv.2,
            }],
            assignments: vec![], modifiers: vec![],
        },
        labels, ir: None,
    }
}

/// SELECT * FROM {table} ORDER BY {field} [DESC]
fn sel_order(parts: &[Part], desc: bool) -> Datum {
    let (nl, spans) = layout(parts);
    let intent_text = spans.intent.unwrap();
    let table = spans.entity.unwrap();
    let mt = spans.mod_text.unwrap();
    let mf = spans.mod_field.unwrap();
    Datum {
        nl: nl.clone(), surql: String::new(),
        semantics: Semantics {
            intent: IntentSpan { text: intent_text.0.clone(), start: intent_text.1, end: intent_text.2 },
            entity: EntitySpan { text: table.0.clone(), start: table.1, end: table.2, record_id: None },
            projections: vec![], conditions: vec![], assignments: vec![],
            modifiers: vec![ModifierSpan {
                text: mt.0.clone(), argument: Some(mf.0.clone()),
                argument_value: None, descending: Some(desc), start: mt.1, end: mf.2,
            }],
        },
        labels: vec![
            SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(septa::Intent::Select) },
            SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(table.0.clone()) },
            SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Modifier(ModifierKind::OrderBy) },
            SpanLabel { span_type: SpanType::Modifier, span_index: 0, target: QueryNode::Field { table: table.0.clone(), name: mf.0.clone() } },
        ],
        ir: None,
    }
}

/// SELECT * FROM {table}:{record_id}
fn sel_record(parts: &[Part], rid: &str) -> Datum {
    let (nl, spans) = layout(parts);
    let intent_text = spans.intent.unwrap();
    let table = spans.entity.unwrap();
    Datum {
        nl, surql: String::new(),
        semantics: Semantics {
            intent: IntentSpan { text: intent_text.0.clone(), start: intent_text.1, end: intent_text.2 },
            entity: EntitySpan { text: table.0.clone(), start: table.1, end: table.2, record_id: Some(ValueRef::Literal(rid.into())) },
            projections: vec![], conditions: vec![], assignments: vec![], modifiers: vec![],
        },
        labels: vec![
            SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(septa::Intent::Select) },
            SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(table.0.clone()) },
        ],
        ir: None,
    }
}

/// CREATE {table} SET {field} = {val}, ...
fn create(parts: &[Part], assigns: &[(&str, ValueRef)]) -> Datum {
    let (nl, spans) = layout(parts);
    let intent_text = spans.intent.unwrap();
    let table = spans.entity.unwrap();
    let mut assignment_spans = Vec::new();
    let mut labels = vec![
        SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(septa::Intent::Create) },
        SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(table.0.clone()) },
    ];
    for (i, af) in spans.assign_fields.iter().enumerate() {
        let av = &spans.assign_vals[i];
        assignment_spans.push(AssignmentSpan {
            field_text: Some(af.0.clone()), value: assigns[i].1.clone(),
            start: af.1, end: av.2,
        });
        labels.push(SpanLabel { span_type: SpanType::Assignment, span_index: i, target: QueryNode::Field { table: table.0.clone(), name: af.0.clone() } });
    }
    Datum {
        nl, surql: String::new(),
        semantics: Semantics {
            intent: IntentSpan { text: intent_text.0.clone(), start: intent_text.1, end: intent_text.2 },
            entity: EntitySpan { text: table.0.clone(), start: table.1, end: table.2, record_id: None },
            projections: vec![], conditions: vec![], assignments: assignment_spans, modifiers: vec![],
        },
        labels, ir: None,
    }
}

/// UPDATE {table} SET {field} = {val} WHERE {cond_field} {cmp} {cond_val}
fn update(parts: &[Part], asgn_val: ValueRef, cmp: Comparator, cond_val: ValueRef) -> Datum {
    let (nl, spans) = layout(parts);
    let intent_text = spans.intent.unwrap();
    let table = spans.entity.unwrap();
    let af = &spans.assign_fields[0];
    let av = &spans.assign_vals[0];
    let cf = spans.cond_field.unwrap();
    let ct = spans.cond_cmp.unwrap();
    let cv = spans.cond_val.unwrap();
    Datum {
        nl, surql: String::new(),
        semantics: Semantics {
            intent: IntentSpan { text: intent_text.0.clone(), start: intent_text.1, end: intent_text.2 },
            entity: EntitySpan { text: table.0.clone(), start: table.1, end: table.2, record_id: None },
            projections: vec![],
            conditions: vec![ConditionSpan {
                field_text: cf.0.clone(), comparator_text: ct.0.clone(),
                value: cond_val, start: cf.1, end: cv.2,
            }],
            assignments: vec![AssignmentSpan {
                field_text: Some(af.0.clone()), value: asgn_val,
                start: af.1, end: av.2,
            }],
            modifiers: vec![],
        },
        labels: vec![
            SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(septa::Intent::Update) },
            SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(table.0.clone()) },
            SpanLabel { span_type: SpanType::Assignment, span_index: 0, target: QueryNode::Field { table: table.0.clone(), name: af.0.clone() } },
            SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: table.0.clone(), name: cf.0.clone() } },
            SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp) },
        ],
        ir: None,
    }
}

/// DELETE {table} WHERE {field} {cmp} {val}
fn delete(parts: &[Part], cmp: Comparator, val: ValueRef) -> Datum {
    let (nl, spans) = layout(parts);
    let intent_text = spans.intent.unwrap();
    let table = spans.entity.unwrap();
    let cf = spans.cond_field.unwrap();
    let ct = spans.cond_cmp.unwrap();
    let cv = spans.cond_val.unwrap();
    Datum {
        nl, surql: String::new(),
        semantics: Semantics {
            intent: IntentSpan { text: intent_text.0.clone(), start: intent_text.1, end: intent_text.2 },
            entity: EntitySpan { text: table.0.clone(), start: table.1, end: table.2, record_id: None },
            projections: vec![], assignments: vec![],
            conditions: vec![ConditionSpan {
                field_text: cf.0.clone(), comparator_text: ct.0.clone(),
                value: val, start: cf.1, end: cv.2,
            }],
            modifiers: vec![],
        },
        labels: vec![
            SpanLabel { span_type: SpanType::Intent, span_index: 0, target: QueryNode::Operation(septa::Intent::Delete) },
            SpanLabel { span_type: SpanType::Entity, span_index: 0, target: QueryNode::Table(table.0.clone()) },
            SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Field { table: table.0.clone(), name: cf.0.clone() } },
            SpanLabel { span_type: SpanType::Condition, span_index: 0, target: QueryNode::Comparator(cmp) },
        ],
        ir: None,
    }
}

// =============================================================================
// Layout engine — declarative parts → positioned spans
// =============================================================================

/// A piece of the NL string, tagged with its semantic role.
enum Part<'a> {
    /// Literal filler text (not a span)
    Lit(&'a str),
    /// Intent verb/phrase
    Intent(&'a str),
    /// Entity (table name) — may have trailing suffix like "s" that's not part of the table name
    Entity(&'a str, &'a str), // (table_name, suffix)
    /// Projection field name
    Proj(&'a str),
    /// Condition field
    CondField(&'a str),
    /// Condition comparator text
    CondCmp(&'a str),
    /// Condition value text
    CondVal(&'a str),
    /// Modifier keyword (e.g. "sorted by", "order by")
    ModText(&'a str),
    /// Modifier field argument
    ModField(&'a str),
    /// Assignment field name
    AsgnField(&'a str),
    /// Assignment value text
    AsgnVal(&'a str),
}

/// Extracted span positions: (text, start, end)
type Span = (String, usize, usize);

struct Spans {
    intent: Option<Span>,
    entity: Option<Span>,
    projections: Vec<Span>,
    cond_field: Option<Span>,
    cond_cmp: Option<Span>,
    cond_val: Option<Span>,
    mod_text: Option<Span>,
    mod_field: Option<Span>,
    assign_fields: Vec<Span>,
    assign_vals: Vec<Span>,
}

fn layout(parts: &[Part]) -> (String, Spans) {
    let mut t = Tk::new();
    let mut spans = Spans {
        intent: None, entity: None, projections: vec![],
        cond_field: None, cond_cmp: None, cond_val: None,
        mod_text: None, mod_field: None,
        assign_fields: vec![], assign_vals: vec![],
    };
    for (i, part) in parts.iter().enumerate() {
        if i > 0 { t.sp(); }
        match part {
            Part::Lit(s) => { t.lit(s); }
            Part::Intent(s) => {
                let (start, end) = t.push(s);
                spans.intent = Some((s.to_string(), start, end));
            }
            Part::Entity(name, suffix) => {
                let start = t.0.len();
                t.lit(name);
                let end = t.0.len();
                t.lit(suffix);
                spans.entity = Some((name.to_string(), start, end));
            }
            Part::Proj(s) => {
                let (start, end) = t.push(s);
                spans.projections.push((s.to_string(), start, end));
            }
            Part::CondField(s) => {
                let (start, end) = t.push(s);
                spans.cond_field = Some((s.to_string(), start, end));
            }
            Part::CondCmp(s) => {
                let (start, end) = t.push(s);
                spans.cond_cmp = Some((s.to_string(), start, end));
            }
            Part::CondVal(s) => {
                let (start, end) = t.push(s);
                spans.cond_val = Some((s.to_string(), start, end));
            }
            Part::ModText(s) => {
                let (start, end) = t.push(s);
                spans.mod_text = Some((s.to_string(), start, end));
            }
            Part::ModField(s) => {
                let (start, end) = t.push(s);
                spans.mod_field = Some((s.to_string(), start, end));
            }
            Part::AsgnField(s) => {
                let (start, end) = t.push(s);
                spans.assign_fields.push((s.to_string(), start, end));
            }
            Part::AsgnVal(s) => {
                let (start, end) = t.push(s);
                spans.assign_vals.push((s.to_string(), start, end));
            }
        }
    }
    (t.done(), spans)
}

use Part::*;

// =============================================================================
// The 100 hand-crafted eval prompts
// =============================================================================

pub fn build_eval_set() -> Vec<Datum> {
    let mut out = Vec::new();

    // ── SELECT ALL (1-10) ───────────────────────────────────────────────
    out.push(sel(&[Intent("what"), Entity("user", "s"), Lit("do we currently have")]));
    out.push(sel(&[Intent("i'd like to see all"), Entity("task", "s"), Lit("please")]));
    out.push(sel(&[Intent("could you pull up the"), Entity("project", ""), Lit("list")]));
    out.push(sel(&[Intent("just give me every"), Entity("comment", ""), Lit("in the system")]));
    out.push(sel(&[Intent("let me browse"), Lit("the"), Entity("post", "s")]));
    out.push(sel(&[Intent("need a dump of all"), Entity("tag", "s")]));
    out.push(sel(&[Intent("yo show me"), Lit("the"), Entity("user", ""), Lit("table")]));
    out.push(sel(&[Intent("what are all the"), Entity("task", "s"), Lit("we have right now")]));
    out.push(sel(&[Intent("everything in"), Entity("project", "s")]));
    out.push(sel(&[Intent("go ahead and list"), Lit("all"), Entity("comment", "s")]));

    // ── SELECT WITH PROJECTIONS (11-22) ─────────────────────────────────
    out.push(sel_proj(&[Intent("i want"), Lit("the"), Proj("name"), Lit("and"), Proj("email"), Lit("of"), Entity("user", "s")]));
    out.push(sel_proj(&[Intent("what's"), Lit("the"), Proj("title"), Lit("and"), Proj("priority"), Lit("on each"), Entity("task", "")]));
    out.push(sel_proj(&[Intent("give me"), Lit("the"), Proj("status"), Lit("and"), Proj("description"), Lit("for every"), Entity("project", "")]));
    out.push(sel_proj(&[Intent("just the"), Proj("body"), Lit("from"), Entity("comment", "s"), Lit("please")]));
    out.push(sel_proj(&[Intent("pull"), Proj("title"), Lit("and"), Proj("tags"), Lit("off the"), Entity("post", "s")]));
    out.push(sel_proj(&[Intent("i only need"), Lit("the"), Proj("name"), Lit("and"), Proj("color"), Lit("of"), Entity("tag", "s")]));
    out.push(sel_proj(&[Intent("return"), Lit("only"), Proj("email"), Lit("and"), Proj("active"), Lit("for"), Entity("user", "s")]));
    out.push(sel_proj(&[Intent("can i get"), Lit("the"), Proj("due_date"), Lit("and"), Proj("status"), Lit("of"), Entity("task", "s")]));
    out.push(sel_proj(&[Intent("let me see"), Proj("name"), Lit("and"), Proj("created"), Lit("from"), Entity("project", "s")]));
    out.push(sel_proj(&[Intent("what is"), Lit("the"), Proj("title"), Lit("and"), Proj("body"), Lit("for"), Entity("post", "s")]));
    out.push(sel_proj(&[Intent("show me just"), Lit("the"), Proj("age"), Lit("from"), Entity("user", "s")]));
    out.push(sel_proj(&[Intent("only grab"), Lit("the"), Proj("description"), Lit("from"), Entity("task", "s")]));

    // ── SELECT WITH CONDITIONS (23-40) ──────────────────────────────────
    out.push(sel_cond(&[Intent("find"), Entity("user", "s"), Lit("whose"), CondField("age"), CondCmp("is above"), CondVal("25")],
        Comparator::Gt, ValueRef::Literal("25".into())));
    out.push(sel_cond(&[Intent("show me"), Entity("task", "s"), Lit("where the"), CondField("priority"), CondCmp("exceeds"), CondVal("3")],
        Comparator::Gt, ValueRef::Literal("3".into())));
    out.push(sel_cond(&[Intent("i need"), Entity("project", "s"), Lit("where"), CondField("status"), CondCmp("is"), CondVal("active")],
        Comparator::Eq, ValueRef::Literal("active".into())));
    out.push(sel_cond(&[Intent("which"), Entity("comment", "s"), Lit("have a"), CondField("body"), Lit("that"), CondCmp("includes"), CondVal("bug")],
        Comparator::Contains, ValueRef::Literal("bug".into())));
    out.push(sel_cond(&[Intent("only show"), Entity("user", "s"), Lit("where"), CondField("active"), CondCmp("is not"), CondVal("true")],
        Comparator::Neq, ValueRef::Literal("true".into())));
    out.push(sel_cond(&[Intent("look for"), Entity("task", "s"), Lit("whose"), CondField("due_date"), CondCmp("is less than"), CondVal("today")],
        Comparator::Lt, ValueRef::Temporal(TemporalExpr::Today)));
    out.push(sel_cond(&[Intent("give me"), Entity("post", "s"), Lit("where"), CondField("title"), CondCmp("matches"), CondVal("weekly roundup")],
        Comparator::Eq, ValueRef::Literal("weekly roundup".into())));
    out.push(sel_cond(&[Intent("show me"), Entity("tag", "s"), Lit("whose"), CondField("color"), CondCmp("is not"), CondVal("red")],
        Comparator::Neq, ValueRef::Literal("red".into())));
    out.push(sel_cond(&[Intent("i want"), Entity("user", "s"), Lit("where"), CondField("email"), CondCmp("contains"), CondVal("gmail")],
        Comparator::Contains, ValueRef::Literal("gmail".into())));
    out.push(sel_cond(&[Intent("search for"), Entity("task", "s"), Lit("with"), CondField("status"), CondCmp("equals"), CondVal("done")],
        Comparator::Eq, ValueRef::Literal("done".into())));
    out.push(sel_cond(&[Intent("get me"), Entity("project", "s"), Lit("where"), CondField("created"), CondCmp("is above"), CondVal("yesterday")],
        Comparator::Gt, ValueRef::Temporal(TemporalExpr::Yesterday)));
    out.push(sel_cond(&[Intent("find"), Entity("comment", "s"), Lit("where"), CondField("created"), CondCmp("is under"), CondVal("today")],
        Comparator::Lt, ValueRef::Temporal(TemporalExpr::Today)));
    out.push(sel_cond(&[Intent("list"), Entity("user", "s"), Lit("with"), CondField("age"), CondCmp("is at most"), CondVal("30")],
        Comparator::Lte, ValueRef::Literal("30".into())));
    out.push(sel_cond(&[Intent("grab"), Entity("task", "s"), Lit("where"), CondField("priority"), CondCmp("is at least"), CondVal("5")],
        Comparator::Gte, ValueRef::Literal("5".into())));
    out.push(sel_cond(&[Intent("look up"), Entity("post", "s"), Lit("where"), CondField("body"), CondCmp("has"), CondVal("update")],
        Comparator::Contains, ValueRef::Literal("update".into())));
    out.push(sel_cond(&[Intent("fetch"), Entity("user", "s"), Lit("whose"), CondField("name"), CondCmp("equals"), CondVal("alice")],
        Comparator::Eq, ValueRef::Literal("alice".into())));
    out.push(sel_cond(&[Intent("retrieve"), Entity("task", "s"), Lit("where"), CondField("title"), CondCmp("contains"), CondVal("deploy")],
        Comparator::Contains, ValueRef::Literal("deploy".into())));
    out.push(sel_cond(&[Intent("which"), Entity("project", "s"), Lit("have"), CondField("name"), Lit("that"), CondCmp("differs from"), CondVal("archive")],
        Comparator::Neq, ValueRef::Literal("archive".into())));

    // ── SELECT WITH ORDER BY (41-52) ────────────────────────────────────
    out.push(sel_order(&[Intent("show me"), Entity("user", "s"), ModText("sorted by"), ModField("name")], false));
    out.push(sel_order(&[Intent("get"), Entity("task", "s"), ModText("in order of"), ModField("priority")], false));
    out.push(sel_order(&[Intent("list"), Entity("project", "s"), ModText("ordered by"), ModField("created"), Lit("desc")], true));
    out.push(sel_order(&[Intent("pull up"), Entity("post", "s"), ModText("sort on"), ModField("created"), Lit("descending")], true));
    out.push(sel_order(&[Intent("show"), Entity("comment", "s"), ModText("sorted by"), ModField("created")], false));
    out.push(sel_order(&[Intent("get"), Entity("tag", "s"), ModText("order by"), ModField("name")], false));
    out.push(sel_order(&[Intent("give me"), Entity("user", "s"), ModText("sorted by"), ModField("age"), Lit("descending")], true));
    out.push(sel_order(&[Intent("show me"), Entity("task", "s"), ModText("order by"), ModField("due_date")], false));
    out.push(sel_order(&[Intent("list"), Entity("project", "s"), ModText("sort on"), ModField("name")], false));
    out.push(sel_order(&[Intent("pull"), Entity("user", "s"), ModText("sorted by"), ModField("created"), Lit("desc")], true));
    out.push(sel_order(&[Intent("i need"), Entity("task", "s"), ModText("sorted by"), ModField("status")], false));
    out.push(sel_order(&[Intent("get"), Entity("post", "s"), ModText("ordered by"), ModField("title")], false));

    // ── SELECT PROJ + CONDITION (53-62) ─────────────────────────────────
    out.push(sel_proj_cond(&[Intent("show me"), Lit("the"), Proj("name"), Lit("and"), Proj("email"), Lit("of"), Entity("user", "s"), Lit("where"), CondField("active"), CondCmp("equals"), CondVal("true")],
        Comparator::Eq, ValueRef::Literal("true".into())));
    out.push(sel_proj_cond(&[Intent("what's"), Lit("the"), Proj("title"), Lit("and"), Proj("status"), Lit("for"), Entity("task", "s"), Lit("where"), CondField("priority"), CondCmp("is above"), CondVal("3")],
        Comparator::Gt, ValueRef::Literal("3".into())));
    out.push(sel_proj_cond(&[Intent("get"), Proj("description"), Lit("and"), Proj("status"), Lit("from"), Entity("project", "s"), Lit("where"), CondField("name"), CondCmp("equals"), CondVal("mycelium")],
        Comparator::Eq, ValueRef::Literal("mycelium".into())));
    out.push(sel_proj_cond(&[Intent("i need"), Lit("the"), Proj("body"), Lit("from"), Entity("comment", "s"), Lit("where"), CondField("created"), CondCmp("is above"), CondVal("yesterday")],
        Comparator::Gt, ValueRef::Temporal(TemporalExpr::Yesterday)));
    out.push(sel_proj_cond(&[Intent("pull"), Proj("title"), Lit("and"), Proj("tags"), Lit("from"), Entity("post", "s"), Lit("where"), CondField("title"), CondCmp("contains"), CondVal("release")],
        Comparator::Contains, ValueRef::Literal("release".into())));
    out.push(sel_proj_cond(&[Intent("show me"), Lit("the"), Proj("name"), Lit("and"), Proj("age"), Lit("of"), Entity("user", "s"), Lit("where"), CondField("age"), CondCmp("is at least"), CondVal("18")],
        Comparator::Gte, ValueRef::Literal("18".into())));
    out.push(sel_proj_cond(&[Intent("give me"), Lit("the"), Proj("title"), Lit("and"), Proj("due_date"), Lit("from"), Entity("task", "s"), Lit("where"), CondField("status"), CondCmp("is not"), CondVal("done")],
        Comparator::Neq, ValueRef::Literal("done".into())));
    out.push(sel_proj_cond(&[Intent("list"), Lit("the"), Proj("email"), Lit("of"), Entity("user", "s"), Lit("where"), CondField("name"), CondCmp("equals"), CondVal("bob")],
        Comparator::Eq, ValueRef::Literal("bob".into())));
    out.push(sel_proj_cond(&[Intent("what is"), Lit("the"), Proj("description"), Lit("of"), Entity("project", "s"), Lit("where"), CondField("status"), CondCmp("is"), CondVal("planning")],
        Comparator::Eq, ValueRef::Literal("planning".into())));
    out.push(sel_proj_cond(&[Intent("find"), Lit("the"), Proj("title"), Lit("and"), Proj("body"), Lit("from"), Entity("post", "s"), Lit("where"), CondField("created"), CondCmp("is under"), CondVal("today")],
        Comparator::Lt, ValueRef::Temporal(TemporalExpr::Today)));

    // ── CREATE (63-72) ──────────────────────────────────────────────────
    out.push(create(&[Intent("register a new"), Entity("user", ""), Lit("with"), AsgnField("name"), AsgnVal("alice"), Lit("and"), AsgnField("email"), AsgnVal("alice@example.com")],
        &[("name", ValueRef::Literal("alice".into())), ("email", ValueRef::Literal("alice@example.com".into()))]));
    out.push(create(&[Intent("spin up a"), Entity("task", ""), Lit("with"), AsgnField("title"), AsgnVal("fix login"), Lit("and"), AsgnField("priority"), AsgnVal("5")],
        &[("title", ValueRef::Literal("fix login".into())), ("priority", ValueRef::Literal("5".into()))]));
    out.push(create(&[Intent("make a"), Entity("project", ""), Lit("with"), AsgnField("name"), AsgnVal("alpha"), Lit("and"), AsgnField("status"), AsgnVal("active")],
        &[("name", ValueRef::Literal("alpha".into())), ("status", ValueRef::Literal("active".into()))]));
    out.push(create(&[Intent("add a"), Entity("comment", ""), Lit("with"), AsgnField("body"), AsgnVal("looks good")],
        &[("body", ValueRef::Literal("looks good".into()))]));
    out.push(create(&[Intent("write a new"), Entity("post", ""), Lit("with"), AsgnField("title"), AsgnVal("hello world"), Lit("and"), AsgnField("body"), AsgnVal("first post")],
        &[("title", ValueRef::Literal("hello world".into())), ("body", ValueRef::Literal("first post".into()))]));
    out.push(create(&[Intent("create a"), Entity("tag", ""), Lit("with"), AsgnField("name"), AsgnVal("urgent"), Lit("and"), AsgnField("color"), AsgnVal("red")],
        &[("name", ValueRef::Literal("urgent".into())), ("color", ValueRef::Literal("red".into()))]));
    out.push(create(&[Intent("add a new"), Entity("user", ""), Lit("with"), AsgnField("name"), AsgnVal("charlie"), Lit("and"), AsgnField("active"), AsgnVal("true")],
        &[("name", ValueRef::Literal("charlie".into())), ("active", ValueRef::Literal("true".into()))]));
    out.push(create(&[Intent("please create a"), Entity("task", ""), Lit("with"), AsgnField("title"), AsgnVal("review PR"), Lit("and"), AsgnField("status"), AsgnVal("open")],
        &[("title", ValueRef::Literal("review PR".into())), ("status", ValueRef::Literal("open".into()))]));
    out.push(create(&[Intent("insert a"), Entity("project", ""), Lit("with"), AsgnField("name"), AsgnVal("beta")],
        &[("name", ValueRef::Literal("beta".into()))]));
    out.push(create(&[Intent("new"), Entity("tag", ""), Lit("with"), AsgnField("name"), AsgnVal("wontfix"), Lit("and"), AsgnField("color"), AsgnVal("grey")],
        &[("name", ValueRef::Literal("wontfix".into())), ("color", ValueRef::Literal("grey".into()))]));

    // ── UPDATE (73-82) ──────────────────────────────────────────────────
    out.push(update(&[Intent("change"), Entity("user", ""), AsgnField("name"), Lit("to"), AsgnVal("admin"), Lit("where"), CondField("email"), CondCmp("is"), CondVal("old@test.com")],
        ValueRef::Literal("admin".into()), Comparator::Eq, ValueRef::Literal("old@test.com".into())));
    out.push(update(&[Intent("update"), Entity("task", "s"), Lit("set"), AsgnField("status"), Lit("to"), AsgnVal("done"), Lit("where"), CondField("title"), CondCmp("equals"), CondVal("fix login")],
        ValueRef::Literal("done".into()), Comparator::Eq, ValueRef::Literal("fix login".into())));
    out.push(update(&[Intent("set"), Entity("project", ""), AsgnField("status"), Lit("to"), AsgnVal("archived"), Lit("where"), CondField("name"), CondCmp("equals"), CondVal("old project")],
        ValueRef::Literal("archived".into()), Comparator::Eq, ValueRef::Literal("old project".into())));
    out.push(update(&[Intent("modify"), Entity("tag", ""), AsgnField("color"), Lit("to"), AsgnVal("blue"), Lit("where"), CondField("name"), CondCmp("is"), CondVal("urgent")],
        ValueRef::Literal("blue".into()), Comparator::Eq, ValueRef::Literal("urgent".into())));
    out.push(update(&[Intent("change"), Entity("user", ""), AsgnField("active"), Lit("to"), AsgnVal("false"), Lit("where"), CondField("name"), CondCmp("equals"), CondVal("charlie")],
        ValueRef::Literal("false".into()), Comparator::Eq, ValueRef::Literal("charlie".into())));
    out.push(update(&[Intent("update"), Entity("task", "s"), Lit("set"), AsgnField("priority"), Lit("to"), AsgnVal("1"), Lit("where"), CondField("status"), CondCmp("equals"), CondVal("done")],
        ValueRef::Literal("1".into()), Comparator::Eq, ValueRef::Literal("done".into())));
    out.push(update(&[Intent("set"), Entity("post", ""), AsgnField("title"), Lit("to"), AsgnVal("updated title"), Lit("where"), CondField("body"), CondCmp("contains"), CondVal("draft")],
        ValueRef::Literal("updated title".into()), Comparator::Contains, ValueRef::Literal("draft".into())));
    out.push(update(&[Intent("for"), Entity("task", "s"), Lit("with"), CondField("priority"), CondCmp("is above"), CondVal("3"), Lit("set"), AsgnField("description"), Lit("to"), AsgnVal("needs review")],
        ValueRef::Literal("needs review".into()), Comparator::Gt, ValueRef::Literal("3".into())));
    out.push(update(&[Intent("change"), Entity("project", ""), AsgnField("description"), Lit("to"), AsgnVal("new plan"), Lit("where"), CondField("status"), CondCmp("is"), CondVal("active")],
        ValueRef::Literal("new plan".into()), Comparator::Eq, ValueRef::Literal("active".into())));
    out.push(update(&[Intent("update"), Entity("user", "s"), Lit("set"), AsgnField("email"), Lit("to"), AsgnVal("new@mail.com"), Lit("where"), CondField("name"), CondCmp("equals"), CondVal("alice")],
        ValueRef::Literal("new@mail.com".into()), Comparator::Eq, ValueRef::Literal("alice".into())));

    // ── DELETE (83-90) ──────────────────────────────────────────────────
    out.push(delete(&[Intent("remove"), Entity("user", "s"), Lit("where"), CondField("active"), CondCmp("is not"), CondVal("true")],
        Comparator::Neq, ValueRef::Literal("true".into())));
    out.push(delete(&[Intent("delete"), Entity("task", "s"), Lit("where"), CondField("status"), CondCmp("equals"), CondVal("cancelled")],
        Comparator::Eq, ValueRef::Literal("cancelled".into())));
    out.push(delete(&[Intent("purge"), Entity("comment", "s"), Lit("where"), CondField("created"), CondCmp("is under"), CondVal("yesterday")],
        Comparator::Lt, ValueRef::Temporal(TemporalExpr::Yesterday)));
    out.push(delete(&[Intent("get rid of"), Entity("post", "s"), Lit("where"), CondField("title"), CondCmp("equals"), CondVal("spam")],
        Comparator::Eq, ValueRef::Literal("spam".into())));
    out.push(delete(&[Intent("wipe"), Entity("tag", "s"), Lit("where"), CondField("color"), CondCmp("is"), CondVal("grey")],
        Comparator::Eq, ValueRef::Literal("grey".into())));
    out.push(delete(&[Intent("drop"), Entity("project", "s"), Lit("where"), CondField("status"), CondCmp("equals"), CondVal("dead")],
        Comparator::Eq, ValueRef::Literal("dead".into())));
    out.push(delete(&[Intent("nuke"), Entity("user", "s"), Lit("where"), CondField("age"), CondCmp("is under"), CondVal("13")],
        Comparator::Lt, ValueRef::Literal("13".into())));
    out.push(delete(&[Intent("clear out"), Entity("task", "s"), Lit("where"), CondField("priority"), CondCmp("is at most"), CondVal("0")],
        Comparator::Lte, ValueRef::Literal("0".into())));

    // ── RECORD ID (91-96) ───────────────────────────────────────────────
    out.push(sel_record(&[Intent("look up"), Entity("user", ""), Lit("usr_42")], "usr_42"));
    out.push(sel_record(&[Intent("get me"), Entity("task", ""), Lit("t-100")], "t-100"));
    out.push(sel_record(&[Intent("pull up"), Entity("project", ""), Lit("proj_alpha")], "proj_alpha"));
    out.push(sel_record(&[Intent("show me"), Entity("comment", ""), Lit("c99")], "c99"));
    out.push(sel_record(&[Intent("i need"), Entity("post", ""), Lit("hello-world")], "hello-world"));
    out.push(sel_record(&[Intent("open"), Entity("tag", ""), Lit("urgent")], "urgent"));

    // ── COMPOUND — proj + cond (97-100) ─────────────────────────────────
    out.push(sel_proj_cond(&[Intent("what are"), Lit("the"), Proj("name"), Lit("and"), Proj("email"), Lit("for"), Entity("user", "s"), Lit("where"), CondField("age"), CondCmp("is more than"), CondVal("21")],
        Comparator::Gt, ValueRef::Literal("21".into())));
    out.push(sel_proj_cond(&[Intent("grab"), Lit("the"), Proj("title"), Lit("and"), Proj("priority"), Lit("from"), Entity("task", "s"), Lit("where"), CondField("due_date"), CondCmp("is less than"), CondVal("today")],
        Comparator::Lt, ValueRef::Temporal(TemporalExpr::Today)));
    out.push(sel_proj_cond(&[Intent("give me"), Lit("the"), Proj("status"), Lit("and"), Proj("created"), Lit("from"), Entity("project", "s"), Lit("where"), CondField("name"), CondCmp("has"), CondVal("beta")],
        Comparator::Contains, ValueRef::Literal("beta".into())));
    out.push(sel_proj_cond(&[Intent("list"), Lit("the"), Proj("name"), Lit("and"), Proj("color"), Lit("of"), Entity("tag", "s"), Lit("where"), CondField("name"), CondCmp("is not"), CondVal("default")],
        Comparator::Neq, ValueRef::Literal("default".into())));

    out
}
