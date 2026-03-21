// septa — semantic parsing of natural language prompts

pub mod model;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Semantics {
    pub intent: IntentSpan,
    pub entity: EntitySpan,
    pub projections: Vec<ProjectionSpan>,
    pub conditions: Vec<ConditionSpan>,
    pub assignments: Vec<AssignmentSpan>,
    pub modifiers: Vec<ModifierSpan>,
}

impl Semantics {
    pub fn parse(_text: &str) -> Self {
        todo!()
    }
}

/// Raw verb/phrase — GNN resolves to an Operation node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentSpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Primary table reference (one per query). record_id holds the :id qualifier when present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub record_id: Option<ValueRef>,
}

/// Field to project in SELECT.
/// fetch_index indexes into Semantics.modifiers when this field lives on a linked table;
/// the NLP learns this binding from NL co-reference. None = field on primary table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionSpan {
    pub field_text: String,
    pub start: usize,
    pub end: usize,
    pub fetch_index: Option<usize>,
}

/// Condition predicate. comparator_text is raw NL ("over", "less than", "equals") —
/// GNN resolves it to a Comparator node.
/// Sub-span ranges allow separate BiGRU pooling for field vs comparator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionSpan {
    pub field_text: String,
    pub comparator_text: String,
    pub value: ValueRef,
    pub start: usize,
    pub end: usize,
    /// Character range of the field name sub-span (e.g. "status" in "status equals done").
    pub field_start: usize,
    pub field_end: usize,
    /// Character range of the comparator sub-span (e.g. "equals" in "status equals done").
    pub cmp_start: usize,
    pub cmp_end: usize,
}

/// Field=value write. field_text is None when the slot value is an object to be
/// expanded field-by-field at render time using schema types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentSpan {
    pub field_text: Option<String>,
    pub value: ValueRef,
    pub start: usize,
    pub end: usize,
    /// Character range of the field name sub-span, when field_text is Some.
    pub field_start: usize,
    pub field_end: usize,
}

/// Generic modifier span — GNN resolves the type (OrderBy/Limit/Fetch) via ModifierToType edges
/// and the target field via multi-hop through ModifierToField schema edges.
/// Sub-span ranges allow separate BiGRU pooling for type keyword vs field argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifierSpan {
    pub text: String,                     // "order by", "fetch", "limit"
    pub argument: Option<String>,         // field text (for OrderBy/Fetch) or raw limit text
    pub argument_value: Option<ValueRef>, // value form (for Limit with slots/temporals/literals)
    pub descending: Option<bool>,         // NLP detects "desc"/"asc" keywords; None if unknown
    pub start: usize,
    pub end: usize,
    /// Character range of the field argument sub-span (e.g. "created_at" in "order by created_at").
    /// Only meaningful when argument is Some.
    pub arg_start: usize,
    pub arg_end: usize,
}

/// Value on the right-hand side of a condition or assignment.
/// Kept opaque through NLP and GNN; substituted or normalised at render time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueRef {
    Literal(String),
    Slot(usize), // {1} → Slot(0),  {2} → Slot(1)  — deterministic pre-processing
    Temporal(TemporalExpr),
}

/// Relative datetime expressions — normalised to SurrealQL at render time.
/// e.g. LastWeek → time::now() - 7d,  Today → time::floor(time::now(), 1d)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalExpr {
    Today,
    Yesterday,
    DaysAgo(u32),
    WeeksAgo(u32),
    MonthsAgo(u32),
    Iso(String),
}

/// Resolved operation type — target for IntentSpan bilinear resolution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    Select,
    Create,
    Update,
    Delete,
}

/// Resolved comparator — target for ConditionSpan comparator bilinear resolution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Comparator {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    Contains,
}
