// septa — semantic parsing of natural language prompts

pub mod model;

pub struct Semantics {
    pub intent: Intent,
    pub entities: Vec<EntitySpan>,
    pub projections: Vec<ProjectionSpan>,
    pub conditions: Vec<ConditionSpan>,
    pub assignments: Vec<AssignmentSpan>,
    pub modifiers: Vec<ModifierSpan>,
}

impl Semantics {
    pub fn parse(text: &str) -> Self {
        todo!()
    }
}

/// Noun phrase referring to a table or record.
pub struct EntitySpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Field name to return (empty projections = SELECT *).
pub struct ProjectionSpan {
    pub field_text: String,
    pub start: usize,
    pub end: usize,
    pub entity_index: usize, // index into Slots.entities
}

/// Comparison predicate — decomposed for direct edge construction.
pub struct ConditionSpan {
    pub field_text: String,
    pub comparator: Comparator,
    pub value: String,
    pub start: usize,
    pub end: usize,
    pub entity_index: usize, // index into Slots.entities
}

/// Field=value write — decomposed for direct edge construction.
pub struct AssignmentSpan {
    pub field_text: String,
    pub value: String,
    pub start: usize,
    pub end: usize,
    pub entity_index: usize, // index into Slots.entities
}

/// Ordering, limiting, or fetching modifier.
pub struct ModifierSpan {
    pub text: String,
    pub kind: ModifierKind,
    pub start: usize,
    pub end: usize,
    pub entity_index: usize, // index into Slots.entities
}

#[derive(Debug, Clone, PartialEq)]
pub enum Intent {
    Select,
    Insert,
    Update,
    Delete,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Comparator {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    Contains,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModifierKind {
    OrderBy { descending: bool },
    Limit,
    Fetch,
    GroupBy,
}
