use septa::{Comparator, Intent, ValueRef};

/// All node types in the grounded graph — each is a bilinear resolution target.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryNode {
    Table(String),
    Field { table: String, name: String },
    Operation(Intent),
    Comparator(Comparator),
    Modifier(ModifierKind),
    /// Placeholder for span nodes added by inject().
    /// Features come from SpanHiddens (BiLSTM output), not from this variant.
    Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModifierKind {
    OrderBy,
    Limit,
    Fetch,
}

pub struct QueryIr {
    pub intent: Intent,
    pub table: String,
    pub record_id: Option<ValueRef>,
    pub projections: Vec<ResolvedField>,
    pub conditions: Vec<ResolvedCondition>,
    pub assignments: Vec<ResolvedAssignment>,
    pub modifiers: Vec<ResolvedModifier>,
}

pub struct ResolvedField {
    pub table: String,
    pub field: String,
}

pub struct ResolvedCondition {
    pub table: String,
    pub field: String,
    pub comparator: Comparator,
    pub value: ValueRef,
}

pub struct ResolvedAssignment {
    pub table: String,
    pub field: Option<String>, // None = expand slot object via schema types at render
    pub value: ValueRef,
}

pub enum ResolvedModifier {
    OrderBy {
        table: String,
        field: String,
        descending: bool,
    },
    Limit {
        value: ValueRef,
    },
    Fetch {
        field: String,
    },
}

pub struct Query {
    pub surql: String,
}

impl QueryIr {
    /// Render to SurrealQL. values[n] is substituted for Slot(n) references.
    pub fn render(&self, _values: &[String]) -> Query {
        todo!()
    }
}
