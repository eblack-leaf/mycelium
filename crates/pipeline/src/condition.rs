use serde::{Deserialize, Serialize};

use crate::schema::Schema;

/// A single condition triple — the IR between the orchestrator and query builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub field: String,
    pub op: Op,
    pub value: Value,
}

/// Operators supported by SurrealDB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Op {
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    Contains,
    ContainsNot,
    ContainsAll,
    ContainsAny,
    ContainsNone,
    Inside,
    Matches,
}

impl Op {
    pub fn to_surreal(&self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Neq => "!=",
            Self::Lt => "<",
            Self::Gt => ">",
            Self::Lte => "<=",
            Self::Gte => ">=",
            Self::Contains => "CONTAINS",
            Self::ContainsNot => "CONTAINSNOT",
            Self::ContainsAll => "CONTAINSALL",
            Self::ContainsAny => "CONTAINSANY",
            Self::ContainsNone => "CONTAINSNONE",
            Self::Inside => "INSIDE",
            Self::Matches => "@@",
        }
    }
}

/// Value in a condition. The orchestrator produces these from spore hints —
/// numeric hints become Int/Float (based on field type from schema),
/// temporal hints stay as marker text, everything else is a raw span.
/// The query builder consumes these to emit SurrealQL literals.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    /// Temporal marker text from the input, e.g. "this week", "3 days ago".
    /// The query builder resolves this to a SurrealQL time expression.
    Temporal(String),
}

impl Value {
    pub fn to_surreal(&self) -> String {
        match self {
            Self::Int(v) => v.to_string(),
            Self::Float(v) => v.to_string(),
            Self::String(v) => format!("'{}'", v.replace('\'', "\\'")),
            Self::Bool(v) => v.to_string(),
            Self::Temporal(expr) => expr.clone(),
        }
    }
}

/// The orchestrator takes spore hints (positioned fragments) and associates
/// them into complete Conditions.
///
/// Spores find pieces independently:
///   Field("stock") at [24..29], Op(Lt) at [30..35], Numeric(10) at [36..38]
///   Field("created") at [33..38], Temporal("this week") at [39..48]
///
/// The orchestrator groups them: which field goes with which op and which value.
/// It also infers missing pieces — e.g. no explicit op near a temporal hint,
/// but the orchestrator can infer Gt from context.
pub trait Orchestrator {
    fn assemble(&self, annotated: &crate::hint::AnnotatedInput, schema: &Schema) -> Vec<Condition>;
}
