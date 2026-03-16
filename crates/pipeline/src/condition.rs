use serde::{Deserialize, Serialize};

use crate::schema::Schema;

/// A single extracted condition triple.
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

/// Extracted value, typed after schema lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    /// Raw temporal expression resolved by TemporalResolver.
    /// e.g. "time::now() - 1w"
    Temporal(String),
    /// Passthrough for types we don't parse — plugin provides the literal.
    Raw(String),
}

impl Value {
    pub fn to_surreal(&self) -> String {
        match self {
            Self::Int(v) => v.to_string(),
            Self::Float(v) => v.to_string(),
            Self::String(v) => format!("'{}'", v.replace('\'', "\\'")),
            Self::Bool(v) => v.to_string(),
            Self::Temporal(expr) => expr.clone(),
            Self::Raw(v) => v.clone(),
        }
    }
}

/// The orchestrator model's job: take spore hints (positioned fragments)
/// and associate them into complete Conditions.
///
/// Spores find the pieces independently:
///   Field("stock") at [8..13], Op(Lt) at [14..23], Numeric(10) at [24..26]
///   Temporal("this week") at [35..44], Field("created") at [27..34]
///
/// The orchestrator decides which field goes with which op goes with which value.
/// It sees the full input + all hints with positions + schema context.
///
/// Architecture: takes the annotated input as a structured feature vector
/// and outputs association indices — grouping hints into condition triples.
pub trait Orchestrator {
    fn assemble(&self, annotated: &crate::hint::AnnotatedInput, schema: &Schema) -> Vec<Condition>;
}
