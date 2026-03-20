// basidium — synthetic training data generation for the mycelium domain

pub mod trainable;
pub mod trainer;

use hyphae::query::{QueryIr, QueryNode};
use hyphae::schema::Schema;
use septa::Semantics;

/// A single labelled training example.
pub struct Datum {
    pub nl: String,
    pub surql: String,
    pub semantics: Semantics,
    /// GNN supervision: for each span node, the correct QueryNode resolution target.
    /// A single span can have multiple labels when it resolves more than one thing
    /// (e.g. a ConditionSpan has a Field label AND a Comparator label).
    pub labels: Vec<SpanLabel>,
    /// Ground truth QueryIr for end-to-end evaluation.
    pub ir: QueryIr,
}

/// Which span vec in Semantics a label refers to.
#[derive(Debug, Clone, PartialEq)]
pub enum SpanType {
    Intent,
    Entity,
    Projection,
    Condition,
    Assignment,
    Modifier,
}

pub struct SpanLabel {
    pub span_type: SpanType, // which span vec this indexes into
    pub span_index: usize,   // index within that vec (0 for Intent/Entity since they're singular)
    pub target: QueryNode, // correct resolution — variant tells the bilinear head which head to score
}

impl Datum {
    /// Generate a batch of labelled training datums for a given schema.
    pub fn generate(schema: &Schema) -> Vec<Datum> {
        todo!()
    }
}
