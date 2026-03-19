// basidium — synthetic training data generation for the mycelium domain
//
// Produces Datum: (nl, surql, slots) — the labelled training unit.
// Slots are derived from surql against the schema, aligned back to NL spans.

pub mod trainable;
pub mod trainer;

use hyphae::{QueryNode, Schema};
use septa::{Semantics};

/// A single training example.
pub struct Datum {
    pub nl: String,
    pub surql: String,
    pub semantics: Semantics,
    pub labels: Vec<SpanLabel>,
}

pub struct SpanLabel {
    pub span_index: usize, // index into the relevant Slots vec (entities, projections, etc.)
    pub target: QueryNode, // the correct resolution — variant implies which vec
}

impl Datum {
    /// Generate a batch of labelled training datums for a given schema.
    pub fn generate(schema: &Schema) -> Vec<Datum> {
        todo!()
    }
}
