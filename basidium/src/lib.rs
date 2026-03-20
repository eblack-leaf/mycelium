// basidium — synthetic training data generation for the mycelium domain

pub mod trainable;
pub mod trainer;

use hyphae::{QueryIr, QueryNode, Schema};
use septa::Semantics;

/// A single labelled training example.
pub struct Datum {
    pub nl:        String,
    pub surql:     String,
    pub semantics: Semantics,
    /// GNN supervision: for each span node, the correct QueryNode resolution target.
    /// QueryNode now covers Table, Field, Operation, Comparator, and Modifier —
    /// the variant tells the bilinear head which resolution head to score against.
    pub labels:    Vec<SpanLabel>,
    /// Ground truth QueryIr for end-to-end evaluation.
    pub ir:        QueryIr,
}

pub struct SpanLabel {
    pub span_index: usize,     // index into the relevant span vec (projections, conditions, etc.)
    pub target:     QueryNode, // correct resolution — variant implies which span type
}

impl Datum {
    /// Generate a batch of labelled training datums for a given schema.
    pub fn generate(schema: &Schema) -> Vec<Datum> {
        todo!()
    }
}
