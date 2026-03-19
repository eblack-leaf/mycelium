// basidium — synthetic training data generation for the mycelium domain
//
// Produces Datum: (nl, surql, slots) — the labelled training unit.
// Slots are derived from surql against the schema, aligned back to NL spans.

pub mod trainer;
pub mod trainable;

pub use septa::{Slots, Intent};
pub use hyphae::Schema;

/// A single training example.
pub struct Datum {
    pub nl:    String,
    pub surql: String,
    pub slots: Slots,
}

impl Datum {
    /// Generate a batch of labelled training datums for a given schema.
    pub fn generate(schema: &Schema) -> Vec<Datum> {
        todo!()
    }
}
