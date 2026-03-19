// basidium — synthetic training data generation for the mycelium domain
//
// Produces Datum: (nl, surql, schema) triples.
// Labels for septa (span extraction) and hyphae (GNN) are derived downstream —
// not baked into this format.

/// A single training example — schema-grounded NL query paired with its SurrealQL.
pub struct Datum {
    pub nl:    String,
    pub surql: String,
}

impl Datum {
    /// Generate a batch of (nl, surql) training pairs for a given schema.
    pub fn generate(schema: &Schema) -> Vec<Datum> {
        todo!()
    }
}
/// Minimal schema reference for data generation context.
/// Kept independent of hyphae::Schema to avoid coupling.
pub struct Schema {
    pub name:   String,
    pub tables: Vec<Table>,
}

pub struct Table {
    pub name:   String,
    pub fields: Vec<Field>,
}

pub struct Field {
    pub name:       String,
    pub field_type: String,
}


