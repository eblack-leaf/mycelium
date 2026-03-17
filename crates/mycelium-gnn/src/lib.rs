#![recursion_limit = "512"]
// =============================================================================
// mycelium-gnn — Stage 1: Schema-grounded query resolution
// =============================================================================

pub mod schema;
pub mod graph;
pub mod conv_graph;
pub mod tensor_ops;
pub mod sage;
pub mod embed;
pub mod intent;
pub mod query_graph;
pub mod operations;
pub mod grounding;
pub mod head;
pub mod training;
pub mod orchestrator;
pub mod nlp;
pub mod candidate_matcher;
pub mod linguistic_graph;

use std::path::Path;
use schema::{Reader, Extractor, Schema, Validation};
use graph::SchemaGraph;

/// Stage 1 entry point. Owns the parsed schema and encoder.
#[derive(Debug)]
pub struct Gnn {
    pub raw_schema: String,
    pub schema: Schema,
    pub graph: SchemaGraph,
    pub validation: Validation,
}

impl Gnn {
    /// Load schema from a file or directory of .surql/.sql files, parse it.
    pub fn from_schema(path: &Path) -> std::io::Result<Self> {
        let raw_schema = Reader::read(path)?;
        let (schema, validation) = Extractor::extract(&raw_schema);
        let graph = SchemaGraph::from_schema(&schema);

        Ok(Self {
            raw_schema,
            schema,
            graph,
            validation,
        })
    }
}
