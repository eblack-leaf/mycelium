// =============================================================================
// orchestrator.rs — Stage 4: Walk resolved graph, emit SurrealQL
//
// Deterministic — no ML. Takes a ResolvedGraph and produces valid SurrealQL.
// =============================================================================

use crate::head::ResolvedGraph;
use crate::schema::Schema;

/// Generated SurrealQL query.
#[derive(Debug, Clone)]
pub struct SurrealQuery {
    pub query: String,
    pub params: Vec<(String, String)>,
}

pub struct Orchestrator {
    // TODO: schema reference for validation
}

impl Orchestrator {
    pub fn new(_schema: &Schema) -> Self {
        todo!()
    }

    pub fn emit(&self, _resolved: &ResolvedGraph) -> SurrealQuery {
        todo!()
    }
}
