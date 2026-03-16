// =============================================================================
// query_graph.rs — Query graph built from intent extraction output
//
// Builds nodes from Extraction. Only edge: filter → its field (model said so).
// All other relationships (field↔collection, traversal↔collection) are
// resolved by cross-edges in ConvGraph::combined + message passing.
// =============================================================================

use super::intent::{Extraction, SchemaMatch, OperationMatch};

// =============================================================================
// Query graph node types
// =============================================================================

#[derive(Debug, Clone)]
pub struct CollectionCandidate {
    pub id: usize,
    pub surface_form: String,
    pub confidence: f32,
    pub schema_matches: Vec<SchemaMatch>,
    pub operation_matches: Vec<OperationMatch>,
}

#[derive(Debug, Clone)]
pub struct FieldCandidate {
    pub id: usize,
    pub surface_form: String,
    pub confidence: f32,
    pub schema_matches: Vec<SchemaMatch>,
    pub operation_matches: Vec<OperationMatch>,
}

#[derive(Debug, Clone)]
pub struct FilterCandidate {
    pub id: usize,
    pub field_candidate_id: usize,
    pub operator: String,
    pub value: String,
    pub confidence: f32,
    pub operation_matches: Vec<OperationMatch>,
}

#[derive(Debug, Clone)]
pub struct TraversalCandidate {
    pub id: usize,
    pub surface_form: String,
    pub confidence: f32,
    pub schema_matches: Vec<SchemaMatch>,
    pub operation_matches: Vec<OperationMatch>,
}

#[derive(Debug, Clone)]
pub struct ModifierCandidate {
    pub id: usize,
    pub surface_form: String,
    pub value: String,
    pub confidence: f32,
    pub operation_matches: Vec<OperationMatch>,
}

#[derive(Debug, Clone)]
pub struct QueryEdge {
    pub src: usize,
    pub dst: usize,
}

// =============================================================================
// QueryGraph
// =============================================================================

#[derive(Debug, Clone)]
pub struct QueryGraph {
    pub collections: Vec<CollectionCandidate>,
    pub fields: Vec<FieldCandidate>,
    pub filters: Vec<FilterCandidate>,
    pub traversals: Vec<TraversalCandidate>,
    pub modifiers: Vec<ModifierCandidate>,

    /// filter → field (only intra-query edge — the model paired these)
    pub filters_on: Vec<QueryEdge>,
}

impl QueryGraph {
    pub fn from_extraction(extraction: &Extraction) -> Self {
        let collections: Vec<_> = extraction.collections.iter().enumerate().map(|(i, c)| {
            CollectionCandidate {
                id: i,
                surface_form: c.surface_form.clone(),
                confidence: c.confidence,
                schema_matches: c.schema_matches.clone(),
                operation_matches: c.operation_matches.clone(),
            }
        }).collect();

        let mut fields: Vec<_> = extraction.fields.iter().enumerate().map(|(i, f)| {
            FieldCandidate {
                id: i,
                surface_form: f.surface_form.clone(),
                confidence: f.confidence,
                schema_matches: f.schema_matches.clone(),
                operation_matches: f.operation_matches.clone(),
            }
        }).collect();

        let mut filters = Vec::new();
        let mut filters_on = Vec::new();

        for fm in &extraction.filters {
            // Dedup: reuse existing field candidate or create one
            let field_id = fields
                .iter()
                .position(|f| f.surface_form == fm.field.surface_form)
                .unwrap_or_else(|| {
                    let id = fields.len();
                    fields.push(FieldCandidate {
                        id,
                        surface_form: fm.field.surface_form.clone(),
                        confidence: fm.field.confidence,
                        schema_matches: fm.field.schema_matches.clone(),
                        operation_matches: fm.field.operation_matches.clone(),
                    });
                    id
                });

            let filter_id = filters.len();
            filters.push(FilterCandidate {
                id: filter_id,
                field_candidate_id: field_id,
                operator: fm.operator.clone(),
                value: fm.value.clone(),
                confidence: fm.confidence,
                operation_matches: fm.operation_matches.clone(),
            });
            filters_on.push(QueryEdge { src: filter_id, dst: field_id });
        }

        let traversals: Vec<_> = extraction.traversals.iter().enumerate().map(|(i, t)| {
            TraversalCandidate {
                id: i,
                surface_form: t.surface_form.clone(),
                confidence: t.confidence,
                schema_matches: t.schema_matches.clone(),
                operation_matches: t.operation_matches.clone(),
            }
        }).collect();

        let modifiers: Vec<_> = extraction.modifiers.iter().enumerate().map(|(i, m)| {
            ModifierCandidate {
                id: i,
                surface_form: m.surface_form.clone(),
                value: m.value.clone(),
                confidence: m.confidence,
                operation_matches: m.operation_matches.clone(),
            }
        }).collect();

        Self {
            collections,
            fields,
            filters,
            traversals,
            modifiers,
            filters_on,
        }
    }

}
