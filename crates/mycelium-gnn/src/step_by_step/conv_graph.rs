// =============================================================================
// conv_graph.rs — Graph-agnostic topology for message passing
//
// Both SchemaGraph and (future) QueryGraph convert into this.
// SAGEConv and HeteroConv only see ConvGraph, never the semantic types.
// =============================================================================

use super::graph::SchemaGraph;

/// A single relation type: source node type → destination node type.
#[derive(Debug, Clone)]
pub struct ConvRelation {
    pub src_type: String,
    pub edge_type: String,
    pub dst_type: String,
    pub src_indices: Vec<usize>,
    pub dst_indices: Vec<usize>,
}

/// Topology stripped to what message passing needs.
#[derive(Debug, Clone)]
pub struct ConvGraph {
    /// Node type → number of nodes
    pub node_counts: Vec<(String, usize)>,
    /// One entry per relation type
    pub relations: Vec<ConvRelation>,
}

impl ConvGraph {
    pub fn from_schema_graph(sg: &SchemaGraph) -> Self {
        let node_counts = vec![
            ("table".to_string(), sg.table_nodes.len()),
            ("field".to_string(), sg.field_nodes.len()),
        ];

        let mut relations = Vec::new();

        if !sg.has_field.is_empty() {
            relations.push(ConvRelation {
                src_type: "table".to_string(),
                edge_type: "has_field".to_string(),
                dst_type: "field".to_string(),
                src_indices: sg.has_field.iter().map(|e| e.src).collect(),
                dst_indices: sg.has_field.iter().map(|e| e.dst).collect(),
            });
        }

        if !sg.field_of.is_empty() {
            relations.push(ConvRelation {
                src_type: "field".to_string(),
                edge_type: "field_of".to_string(),
                dst_type: "table".to_string(),
                src_indices: sg.field_of.iter().map(|e| e.src).collect(),
                dst_indices: sg.field_of.iter().map(|e| e.dst).collect(),
            });
        }

        if !sg.links_to.is_empty() {
            relations.push(ConvRelation {
                src_type: "table".to_string(),
                edge_type: "links_to".to_string(),
                dst_type: "table".to_string(),
                src_indices: sg.links_to.iter().map(|e| e.src).collect(),
                dst_indices: sg.links_to.iter().map(|e| e.dst).collect(),
            });
        }

        if !sg.linked_from.is_empty() {
            relations.push(ConvRelation {
                src_type: "table".to_string(),
                edge_type: "linked_from".to_string(),
                dst_type: "table".to_string(),
                src_indices: sg.linked_from.iter().map(|e| e.src).collect(),
                dst_indices: sg.linked_from.iter().map(|e| e.dst).collect(),
            });
        }

        Self { node_counts, relations }
    }
}
