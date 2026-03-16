// =============================================================================
// conv_graph.rs — ResolverConv: the single message-passing topology
//
// Built per-query from SchemaGraph + QueryGraph + cross-edges.
// =============================================================================

use super::graph::SchemaGraph;
use super::query_graph::QueryGraph;

/// A single relation type: source node type → destination node type.
#[derive(Debug, Clone)]
pub struct ConvRelation {
    pub src_type: String,
    pub edge_type: String,
    pub dst_type: String,
    pub src_indices: Vec<usize>,
    pub dst_indices: Vec<usize>,
}

/// Combined schema + query topology with cross-edges. Built per-query.
#[derive(Debug, Clone)]
pub struct ResolverConv {
    pub node_counts: Vec<(String, usize)>,
    pub relations: Vec<ConvRelation>,
}

impl ResolverConv {
    pub fn new(schema_graph: &SchemaGraph, query_graph: &QueryGraph) -> Self {
        let mut node_counts = vec![
            ("table".to_string(), schema_graph.table_nodes.len()),
            ("field".to_string(), schema_graph.field_nodes.len()),
            ("q_collection".to_string(), query_graph.collections.len()),
            ("q_field".to_string(), query_graph.fields.len()),
            ("q_filter".to_string(), query_graph.filters.len()),
            ("q_traversal".to_string(), query_graph.traversals.len()),
        ];

        let mut relations = Vec::new();

        // --- Schema intra-edges ---

        if !schema_graph.has_field.is_empty() {
            relations.push(ConvRelation {
                src_type: "table".to_string(),
                edge_type: "has_field".to_string(),
                dst_type: "field".to_string(),
                src_indices: schema_graph.has_field.iter().map(|e| e.src).collect(),
                dst_indices: schema_graph.has_field.iter().map(|e| e.dst).collect(),
            });
        }

        if !schema_graph.field_of.is_empty() {
            relations.push(ConvRelation {
                src_type: "field".to_string(),
                edge_type: "field_of".to_string(),
                dst_type: "table".to_string(),
                src_indices: schema_graph.field_of.iter().map(|e| e.src).collect(),
                dst_indices: schema_graph.field_of.iter().map(|e| e.dst).collect(),
            });
        }

        if !schema_graph.links_to.is_empty() {
            relations.push(ConvRelation {
                src_type: "table".to_string(),
                edge_type: "links_to".to_string(),
                dst_type: "table".to_string(),
                src_indices: schema_graph.links_to.iter().map(|e| e.src).collect(),
                dst_indices: schema_graph.links_to.iter().map(|e| e.dst).collect(),
            });
        }

        if !schema_graph.linked_from.is_empty() {
            relations.push(ConvRelation {
                src_type: "table".to_string(),
                edge_type: "linked_from".to_string(),
                dst_type: "table".to_string(),
                src_indices: schema_graph.linked_from.iter().map(|e| e.src).collect(),
                dst_indices: schema_graph.linked_from.iter().map(|e| e.dst).collect(),
            });
        }

        // --- Query intra-edges ---

        if !query_graph.filters_on.is_empty() {
            relations.push(ConvRelation {
                src_type: "q_filter".to_string(),
                edge_type: "filters_on".to_string(),
                dst_type: "q_field".to_string(),
                src_indices: query_graph.filters_on.iter().map(|e| e.src).collect(),
                dst_indices: query_graph.filters_on.iter().map(|e| e.dst).collect(),
            });
        }

        // --- Cross-edges: q_collection ↔ table ---

        let (mut cs, mut cd, mut ts, mut td) = (vec![], vec![], vec![], vec![]);
        for coll in &query_graph.collections {
            for m in &coll.schema_matches {
                if m.schema_node_type == "table" {
                    cs.push(coll.id); cd.push(m.schema_node_id);
                    ts.push(m.schema_node_id); td.push(coll.id);
                }
            }
        }
        if !cs.is_empty() {
            relations.push(ConvRelation {
                src_type: "q_collection".to_string(), edge_type: "matches_table".to_string(),
                dst_type: "table".to_string(), src_indices: cs, dst_indices: cd,
            });
            relations.push(ConvRelation {
                src_type: "table".to_string(), edge_type: "matched_by_collection".to_string(),
                dst_type: "q_collection".to_string(), src_indices: ts, dst_indices: td,
            });
        }

        // --- Cross-edges: q_field ↔ field ---

        let (mut fs, mut fd, mut rfs, mut rfd) = (vec![], vec![], vec![], vec![]);
        for f in &query_graph.fields {
            for m in &f.schema_matches {
                if m.schema_node_type == "field" {
                    fs.push(f.id); fd.push(m.schema_node_id);
                    rfs.push(m.schema_node_id); rfd.push(f.id);
                }
            }
        }
        if !fs.is_empty() {
            relations.push(ConvRelation {
                src_type: "q_field".to_string(), edge_type: "matches_field".to_string(),
                dst_type: "field".to_string(), src_indices: fs, dst_indices: fd,
            });
            relations.push(ConvRelation {
                src_type: "field".to_string(), edge_type: "matched_by_field".to_string(),
                dst_type: "q_field".to_string(), src_indices: rfs, dst_indices: rfd,
            });
        }

        // --- Cross-edges: q_traversal ↔ table ---

        let (mut tvs, mut tvd, mut rts, mut rtd) = (vec![], vec![], vec![], vec![]);
        for t in &query_graph.traversals {
            for m in &t.schema_matches {
                if m.schema_node_type == "table" {
                    tvs.push(t.id); tvd.push(m.schema_node_id);
                    rts.push(m.schema_node_id); rtd.push(t.id);
                }
            }
        }
        if !tvs.is_empty() {
            relations.push(ConvRelation {
                src_type: "q_traversal".to_string(), edge_type: "matches_table".to_string(),
                dst_type: "table".to_string(), src_indices: tvs, dst_indices: tvd,
            });
            relations.push(ConvRelation {
                src_type: "table".to_string(), edge_type: "matched_by_traversal".to_string(),
                dst_type: "q_traversal".to_string(), src_indices: rts, dst_indices: rtd,
            });
        }

        Self { node_counts, relations }
    }
}
