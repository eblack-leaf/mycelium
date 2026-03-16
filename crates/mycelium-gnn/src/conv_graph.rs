// =============================================================================
// conv_graph.rs — ResolverConv: the single message-passing topology
//
// Built per-query from SchemaGraph + QueryGraph + cross-edges.
// =============================================================================

use crate::graph::SchemaGraph;
use crate::query_graph::QueryGraph;
use crate::operations::{all_operations, is_compatible, OpNode, ConnectsTo};

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
    pub operations: Vec<OpNode>,
}

impl ResolverConv {
    pub fn new(schema_graph: &SchemaGraph, query_graph: &QueryGraph) -> Self {
        let operations = all_operations();

        let node_counts = vec![
            ("table".to_string(), schema_graph.table_nodes.len()),
            ("field".to_string(), schema_graph.field_nodes.len()),
            ("operation".to_string(), operations.len()),
            ("q_collection".to_string(), query_graph.collections.len()),
            ("q_field".to_string(), query_graph.fields.len()),
            ("q_filter".to_string(), query_graph.filters.len()),
            ("q_traversal".to_string(), query_graph.traversals.len()),
            ("q_modifier".to_string(), query_graph.modifiers.len()),
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
            relations.push(ConvRelation {
                src_type: "q_field".to_string(),
                edge_type: "filtered_by".to_string(),
                dst_type: "q_filter".to_string(),
                src_indices: query_graph.filters_on.iter().map(|e| e.dst).collect(),
                dst_indices: query_graph.filters_on.iter().map(|e| e.src).collect(),
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

        // --- Schema edges: field ↔ operation (compatible_op) ---
        // Only for operations that connect to fields, filtered by type.

        let (mut cop_fs, mut cop_fd, mut cop_os, mut cop_od) = (vec![], vec![], vec![], vec![]);
        for field_node in &schema_graph.field_nodes {
            if let Some(ref ft) = field_node.field_type {
                for op in &operations {
                    if op.connects_to == ConnectsTo::Field && is_compatible(op, ft) {
                        cop_fs.push(field_node.id); cop_fd.push(op.id);
                        cop_os.push(op.id); cop_od.push(field_node.id);
                    }
                }
            }
        }
        if !cop_fs.is_empty() {
            relations.push(ConvRelation {
                src_type: "field".to_string(), edge_type: "compatible_op".to_string(),
                dst_type: "operation".to_string(), src_indices: cop_fs, dst_indices: cop_fd,
            });
            relations.push(ConvRelation {
                src_type: "operation".to_string(), edge_type: "compatible_field".to_string(),
                dst_type: "field".to_string(), src_indices: cop_os, dst_indices: cop_od,
            });
        }

        // --- Schema edges: table ↔ operation (table_op) ---
        // Statements and universal aggregates connect to all tables.

        let (mut top_ts, mut top_td, mut top_os, mut top_od) = (vec![], vec![], vec![], vec![]);
        for table_node in &schema_graph.table_nodes {
            for op in &operations {
                if op.connects_to == ConnectsTo::Table {
                    top_ts.push(table_node.id); top_td.push(op.id);
                    top_os.push(op.id); top_od.push(table_node.id);
                }
            }
        }
        if !top_ts.is_empty() {
            relations.push(ConvRelation {
                src_type: "table".to_string(), edge_type: "table_op".to_string(),
                dst_type: "operation".to_string(), src_indices: top_ts, dst_indices: top_td,
            });
            relations.push(ConvRelation {
                src_type: "operation".to_string(), edge_type: "op_table".to_string(),
                dst_type: "table".to_string(), src_indices: top_os, dst_indices: top_od,
            });
        }

        // --- Cross-edges: query candidates ↔ operation (matches_op) ---
        // Built from Grounding model operation_matches on all candidate types.

        for (node_type, op_sources) in [
            ("q_filter", query_graph.filters.iter().map(|f| (f.id, &f.operation_matches)).collect::<Vec<_>>()),
            ("q_field", query_graph.fields.iter().map(|f| (f.id, &f.operation_matches)).collect::<Vec<_>>()),
            ("q_collection", query_graph.collections.iter().map(|c| (c.id, &c.operation_matches)).collect::<Vec<_>>()),
            ("q_traversal", query_graph.traversals.iter().map(|t| (t.id, &t.operation_matches)).collect::<Vec<_>>()),
            ("q_modifier", query_graph.modifiers.iter().map(|m| (m.id, &m.operation_matches)).collect::<Vec<_>>()),
        ] {
            let (mut qs, mut qd, mut ros, mut rod) = (vec![], vec![], vec![], vec![]);
            for (candidate_id, matches) in op_sources {
                for m in matches {
                    qs.push(candidate_id); qd.push(m.operation_id);
                    ros.push(m.operation_id); rod.push(candidate_id);
                }
            }
            if !qs.is_empty() {
                relations.push(ConvRelation {
                    src_type: node_type.to_string(), edge_type: "matches_op".to_string(),
                    dst_type: "operation".to_string(), src_indices: qs, dst_indices: qd,
                });
                relations.push(ConvRelation {
                    src_type: "operation".to_string(),
                    edge_type: format!("matched_by_{}", node_type.strip_prefix("q_").unwrap_or(node_type)),
                    dst_type: node_type.to_string(), src_indices: ros, dst_indices: rod,
                });
            }
        }

        Self { node_counts, relations, operations }
    }
}
