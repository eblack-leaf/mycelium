// =============================================================================
// linguistic_graph.rs — Builds the GNN-ready conv topology from linguistic parse
//
// Replaces query_graph.rs for the new pipeline. Instead of pre-typed candidates
// (q_collection, q_field, etc.), linguistic nodes are untyped — the GNN assigns
// roles through message passing.
//
// Node types in the combined graph:
//   Schema:     table, field, operation        (same as before)
//   Linguistic: np, quantifier, comparator     (from NLP parser)
//
// Edge types:
//   Schema intra:      has_field, field_of, links_to, linked_from,
//                      compatible_op, compatible_field, table_op, op_table
//   Linguistic intra:  possessive, possessive_inv, quantifies, quantifies_inv,
//                      comparison, comparison_inv
//   Cross (candidate): candidate_table, candidate_table_inv,
//                      candidate_field, candidate_field_inv,
//                      candidate_op, candidate_op_inv
// =============================================================================

use crate::graph::SchemaGraph;
use crate::nlp::{LinguisticGraph, DepRelation, SpanType};
use crate::candidate_matcher::CandidateSet;
use crate::conv_graph::ConvRelation;
use crate::operations::{all_operations, is_compatible, OpNode, ConnectsTo};

/// Combined schema + linguistic topology for the GNN.
pub struct LinguisticConv {
    pub node_counts: Vec<(String, usize)>,
    pub relations: Vec<ConvRelation>,
    pub operations: Vec<OpNode>,
}

/// Map SpanType to the node type name used in the conv graph.
fn span_node_type(span_type: SpanType) -> &'static str {
    match span_type {
        SpanType::NounPhrase => "np",
        SpanType::Quantifier => "quantifier",
        SpanType::Comparator => "comparator",
        SpanType::Intent => "intent",
    }
}

impl LinguisticConv {
    /// Template with all relation types (empty edges). Used to initialize
    /// the Encoder so every SAGEConv exists regardless of query shape.
    pub fn template(schema_graph: &SchemaGraph) -> Self {
        let operations = all_operations();
        let node_counts = vec![
            // Schema
            ("table".into(), schema_graph.table_nodes.len()),
            ("field".into(), schema_graph.field_nodes.len()),
            ("operation".into(), operations.len()),
            // Linguistic (sizes vary per query)
            ("np".into(), 0),
            ("quantifier".into(), 0),
            ("comparator".into(), 0),
            ("intent".into(), 0),
        ];

        let mut relations = Vec::new();

        // Schema intra (same as ResolverConv)
        push_empty(&mut relations, "table", "has_field", "field");
        push_empty(&mut relations, "field", "field_of", "table");
        push_empty(&mut relations, "table", "links_to", "table");
        push_empty(&mut relations, "table", "linked_from", "table");
        push_empty(&mut relations, "field", "compatible_op", "operation");
        push_empty(&mut relations, "operation", "compatible_field", "field");
        push_empty(&mut relations, "table", "table_op", "operation");
        push_empty(&mut relations, "operation", "op_table", "table");

        // Linguistic intra (dep parse edges, bidirectional)
        push_empty(&mut relations, "np", "possessive", "np");
        push_empty(&mut relations, "np", "possessive_inv", "np");
        push_empty(&mut relations, "quantifier", "quantifies", "np");
        push_empty(&mut relations, "np", "quantified_by", "quantifier");
        push_empty(&mut relations, "comparator", "comparison", "np");
        push_empty(&mut relations, "np", "compared_by", "comparator");
        push_empty(&mut relations, "intent", "targets", "np");
        push_empty(&mut relations, "np", "targeted_by", "intent");

        // Cross: linguistic ↔ schema (candidate match edges)
        // Each linguistic node type can match any schema node type
        for ling_type in &["np", "quantifier", "comparator", "intent"] {
            push_empty(&mut relations, ling_type, "candidate_table", "table");
            push_empty(&mut relations, "table", &format!("candidate_{}_inv", ling_type), ling_type);
            push_empty(&mut relations, ling_type, "candidate_field", "field");
            push_empty(&mut relations, "field", &format!("candidate_{}_inv", ling_type), ling_type);
            push_empty(&mut relations, ling_type, "candidate_op", "operation");
            push_empty(&mut relations, "operation", &format!("candidate_{}_inv", ling_type), ling_type);
        }

        Self { node_counts, relations, operations }
    }

    /// Build the conv topology for a specific query.
    pub fn new(
        schema_graph: &SchemaGraph,
        ling_graph: &LinguisticGraph,
        candidates: &CandidateSet,
    ) -> Self {
        let operations = all_operations();

        // Count linguistic nodes by type
        let mut n_np = 0;
        let mut n_quant = 0;
        let mut n_comp = 0;
        let mut n_intent = 0;
        // Map from LinguisticNode.id → local index within its type
        let mut ling_local_id: Vec<usize> = Vec::new();

        for node in &ling_graph.nodes {
            let local = match node.span_type {
                SpanType::NounPhrase => { let l = n_np; n_np += 1; l },
                SpanType::Quantifier => { let l = n_quant; n_quant += 1; l },
                SpanType::Comparator => { let l = n_comp; n_comp += 1; l },
                SpanType::Intent => { let l = n_intent; n_intent += 1; l },
            };
            ling_local_id.push(local);
        }

        let node_counts = vec![
            ("table".into(), schema_graph.table_nodes.len()),
            ("field".into(), schema_graph.field_nodes.len()),
            ("operation".into(), operations.len()),
            ("np".into(), n_np),
            ("quantifier".into(), n_quant),
            ("comparator".into(), n_comp),
            ("intent".into(), n_intent),
        ];

        let mut relations = Vec::new();

        // --- Schema intra-edges (same as ResolverConv::new) ---
        add_schema_edges(&mut relations, schema_graph, &operations);

        // --- Linguistic intra-edges (from dep parse) ---
        add_linguistic_edges(&mut relations, ling_graph, &ling_local_id);

        // --- Cross-edges (from candidate matcher) ---
        add_candidate_edges(&mut relations, ling_graph, &ling_local_id, candidates);

        Self { node_counts, relations, operations }
    }
}

// =============================================================================
// Edge builders
// =============================================================================

fn push_empty(relations: &mut Vec<ConvRelation>, src: &str, edge: &str, dst: &str) {
    relations.push(ConvRelation {
        src_type: src.into(), edge_type: edge.into(), dst_type: dst.into(),
        src_indices: vec![], dst_indices: vec![],
    });
}

fn add_schema_edges(
    relations: &mut Vec<ConvRelation>,
    schema_graph: &SchemaGraph,
    operations: &[OpNode],
) {
    // has_field / field_of
    if !schema_graph.has_field.is_empty() {
        relations.push(ConvRelation {
            src_type: "table".into(), edge_type: "has_field".into(), dst_type: "field".into(),
            src_indices: schema_graph.has_field.iter().map(|e| e.src).collect(),
            dst_indices: schema_graph.has_field.iter().map(|e| e.dst).collect(),
        });
        relations.push(ConvRelation {
            src_type: "field".into(), edge_type: "field_of".into(), dst_type: "table".into(),
            src_indices: schema_graph.field_of.iter().map(|e| e.src).collect(),
            dst_indices: schema_graph.field_of.iter().map(|e| e.dst).collect(),
        });
    }

    // links_to / linked_from
    if !schema_graph.links_to.is_empty() {
        relations.push(ConvRelation {
            src_type: "table".into(), edge_type: "links_to".into(), dst_type: "table".into(),
            src_indices: schema_graph.links_to.iter().map(|e| e.src).collect(),
            dst_indices: schema_graph.links_to.iter().map(|e| e.dst).collect(),
        });
        relations.push(ConvRelation {
            src_type: "table".into(), edge_type: "linked_from".into(), dst_type: "table".into(),
            src_indices: schema_graph.linked_from.iter().map(|e| e.src).collect(),
            dst_indices: schema_graph.linked_from.iter().map(|e| e.dst).collect(),
        });
    }

    // field ↔ operation (compatible_op)
    let (mut fs, mut fd, mut os, mut od) = (vec![], vec![], vec![], vec![]);
    for field_node in &schema_graph.field_nodes {
        if let Some(ref ft) = field_node.field_type {
            for op in operations {
                if op.connects_to == ConnectsTo::Field && is_compatible(op, ft) {
                    fs.push(field_node.id); fd.push(op.id);
                    os.push(op.id); od.push(field_node.id);
                }
            }
        }
    }
    if !fs.is_empty() {
        relations.push(ConvRelation {
            src_type: "field".into(), edge_type: "compatible_op".into(), dst_type: "operation".into(),
            src_indices: fs, dst_indices: fd,
        });
        relations.push(ConvRelation {
            src_type: "operation".into(), edge_type: "compatible_field".into(), dst_type: "field".into(),
            src_indices: os, dst_indices: od,
        });
    }

    // table ↔ operation (table_op)
    let (mut ts, mut td, mut tos, mut tod) = (vec![], vec![], vec![], vec![]);
    for table_node in &schema_graph.table_nodes {
        for op in operations {
            if op.connects_to == ConnectsTo::Table {
                ts.push(table_node.id); td.push(op.id);
                tos.push(op.id); tod.push(table_node.id);
            }
        }
    }
    if !ts.is_empty() {
        relations.push(ConvRelation {
            src_type: "table".into(), edge_type: "table_op".into(), dst_type: "operation".into(),
            src_indices: ts, dst_indices: td,
        });
        relations.push(ConvRelation {
            src_type: "operation".into(), edge_type: "op_table".into(), dst_type: "table".into(),
            src_indices: tos, dst_indices: tod,
        });
    }
}

fn add_linguistic_edges(
    relations: &mut Vec<ConvRelation>,
    ling_graph: &LinguisticGraph,
    local_ids: &[usize],
) {
    // Group edges by (src_type, relation, dst_type) to batch into ConvRelations
    use std::collections::HashMap;
    let mut edge_groups: HashMap<(String, String, String), (Vec<usize>, Vec<usize>)> =
        HashMap::new();

    for edge in &ling_graph.edges {
        let src_node = &ling_graph.nodes[edge.src];
        let dst_node = &ling_graph.nodes[edge.dst];
        let src_type = span_node_type(src_node.span_type);
        let dst_type = span_node_type(dst_node.span_type);

        let (fwd_label, inv_label) = match edge.relation {
            DepRelation::Possessive => ("possessive", "possessive_inv"),
            DepRelation::Quantifies => ("quantifies", "quantified_by"),
            DepRelation::Comparison => ("comparison", "compared_by"),
            DepRelation::IntentTarget => ("targets", "targeted_by"),
        };

        // Forward edge
        let key = (src_type.into(), fwd_label.into(), dst_type.into());
        let entry = edge_groups.entry(key).or_default();
        entry.0.push(local_ids[edge.src]);
        entry.1.push(local_ids[edge.dst]);

        // Inverse edge
        let key = (dst_type.into(), inv_label.into(), src_type.into());
        let entry = edge_groups.entry(key).or_default();
        entry.0.push(local_ids[edge.dst]);
        entry.1.push(local_ids[edge.src]);
    }

    for ((src_type, edge_type, dst_type), (src_indices, dst_indices)) in edge_groups {
        relations.push(ConvRelation {
            src_type, edge_type, dst_type,
            src_indices, dst_indices,
        });
    }
}

fn add_candidate_edges(
    relations: &mut Vec<ConvRelation>,
    ling_graph: &LinguisticGraph,
    local_ids: &[usize],
    candidates: &CandidateSet,
) {
    use std::collections::HashMap;
    // Group by (ling_node_type, schema_node_type)
    let mut groups: HashMap<(String, String), (Vec<usize>, Vec<usize>)> = HashMap::new();
    let mut inv_groups: HashMap<(String, String), (Vec<usize>, Vec<usize>)> = HashMap::new();

    for edge in &candidates.edges {
        let ling_node = &ling_graph.nodes[edge.linguistic_node];
        let ling_type = span_node_type(ling_node.span_type).to_string();
        let local_id = local_ids[edge.linguistic_node];

        let _edge_label = format!("candidate_{}", edge.schema_node_type);
        let _inv_label = format!("candidate_{}_inv", ling_type);

        // Forward: linguistic → schema
        let key = (ling_type.clone(), edge.schema_node_type.clone());
        let entry = groups.entry(key).or_default();
        entry.0.push(local_id);
        entry.1.push(edge.schema_node_id);

        // Inverse: schema → linguistic
        let key = (edge.schema_node_type.clone(), ling_type);
        let entry = inv_groups.entry(key).or_default();
        entry.0.push(edge.schema_node_id);
        entry.1.push(local_id);
    }

    for ((ling_type, schema_type), (src_indices, dst_indices)) in groups {
        relations.push(ConvRelation {
            src_type: ling_type,
            edge_type: format!("candidate_{}", schema_type),
            dst_type: schema_type,
            src_indices, dst_indices,
        });
    }

    for ((schema_type, ling_type), (src_indices, dst_indices)) in inv_groups {
        relations.push(ConvRelation {
            src_type: schema_type,
            edge_type: format!("candidate_{}_inv", ling_type),
            dst_type: ling_type,
            src_indices, dst_indices,
        });
    }
}
