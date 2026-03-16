// =============================================================================
// graph.rs — Heterogeneous graph built from a parsed Schema
//
// Node types: table, field
// Edge types: HAS_FIELD, FIELD_OF, LINKS_TO, LINKED_FROM
// =============================================================================

use super::schema::{Schema, FieldType};

#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub name: String,
    /// Field type, only set for field nodes (used for operation compatibility edges).
    pub field_type: Option<FieldType>,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// table → field
    HasField,
    /// field → table (reverse)
    FieldOf,
    /// table → table (via record link)
    LinksTo,
    /// table → table (reverse of LinksTo)
    LinkedFrom,
}

#[derive(Debug, Clone)]
pub struct SchemaGraph {
    pub table_nodes: Vec<Node>,
    pub field_nodes: Vec<Node>,
    pub has_field: Vec<Edge>,
    pub field_of: Vec<Edge>,
    pub links_to: Vec<Edge>,
    pub linked_from: Vec<Edge>,
}

impl SchemaGraph {
    pub fn from_schema(schema: &Schema) -> Self {
        let mut table_nodes = Vec::new();
        let mut field_nodes = Vec::new();
        let mut has_field = Vec::new();
        let mut field_of = Vec::new();
        let mut links_to = Vec::new();
        let mut linked_from = Vec::new();

        // Build table nodes
        for (table_id, table) in schema.tables.iter().enumerate() {
            table_nodes.push(Node {
                id: table_id,
                name: table.name.clone(),
                field_type: None,
            });
        }

        // Build field nodes + edges
        let mut field_id = 0;
        for (table_id, table) in schema.tables.iter().enumerate() {
            for field in &table.fields {
                field_nodes.push(Node {
                    id: field_id,
                    name: format!("{}.{}", table.name, field.name),
                    field_type: Some(field.field_type.clone()),
                });

                has_field.push(Edge { src: table_id, dst: field_id });
                field_of.push(Edge { src: field_id, dst: table_id });

                // Record links create table→table edges
                if let Some(targets) = record_targets(&field.field_type) {
                    for target_name in targets {
                        if let Some(target_id) = schema.tables.iter().position(|t| t.name == *target_name) {
                            links_to.push(Edge { src: table_id, dst: target_id });
                            linked_from.push(Edge { src: target_id, dst: table_id });
                        }
                    }
                }

                field_id += 1;
            }
        }

        Self {
            table_nodes,
            field_nodes,
            has_field,
            field_of,
            links_to,
            linked_from,
        }
    }
}

/// Extract record link target table names from a FieldType, unwrapping
/// through option/array/set wrappers.
fn record_targets(ft: &FieldType) -> Option<&Vec<String>> {
    match ft {
        FieldType::Record { tables } if !tables.is_empty() => Some(tables),
        FieldType::Option { inner } => record_targets(inner),
        FieldType::Array { inner: Some(inner), .. } => record_targets(inner),
        FieldType::Set { inner: Some(inner), .. } => record_targets(inner),
        _ => None,
    }
}
