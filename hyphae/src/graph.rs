use std::collections::HashMap;
use septa::{Comparator, Intent, Semantics};
use crate::query::{ModifierKind, QueryIr, QueryNode};
use crate::sage::{Edge, EdgeType, TypedEdges};
use crate::schema::{FieldType, Schema};

pub struct SchemaGraph {
    schema: Schema,
    nodes: Vec<QueryNode>,
    edges: TypedEdges,
}

impl SchemaGraph {
    pub fn new(schema: Schema) -> Self {
        let mut nodes: Vec<QueryNode> = Vec::new();
        let mut edges = TypedEdges::new();

        // Initialise all edge type buckets
        for et in EdgeType::all() {
            edges.insert(et.clone(), vec![]);
        }

        // ── Fixed vocabulary nodes ──────────────────────────────────────────
        // Indices are stable regardless of schema — span cross edges in inject()
        // connect to these by iterating QueryNode::Operation/Comparator/Modifier.

        // Operation nodes  [0..3]
        let op_base = nodes.len();
        for op in [
            Intent::Select,
            Intent::Create,
            Intent::Update,
            Intent::Delete,
        ] {
            nodes.push(QueryNode::Operation(op));
        }

        // Comparator nodes  [4..10]
        let cmp_base = nodes.len();
        for cmp in [
            Comparator::Eq,
            Comparator::Neq,
            Comparator::Gt,
            Comparator::Gte,
            Comparator::Lt,
            Comparator::Lte,
            Comparator::Contains,
        ] {
            nodes.push(QueryNode::Comparator(cmp));
        }

        // Modifier nodes  [11..13]
        let mod_base = nodes.len();
        let fetch_mod_idx = mod_base;
        let orderby_mod_idx = mod_base + 1;
        nodes.push(QueryNode::Modifier(ModifierKind::Fetch));
        nodes.push(QueryNode::Modifier(ModifierKind::OrderBy));
        nodes.push(QueryNode::Modifier(ModifierKind::Limit));

        // ── Schema nodes ────────────────────────────────────────────────────

        let schema_node_base = nodes.len();
        let mut table_map: HashMap<String, usize> = HashMap::new();

        // Pass 1: table nodes
        for table in schema.tables.iter() {
            let idx = nodes.len();
            nodes.push(QueryNode::Table(table.name.clone()));
            table_map.insert(table.name.clone(), idx);
        }

        // Pass 2: field nodes + structural edges
        // Collect record-link field indices for ModifierToField after all fields are created.
        let mut record_field_indices: Vec<usize> = Vec::new();
        let mut all_field_indices: Vec<usize> = Vec::new();

        for table in schema.tables.iter() {
            let table_idx = *table_map.get(&table.name).unwrap();

            for field in table.fields.iter() {
                let field_idx = nodes.len();
                nodes.push(QueryNode::Field {
                    table: table.name.clone(),
                    name: field.name.clone(),
                });

                // HasField / FieldOf
                edges.get_mut(&EdgeType::HasField).unwrap().push(Edge {
                    src: table_idx,
                    dst: field_idx,
                });
                edges.get_mut(&EdgeType::FieldOf).unwrap().push(Edge {
                    src: field_idx,
                    dst: table_idx,
                });

                // LinksTo / LinkedFrom  — Field → linked Table (fixes Table→Table bug)
                if let FieldType::Record { ref tables } = field.field_type {
                    for linked_name in tables {
                        if let Some(&linked_idx) = table_map.get(linked_name) {
                            edges.get_mut(&EdgeType::LinksTo).unwrap().push(Edge {
                                src: field_idx,
                                dst: linked_idx,
                            });
                            edges.get_mut(&EdgeType::LinkedFrom).unwrap().push(Edge {
                                src: linked_idx,
                                dst: field_idx,
                            });
                        }
                    }
                    record_field_indices.push(field_idx);
                }

                all_field_indices.push(field_idx);
            }
        }

        // ModifierToField — multi-hop routing for modifier span resolution
        // Fetch  → record-link fields only
        for &f in &record_field_indices {
            edges
                .get_mut(&EdgeType::ModifierToField)
                .unwrap()
                .push(Edge {
                    src: fetch_mod_idx,
                    dst: f,
                });
        }
        // OrderBy → all fields
        for &f in &all_field_indices {
            edges
                .get_mut(&EdgeType::ModifierToField)
                .unwrap()
                .push(Edge {
                    src: orderby_mod_idx,
                    dst: f,
                });
        }
        // Limit → nothing

        let _ = (op_base, cmp_base, schema_node_base); // suppress unused warnings

        Self {
            schema,
            nodes,
            edges,
        }
    }

    pub fn inject(&self, semantics: &Semantics) -> GroundedGraph {
        let mut nodes = self.nodes.clone();
        let mut edges = self.edges.clone();

        // ── Collect schema node indices by type ───────────────────────────
        let op_indices: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, QueryNode::Operation(_)).then_some(i))
            .collect();
        let cmp_indices: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, QueryNode::Comparator(_)).then_some(i))
            .collect();
        let mod_indices: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, QueryNode::Modifier(_)).then_some(i))
            .collect();
        let table_indices: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, QueryNode::Table(_)).then_some(i))
            .collect();
        let field_indices: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, QueryNode::Field { .. }).then_some(i))
            .collect();

        // Helper: push fan-out edges into a bucket
        fn fan_out(edges: &mut TypedEdges, et: &EdgeType, src: usize, dsts: &[usize]) {
            let bucket = edges.get_mut(et).unwrap();
            for &dst in dsts {
                bucket.push(Edge { src, dst });
            }
        }

        // ── Intent span node ──────────────────────────────────────────────
        let intent_idx = nodes.len();
        nodes.push(QueryNode::Operation(Intent::Select)); // placeholder — resolved by head
        fan_out(
            &mut edges,
            &EdgeType::IntentToOperation,
            intent_idx,
            &op_indices,
        );

        let intent_resolution = Resolution {
            span_index: intent_idx,
            candidates: op_indices.clone(),
        };

        // ── Entity span node ──────────────────────────────────────────────
        let entity_idx = nodes.len();
        nodes.push(QueryNode::Table(String::new())); // placeholder
        fan_out(
            &mut edges,
            &EdgeType::EntityToTable,
            entity_idx,
            &table_indices,
        );

        let entity_resolution = Resolution {
            span_index: entity_idx,
            candidates: table_indices.clone(),
        };

        // ── Projection span nodes ─────────────────────────────────────────
        let mut projection_resolutions = Vec::new();
        let mut proj_span_indices: Vec<usize> = Vec::new();

        for _proj in &semantics.projections {
            let idx = nodes.len();
            nodes.push(QueryNode::Field {
                table: String::new(),
                name: String::new(),
            });
            fan_out(
                &mut edges,
                &EdgeType::ProjectionToField,
                idx,
                &field_indices,
            );
            proj_span_indices.push(idx);
            projection_resolutions.push(Resolution {
                span_index: idx,
                candidates: field_indices.clone(),
            });
        }

        // ── Modifier span nodes (before ProjectionToFetch so indices exist) ──
        let mut modifier_type_resolutions = Vec::new();
        let mut modifier_field_resolutions = Vec::new();
        let mut mod_span_indices: Vec<usize> = Vec::new();

        for modifier in &semantics.modifiers {
            let idx = nodes.len();
            nodes.push(QueryNode::Modifier(ModifierKind::Fetch)); // placeholder
            fan_out(&mut edges, &EdgeType::ModifierToType, idx, &mod_indices);
            mod_span_indices.push(idx);

            modifier_type_resolutions.push(Resolution {
                span_index: idx,
                candidates: mod_indices.clone(),
            });

            // Field resolution only when the modifier has an argument (OrderBy/Fetch).
            // Limit has no field target — skip.
            if modifier.argument.is_some() {
                modifier_field_resolutions.push(Resolution {
                    span_index: idx,
                    candidates: field_indices.clone(),
                });
            }
        }

        // ── ProjectionToFetch inter-span edges ───────────────────────────
        for (pi, proj) in semantics.projections.iter().enumerate() {
            if let Some(fi) = proj.fetch_index {
                if fi < mod_span_indices.len() {
                    edges
                        .get_mut(&EdgeType::ProjectionToFetch)
                        .unwrap()
                        .push(Edge {
                            src: proj_span_indices[pi],
                            dst: mod_span_indices[fi],
                        });
                }
            }
        }

        // ── Condition span nodes ──────────────────────────────────────────
        let mut condition_field_resolutions = Vec::new();
        let mut condition_cmp_resolutions = Vec::new();

        for _cond in &semantics.conditions {
            let idx = nodes.len();
            nodes.push(QueryNode::Field {
                table: String::new(),
                name: String::new(),
            });

            fan_out(&mut edges, &EdgeType::ConditionToField, idx, &field_indices);
            fan_out(
                &mut edges,
                &EdgeType::ConditionToComparator,
                idx,
                &cmp_indices,
            );

            condition_field_resolutions.push(Resolution {
                span_index: idx,
                candidates: field_indices.clone(),
            });
            condition_cmp_resolutions.push(Resolution {
                span_index: idx,
                candidates: cmp_indices.clone(),
            });
        }

        // ── Assignment span nodes ─────────────────────────────────────────
        let mut assignment_resolutions = Vec::new();

        for assign in &semantics.assignments {
            let idx = nodes.len();
            nodes.push(QueryNode::Field {
                table: String::new(),
                name: String::new(),
            });

            // Only wire field resolution when field_text is present.
            // Object-expansion assignments (field_text: None) have no field to resolve.
            if assign.field_text.is_some() {
                fan_out(
                    &mut edges,
                    &EdgeType::AssignmentToField,
                    idx,
                    &field_indices,
                );
                assignment_resolutions.push(Resolution {
                    span_index: idx,
                    candidates: field_indices.clone(),
                });
            }
        }

        GroundedGraph {
            nodes,
            edges,
            intent_resolution,
            entity_resolution,
            projection_resolutions,
            condition_field_resolutions,
            condition_cmp_resolutions,
            assignment_resolutions,
            modifier_type_resolutions,
            modifier_field_resolutions,
        }
    }
}

/// A single resolution task: one span node scored against a set of candidate schema nodes.
pub struct Resolution {
    pub span_index: usize,      // index into GroundedGraph.nodes
    pub candidates: Vec<usize>, // schema node indices the bilinear head scores against
}

pub struct GroundedGraph {
    /// Flat node index space: schema nodes followed by span nodes added by inject().
    pub nodes: Vec<QueryNode>,

    /// Typed edges over node indices.
    pub edges: TypedEdges,

    /// Each resolution task targets one bilinear head.
    /// A single span node can appear in multiple resolution lists
    /// (e.g. ConditionSpan resolves both a Field and a Comparator).
    pub intent_resolution: Resolution,
    pub entity_resolution: Resolution,
    pub projection_resolutions: Vec<Resolution>, // ProjectionSpan → Field
    pub condition_field_resolutions: Vec<Resolution>, // ConditionSpan → Field
    pub condition_cmp_resolutions: Vec<Resolution>, // ConditionSpan → Comparator
    pub assignment_resolutions: Vec<Resolution>, // AssignmentSpan → Field
    pub modifier_type_resolutions: Vec<Resolution>, // ModifierSpan → Modifier
    pub modifier_field_resolutions: Vec<Resolution>, // ModifierSpan → Field (OrderBy/Fetch only)
}

impl GroundedGraph {
    pub fn forward(&self) -> QueryIr {
        todo!()
    }
}