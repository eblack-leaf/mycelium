// hyphae — graph neural network

pub mod model;
pub mod ops;
pub mod sage;

use crate::sage::{Edge, EdgeType, TypedEdges};
use regex::Regex;
use septa::{Comparator, Intent, Semantics, ValueRef};
use std::collections::HashMap;
use std::path::Path;

// =============================================================================
// Schema
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct Schema {
    pub tables: Vec<Table>,
}

impl Schema {
    pub fn from_dir(path: &Path) -> std::io::Result<Self> {
        let mut entries: Vec<_> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "surql"))
            .collect();
        entries.sort_by_key(|e| e.path());

        let mut raw = String::new();
        for entry in entries {
            raw.push_str(&std::fs::read_to_string(entry.path())?);
            raw.push('\n');
        }

        Ok(Self::parse(&raw))
    }

    fn parse(raw: &str) -> Self {
        let table_re =
            Regex::new(r"(?i)DEFINE\s+TABLE\s+(?:(?:OVERWRITE|IF\s+NOT\s+EXISTS)\s+)?(\w+)")
                .unwrap();
        let field_re = Regex::new(
            r"(?i)DEFINE\s+FIELD\s+(?:(?:OVERWRITE|IF\s+NOT\s+EXISTS)\s+)?(\S+)\s+ON\s+(?:TABLE\s+)?(\w+)(?:\s+TYPE\s+(.+?))?\s*;"
        ).unwrap();

        let mut tables: Vec<Table> = table_re
            .captures_iter(raw)
            .map(|c| Table {
                name: c[1].to_string(),
                fields: Vec::new(),
            })
            .collect();

        for cap in field_re.captures_iter(raw) {
            let field_name = cap[1].to_string();
            let table_name = &cap[2];
            let field_type = cap
                .get(3)
                .map(|m| FieldType::parse(m.as_str().trim()))
                .unwrap_or(FieldType::Any);

            if let Some(table) = tables.iter_mut().find(|t| t.name == table_name) {
                table.fields.push(Field {
                    name: field_name,
                    field_type,
                });
            }
        }

        Self { tables }
    }
}

#[derive(Debug, Clone)]
pub struct Table {
    pub name: String,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub field_type: FieldType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    Any,
    Bool,
    String,
    Int,
    Float,
    Decimal,
    Number,
    Datetime,
    Duration,
    Bytes,
    Object,
    Regex,
    Record {
        tables: Vec<String>,
    },
    Array {
        inner: Option<Box<FieldType>>,
        max_len: Option<usize>,
    },
    Set {
        inner: Option<Box<FieldType>>,
        max_len: Option<usize>,
    },
    Option {
        inner: Box<FieldType>,
    },
    Geometry {
        variant: GeometryVariant,
    },
    Literal {
        raw: String,
    },
    Range {
        raw: String,
    },
}

impl FieldType {
    fn parse(s: &str) -> Self {
        if let Some(inner) = wrap(s, "option") {
            return Self::Option {
                inner: Box::new(Self::parse(inner)),
            };
        }
        if s.to_lowercase() == "array" {
            return Self::Array {
                inner: None,
                max_len: None,
            };
        }
        if let Some(inner) = wrap(s, "array") {
            let (t, n) = split_len(inner);
            return Self::Array {
                inner: Some(Box::new(Self::parse(t))),
                max_len: n,
            };
        }
        if s.to_lowercase() == "set" {
            return Self::Set {
                inner: None,
                max_len: None,
            };
        }
        if let Some(inner) = wrap(s, "set") {
            let (t, n) = split_len(inner);
            return Self::Set {
                inner: Some(Box::new(Self::parse(t))),
                max_len: n,
            };
        }
        if s.to_lowercase() == "record" {
            return Self::Record { tables: Vec::new() };
        }
        if let Some(inner) = wrap(s, "record") {
            let tables = inner.split('|').map(|t| t.trim().to_string()).collect();
            return Self::Record { tables };
        }
        if let Some(inner) = wrap(s, "geometry") {
            let variant = match inner.trim() {
                "feature" => GeometryVariant::Feature,
                "point" => GeometryVariant::Point,
                "line" => GeometryVariant::Line,
                "polygon" => GeometryVariant::Polygon,
                "multipoint" => GeometryVariant::MultiPoint,
                "multiline" => GeometryVariant::MultiLine,
                "multipolygon" => GeometryVariant::MultiPolygon,
                "collection" => GeometryVariant::Collection,
                _ => GeometryVariant::Point,
            };
            return Self::Geometry { variant };
        }
        match s.to_lowercase().as_str() {
            "any" => Self::Any,
            "bool" => Self::Bool,
            "string" => Self::String,
            "int" => Self::Int,
            "float" => Self::Float,
            "decimal" => Self::Decimal,
            "number" => Self::Number,
            "datetime" => Self::Datetime,
            "duration" => Self::Duration,
            "bytes" => Self::Bytes,
            "object" => Self::Object,
            "regex" => Self::Regex,
            other if other.contains("..") => Self::Range {
                raw: other.to_string(),
            },
            other if other.contains('|') => Self::Literal {
                raw: other.to_string(),
            },
            _ => Self::Any,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeometryVariant {
    Feature,
    Point,
    Line,
    Polygon,
    MultiPoint,
    MultiLine,
    MultiPolygon,
    Collection,
}

fn wrap<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if !s.to_lowercase().starts_with(prefix) {
        return None;
    }
    let rest = s[prefix.len()..].trim_start();
    if rest.starts_with('<') && rest.ends_with('>') {
        Some(&rest[1..rest.len() - 1])
    } else {
        None
    }
}

fn split_len(s: &str) -> (&str, Option<usize>) {
    if let Some((t, n)) = s.rsplit_once(',') {
        if let Ok(n) = n.trim().parse::<usize>() {
            return (t.trim(), Some(n));
        }
    }
    (s, None)
}

// =============================================================================
// QueryNode
// =============================================================================

/// All node types in the grounded graph — each is a bilinear resolution target.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryNode {
    Table(String),
    Field { table: String, name: String },
    Operation(Intent),
    Comparator(Comparator),
    Modifier(ModifierKind),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModifierKind {
    OrderBy,
    Limit,
    Fetch,
}

// =============================================================================
// QueryIr — GNN resolution output, moved here from stipe to avoid circular deps
// =============================================================================

pub struct QueryIr {
    pub intent: Intent,
    pub table: String,
    pub record_id: Option<ValueRef>,
    pub projections: Vec<ResolvedField>,
    pub conditions: Vec<ResolvedCondition>,
    pub assignments: Vec<ResolvedAssignment>,
    pub modifiers: Vec<ResolvedModifier>,
}

pub struct ResolvedField {
    pub table: String,
    pub field: String,
}

pub struct ResolvedCondition {
    pub table: String,
    pub field: String,
    pub comparator: Comparator,
    pub value: ValueRef,
}

pub struct ResolvedAssignment {
    pub table: String,
    pub field: Option<String>, // None = expand slot object via schema types at render
    pub value: ValueRef,
}

pub enum ResolvedModifier {
    OrderBy {
        table: String,
        field: String,
        descending: bool,
    },
    Limit {
        value: ValueRef,
    },
    Fetch {
        field: String,
    },
}

pub struct Query {
    pub surql: String,
}

impl QueryIr {
    /// Render to SurrealQL. values[n] is substituted for Slot(n) references.
    pub fn render(&self, _values: &[String]) -> Query {
        todo!()
    }
}

// =============================================================================
// SchemaGraph
// =============================================================================

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

// =============================================================================
// GroundedGraph
// =============================================================================

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
