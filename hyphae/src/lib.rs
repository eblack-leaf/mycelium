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
    pub name:   String,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name:       String,
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
    Record { tables: Vec<std::string::String> },
    Array  { inner: Option<Box<FieldType>>, max_len: Option<usize> },
    Set    { inner: Option<Box<FieldType>>, max_len: Option<usize> },
    Option { inner: Box<FieldType> },
    Geometry { variant: GeometryVariant },
    Literal  { raw: std::string::String },
    Range    { raw: std::string::String },
}

impl FieldType {
    fn parse(s: &str) -> Self {
        if let Some(inner) = wrap(s, "option") {
            return Self::Option { inner: Box::new(Self::parse(inner)) };
        }
        if s.to_lowercase() == "array" {
            return Self::Array { inner: None, max_len: None };
        }
        if let Some(inner) = wrap(s, "array") {
            let (t, n) = split_len(inner);
            return Self::Array { inner: Some(Box::new(Self::parse(t))), max_len: n };
        }
        if s.to_lowercase() == "set" {
            return Self::Set { inner: None, max_len: None };
        }
        if let Some(inner) = wrap(s, "set") {
            let (t, n) = split_len(inner);
            return Self::Set { inner: Some(Box::new(Self::parse(t))), max_len: n };
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
                "feature"      => GeometryVariant::Feature,
                "point"        => GeometryVariant::Point,
                "line"         => GeometryVariant::Line,
                "polygon"      => GeometryVariant::Polygon,
                "multipoint"   => GeometryVariant::MultiPoint,
                "multiline"    => GeometryVariant::MultiLine,
                "multipolygon" => GeometryVariant::MultiPolygon,
                "collection"   => GeometryVariant::Collection,
                _              => GeometryVariant::Point,
            };
            return Self::Geometry { variant };
        }
        match s.to_lowercase().as_str() {
            "any"      => Self::Any,
            "bool"     => Self::Bool,
            "string"   => Self::String,
            "int"      => Self::Int,
            "float"    => Self::Float,
            "decimal"  => Self::Decimal,
            "number"   => Self::Number,
            "datetime" => Self::Datetime,
            "duration" => Self::Duration,
            "bytes"    => Self::Bytes,
            "object"   => Self::Object,
            "regex"    => Self::Regex,
            other if other.contains("..") => Self::Range   { raw: other.to_string() },
            other if other.contains('|')  => Self::Literal { raw: other.to_string() },
            _ => Self::Any,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeometryVariant {
    Feature, Point, Line, Polygon,
    MultiPoint, MultiLine, MultiPolygon, Collection,
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
    Table(std::string::String),
    Field { table: std::string::String, name: std::string::String },
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
    pub intent:      Intent,
    pub table:       std::string::String,
    pub record_id:   Option<ValueRef>,
    pub projections: Vec<ResolvedField>,
    pub conditions:  Vec<ResolvedCondition>,
    pub assignments: Vec<ResolvedAssignment>,
    pub modifiers:   Vec<ResolvedModifier>,
}

pub struct ResolvedField {
    pub table: std::string::String,
    pub field: std::string::String,
}

pub struct ResolvedCondition {
    pub table:      std::string::String,
    pub field:      std::string::String,
    pub comparator: Comparator,
    pub value:      ValueRef,
}

pub struct ResolvedAssignment {
    pub table: std::string::String,
    pub field: Option<std::string::String>, // None = expand slot object via schema types at render
    pub value: ValueRef,
}

pub enum ResolvedModifier {
    OrderBy { table: std::string::String, field: std::string::String, descending: bool },
    Limit   { value: ValueRef },
    Fetch   { field: std::string::String },
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
    nodes:  Vec<QueryNode>,
    edges:  TypedEdges,
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
        for op in [Intent::Select, Intent::Create, Intent::Update, Intent::Delete] {
            nodes.push(QueryNode::Operation(op));
        }

        // Comparator nodes  [4..10]
        let cmp_base = nodes.len();
        for cmp in [
            Comparator::Eq, Comparator::Neq,
            Comparator::Gt, Comparator::Gte,
            Comparator::Lt, Comparator::Lte,
            Comparator::Contains,
        ] {
            nodes.push(QueryNode::Comparator(cmp));
        }

        // Modifier nodes  [11..13]
        let mod_base = nodes.len();
        let fetch_mod_idx   = mod_base;
        let orderby_mod_idx = mod_base + 1;
        nodes.push(QueryNode::Modifier(ModifierKind::Fetch));
        nodes.push(QueryNode::Modifier(ModifierKind::OrderBy));
        nodes.push(QueryNode::Modifier(ModifierKind::Limit));

        // ── Schema nodes ────────────────────────────────────────────────────

        let schema_node_base = nodes.len();
        let mut table_map: HashMap<std::string::String, usize> = HashMap::new();

        // Pass 1: table nodes
        for table in schema.tables.iter() {
            let idx = nodes.len();
            nodes.push(QueryNode::Table(table.name.clone()));
            table_map.insert(table.name.clone(), idx);
        }

        // Pass 2: field nodes + structural edges
        // Collect record-link field indices for ModifierToField after all fields are created.
        let mut record_field_indices: Vec<usize> = Vec::new();
        let mut all_field_indices:    Vec<usize> = Vec::new();

        for table in schema.tables.iter() {
            let table_idx = *table_map.get(&table.name).unwrap();

            for field in table.fields.iter() {
                let field_idx = nodes.len();
                nodes.push(QueryNode::Field {
                    table: table.name.clone(),
                    name:  field.name.clone(),
                });

                // HasField / FieldOf
                edges.get_mut(&EdgeType::HasField).unwrap().push(Edge { src: table_idx, dst: field_idx });
                edges.get_mut(&EdgeType::FieldOf).unwrap().push(Edge  { src: field_idx, dst: table_idx });

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
            edges.get_mut(&EdgeType::ModifierToField).unwrap().push(Edge {
                src: fetch_mod_idx,
                dst: f,
            });
        }
        // OrderBy → all fields
        for &f in &all_field_indices {
            edges.get_mut(&EdgeType::ModifierToField).unwrap().push(Edge {
                src: orderby_mod_idx,
                dst: f,
            });
        }
        // Limit → nothing

        let _ = (op_base, cmp_base, schema_node_base); // suppress unused warnings

        Self { schema, nodes, edges }
    }

    pub fn inject(&self, semantics: &Semantics) -> GroundedGraph {
        todo!()
    }
}

// =============================================================================
// GroundedGraph
// =============================================================================

pub struct GroundedGraph {
    /// Flat node index space: schema nodes followed by span nodes added by inject().
    pub nodes: Vec<QueryNode>,

    /// Typed edges over node indices.
    pub edges: TypedEdges,

    /// Node indices of span nodes, in order:
    /// [intent, entity, projections..., conditions..., assignments..., modifiers...]
    pub span_indices: Vec<usize>,

    /// For each span node, the schema node indices it can resolve to (candidates).
    /// Used by the bilinear head to restrict scoring.
    pub span_candidates: Vec<Vec<usize>>,
}

impl GroundedGraph {
    pub fn forward(&self) -> QueryIr {
        todo!()
    }
}
