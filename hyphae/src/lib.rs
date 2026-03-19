// hyphae — graph neural network

use regex::Regex;
use septa::Semantics;
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
        tables: Vec<std::string::String>,
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
        raw: std::string::String,
    },
    Range {
        raw: std::string::String,
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
// SchemaGraph
// =============================================================================

pub struct SchemaGraph {
    pub schema: Schema,
}

impl SchemaGraph {
    pub fn new(schema: Schema) -> Self {
        Self { schema }
    }

    pub fn inject(&self, semantics: &Semantics) -> GroundedGraph {
        todo!()
    }
}

pub struct GroundedGraph {}

impl GroundedGraph {
    pub fn forward(&self) -> Predictions {
        todo!()
    }
}

/// Unordered output from the GNN head — scored nodes, fields, and operation.
/// Ordering is recovered later by threading through the original Semantics.
pub struct Predictions {
    pub tables: Vec<ScoredTable>,
    pub fields: Vec<ScoredField>,
    pub operation: Operation,
}

pub struct ScoredTable {
    pub name: String,
    pub score: f32,
}

pub struct ScoredField {
    pub table: String,
    pub name: String,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Select,
    Insert,
    Update,
    Delete,
    Relate,
}
