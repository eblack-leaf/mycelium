// =============================================================================
// schema.rs — Parsed SurrealQL schema: tables, fields, types
// =============================================================================

use std::path::Path;
use regex::Regex;

// =============================================================================
// Reader — stateless file/directory concatenator
// =============================================================================

pub struct Reader;

impl Reader {
    /// Read a file or directory of .surql/.sql files into a single string.
    pub fn read(path: &Path) -> std::io::Result<String> {
        if path.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|ext| ext == "surql" || ext == "sql")
                })
                .collect();
            entries.sort_by_key(|e| e.path());

            let mut combined = String::new();
            for entry in entries {
                combined.push_str(&std::fs::read_to_string(entry.path())?);
                combined.push('\n');
            }
            Ok(combined)
        } else {
            std::fs::read_to_string(path)
        }
    }
}

// =============================================================================
// Extractor — stateless parser, returns (ParsedSchema, Validation)
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct Validation {
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl Validation {
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

pub struct Extractor;

impl Extractor {
    pub fn extract(raw: &str) -> (Schema, Validation) {
        let mut validation = Validation::default();

        let table_re = Regex::new(
            r"(?i)DEFINE\s+TABLE\s+(?:(?:OVERWRITE|IF\s+NOT\s+EXISTS)\s+)?(\w+)"
        ).unwrap();

        let field_re = Regex::new(
            r"(?i)DEFINE\s+FIELD\s+(?:(?:OVERWRITE|IF\s+NOT\s+EXISTS)\s+)?(\S+)\s+ON\s+(?:TABLE\s+)?(\w+)(?:\s+TYPE\s+(.+?))?\s*;"
        ).unwrap();

        let mut schema = Schema::default();

        // Pass 1: collect all tables
        for cap in table_re.captures_iter(raw) {
            schema.tables.push(Table {
                name: cap[1].to_string(),
                fields: Vec::new(),
            });
        }

        // Pass 2: collect fields and attach to their tables
        for cap in field_re.captures_iter(raw) {
            let field_name = cap[1].to_string();
            let table_name = cap[2].to_string();
            let type_str = cap.get(3).map(|m| m.as_str().trim());

            let field_type = match type_str {
                Some(t) => parse_type(t, &mut validation),
                None => {
                    validation.warnings.push(format!(
                        "field `{}` on `{}` has no TYPE, defaulting to Any",
                        field_name, table_name
                    ));
                    FieldType::Any
                }
            };

            let field = Field {
                name: field_name.clone(),
                field_type,
            };

            match schema.tables.iter_mut().find(|t| t.name == table_name) {
                Some(table) => table.fields.push(field),
                None => {
                    validation.errors.push(format!(
                        "field `{}` references table `{}` which was not defined",
                        field_name, table_name
                    ));
                }
            }
        }

        (schema, validation)
    }
}

fn parse_type(raw: &str, validation: &mut Validation) -> FieldType {
    let s = raw.trim();

    // option<T>
    if let Some(inner) = strip_wrapper(s, "option") {
        return FieldType::Option {
            inner: Box::new(parse_type(inner, validation)),
        };
    }

    // array<T> or array<T, N>
    if s == "array" {
        return FieldType::Array { inner: None, max_len: None };
    }
    if let Some(inner) = strip_wrapper(s, "array") {
        let (type_str, len) = split_type_and_len(inner);
        return FieldType::Array {
            inner: Some(Box::new(parse_type(type_str, validation))),
            max_len: len,
        };
    }

    // set<T> or set<T, N>
    if s == "set" {
        return FieldType::Set { inner: None, max_len: None };
    }
    if let Some(inner) = strip_wrapper(s, "set") {
        let (type_str, len) = split_type_and_len(inner);
        return FieldType::Set {
            inner: Some(Box::new(parse_type(type_str, validation))),
            max_len: len,
        };
    }

    // record or record<table> or record<a | b>
    if s == "record" {
        return FieldType::Record { tables: Vec::new() };
    }
    if let Some(inner) = strip_wrapper(s, "record") {
        let tables: Vec<String> = inner
            .split('|')
            .map(|t| t.trim().to_string())
            .collect();
        return FieldType::Record { tables };
    }

    // geometry<variant>
    if let Some(inner) = strip_wrapper(s, "geometry") {
        let variant = match inner.trim() {
            "feature" => GeometryVariant::Feature,
            "point" => GeometryVariant::Point,
            "line" => GeometryVariant::Line,
            "polygon" => GeometryVariant::Polygon,
            "multipoint" => GeometryVariant::MultiPoint,
            "multiline" => GeometryVariant::MultiLine,
            "multipolygon" => GeometryVariant::MultiPolygon,
            "collection" => GeometryVariant::Collection,
            other => {
                validation.warnings.push(format!("unknown geometry variant: `{}`", other));
                GeometryVariant::Point
            }
        };
        return FieldType::Geometry { variant };
    }

    // scalar types
    match s.to_lowercase().as_str() {
        "any" => FieldType::Any,
        "bool" => FieldType::Bool,
        "string" => FieldType::String,
        "int" => FieldType::Int,
        "float" => FieldType::Float,
        "decimal" => FieldType::Decimal,
        "number" => FieldType::Number,
        "datetime" => FieldType::Datetime,
        "duration" => FieldType::Duration,
        "bytes" => FieldType::Bytes,
        "object" => FieldType::Object,
        "regex" => FieldType::Regex,
        other => {
            if other.contains("..") {
                FieldType::Range { raw: other.to_string() }
            } else if other.contains('|') {
                FieldType::Literal { raw: other.to_string() }
            } else {
                validation.warnings.push(format!("unrecognized type `{}`, defaulting to Any", other));
                FieldType::Any
            }
        }
    }
}

/// Strip a wrapper like `option<...>` and return the inner content.
fn strip_wrapper<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let lower = s.to_lowercase();
    if lower.starts_with(prefix) {
        let rest = &s[prefix.len()..];
        let rest = rest.trim();
        if rest.starts_with('<') && rest.ends_with('>') {
            Some(&rest[1..rest.len() - 1])
        } else {
            None
        }
    } else {
        None
    }
}

/// Split "string, 10" into ("string", Some(10))
fn split_type_and_len(s: &str) -> (&str, Option<usize>) {
    if let Some((type_part, len_part)) = s.rsplit_once(',') {
        match len_part.trim().parse::<usize>() {
            Ok(n) => (type_part.trim(), Some(n)),
            Err(_) => (s, None),
        }
    } else {
        (s, None)
    }
}

// =============================================================================
// Schema types
// =============================================================================

/// Structured schema extracted from DEFINE TABLE + DEFINE FIELD statements.
#[derive(Debug, Clone, Default)]
pub struct Schema {
    pub tables: Vec<Table>,
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
    Record { tables: Vec<String> },
    Array { inner: Option<Box<FieldType>>, max_len: Option<usize> },
    Set { inner: Option<Box<FieldType>>, max_len: Option<usize> },
    Option { inner: Box<FieldType> },
    Geometry { variant: GeometryVariant },
    Literal { raw: String },
    Range { raw: String },
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
