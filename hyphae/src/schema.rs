use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Parsed output of `INFO FOR DB` — top-level DB info.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DbInfo {
    pub tables: HashMap<String, String>,     // name → definition
    pub functions: HashMap<String, String>,
    pub analyzers: HashMap<String, String>,
}

/// Parsed output of `INFO FOR TABLE <name>`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TableInfo {
    pub name: String,
    pub fields: HashMap<String, FieldInfo>,
    pub indexes: HashMap<String, String>,
    pub events: HashMap<String, String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FieldInfo {
    pub kind: Option<String>,  // "string", "int", "record<user>", etc.
    pub value: Option<String>,
    pub assert: Option<String>,
}

/// Flat list of completable schema tokens derived from DB info.
#[derive(Debug)]
pub struct SchemaCompletions {
    pub table_names: Vec<String>,
    pub field_names: Vec<String>, // deduplicated across all tables
}

impl SchemaCompletions {
    pub fn from_db(db: &DbInfo, tables: &[TableInfo]) -> Self {
        let table_names = db.tables.keys().cloned().collect();
        let mut field_names: Vec<String> = tables
            .iter()
            .flat_map(|t| t.fields.keys().cloned())
            .collect();
        field_names.sort();
        field_names.dedup();
        Self { table_names, field_names }
    }

    /// Flatten into (text, metadata) pairs for the completion suggestion pool.
    pub fn to_suggestions(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for t in &self.table_names {
            out.push((t.clone(), "table".to_string()));
        }
        for f in &self.field_names {
            out.push((f.clone(), "field".to_string()));
        }
        out
    }
}

/// Parse the raw JSON body from POST /sql into `DbInfo`.
/// REST response: `[{ "status": "OK", "result": { "tables": {...}, ... }, "time": "..." }]`
pub fn parse_db_info(json: &str) -> Option<DbInfo> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let envelope = v.get(0).unwrap_or(&v);
    let inner = envelope.get("result").unwrap_or(envelope);
    serde_json::from_value(inner.clone()).ok()
}

/// Parse the raw JSON body from POST /sql into `TableInfo`.
pub fn parse_table_info(name: &str, json: &str) -> Option<TableInfo> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let envelope = v.get(0).unwrap_or(&v);
    let inner = envelope.get("result").unwrap_or(envelope);
    let mut info: TableInfo = serde_json::from_value(inner.clone()).ok()?;
    info.name = name.to_string();
    Some(info)
}
