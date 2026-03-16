use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::condition::Op;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub fields: Vec<Field>,
    #[serde(default)]
    pub phrases: Vec<PhraseDef>,
    #[serde(default)]
    pub temporal_markers: Vec<String>,
    #[serde(default)]
    pub op_phrases: Vec<OpPhrase>,
}

/// Maps natural language phrases to an operator.
/// e.g. ["less than", "under", "below", "fewer than"] → Lt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpPhrase {
    pub triggers: Vec<String>,
    pub op: Op,
}

/// A compound phrase that expands to a known condition.
/// e.g. "out of stock" → stock = 0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseDef {
    pub triggers: Vec<String>,
    pub field: String,
    pub op: Op,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub ty: FieldType,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub common_ops: Vec<Op>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    Int,
    Float,
    String,
    Bool,
    Datetime,
    Duration,
    Array,
    Record,
    Object,
}

impl Schema {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let data = std::fs::read_to_string(path).map_err(SchemaError::Io)?;
        serde_json::from_str(&data).map_err(SchemaError::Parse)
    }

    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// All names + aliases mapped to their parent field.
    pub fn field_vocabulary(&self) -> HashMap<&str, &Field> {
        let mut map = HashMap::new();
        for field in &self.fields {
            map.insert(field.name.as_str(), field);
            for alias in &field.aliases {
                map.insert(alias.as_str(), field);
            }
        }
        map
    }
}

#[derive(Debug)]
pub enum SchemaError {
    Io(std::io::Error),
    Parse(serde_json::Error),
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "schema io: {e}"),
            Self::Parse(e) => write!(f, "schema parse: {e}"),
        }
    }
}

impl std::error::Error for SchemaError {}
