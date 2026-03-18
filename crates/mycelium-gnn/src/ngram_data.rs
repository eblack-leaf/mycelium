// =============================================================================
// ngram_data.rs — Data types + concept mapping for n-gram cross-attention
//
// ConceptMap flattens all schema nodes (tables + fields + operations) into
// a single contiguous index space for the learned embedding table:
//   [0, n_tables)                          → tables
//   [n_tables, n_tables + n_fields)        → fields
//   [n_tables + n_fields, total)           → operations
// =============================================================================

use serde::{Serialize, Deserialize};

/// One labeled span in the n-gram dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramSpanLabel {
    pub start_word: usize,
    pub end_word: usize,       // exclusive
    pub span_type: usize,      // 0=NP, 1=Quant, 2=Comp, 3=Intent
    pub concept_idx: usize,    // index into flattened concept table
}

/// One training sample: a query with its labeled spans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramSample {
    pub query: String,
    pub spans: Vec<NgramSpanLabel>,
}

/// Full dataset of n-gram training samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramDataset {
    pub samples: Vec<NgramSample>,
}

impl NgramDataset {
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let ds: Self = serde_json::from_str(&data)?;
        Ok(ds)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let data = serde_json::to_string(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}

/// Maps (schema_type, schema_id) ↔ flat concept index.
#[derive(Debug, Clone)]
pub struct ConceptMap {
    pub n_tables: usize,
    pub n_fields: usize,
    pub n_ops: usize,
    /// Human-readable names for each concept index (for debug printing).
    pub names: Vec<String>,
}

impl ConceptMap {
    pub fn new(
        table_names: &[String],
        field_names: &[String],
        op_names: &[String],
    ) -> Self {
        let mut names = Vec::with_capacity(table_names.len() + field_names.len() + op_names.len());
        names.extend(table_names.iter().cloned());
        names.extend(field_names.iter().cloned());
        names.extend(op_names.iter().cloned());
        Self {
            n_tables: table_names.len(),
            n_fields: field_names.len(),
            n_ops: op_names.len(),
            names,
        }
    }

    pub fn total(&self) -> usize {
        self.n_tables + self.n_fields + self.n_ops
    }

    /// Convert (schema_type, schema_id) → flat concept index.
    pub fn to_idx(&self, schema_type: &str, schema_id: usize) -> usize {
        match schema_type {
            "table" => schema_id,
            "field" => self.n_tables + schema_id,
            "operation" => self.n_tables + self.n_fields + schema_id,
            _ => 0,
        }
    }

    /// Convert flat concept index → (schema_type, schema_id).
    pub fn from_idx(&self, idx: usize) -> (&str, usize) {
        if idx < self.n_tables {
            ("table", idx)
        } else if idx < self.n_tables + self.n_fields {
            ("field", idx - self.n_tables)
        } else {
            ("operation", idx - self.n_tables - self.n_fields)
        }
    }

    /// Get human-readable name for a concept index.
    pub fn name(&self, idx: usize) -> &str {
        self.names.get(idx).map(|s| s.as_str()).unwrap_or("?")
    }
}
