// =============================================================================
// reranker_data.rs — Training data types for the schema re-ranker
// =============================================================================

use serde::{Serialize, Deserialize};
use std::path::Path;

/// One training pair: phrase embedding + schema name embedding + label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerPair {
    /// 384-dim MiniLM embedding of the linguistic phrase
    pub phrase_emb: Vec<f32>,
    /// 384-dim MiniLM embedding of the schema node name
    pub schema_emb: Vec<f32>,
    /// 1.0 = match (ground truth), 0.0 = non-match
    pub label: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerDataset {
    pub pairs: Vec<RerankerPair>,
}

impl RerankerDataset {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let ds: Self = serde_json::from_str(&data)?;
        Ok(ds)
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
