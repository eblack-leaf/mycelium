// =============================================================================
// grounding.rs — NL extraction model (Grounding)
//
// Takes NL text + schema + operation vocabulary, produces an Extraction.
// =============================================================================

use std::path::Path;
use crate::schema::Schema;
use crate::operations::OpNode;
use crate::intent::Extraction;

pub struct GroundingConfig {
    pub model_path: String,
}

pub struct GroundingModel {
    // TODO: model weights, tokenizer, etc.
}

impl GroundingModel {
    pub fn load(_config: &GroundingConfig) -> Self {
        todo!()
    }

    pub fn extract(
        &self,
        _nl_query: &str,
        _schema: &Schema,
        _operations: &[OpNode],
    ) -> Extraction {
        todo!()
    }
}
