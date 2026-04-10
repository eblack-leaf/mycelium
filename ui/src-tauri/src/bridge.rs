use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) enum BlockState {
    Composing,
    Executing,
    Done,
}

#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) struct Block {
    pub(crate) id: String,
    pub(crate) query: String,
    pub(crate) state: BlockState,
    pub(crate) result: Option<String>,
}

#[derive(Serialize, Deserialize, TS, Default, Clone)]
#[ts(export)]
pub(crate) struct Suggestion {
    pub(crate) text: String,
    pub(crate) metadata: String,
}

#[derive(Serialize, Deserialize, TS, Default, Clone)]
#[ts(export)]
pub(crate) struct Suggestions {
    pub(crate) placeholders: Vec<Suggestion>,
    pub(crate) schema: Vec<Suggestion>,
    pub(crate) other: Vec<Suggestion>,
}

#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) struct PlaceholderValue {
    pub(crate) name: String,
    pub(crate) value: String,
}

#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) struct Settings {
    pub(crate) surreal_endpoint: String,
    pub(crate) placeholder_prefix: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            surreal_endpoint: "ws://localhost:8000".to_string(),
            placeholder_prefix: "@".to_string(),
        }
    }
}
