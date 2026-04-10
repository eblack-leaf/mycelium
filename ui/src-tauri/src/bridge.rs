use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Serialize, Deserialize, TS, Default)]
#[ts(export)]
pub(crate) struct Suggestion {
    pub(crate) text: String,
    pub(crate) metadata: String,
}
#[derive(Serialize, Deserialize, TS, Default)]
#[ts(export)]
pub(crate) struct Suggestions {
    pub(crate) placeholders: Vec<Suggestion>,
    pub(crate) ids: Vec<Suggestion>,
    pub(crate) schema: Vec<Suggestion>,
}
#[derive(Serialize, Deserialize, TS)]
#[ts(export)]
pub(crate) struct Block {
    pub(crate) text: String,
}