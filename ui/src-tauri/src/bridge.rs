use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Serialize, Deserialize, TS)]
#[ts(export)]
pub(crate) struct Suggestion {
    pub(crate) text: String,
    pub(crate) metadata: String,
}
#[derive(Serialize, Deserialize, TS)]
#[ts(export)]
pub(crate) struct Block {
    pub(crate) text: String,
}