use std::sync::Mutex;
use crate::bridge::{Block, Suggestion, Suggestions};

pub(crate) type DataM = Mutex<Data>;
pub(crate) struct Data {
    pub(crate) blocks: Vec<Block>,
    pub(crate) suggestions: Suggestions
}
impl Data {
    pub fn new() -> Self {
        Self {
            blocks: vec![],
            suggestions: Suggestions::default()
        }
    }
}