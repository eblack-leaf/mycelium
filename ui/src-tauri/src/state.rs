use crate::bridge::{Block, BlockState, PlaceholderValue, Settings, Suggestions};
use std::sync::Mutex;

pub(crate) type DataM = Mutex<Data>;

pub(crate) struct Data {
    pub(crate) blocks: Vec<Block>,
    pub(crate) suggestions: Suggestions,
    pub(crate) values: Vec<PlaceholderValue>,
    pub(crate) settings: Settings,
    next_id: u64,
}

impl Data {
    pub fn new() -> Self {
        let mut data = Self {
            blocks: vec![],
            suggestions: Suggestions::default(),
            values: vec![],
            settings: Settings::default(),
            next_id: 0,
        };
        // Seed schema suggestions with SurrealQL keywords
        data.suggestions.schema = SURREAL_KEYWORDS
            .iter()
            .map(|kw| crate::bridge::Suggestion {
                text: kw.to_string(),
                metadata: "keyword".to_string(),
            })
            .collect();
        // Start with one empty composing block
        let id = data.new_id();
        data.blocks.push(Block {
            id,
            query: String::new(),
            state: BlockState::Composing,
            result: None,
        });
        data
    }

    pub(crate) fn new_id(&mut self) -> String {
        let id = format!("block-{}", self.next_id);
        self.next_id += 1;
        id
    }
}

const SURREAL_KEYWORDS: &[&str] = &[
    "SELECT", "CREATE", "UPDATE", "DELETE", "RELATE", "RETURN", "INSERT",
    "UPSERT", "DEFINE", "REMOVE", "INFO", "USE", "LET", "IF", "ELSE",
    "FOR", "BREAK", "CONTINUE", "BEGIN", "COMMIT", "CANCEL", "THROW",
    "SLEEP", "SHOW", "LIVE", "KILL",
    "FROM", "WHERE", "SET", "MERGE", "CONTENT", "REPLACE", "UNSET",
    "LIMIT", "ORDER", "GROUP", "SPLIT", "FETCH", "START", "BY", "ONLY",
    "WITH", "TIMEOUT", "PARALLEL", "EXPLAIN", "TEMPFILES",
    "ASC", "DESC", "AND", "OR", "NOT", "IS", "IN", "NONE", "NULL",
    "TRUE", "FALSE", "TYPE", "ASSERT", "VALUE", "DEFAULT", "READONLY",
    "PERMISSIONS", "FLEXIBLE", "SCHEMAFULL", "SCHEMALESS",
    "ON", "FOR", "FIELD", "INDEX", "TABLE", "SCOPE", "PARAM", "FUNCTION",
    "UNIQUE", "SEARCH", "ANALYZER", "NAMESPACE", "DATABASE",
    "math::sum", "math::mean", "math::min", "math::max",
    "array::len", "array::push", "array::pop", "array::distinct",
    "string::len", "string::lowercase", "string::uppercase",
    "time::now", "type::thing", "type::string", "type::int", "type::float",
    "count", "rand", "rand::uuid",
];
