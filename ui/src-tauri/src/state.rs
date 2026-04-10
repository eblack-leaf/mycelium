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
    // Statements
    "SELECT", "CREATE", "UPDATE", "DELETE", "RELATE", "RETURN", "INSERT",
    "UPSERT", "DEFINE", "REMOVE", "INFO", "USE", "LET", "IF", "ELSE",
    "THEN", "END", "FOR", "BREAK", "CONTINUE", "BEGIN", "COMMIT", "CANCEL",
    "THROW", "SLEEP", "SHOW", "LIVE", "KILL", "REBUILD", "OPTION",
    // Clauses
    "FROM", "WHERE", "SET", "MERGE", "CONTENT", "REPLACE", "UNSET",
    "LIMIT", "ORDER", "GROUP", "SPLIT", "FETCH", "START", "BY", "ONLY",
    "WITH", "TIMEOUT", "PARALLEL", "EXPLAIN", "TEMPFILES", "OMIT",
    "BEFORE", "AFTER", "DIFF", "WHEN", "OVERWRITE", "NOINDEX",
    // Operators / logic
    "ASC", "DESC", "AND", "OR", "NOT", "IS", "IN", "NONE", "NULL",
    "CONTAINS", "CONTAINSALL", "CONTAINSANY", "CONTAINSNONE",
    "INSIDE", "NOTINSIDE", "ALLINSIDE", "ANYINSIDE", "NONEINSIDE",
    "OUTSIDE", "INTERSECTS",
    // Values
    "TRUE", "FALSE", "FUTURE",
    // Schema
    "TYPE", "ASSERT", "VALUE", "DEFAULT", "READONLY", "FLEXIBLE",
    "PERMISSIONS", "SCHEMAFULL", "SCHEMALESS", "ENFORCED",
    "ON", "FIELD", "INDEX", "TABLE", "SCOPE", "PARAM", "FUNCTION",
    "UNIQUE", "SEARCH", "ANALYZER", "NAMESPACE", "DATABASE",
    "EVENT", "RELATION", "REFERENCES",
    // Types
    "ANY", "ARRAY", "BOOL", "BYTES", "DATETIME", "DECIMAL", "DURATION",
    "FLOAT", "GEOMETRY", "INT", "NUMBER", "OBJECT", "RECORD", "STRING",
    "UUID",
    // Auth
    "SIGNIN", "SIGNUP", "AUTHENTICATE", "TOKEN", "SESSION",
    // Built-in functions
    "math::sum", "math::mean", "math::min", "math::max", "math::abs",
    "array::len", "array::push", "array::pop", "array::distinct", "array::flatten",
    "string::len", "string::lowercase", "string::uppercase", "string::trim", "string::concat",
    "time::now", "time::day", "time::month", "time::year", "time::format",
    "type::thing", "type::string", "type::int", "type::float", "type::bool", "type::uuid",
    "crypto::md5", "crypto::sha1", "crypto::sha256",
    "count", "rand", "rand::uuid", "rand::string", "rand::int", "rand::float",
    "meta::id", "meta::tb",
];
