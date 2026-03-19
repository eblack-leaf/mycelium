// septa — semantic parsing of natural language prompts

pub mod model;

pub struct Semantics {
    pub slots: Slots,
}

impl Semantics {
    pub fn parse(text: &str) -> Self {
        todo!()
    }
}

pub struct Slots {
    pub intent:      Intent,
    pub entities:    Vec<Span>,   // noun phrases → tables/records
    pub projections: Vec<Span>,   // fields to return (empty = *)
    pub conditions:  Vec<Span>,   // comparison predicates → WHERE clauses
    pub assignments: Vec<Span>,   // field=value pairs → SET clauses (INSERT/UPDATE)
    pub modifiers:   Vec<Span>,   // ORDER BY, LIMIT, FETCH
}

/// Word-level span into the NL string.
pub struct Span {
    pub text:  String,
    pub start: usize,
    pub end:   usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Intent {
    Select,
    Insert,
    Update,
    Delete,
}
