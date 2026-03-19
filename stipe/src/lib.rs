// stipe — main interface, connects UI to the network

use hyphae::{Predictions, Schema, SchemaGraph};
use septa::{Comparator, Semantics};
use std::path::Path;

pub struct Prompt {
    pub text: String,
}

/// GNN-resolved query structure ready for deterministic rendering.
pub struct QueryIr {
    pub intent:      septa::Intent,
    pub entities:    Vec<String>,             // resolved table names → FROM clause
    pub projections: Vec<ResolvedField>,      // empty = SELECT *
    pub conditions:  Vec<ResolvedCondition>,
    pub assignments: Vec<ResolvedAssignment>,
    pub modifiers:   Vec<ResolvedModifier>,
}

/// A field resolved to a specific table by the GNN.
pub struct ResolvedField {
    pub table: String,
    pub field: String,
}

pub struct ResolvedCondition {
    pub table:      String,
    pub field:      String,
    pub comparator: Comparator,
    pub value:      String,
}

pub struct ResolvedAssignment {
    pub table: String,
    pub field: String,
    pub value: String,
}

pub enum ResolvedModifier {
    OrderBy { field: String, descending: bool },
    Limit   { n: usize },
    Fetch   { field: String },
}

impl QueryIr {
    pub fn new(predictions: Predictions, semantics: &Semantics) -> Self {
        todo!()
    }

    pub fn render(&self) -> Query {
        todo!()
    }
}

pub struct Query {
    pub surql: String,
}

pub struct Mycelium {
    pub schema: Schema,
    pub schema_graph: SchemaGraph,
}

impl Mycelium {
    pub fn new(schema_dir: &Path) -> std::io::Result<Self> {
        let schema = Schema::from_dir(schema_dir)?;
        let schema_graph = SchemaGraph::new(schema.clone());
        Ok(Self {
            schema,
            schema_graph,
        })
    }

    pub fn query(&self, prompt: Prompt) -> Query {
        let semantics = Semantics::parse(&prompt.text);
        let grounded = self.schema_graph.inject(&semantics);
        let predictions = grounded.forward();
        let ir = QueryIr::new(predictions, &semantics);
        ir.render()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_translates_nl_to_surql() {
        let mycelium = Mycelium::new(Path::new("fixtures/schema")).unwrap();
        let prompt = Prompt {
            text: "find all users older than 30".to_string(),
        };
        let query = mycelium.query(prompt);
        assert_eq!(query.surql, "SELECT * FROM user WHERE age > 30");
    }
}
