// stipe — main interface, connects UI to the network

use hyphae::{Predictions, Schema, SchemaGraph};
use septa::Semantics;
use std::path::Path;

pub struct Prompt {
    pub text: String,
}

pub struct QueryIr {}

impl QueryIr {
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
        let ir = orchestrate(predictions, &semantics);
        ir.render()
    }
}

/// Re-orders unordered GNN predictions using the semantic structure of the
/// original prompt as a skeleton, producing a QueryIr ready for rendering.
fn orchestrate(predictions: Predictions, semantics: &Semantics) -> QueryIr {
    todo!()
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
