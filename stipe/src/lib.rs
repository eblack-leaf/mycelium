// stipe — main interface, connects UI to the network

use hyphae::{Query, Schema, SchemaGraph};
use septa::Semantics;
use std::path::Path;

pub struct Prompt {
    pub text: String,
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

    /// values: slot substitution strings indexed by slot number (0-based).
    pub fn query(&self, prompt: Prompt, values: &[String]) -> Query {
        let semantics = Semantics::parse(&prompt.text);
        let grounded = self.schema_graph.inject(&semantics);
        let ir = grounded.forward();
        ir.render(values)
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
        let query = mycelium.query(prompt, &[]);
        assert_eq!(query.surql, "SELECT * FROM user WHERE age > 30");
    }
}
