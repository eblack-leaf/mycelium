// stipe — main interface, connects UI to the network

use std::path::Path;
use septa::Semantics;
use hyphae::{Schema, SchemaGraph};

pub struct Prompt {
    pub text: String,
}

pub struct Query {
    pub surql: String,
}

pub struct Mycelium {
    pub schema:       Schema,
    pub schema_graph: SchemaGraph,
}

impl Mycelium {
    pub fn new(schema_dir: &Path) -> std::io::Result<Self> {
        let schema       = Schema::from_dir(schema_dir)?;
        let schema_graph = SchemaGraph::new(schema.clone());
        Ok(Self { schema, schema_graph })
    }

    pub fn query(&self, prompt: Prompt) -> Query {
        let semantics = Semantics::parse(&prompt.text);
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_translates_nl_to_surql() {
        let mycelium = Mycelium::new(Path::new("fixtures/schema")).unwrap();
        let prompt = Prompt { text: "find all users older than 30".to_string() };
        let query = mycelium.query(prompt);
        assert_eq!(query.surql, "SELECT * FROM user WHERE age > 30");
    }
}
