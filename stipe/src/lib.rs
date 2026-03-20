// stipe — main interface, connects UI to the network

use burn::tensor::backend::Backend;
use hyphae::graph::SchemaGraph;
use hyphae::model::Hyphae;
use hyphae::query::Query;
use hyphae::schema::Schema;
use septa::model::Septa;
use std::path::Path;

pub struct Prompt {
    pub text: String,
}

pub struct Mycelium<B: Backend> {
    pub schema:       Schema,
    pub schema_graph: SchemaGraph,
    pub septa:        Septa<B>,
    pub hyphae:       Hyphae<B>,
}

impl<B: Backend> Mycelium<B> {
    pub fn new(
        schema_dir:    &Path,
        ngram_buckets: usize,
        septa:         Septa<B>,
        hyphae:        Hyphae<B>,
    ) -> std::io::Result<Self> {
        let schema = Schema::from_dir(schema_dir)?;
        let schema_graph = SchemaGraph::new(schema.clone(), ngram_buckets);
        Ok(Self { schema, schema_graph, septa, hyphae })
    }

    /// values: slot substitution strings indexed by slot number (0-based).
    pub fn query(&self, prompt: Prompt, values: &[String], device: &B::Device) -> Query {
        let tokens: Vec<&str> = prompt.text.split_whitespace().collect();
        let (semantics, hiddens) = self.septa.forward(&tokens, device);
        let grounded = self.schema_graph.inject(&semantics);
        let ir = self.hyphae.forward(&grounded, &hiddens, device);
        ir.render(values)
    }
}
