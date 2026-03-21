// stipe — main interface: text → SurrealQL via pretrained Basidium

use basidium::trainable::Basidium;
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use hyphae::graph::SchemaGraph;
use hyphae::model::{Hyphae, HyphaeConfig};
use hyphae::query::Query;
use hyphae::schema::Schema;
use septa::model::SeptaConfig;
use std::path::Path;

pub struct Mycelium<B: Backend> {
    pub schema: Schema,
    pub schema_graph: SchemaGraph,
    pub model: Basidium<B>,
    pub septa_config: SeptaConfig,
}

impl<B: Backend> Mycelium<B> {
    /// Load a pretrained Basidium from weights on disk.
    pub fn load(
        schema_dir: &Path,
        weights_path: &Path,
        hyphae_config: &HyphaeConfig,
        septa_config: &SeptaConfig,
        device: &B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let schema = Schema::from_dir(schema_dir)?;
        let schema_graph = SchemaGraph::new(schema.clone(), hyphae_config.ngram_buckets);

        let hyphae = Hyphae::new(hyphae_config, device);
        let septa = septa::model::Septa::new(septa_config, device);
        let mut model = Basidium { septa, hyphae };

        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(weights_path.to_path_buf(), device)
            .map_err(|e| format!("failed to load weights: {e}"))?;
        model = model.load_record(record);

        Ok(Self { schema, schema_graph, model, septa_config: septa_config.clone() })
    }

    /// Full pipeline: raw text → SurrealQL.
    /// Septa parses text into Semantics + SpanHiddens, Hyphae resolves, render to SurrealQL.
    /// `values` provides slot substitutions (values[n] replaces Slot(n)).
    pub fn query(&self, text: &str, values: &[String], device: &B::Device) -> Query {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let (semantics, hiddens) = self.model.septa.forward(&tokens, device);
        let grounded = self.schema_graph.inject(&semantics);
        let logits = self.model.hyphae.forward(&grounded, &hiddens, device);
        let ir = Hyphae::resolve(&logits, &grounded, &semantics);
        ir.render(values)
    }
}
