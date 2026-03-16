// =============================================================================
// embed.rs — Initial node embeddings from GloVe + type vectors
//
// Produces HashMap<node_type, Tensor[n_nodes, embed_dim]> for the Encoder.
// =============================================================================

use std::collections::HashMap;
use std::path::Path;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use super::graph::SchemaGraph;
use super::query_graph::QueryGraph;
use super::schema::{Schema, FieldType};

/// Preloaded GloVe vectors: word → f32 vector.
#[derive(Debug, Clone)]
pub struct GloveVocab {
    pub vectors: HashMap<String, Vec<f32>>,
    pub dim: usize,
    pub seed: u64,
}

impl GloveVocab {
    /// Load from a GloVe text file (e.g. glove.6B.50d.txt).
    /// Each line: word f1 f2 f3 ...
    pub fn load(path: &Path, seed: u64) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut vectors = HashMap::new();
        let mut dim = 0;

        for line in content.lines() {
            let mut parts = line.split_whitespace();
            let word = match parts.next() {
                Some(w) => w.to_string(),
                None => continue,
            };
            let vals: Vec<f32> = parts.filter_map(|v| v.parse().ok()).collect();
            if dim == 0 {
                dim = vals.len();
            }
            if vals.len() == dim {
                vectors.insert(word, vals);
            }
        }

        Ok(Self { vectors, dim, seed })
    }

    /// Look up a word, averaging tokens if multi-word. Random init for unknown words.
    fn embed(&self, text: &str) -> Vec<f32> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.is_empty() {
            return self.random_init(0);
        }

        let mut sum = vec![0.0f32; self.dim];
        let mut count = 0;
        let mut unknown = 0u64;

        for token in &tokens {
            let lower = token.to_lowercase();
            if let Some(vec) = self.vectors.get(&lower) {
                for (i, v) in vec.iter().enumerate() {
                    sum[i] += v;
                }
                count += 1;
            } else {
                // Unknown word — add random init so it's not a dead signal
                let rand_vec = self.random_init(unknown);
                for (i, v) in rand_vec.iter().enumerate() {
                    sum[i] += v;
                }
                count += 1;
                unknown += 1;
            }
        }

        let scale = 1.0 / count as f32;
        sum.iter_mut().for_each(|v| *v *= scale);

        sum
    }

    fn random_init(&self, salt: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(salt));
        (0..self.dim).map(|_| rng.random_range(-0.1..0.1)).collect()
    }
}

/// Learned type embeddings — one vector per FieldType variant + "table".
/// These are trainable parameters.
#[derive(Debug, Clone)]
pub struct TypeVocab {
    pub vectors: HashMap<String, Vec<f32>>,
    pub dim: usize,
    pub seed: u64,
}

impl TypeVocab {
    /// Initialize with small random values from a seeded RNG.
    pub fn new(dim: usize, seed: u64) -> Self {
        let type_names = [
            "table", "any", "bool", "string", "int", "float", "decimal",
            "number", "datetime", "duration", "bytes", "object", "regex",
            "record", "array", "set", "option", "geometry", "literal", "range",
        ];

        let mut rng = StdRng::seed_from_u64(seed);
        let mut vectors = HashMap::new();

        for name in &type_names {
            let vec: Vec<f32> = (0..dim)
                .map(|_| rng.random_range(-0.1..0.1))
                .collect();
            vectors.insert(name.to_string(), vec);
        }

        Self { vectors, dim, seed }
    }

    fn embed(&self, type_key: &str) -> Vec<f32> {
        self.vectors
            .get(type_key)
            .cloned()
            .unwrap_or_else(|| self.random_init())
    }

    /// Small random vector for query nodes whose type is unknown.
    fn random_init(&self) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(0xDEAD));
        (0..self.dim).map(|_| rng.random_range(-0.1..0.1)).collect()
    }
}

/// Produces initial embeddings for all nodes in the combined graph.
pub struct Embedder {
    pub glove: GloveVocab,
    pub types: TypeVocab,
}

impl Embedder {
    pub fn new(glove: GloveVocab, type_dim: usize, seed: u64) -> Self {
        Self {
            glove,
            types: TypeVocab::new(type_dim, seed),
        }
    }

    /// Total embedding dim: glove_dim + type_dim
    pub fn dim(&self) -> usize {
        self.glove.dim + self.types.dim
    }

    /// Produce initial embeddings for every node type.
    pub fn embed_all<B: Backend>(
        &self,
        schema: &Schema,
        schema_graph: &SchemaGraph,
        query_graph: &QueryGraph,
        device: &B::Device,
    ) -> HashMap<String, Tensor<B, 2>> {
        let mut result = HashMap::new();
        let dim = self.dim();

        // --- Schema: table nodes ---
        let table_feats: Vec<f32> = schema_graph.table_nodes.iter().map(|node| {
            let word = self.glove.embed(&node.name);
            let typ = self.types.embed("table");
            concat(&word, &typ)
        }).flatten().collect();

        if !schema_graph.table_nodes.is_empty() {
            result.insert("table".to_string(), to_tensor::<B>(
                table_feats, schema_graph.table_nodes.len(), dim, device,
            ));
        }

        // --- Schema: field nodes ---
        let field_feats: Vec<f32> = schema_graph.field_nodes.iter().map(|node| {
            // Find this field's type from the schema
            let type_key = find_field_type(schema, &node.name);
            let word = self.glove.embed(&node.name);
            let typ = self.types.embed(&type_key);
            concat(&word, &typ)
        }).flatten().collect();

        if !schema_graph.field_nodes.is_empty() {
            result.insert("field".to_string(), to_tensor::<B>(
                field_feats, schema_graph.field_nodes.len(), dim, device,
            ));
        }

        // --- Query: collection candidates ---
        let coll_feats: Vec<f32> = query_graph.collections.iter().map(|c| {
            let word = self.glove.embed(&c.surface_form);
            let typ = self.types.embed("table");
            concat(&word, &typ)
        }).flatten().collect();

        if !query_graph.collections.is_empty() {
            result.insert("q_collection".to_string(), to_tensor::<B>(
                coll_feats, query_graph.collections.len(), dim, device,
            ));
        }

        // --- Query: field candidates ---
        let field_cand_feats: Vec<f32> = query_graph.fields.iter().map(|f| {
            let word = self.glove.embed(&f.surface_form);
            let typ = self.types.random_init();
            concat(&word, &typ)
        }).flatten().collect();

        if !query_graph.fields.is_empty() {
            result.insert("q_field".to_string(), to_tensor::<B>(
                field_cand_feats, query_graph.fields.len(), dim, device,
            ));
        }

        // --- Query: filter candidates ---
        let filter_feats: Vec<f32> = query_graph.filters.iter().map(|f| {
            let word = self.glove.embed(&format!("{} {}", f.operator, f.value));
            let typ = vec![0.0; self.types.dim];
            concat(&word, &typ)
        }).flatten().collect();

        if !query_graph.filters.is_empty() {
            result.insert("q_filter".to_string(), to_tensor::<B>(
                filter_feats, query_graph.filters.len(), dim, device,
            ));
        }

        // --- Query: traversal candidates ---
        let trav_feats: Vec<f32> = query_graph.traversals.iter().map(|t| {
            let word = self.glove.embed(&t.surface_form);
            let typ = vec![0.0; self.types.dim];
            concat(&word, &typ)
        }).flatten().collect();

        if !query_graph.traversals.is_empty() {
            result.insert("q_traversal".to_string(), to_tensor::<B>(
                trav_feats, query_graph.traversals.len(), dim, device,
            ));
        }

        result
    }
}

fn concat(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

fn to_tensor<B: Backend>(
    data: Vec<f32>,
    n_nodes: usize,
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(
        TensorData::new(data, [n_nodes, dim]),
        device,
    )
}

/// Look up a field's type key from the schema. Field names are "table.field".
fn find_field_type(schema: &Schema, full_name: &str) -> String {
    let parts: Vec<&str> = full_name.splitn(2, '.').collect();
    if parts.len() != 2 {
        return "any".to_string();
    }
    let (table_name, field_name) = (parts[0], parts[1]);

    for table in &schema.tables {
        if table.name == table_name {
            for field in &table.fields {
                if field.name == field_name {
                    return field_type_key(&field.field_type);
                }
            }
        }
    }
    "any".to_string()
}

fn field_type_key(ft: &FieldType) -> String {
    match ft {
        FieldType::Any => "any",
        FieldType::Bool => "bool",
        FieldType::String => "string",
        FieldType::Int => "int",
        FieldType::Float => "float",
        FieldType::Decimal => "decimal",
        FieldType::Number => "number",
        FieldType::Datetime => "datetime",
        FieldType::Duration => "duration",
        FieldType::Bytes => "bytes",
        FieldType::Object => "object",
        FieldType::Regex => "regex",
        FieldType::Record { .. } => "record",
        FieldType::Array { .. } => "array",
        FieldType::Set { .. } => "set",
        FieldType::Option { .. } => "option",
        FieldType::Geometry { .. } => "geometry",
        FieldType::Literal { .. } => "literal",
        FieldType::Range { .. } => "range",
    }.to_string()
}
