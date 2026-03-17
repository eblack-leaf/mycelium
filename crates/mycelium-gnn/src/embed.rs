// =============================================================================
// embed.rs — Initial node embeddings from GloVe + learned type embeddings
//
// GloVe: frozen pretrained word vectors (not trainable)
// TypeVocab: burn Embedding module (trainable) — one vector per schema type
//            and operation identity
//
// Produces HashMap<node_type, Tensor[n_nodes, embed_dim]> for the Encoder.
// =============================================================================

use std::collections::HashMap;
use std::path::Path;
use burn::tensor::{backend::Backend, Tensor, Int, TensorData};
use burn::nn::{Embedding, EmbeddingConfig, Initializer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use crate::graph::SchemaGraph;
use crate::query_graph::QueryGraph;
use crate::schema::{Schema, FieldType};
use crate::operations::OpNode;

// =============================================================================
// Type vocabulary — name → index mapping
// =============================================================================

/// All type names in fixed order. Index into this = index into the Embedding.
const TYPE_NAMES: &[&str] = &[
    // Schema node types
    "table", "any", "bool", "string", "int", "float", "decimal",
    "number", "datetime", "duration", "bytes", "object", "regex",
    "record", "array", "set", "option", "geometry", "literal", "range",
    // Operation identities — prefixed to avoid collision
    "op_SELECT", "op_CREATE", "op_UPDATE", "op_DELETE", "op_RELATE", "op_INSERT",
    "op_ORDER_BY", "op_GROUP_BY", "op_LIMIT", "op_FETCH", "op_SPLIT",
    "op_eq", "op_neq", "op_gt", "op_lt", "op_gte", "op_lte",
    "op_LIKE", "op_CONTAINS", "op_STARTS_WITH", "op_ENDS_WITH",
    "op_add", "op_sub", "op_mul", "op_div",
    "op_count", "op_sum", "op_avg", "op_min", "op_max", "op_array_group",
    "op_arrow_right", "op_arrow_left", "op_arrow_both",
    // Fallback for unknown types
    "unknown",
];

/// Name → index into the Embedding weight matrix.
#[derive(Debug, Clone)]
pub struct TypeIndex {
    names: HashMap<String, usize>,
    unknown_idx: usize,
}

impl TypeIndex {
    pub fn new() -> Self {
        let names: HashMap<String, usize> = TYPE_NAMES.iter()
            .enumerate()
            .map(|(i, &name)| (name.to_string(), i))
            .collect();
        let unknown_idx = names["unknown"];
        Self { names, unknown_idx }
    }

    pub fn index_of(&self, name: &str) -> usize {
        self.names.get(name).copied().unwrap_or(self.unknown_idx)
    }

    pub fn n_types(&self) -> usize {
        TYPE_NAMES.len()
    }
}

/// Create the trainable type Embedding module.
pub fn create_type_embedding<B: Backend>(type_dim: usize, device: &B::Device) -> Embedding<B> {
    EmbeddingConfig::new(TYPE_NAMES.len(), type_dim)
        .with_initializer(Initializer::Uniform { min: -0.1, max: 0.1 })
        .init(device)
}

// =============================================================================
// GloVe — frozen pretrained word vectors
// =============================================================================

#[derive(Debug, Clone)]
pub struct GloveVocab {
    pub vectors: HashMap<String, Vec<f32>>,
    pub dim: usize,
    pub seed: u64,
}

impl GloveVocab {
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

// =============================================================================
// Embedder — combines frozen GloVe + trainable type embeddings
// =============================================================================

pub struct Embedder {
    pub glove: GloveVocab,
    pub type_index: TypeIndex,
    pub type_dim: usize,
}

impl Embedder {
    pub fn new(glove: GloveVocab, type_dim: usize) -> Self {
        Self {
            glove,
            type_index: TypeIndex::new(),
            type_dim,
        }
    }

    /// Total embedding dim: glove_dim + type_dim + 2 (confidence + is_nl flag)
    pub fn dim(&self) -> usize {
        self.glove.dim + self.type_dim + 2
    }

    /// Produce initial embeddings for every node type.
    /// type_embed is the trainable Embedding from GnnModel.
    pub fn embed_all<B: Backend>(
        &self,
        type_embed: &Embedding<B>,
        schema: &Schema,
        schema_graph: &SchemaGraph,
        query_graph: &QueryGraph,
        operations: &[OpNode],
        device: &B::Device,
    ) -> HashMap<String, Tensor<B, 2>> {
        let mut result = HashMap::new();

        // --- Schema: table nodes ---
        if !schema_graph.table_nodes.is_empty() {
            let names: Vec<&str> = schema_graph.table_nodes.iter()
                .map(|n| n.name.as_str()).collect();
            let type_keys: Vec<&str> = vec!["table"; names.len()];
            let confs: Vec<f32> = vec![0.0; names.len()];
            result.insert("table".into(),
                self.embed_group(type_embed, &names, &type_keys, &confs, false, device));
        }

        // --- Schema: field nodes ---
        if !schema_graph.field_nodes.is_empty() {
            let names: Vec<&str> = schema_graph.field_nodes.iter()
                .map(|n| n.name.as_str()).collect();
            let type_key_strings: Vec<String> = schema_graph.field_nodes.iter()
                .map(|n| find_field_type(schema, &n.name)).collect();
            let type_keys: Vec<&str> = type_key_strings.iter()
                .map(|s| s.as_str()).collect();
            let confs: Vec<f32> = vec![0.0; names.len()];
            result.insert("field".into(),
                self.embed_group(type_embed, &names, &type_keys, &confs, false, device));
        }

        // --- Query: collection candidates ---
        if !query_graph.collections.is_empty() {
            let names: Vec<&str> = query_graph.collections.iter()
                .map(|c| c.surface_form.as_str()).collect();
            let type_keys: Vec<&str> = vec!["table"; names.len()];
            let confs: Vec<f32> = query_graph.collections.iter()
                .map(|c| c.confidence).collect();
            result.insert("q_collection".into(),
                self.embed_group(type_embed, &names, &type_keys, &confs, true, device));
        }

        // --- Query: field candidates ---
        if !query_graph.fields.is_empty() {
            let names: Vec<&str> = query_graph.fields.iter()
                .map(|f| f.surface_form.as_str()).collect();
            let type_keys: Vec<&str> = vec!["unknown"; names.len()];
            let confs: Vec<f32> = query_graph.fields.iter()
                .map(|f| f.confidence).collect();
            result.insert("q_field".into(),
                self.embed_group(type_embed, &names, &type_keys, &confs, true, device));
        }

        // --- Query: filter candidates ---
        if !query_graph.filters.is_empty() {
            let texts: Vec<String> = query_graph.filters.iter()
                .map(|f| format!("{} {}", f.operator, f.value)).collect();
            let names: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let type_keys: Vec<&str> = vec!["unknown"; names.len()];
            let confs: Vec<f32> = query_graph.filters.iter()
                .map(|f| f.confidence).collect();
            result.insert("q_filter".into(),
                self.embed_group(type_embed, &names, &type_keys, &confs, true, device));
        }

        // --- Query: traversal candidates ---
        if !query_graph.traversals.is_empty() {
            let names: Vec<&str> = query_graph.traversals.iter()
                .map(|t| t.surface_form.as_str()).collect();
            let type_keys: Vec<&str> = vec!["unknown"; names.len()];
            let confs: Vec<f32> = query_graph.traversals.iter()
                .map(|t| t.confidence).collect();
            result.insert("q_traversal".into(),
                self.embed_group(type_embed, &names, &type_keys, &confs, true, device));
        }

        // --- Query: modifier candidates ---
        if !query_graph.modifiers.is_empty() {
            let texts: Vec<String> = query_graph.modifiers.iter()
                .map(|m| format!("{} {}", m.surface_form, m.value)).collect();
            let names: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let type_keys: Vec<&str> = vec!["unknown"; names.len()];
            let confs: Vec<f32> = query_graph.modifiers.iter()
                .map(|m| m.confidence).collect();
            result.insert("q_modifier".into(),
                self.embed_group(type_embed, &names, &type_keys, &confs, true, device));
        }

        // --- Operation nodes ---
        if !operations.is_empty() {
            let texts: Vec<String> = operations.iter()
                .map(|op| op.name.to_lowercase().replace('_', " ")).collect();
            let names: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let type_key_strings: Vec<String> = operations.iter()
                .map(|op| format!("op_{}", op.name)).collect();
            let type_keys: Vec<&str> = type_key_strings.iter()
                .map(|s| s.as_str()).collect();
            let confs: Vec<f32> = vec![0.0; names.len()];
            result.insert("operation".into(),
                self.embed_group(type_embed, &names, &type_keys, &confs, false, device));
        }

        result
    }

    /// Embed a group of nodes: GloVe (frozen) + type embedding (trainable) + meta.
    /// Returns [n_nodes, glove_dim + type_dim + 2].
    fn embed_group<B: Backend>(
        &self,
        type_embed: &Embedding<B>,
        names: &[&str],
        type_keys: &[&str],
        confidences: &[f32],
        is_nl: bool,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let n = names.len();
        let glove_dim = self.glove.dim;
        let total_frozen_dim = glove_dim + 2; // glove + confidence + is_nl

        // Build frozen part as single tensor: [glove... | confidence | is_nl]
        // Avoids separate [n, 2] meta tensor that triggers fusion shape conflicts
        let is_nl_val = if is_nl { 1.0f32 } else { 0.0 };
        let mut frozen_data = Vec::with_capacity(n * total_frozen_dim);
        for (i, name) in names.iter().enumerate() {
            frozen_data.extend(self.glove.embed(name));
            frozen_data.push(confidences[i]);
            frozen_data.push(is_nl_val);
        }
        let frozen_t: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(frozen_data, [n, total_frozen_dim]), device,
        );

        // Type indices → trainable embedding lookup [1, n] → [1, n, type_dim] → [n, type_dim]
        let type_indices: Vec<i32> = type_keys.iter()
            .map(|k| self.type_index.index_of(k) as i32)
            .collect();
        let idx_t = Tensor::<B, 2, Int>::from_data(
            TensorData::new(type_indices, [1, n]), device,
        );
        let type_t = type_embed.forward(idx_t).reshape([n, self.type_dim]);

        // Concat: [n, glove_dim + 2] | [n, type_dim] → [n, glove_dim + type_dim + 2]
        Tensor::cat(vec![frozen_t, type_t], 1)
    }
}

// =============================================================================
// Helpers
// =============================================================================

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
