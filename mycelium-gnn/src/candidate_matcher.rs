// =============================================================================
// candidate_matcher.rs — Stage 2: Candidate matching
//
// Scores (phrase, schema_name) pairs using either:
//   - Pretrained cross-encoder (MiniLM)
//   - Trained re-ranker (Burn MLP on MiniLM embeddings)
// No role assignment — every linguistic node scored against all schema nodes.
// =============================================================================

use serde::{Serialize, Deserialize};
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use crate::graph::SchemaGraph;
use crate::nlp::{NlpModel, LinguisticGraph};
use crate::operations::OpNode;
use crate::reranker::Reranker;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateEdge {
    pub linguistic_node: usize,
    pub schema_node_type: String,
    pub schema_node_id: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateSet {
    pub edges: Vec<CandidateEdge>,
}

#[derive(Debug, Clone)]
pub struct CandidateMatcherConfig {
    pub top_k: usize,
    pub min_score: f32,
}

impl Default for CandidateMatcherConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: 0.3,
        }
    }
}

/// Schema node names for cross-encoder pairing.
pub struct SchemaNames {
    pub table_names: Vec<String>,
    pub field_names: Vec<String>,
    pub operation_names: Vec<String>,
}

pub struct CandidateMatcher {
    schema_names: SchemaNames,
    /// Pre-computed MiniLM embeddings for each schema name (for re-ranker path)
    schema_embeddings: Option<SchemaEmbeddings>,
    reranker: Option<Reranker<NdArray>>,
    config: CandidateMatcherConfig,
}

/// Pre-computed 384-dim embeddings for schema names.
pub struct SchemaEmbeddings {
    pub table_embs: Vec<Vec<f32>>,  // [n_tables][384]
    pub field_embs: Vec<Vec<f32>>,  // [n_fields][384]
    pub op_embs: Vec<Vec<f32>>,     // [n_ops][384]
}

impl CandidateMatcher {
    pub fn new(
        schema_graph: &SchemaGraph,
        operations: &[OpNode],
        config: CandidateMatcherConfig,
    ) -> Self {
        let table_names: Vec<String> = schema_graph.table_nodes.iter()
            .map(|n| n.name.clone())
            .collect();

        // Strip table prefix for scoring — "users.email" → "email"
        // Prevents matching "users" to "users.email" on the shared prefix.
        let field_names: Vec<String> = schema_graph.field_nodes.iter()
            .map(|n| {
                n.name.splitn(2, '.').nth(1).unwrap_or(&n.name).to_string()
            })
            .collect();

        let operation_names: Vec<String> = operations.iter()
            .map(|op| op.name.clone())
            .collect();

        Self {
            schema_names: SchemaNames { table_names, field_names, operation_names },
            schema_embeddings: None,
            reranker: None,
            config,
        }
    }

    /// Load the trained re-ranker and pre-compute schema embeddings.
    /// If this succeeds, `match_candidates` will use the re-ranker instead
    /// of the pretrained cross-encoder.
    pub fn load_reranker(&mut self, reranker: Reranker<NdArray>, nlp: &NlpModel) {
        // Pre-compute schema name embeddings
        let table_embs: Vec<Vec<f32>> = self.schema_names.table_names.iter()
            .map(|n| nlp.encode_pooled(n).expect("encode table name"))
            .collect();
        let field_embs: Vec<Vec<f32>> = self.schema_names.field_names.iter()
            .map(|n| nlp.encode_pooled(n).expect("encode field name"))
            .collect();
        let op_embs: Vec<Vec<f32>> = self.schema_names.operation_names.iter()
            .map(|n| nlp.encode_pooled(n).expect("encode op name"))
            .collect();

        self.schema_embeddings = Some(SchemaEmbeddings { table_embs, field_embs, op_embs });
        self.reranker = Some(reranker);
    }

    /// Score all linguistic nodes against all schema nodes.
    /// Uses re-ranker if loaded, otherwise falls back to cross-encoder.
    pub fn match_candidates(
        &self,
        nlp: &NlpModel,
        ling_graph: &LinguisticGraph,
    ) -> CandidateSet {
        if let (Some(reranker), Some(schema_embs)) = (&self.reranker, &self.schema_embeddings) {
            return self.match_with_reranker(nlp, ling_graph, reranker, schema_embs);
        }
        self.match_with_cross_encoder(nlp, ling_graph)
    }

    /// Cross-encoder path (original).
    fn match_with_cross_encoder(
        &self,
        nlp: &NlpModel,
        ling_graph: &LinguisticGraph,
    ) -> CandidateSet {
        let mut edges = Vec::new();

        for node in &ling_graph.nodes {
            let table_scores: Vec<f32> = self.schema_names.table_names.iter()
                .map(|name| nlp.cross_encode(&node.text, name))
                .collect();
            self.collect_top_k(&mut edges, node.id, "table", &table_scores);

            let field_scores: Vec<f32> = self.schema_names.field_names.iter()
                .map(|name| nlp.cross_encode(&node.text, name))
                .collect();
            self.collect_top_k(&mut edges, node.id, "field", &field_scores);

            let op_scores: Vec<f32> = self.schema_names.operation_names.iter()
                .map(|name| nlp.cross_encode(&node.text, name))
                .collect();
            self.collect_top_k(&mut edges, node.id, "operation", &op_scores);
        }

        CandidateSet { edges }
    }

    /// Re-ranker path: encode phrase with MiniLM, score against pre-computed schema embeddings.
    fn match_with_reranker(
        &self,
        nlp: &NlpModel,
        ling_graph: &LinguisticGraph,
        reranker: &Reranker<NdArray>,
        schema_embs: &SchemaEmbeddings,
    ) -> CandidateSet {
        let device = Default::default();
        let mut edges = Vec::new();

        for node in &ling_graph.nodes {
            let phrase_emb = match nlp.encode_pooled(&node.text) {
                Ok(e) => e,
                Err(_) => continue,
            };

            // Score against tables
            let table_scores = self.reranker_scores(
                reranker, &phrase_emb, &schema_embs.table_embs, &device,
            );
            self.collect_top_k(&mut edges, node.id, "table", &table_scores);

            // Score against fields
            let field_scores = self.reranker_scores(
                reranker, &phrase_emb, &schema_embs.field_embs, &device,
            );
            self.collect_top_k(&mut edges, node.id, "field", &field_scores);

            // Score against operations
            let op_scores = self.reranker_scores(
                reranker, &phrase_emb, &schema_embs.op_embs, &device,
            );
            self.collect_top_k(&mut edges, node.id, "operation", &op_scores);
        }

        CandidateSet { edges }
    }

    /// Batch-score one phrase against all schema embeddings of a type.
    fn reranker_scores(
        &self,
        reranker: &Reranker<NdArray>,
        phrase_emb: &[f32],
        schema_embs: &[Vec<f32>],
        device: &<NdArray as burn::tensor::backend::Backend>::Device,
    ) -> Vec<f32> {
        if schema_embs.is_empty() { return vec![]; }
        let n = schema_embs.len();

        // Broadcast phrase to [n, 384]
        let phrase_data: Vec<f32> = phrase_emb.iter().copied().cycle().take(n * 384).collect();
        let phrase_t: Tensor<NdArray, 2> = Tensor::from_data(
            TensorData::new(phrase_data, [n, 384]), device,
        );

        // Stack schema embeddings to [n, 384]
        let schema_data: Vec<f32> = schema_embs.iter().flat_map(|e| e.iter().copied()).collect();
        let schema_t: Tensor<NdArray, 2> = Tensor::from_data(
            TensorData::new(schema_data, [n, 384]), device,
        );

        let probs = reranker.predict(phrase_t, schema_t); // [n]
        probs.into_data().to_vec().unwrap()
    }

    fn collect_top_k(
        &self,
        edges: &mut Vec<CandidateEdge>,
        ling_node: usize,
        schema_type: &str,
        scores: &[f32],
    ) {
        // Sort by score descending, take top-k above threshold
        let mut indexed: Vec<(usize, f32)> = scores.iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (schema_id, score) in indexed.into_iter().take(self.config.top_k) {
            if score < self.config.min_score { break; }
            edges.push(CandidateEdge {
                linguistic_node: ling_node,
                schema_node_type: schema_type.to_string(),
                schema_node_id: schema_id,
                score,
            });
        }
    }
}
