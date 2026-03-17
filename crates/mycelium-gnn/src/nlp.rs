// =============================================================================
// nlp.rs — Transformer-based NL frontend (MiniLM via ONNX Runtime)
//
// Architecture (decided 2026-03-17):
//   The old "grounding model" approach (single LLM produces structured Extraction
//   with schema matches) was rejected — too much responsibility in one model.
//   Instead we use a 3-stage pipeline:
//
//   Stage 1 (this module): NL query → LinguisticGraph
//     MiniLM transformer encoder (loaded via `ort` crate, ONNX Runtime).
//     Two heads on top of the shared token embeddings:
//       a) BIO span tagger → extracts span boundaries (NounPhrase, Quantifier,
//          Comparator, Intent). Pure grammar, no schema knowledge.
//       b) Biaffine dependency head (Dozat & Manning style) → predicts directed
//          edges between spans with relation labels (Possessive, Quantifies,
//          Comparison, IntentTarget).
//     Also produces mean-pooled span embeddings (used by Stage 2 cross-encoder
//     as an optimization — can skip re-encoding the phrase).
//
//   Stage 2 (candidate_matcher.rs): LinguisticGraph + Schema → CandidateSet
//     Cross-encoder using the SAME MiniLM model. For each (phrase, schema_name)
//     pair, feeds "[CLS] phrase [SEP] schema_name [SEP]" through the transformer
//     and scores from [CLS] output. This works for novel schemas because the
//     cross-encoder understands language relationships ("the goods" ≈ "products")
//     without memorized mappings.
//     Cost: O(n_phrases × n_schema_nodes) forward passes per query.
//     With ~4 phrases × ~63 schema nodes = ~252 tiny forward passes.
//
//   Stage 3 (GNN — sage.rs + linguistic_graph.rs + head.rs):
//     Heterogeneous GNN over combined schema graph + linguistic graph +
//     candidate match edges. Resolves which schema node each linguistic
//     node maps to, AND assigns roles (which phrase is the collection,
//     which is a field, etc.) through message passing.
//
// Key dependencies:
//   - `ort` crate: ONNX Runtime bindings (load MiniLM)
//   - `tokenizers` crate: HuggingFace tokenizer (tokenize input)
//   - Model: sentence-transformers/all-MiniLM-L6-v2 (ONNX format from HuggingFace)
//   - Biaffine head weights: trained separately on dependency parsing data
//
// What DOESN'T happen here:
//   - No schema knowledge. This module never sees table/field names.
//   - No candidate matching. That's Stage 2 (candidate_matcher.rs).
//   - No role assignment (collection vs field). That's the GNN's job.
// =============================================================================

use serde::{Serialize, Deserialize};

// =============================================================================
// Linguistic graph types — output of NL parsing
// =============================================================================

/// A span extracted from the NL query (noun phrase, quantifier, comparator).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticNode {
    pub id: usize,
    pub text: String,
    /// Token indices [start, end) in the original query
    pub token_span: (usize, usize),
    /// Span type from BIO tagger
    pub span_type: SpanType,
    /// Transformer embedding (mean-pooled over span tokens), set during forward pass.
    /// Used by cross-encoder as optional optimization.
    #[serde(skip)]
    pub embedding: Vec<f32>,
}

/// What kind of linguistic span this is (no schema knowledge — pure grammar).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanType {
    /// Noun phrase: "the goods", "timestamp", "cost"
    NounPhrase,
    /// Quantifier with a value: "first 49", "top 10", "at most 5"
    Quantifier,
    /// Comparison with a value: "over 100", "before yesterday", "at least 3"
    Comparator,
    /// Intent verb: "show", "find", "count", "delete"
    Intent,
}

/// Grammatical relationship between two linguistic nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DepRelation {
    /// Possessive/genitive: "the goods' timestamp" → goods→timestamp
    Possessive,
    /// Quantifier modifying a noun: "first 49 goods" → quantifier→goods
    Quantifies,
    /// Comparison on a noun: "cost over 100" → comparator→cost
    Comparison,
    /// Intent targeting a noun: "show me the goods" → intent→goods
    IntentTarget,
}

/// Directed edge in the linguistic graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticEdge {
    pub src: usize,
    pub dst: usize,
    pub relation: DepRelation,
}

/// Complete linguistic parse of an NL query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticGraph {
    pub raw_query: String,
    pub nodes: Vec<LinguisticNode>,
    pub edges: Vec<LinguisticEdge>,
}

// =============================================================================
// Biaffine head — trainable dep parsing layer
// =============================================================================

/// Configuration for the biaffine dependency parsing head.
pub struct BiaffineConfig {
    /// Hidden dim of transformer encoder (384 for MiniLM-L6)
    pub encoder_dim: usize,
    /// Reduced dim for arc/label prediction
    pub arc_dim: usize,
    /// Number of span types (BIO tags: B-NP, I-NP, B-QUANT, I-QUANT, ...)
    pub n_span_types: usize,
    /// Number of dependency relation types
    pub n_relations: usize,
}

impl Default for BiaffineConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 384,
            arc_dim: 128,
            n_span_types: SpanType::COUNT,
            n_relations: DepRelation::COUNT,
        }
    }
}

impl SpanType {
    const COUNT: usize = 4;
}

impl DepRelation {
    const COUNT: usize = 4;
}

// =============================================================================
// NLP model — transformer encoder + biaffine head
// =============================================================================

/// Full NLP frontend: loads MiniLM via ONNX, runs biaffine head for parsing,
/// mean-pools span tokens for phrase embeddings.
pub struct NlpModel {
    // TODO: ort::Session for transformer encoder
    // TODO: biaffine weights (arc head, arc dep, label head, label dep, BIO classifier)
    // TODO: tokenizer (from `tokenizers` crate)
    _config: BiaffineConfig,
}

pub struct NlpConfig {
    /// Path to ONNX model file (e.g., "models/minilm-l6-v2.onnx")
    pub model_path: String,
    /// Path to tokenizer.json
    pub tokenizer_path: String,
    /// Biaffine head config
    pub biaffine: BiaffineConfig,
}

impl NlpModel {
    pub fn load(_config: NlpConfig) -> Self {
        todo!("load ONNX session + tokenizer + biaffine weights")
    }

    /// Run the full NL frontend: tokenize → encode → parse spans → parse deps → embed spans.
    ///
    /// Steps:
    ///   1. Tokenize query → input_ids, attention_mask
    ///   2. Run transformer encoder → token_embeddings [seq_len, 384]
    ///   3. BIO tagger: classify each token → extract span boundaries
    ///   4. Group tokens into spans by BIO tags
    ///   5. Biaffine arc predictor: for each span pair, predict if dep edge exists
    ///   6. Biaffine label predictor: for each predicted edge, classify relation type
    ///   7. Mean-pool token embeddings per span → phrase embeddings
    ///   8. Build LinguisticGraph
    pub fn parse(&self, _query: &str) -> LinguisticGraph {
        todo!()
    }

    /// Run cross-encoder scoring for candidate matching (Stage 2).
    /// Feeds "[CLS] phrase [SEP] schema_name [SEP]" through the transformer.
    /// Returns a relevance score from the [CLS] representation.
    ///
    /// This reuses the same ONNX session as parse() but with different input.
    pub fn cross_encode(&self, _phrase: &str, _schema_name: &str) -> f32 {
        todo!()
    }

    /// Batch version of cross_encode for efficiency.
    /// Scores all (phrase, schema_name) pairs in one call.
    pub fn cross_encode_batch(
        &self,
        _pairs: &[(&str, &str)],
    ) -> Vec<f32> {
        todo!()
    }
}
