#![recursion_limit = "512"]
// =============================================================================
// mycelium-gnn — NL→SurrealQL query resolution
//
// Pipeline:
//   1. NL parse (nlp.rs)                      → LinguisticGraph
//   2. Candidate match (candidate_matcher.rs)  → CandidateSet
//   3. GNN resolve (linguistic_graph.rs + sage.rs + head.rs) → Resolution
//   4. SQL emit (orchestrator.rs)              → SurrealQL
// =============================================================================

pub mod schema;
pub mod graph;
pub mod operations;
pub mod nlp;
pub mod biaffine;
pub mod biaffine_data;
pub mod candidate_matcher;
pub mod linguistic_graph;
pub mod tensor_ops;
pub mod sage;
pub mod embed;
pub mod head;
pub mod orchestrator;
pub mod reranker;
pub mod reranker_data;
pub mod ngram_attn;
pub mod ngram_data;
pub mod training;

use std::path::Path;
use schema::{Reader, Extractor, Schema, Validation};
use graph::SchemaGraph;
use operations::{all_operations, OpNode};
use nlp::{NlpModel, NlpConfig};
use candidate_matcher::{CandidateMatcher, CandidateMatcherConfig};

// =============================================================================
// Pipeline
// =============================================================================

pub struct Pipeline {
    pub schema: Schema,
    pub graph: SchemaGraph,
    pub validation: Validation,
    pub operations: Vec<OpNode>,
    pub nlp: NlpModel,
    pub matcher: CandidateMatcher,
}

pub struct PipelineConfig {
    pub schema_path: String,
    pub model_path: String,
    pub tokenizer_path: String,
    pub cross_model_path: String,
    pub cross_tokenizer_path: String,
    pub biaffine_model_path: Option<String>,
    pub ngram_model_path: Option<String>,
    pub matcher_config: CandidateMatcherConfig,
}

pub struct PipelineResult {
    pub linguistic_graph: nlp::LinguisticGraph,
    pub candidates: candidate_matcher::CandidateSet,
}

impl Pipeline {
    pub fn load(config: &PipelineConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = Reader::read(Path::new(&config.schema_path))?;
        let (schema, validation) = Extractor::extract(&raw);
        let graph = SchemaGraph::from_schema(&schema);
        let operations = all_operations();

        let mut nlp = NlpModel::load(&NlpConfig {
            model_path: config.model_path.clone(),
            tokenizer_path: config.tokenizer_path.clone(),
            cross_model_path: config.cross_model_path.clone(),
            cross_tokenizer_path: config.cross_tokenizer_path.clone(),
            biaffine_model_path: config.biaffine_model_path.clone(),
            ngram_model_path: config.ngram_model_path.clone(),
        })?;

        // Initialize n-gram model with schema if loaded
        nlp.init_ngram(&graph, &operations);

        let matcher = CandidateMatcher::new(&graph, &operations, config.matcher_config.clone());

        Ok(Self { schema, graph, validation, operations, nlp, matcher })
    }

    pub fn run(&self, query: &str) -> PipelineResult {
        let (linguistic_graph, cands) = self.nlp.parse_with_candidates(query);
        let candidates = cands.unwrap_or_else(|| self.matcher.match_candidates(&self.nlp, &linguistic_graph));
        PipelineResult { linguistic_graph, candidates }
    }
}
