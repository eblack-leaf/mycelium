// =============================================================================
// nlp.rs — Transformer-based NL frontend (MiniLM via ONNX Runtime)
//
// 3-stage pipeline:
//   Stage 1 (this module): NL query → LinguisticGraph
//     Currently: rule-based span/dep extraction (biaffine head TODO).
//     Transformer produces span embeddings.
//   Stage 2 (candidate_matcher.rs): cross-encoder scoring
//   Stage 3 (GNN): resolve using graph structure
// =============================================================================

use std::cell::RefCell;
use std::path::Path;
use serde::{Serialize, Deserialize};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;
use burn::backend::NdArray;
use burn::record::CompactRecorder;
use burn::module::Module;
use burn::tensor::TensorData;

use crate::biaffine::{BiaffineHead, mean_pool_spans, HIDDEN_DIM};
use crate::biaffine_data::{BioTag, build_subword_to_word, decode_bio_spans};
use crate::ngram_attn::{NgramCrossAttn, words_from_subwords, generate_ngrams, greedy_select, MINILM_DIM};
use crate::ngram_data::ConceptMap;
use crate::candidate_matcher::{CandidateSet, CandidateEdge};
use crate::graph::SchemaGraph;
use crate::operations::OpNode;

// =============================================================================
// Linguistic graph types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticNode {
    pub id: usize,
    pub text: String,
    pub token_span: (usize, usize),
    pub span_type: SpanType,
    #[serde(skip)]
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanType {
    NounPhrase,
    Quantifier,
    Comparator,
    Intent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DepRelation {
    Possessive,
    Quantifies,
    Comparison,
    IntentTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticEdge {
    pub src: usize,
    pub dst: usize,
    pub relation: DepRelation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticGraph {
    pub raw_query: String,
    pub nodes: Vec<LinguisticNode>,
    pub edges: Vec<LinguisticEdge>,
}

impl SpanType {
    pub const COUNT: usize = 4;
}

impl DepRelation {
    pub const COUNT: usize = 4;
}

// =============================================================================
// NLP model
// =============================================================================

pub struct NlpModel {
    /// Bi-encoder (MiniLM) for span embeddings
    session: RefCell<Session>,
    tokenizer: Tokenizer,
    /// Cross-encoder (ms-marco-MiniLM) for candidate scoring
    cross_session: RefCell<Session>,
    cross_tokenizer: Tokenizer,
    /// Trained biaffine head (None = fall back to rule-based)
    biaffine: Option<BiaffineHead<NdArray>>,
    /// N-gram cross-attention model (None = not loaded)
    ngram: Option<NgramCrossAttn<NdArray>>,
    /// Concept map for n-gram model (built from schema + operations)
    concept_map: Option<ConceptMap>,
}

pub struct NlpConfig {
    /// Bi-encoder model (sentence-transformers/all-MiniLM-L6-v2)
    pub model_path: String,
    pub tokenizer_path: String,
    /// Cross-encoder model (cross-encoder/ms-marco-MiniLM-L6-v2)
    pub cross_model_path: String,
    pub cross_tokenizer_path: String,
    /// Optional: path to trained biaffine model (.mpk)
    pub biaffine_model_path: Option<String>,
    /// Optional: path to trained n-gram cross-attention model (.mpk)
    pub ngram_model_path: Option<String>,
}

impl NlpModel {
    pub fn load(config: &NlpConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(&config.model_path)?;
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| format!("tokenizer: {}", e))?;
        let cross_session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(&config.cross_model_path)?;
        let cross_tokenizer = Tokenizer::from_file(&config.cross_tokenizer_path)
            .map_err(|e| format!("cross tokenizer: {}", e))?;

        // Optionally load biaffine head
        let biaffine = config.biaffine_model_path.as_ref().and_then(|path| {
            let mpk_path = format!("{}.mpk", path);
            if Path::new(&mpk_path).exists() {
                let device = Default::default();
                match BiaffineHead::<NdArray>::new(&device)
                    .load_file(path, &CompactRecorder::new(), &device)
                {
                    Ok(m) => {
                        eprintln!("biaffine head loaded from {}", path);
                        Some(m)
                    }
                    Err(e) => {
                        eprintln!("biaffine head load failed: {} — falling back to rule-based", e);
                        None
                    }
                }
            } else {
                None
            }
        });

        // Optionally load n-gram cross-attention model
        let ngram = config.ngram_model_path.as_ref().and_then(|path| {
            let mpk_path = format!("{}.mpk", path);
            if Path::new(&mpk_path).exists() {
                let device = Default::default();
                // Need a dummy n_concepts to create the model struct, then load overwrites weights.
                // We use 1 as placeholder — the loaded weights will have the correct shape.
                match NgramCrossAttn::<NdArray>::new(1, &device)
                    .load_file(path, &CompactRecorder::new(), &device)
                {
                    Ok(m) => {
                        eprintln!("ngram cross-attention loaded from {}", path);
                        Some(m)
                    }
                    Err(e) => {
                        eprintln!("ngram model load failed: {} — will not use n-gram path", e);
                        None
                    }
                }
            } else {
                None
            }
        });

        Ok(Self {
            session: RefCell::new(session), tokenizer,
            cross_session: RefCell::new(cross_session), cross_tokenizer,
            biaffine,
            ngram,
            concept_map: None,
        })
    }

    /// Encode single text, return mean-pooled embedding [384].
    pub fn encode_pooled(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| format!("encode: {}", e))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let seq_len = ids.len();

        let ids_tensor = Tensor::from_array((vec![1i64, seq_len as i64], ids))?;
        let mask_tensor = Tensor::from_array((vec![1i64, seq_len as i64], mask))?;
        let type_tensor = Tensor::from_array((vec![1i64, seq_len as i64], type_ids))?;

        let mut session = self.session.borrow_mut();
        let outputs = session.run(ort::inputs![ids_tensor, mask_tensor, type_tensor])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let hidden_dim = shape[2] as usize;

        // Mean pool (skip [CLS]=0 and [SEP]=last)
        let mut pooled = vec![0.0f32; hidden_dim];
        let n_tokens = seq_len.saturating_sub(2).max(1);
        for i in 1..seq_len.saturating_sub(1) {
            let offset = i * hidden_dim;
            for j in 0..hidden_dim {
                pooled[j] += data[offset + j];
            }
        }
        for j in 0..hidden_dim {
            pooled[j] /= n_tokens as f32;
        }
        // L2 normalize
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut pooled {
                *x /= norm;
            }
        }

        Ok(pooled)
    }

    /// Cross-encoder scoring: feeds "[CLS] phrase [SEP] schema_name [SEP]"
    /// through the ms-marco cross-encoder. Returns sigmoid(logit) as 0-1 score.
    pub fn cross_encode(&self, phrase: &str, schema_name: &str) -> f32 {
        let encoding = match self.cross_tokenizer.encode((phrase, schema_name), true) {
            Ok(e) => e,
            Err(_) => return 0.0,
        };
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let seq_len = ids.len();

        let ids_tensor = match Tensor::from_array((vec![1i64, seq_len as i64], ids)) {
            Ok(t) => t, Err(_) => return 0.0,
        };
        let mask_tensor = match Tensor::from_array((vec![1i64, seq_len as i64], mask)) {
            Ok(t) => t, Err(_) => return 0.0,
        };
        let type_tensor = match Tensor::from_array((vec![1i64, seq_len as i64], type_ids)) {
            Ok(t) => t, Err(_) => return 0.0,
        };

        let mut session = self.cross_session.borrow_mut();
        let outputs = match session.run(ort::inputs![ids_tensor, mask_tensor, type_tensor]) {
            Ok(o) => o, Err(_) => return 0.0,
        };

        // Output: [1, 1] logit — apply sigmoid
        let (_, data) = match outputs[0].try_extract_tensor::<f32>() {
            Ok(d) => d, Err(_) => return 0.0,
        };
        let logit = data[0];
        1.0 / (1.0 + (-logit).exp()) // sigmoid
    }

    /// Batch cross-encode.
    pub fn cross_encode_batch(&self, pairs: &[(&str, &str)]) -> Vec<f32> {
        pairs.iter().map(|(p, s)| self.cross_encode(p, s)).collect()
    }

    /// Encode text, return raw token-level embeddings [seq_len-2, 384] (no CLS/SEP)
    /// plus the tokenizer Encoding for offset information.
    fn encode_tokens(&self, text: &str) -> Result<(Vec<f32>, tokenizers::Encoding), Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| format!("encode: {}", e))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let seq_len = ids.len();

        let ids_tensor = Tensor::from_array((vec![1i64, seq_len as i64], ids))?;
        let mask_tensor = Tensor::from_array((vec![1i64, seq_len as i64], mask))?;
        let type_tensor = Tensor::from_array((vec![1i64, seq_len as i64], type_ids))?;

        let mut session = self.session.borrow_mut();
        let outputs = session.run(ort::inputs![ids_tensor, mask_tensor, type_tensor])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let hidden_dim = shape[2] as usize;

        // Extract content tokens (skip CLS=0 and SEP=last)
        let n_content = seq_len.saturating_sub(2);
        let mut token_embs = Vec::with_capacity(n_content * hidden_dim);
        for i in 1..=n_content {
            let offset = i * hidden_dim;
            token_embs.extend_from_slice(&data[offset..offset + hidden_dim]);
        }

        Ok((token_embs, encoding))
    }

    /// Parse NL query into a LinguisticGraph.
    /// Uses biaffine head if available, otherwise falls back to rule-based.
    pub fn parse(&self, query: &str) -> LinguisticGraph {
        if let Some(ref biaffine) = self.biaffine {
            if let Some(graph) = self.biaffine_parse(biaffine, query) {
                return graph;
            }
            // Fall through to rule-based on error
        }

        // Rule-based fallback
        let (nodes, edges) = rule_based_parse(query);
        let mut enriched_nodes = nodes;
        for node in &mut enriched_nodes {
            if let Ok(embed) = self.encode_pooled(&node.text) {
                node.embedding = embed;
            }
        }
        LinguisticGraph {
            raw_query: query.to_string(),
            nodes: enriched_nodes,
            edges,
        }
    }

    /// Parse using the trained biaffine head.
    fn biaffine_parse(&self, biaffine: &BiaffineHead<NdArray>, query: &str) -> Option<LinguisticGraph> {
        let (token_data, encoding) = self.encode_tokens(query).ok()?;
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
        let subword_to_word = build_subword_to_word(&offsets, query);
        let seq_len = subword_to_word.len();
        if seq_len == 0 { return None; }

        let device: <NdArray as burn::tensor::backend::Backend>::Device = Default::default();
        let use_len = seq_len.min(token_data.len() / HIDDEN_DIM);
        if use_len == 0 { return None; }

        let token_embs = burn::tensor::Tensor::<NdArray, 2>::from_data(
            TensorData::new(token_data[..use_len * HIDDEN_DIM].to_vec(), [use_len, HIDDEN_DIM]),
            &device,
        );

        // Task 1: BIO tagging → argmax → decode spans
        let bio_logits = biaffine.forward_bio(token_embs.clone());
        let bio_data = bio_logits.into_data();
        let bio_vals: Vec<f32> = bio_data.to_vec().unwrap();

        let bio_preds: Vec<usize> = (0..use_len).map(|i| {
            let row = &bio_vals[i * BioTag::COUNT..(i + 1) * BioTag::COUNT];
            row.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0
        }).collect();

        let decoded_spans = decode_bio_spans(&bio_preds, &subword_to_word[..use_len]);
        if decoded_spans.is_empty() { return None; }

        // Build span embeddings by mean-pooling
        let span_bounds: Vec<(usize, usize)> = decoded_spans.iter()
            .map(|s| (s.start_word, s.end_word))
            .collect();
        let span_embs = mean_pool_spans(&token_embs, &subword_to_word[..use_len], &span_bounds);

        // Task 2: Dependency parsing
        let (arc_scores, rel_logits) = biaffine.forward_deps(span_embs);
        let n_spans = decoded_spans.len();

        let arc_data = arc_scores.into_data();
        let arc_vals: Vec<f32> = arc_data.to_vec().unwrap();
        let rel_data = rel_logits.into_data();
        let rel_vals: Vec<f32> = rel_data.to_vec().unwrap();

        // Build nodes with embeddings
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut nodes: Vec<LinguisticNode> = Vec::new();
        for (i, span) in decoded_spans.iter().enumerate() {
            let text: String = words[span.start_word..span.end_word.min(words.len())]
                .join(" ");
            let embed = self.encode_pooled(&text).unwrap_or_default();
            nodes.push(LinguisticNode {
                id: i,
                text,
                token_span: (span.start_word, span.end_word),
                span_type: span.span_type,
                embedding: embed,
            });
        }

        // Build edges: threshold arc scores > 0.5, pick argmax relation
        let arc_threshold = 0.5;
        let mut edges: Vec<LinguisticEdge> = Vec::new();
        for src in 0..n_spans {
            for dst in 0..n_spans {
                if src == dst { continue; }
                let arc_score = arc_vals[src * n_spans + dst];
                if arc_score > arc_threshold {
                    // Argmax over relation logits at (src, dst)
                    let base = (src * n_spans + dst) * DepRelation::COUNT;
                    let rel_idx = (0..DepRelation::COUNT)
                        .max_by(|&a, &b| {
                            rel_vals[base + a].partial_cmp(&rel_vals[base + b]).unwrap()
                        })
                        .unwrap_or(0);
                    let relation = match rel_idx {
                        0 => DepRelation::Possessive,
                        1 => DepRelation::Quantifies,
                        2 => DepRelation::Comparison,
                        3 => DepRelation::IntentTarget,
                        _ => DepRelation::Possessive,
                    };
                    edges.push(LinguisticEdge { src, dst, relation });
                }
            }
        }

        Some(LinguisticGraph {
            raw_query: query.to_string(),
            nodes,
            edges,
        })
    }

    /// Initialize the n-gram concept map from the schema and operations.
    /// Must be called after loading the schema if the n-gram model was loaded.
    pub fn init_ngram(&mut self, schema_graph: &SchemaGraph, operations: &[OpNode]) {
        if self.ngram.is_none() { return; }

        let table_names: Vec<String> = schema_graph.table_nodes.iter()
            .map(|n| n.name.clone())
            .collect();
        // Strip table prefix for field names (same as candidate_matcher)
        let field_names: Vec<String> = schema_graph.field_nodes.iter()
            .map(|n| n.name.splitn(2, '.').nth(1).unwrap_or(&n.name).to_string())
            .collect();
        let op_names: Vec<String> = operations.iter()
            .map(|op| op.name.clone())
            .collect();

        let concept_map = ConceptMap::new(&table_names, &field_names, &op_names);
        eprintln!("ngram concept map: {} concepts ({} tables, {} fields, {} ops)",
            concept_map.total(), concept_map.n_tables, concept_map.n_fields, concept_map.n_ops);
        self.concept_map = Some(concept_map);
    }

    /// Parse using the n-gram cross-attention model (standalone mode).
    /// Returns (LinguisticGraph, CandidateSet) in one pass.
    /// Currently unused — hybrid approach (biaffine spans + ngram scoring) preferred.
    #[allow(dead_code)]
    fn ngram_parse(
        &self,
        ngram: &NgramCrossAttn<NdArray>,
        concept_map: &ConceptMap,
        query: &str,
    ) -> Option<(LinguisticGraph, CandidateSet)> {
        let (token_data, encoding) = self.encode_tokens(query).ok()?;
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
        let subword_to_word = build_subword_to_word(&offsets, query);
        if subword_to_word.is_empty() { return None; }

        // Mean-pool subwords per word
        let (word_embs, n_words) = words_from_subwords(&token_data, &subword_to_word);
        if n_words == 0 { return None; }

        // Generate n-gram candidates
        let (ngram_embs, spans) = generate_ngrams(&word_embs, n_words);
        let n_ngrams = spans.len();
        if n_ngrams == 0 { return None; }

        // Run model forward pass
        let device: <NdArray as burn::tensor::backend::Backend>::Device = Default::default();
        let ngram_tensor = burn::tensor::Tensor::<NdArray, 2>::from_data(
            TensorData::new(ngram_embs.clone(), [n_ngrams, MINILM_DIM]),
            &device,
        );
        let (affinity_tensor, type_tensor) = ngram.forward_scores(ngram_tensor);

        let affinity: Vec<f32> = affinity_tensor.into_data().to_vec().unwrap();
        let type_logits: Vec<f32> = type_tensor.into_data().to_vec().unwrap();

        // Greedy selection — use discriminability (peakiness) instead of absolute threshold
        // min_peak = minimum (max - mean) across concepts for an n-gram to be considered
        let words: Vec<&str> = query.split_whitespace().collect();
        let selected = greedy_select(
            &affinity, &type_logits, &ngram_embs, &spans,
            &words, 12.0, 5, concept_map,
        );

        if selected.is_empty() { return None; }

        // Build LinguisticGraph
        let mut nodes = Vec::new();
        let edges_out = Vec::new();

        for (i, span) in selected.iter().enumerate() {
            nodes.push(LinguisticNode {
                id: i,
                text: span.text.clone(),
                token_span: (span.word_start, span.word_end),
                span_type: span.span_type,
                embedding: span.embedding.clone(),
            });
        }

        // Build CandidateSet from selected spans' top-k candidates
        let mut cand_edges = Vec::new();
        for (i, span) in selected.iter().enumerate() {
            for &(ref schema_type, schema_id, score) in &span.candidates {
                cand_edges.push(CandidateEdge {
                    linguistic_node: i,
                    schema_node_type: schema_type.clone(),
                    schema_node_id: schema_id,
                    score,
                });
            }
        }

        // No dependency edges — GNN relies on schema structure + cross-candidate edges
        let ling_graph = LinguisticGraph {
            raw_query: query.to_string(),
            nodes,
            edges: edges_out,
        };

        Some((ling_graph, CandidateSet { edges: cand_edges }))
    }

    /// Parse query, returning both LinguisticGraph and optional CandidateSet.
    ///
    /// If n-gram model is loaded: standalone n-gram parse (span detection + concept scoring).
    /// Falls back to biaffine/rule-based parse() + None if n-gram unavailable.
    pub fn parse_with_candidates(&self, query: &str) -> (LinguisticGraph, Option<CandidateSet>) {
        // If n-gram model is loaded, use standalone n-gram for both spans and concepts
        if let (Some(ref ngram), Some(ref concept_map)) = (&self.ngram, &self.concept_map) {
            if let Some((ling_graph, cands)) = self.ngram_parse(ngram, concept_map, query) {
                return (ling_graph, Some(cands));
            }
        }

        // Fallback: biaffine/rule-based parse, no candidates
        let ling_graph = self.parse(query);
        (ling_graph, None)
    }

    /// Score pre-detected spans against learned concept embeddings.
    ///
    /// Takes a LinguisticGraph (from biaffine/rule-based) and scores each node
    /// using the same embedding path as n-gram training:
    ///   1. encode_tokens() on full query → contextual subword embeddings
    ///   2. words_from_subwords() → word-level embeddings
    ///   3. Mean-pool words within each node's span
    ///   4. Project through ngram_proj + dot against concept_embs
    ///
    /// Returns CandidateSet with top-k concept matches per node.
    fn ngram_score_spans(
        &self,
        ngram: &NgramCrossAttn<NdArray>,
        concept_map: &ConceptMap,
        ling_graph: &LinguisticGraph,
    ) -> Option<CandidateSet> {
        if ling_graph.nodes.is_empty() { return None; }

        let n_nodes = ling_graph.nodes.len();
        let n_concepts = concept_map.total();
        let device: <NdArray as burn::tensor::backend::Backend>::Device = Default::default();

        // Get contextual token embeddings from full query (same as training)
        let (token_data, encoding) = self.encode_tokens(&ling_graph.raw_query).ok()?;
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
        let subword_to_word = build_subword_to_word(&offsets, &ling_graph.raw_query);
        if subword_to_word.is_empty() { return None; }

        // Mean-pool subwords per word
        let (word_embs, n_words) = words_from_subwords(&token_data, &subword_to_word);
        if n_words == 0 { return None; }

        // For each linguistic node, generate all n-gram sub-spans within
        // its word boundaries, score them all, and take the best concept scores.
        // This matches the n-gram training distribution (uni/bi/trigrams).
        let top_k = 5;
        let mut edges = Vec::new();

        for node in &ling_graph.nodes {
            let (start_word, end_word) = node.token_span;
            let end_clamped = end_word.min(n_words);
            let start_clamped = start_word.min(end_clamped);
            let span_n_words = end_clamped.saturating_sub(start_clamped);

            if span_n_words == 0 { continue; }

            // Extract word embeddings for this span
            let span_word_embs: Vec<f32> = (start_clamped..end_clamped)
                .flat_map(|w| {
                    let off = w * MINILM_DIM;
                    word_embs[off..off + MINILM_DIM].to_vec()
                })
                .collect();

            // Generate all n-gram sub-spans within this span
            let (sub_ngrams, _sub_spans) = generate_ngrams(&span_word_embs, span_n_words);
            let n_sub = sub_ngrams.len() / MINILM_DIM;
            if n_sub == 0 { continue; }

            // Score all sub-ngrams at once
            let sub_tensor = burn::tensor::Tensor::<NdArray, 2>::from_data(
                TensorData::new(sub_ngrams, [n_sub, MINILM_DIM]),
                &device,
            );
            let (sub_affinity, _) = ngram.forward_scores(sub_tensor);
            let sub_aff: Vec<f32> = sub_affinity.into_data().to_vec().unwrap();

            // For each concept, take the max score across all sub-ngrams
            let mut best_per_concept = vec![f32::NEG_INFINITY; n_concepts];
            for ng in 0..n_sub {
                let base = ng * n_concepts;
                for c in 0..n_concepts {
                    let score = sub_aff[base + c];
                    if score > best_per_concept[c] {
                        best_per_concept[c] = score;
                    }
                }
            }

            // Take top-k concepts
            let mut indexed: Vec<(usize, f32)> = best_per_concept.iter()
                .enumerate()
                .map(|(i, &s)| (i, s))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for &(concept_idx, score) in indexed.iter().take(top_k) {
                let (schema_type, schema_id) = concept_map.from_idx(concept_idx);
                edges.push(CandidateEdge {
                    linguistic_node: node.id,
                    schema_node_type: schema_type.to_string(),
                    schema_node_id: schema_id,
                    score,
                });
            }
        }

        Some(CandidateSet { edges })
    }

    /// Get a reference to the concept map (for external use, e.g. dataset generation).
    pub fn concept_map(&self) -> Option<&ConceptMap> {
        self.concept_map.as_ref()
    }

    /// Check if the n-gram model is loaded and initialized.
    pub fn has_ngram(&self) -> bool {
        self.ngram.is_some() && self.concept_map.is_some()
    }
}

// =============================================================================
// Rule-based parsing (stand-in for biaffine head)
// =============================================================================

fn rule_based_parse(query: &str) -> (Vec<LinguisticNode>, Vec<LinguisticEdge>) {
    let lower = query.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut word_idx = 0;

    // Intent detection
    let intent_words = ["show", "find", "get", "list", "count", "delete", "give", "fetch", "display"];
    if let Some(&first) = words.first() {
        if intent_words.contains(&first) {
            nodes.push(LinguisticNode {
                id: 0, text: first.to_string(),
                token_span: (0, 1), span_type: SpanType::Intent, embedding: vec![],
            });
            word_idx = 1;
            while word_idx < words.len() && ["me", "all", "every", "the"].contains(&words[word_idx]) {
                word_idx += 1;
            }
        }
    }

    let quant_words = ["first", "top", "last", "limit"];
    let comp_words = ["over", "above", "below", "under", "before", "after", "more", "less", "greater", "fewer"];

    let mut phrases: Vec<(String, usize, usize, SpanType)> = Vec::new();
    let mut current_phrase: Vec<String> = Vec::new();
    let mut phrase_start = word_idx;

    while word_idx < words.len() {
        let word = words[word_idx];

        if quant_words.contains(&word) {
            if !current_phrase.is_empty() {
                phrases.push((current_phrase.join(" "), phrase_start, word_idx, SpanType::NounPhrase));
                current_phrase.clear();
            }
            let mut quant_text = vec![word.to_string()];
            let q_start = word_idx;
            word_idx += 1;
            if word_idx < words.len() && words[word_idx].parse::<f64>().is_ok() {
                quant_text.push(words[word_idx].to_string());
                word_idx += 1;
            }
            phrases.push((quant_text.join(" "), q_start, word_idx, SpanType::Quantifier));
            phrase_start = word_idx;
            continue;
        }

        if comp_words.contains(&word) {
            if !current_phrase.is_empty() {
                phrases.push((current_phrase.join(" "), phrase_start, word_idx, SpanType::NounPhrase));
                current_phrase.clear();
            }
            let mut comp_text = vec![word.to_string()];
            let c_start = word_idx;
            word_idx += 1;
            if word_idx < words.len() && words[word_idx] == "than" {
                comp_text.push("than".to_string());
                word_idx += 1;
            }
            if word_idx < words.len() && words[word_idx].parse::<f64>().is_ok() {
                comp_text.push(words[word_idx].to_string());
                word_idx += 1;
            }
            phrases.push((comp_text.join(" "), c_start, word_idx, SpanType::Comparator));
            phrase_start = word_idx;
            continue;
        }

        let is_delim = ["where", "with", "and", "or", "by", "from", "in", "is", "are", "that", "which", "whose"]
            .contains(&word);
        let is_possessive = word.ends_with("'s") || word.ends_with("'");

        if is_possessive {
            let clean = word.trim_end_matches("'s").trim_end_matches("'");
            current_phrase.push(clean.to_string());
            phrases.push((current_phrase.join(" "), phrase_start, word_idx + 1, SpanType::NounPhrase));
            current_phrase.clear();
            word_idx += 1;
            phrase_start = word_idx;
            continue;
        }

        if is_delim || word.ends_with(',') {
            if !current_phrase.is_empty() {
                phrases.push((current_phrase.join(" "), phrase_start, word_idx, SpanType::NounPhrase));
                current_phrase.clear();
            }
            word_idx += 1;
            phrase_start = word_idx;
            continue;
        }

        if ["the", "a", "an", "my", "their", "its"].contains(&word) && current_phrase.is_empty() {
            word_idx += 1;
            phrase_start = word_idx;
            continue;
        }

        current_phrase.push(word.to_string());
        word_idx += 1;
    }

    if !current_phrase.is_empty() {
        phrases.push((current_phrase.join(" "), phrase_start, word_idx, SpanType::NounPhrase));
    }

    for (text, start, end, span_type) in &phrases {
        if text.is_empty() { continue; }
        nodes.push(LinguisticNode {
            id: nodes.len(), text: text.clone(),
            token_span: (*start, *end), span_type: *span_type, embedding: vec![],
        });
    }

    // Edges
    let intent_id = nodes.iter().position(|n| n.span_type == SpanType::Intent);
    let first_np = nodes.iter().position(|n| n.span_type == SpanType::NounPhrase);
    if let (Some(i), Some(np)) = (intent_id, first_np) {
        edges.push(LinguisticEdge { src: i, dst: np, relation: DepRelation::IntentTarget });
    }

    let np_ids: Vec<usize> = nodes.iter()
        .filter(|n| n.span_type == SpanType::NounPhrase)
        .map(|n| n.id).collect();

    for window in np_ids.windows(2) {
        let a = &nodes[window[0]];
        let b = &nodes[window[1]];
        if a.token_span.1 == b.token_span.0 {
            edges.push(LinguisticEdge { src: window[0], dst: window[1], relation: DepRelation::Possessive });
        }
    }

    for node in &nodes {
        if node.span_type == SpanType::Quantifier {
            let target = np_ids.iter()
                .filter(|&&id| nodes[id].token_span.0 < node.token_span.0)
                .last().copied()
                .or(first_np);
            if let Some(t) = target {
                edges.push(LinguisticEdge { src: node.id, dst: t, relation: DepRelation::Quantifies });
            }
        }
        if node.span_type == SpanType::Comparator {
            let target = np_ids.iter()
                .filter(|&&id| nodes[id].token_span.0 < node.token_span.0)
                .last().copied();
            if let Some(t) = target {
                edges.push(LinguisticEdge { src: node.id, dst: t, relation: DepRelation::Comparison });
            }
        }
    }

    (nodes, edges)
}

