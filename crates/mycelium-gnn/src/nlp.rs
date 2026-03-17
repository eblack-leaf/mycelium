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
use serde::{Serialize, Deserialize};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

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
}

pub struct NlpConfig {
    /// Bi-encoder model (sentence-transformers/all-MiniLM-L6-v2)
    pub model_path: String,
    pub tokenizer_path: String,
    /// Cross-encoder model (cross-encoder/ms-marco-MiniLM-L6-v2)
    pub cross_model_path: String,
    pub cross_tokenizer_path: String,
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
        Ok(Self {
            session: RefCell::new(session), tokenizer,
            cross_session: RefCell::new(cross_session), cross_tokenizer,
        })
    }

    /// Encode single text, return mean-pooled embedding [384].
    fn encode_pooled(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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

    /// Parse NL query into a LinguisticGraph.
    /// Currently rule-based spans + transformer embeddings.
    pub fn parse(&self, query: &str) -> LinguisticGraph {
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

