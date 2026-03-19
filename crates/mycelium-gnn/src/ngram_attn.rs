// =============================================================================
// ngram_attn.rs — N-gram cross-attention module (Burn)
//
// Single model replacing biaffine parser + re-ranker:
//   1. Mean-pool subwords per word → word-level embeddings
//   2. Generate unigram/bigram/trigram candidates
//   3. Project n-grams to 128-dim, cross-attend against learned concept embeddings
//   4. Concept-centric selection: for each concept, pick best n-gram (sigmoid)
//      → spans can overlap, each span can map to multiple concepts
//
// Multi-label training: n-grams inherit ALL concept labels from GT spans they
// contain. "age over 25" (trigram) inherits {field:age, op:gt} because it
// contains both the "age" unigram and the "over" unigram.
//
// The concept embedding table (nn::Embedding) learns from training data that
// natural language phrases map to SQL concepts. No MiniLM on the schema side.
// =============================================================================

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Embedding, EmbeddingConfig, Dropout, DropoutConfig},
    tensor::{backend::Backend, Tensor, Int, TensorData},
};

use crate::nlp::SpanType;
use crate::ngram_data::ConceptMap;

pub const PROJ_DIM: usize = 128;
pub const MINILM_DIM: usize = 384;

// =============================================================================
// Module
// =============================================================================

#[derive(Module, Debug)]
pub struct NgramCrossAttn<B: Backend> {
    /// Project 384-dim n-gram embeddings → 128-dim for affinity scoring
    pub ngram_proj: Linear<B>,
    /// Dropout after projection (regularization)
    pub dropout: Dropout,
    /// Learned concept embeddings: [n_concepts, 128]
    pub concept_embs: Embedding<B>,
    /// Classify n-gram span type: 384 → 4 (NP, Quant, Comp, Intent)
    pub type_classifier: Linear<B>,
}

impl<B: Backend> NgramCrossAttn<B> {
    pub fn new(n_concepts: usize, device: &B::Device) -> Self {
        Self {
            ngram_proj: LinearConfig::new(MINILM_DIM, PROJ_DIM).init(device),
            dropout: DropoutConfig::new(0.2).init(),
            concept_embs: EmbeddingConfig::new(n_concepts, PROJ_DIM).init(device),
            type_classifier: LinearConfig::new(MINILM_DIM, SpanType::COUNT).init(device),
        }
    }

    /// Compute affinity scores and type logits from n-gram embeddings.
    ///
    /// `ngram_embs`: [N, 384] — raw MiniLM-space n-gram embeddings
    /// Returns: (affinity [N, n_concepts], type_logits [N, 4])
    pub fn forward_scores(
        &self,
        ngram_embs: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let projected = self.dropout.forward(self.ngram_proj.forward(ngram_embs.clone())); // [N, 128]

        // Get all concept embeddings as a matrix [n_concepts, 128]
        let n_concepts = self.concept_embs.weight.val().dims()[0];
        let device = ngram_embs.device();
        // Embedding::forward wants [batch, seq_len] → [batch, seq_len, emb_dim]
        // Use [1, n_concepts] → [1, n_concepts, 128] → squeeze to [n_concepts, 128]
        let indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                (0..n_concepts as i32).collect::<Vec<_>>(),
                [1, n_concepts],
            ),
            &device,
        );
        let concept_3d = self.concept_embs.forward(indices); // [1, n_concepts, 128]
        let concept_matrix = concept_3d.squeeze::<2>(); // [n_concepts, 128]

        // Affinity: projected @ concept_matrix^T → [N, n_concepts]
        let affinity = projected.matmul(concept_matrix.transpose());

        // Type logits from the full 384-dim embeddings
        let type_logits = self.type_classifier.forward(ngram_embs); // [N, 4]

        (affinity, type_logits)
    }
}

// =============================================================================
// Word-level pooling from subword embeddings
// =============================================================================

/// Mean-pool subword embeddings per whitespace word.
///
/// `token_embs`: flat [seq_len * 384] content token embeddings (no CLS/SEP)
/// `subword_to_word`: maps each subword index → word index
///
/// Returns: (word_embs [n_words, 384] as flat Vec, n_words)
pub fn words_from_subwords(
    token_embs: &[f32],
    subword_to_word: &[usize],
) -> (Vec<f32>, usize) {
    if subword_to_word.is_empty() {
        return (vec![], 0);
    }

    let n_words = subword_to_word.iter().max().unwrap_or(&0) + 1;
    let mut word_sums = vec![0.0f32; n_words * MINILM_DIM];
    let mut word_counts = vec![0usize; n_words];

    for (sw_idx, &word_idx) in subword_to_word.iter().enumerate() {
        let src_offset = sw_idx * MINILM_DIM;
        if src_offset + MINILM_DIM > token_embs.len() { break; }
        let dst_offset = word_idx * MINILM_DIM;
        for j in 0..MINILM_DIM {
            word_sums[dst_offset + j] += token_embs[src_offset + j];
        }
        word_counts[word_idx] += 1;
    }

    // Divide by count
    for w in 0..n_words {
        let count = word_counts[w].max(1) as f32;
        let offset = w * MINILM_DIM;
        for j in 0..MINILM_DIM {
            word_sums[offset + j] /= count;
        }
    }

    (word_sums, n_words)
}

/// Generate unigram, bigram, and trigram candidates from word embeddings.
///
/// `word_embs`: flat [n_words * 384]
/// `n_words`: number of words
///
/// Returns: (ngram_embs flat [N * 384], spans Vec<(start_word, end_word)>)
/// where N = n_words + (n_words-1) + (n_words-2) = 3*n_words - 3 for n_words >= 3
pub fn generate_ngrams(
    word_embs: &[f32],
    n_words: usize,
) -> (Vec<f32>, Vec<(usize, usize)>) {
    if n_words == 0 {
        return (vec![], vec![]);
    }

    let mut ngram_embs = Vec::new();
    let mut spans = Vec::new();

    // Unigrams: each word as-is
    for i in 0..n_words {
        let offset = i * MINILM_DIM;
        ngram_embs.extend_from_slice(&word_embs[offset..offset + MINILM_DIM]);
        spans.push((i, i + 1));
    }

    // Bigrams: mean of consecutive word pairs
    if n_words >= 2 {
        for i in 0..n_words - 1 {
            let off_a = i * MINILM_DIM;
            let off_b = (i + 1) * MINILM_DIM;
            for j in 0..MINILM_DIM {
                ngram_embs.push((word_embs[off_a + j] + word_embs[off_b + j]) / 2.0);
            }
            spans.push((i, i + 2));
        }
    }

    // Trigrams: mean of consecutive word triples
    if n_words >= 3 {
        for i in 0..n_words - 2 {
            let off_a = i * MINILM_DIM;
            let off_b = (i + 1) * MINILM_DIM;
            let off_c = (i + 2) * MINILM_DIM;
            for j in 0..MINILM_DIM {
                ngram_embs.push(
                    (word_embs[off_a + j] + word_embs[off_b + j] + word_embs[off_c + j]) / 3.0
                );
            }
            spans.push((i, i + 3));
        }
    }

    (ngram_embs, spans)
}

// =============================================================================
// Greedy non-overlapping selection (inference)
// =============================================================================

/// A selected span from greedy decoding.
#[derive(Debug, Clone)]
pub struct SelectedSpan {
    pub word_start: usize,
    pub word_end: usize,     // exclusive
    pub text: String,
    pub span_type: SpanType,
    pub embedding: Vec<f32>, // 384-dim original n-gram embedding
    /// Top-k candidate matches: (schema_type, schema_id, score)
    pub candidates: Vec<(String, usize, f32)>,
}

/// Greedy non-overlapping selection of (n-gram, concept) pairs.
///
/// Uses discriminability-based stopping: for each n-gram, measures how "peaky"
/// its concept distribution is (top score minus mean). N-grams that strongly
/// prefer one concept are informative; flat distributions are noise words.
///
/// `affinity`: flat [N * n_concepts] affinity scores
/// `type_logits`: flat [N * 4] span type logits
/// `ngram_embs`: flat [N * 384] original n-gram embeddings
/// `spans`: [(start_word, end_word)] for each n-gram
/// `words`: whitespace-split words of the original query
/// `min_peak`: minimum (max - mean) to consider an n-gram informative
/// `top_k`: how many concept candidates to keep per span
/// `concept_map`: for converting concept indices back to (type, id)
pub fn greedy_select(
    affinity: &[f32],
    type_logits: &[f32],
    ngram_embs: &[f32],
    spans: &[(usize, usize)],
    words: &[&str],
    min_peak: f32,
    top_k: usize,
    concept_map: &ConceptMap,
) -> Vec<SelectedSpan> {
    let n = spans.len();
    let n_concepts = concept_map.total();
    if n == 0 || n_concepts == 0 { return vec![]; }

    // Pre-compute peakiness (max - mean) for each n-gram
    let mut peakiness = vec![0.0f32; n];
    let mut max_scores = vec![f32::NEG_INFINITY; n];
    for i in 0..n {
        let base = i * n_concepts;
        let mut sum = 0.0f32;
        let mut mx = f32::NEG_INFINITY;
        for c in 0..n_concepts {
            let s = affinity[base + c];
            sum += s;
            if s > mx { mx = s; }
        }
        let mean = sum / n_concepts as f32;
        peakiness[i] = mx - mean;
        max_scores[i] = mx;
    }

    let mut selected = Vec::new();
    let mut masked = vec![false; n]; // masked n-gram indices

    loop {
        // Find the unmasked n-gram with highest max concept score,
        // but only consider n-grams with sufficient peakiness
        let mut best_score = f32::NEG_INFINITY;
        let mut best_ngram = 0;
        let mut found = false;

        for i in 0..n {
            if masked[i] { continue; }
            if peakiness[i] < min_peak { continue; }
            if max_scores[i] > best_score {
                best_score = max_scores[i];
                best_ngram = i;
                found = true;
            }
        }

        if !found { break; }

        let (start, end) = spans[best_ngram];

        // Determine span type from type_logits argmax
        let type_base = best_ngram * SpanType::COUNT;
        let span_type_idx = (0..SpanType::COUNT)
            .max_by(|&a, &b| {
                type_logits[type_base + a]
                    .partial_cmp(&type_logits[type_base + b])
                    .unwrap()
            })
            .unwrap_or(0);
        let span_type = match span_type_idx {
            0 => SpanType::NounPhrase,
            1 => SpanType::Quantifier,
            2 => SpanType::Comparator,
            3 => SpanType::Intent,
            _ => SpanType::NounPhrase,
        };

        // Extract 384-dim embedding for this n-gram
        let emb_offset = best_ngram * MINILM_DIM;
        let embedding = ngram_embs[emb_offset..emb_offset + MINILM_DIM].to_vec();

        // Collect top-k concept candidates for this n-gram
        let mut concept_scores: Vec<(usize, f32)> = (0..n_concepts)
            .map(|c| (c, affinity[best_ngram * n_concepts + c]))
            .collect();
        concept_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let candidates: Vec<(String, usize, f32)> = concept_scores.iter()
            .take(top_k)
            .map(|&(c, score)| {
                let (schema_type, schema_id) = concept_map.from_idx(c);
                (schema_type.to_string(), schema_id, score)
            })
            .collect();

        let text = words[start..end.min(words.len())].join(" ");

        selected.push(SelectedSpan {
            word_start: start,
            word_end: end,
            text,
            span_type,
            embedding,
            candidates,
        });

        // Mask all n-grams overlapping these word positions
        for i in 0..n {
            if masked[i] { continue; }
            let (s, e) = spans[i];
            if s < end && e > start {
                masked[i] = true;
            }
        }
    }

    // Sort by word position for consistent output order
    selected.sort_by_key(|s| s.word_start);
    selected
}

// =============================================================================
// Concept-centric selection (multi-label, overlapping spans allowed)
// =============================================================================

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Concept-centric selection: for each concept, find the n-gram with highest
/// sigmoid score. Spans can overlap. Each span can map to multiple concepts.
///
/// This replaces greedy_select. Instead of "pick best n-gram, mask overlapping",
/// we flip it: "for each concept, pick best n-gram". The result is a set of
/// SelectedSpans where overlapping spans coexist and each span can carry
/// multiple concept candidates.
///
/// `affinity`: flat [N * n_concepts] raw affinity scores (pre-sigmoid)
/// `type_logits`: flat [N * 4] span type logits
/// `ngram_embs`: flat [N * 384] original n-gram embeddings
/// `spans`: [(start_word, end_word)] for each n-gram
/// `words`: whitespace-split words of the original query
/// `threshold`: minimum sigmoid score to activate a concept (e.g. 0.5)
/// `concept_map`: for converting concept indices back to (type, id)
pub fn concept_select(
    affinity: &[f32],
    type_logits: &[f32],
    ngram_embs: &[f32],
    spans: &[(usize, usize)],
    words: &[&str],
    threshold: f32,
    concept_map: &ConceptMap,
) -> Vec<SelectedSpan> {
    let n = spans.len();
    let n_concepts = concept_map.total();
    if n == 0 || n_concepts == 0 { return vec![]; }

    // For each concept, find the n-gram with the highest sigmoid score
    // concept_winner[c] = (best_ngram_idx, best_sigmoid_score)
    let mut concept_winner: Vec<(usize, f32)> = vec![(0, f32::NEG_INFINITY); n_concepts];

    for c in 0..n_concepts {
        for i in 0..n {
            let raw = affinity[i * n_concepts + c];
            let sig = sigmoid(raw);
            if sig > concept_winner[c].1 {
                concept_winner[c] = (i, sig);
            }
        }
    }

    // Group activated concepts by their winning n-gram
    // ngram_concepts[ngram_idx] = vec of (concept_idx, sigmoid_score)
    let mut ngram_concepts: std::collections::HashMap<usize, Vec<(usize, f32)>>
        = std::collections::HashMap::new();

    for (c, &(ng_idx, score)) in concept_winner.iter().enumerate() {
        if score >= threshold {
            ngram_concepts.entry(ng_idx).or_default().push((c, score));
        }
    }

    // Build SelectedSpans from the grouped concepts
    let mut selected: Vec<SelectedSpan> = ngram_concepts.into_iter()
        .map(|(ng_idx, mut concepts)| {
            let (start, end) = spans[ng_idx];

            // Sort concepts by score descending
            concepts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Determine span type from type_logits argmax
            let type_base = ng_idx * SpanType::COUNT;
            let span_type_idx = (0..SpanType::COUNT)
                .max_by(|&a, &b| {
                    type_logits[type_base + a]
                        .partial_cmp(&type_logits[type_base + b])
                        .unwrap()
                })
                .unwrap_or(0);
            let span_type = match span_type_idx {
                0 => SpanType::NounPhrase,
                1 => SpanType::Quantifier,
                2 => SpanType::Comparator,
                3 => SpanType::Intent,
                _ => SpanType::NounPhrase,
            };

            // Extract 384-dim embedding
            let emb_offset = ng_idx * MINILM_DIM;
            let embedding = ngram_embs[emb_offset..emb_offset + MINILM_DIM].to_vec();

            // Convert concept indices to (schema_type, schema_id, score)
            let candidates: Vec<(String, usize, f32)> = concepts.iter()
                .map(|&(c, score)| {
                    let (schema_type, schema_id) = concept_map.from_idx(c);
                    (schema_type.to_string(), schema_id, score)
                })
                .collect();

            let text = words[start..end.min(words.len())].join(" ");

            SelectedSpan {
                word_start: start,
                word_end: end,
                text,
                span_type,
                embedding,
                candidates,
            }
        })
        .collect();

    // Sort by word position
    selected.sort_by_key(|s| s.word_start);
    selected
}
