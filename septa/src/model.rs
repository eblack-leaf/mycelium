// model.rs — BiGRU architecture for semantic span extraction

use crate::Semantics;
use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, gru::{Gru, GruConfig}},
    tensor::{backend::Backend, Int, Tensor},
};

#[derive(Debug, Config)]
pub struct SeptaConfig {
    #[config(default = 256)]
    pub hidden_dim: usize,
    #[config(default = 2)]
    pub num_layers: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = 10_000)]
    pub vocab_size: usize,
    #[config(default = 128)]
    pub embed_dim: usize,
    /// Number of BIO tag classes — set from the tag set size, no default.
    pub num_tags: usize,
}

/// BiGRU hidden states for every span in one query, shape [2 * hidden_dim] per entry.
/// Produced by Septa; consumed by Hyphae to initialise span node features.
pub struct SpanHiddens<B: Backend> {
    pub intent:      Tensor<B, 1>,
    pub entity:      Tensor<B, 1>,
    pub projections: Vec<Tensor<B, 1>>,
    /// Per-condition field sub-span hiddens (pooled over field_start..field_end).
    pub cond_fields: Vec<Tensor<B, 1>>,
    /// Per-condition comparator sub-span hiddens (pooled over cmp_start..cmp_end).
    pub cond_cmps:   Vec<Tensor<B, 1>>,
    /// Per-assignment field sub-span hiddens (pooled over field_start..field_end).
    pub asgn_fields: Vec<Tensor<B, 1>>,
    /// Per-modifier type sub-span hiddens (pooled over start..end, the keyword text).
    pub mod_types:   Vec<Tensor<B, 1>>,
    /// Per-modifier field sub-span hiddens (pooled over arg_start..arg_end).
    /// Only present for modifiers with argument (OrderBy field, Fetch field).
    pub mod_fields:  Vec<Tensor<B, 1>>,
    /// Which modifiers have a field argument (true = has ModFieldSpan node).
    /// Needed by init_node_features to interleave type/field spans correctly.
    pub mod_has_field: Vec<bool>,
}

/// Manual BiGRU layer: forward Gru + reverse Gru, outputs concatenated.
#[derive(Module, Debug)]
pub struct BiGruLayer<B: Backend> {
    pub forward_gru: Gru<B>,
    pub reverse_gru: Gru<B>,
}

impl<B: Backend> BiGruLayer<B> {
    pub fn new(d_input: usize, d_hidden: usize, device: &B::Device) -> Self {
        Self {
            forward_gru: GruConfig::new(d_input, d_hidden, true).init(device),
            reverse_gru: GruConfig::new(d_input, d_hidden, true).init(device),
        }
    }

    /// [batch, seq_len, d_input] → [batch, seq_len, 2 * d_hidden]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let fwd = self.forward_gru.forward(x.clone(), None); // [batch, seq, hidden]
        let rev = self.reverse_gru.forward(x.flip([1]), None).flip([1]); // reverse in, reverse out
        Tensor::cat(vec![fwd, rev], 2)
    }
}

/// BiGRU span encoder. Embeds tokens via char n-gram hashing, runs stacked BiGRU,
/// then mean-pools outputs over known span boundaries to produce SpanHiddens.
#[derive(Module, Debug)]
pub struct Septa<B: Backend> {
    pub ngram_table: Embedding<B>,
    pub layers: Vec<BiGruLayer<B>>,
    pub embed_dim: usize,
    pub hidden_dim: usize,
}

impl<B: Backend> Septa<B> {
    pub fn new(config: &SeptaConfig, device: &B::Device) -> Self {
        let ngram_table = EmbeddingConfig::new(config.vocab_size, config.embed_dim).init(device);

        let mut layers: Vec<BiGruLayer<B>> = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let d_input = if i == 0 { config.embed_dim } else { 2 * config.hidden_dim };
            layers.push(BiGruLayer::new(d_input, config.hidden_dim, device));
        }

        Self { ngram_table, layers, embed_dim: config.embed_dim, hidden_dim: config.hidden_dim }
    }

    /// Embed a single token → [embed_dim].
    fn embed_token(&self, token: &str, num_buckets: usize, device: &B::Device) -> Tensor<B, 1> {
        let buckets = char_ngram_buckets(token, num_buckets);
        let idx_data: Vec<i32> = buckets.iter().map(|&b| b as i32).collect();
        let idx = Tensor::<B, 1, Int>::from_data(idx_data.as_slice(), device)
            .unsqueeze::<2>(); // [1, n_grams]
        let emb = self.ngram_table.forward(idx); // [1, n_grams, embed_dim]
        let n = buckets.len();
        let embed_dim = emb.dims()[2];
        let rows: Tensor<B, 2> = emb.reshape([n, embed_dim]);
        rows.mean_dim(0).squeeze::<1>()
    }

    /// Embed and pad a batch of token sequences → [batch, max_len, embed_dim].
    fn embed_batch(
        &self,
        token_seqs: &[Vec<String>],
        max_len: usize,
        num_buckets: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let batch: Vec<Tensor<B, 2>> = token_seqs.iter().map(|tokens| {
            let mut embedded: Vec<Tensor<B, 1>> = tokens.iter()
                .map(|t| self.embed_token(t, num_buckets, device))
                .collect();
            // Zero-pad to max_len
            for _ in tokens.len()..max_len {
                embedded.push(Tensor::zeros([self.embed_dim], device));
            }
            Tensor::<B, 1>::stack::<2>(embedded, 0) // [max_len, embed_dim]
        }).collect();

        Tensor::<B, 2>::stack::<3>(batch, 0) // [batch, max_len, embed_dim]
    }

    /// Batch encode: tokenized sequences → [batch, max_len, 2 * hidden_dim].
    pub fn encode_batch(
        &self,
        token_seqs: &[Vec<String>],
        max_len: usize,
        num_buckets: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let mut x = self.embed_batch(token_seqs, max_len, num_buckets, device);
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    /// Batch forward: encode a batch of datums, extract per-datum SpanHiddens.
    pub fn batch_forward_with_spans(
        &self,
        texts: &[&str],
        semantics: &[&Semantics],
        num_buckets: usize,
        device: &B::Device,
    ) -> Vec<SpanHiddens<B>> {
        // Tokenize all texts
        let tokenized: Vec<(Vec<String>, Vec<(usize, usize)>)> =
            texts.iter().map(|t| tokenize(t)).collect();

        let token_seqs: Vec<Vec<String>> = tokenized.iter().map(|(t, _)| t.clone()).collect();
        let max_len = token_seqs.iter().map(|s| s.len()).max().unwrap_or(1);

        let h = self.encode_batch(&token_seqs, max_len, num_buckets, device);
        // h: [batch, max_len, 2 * hidden_dim]

        tokenized.iter().enumerate().map(|(i, (_, char_ranges))| {
            let seq_len = char_ranges.len();
            // Extract this datum's hidden states: [seq_len, 2*hidden]
            let h_i = h.clone().slice([i..i+1, 0..seq_len]).squeeze_dim::<2>(0);
            let sem = semantics[i];

            let pool = |start: usize, end: usize| {
                pool_span(&h_i, char_ranges, start, end)
            };

            SpanHiddens {
                intent: pool(sem.intent.start, sem.intent.end),
                entity: pool(sem.entity.start, sem.entity.end),
                projections: sem.projections.iter().map(|s| pool(s.start, s.end)).collect(),
                cond_fields: sem.conditions.iter().map(|s| pool(s.field_start, s.field_end)).collect(),
                cond_cmps: sem.conditions.iter().map(|s| pool(s.cmp_start, s.cmp_end)).collect(),
                asgn_fields: sem.assignments.iter()
                    .filter(|s| s.field_text.is_some())
                    .map(|s| pool(s.field_start, s.field_end))
                    .collect(),
                mod_types: sem.modifiers.iter().map(|s| pool(s.start, s.end)).collect(),
                mod_fields: sem.modifiers.iter()
                    .filter(|s| s.argument.is_some())
                    .map(|s| pool(s.arg_start, s.arg_end))
                    .collect(),
                mod_has_field: sem.modifiers.iter().map(|s| s.argument.is_some()).collect(),
            }
        }).collect()
    }

    /// Single-datum forward (convenience wrapper).
    pub fn forward_with_spans(
        &self,
        text: &str,
        semantics: &Semantics,
        num_buckets: usize,
        device: &B::Device,
    ) -> SpanHiddens<B> {
        self.batch_forward_with_spans(&[text], &[semantics], num_buckets, device)
            .into_iter().next().unwrap()
    }

    /// Full forward: encode → CRF decode → Semantics + SpanHiddens.
    /// CRF not yet implemented; use forward_with_spans for training.
    pub fn forward(&self, _tokens: &[&str], _device: &B::Device) -> (Semantics, SpanHiddens<B>) {
        todo!()
    }
}

// =============================================================================
// Span pooling
// =============================================================================

/// Mean-pool BiGRU outputs over a character span, converting char offsets
/// to token indices. Returns [2 * hidden_dim].
fn pool_span<B: Backend>(
    h: &Tensor<B, 2>,
    token_char_ranges: &[(usize, usize)],
    char_start: usize,
    char_end: usize,
) -> Tensor<B, 1> {
    let mut tok_start = None;
    let mut tok_end = None;
    for (i, &(cs, ce)) in token_char_ranges.iter().enumerate() {
        if cs <= char_start && char_start < ce && tok_start.is_none() {
            tok_start = Some(i);
        }
        if cs < char_end && char_end <= ce {
            tok_end = Some(i + 1);
        }
    }
    let tok_start = tok_start.unwrap_or(0);
    let tok_end = tok_end.unwrap_or(token_char_ranges.len()).max(tok_start + 1);

    let span_h = h.clone().slice([tok_start..tok_end]); // [span_len, hidden]
    span_h.mean_dim(0).squeeze::<1>()
}

// =============================================================================
// Tokenization
// =============================================================================

/// Whitespace tokenizer that tracks character offsets for each token.
pub fn tokenize(text: &str) -> (Vec<String>, Vec<(usize, usize)>) {
    let mut tokens = Vec::new();
    let mut ranges = Vec::new();
    let mut i = 0;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        tokens.push(text[start..i].to_lowercase());
        ranges.push((start, i));
    }

    (tokens, ranges)
}

// =============================================================================
// Char n-gram hashing (local copy — septa can't depend on hyphae)
// =============================================================================

fn fnv1a(s: &str) -> u64 {
    const PRIME: u64 = 1_099_511_628_211;
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    let mut h = OFFSET;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

fn char_ngram_buckets(name: &str, num_buckets: usize) -> Vec<usize> {
    let chars: Vec<char> = name.chars().collect();
    let mut seen = std::collections::HashSet::new();
    let mut buckets = Vec::new();

    for n in [2usize, 3] {
        for window in chars.windows(n) {
            let s: String = window.iter().collect();
            let bucket = fnv1a(&s) as usize % num_buckets;
            if seen.insert(bucket) {
                buckets.push(bucket);
            }
        }
    }

    if buckets.is_empty() {
        buckets.push(fnv1a(name) as usize % num_buckets);
    }

    buckets
}
