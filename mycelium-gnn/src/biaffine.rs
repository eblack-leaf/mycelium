// =============================================================================
// biaffine.rs — Biaffine dependency parser head (Burn module)
//
// Two tasks on MiniLM token-level embeddings:
//   Task 1: BIO sequence tagger (Linear 384→9)
//   Task 2: Biaffine dependency parser on extracted span representations
//           - Head/Dep MLPs (384→128→128)
//           - Arc scorer (biaffine W + bias)
//           - Relation classifier (4 biaffine scorers, one per DepRelation)
//
// Output: LinguisticGraph (same structure as rule_based_parse)
// =============================================================================

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor, activation},
};
use crate::nlp::DepRelation;
use crate::biaffine_data::BioTag;

pub const HIDDEN_DIM: usize = 384;
pub const ARC_DIM: usize = 128;

// =============================================================================
// Biaffine head module
// =============================================================================

#[derive(Module, Debug)]
pub struct BiaffineHead<B: Backend> {
    /// BIO sequence classifier: hidden_dim → 9 tags
    pub bio_classifier: Linear<B>,

    /// Head MLP: hidden_dim → arc_dim → arc_dim
    pub head_mlp_1: Linear<B>,
    pub head_mlp_2: Linear<B>,

    /// Dep MLP: hidden_dim → arc_dim → arc_dim
    pub dep_mlp_1: Linear<B>,
    pub dep_mlp_2: Linear<B>,

    /// Arc scorer: biaffine weight (arc_dim → arc_dim) + bias (arc_dim → 1)
    pub arc_weight: Linear<B>,
    pub arc_bias: Linear<B>,

    /// Relation classifiers: one biaffine weight per DepRelation (4 total)
    pub rel_weight_0: Linear<B>,
    pub rel_weight_1: Linear<B>,
    pub rel_weight_2: Linear<B>,
    pub rel_weight_3: Linear<B>,
}

impl<B: Backend> BiaffineHead<B> {
    /// Create a new biaffine head with random weights.
    pub fn new(device: &B::Device) -> Self {
        Self {
            bio_classifier: LinearConfig::new(HIDDEN_DIM, BioTag::COUNT).init(device),

            head_mlp_1: LinearConfig::new(HIDDEN_DIM, ARC_DIM).init(device),
            head_mlp_2: LinearConfig::new(ARC_DIM, ARC_DIM).init(device),

            dep_mlp_1: LinearConfig::new(HIDDEN_DIM, ARC_DIM).init(device),
            dep_mlp_2: LinearConfig::new(ARC_DIM, ARC_DIM).init(device),

            arc_weight: LinearConfig::new(ARC_DIM, ARC_DIM).init(device),
            arc_bias: LinearConfig::new(ARC_DIM, 1).init(device),

            rel_weight_0: LinearConfig::new(ARC_DIM, ARC_DIM).init(device),
            rel_weight_1: LinearConfig::new(ARC_DIM, ARC_DIM).init(device),
            rel_weight_2: LinearConfig::new(ARC_DIM, ARC_DIM).init(device),
            rel_weight_3: LinearConfig::new(ARC_DIM, ARC_DIM).init(device),
        }
    }

    /// BIO logits: [seq_len, hidden_dim] → [seq_len, 9]
    pub fn forward_bio(&self, token_embs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.bio_classifier.forward(token_embs)
    }

    /// Head MLP: [n, hidden_dim] → [n, arc_dim]
    fn head_mlp(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::relu(self.head_mlp_1.forward(x));
        self.head_mlp_2.forward(h)
    }

    /// Dep MLP: [n, hidden_dim] → [n, arc_dim]
    fn dep_mlp(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::relu(self.dep_mlp_1.forward(x));
        self.dep_mlp_2.forward(h)
    }

    /// Dependency scores from span embeddings.
    ///
    /// Input: span_embs [n_spans, hidden_dim]
    /// Returns: (arc_scores [n_spans, n_spans], rel_logits [n_spans, n_spans, 4])
    ///
    /// arc_scores: sigmoid of biaffine(dep, head) — probability of arc presence
    /// rel_logits: raw logits for each relation type at each (src, dst) pair
    pub fn forward_deps(
        &self,
        span_embs: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let n_spans = span_embs.dims()[0];
        let device = span_embs.device();

        let h_head = self.head_mlp(span_embs.clone()); // [n, arc_dim]
        let h_dep = self.dep_mlp(span_embs);            // [n, arc_dim]

        // Arc scores: dep * W * head^T + dep * b
        // arc_weight.forward(h_dep) → [n, arc_dim], matmul h_head^T → [n, n]
        let dep_w = self.arc_weight.forward(h_dep.clone()); // [n, arc_dim]
        let arc_bilinear = dep_w.matmul(h_head.clone().transpose()); // [n, n]

        // Bias term: dep * b → [n, 1] → broadcast
        let arc_bias = self.arc_bias.forward(h_dep.clone()); // [n, 1]
        let arc_scores = activation::sigmoid(arc_bilinear + arc_bias);

        // Relation logits: 4 biaffine scorers
        let rel_weights = [&self.rel_weight_0, &self.rel_weight_1, &self.rel_weight_2, &self.rel_weight_3];
        let mut rel_planes: Vec<Tensor<B, 2>> = Vec::new();
        for w in &rel_weights {
            let dep_wr = w.forward(h_dep.clone()); // [n, arc_dim]
            let plane = dep_wr.matmul(h_head.clone().transpose()); // [n, n]
            rel_planes.push(plane);
        }

        // Stack into [n, n, 4]
        let rel_logits = if n_spans > 0 {
            // Reshape each [n, n] → [n, n, 1] then cat
            let planes: Vec<Tensor<B, 3>> = rel_planes.into_iter()
                .map(|p| p.reshape([n_spans, n_spans, 1]))
                .collect();
            Tensor::cat(planes, 2)
        } else {
            Tensor::zeros([0, 0, DepRelation::COUNT], &device)
        };

        (arc_scores, rel_logits)
    }
}

// =============================================================================
// Mean-pool token embeddings within span boundaries
// =============================================================================

/// Mean-pool token embeddings within each span to produce span-level embeddings.
///
/// `token_embs`: [seq_len, hidden_dim] — content token embeddings (no CLS/SEP)
/// `subword_to_word`: maps subword index → word index
/// `spans`: (start_word, end_word_exclusive) per span
///
/// Returns: [n_spans, hidden_dim]
pub fn mean_pool_spans<B: Backend>(
    token_embs: &Tensor<B, 2>,
    subword_to_word: &[usize],
    spans: &[(usize, usize)],
) -> Tensor<B, 2> {
    let [_seq_len, hidden_dim] = token_embs.dims();
    let device = token_embs.device();
    let n_spans = spans.len();

    if n_spans == 0 {
        return Tensor::zeros([0, hidden_dim], &device);
    }

    let mut span_vecs: Vec<Tensor<B, 2>> = Vec::new();

    for &(start_word, end_word) in spans {
        // Collect subword indices that fall within this word-level span
        let indices: Vec<usize> = subword_to_word.iter()
            .enumerate()
            .filter(|&(_, &w)| w >= start_word && w < end_word)
            .map(|(i, _)| i)
            .collect();

        if indices.is_empty() {
            span_vecs.push(Tensor::zeros([1, hidden_dim], &device));
            continue;
        }

        // Gather and mean-pool
        let idx_data: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
        let idx_tensor = Tensor::<B, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::from(idx_data.as_slice()),
            &device,
        );
        let gathered = token_embs.clone().select(0, idx_tensor); // [k, hidden_dim]
        let pooled = gathered.mean_dim(0); // [1, hidden_dim]
        span_vecs.push(pooled);
    }

    Tensor::cat(span_vecs, 0) // [n_spans, hidden_dim]
}
