// model.rs — SageConv + bilinear heads GNN architecture

use crate::graph::{GroundedGraph, Resolution, VOCAB_NODE_COUNT};
use crate::ops;
use crate::query::QueryIr;
use crate::sage::SageConv;
use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Int, Tensor},
};
use septa::model::SpanHiddens;

#[derive(Debug, Config)]
pub struct HyphaeConfig {
    #[config(default = 256)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub num_layers: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Node feature dimensionality throughout the GNN.
    #[config(default = 128)]
    pub node_feat_dim: usize,
    /// Septa's BiLSTM hidden_dim — span_proj input is 2× this (bidirectional).
    #[config(default = 256)]
    pub septa_hidden_dim: usize,
    /// Number of char n-gram hash buckets for schema node embeddings.
    #[config(default = 50_000)]
    pub ngram_buckets: usize,
}

/// Training output — per-head logit tensors carrying gradient.
pub struct HyphaeLogits<B: Backend> {
    pub intent:          Tensor<B, 1>,
    pub entity:          Tensor<B, 1>,
    pub projection:      Vec<Tensor<B, 1>>,
    pub condition_field: Vec<Tensor<B, 1>>,
    pub condition_cmp:   Vec<Tensor<B, 1>>,
    pub assignment:      Vec<Tensor<B, 1>>,
    pub modifier_type:   Vec<Tensor<B, 1>>,
    pub modifier_field:  Vec<Tensor<B, 1>>,
}

/// R-GCN / GraphSAGE GNN with bilinear resolution heads.
#[derive(Module, Debug)]
pub struct Hyphae<B: Backend> {
    pub sage: SageConv<B>,

    /// Learned embeddings for the 14 fixed vocab nodes (ops, comparators, modifiers).
    /// Shape [14, node_feat_dim]. Looked up by enum variant index — same index every
    /// forward pass, so the same row trains consistently.
    pub vocab_emb: Embedding<B>,

    /// Char n-gram hash table for schema nodes (Table/Field names).
    /// Shape [ngram_buckets, node_feat_dim]. A name's bucket indices are fixed by
    /// the FNV hash — only the values at those rows change (via optimizer updates).
    pub ngram_table: Embedding<B>,

    /// Projects BiLSTM span hiddens [2 * septa_hidden_dim] → [node_feat_dim].
    pub span_proj: Linear<B>,

    // Bilinear resolution heads — [hidden_dim, hidden_dim], no bias.
    pub bilinear_intent:    Linear<B>,
    pub bilinear_entity:    Linear<B>,
    pub bilinear_proj:      Linear<B>,
    pub bilinear_cond_f:    Linear<B>,
    pub bilinear_cond_c:    Linear<B>,
    pub bilinear_assign:    Linear<B>,
    pub bilinear_mod_type:  Linear<B>,
    pub bilinear_mod_field: Linear<B>,
}

impl<B: Backend> Hyphae<B> {
    pub fn new(config: &HyphaeConfig, device: &B::Device) -> Self {
        let bilinear = |device: &B::Device| {
            LinearConfig::new(config.hidden_dim, config.hidden_dim)
                .with_bias(false)
                .init(device)
        };
        Self {
            sage: SageConv::new(config.node_feat_dim, config.hidden_dim, config.num_layers, device),
            vocab_emb:   EmbeddingConfig::new(14, config.node_feat_dim).init(device),
            ngram_table: EmbeddingConfig::new(config.ngram_buckets, config.node_feat_dim).init(device),
            span_proj:   LinearConfig::new(2 * config.septa_hidden_dim, config.node_feat_dim).init(device),
            bilinear_intent:    bilinear(device),
            bilinear_entity:    bilinear(device),
            bilinear_proj:      bilinear(device),
            bilinear_cond_f:    bilinear(device),
            bilinear_cond_c:    bilinear(device),
            bilinear_assign:    bilinear(device),
            bilinear_mod_type:  bilinear(device),
            bilinear_mod_field: bilinear(device),
        }
    }

    /// Build the initial [num_nodes, node_feat_dim] feature matrix before message passing.
    ///
    /// Node ranges (matching the order inject() builds them):
    ///   [0..VOCAB_NODE_COUNT)          vocab nodes   → vocab_emb lookup by index
    ///   [VOCAB_NODE_COUNT..schema_end) schema nodes  → ngram_table lookup + mean
    ///   [schema_end..)                 span nodes    → span_proj(BiLSTM hidden)
    ///
    /// The embedding lookups read rows from parameter matrices; they do not replace them.
    /// Backprop writes gradients to those rows; the optimizer then updates the values.
    fn init_node_features(
        &self,
        graph: &GroundedGraph,
        hiddens: &SpanHiddens<B>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let mut feats: Vec<Tensor<B, 1>> = Vec::with_capacity(graph.nodes.len());

        // ── Vocab nodes: one embedding row per enum variant index ─────────
        // Embedding::forward takes [batch, seq] Int tensor → [batch, seq, feat_dim].
        // We use [1, 1] to look up a single row, then flatten to [feat_dim].
        for i in 0..VOCAB_NODE_COUNT {
            let idx = Tensor::<B, 1, Int>::from_data([i as i32], device)
                .unsqueeze::<2>();                                    // [1, 1]
            feats.push(self.vocab_emb.forward(idx).flatten(0, 2));   // [feat_dim]
        }

        // ── Schema nodes: mean of n-gram bucket rows ──────────────────────
        // Bucket indices are fixed for a given name (FNV hash).
        // Only the values at those rows in ngram_table change during training.
        for ngram_idx in &graph.schema_ngram_indices {
            let idx_data: Vec<i32> = ngram_idx.iter().map(|&b| b as i32).collect();
            let idx = Tensor::<B, 1, Int>::from_data(idx_data.as_slice(), device)
                .unsqueeze::<2>();                                         // [1, n_grams]
            let rows = self.ngram_table.forward(idx).squeeze::<2>();       // [n_grams, feat_dim]
            feats.push(rows.mean_dim(0).squeeze::<1>());                   // [feat_dim]
        }

        // ── Span nodes: project BiLSTM hiddens ───────────────────────────
        // Must match the order inject() pushes QueryNode::Span:
        //   intent, entity, projections, modifiers, conditions, assignments.
        feats.push(self.span_proj.forward(hiddens.intent.clone()));
        feats.push(self.span_proj.forward(hiddens.entity.clone()));
        for h in &hiddens.projections  { feats.push(self.span_proj.forward(h.clone())); }
        for h in &hiddens.modifiers    { feats.push(self.span_proj.forward(h.clone())); }
        for h in &hiddens.conditions   { feats.push(self.span_proj.forward(h.clone())); }
        for h in &hiddens.assignments  { feats.push(self.span_proj.forward(h.clone())); }

        Tensor::stack(feats, 0)
    }

    pub fn forward(
        &self,
        graph: &GroundedGraph,
        hiddens: &SpanHiddens<B>,
        device: &B::Device,
    ) -> QueryIr {
        let _features = self.init_node_features(graph, hiddens, device);
        // Next: self.sage.forward(_features, &graph.edges, graph.nodes.len(), device)
        // Then: bilinear heads score span embeddings vs candidate embeddings → QueryIr
        todo!()
    }

    /// Score a single resolution: span embedding projected through bilinear head,
    /// then dot-product against each candidate embedding → logits [num_candidates].
    fn score_resolution(
        bilinear: &Linear<B>,
        h: &Tensor<B, 2>,
        res: &Resolution,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let span = ops::gather(h.clone(), &[res.span_index], device);  // [1, hidden_dim]
        let cands = ops::gather(h.clone(), &res.candidates, device);   // [n, hidden_dim]
        let proj = bilinear.forward(span);                              // [1, hidden_dim]
        cands.matmul(proj.transpose()).squeeze::<1>()                   // [n]
    }

    /// Training forward: init features → message passing → bilinear head logits.
    pub fn forward_train(
        &self,
        graph: &GroundedGraph,
        hiddens: &SpanHiddens<B>,
        device: &B::Device,
    ) -> HyphaeLogits<B> {
        let features = self.init_node_features(graph, hiddens, device);
        let h = self.sage.forward(features, &graph.edges, graph.nodes.len(), device);

        let score = |bilinear: &Linear<B>, res: &Resolution| {
            Self::score_resolution(bilinear, &h, res, device)
        };

        HyphaeLogits {
            intent: score(&self.bilinear_intent, &graph.intent_resolution),
            entity: score(&self.bilinear_entity, &graph.entity_resolution),
            projection:      graph.projection_resolutions.iter().map(|r| score(&self.bilinear_proj, r)).collect(),
            condition_field: graph.condition_field_resolutions.iter().map(|r| score(&self.bilinear_cond_f, r)).collect(),
            condition_cmp:   graph.condition_cmp_resolutions.iter().map(|r| score(&self.bilinear_cond_c, r)).collect(),
            assignment:      graph.assignment_resolutions.iter().map(|r| score(&self.bilinear_assign, r)).collect(),
            modifier_type:   graph.modifier_type_resolutions.iter().map(|r| score(&self.bilinear_mod_type, r)).collect(),
            modifier_field:  graph.modifier_field_resolutions.iter().map(|r| score(&self.bilinear_mod_field, r)).collect(),
        }
    }
}
