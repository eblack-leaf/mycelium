// =============================================================================
// reranker.rs — Schema-trained re-ranker (Burn module)
//
// Replaces pretrained cross-encoder scoring with a learned match function.
// Input:  concat(phrase_embedding, schema_name_embedding) = 768-dim
// Output: match probability (sigmoid)
//
// Trained on (phrase, schema_name) pairs from template ground truth,
// with MiniLM encode_pooled embeddings for both sides.
// =============================================================================

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor, activation},
};

pub const EMB_DIM: usize = 384;

#[derive(Module, Debug)]
pub struct Reranker<B: Backend> {
    pub fc1: Linear<B>,    // 768 → 128
    pub fc2: Linear<B>,    // 128 → 64
    pub fc3: Linear<B>,    // 64 → 1
}

impl<B: Backend> Reranker<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(EMB_DIM * 2, 128).init(device),
            fc2: LinearConfig::new(128, 64).init(device),
            fc3: LinearConfig::new(64, 1).init(device),
        }
    }

    /// Score a batch of (phrase, schema_name) pairs.
    /// phrase_embs:  [batch, 384]
    /// schema_embs:  [batch, 384]
    /// Returns:      [batch] logits (pre-sigmoid)
    pub fn forward(
        &self,
        phrase_embs: Tensor<B, 2>,
        schema_embs: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let concat = Tensor::cat(vec![phrase_embs, schema_embs], 1); // [batch, 768]
        let h = activation::relu(self.fc1.forward(concat));
        let h = activation::relu(self.fc2.forward(h));
        self.fc3.forward(h).squeeze::<1>() // [batch]
    }

    /// Score and return probabilities.
    pub fn predict(
        &self,
        phrase_embs: Tensor<B, 2>,
        schema_embs: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        activation::sigmoid(self.forward(phrase_embs, schema_embs))
    }
}
