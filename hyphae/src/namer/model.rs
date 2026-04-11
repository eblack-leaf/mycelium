use crate::namer::vocab::CHAR_VOCAB_SIZE;
use burn::{
    config::Config,
    module::Module,
    nn::{
        gru::{Gru, GruConfig},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};

/// Config — `word_vocab_size` is set at training time from the profile's WordVocab.
#[derive(Config, Debug)]
pub struct NamerConfig {
    /// Size of the word output vocabulary (varies per profile, set after vocab is built).
    pub word_vocab_size: usize,
    #[config(default = 32)]
    pub char_embed_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
}

/// Char-level encoder → two word-level prediction heads.
///
/// The encoder reads the raw value string character by character and produces
/// a fixed context vector. Two independent linear heads then predict:
///   - `word1`: the first word of the name (required)
///   - `word2`: the second word, or STOP (index 0) for single-word names
///
/// Style lives entirely in the weights — each profile checkpoint maps the same
/// input to different word choices.
#[derive(Module, Debug)]
pub struct NamerModel<B: Backend> {
    char_embed: Embedding<B>,
    encoder:    Gru<B>,
    word1_head: Linear<B>,
    word2_head: Linear<B>,
}

impl NamerConfig {
    pub fn build<B: Backend>(&self, device: &B::Device) -> NamerModel<B> {
        NamerModel {
            char_embed: EmbeddingConfig::new(CHAR_VOCAB_SIZE, self.char_embed_dim).init(device),
            encoder:    GruConfig::new(self.char_embed_dim, self.hidden_dim, true).init(device),
            word1_head: LinearConfig::new(self.hidden_dim, self.word_vocab_size).init(device),
            word2_head: LinearConfig::new(self.hidden_dim, self.word_vocab_size).init(device),
        }
    }
}

impl<B: Backend> NamerModel<B> {
    /// Encode the value string into a context vector: [batch, hidden_dim]
    fn encode(&self, chars: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let embedded = self.char_embed.forward(chars); // [batch, seq, char_embed]
        let output   = self.encoder.forward(embedded, None); // [batch, seq, hidden]
        let [batch, seq, hidden] = output.dims();
        output
            .slice([0..batch, (seq - 1)..seq, 0..hidden])
            .squeeze_dim::<2>(1) // [batch, hidden]
    }

    /// Training forward — returns (word1_logits, word2_logits).
    pub fn forward(&self, chars: Tensor<B, 2, Int>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let ctx = self.encode(chars);
        let w1  = self.word1_head.forward(ctx.clone()); // [batch, vocab]
        let w2  = self.word2_head.forward(ctx);         // [batch, vocab]
        (w1, w2)
    }

    /// Training loss: sum of cross-entropy on both word positions.
    pub fn forward_loss(
        &self,
        chars:   Tensor<B, 2, Int>,
        target1: Tensor<B, 1, Int>,
        target2: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        use burn::tensor::activation::log_softmax;

        let (logits1, logits2) = self.forward(chars);
        let _vocab = self.word_vocab_size();

        let lp1 = log_softmax(logits1, 1);
        let lp2 = log_softmax(logits2, 1);

        let loss1 = lp1.gather(1, target1.unsqueeze_dim::<2>(1)).mean().neg();
        let loss2 = lp2.gather(1, target2.unsqueeze_dim::<2>(1)).mean().neg();
        loss1 + loss2
    }

    fn word_vocab_size(&self) -> usize {
        self.word1_head.weight.dims()[0]
    }
}
