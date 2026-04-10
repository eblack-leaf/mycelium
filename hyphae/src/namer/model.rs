use crate::namer::vocab::{IN_VOCAB_SIZE, OUT_VOCAB_SIZE};
use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig,
        Linear, LinearConfig,
        gru::{Gru, GruConfig},
    },
    tensor::{backend::Backend, Int, Tensor},
};

// ── Encoder ──────────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct EncoderConfig {
    #[config(default = "IN_VOCAB_SIZE")]
    pub vocab_size: usize,
    #[config(default = 64)]
    pub embed_dim: usize,
    #[config(default = 128)]
    pub hidden_dim: usize,
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    embed: Embedding<B>,
    gru: Gru<B>,
}

impl EncoderConfig {
    pub fn build<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        Encoder {
            embed: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            gru: GruConfig::new(self.embed_dim, self.hidden_dim, true).init(device),
        }
    }
}

impl<B: Backend> Encoder<B> {
    /// Returns the final hidden state: [batch, hidden_dim]
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let embedded = self.embed.forward(input); // [batch, seq, embed_dim]
        let output = self.gru.forward(embedded, None); // [batch, seq, hidden_dim]
        let [batch, seq, hidden] = output.dims();
        // Take last timestep as context vector
        output
            .slice([0..batch, (seq - 1)..seq, 0..hidden])
            .squeeze_dim::<2>(1)
    }
}

// ── Decoder ──────────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct DecoderConfig {
    #[config(default = "OUT_VOCAB_SIZE")]
    pub vocab_size: usize,
    #[config(default = 64)]
    pub embed_dim: usize,
    #[config(default = 128)]
    pub hidden_dim: usize,
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    embed: Embedding<B>,
    gru: Gru<B>,
    proj: Linear<B>,
}

impl DecoderConfig {
    pub fn build<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        Decoder {
            embed: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            gru: GruConfig::new(self.embed_dim, self.hidden_dim, true).init(device),
            proj: LinearConfig::new(self.hidden_dim, self.vocab_size).init(device),
        }
    }
}

impl<B: Backend> Decoder<B> {
    /// Teacher-forcing forward pass.
    /// `targets`: [batch, out_seq] — the expected output tokens (including EOS).
    /// `context`: [batch, hidden_dim] — encoder final hidden state (used as initial GRU state).
    /// Returns logits: [batch, out_seq, out_vocab_size]
    pub fn forward(
        &self,
        targets: Tensor<B, 2, Int>,
        context: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let embedded = self.embed.forward(targets); // [batch, out_seq, embed_dim]
        let output = self.gru.forward(embedded, Some(context)); // [batch, out_seq, hidden_dim]
        self.proj.forward(output) // [batch, out_seq, out_vocab_size]
    }

    /// Step-wise inference: given all tokens generated so far, return logits for next token.
    /// `so_far`: [batch, step] — tokens generated so far.
    /// `context`: [batch, hidden_dim] — encoder context (initial hidden state).
    /// Returns logits for the next position: [batch, out_vocab_size]
    pub fn step(
        &self,
        so_far: Tensor<B, 2, Int>,
        context: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch, step] = so_far.dims();
        let embedded = self.embed.forward(so_far); // [batch, step, embed_dim]
        let output = self.gru.forward(embedded, Some(context)); // [batch, step, hidden_dim]
        // Take last step's output
        self.proj
            .forward(output.slice([0..batch, (step - 1)..step, 0..128]).squeeze_dim::<2>(1))
    }
}

// ── NamerModel ───────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct NamerConfig {
    #[config(default = "EncoderConfig::new()")]
    pub encoder: EncoderConfig,
    #[config(default = "DecoderConfig::new()")]
    pub decoder: DecoderConfig,
}

#[derive(Module, Debug)]
pub struct NamerModel<B: Backend> {
    pub encoder: Encoder<B>,
    pub decoder: Decoder<B>,
}

impl NamerConfig {
    pub fn build<B: Backend>(&self, device: &B::Device) -> NamerModel<B> {
        NamerModel {
            encoder: self.encoder.build(device),
            decoder: self.decoder.build(device),
        }
    }
}

impl<B: Backend> NamerModel<B> {
    /// Training forward: returns cross-entropy loss.
    pub fn forward_loss(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        use burn::tensor::activation::log_softmax;

        let context = self.encoder.forward(input);
        // Decoder input = targets shifted right (drop last token, prepend 0 as BOS)
        let [batch, seq] = targets.dims();
        let bos = Tensor::<B, 2, Int>::zeros([batch, 1], &targets.device());
        let decoder_input = Tensor::cat(vec![bos, targets.clone().slice([0..batch, 0..(seq - 1)])], 1);
        let logits = self.decoder.forward(decoder_input, context); // [batch, seq, out_vocab]

        // Cross-entropy: log_softmax + NLL
        let log_probs = log_softmax(logits, 2); // [batch, seq, out_vocab]
        let [_, _, vocab] = log_probs.dims();
        let log_probs_flat = log_probs.reshape([batch * seq, vocab]);
        let targets_flat = targets.reshape([batch * seq]);
        // Gather log probs at target indices
        let gathered = log_probs_flat.gather(1, targets_flat.unsqueeze_dim::<2>(1));
        gathered.mean().neg()
    }
}
