use crate::namer::{
    model::NamerModel,
    vocab::{encode_value, WordVocab},
};
use burn::tensor::{backend::Backend, ElementConversion, Int, Tensor};

/// Generate a name for `value` using the loaded model and vocabulary.
pub fn generate<B: Backend>(
    model: &NamerModel<B>,
    vocab: &WordVocab,
    value: &str,
    device: &B::Device,
) -> String {
    let ids: Vec<i32> = encode_value(value).iter().map(|&x| x as i32).collect();
    let chars = Tensor::<B, 2, Int>::from_ints(ids.as_slice(), device)
        .reshape([1, ids.len()]);

    let (logits1, logits2) = model.forward(chars);

    let w1 = logits1.squeeze::<1>().argmax(0).into_scalar().elem::<i32>() as usize;
    let w2 = logits2.squeeze::<1>().argmax(0).into_scalar().elem::<i32>() as usize;

    vocab.decode_name(w1, w2)
}
