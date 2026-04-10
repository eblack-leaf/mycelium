use crate::namer::{
    model::NamerModel,
    vocab::{decode_output, encode_input, OUT_EOS, OUT_MAX_LEN},
};
use burn::{
    tensor::{backend::Backend, ElementConversion, Int, Tensor},
};

/// Greedy decoding: at each step pick the highest-probability token.
pub fn generate<B: Backend>(
    model: &NamerModel<B>,
    value: &str,
    context: &str,
    device: &B::Device,
) -> String {
    // Build input: value + "|" + context, encoded and padded
    let combined = format!("{}|{}", value, context);
    let input_ids = encode_input(&combined);
    let input = Tensor::<B, 2, Int>::from_ints(
        input_ids.iter().map(|&x| x as i32).collect::<Vec<_>>().as_slice(),
        device,
    )
    .reshape([1, input_ids.len()]);

    // Encode
    let context_vec = model.encoder.forward(input);

    // Greedy decode
    let mut generated: Vec<usize> = Vec::with_capacity(OUT_MAX_LEN);
    // Start with a single BOS (token 0)
    generated.push(0);

    for _ in 0..OUT_MAX_LEN {
        let so_far = Tensor::<B, 2, Int>::from_ints(
            generated.iter().map(|&x| x as i32).collect::<Vec<_>>().as_slice(),
            device,
        )
        .reshape([1, generated.len()]);

        let logits = model.decoder.step(so_far, context_vec.clone()); // [1, out_vocab]
        let next = logits
            .squeeze::<1>()
            .argmax(0)
            .into_scalar()
            .elem::<i32>() as usize;

        if next == OUT_EOS {
            break;
        }
        generated.push(next);
    }

    // Strip BOS token before decoding
    decode_output(&generated[1..])
}
