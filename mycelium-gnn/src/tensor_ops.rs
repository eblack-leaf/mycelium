// =============================================================================
// ops.rs — Scatter / Gather / L2 normalize tensor primitives
// =============================================================================

use burn::{
    prelude::*,
    tensor::{Int, IndexingUpdateOp, TensorData, backend::Backend},
};

/// Gather source node features along edges.
/// node_features: [n_nodes, feat_dim], src_indices: per-edge source node ids
/// returns: [n_edges, feat_dim]
pub fn gather<B: Backend>(
    node_features: Tensor<B, 2>,
    src_indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let n_edges = src_indices.len();
    let feat_dim = node_features.dims()[1];

    let flat: Vec<i32> = src_indices
        .iter()
        .flat_map(|&idx| std::iter::repeat(idx as i32).take(feat_dim))
        .collect();

    let indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(flat, [n_edges, feat_dim]),
        device,
    );

    node_features.gather(0, indices)
}

/// Scatter-mean: aggregate per-edge messages to destination nodes.
/// messages: [n_edges, feat_dim], dst_indices: per-edge destination node ids
/// returns: [n_dst, feat_dim]
pub fn scatter_mean<B: Backend>(
    messages: Tensor<B, 2>,
    dst_indices: &[usize],
    n_dst: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let n_edges = dst_indices.len();
    let feat_dim = messages.dims()[1];

    let flat: Vec<i32> = dst_indices
        .iter()
        .flat_map(|&dst| std::iter::repeat(dst as i32).take(feat_dim))
        .collect();

    let indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(flat, [n_edges, feat_dim]),
        device,
    );

    let acc = Tensor::<B, 2>::zeros([n_dst, feat_dim], device);
    let summed = acc.scatter(0, indices, messages, IndexingUpdateOp::Add);

    // Per-node degree, clamped to 1.0 to avoid div-by-zero
    let mut degrees = vec![0.0f32; n_dst];
    for &dst in dst_indices {
        degrees[dst] += 1.0;
    }
    let degrees: Vec<f32> = degrees.iter().map(|&d| d.max(1.0)).collect();

    let degree_t = Tensor::<B, 1>::from_data(
        TensorData::new(degrees, [n_dst]),
        device,
    )
    .reshape([n_dst, 1])
    .expand([n_dst, feat_dim]);

    summed / degree_t
}

/// Weighted scatter-mean: same as scatter_mean but each edge message is scaled
/// by its weight before summing, and degree counts are replaced by weight sums.
/// weights must have the same length as dst_indices.
pub fn scatter_weighted_mean<B: Backend>(
    messages: Tensor<B, 2>,
    dst_indices: &[usize],
    weights: &[f32],
    n_dst: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let n_edges = dst_indices.len();
    let feat_dim = messages.dims()[1];

    // Scale each message by its weight: messages[i] *= weights[i]
    let weight_col = Tensor::<B, 1>::from_data(
        TensorData::new(weights.to_vec(), [n_edges]),
        device,
    )
    .reshape([n_edges, 1])
    .expand([n_edges, feat_dim]);

    let weighted_msgs = messages * weight_col;

    let flat: Vec<i32> = dst_indices
        .iter()
        .flat_map(|&dst| std::iter::repeat(dst as i32).take(feat_dim))
        .collect();

    let indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(flat, [n_edges, feat_dim]),
        device,
    );

    let acc = Tensor::<B, 2>::zeros([n_dst, feat_dim], device);
    let summed = acc.scatter(0, indices, weighted_msgs, IndexingUpdateOp::Add);

    // Normalize by sum of weights per destination (not count)
    let mut weight_sums = vec![0.0f32; n_dst];
    for (i, &dst) in dst_indices.iter().enumerate() {
        weight_sums[dst] += weights[i];
    }
    let weight_sums: Vec<f32> = weight_sums.iter().map(|&w| w.max(1e-6)).collect();

    let denom = Tensor::<B, 1>::from_data(
        TensorData::new(weight_sums, [n_dst]),
        device,
    )
    .reshape([n_dst, 1])
    .expand([n_dst, feat_dim]);

    summed / denom
}

/// Row-wise L2 normalization.
/// input: [n, feat_dim], returns: [n, feat_dim] with unit-norm rows.
pub fn l2_normalize<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let [n, feat_dim] = x.dims();

    let norm = x
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-6)
        .expand([n, feat_dim]);

    x / norm
}
