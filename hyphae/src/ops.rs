// ops.rs — scatter / gather / normalize primitives

use burn::tensor::{backend::Backend, IndexingUpdateOp, Int, Tensor};

/// Gather rows from `src` by index.
/// src: [n, dim], indices: node indices → output: [indices.len(), dim]
pub fn gather<B: Backend>(
    src: Tensor<B, 2>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let idx = Tensor::<B, 1, Int>::from_ints(
        indices
            .iter()
            .map(|&i| i as i32)
            .collect::<Vec<_>>()
            .as_slice(),
        device,
    );
    src.select(0, idx)
}

/// Scatter-add `values` into a zero tensor of shape [n_dst, dim] at `dst_indices`.
/// values: [n_edges, dim], dst_indices: one per edge → output: [n_dst, dim]
///
/// Uses Burn's native scatter with IndexingUpdateOp::Add. Backward is a simple gather.
pub fn scatter_add<B: Backend>(
    values: Tensor<B, 2>,
    dst_indices: &[usize],
    n_dst: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let dim = values.dims()[1];
    let n_edges = dst_indices.len();
    if n_edges == 0 {
        return Tensor::zeros([n_dst, dim], device);
    }

    // Expand dst_indices [n_edges] → [n_edges, dim] to match values shape
    let idx = Tensor::<B, 1, Int>::from_ints(
        dst_indices.iter().map(|&i| i as i32).collect::<Vec<_>>().as_slice(),
        device,
    ).unsqueeze_dim::<2>(1).expand([n_edges, dim]);

    let out: Tensor<B, 2> = Tensor::zeros([n_dst, dim], device);
    out.scatter(0, idx, values, IndexingUpdateOp::Add)
}

/// Row-wise L2 normalization. Adds epsilon to denominator to avoid div-by-zero.
/// x: [n, dim] → [n, dim]
/// sum_dim(1) keeps the dimension → [n, 1], which broadcasts over [n, dim].
pub fn l2_normalize<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = (x.clone() * x.clone()).sum_dim(1).sqrt() + 1e-8;
    x / norm
}
