// =============================================================================
// ops.rs — Scatter / Gather tensor primitives
//
// These are the two fundamental GNN tensor operations:
//
//   gather_node_features  — index into node feature matrix by edge source indices
//                           result[k] = node_features[src_indices[k]]
//
//   scatter_mean          — aggregate edge messages back to destination nodes
//                           result[dst] = mean of all messages where dst_indices[k] == dst
//
// Burn 0.20.1 API notes used here:
//   - tensor.gather(dim, indices)           — gather rows/cols by index tensor
//   - tensor.scatter(dim, indices, values, IndexingUpdateOp::Add)
//   - Tensor::<B,1,Int>::from_data(TensorData::new(vec, shape), device)
//   - tensor.zeros_like() / Tensor::zeros(shape, device)
// =============================================================================

use burn::{
    prelude::*,
    tensor::{
        Int, IndexingUpdateOp, TensorData,
        backend::Backend,
    },
};

// -----------------------------------------------------------------------------
// gather_node_features
//
// Given node feature matrix and a list of source node indices (one per edge),
// return a matrix of shape [n_edges, feat_dim] where row k is the feature
// vector of the source node of edge k.
//
// node_features : [n_nodes, feat_dim]
// src_indices   : &[usize]  — length n_edges
// returns       : [n_edges, feat_dim]
//
// Burn op: tensor.gather(dim=0, indices)
// indices must be shape [n_edges, feat_dim] with the same index repeated
// across columns (selecting the same row for all features of that node).
// -----------------------------------------------------------------------------
pub fn gather_node_features<B: Backend>(
    node_features: Tensor<B, 2>,
    src_indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let n_edges = src_indices.len();
    let feat_dim = node_features.dims()[1];

    // Build index tensor [n_edges, feat_dim]
    // Each row k contains the same value src_indices[k] repeated feat_dim times
    // because gather(dim=0) selects row src_indices[k][col] for each column.
    let flat_indices: Vec<i32> = src_indices
        .iter()
        .flat_map(|&idx| std::iter::repeat(idx as i32).take(feat_dim))
        .collect();

    let indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(flat_indices, [n_edges, feat_dim]),
        device,
    );

    // gather(dim=0, indices) — for each (i,j): output[i,j] = input[indices[i,j], j]
    node_features.gather(0, indices)
}

// -----------------------------------------------------------------------------
// scatter_mean
//
// Aggregate per-edge messages into destination nodes by mean pooling.
//
// messages    : [n_edges, feat_dim]  — one message vector per edge
// dst_indices : &[usize]             — which dest node each message goes to
// n_dst_nodes : usize
// returns     : [n_dst_nodes, feat_dim]
//
// Burn op: tensor.scatter(dim=0, indices, values, IndexingUpdateOp::Add)
// This is the native scatter-add in 0.20.x — no matmul trick needed.
// -----------------------------------------------------------------------------
pub fn scatter_mean<B: Backend>(
    messages: Tensor<B, 2>,
    dst_indices: &[usize],
    n_dst_nodes: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let n_edges = dst_indices.len();
    let feat_dim = messages.dims()[1];

    // Build scatter index tensor [n_edges, feat_dim]
    // Same pattern as gather: repeat the destination index across all feat columns
    let flat_indices: Vec<i32> = dst_indices
        .iter()
        .flat_map(|&dst| std::iter::repeat(dst as i32).take(feat_dim))
        .collect();

    let indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(flat_indices, [n_edges, feat_dim]),
        device,
    );

    // Zero accumulator for destination nodes
    let accumulator = Tensor::<B, 2>::zeros([n_dst_nodes, feat_dim], device);

    // scatter(dim=0, indices, values, Add) — atomically adds messages into accumulator
    // output[indices[i,j], j] += messages[i,j]
    let summed = accumulator.scatter(0, indices, messages, IndexingUpdateOp::Add);

    // Compute per-node degree for mean normalization
    let mut degrees = vec![0.0f32; n_dst_nodes];
    for &dst in dst_indices {
        degrees[dst] += 1.0;
    }
    // Clamp to 1.0 so isolated nodes get identity (sum/1 = sum) not NaN (sum/0)
    let degrees: Vec<f32> = degrees.iter().map(|&d| d.max(1.0)).collect();

    let degree_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(degrees, [n_dst_nodes]),
        device,
    )
    // Reshape to [n_dst_nodes, 1] then expand to [n_dst_nodes, feat_dim] for broadcast
    .reshape([n_dst_nodes, 1])
    .expand([n_dst_nodes, feat_dim]);

    summed / degree_tensor
}

// -----------------------------------------------------------------------------
// l2_normalize
//
// Row-wise L2 normalization. Standard in GraphSAGE after combining
// self + neighbor representations.
//
// input  : [n, feat_dim]
// returns: [n, feat_dim]  each row has unit L2 norm
// -----------------------------------------------------------------------------
pub fn l2_normalize<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let dims = x.dims();
    let n = dims[0];
    let feat_dim = dims[1];

    // Row-wise L2 norm: [n, 1]
    let norm = x
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)       // [n, 1]
        .sqrt()
        .clamp_min(1e-6); // avoid division by zero

    // Expand norm to [n, feat_dim] for element-wise division
    let norm_expanded = norm.expand([n, feat_dim]);

    x / norm_expanded
}
