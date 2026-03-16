// =============================================================================
// sage_conv.rs — Single SAGEConv Layer
//
// GraphSAGE convolution for ONE relation type (one src_type → dst_type edge set).
//
// What it does per forward pass:
//   1. Gather source node features along edges         [n_edges, feat_dim]
//   2. Aggregate gathered features to dst nodes (mean) [n_dst, feat_dim]
//   3. Project aggregated neighbors: W_neigh * agg     [n_dst, out_dim]
//   4. Project self features:        W_self  * dst     [n_dst, out_dim]
//   5. Sum projections + activate + L2 normalize
//
// Why GraphSAGE and not GCN?
//   GCN requires the full adjacency matrix at training time (transductive).
//   SAGE is inductive — it works on any graph topology seen at inference,
//   including schemas you've never trained on. Essential for a generic framework.
//
// Module fields (learned parameters):
//   neighbor_proj : Linear(in_dim → out_dim)  — transforms neighbor aggregation
//   self_proj     : Linear(in_dim → out_dim)  — transforms self features
// =============================================================================

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use burn::tensor::activation;
use crate::ops::{gather_node_features, scatter_mean, l2_normalize};

#[derive(Module, Debug)]
pub struct SAGEConv<B: Backend> {
    pub neighbor_proj: Linear<B>,
    pub self_proj: Linear<B>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl<B: Backend> SAGEConv<B> {
    /// Create a new SAGEConv layer.
    /// in_dim  — input feature dimension (same for src and dst in homo case;
    ///           in hetero case this is the src embedding dim after projection)
    /// out_dim — output feature dimension
    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self {
            neighbor_proj: LinearConfig::new(in_dim, out_dim)
                .with_bias(true)
                .init(device),
            self_proj: LinearConfig::new(in_dim, out_dim)
                .with_bias(false)
                .init(device),
            in_dim,
            out_dim,
        }
    }

    /// Forward pass for this relation type.
    ///
    /// src_features : [n_src_nodes, in_dim]  — source node embeddings
    /// dst_features : [n_dst_nodes, in_dim]  — destination node embeddings
    /// src_indices  : edge source node indices (into src_features rows)
    /// dst_indices  : edge destination node indices (into dst_features rows)
    ///
    /// returns      : [n_dst_nodes, out_dim]  updated destination embeddings
    pub fn forward(
        &self,
        src_features: Tensor<B, 2>,
        dst_features: Tensor<B, 2>,
        src_indices: &[usize],
        dst_indices: &[usize],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let n_dst = dst_features.dims()[0];

        // --- Step 1 & 2: gather then aggregate neighbor features ---
        // gathered : [n_edges, in_dim]
        let gathered = gather_node_features(src_features, src_indices, device);

        // neighbor_agg : [n_dst, in_dim]
        let neighbor_agg = scatter_mean(gathered, dst_indices, n_dst, device);

        // --- Step 3: transform neighbor aggregation ---
        // [n_dst, out_dim]
        let neighbor_out = self.neighbor_proj.forward(neighbor_agg);

        // --- Step 4: transform self features ---
        // [n_dst, out_dim]
        let self_out = self.self_proj.forward(dst_features);

        // --- Step 5: combine, activate, normalize ---
        // SAGE original paper concatenates then projects; here we sum the two
        // projections (equivalent when W_concat = [W_neigh | W_self] and
        // the input is [agg; self]) — simpler and same expressiveness.
        let combined = activation::relu(neighbor_out + self_out);

        // L2 normalize rows — keeps embeddings on unit hypersphere,
        // prevents scale explosion across layers
        l2_normalize(combined)
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray;

    #[test]
    fn test_sage_conv_output_shape() {
        let device = Default::default();
        let in_dim = 4;
        let out_dim = 8;
        let n_src = 3;
        let n_dst = 2;

        let conv = SAGEConv::<B>::new(in_dim, out_dim, &device);

        // src: 3 nodes, dst: 2 nodes
        let src_feat = Tensor::<B, 2>::from_data(
            TensorData::new(
                (0..n_src * in_dim).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
                [n_src, in_dim],
            ),
            &device,
        );
        let dst_feat = Tensor::<B, 2>::from_data(
            TensorData::new(
                (0..n_dst * in_dim).map(|i| i as f32 * 0.2).collect::<Vec<_>>(),
                [n_dst, in_dim],
            ),
            &device,
        );

        // Edges: src[0]->dst[0], src[1]->dst[0], src[2]->dst[1]
        let src_idx = vec![0, 1, 2];
        let dst_idx = vec![0, 0, 1];

        let out = conv.forward(src_feat, dst_feat, &src_idx, &dst_idx, &device);

        assert_eq!(out.dims(), [n_dst, out_dim]);
    }

    #[test]
    fn test_sage_conv_l2_normalized() {
        let device = Default::default();
        let conv = SAGEConv::<B>::new(4, 4, &device);

        let src_feat = Tensor::<B, 2>::ones([3, 4], &device);
        let dst_feat = Tensor::<B, 2>::ones([2, 4], &device);

        let out = conv.forward(src_feat, dst_feat, &[0, 1, 2], &[0, 0, 1], &device);

        // Each row should have approximately unit L2 norm
        let norms = out
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt();

        let norms_data = norms.into_data();
        for val in norms_data.as_slice::<f32>().unwrap() {
            assert!((val - 1.0).abs() < 1e-5, "norm {} not close to 1.0", val);
        }
    }
}
