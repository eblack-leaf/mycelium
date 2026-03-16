// =============================================================================
// GNN in Burn 0.20.1 — Isolated, Self-Contained Implementation
// GraphSAGE-style heterogeneous message passing
//
// Cargo.toml deps:
//   burn = { version = "0.20.1", features = ["ndarray"] }
//
// Structure:
//   graph.rs          — graph data structures (no tensors)
//   ops.rs            — scatter / gather tensor primitives
//   sage_conv.rs      — single SAGEConv layer
//   hetero_conv.rs    — one SAGEConv per relation type
//   encoder.rs        — stacked hetero_conv layers
//   lib.rs            — re-exports + usage example
// =============================================================================

pub mod graph;
pub mod ops;
pub mod sage_conv;
pub mod hetero_conv;
pub mod encoder;
pub mod step_by_step;
