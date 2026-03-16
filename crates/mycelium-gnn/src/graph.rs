// =============================================================================
// graph.rs — Pure Rust graph structures, no tensors
//
// Two graphs live here:
//   HeteroGraph  — the full heterogeneous graph container
//   NodeStore    — per-node-type feature storage (flat f32 vecs)
//   EdgeStore    — per-relation-type edge index storage
//
// The graph is topology + raw features. Tensors are only created in ops.rs
// and above when needed for actual computation.
// =============================================================================

use std::collections::HashMap;

// -----------------------------------------------------------------------------
// NodeStore
// Holds flat feature vectors for all nodes of one type.
// features is row-major: node i lives at features[i*feat_dim .. (i+1)*feat_dim]
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct NodeStore {
    /// Raw features, shape logically [n_nodes, feat_dim]
    pub features: Vec<f32>,
    pub n_nodes: usize,
    pub feat_dim: usize,
}

impl NodeStore {
    pub fn new(features: Vec<f32>, n_nodes: usize) -> Self {
        assert_eq!(features.len() % n_nodes, 0, "features.len() must be divisible by n_nodes");
        let feat_dim = features.len() / n_nodes;
        Self { features, n_nodes, feat_dim }
    }

    /// Get the feature slice for node i
    pub fn node_features(&self, i: usize) -> &[f32] {
        &self.features[i * self.feat_dim..(i + 1) * self.feat_dim]
    }
}

// -----------------------------------------------------------------------------
// EdgeStore
// Holds edge indices for one relation type (src_type, edge_type, dst_type).
// src_indices[k] and dst_indices[k] define the k-th directed edge.
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct EdgeStore {
    /// Source node index for each edge (into src_type's NodeStore)
    pub src_indices: Vec<usize>,
    /// Destination node index for each edge (into dst_type's NodeStore)
    pub dst_indices: Vec<usize>,
}

impl EdgeStore {
    pub fn new(src_indices: Vec<usize>, dst_indices: Vec<usize>) -> Self {
        assert_eq!(
            src_indices.len(),
            dst_indices.len(),
            "src and dst index lists must be same length"
        );
        Self { src_indices, dst_indices }
    }

    pub fn n_edges(&self) -> usize {
        self.src_indices.len()
    }
}

// -----------------------------------------------------------------------------
// RelationKey
// Identifies a relation type as (src_node_type, edge_type, dst_node_type).
// Used as HashMap key throughout.
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RelationKey {
    pub src_type: String,
    pub edge_type: String,
    pub dst_type: String,
}

impl RelationKey {
    pub fn new(src: &str, edge: &str, dst: &str) -> Self {
        Self {
            src_type: src.to_string(),
            edge_type: edge.to_string(),
            dst_type: dst.to_string(),
        }
    }
}

// -----------------------------------------------------------------------------
// HeteroGraph
// The top-level container. Holds:
//   node_stores: node_type → NodeStore
//   edge_stores: RelationKey → EdgeStore
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct HeteroGraph {
    pub node_stores: HashMap<String, NodeStore>,
    pub edge_stores: HashMap<RelationKey, EdgeStore>,
}

impl HeteroGraph {
    pub fn new() -> Self {
        Self {
            node_stores: HashMap::new(),
            edge_stores: HashMap::new(),
        }
    }

    pub fn add_nodes(&mut self, node_type: &str, features: Vec<f32>, n_nodes: usize) {
        self.node_stores.insert(
            node_type.to_string(),
            NodeStore::new(features, n_nodes),
        );
    }

    pub fn add_edges(
        &mut self,
        src_type: &str,
        edge_type: &str,
        dst_type: &str,
        src_indices: Vec<usize>,
        dst_indices: Vec<usize>,
    ) {
        self.edge_stores.insert(
            RelationKey::new(src_type, edge_type, dst_type),
            EdgeStore::new(src_indices, dst_indices),
        );
    }

    /// All relation keys in this graph
    pub fn relation_keys(&self) -> Vec<&RelationKey> {
        self.edge_stores.keys().collect()
    }

    /// Collect all unique node types referenced by edges
    pub fn node_types(&self) -> Vec<&str> {
        let mut types: Vec<&str> = self.node_stores.keys().map(|s| s.as_str()).collect();
        types.sort();
        types.dedup();
        types
    }
}

impl Default for HeteroGraph {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------------
// Example: build a small schema graph for testing
//
// Collections: "person", "order"
// Fields:      "person.name", "person.age", "order.total"
// Relations:   person -[HAS_FIELD]-> person.name
//              person -[HAS_FIELD]-> person.age
//              order  -[HAS_FIELD]-> order.total
//              person -[LINKS_TO]->  order
// -----------------------------------------------------------------------------
pub fn make_example_graph(feat_dim: usize) -> HeteroGraph {
    let mut g = HeteroGraph::new();

    // 2 collection nodes, 3 field nodes
    // features are random-ish for testing — in real use these come from
    // an embedding of the schema entity name + type
    let collection_feats: Vec<f32> = (0..2 * feat_dim).map(|i| i as f32 * 0.1).collect();
    let field_feats: Vec<f32> = (0..3 * feat_dim).map(|i| i as f32 * 0.05).collect();

    g.add_nodes("collection", collection_feats, 2); // [person=0, order=1]
    g.add_nodes("field", field_feats, 3);            // [name=0, age=1, total=2]

    // collection -[HAS_FIELD]-> field
    // person(0) -> name(0), person(0) -> age(1), order(1) -> total(2)
    g.add_edges(
        "collection", "HAS_FIELD", "field",
        vec![0, 0, 1],
        vec![0, 1, 2],
    );

    // collection -[LINKS_TO]-> collection
    // person(0) -> order(1)
    g.add_edges(
        "collection", "LINKS_TO", "collection",
        vec![0],
        vec![1],
    );

    g
}
