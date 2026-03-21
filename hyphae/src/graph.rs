use crate::query::{ModifierKind, QueryNode};
use crate::rgcn::{Edge, EdgeType, TypedEdges};
use crate::schema::{FieldType, Schema};
use septa::{Comparator, Intent, Semantics};
use std::collections::HashMap;

/// Fixed vocab node count at the head of every node list: 4 ops + 7 cmps + 3 modifiers.
pub(crate) const VOCAB_NODE_COUNT: usize = 14;

// =============================================================================
// Char n-gram hashing
// =============================================================================

/// FNV-1a 64-bit hash — deterministic, no std dependency.
pub fn fnv1a(s: &str) -> u64 {
    const PRIME: u64 = 1_099_511_628_211;
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    let mut h = OFFSET;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Extract character 2- and 3-gram bucket indices for `name`.
/// Same name → same indices every call (deterministic pure function).
/// Duplicates removed so no bucket is double-weighted in the mean.
pub(crate) fn char_ngram_buckets(name: &str, num_buckets: usize) -> Vec<usize> {
    let chars: Vec<char> = name.chars().collect();
    let mut seen = std::collections::HashSet::new();
    let mut buckets = Vec::new();

    for n in [2usize, 3] {
        for window in chars.windows(n) {
            let s: String = window.iter().collect();
            let bucket = fnv1a(&s) as usize % num_buckets;
            if seen.insert(bucket) {
                buckets.push(bucket);
            }
        }
    }

    if buckets.is_empty() {
        // single-char or empty name fallback
        buckets.push(fnv1a(name) as usize % num_buckets);
    }

    buckets
}

// =============================================================================
// SchemaGraph
// =============================================================================

pub struct SchemaGraph {
    #[allow(dead_code)]
    schema: Schema,
    nodes: Vec<QueryNode>,
    edges: TypedEdges,
    #[allow(dead_code)]
    ngram_buckets: usize,
    /// Precomputed n-gram bucket indices for nodes[VOCAB_NODE_COUNT..].
    /// Computed once in new() and copied into every GroundedGraph by inject().
    schema_ngram_indices: Vec<Vec<usize>>,
}

impl SchemaGraph {
    pub fn new(schema: Schema, ngram_buckets: usize) -> Self {
        let mut nodes: Vec<QueryNode> = Vec::new();
        let mut edges = TypedEdges::new();

        for et in EdgeType::all() {
            edges.insert(et.clone(), vec![]);
        }

        // ── Fixed vocabulary nodes ──────────────────────────────────────────

        for op in [Intent::Select, Intent::Create, Intent::Update, Intent::Delete] {
            nodes.push(QueryNode::Operation(op));
        }
        for cmp in [
            Comparator::Eq, Comparator::Neq,
            Comparator::Gt, Comparator::Gte,
            Comparator::Lt, Comparator::Lte,
            Comparator::Contains,
        ] {
            nodes.push(QueryNode::Comparator(cmp));
        }
        nodes.push(QueryNode::Modifier(ModifierKind::Fetch));
        nodes.push(QueryNode::Modifier(ModifierKind::OrderBy));
        nodes.push(QueryNode::Modifier(ModifierKind::Limit));

        debug_assert_eq!(nodes.len(), VOCAB_NODE_COUNT);

        // ── Schema nodes ────────────────────────────────────────────────────

        let mut table_map: HashMap<String, usize> = HashMap::new();

        for table in schema.tables.iter() {
            let idx = nodes.len();
            nodes.push(QueryNode::Table(table.name.clone()));
            table_map.insert(table.name.clone(), idx);
        }

        for table in schema.tables.iter() {
            let table_idx = *table_map.get(&table.name).unwrap();

            for field in table.fields.iter() {
                let field_idx = nodes.len();
                nodes.push(QueryNode::Field { table: table.name.clone(), name: field.name.clone() });

                edges.get_mut(&EdgeType::HasField).unwrap().push(Edge { src: table_idx, dst: field_idx });
                edges.get_mut(&EdgeType::FieldOf).unwrap().push(Edge  { src: field_idx, dst: table_idx });

                if let FieldType::Record { ref tables } = field.field_type {
                    for linked_name in tables {
                        if let Some(&linked_idx) = table_map.get(linked_name) {
                            edges.get_mut(&EdgeType::LinksTo).unwrap().push(Edge { src: field_idx, dst: linked_idx });
                            edges.get_mut(&EdgeType::LinkedFrom).unwrap().push(Edge { src: linked_idx, dst: field_idx });
                        }
                    }
                }
            }
        }

        // ── Precompute n-gram indices for schema nodes ──────────────────────
        // nodes[VOCAB_NODE_COUNT..] are Table and Field nodes with real name strings.
        // The hash function is deterministic: identical name → identical bucket list
        // every time, so the same rows of ngram_table receive gradient each step.
        let schema_ngram_indices: Vec<Vec<usize>> = nodes[VOCAB_NODE_COUNT..]
            .iter()
            .map(|node| match node {
                QueryNode::Table(name)        => char_ngram_buckets(name, ngram_buckets),
                QueryNode::Field { name, .. } => char_ngram_buckets(name, ngram_buckets),
                _ => vec![],
            })
            .collect();

        Self { schema, nodes, edges, ngram_buckets, schema_ngram_indices }
    }

    pub fn inject(&self, semantics: &Semantics) -> GroundedGraph {
        let mut nodes = self.nodes.clone();
        let mut edges = self.edges.clone();

        let op_indices: Vec<usize> = self.nodes.iter().enumerate()
            .filter_map(|(i, n)| matches!(n, QueryNode::Operation(_)).then_some(i))
            .collect();
        let cmp_indices: Vec<usize> = self.nodes.iter().enumerate()
            .filter_map(|(i, n)| matches!(n, QueryNode::Comparator(_)).then_some(i))
            .collect();
        let mod_indices: Vec<usize> = self.nodes.iter().enumerate()
            .filter_map(|(i, n)| matches!(n, QueryNode::Modifier(_)).then_some(i))
            .collect();
        let table_indices: Vec<usize> = self.nodes.iter().enumerate()
            .filter_map(|(i, n)| matches!(n, QueryNode::Table(_)).then_some(i))
            .collect();
        let field_indices: Vec<usize> = self.nodes.iter().enumerate()
            .filter_map(|(i, n)| matches!(n, QueryNode::Field { .. }).then_some(i))
            .collect();

        let schema_node_count = nodes.len();

        // ── Typed span nodes ─────────────────────────────────────────────
        //   Order must match init_node_features in Hyphae:
        //   intent, entity, projections, modifiers, conditions, assignments.
        //
        // Each span type has role-specific edges:
        //   IntentSpan/EntitySpan: no edges (BiGRU sufficient)
        //   ProjSpan:  ProjToTable → all tables
        //   CondSpan:  CondToTable → all tables, CondToCmp → comparator vocab
        //   AsgnSpan:  AsgnToTable → all tables
        //   ModSpan:   ModToTable → all tables (if has field arg)

        let intent_idx = nodes.len();
        nodes.push(QueryNode::IntentSpan);
        let intent_resolution = Resolution { span_index: intent_idx, candidates: op_indices.clone() };

        let entity_idx = nodes.len();
        nodes.push(QueryNode::EntitySpan);
        let entity_resolution = Resolution { span_index: entity_idx, candidates: table_indices.clone() };

        let mut projection_resolutions = Vec::new();
        let mut proj_span_indices: Vec<usize> = Vec::new();
        for _proj in &semantics.projections {
            let idx = nodes.len();
            nodes.push(QueryNode::ProjSpan);
            for &t in &table_indices {
                edges.get_mut(&EdgeType::ProjToTable).unwrap().push(Edge { src: idx, dst: t });
            }
            proj_span_indices.push(idx);
            projection_resolutions.push(Resolution { span_index: idx, candidates: field_indices.clone() });
        }

        let mut modifier_type_resolutions  = Vec::new();
        let mut modifier_field_resolutions = Vec::new();
        let mut mod_span_indices: Vec<usize> = Vec::new();
        for modifier in &semantics.modifiers {
            let idx = nodes.len();
            nodes.push(QueryNode::ModSpan);
            mod_span_indices.push(idx);
            modifier_type_resolutions.push(Resolution { span_index: idx, candidates: mod_indices.clone() });
            // Connect to modifier vocab nodes (Fetch/OrderBy/Limit) — analogous to CondToCmp
            for &m in &mod_indices {
                edges.get_mut(&EdgeType::ModToModifier).unwrap().push(Edge { src: idx, dst: m });
            }
            if modifier.argument.is_some() {
                for &t in &table_indices {
                    edges.get_mut(&EdgeType::ModToTable).unwrap().push(Edge { src: idx, dst: t });
                }
                modifier_field_resolutions.push(Resolution { span_index: idx, candidates: field_indices.clone() });
            }
        }

        for (pi, proj) in semantics.projections.iter().enumerate() {
            if let Some(fi) = proj.fetch_index {
                if fi < mod_span_indices.len() {
                    edges.get_mut(&EdgeType::ProjectionToFetch).unwrap().push(Edge {
                        src: proj_span_indices[pi],
                        dst: mod_span_indices[fi],
                    });
                }
            }
        }

        let mut condition_field_resolutions = Vec::new();
        let mut condition_cmp_resolutions   = Vec::new();
        for _cond in &semantics.conditions {
            let idx = nodes.len();
            nodes.push(QueryNode::CondSpan);
            for &t in &table_indices {
                edges.get_mut(&EdgeType::CondToTable).unwrap().push(Edge { src: idx, dst: t });
            }
            for &c in &cmp_indices {
                edges.get_mut(&EdgeType::CondToCmp).unwrap().push(Edge { src: idx, dst: c });
            }
            condition_field_resolutions.push(Resolution { span_index: idx, candidates: field_indices.clone() });
            condition_cmp_resolutions.push(Resolution   { span_index: idx, candidates: cmp_indices.clone() });
        }

        let mut assignment_resolutions = Vec::new();
        for assign in &semantics.assignments {
            let idx = nodes.len();
            nodes.push(QueryNode::AsgnSpan);
            if assign.field_text.is_some() {
                for &t in &table_indices {
                    edges.get_mut(&EdgeType::AsgnToTable).unwrap().push(Edge { src: idx, dst: t });
                }
                assignment_resolutions.push(Resolution { span_index: idx, candidates: field_indices.clone() });
            }
        }

        GroundedGraph {
            nodes,
            edges,
            schema_node_count,
            schema_ngram_indices: self.schema_ngram_indices.clone(),
            intent_resolution,
            entity_resolution,
            projection_resolutions,
            condition_field_resolutions,
            condition_cmp_resolutions,
            assignment_resolutions,
            modifier_type_resolutions,
            modifier_field_resolutions,
        }
    }
}

// =============================================================================
// GroundedGraph
// =============================================================================

pub struct Resolution {
    pub span_index: usize,
    pub candidates: Vec<usize>,
}

pub struct GroundedGraph {
    pub nodes: Vec<QueryNode>,
    pub edges: TypedEdges,

    /// Where schema nodes end and span nodes begin.
    pub schema_node_count: usize,

    /// Precomputed char n-gram bucket indices for nodes[VOCAB_NODE_COUNT..schema_node_count].
    /// schema_ngram_indices[i] → nodes[VOCAB_NODE_COUNT + i].
    ///
    /// These are *lookup indices* into ngram_table (a learned Embedding parameter).
    /// The indices themselves never change for a given schema — only the embedding
    /// values at those buckets are updated by the optimizer. Same name → same
    /// buckets → same rows receive gradient every step.
    pub schema_ngram_indices: Vec<Vec<usize>>,

    pub intent_resolution:           Resolution,
    pub entity_resolution:           Resolution,
    pub projection_resolutions:      Vec<Resolution>,
    pub condition_field_resolutions: Vec<Resolution>,
    pub condition_cmp_resolutions:   Vec<Resolution>,
    pub assignment_resolutions:      Vec<Resolution>,
    pub modifier_type_resolutions:   Vec<Resolution>,
    pub modifier_field_resolutions:  Vec<Resolution>,
}

