use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    Embedding, EmbeddingConfig,
    Linear, LinearConfig,
    transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
};
use burn::tensor::{Bool, ElementConversion, Int, Tensor, activation, backend::Backend};

use crate::catalog::SchemaCatalog;
use crate::embed::char_ngram_buckets;
use crate::query::{
    ModifierKind, QueryIr,
    ResolvedAssignment, ResolvedCondition, ResolvedField, ResolvedModifier, ValueRef,
};

// =============================================================================
// Config
// =============================================================================

#[derive(Config, Debug)]
pub struct LamellaConfig {
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 512)]
    pub d_ff: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    #[config(default = 0.2)]
    pub dropout: f64,
    #[config(default = 10_000)]
    pub token_buckets: usize,
    #[config(default = 50_000)]
    pub schema_buckets: usize,
    #[config(default = 128)]
    pub embed_dim: usize,
    #[config(default = 8)]
    pub max_slots: usize,
}

// =============================================================================
// Gather cache — fixed per catalog, computed once at startup
// =============================================================================

/// Per-entity gather index and field mask tensors, resident on device.
/// Built once from the catalog and reused across all batches.
pub struct GatherCache<B: Backend> {
    /// [n_tables, max_fields] — global field indices per table row, padded with 0
    pub gather_idx: Tensor<B, 2, Int>,
    /// [n_tables, max_fields] — 1.0 for valid field slots, 0.0 for padding
    pub field_mask: Tensor<B, 2>,
    pub max_fields: usize,
}

impl<B: Backend> GatherCache<B> {
    pub fn new(catalog: &SchemaCatalog, device: &B::Device) -> Self {
        let n_tables  = catalog.table_field_indices.len();
        let max_fields = catalog.table_field_indices.iter()
            .map(|f| f.len()).max().unwrap_or(0).max(1);

        let mut idx_flat  = vec![0i32;   n_tables * max_fields];
        let mut mask_flat = vec![0.0f32; n_tables * max_fields];
        for (t, fields) in catalog.table_field_indices.iter().enumerate() {
            for (j, &gf) in fields.iter().enumerate() {
                idx_flat [t * max_fields + j] = gf as i32;
                mask_flat[t * max_fields + j] = 1.0;
            }
        }
        Self {
            gather_idx: Tensor::<B, 1, Int>::from_ints(idx_flat.as_slice(), device)
                .reshape([n_tables as i32, max_fields as i32]),
            field_mask: Tensor::<B, 1>::from_floats(mask_flat.as_slice(), device)
                .reshape([n_tables as i32, max_fields as i32]),
            max_fields,
        }
    }
}

// =============================================================================
// Cached schema embeddings — computed once per batch
// =============================================================================

pub struct SchemaEmbs<B: Backend> {
    pub table_embs: Tensor<B, 2>,
    pub field_embs: Tensor<B, 2>,
    pub op_embs: Tensor<B, 2>,
    pub cmp_embs: Tensor<B, 2>,
    pub mod_embs: Tensor<B, 2>,
}

// =============================================================================
// Logits
// =============================================================================

pub struct LamellaLogits<B: Backend> {
    pub intent: Tensor<B, 1>,
    pub entity: Tensor<B, 1>,
    pub projection: Vec<Tensor<B, 1>>,
    pub cond_field: Vec<Tensor<B, 1>>,
    pub cond_cmp: Vec<Tensor<B, 1>>,
    pub assignment: Vec<Tensor<B, 1>>,
    pub mod_type: Vec<Tensor<B, 1>>,
    pub mod_field: Vec<Tensor<B, 1>>,
}

// =============================================================================
// Model
// =============================================================================

#[derive(Module, Debug)]
pub struct Lamella<B: Backend> {
    // NL encoder
    token_table: Embedding<B>,
    input_proj: Linear<B>,
    transformer: TransformerEncoder<B>,

    // Schema node embeddings (flat)
    schema_table: Embedding<B>,

    // Fixed vocab embeddings
    op_emb: Embedding<B>,
    cmp_emb: Embedding<B>,
    mod_emb: Embedding<B>,

    // Slot embeddings: positional (which slot within a head) + type (which SurrealQL slot kind)
    slot_pos_emb: Embedding<B>,
    slot_type_emb: Embedding<B>,

    // 8 head projections
    head_intent: Linear<B>,
    head_entity: Linear<B>,
    head_proj: Linear<B>,
    head_cond_f: Linear<B>,
    head_cond_c: Linear<B>,
    head_asgn: Linear<B>,
    head_mod_t: Linear<B>,
    head_mod_f: Linear<B>,

    // bilinear scoring; proj+asgn use asymmetric left/right projections
    bi_intent: Linear<B>,
    bi_entity: Linear<B>,
    bi_proj_l: Linear<B>,  // left (query) projection for projection head
    bi_proj_r: Linear<B>,  // right (field) projection for projection head
    bi_cond_f: Linear<B>,
    bi_cond_c: Linear<B>,
    bi_asgn_l: Linear<B>,  // left (query) projection for assignment
    bi_asgn_r: Linear<B>,  // right (field) projection for assignment
    bi_mod_t: Linear<B>,
    bi_mod_f: Linear<B>,

    // Value-type embedding for assignment slots: 5 types × d_model
    asgn_val_type_emb: Embedding<B>,

    // Cross-attention Q projections — one per field head so gradients don't compete.
    xattn_table_q: Linear<B>,  // NL tokens → table cross-attention queries
    xattn_proj_q:  Linear<B>,  // proj-specific field cross-attention
    xattn_cond_q:  Linear<B>,  // cond_field-specific field cross-attention
    xattn_asgn_q:  Linear<B>,  // assignment-specific field cross-attention
    xattn_field_q: Linear<B>,  // mod_field-specific field cross-attention

    d_model: usize,
    embed_dim: usize,
}

impl LamellaConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Lamella<B> {
        let d = self.d_model;

        let transformer = TransformerEncoderConfig::new(d, self.d_ff, self.n_heads, self.n_layers)
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .init(device);

        let bilinear = |dev: &B::Device| -> Linear<B> {
            LinearConfig::new(d, d).with_bias(false).init(dev)
        };
        let head = |dev: &B::Device| -> Linear<B> {
            LinearConfig::new(d, d).init(dev)
        };

        Lamella {
            token_table: EmbeddingConfig::new(self.token_buckets, self.embed_dim).init(device),
            input_proj: LinearConfig::new(self.embed_dim, d).init(device),
            transformer,
            schema_table: EmbeddingConfig::new(self.schema_buckets, d).init(device),
            op_emb: EmbeddingConfig::new(4, d).init(device),
            cmp_emb: EmbeddingConfig::new(7, d).init(device),
            mod_emb: EmbeddingConfig::new(3, d).init(device),
            slot_pos_emb: EmbeddingConfig::new(self.max_slots, d).init(device),
            slot_type_emb: EmbeddingConfig::new(8, d).init(device),
            head_intent: head(device),
            head_entity: head(device),
            head_proj: head(device),
            head_cond_f: head(device),
            head_cond_c: head(device),
            head_asgn: head(device),
            head_mod_t: head(device),
            head_mod_f: head(device),
            bi_intent: bilinear(device),
            bi_entity: bilinear(device),
            bi_proj_l: bilinear(device),
            bi_proj_r: bilinear(device),
            bi_cond_f: bilinear(device),
            bi_cond_c: bilinear(device),
            bi_asgn_l: bilinear(device),
            bi_asgn_r: bilinear(device),
            bi_mod_t: bilinear(device),
            bi_mod_f: bilinear(device),
            asgn_val_type_emb: EmbeddingConfig::new(5, d).init(device),
            xattn_table_q: head(device),
            xattn_proj_q:  head(device),
            xattn_cond_q:  head(device),
            xattn_asgn_q:  head(device),
            xattn_field_q: head(device),
            d_model: d,
            embed_dim: self.embed_dim,
        }
    }
}

// =============================================================================
// Slot counts — tells forward() how many resolution slots each head needs
// =============================================================================

pub struct SlotCounts {
    pub projections: usize,
    pub conditions: usize,
    pub assignments: usize,
    pub asgn_val_types: Vec<usize>, // 0=placeholder 1=bool 2=numeric 3=string 4=temporal
    pub mod_types: usize,
    pub mod_fields: usize,
}

// =============================================================================
// Forward pass
// =============================================================================

impl<B: Backend> Lamella<B> {
    /// Encode NL tokens → mean-pooled representation [d_model].
    ///
    /// Batches all n-gram indices into a single padded tensor for one embedding
    /// lookup, then uses scatter-mean to pool per-token.
    pub fn encode_nl(
        &self,
        tokens: &[String],
        token_buckets: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        if tokens.is_empty() {
            return Tensor::zeros([self.d_model], device);
        }

        let ed = self.embed_dim;
        let seq_len = tokens.len();

        // Gather all n-gram bucket indices per token, pad to max_ngrams
        let per_token: Vec<Vec<usize>> = tokens.iter()
            .map(|tok| char_ngram_buckets(tok, token_buckets))
            .collect();
        let max_ng = per_token.iter().map(|b| b.len()).max().unwrap_or(1);

        // Build padded [seq_len, max_ngrams] index tensor + mask for true counts
        let mut flat = vec![0i32; seq_len * max_ng];
        let mut mask_flat = vec![0.0f32; seq_len * max_ng];
        for (i, buckets) in per_token.iter().enumerate() {
            for (j, &b) in buckets.iter().enumerate() {
                flat[i * max_ng + j] = b as i32;
                mask_flat[i * max_ng + j] = 1.0;
            }
        }
        let idx = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device)
            .reshape([seq_len as i32, max_ng as i32]); // [seq_len, max_ng]

        // Single embedding lookup: [seq_len, max_ng, embed_dim]
        let all_embs = self.token_table.forward(idx);

        // Masked mean: zero out pad positions, sum, divide by real count
        let mask = Tensor::<B, 1>::from_floats(mask_flat.as_slice(), device)
            .reshape([seq_len as i32, max_ng as i32, 1]); // [seq_len, max_ng, 1]
        let counts = mask.clone().sum_dim(1).clamp_min(1.0); // [seq_len, 1, 1]
        let seq = (all_embs * mask).sum_dim(1) / counts; // [seq_len, 1, embed_dim]
        let seq = seq.reshape([seq_len as i32, ed as i32]);

        let seq = self.input_proj.forward(seq); // [seq_len, d_model]
        let seq = seq.unsqueeze::<3>(); // [1, seq_len, d_model]
        let h = self.transformer.forward(TransformerEncoderInput::new(seq));
        // [1, seq_len, d_model] → [seq_len, d_model] → mean → [d_model]
        let d = self.d_model;
        h.reshape([seq_len as i32, d as i32]).mean_dim(0).reshape([d])
    }

    /// Encode a batch of tokenized NL inputs in one transformer pass.
    /// Returns (full_seq [bs, max_seq, d_model], pool [bs, d_model], seq_lens, seq_mask [bs, max_seq, 1]).
    /// seq_mask is 1.0 for real tokens, 0.0 for padding — reused by head_scoring_batch.
    pub fn encode_nl_batch(
        &self,
        batch_tokens: &[Vec<String>],
        token_buckets: usize,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Vec<usize>, Tensor<B, 3>) {
        let d = self.d_model;
        let ed = self.embed_dim;
        let bs = batch_tokens.len();
        if bs == 0 {
            return (
                Tensor::zeros([0, 1, d], device),
                Tensor::zeros([0, d], device),
                vec![],
                Tensor::zeros([0, 1, 1], device),
            );
        }

        // Per-token n-gram indices for each datum
        let batch_ngrams: Vec<Vec<Vec<usize>>> = batch_tokens.iter()
            .map(|toks| toks.iter().map(|t| char_ngram_buckets(t, token_buckets)).collect())
            .collect();

        let seq_lens: Vec<usize> = batch_ngrams.iter().map(|toks| toks.len().max(1)).collect();
        let max_seq = seq_lens.iter().copied().max().unwrap();
        let max_ng = batch_ngrams.iter()
            .flat_map(|toks| toks.iter().map(|b| b.len()))
            .max().unwrap_or(1);

        // Build padded [bs * max_seq, max_ng] index tensor + n-gram mask
        let total = bs * max_seq;
        let mut idx_flat = vec![0i32; total * max_ng];
        let mut ng_mask_flat = vec![0.0f32; total * max_ng];
        // Sequence mask: true for real tokens, false for padding
        let mut seq_mask_flat = vec![false; bs * max_seq];

        for (b, toks) in batch_ngrams.iter().enumerate() {
            for (s, ngrams) in toks.iter().enumerate() {
                seq_mask_flat[b * max_seq + s] = true;
                let row = (b * max_seq + s) * max_ng;
                for (j, &bucket) in ngrams.iter().enumerate() {
                    idx_flat[row + j] = bucket as i32;
                    ng_mask_flat[row + j] = 1.0;
                }
            }
        }

        // Embed n-grams: [bs*max_seq, max_ng, embed_dim]
        let idx = Tensor::<B, 1, Int>::from_ints(idx_flat.as_slice(), device)
            .reshape([total as i32, max_ng as i32]);
        let all_embs = self.token_table.forward(idx);

        // Masked mean over n-grams → [bs*max_seq, embed_dim]
        let ng_mask = Tensor::<B, 1>::from_floats(ng_mask_flat.as_slice(), device)
            .reshape([total as i32, max_ng as i32, 1]);
        let ng_counts = ng_mask.clone().sum_dim(1).clamp_min(1.0);
        let tok_embs = (all_embs * ng_mask).sum_dim(1) / ng_counts;
        let tok_embs = tok_embs.reshape([total as i32, ed as i32]);

        // Project to d_model: [bs*max_seq, d_model] → [bs, max_seq, d_model]
        let projected = self.input_proj.forward(tok_embs)
            .reshape([bs as i32, max_seq as i32, d as i32]);

        // Padding mask for transformer: [bs, max_seq]
        let pad_mask = Tensor::<B, 1, Bool>::from_bool(seq_mask_flat.as_slice().into(), device)
            .reshape([bs as i32, max_seq as i32]);

        // One transformer forward: [bs, max_seq, d_model]
        let input = TransformerEncoderInput::new(projected).mask_pad(pad_mask.clone());
        let h = self.transformer.forward(input); // [bs, max_seq, d_model]

        // Masked mean-pool over sequence dim → [bs, d_model]
        let seq_mask = pad_mask.float().reshape([bs as i32, max_seq as i32, 1]);
        let seq_counts = seq_mask.clone().sum_dim(1).clamp_min(1.0); // [bs, 1, 1]
        let pooled = (h.clone() * seq_mask.clone()).sum_dim(1) / seq_counts; // [bs, 1, d_model]
        let pool = pooled.reshape([bs as i32, d as i32]);

        (h, pool, seq_lens, seq_mask)
    }

    /// Compute schema node embeddings from precomputed n-gram indices.
    /// Same batched approach as encode_nl.
    pub fn embed_schema_nodes(
        &self,
        ngram_indices: &[Vec<usize>],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let d = self.d_model;
        let n_nodes = ngram_indices.len();
        if n_nodes == 0 {
            return Tensor::zeros([0, d], device);
        }

        let max_ng = ngram_indices.iter().map(|b| b.len()).max().unwrap_or(1);

        let mut flat = vec![0i32; n_nodes * max_ng];
        let mut mask_flat = vec![0.0f32; n_nodes * max_ng];
        for (i, buckets) in ngram_indices.iter().enumerate() {
            for (j, &b) in buckets.iter().enumerate() {
                flat[i * max_ng + j] = b as i32;
                mask_flat[i * max_ng + j] = 1.0;
            }
        }
        let idx = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device)
            .reshape([n_nodes as i32, max_ng as i32]);

        let all_embs = self.schema_table.forward(idx); // [n_nodes, max_ng, d_model]
        let mask = Tensor::<B, 1>::from_floats(mask_flat.as_slice(), device)
            .reshape([n_nodes as i32, max_ng as i32, 1]); // [n_nodes, max_ng, 1]
        let counts = mask.clone().sum_dim(1).clamp_min(1.0); // [n_nodes, 1, 1]
        let result = (all_embs * mask).sum_dim(1) / counts; // [n_nodes, 1, d]
        result.reshape([n_nodes as i32, d as i32])
    }

    /// Score a query vector against candidate embeddings via a bilinear head.
    fn score(
        &self,
        query: Tensor<B, 1>,
        bilinear: &Linear<B>,
        candidates: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let proj = bilinear.forward(query.unsqueeze::<2>()); // [1, d_model]
        candidates.matmul(proj.transpose()).squeeze::<1>() // [n_candidates]
    }

    /// Positional slot embedding for index i within a head.
    fn slot(&self, i: usize, device: &B::Device) -> Tensor<B, 1> {
        let idx = Tensor::<B, 1, Int>::from_ints([i as i32].as_slice(), device);
        self.slot_pos_emb.forward(idx.unsqueeze::<2>()).reshape([self.d_model])
    }

    /// SurrealQL slot-type embedding — encodes which part of the query grammar
    /// this head is responsible for (intent, entity, projection, condition, etc.).
    fn stype(&self, t: usize, device: &B::Device) -> Tensor<B, 1> {
        let idx = Tensor::<B, 1, Int>::from_ints([t as i32].as_slice(), device);
        self.slot_type_emb.forward(idx.unsqueeze::<2>()).reshape([self.d_model])
    }

    /// Precompute schema + fixed vocab embeddings (call once per batch).
    pub fn precompute_schema_embs(
        &self,
        catalog: &SchemaCatalog,
        device: &B::Device,
    ) -> SchemaEmbs<B> {
        let table_embs = self.embed_schema_nodes(&catalog.table_ngrams, device);
        let field_embs = self.embed_schema_nodes(&catalog.field_ngrams, device);

        let op_indices = Tensor::<B, 1, Int>::from_ints([0, 1, 2, 3].as_slice(), device);
        let op_embs = self.op_emb.forward(op_indices.unsqueeze::<2>()).squeeze::<2>();

        let cmp_indices = Tensor::<B, 1, Int>::from_ints([0, 1, 2, 3, 4, 5, 6].as_slice(), device);
        let cmp_embs = self.cmp_emb.forward(cmp_indices.unsqueeze::<2>()).squeeze::<2>();

        let mod_indices = Tensor::<B, 1, Int>::from_ints([0, 1, 2].as_slice(), device);
        let mod_embs = self.mod_emb.forward(mod_indices.unsqueeze::<2>()).squeeze::<2>();

        SchemaEmbs { table_embs, field_embs, op_embs, cmp_embs, mod_embs }
    }

    /// Batched level-1: table cross-attention + entity head across all datums at once.
    /// Replaces bs × [1, d] ops with one [bs, d] pass — avoids repeated small matmuls.
    /// Returns (pool1s [bs, d], entity_logits [bs, n_tables]).
    pub fn level1_batch(
        &self,
        pools: Tensor<B, 2>,    // [bs, d_model]
        embs: &SchemaEmbs<B>,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let d = self.d_model;
        let scale = 1.0f32 / (d as f32).sqrt();

        // Table cross-attention: [bs, d] → pool_1s [bs, d]
        let q1 = self.xattn_table_q.forward(pools.clone());          // [bs, d]
        let scores1 = q1.matmul(embs.table_embs.clone().transpose()) * scale; // [bs, n_tables]
        let attn1 = activation::softmax(scores1, 1);
        let ctx_1 = attn1.matmul(embs.table_embs.clone());           // [bs, d]
        let pool1s = pools + ctx_1;                                   // [bs, d]

        // Entity head: pool_1 + T_ENTITY type signal → logits over tables
        // Broadcast type emb [1, d] over batch dim.
        let entity_type = self.stype(1 /*T_ENTITY*/, device).unsqueeze::<2>(); // [1, d]
        let entity_q = self.head_entity.forward(pool1s.clone() + entity_type); // [bs, d]
        let entity_proj = self.bi_entity.forward(entity_q);          // [bs, d]
        let entity_logits = entity_proj.matmul(embs.table_embs.clone().transpose()); // [bs, n_tables]

        (pool1s, entity_logits)
    }

    /// Full forward pass: NL → logits for all heads.
    pub fn forward(
        &self,
        tokens: &[String],
        token_buckets: usize,
        slots: &SlotCounts,
        catalog: &SchemaCatalog,
        entity_table_idx: Option<usize>,
        device: &B::Device,
    ) -> LamellaLogits<B> {
        let embs = self.precompute_schema_embs(catalog, device);
        self.forward_with_embs(tokens, token_buckets, slots, catalog, entity_table_idx, &embs, device)
    }

    /// Forward pass using precomputed schema embeddings.
    pub fn forward_with_embs(
        &self,
        tokens: &[String],
        token_buckets: usize,
        slots: &SlotCounts,
        catalog: &SchemaCatalog,
        entity_table_idx: Option<usize>,
        embs: &SchemaEmbs<B>,
        device: &B::Device,
    ) -> LamellaLogits<B> {
        let d = self.d_model;
        let (full_seq, pools, seq_lens, _seq_mask) = self.encode_nl_batch(
            &[tokens.to_vec()], token_buckets, device,
        );
        let (pool1s, entity_logits) = self.level1_batch(pools, embs, device);
        let pool_1 = pool1s.reshape([d]);
        let entity = entity_logits.reshape([embs.table_embs.dims()[0]]);
        let sl = seq_lens[0];
        let nl_seq = full_seq.slice([0..1, 0..sl, 0..d]).reshape([sl, d]);
        self.head_scoring(pool_1, entity, nl_seq, slots, catalog, entity_table_idx, embs, device)
    }

    /// Head scoring via two-level hierarchical schema cross-attention.
    ///
    /// Level 1 — Q-projected pool attends over all table embeddings.
    ///   pool_1 → entity head (no oracle, no circular dependency).
    ///
    /// Level 2 — Q-projected pool_1 attends over the resolved entity's fields.
    ///   entity_table_idx: Some(idx) = teacher forcing (training),
    ///                     None      = argmax of entity logits (inference).
    ///   pool_2 → intent + all field heads.
    ///
    /// Cross-attention is pool-level ([1, n_schema] attention weights) rather
    /// than sequence-level to stay compatible with CubeCL 0.9.0 kernel codegen.
    /// Per-slot attention over nl_seq resolves individual field slots.
    pub fn head_scoring(
        &self,
        pool_1: Tensor<B, 1>,            // [d_model] — level-1 enriched pool (from level1_batch)
        entity: Tensor<B, 1>,            // [n_tables] — entity logits (from level1_batch)
        nl_seq: Tensor<B, 2>,            // [seq_len, d_model] — full transformer output
        slots: &SlotCounts,
        catalog: &SchemaCatalog,
        entity_table_idx: Option<usize>, // Some = teacher forcing, None = inference
        embs: &SchemaEmbs<B>,
        device: &B::Device,
    ) -> LamellaLogits<B> {
        let d = self.d_model;
        let scale = 1.0f32 / (d as f32).sqrt();

        // Slot type indices — which part of SurrealQL grammar each head handles.
        const T_INTENT:     usize = 0;
        const T_ENTITY:     usize = 1;
        const T_PROJ:       usize = 2;
        const T_COND_FIELD: usize = 3;
        const T_COND_CMP:   usize = 4;
        const T_ASGN:       usize = 5;
        const T_MOD_TYPE:   usize = 6;
        const T_MOD_FIELD:  usize = 7;

        // Precompute all 8 type embeddings once — reused across slot positions.
        let st: [Tensor<B, 1>; 8] = [
            self.stype(T_INTENT, device),
            self.stype(T_ENTITY, device),
            self.stype(T_PROJ, device),
            self.stype(T_COND_FIELD, device),
            self.stype(T_COND_CMP, device),
            self.stype(T_ASGN, device),
            self.stype(T_MOD_TYPE, device),
            self.stype(T_MOD_FIELD, device),
        ];

        // pool_1 is already level-1 enriched (table cross-attn + entity type signal
        // applied outside in a batched pass — see level1_batch). Entity logits also
        // arrive pre-computed from that batch pass.
        //
        // ── Level 2: pool_1 attends over resolved entity's field embeddings ──
        let resolved_entity_idx: usize = match entity_table_idx {
            Some(idx) => idx,
            None => entity.clone().argmax(0).into_scalar().elem::<i32>() as usize,
        };

        let valid_field_indices = &catalog.table_field_indices[resolved_entity_idx];
        let masked_field_embs = self.gather_rows(&embs.field_embs, valid_field_indices, device);

        // Per-head field cross-attention pools — each head gets its own Q projection
        // so gradients don't compete. Structural heads (intent, cond_cmp, mod_type)
        // use pool_1 directly; field heads attend their own copy of the field set.
        let field_pool = |q_proj: &Linear<B>| -> Tensor<B, 1> {
            if valid_field_indices.is_empty() { return pool_1.clone(); }
            let q = q_proj.forward(pool_1.clone().unsqueeze::<2>()).squeeze::<1>();
            let scores = q.unsqueeze::<2>().matmul(masked_field_embs.clone().transpose());
            let attn = activation::softmax(scores, 1);
            let ctx = attn.matmul(masked_field_embs.clone()).reshape([d]);
            pool_1.clone() + ctx
        };

        let proj_pool = field_pool(&self.xattn_proj_q);
        let cond_pool = field_pool(&self.xattn_cond_q);
        let asgn_pool = field_pool(&self.xattn_asgn_q);
        let modf_pool = field_pool(&self.xattn_field_q);

        // Intent — pool_1 directly; intent is a structural choice not tied to any
        // particular field, so field cross-attention adds noise rather than signal.
        let intent_q = self.head_intent.forward(
            (pool_1.clone() + st[T_INTENT].clone()).unsqueeze::<2>()
        ).squeeze::<1>();
        let intent = self.score(intent_q, &self.bi_intent, embs.op_embs.clone());

        let seq_len = nl_seq.dims()[0];

        // -- Projection: uses proj_pool (decoupled from shared pool_2) --
        let mut proj_cov: Tensor<B, 2> = Tensor::zeros([seq_len, 1], device);
        let mut projection = Vec::with_capacity(slots.projections);
        for i in 0..slots.projections {
            let q = self.head_proj.forward(
                (proj_pool.clone() + st[T_PROJ].clone() + self.slot(i, device)).unsqueeze::<2>()
            ).squeeze::<1>();
            let raw = nl_seq.clone().matmul(q.unsqueeze::<2>().transpose()) * scale; // [seq_len, 1]
            let attn_w = activation::softmax(raw - proj_cov.clone(), 0);
            proj_cov = proj_cov + attn_w.clone();
            let ctx = nl_seq.clone().transpose().matmul(attn_w).reshape([d]);
            let ctx_l = self.bi_proj_l.forward(ctx.unsqueeze::<2>()).squeeze::<1>(); // [d]
            let fields_r = self.bi_proj_r.forward(masked_field_embs.clone());        // [n_fields, d]
            projection.push(fields_r.matmul(ctx_l.unsqueeze::<2>()).squeeze::<1>()); // [n_fields]
        }

        // -- Condition field: NL attention with coverage --
        let mut cond_cov: Tensor<B, 2> = Tensor::zeros([seq_len, 1], device);
        let mut cond_field = Vec::with_capacity(slots.conditions);
        for i in 0..slots.conditions {
            let q = self.head_cond_f.forward(
                (cond_pool.clone() + st[T_COND_FIELD].clone() + self.slot(i, device)).unsqueeze::<2>()
            ).squeeze::<1>();
            let raw = nl_seq.clone().matmul(q.unsqueeze::<2>().transpose()) * scale;
            let attn_w = activation::softmax(raw - cond_cov.clone(), 0);
            cond_cov = cond_cov + attn_w.clone();
            let ctx = nl_seq.clone().transpose().matmul(attn_w).reshape([d]);
            cond_field.push(self.score(ctx, &self.bi_cond_f, masked_field_embs.clone()));
        }

        // Condition comparator — pool only; comparators are structural SurrealQL choices.
        let cond_cmp: Vec<Tensor<B, 1>> = (0..slots.conditions).map(|i| {
            let q = self.head_cond_c.forward(
                (pool_1.clone() + st[T_COND_CMP].clone() + self.slot(i, device)).unsqueeze::<2>()
            ).squeeze::<1>();
            self.score(q, &self.bi_cond_c, embs.cmp_embs.clone())
        }).collect();

        // -- Assignment field: NL attention with coverage + asymmetric bilinear --
        let mut asgn_cov: Tensor<B, 2> = Tensor::zeros([seq_len, 1], device);
        let mut assignment = Vec::with_capacity(slots.assignments);
        for i in 0..slots.assignments {
            let vtype_idx = slots.asgn_val_types.get(i).copied().unwrap_or(0);
            let vtype_emb = self.asgn_val_type_emb
                .forward(Tensor::<B, 1, Int>::from_ints([vtype_idx as i32].as_slice(), device).unsqueeze::<2>())
                .reshape([d]);
            let q = self.head_asgn.forward(
                (asgn_pool.clone() + st[T_ASGN].clone() + self.slot(i, device) + vtype_emb).unsqueeze::<2>()
            ).squeeze::<1>();
            let raw = nl_seq.clone().matmul(q.unsqueeze::<2>().transpose()) * scale;
            let attn_w = activation::softmax(raw - asgn_cov.clone(), 0);
            asgn_cov = asgn_cov + attn_w.clone();
            let ctx = nl_seq.clone().transpose().matmul(attn_w).reshape([d]);
            // Asymmetric bilinear: left projects query, right projects field candidates
            let ctx_l = self.bi_asgn_l.forward(ctx.unsqueeze::<2>()).squeeze::<1>(); // [d]
            let fields_r = self.bi_asgn_r.forward(masked_field_embs.clone());        // [n_fields, d]
            assignment.push(fields_r.matmul(ctx_l.unsqueeze::<2>()).squeeze::<1>()); // [n_fields]
        }

        // Modifier type — pool only; ORDER BY / LIMIT / FETCH are structural choices.
        let mod_type: Vec<Tensor<B, 1>> = (0..slots.mod_types).map(|i| {
            let q = self.head_mod_t.forward(
                (pool_1.clone() + st[T_MOD_TYPE].clone() + self.slot(i, device)).unsqueeze::<2>()
            ).squeeze::<1>();
            self.score(q, &self.bi_mod_t, embs.mod_embs.clone())
        }).collect();

        // Modifier field — usually 1 slot, no coverage needed
        let mod_field: Vec<Tensor<B, 1>> = (0..slots.mod_fields).map(|i| {
            let q = self.head_mod_f.forward(
                (modf_pool.clone() + st[T_MOD_FIELD].clone() + self.slot(i, device)).unsqueeze::<2>()
            ).squeeze::<1>();
            let raw = nl_seq.clone().matmul(q.unsqueeze::<2>().transpose()) * scale;
            let attn_w = activation::softmax(raw, 0);
            let ctx = nl_seq.clone().transpose().matmul(attn_w).reshape([d]);
            self.score(ctx, &self.bi_mod_f, masked_field_embs.clone())
        }).collect();

        LamellaLogits { intent, entity, projection, cond_field, cond_cmp, assignment, mod_type, mod_field }
    }

    /// Batched head scoring: processes all datums in a batch simultaneously.
    /// Eliminates bs × n_slots sequential GPU dispatches — all slot heads
    /// run as single [bs, ...] ops instead of many [1, ...] ops.
    pub fn head_scoring_batch(
        &self,
        pool1s: Tensor<B, 2>,              // [bs, d_model]
        entity_logits: Tensor<B, 2>,       // [bs, n_tables]
        full_seqs: Tensor<B, 3>,           // [bs, max_seq, d_model]
        seq_mask: Tensor<B, 3>,            // [bs, max_seq, 1] — 1.0 real, 0.0 pad
        all_slots: &[SlotCounts],
        entity_indices: &[Option<usize>],  // Some = teacher forcing, None = argmax
        cache: &GatherCache<B>,
        embs: &SchemaEmbs<B>,
        device: &B::Device,
    ) -> Vec<LamellaLogits<B>> {
        let d = self.d_model;
        let scale = 1.0f32 / (d as f32).sqrt();
        let bs = all_slots.len();
        let max_seq = full_seqs.dims()[1];
        let n_tables = embs.table_embs.dims()[0];
        let n_ops    = embs.op_embs.dims()[0];
        let n_cmps   = embs.cmp_embs.dims()[0];
        let n_mods   = embs.mod_embs.dims()[0];
        let max_fields = cache.max_fields;

        const T_INTENT:     usize = 0;
        const T_PROJ:       usize = 2;
        const T_COND_FIELD: usize = 3;
        const T_COND_CMP:   usize = 4;
        const T_ASGN:       usize = 5;
        const T_MOD_TYPE:   usize = 6;
        const T_MOD_FIELD:  usize = 7;

        // ── Resolve entity indices ──────────────────────────────────────
        let entity_idxs: Vec<usize> = entity_indices.iter().enumerate().map(|(b, opt)| {
            match opt {
                Some(idx) => *idx,
                None => entity_logits.clone()
                    .slice([b..b+1, 0..n_tables]).reshape([n_tables])
                    .argmax(0).into_scalar().elem::<i32>() as usize,
            }
        }).collect();

        // ── Padded field embeddings [bs, max_fields, d] ─────────────────
        // Gather entity rows from the pre-built cache — no CPU loop, no host transfer
        // beyond the bs-element entity index array.
        let batch_entity_idx = Tensor::<B, 1, Int>::from_ints(
            entity_idxs.iter().map(|&i| i as i32).collect::<Vec<_>>().as_slice(), device,
        );
        let flat_gather = cache.gather_idx.clone()
            .select(0, batch_entity_idx.clone())
            .reshape([(bs * max_fields) as i32]);
        let field_mask_2d = cache.field_mask.clone()
            .select(0, batch_entity_idx);              // [bs, max_fields]
        let padded_field_embs = embs.field_embs.clone()
            .select(0, flat_gather)
            .reshape([bs as i32, max_fields as i32, d as i32]); // [bs, max_fields, d]

        // field_bias [bs, max_fields]: -1e9 for padding, 0 for valid
        let field_bias = (field_mask_2d - 1.0f32) * 1e9f32;

        // seq_bias [bs, max_seq, 1]: -1e9 for padding tokens
        let seq_bias = (seq_mask - 1.0f32) * 1e9f32;

        // ── Per-head field cross-attention pools [bs, d] ────────────────────
        // Each head gets its own Q projection so gradients don't compete.
        // Structural heads (intent, cond_cmp, mod_type) use pool1s directly.
        let batch_field_pool = |q_proj: &Linear<B>| -> Tensor<B, 2> {
            if max_fields == 0 { return pool1s.clone(); }
            let q = q_proj.forward(pool1s.clone());
            let scores = q.reshape([bs as i32, 1, d as i32])
                .matmul(padded_field_embs.clone().transpose())
                .reshape([bs as i32, max_fields as i32])
                + field_bias.clone();
            let attn = activation::softmax(scores, 1);
            let ctx = attn.reshape([bs as i32, 1, max_fields as i32])
                .matmul(padded_field_embs.clone())
                .reshape([bs as i32, d as i32]);
            pool1s.clone() + ctx
        };

        let proj_pool = batch_field_pool(&self.xattn_proj_q);
        let cond_pool = batch_field_pool(&self.xattn_cond_q);
        let asgn_pool = batch_field_pool(&self.xattn_asgn_q);
        let modf_pool = batch_field_pool(&self.xattn_field_q);

        // ── Intent (pool1s — structural, not field-specific) ───────────────
        let st_intent = self.stype(T_INTENT, device).unsqueeze::<2>(); // [1, d]
        let intent_logits_b = self.bi_intent
            .forward(self.head_intent.forward(pool1s.clone() + st_intent))
            .matmul(embs.op_embs.clone().transpose()); // [bs, n_ops]

        // ── Precompute type embs (broadcast [1, d]) ─────────────────────
        let st_proj   = self.stype(T_PROJ,       device).unsqueeze::<2>();
        let st_cf     = self.stype(T_COND_FIELD,  device).unsqueeze::<2>();
        let st_cc     = self.stype(T_COND_CMP,    device).unsqueeze::<2>();
        let st_asgn   = self.stype(T_ASGN,        device).unsqueeze::<2>();
        let st_mod_t  = self.stype(T_MOD_TYPE,    device).unsqueeze::<2>();
        let st_mod_f  = self.stype(T_MOD_FIELD,   device).unsqueeze::<2>();

        let max_proj  = all_slots.iter().map(|s| s.projections).max().unwrap_or(0);
        let max_cond  = all_slots.iter().map(|s| s.conditions).max().unwrap_or(0);
        let max_asgn  = all_slots.iter().map(|s| s.assignments).max().unwrap_or(0);
        let max_mod_t = all_slots.iter().map(|s| s.mod_types).max().unwrap_or(0);
        let max_mod_f = all_slots.iter().map(|s| s.mod_fields).max().unwrap_or(0);

        // ── Helper: NL attention → ctx [bs, d] ─────────────────────────
        // raw_scores [bs, max_seq, 1] = full_seqs @ q [bs, d, 1] * scale
        // attn = softmax(raw + seq_bias - cov, dim=1)
        // ctx  = full_seqs.T [bs, d, max_seq] @ attn [bs, max_seq, 1] → [bs, d]
        let nl_attn = |q: Tensor<B, 2>, seq_bias: Tensor<B, 3>, cov: Tensor<B, 3>| {
            let raw = full_seqs.clone()
                .matmul(q.reshape([bs as i32, d as i32, 1])) * scale; // [bs, max_seq, 1]
            let attn_w = activation::softmax(raw + seq_bias - cov, 1);
            let ctx = full_seqs.clone().transpose()
                .matmul(attn_w.clone())
                .reshape([bs as i32, d as i32]); // [bs, d]
            (ctx, attn_w)
        };

        // ── Field scoring: bi(ctx) [bs, d] → [bs, max_fields] ──────────
        let field_score = |bi: &Linear<B>, ctx: Tensor<B, 2>| -> Tensor<B, 2> {
            bi.forward(ctx)
                .reshape([bs as i32, 1, d as i32])
                .matmul(padded_field_embs.clone().transpose()) // [bs, 1, max_fields]
                .reshape([bs as i32, max_fields as i32])
                + field_bias.clone()
        };

        // ── Projection (uses proj_pool, not pool_2) ─────────────────────
        let mut proj_cov: Tensor<B, 3> = Tensor::zeros([bs, max_seq, 1], device);
        let mut proj_logits: Vec<Tensor<B, 2>> = Vec::with_capacity(max_proj);
        for i in 0..max_proj {
            let slot_i = self.slot(i, device).unsqueeze::<2>();
            let q = self.head_proj.forward(proj_pool.clone() + st_proj.clone() + slot_i);
            let (ctx, attn_w) = nl_attn(q, seq_bias.clone(), proj_cov.clone());
            proj_cov = proj_cov + attn_w;
            // Asymmetric bilinear: separate projections for query and field candidates
            let ctx_l = self.bi_proj_l.forward(ctx); // [bs, d]
            let fields_r = self.bi_proj_r.forward(padded_field_embs.clone()); // [bs, max_fields, d]
            let scores = ctx_l
                .reshape([bs as i32, 1, d as i32])
                .matmul(fields_r.transpose()) // [bs, 1, max_fields]
                .reshape([bs as i32, max_fields as i32])
                + field_bias.clone();
            proj_logits.push(scores);
        }

        // ── Condition field + comparator ────────────────────────────────
        let mut cond_cov: Tensor<B, 3> = Tensor::zeros([bs, max_seq, 1], device);
        let mut cond_field_logits: Vec<Tensor<B, 2>> = Vec::with_capacity(max_cond);
        let mut cond_cmp_logits:   Vec<Tensor<B, 2>> = Vec::with_capacity(max_cond);
        for i in 0..max_cond {
            let slot_i = self.slot(i, device).unsqueeze::<2>();
            let qf = self.head_cond_f.forward(cond_pool.clone() + st_cf.clone() + slot_i.clone());
            let (ctx, attn_w) = nl_attn(qf, seq_bias.clone(), cond_cov.clone());
            cond_cov = cond_cov + attn_w;
            cond_field_logits.push(field_score(&self.bi_cond_f, ctx));
            let qc = self.head_cond_c.forward(pool1s.clone() + st_cc.clone() + slot_i);
            cond_cmp_logits.push(
                self.bi_cond_c.forward(qc).matmul(embs.cmp_embs.clone().transpose())
            );
        }

        // ── Assignment ─────────────────────────────────────────────────
        let mut asgn_cov: Tensor<B, 3> = Tensor::zeros([bs, max_seq, 1], device);
        let mut asgn_logits: Vec<Tensor<B, 2>> = Vec::with_capacity(max_asgn);
        for i in 0..max_asgn {
            let slot_i = self.slot(i, device).unsqueeze::<2>(); // [1, d]
            // Value-type embedding: one per datum in batch → [bs, d]
            let vtype_indices: Vec<i32> = all_slots.iter()
                .map(|s| s.asgn_val_types.get(i).copied().unwrap_or(0) as i32)
                .collect();
            let vtype_embs = self.asgn_val_type_emb
                .forward(Tensor::<B, 1, Int>::from_ints(vtype_indices.as_slice(), device).unsqueeze::<2>())
                .reshape([bs as i32, d as i32]); // [bs, d]
            let q = self.head_asgn.forward(asgn_pool.clone() + st_asgn.clone() + slot_i + vtype_embs);
            let (ctx, attn_w) = nl_attn(q, seq_bias.clone(), asgn_cov.clone());
            asgn_cov = asgn_cov + attn_w;
            // Asymmetric bilinear: left projects query ctx, right projects field candidates
            let ctx_l = self.bi_asgn_l.forward(ctx); // [bs, d]
            let fields_r = self.bi_asgn_r.forward(padded_field_embs.clone()); // [bs, max_fields, d]
            let scores = ctx_l
                .reshape([bs as i32, 1, d as i32])
                .matmul(fields_r.transpose()) // [bs, 1, max_fields]
                .reshape([bs as i32, max_fields as i32])
                + field_bias.clone();
            asgn_logits.push(scores);
        }

        // ── Modifier type (pool only) ───────────────────────────────────
        let mut mod_type_logits: Vec<Tensor<B, 2>> = Vec::with_capacity(max_mod_t);
        for i in 0..max_mod_t {
            let slot_i = self.slot(i, device).unsqueeze::<2>();
            let q = self.head_mod_t.forward(pool1s.clone() + st_mod_t.clone() + slot_i);
            mod_type_logits.push(
                self.bi_mod_t.forward(q).matmul(embs.mod_embs.clone().transpose())
            );
        }

        // ── Modifier field (NL attention, no coverage) ──────────────────
        let mut mod_field_logits: Vec<Tensor<B, 2>> = Vec::with_capacity(max_mod_f);
        for i in 0..max_mod_f {
            let slot_i = self.slot(i, device).unsqueeze::<2>();
            let q = self.head_mod_f.forward(modf_pool.clone() + st_mod_f.clone() + slot_i);
            let (ctx, _) = nl_attn(q, seq_bias.clone(), Tensor::zeros([bs, max_seq, 1], device));
            mod_field_logits.push(field_score(&self.bi_mod_f, ctx));
        }

        // ── Split to Vec<LamellaLogits> ─────────────────────────────────
        (0..bs).map(|b| {
            let intent = intent_logits_b.clone().slice([b..b+1, 0..n_ops]).reshape([n_ops]);
            let entity = entity_logits.clone().slice([b..b+1, 0..n_tables]).reshape([n_tables]);
            let nf = max_fields;
            let projection = (0..all_slots[b].projections).map(|i|
                proj_logits[i].clone().slice([b..b+1, 0..nf]).reshape([nf])
            ).collect();
            let cond_field = (0..all_slots[b].conditions).map(|i|
                cond_field_logits[i].clone().slice([b..b+1, 0..nf]).reshape([nf])
            ).collect();
            let cond_cmp = (0..all_slots[b].conditions).map(|i|
                cond_cmp_logits[i].clone().slice([b..b+1, 0..n_cmps]).reshape([n_cmps])
            ).collect();
            let assignment = (0..all_slots[b].assignments).map(|i|
                asgn_logits[i].clone().slice([b..b+1, 0..nf]).reshape([nf])
            ).collect();
            let mod_type = (0..all_slots[b].mod_types).map(|i|
                mod_type_logits[i].clone().slice([b..b+1, 0..n_mods]).reshape([n_mods])
            ).collect();
            let mod_field = (0..all_slots[b].mod_fields).map(|i|
                mod_field_logits[i].clone().slice([b..b+1, 0..nf]).reshape([nf])
            ).collect();
            LamellaLogits { intent, entity, projection, cond_field, cond_cmp, assignment, mod_type, mod_field }
        }).collect()
    }

    /// Gather specific rows from a [N, d] tensor.
    fn gather_rows(
        &self,
        embs: &Tensor<B, 2>,
        indices: &[usize],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        if indices.is_empty() {
            return Tensor::zeros([0, self.d_model], device);
        }
        let idx = Tensor::<B, 1, Int>::from_ints(
            indices.iter().map(|&i| i as i32).collect::<Vec<_>>().as_slice(),
            device,
        );
        embs.clone().select(0, idx)
    }
}

// =============================================================================
// Resolve: logits → QueryIr
// =============================================================================

impl<B: Backend> Lamella<B> {
    pub fn resolve(
        &self,
        logits: &LamellaLogits<B>,
        catalog: &SchemaCatalog,
        values: &ResolveValues,
    ) -> QueryIr {
        let intent_idx: usize = logits.intent.clone().argmax(0).into_scalar().elem::<i32>() as usize;
        let intent = catalog.ops[intent_idx].clone();

        let entity_idx: usize = logits.entity.clone().argmax(0).into_scalar().elem::<i32>() as usize;
        let table = catalog.tables[entity_idx].clone();
        let valid_fields = &catalog.table_field_indices[entity_idx];

        let resolve_field = |logit: &Tensor<B, 1>| -> String {
            let local_idx: usize = logit.clone().argmax(0).into_scalar().elem::<i32>() as usize;
            if local_idx < valid_fields.len() {
                catalog.fields[valid_fields[local_idx]].1.clone()
            } else {
                "unknown".into()
            }
        };

        let projections: Vec<ResolvedField> = logits.projection.iter()
            .map(|l| ResolvedField { table: table.clone(), field: resolve_field(l) })
            .collect();

        let conditions: Vec<ResolvedCondition> = logits.cond_field.iter().enumerate()
            .map(|(i, l)| {
                let field = resolve_field(l);
                let cmp_idx: usize = logits.cond_cmp[i].clone().argmax(0).into_scalar().elem::<i32>() as usize;
                let comparator = catalog.cmps[cmp_idx].clone();
                let value = values.cond_values.get(i).cloned().unwrap_or(ValueRef::Literal("?".into()));
                ResolvedCondition { table: table.clone(), field, comparator, value }
            })
            .collect();

        let assignments: Vec<ResolvedAssignment> = logits.assignment.iter().enumerate()
            .map(|(i, l)| {
                let field = Some(resolve_field(l));
                let value = values.asgn_values.get(i).cloned().unwrap_or(ValueRef::Literal("?".into()));
                ResolvedAssignment { table: table.clone(), field, value }
            })
            .collect();

        let modifiers: Vec<ResolvedModifier> = logits.mod_type.iter().enumerate()
            .map(|(i, l)| {
                let type_idx: usize = l.clone().argmax(0).into_scalar().elem::<i32>() as usize;
                let kind = &catalog.mods[type_idx];
                match kind {
                    ModifierKind::OrderBy => {
                        let field = logits.mod_field.get(i)
                            .map(|f| resolve_field(f))
                            .unwrap_or("id".into());
                        let descending = values.mod_descending.get(i).copied().unwrap_or(false);
                        ResolvedModifier::OrderBy { table: table.clone(), field, descending }
                    }
                    ModifierKind::Limit => {
                        let value = values.mod_values.get(i).cloned().unwrap_or(ValueRef::Literal("10".into()));
                        ResolvedModifier::Limit { value }
                    }
                    ModifierKind::Fetch => {
                        let field = logits.mod_field.get(i)
                            .map(|f| resolve_field(f))
                            .unwrap_or("id".into());
                        ResolvedModifier::Fetch { field }
                    }
                }
            })
            .collect();

        QueryIr {
            intent,
            table,
            record_id: values.record_id.clone(),
            projections,
            conditions,
            assignments,
            modifiers,
        }
    }
}

/// Carry-through values needed by resolve() that the model doesn't predict.
pub struct ResolveValues {
    pub record_id: Option<ValueRef>,
    pub cond_values: Vec<ValueRef>,
    pub asgn_values: Vec<ValueRef>,
    pub mod_values: Vec<ValueRef>,
    pub mod_descending: Vec<bool>,
}
