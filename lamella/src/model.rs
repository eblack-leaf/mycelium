use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    Embedding, EmbeddingConfig,
    Linear, LinearConfig,
    transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
};
use burn::tensor::{Bool, ElementConversion, Int, Tensor, backend::Backend};

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
    #[config(default = 0.1)]
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

    // Slot positional embedding
    slot_emb: Embedding<B>,

    // 8 head projections
    head_intent: Linear<B>,
    head_entity: Linear<B>,
    head_proj: Linear<B>,
    head_cond_f: Linear<B>,
    head_cond_c: Linear<B>,
    head_asgn: Linear<B>,
    head_mod_t: Linear<B>,
    head_mod_f: Linear<B>,

    // 8 bilinear scoring matrices (no bias)
    bi_intent: Linear<B>,
    bi_entity: Linear<B>,
    bi_proj: Linear<B>,
    bi_cond_f: Linear<B>,
    bi_cond_c: Linear<B>,
    bi_asgn: Linear<B>,
    bi_mod_t: Linear<B>,
    bi_mod_f: Linear<B>,

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
            slot_emb: EmbeddingConfig::new(self.max_slots, d).init(device),
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
            bi_proj: bilinear(device),
            bi_cond_f: bilinear(device),
            bi_cond_c: bilinear(device),
            bi_asgn: bilinear(device),
            bi_mod_t: bilinear(device),
            bi_mod_f: bilinear(device),
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
    /// Returns [batch, d_model] — one mean-pooled vector per datum.
    pub fn encode_nl_batch(
        &self,
        batch_tokens: &[Vec<String>],
        token_buckets: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let d = self.d_model;
        let ed = self.embed_dim;
        let bs = batch_tokens.len();
        if bs == 0 {
            return Tensor::zeros([0, d], device);
        }

        // Per-token n-gram indices for each datum
        let batch_ngrams: Vec<Vec<Vec<usize>>> = batch_tokens.iter()
            .map(|toks| toks.iter().map(|t| char_ngram_buckets(t, token_buckets)).collect())
            .collect();

        let max_seq = batch_ngrams.iter().map(|toks| toks.len().max(1)).max().unwrap();
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
        // Expand pad_mask to [bs, max_seq, 1] for broadcasting
        let seq_mask = pad_mask.float().reshape([bs as i32, max_seq as i32, 1]);
        let seq_counts = seq_mask.clone().sum_dim(1).clamp_min(1.0); // [bs, 1, 1]
        let pooled = (h * seq_mask).sum_dim(1) / seq_counts; // [bs, 1, d_model]
        pooled.reshape([bs as i32, d as i32])
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

    /// Slot embedding for index i.
    fn slot(&self, i: usize, device: &B::Device) -> Tensor<B, 1> {
        let idx = Tensor::<B, 1, Int>::from_ints([i as i32].as_slice(), device);
        // [1, 1, d_model] → [d_model]
        self.slot_emb.forward(idx.unsqueeze::<2>()).reshape([self.d_model])
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

    /// Full forward pass: NL → logits for all heads.
    pub fn forward(
        &self,
        tokens: &[String],
        token_buckets: usize,
        slots: &SlotCounts,
        catalog: &SchemaCatalog,
        entity_table_idx: usize,
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
        entity_table_idx: usize,
        embs: &SchemaEmbs<B>,
        device: &B::Device,
    ) -> LamellaLogits<B> {
        let pool = self.encode_nl(tokens, token_buckets, device);
        self.head_scoring(pool, slots, catalog, entity_table_idx, embs, device)
    }

    /// Head scoring from a precomputed pool vector [d_model].
    pub fn head_scoring(
        &self,
        pool: Tensor<B, 1>,
        slots: &SlotCounts,
        catalog: &SchemaCatalog,
        entity_table_idx: usize,
        embs: &SchemaEmbs<B>,
        device: &B::Device,
    ) -> LamellaLogits<B> {

        // Field candidate mask: only fields of the resolved entity table
        let valid_field_indices = &catalog.table_field_indices[entity_table_idx];
        let masked_field_embs = self.gather_rows(&embs.field_embs, valid_field_indices, device);

        // Intent
        let intent_q = self.head_intent.forward(pool.clone().unsqueeze::<2>()).squeeze::<1>();
        let intent = self.score(intent_q, &self.bi_intent, embs.op_embs.clone());

        // Entity
        let entity_q = self.head_entity.forward(pool.clone().unsqueeze::<2>()).squeeze::<1>();
        let entity = self.score(entity_q, &self.bi_entity, embs.table_embs.clone());

        // Projection slots
        let projection: Vec<Tensor<B, 1>> = (0..slots.projections).map(|i| {
            let q = self.head_proj.forward((pool.clone() + self.slot(i, device)).unsqueeze::<2>()).squeeze::<1>();
            self.score(q, &self.bi_proj, masked_field_embs.clone())
        }).collect();

        // Condition field slots
        let cond_field: Vec<Tensor<B, 1>> = (0..slots.conditions).map(|i| {
            let q = self.head_cond_f.forward((pool.clone() + self.slot(i, device)).unsqueeze::<2>()).squeeze::<1>();
            self.score(q, &self.bi_cond_f, masked_field_embs.clone())
        }).collect();

        // Condition comparator slots
        let cond_cmp: Vec<Tensor<B, 1>> = (0..slots.conditions).map(|i| {
            let q = self.head_cond_c.forward((pool.clone() + self.slot(i, device)).unsqueeze::<2>()).squeeze::<1>();
            self.score(q, &self.bi_cond_c, embs.cmp_embs.clone())
        }).collect();

        // Assignment field slots
        let assignment: Vec<Tensor<B, 1>> = (0..slots.assignments).map(|i| {
            let q = self.head_asgn.forward((pool.clone() + self.slot(i, device)).unsqueeze::<2>()).squeeze::<1>();
            self.score(q, &self.bi_asgn, masked_field_embs.clone())
        }).collect();

        // Modifier type slots
        let mod_type: Vec<Tensor<B, 1>> = (0..slots.mod_types).map(|i| {
            let q = self.head_mod_t.forward((pool.clone() + self.slot(i, device)).unsqueeze::<2>()).squeeze::<1>();
            self.score(q, &self.bi_mod_t, embs.mod_embs.clone())
        }).collect();

        // Modifier field slots
        let mod_field: Vec<Tensor<B, 1>> = (0..slots.mod_fields).map(|i| {
            let q = self.head_mod_f.forward((pool.clone() + self.slot(i, device)).unsqueeze::<2>()).squeeze::<1>();
            self.score(q, &self.bi_mod_f, masked_field_embs.clone())
        }).collect();

        LamellaLogits { intent, entity, projection, cond_field, cond_cmp, assignment, mod_type, mod_field }
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
