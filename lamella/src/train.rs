use burn::{
    config::Config,
    lr_scheduler::{cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig}, LrScheduler},
    module::{AutodiffModule, Module},
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{ElementConversion, Tensor, activation, backend::AutodiffBackend, backend::Backend},
};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;

use crate::LamellaDatum;
use crate::catalog::SchemaCatalog;
use crate::embed::tokenize;
use crate::model::{GatherCache, Lamella, LamellaConfig, LamellaLogits};

// =============================================================================
// Metrics + HeadAcc
// =============================================================================

pub struct Metrics {
    pub train_loss: f32,
    pub val_loss: f32,
    pub val_acc: f32,
    pub head_acc: HeadAcc,
}

#[derive(Default)]
pub struct HeadAcc {
    pub intent:     (usize, usize),
    pub entity:     (usize, usize),
    pub proj:       (usize, usize),
    pub cond_field: (usize, usize),
    pub cond_cmp:   (usize, usize),
    pub assign:     (usize, usize),
    pub mod_type:   (usize, usize),
    pub mod_field:  (usize, usize),
}

impl HeadAcc {
    pub fn display(&self) -> String {
        let a = |pair: (usize, usize)| -> String {
            if pair.1 > 0 { format!("{:.0}%", pair.0 as f32 / pair.1 as f32 * 100.0) }
            else { "-".into() }
        };
        format!(
            "int={} ent={} proj={} cf={} cc={} asgn={} mt={} mf={}",
            a(self.intent), a(self.entity), a(self.proj),
            a(self.cond_field), a(self.cond_cmp), a(self.assign),
            a(self.mod_type), a(self.mod_field),
        )
    }
}

// =============================================================================
// Training config
// =============================================================================

#[derive(Config, Debug)]
pub struct TrainConfig {
    #[config(default = 50)]
    pub epochs: usize,
    #[config(default = 3e-3)]
    pub learning_rate: f64,
    #[config(default = 5)]
    pub patience: usize,
    #[config(default = 32)]
    pub batch_size: usize,
}

// =============================================================================
// Training context
// =============================================================================

type OptimizerAdaptor<B> = burn::optim::adaptor::OptimizerAdaptor<AdamW, Lamella<B>, B>;

pub struct LamellaTrainCtx<B: AutodiffBackend> {
    pub model: Lamella<B>,
    pub catalog: SchemaCatalog,
    pub config: LamellaConfig,
    pub optimizer: OptimizerAdaptor<B>,
    pub lr_scheduler: CosineAnnealingLrScheduler,
    pub device: B::Device,
    pub gather_cache: GatherCache<B>,
}

impl<B: AutodiffBackend> LamellaTrainCtx<B> {
    pub fn new(
        config: LamellaConfig,
        catalog: SchemaCatalog,
        lr: f64,
        num_iters: usize,
        device: &B::Device,
    ) -> Self {
        let model = config.init(device);
        let optimizer = AdamWConfig::new().init();
        let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(lr, num_iters)
            .with_min_lr(lr * 0.01)
            .init()
            .unwrap();
        let gather_cache = GatherCache::new(&catalog, device);
        Self { model, catalog, config, optimizer, lr_scheduler, device: device.clone(), gather_cache }
    }
}

// =============================================================================
// Loss helpers
// =============================================================================

fn cross_entropy<B: Backend>(logits: Tensor<B, 1>, target_idx: usize) -> Tensor<B, 1> {
    let log_softmax = activation::log_softmax(logits.unsqueeze::<2>(), 1).squeeze::<1>();
    log_softmax.slice([target_idx..target_idx + 1]).neg()
}

/// Returns a [1] float tensor: 1.0 if argmax matches target, 0.0 otherwise.
/// Stays on GPU — no sync until the caller calls into_scalar().
fn tensor_correct<B: Backend>(logit: &Tensor<B, 1>, target: usize) -> Tensor<B, 1> {
    let device = logit.device();
    let pred = logit.clone().argmax(0).float();                               // [1]
    let tgt  = Tensor::<B, 1>::from_floats([target as f32].as_slice(), &device); // [1]
    (1.0_f32 - (pred - tgt).abs()).clamp_min(0.0_f32)
}

// =============================================================================
// step_batch + evaluate
// =============================================================================

impl<B: AutodiffBackend> LamellaTrainCtx<B> {
    pub fn step_batch(&mut self, batch: &[&LamellaDatum]) -> f32 {
        let embs = self.model.precompute_schema_embs(&self.catalog, &self.device);

        // Batch-encode all NL in one transformer pass
        let batch_tokens: Vec<Vec<String>> = batch.iter()
            .map(|d| tokenize(&d.nl).0)
            .collect();
        let (full_seqs, pools, _seq_lens, seq_mask) = self.model.encode_nl_batch(
            &batch_tokens, self.config.token_buckets, &self.device,
        );

        // Level-1 in one batched pass: table cross-attn + entity head for all datums.
        let (pool1s, entity_logits) = self.model.level1_batch(pools, &embs, &self.device);

        // Batch all datums through slot heads simultaneously
        let all_slots: Vec<_> = batch.iter().map(|d| d.slot_counts()).collect();
        let entity_indices: Vec<Option<usize>> = batch.iter().map(|d| Some(d.entity)).collect();
        let all_logits = self.model.head_scoring_batch(
            pool1s, entity_logits, full_seqs, seq_mask,
            &all_slots, &entity_indices, &self.gather_cache, &embs, &self.device,
        );

        let mut losses: Vec<Tensor<B, 1>> = Vec::new();
        let mut n_total = 0usize;

        for (datum, logits) in batch.iter().zip(all_logits.iter()) {
            let (loss, n) = self.datum_loss(logits, datum);
            if n == 0 { continue; }
            n_total += n;
            losses.push(loss);
        }

        if n_total == 0 { return 0.0; }

        // Normalize by total slot predictions, then one backward pass
        let batch_loss = losses.into_iter().reduce(|a, b| a + b).unwrap() / n_total as f32;
        let grads = batch_loss.clone().backward();
        let grads = GradientsParams::from_grads(grads, &self.model);

        let lr = self.lr_scheduler.step();
        self.model = self.optimizer.step(lr, self.model.clone(), grads);

        // Single GPU→CPU sync per batch (was 32× per-datum syncs before backward)
        batch_loss.inner().into_scalar().elem::<f32>()
    }

    fn datum_loss(&self, logits: &LamellaLogits<B>, datum: &LamellaDatum) -> (Tensor<B, 1>, usize) {
        let mut losses: Vec<Tensor<B, 1>> = Vec::new();
        let valid_fields = &self.catalog.table_field_indices[datum.entity];

        // Intent
        losses.push(cross_entropy(logits.intent.clone(), datum.intent));

        // Entity
        losses.push(cross_entropy(logits.entity.clone(), datum.entity));

        // Projections — target is index within the masked field set (1.5× weight)
        for (i, &global_idx) in datum.proj_fields.iter().enumerate() {
            if i < logits.projection.len() {
                if let Some(local) = valid_fields.iter().position(|&f| f == global_idx) {
                    losses.push(cross_entropy(logits.projection[i].clone(), local) * 1.5f32);
                }
            }
        }

        // Condition fields
        for (i, &global_idx) in datum.cond_fields.iter().enumerate() {
            if i < logits.cond_field.len() {
                if let Some(local) = valid_fields.iter().position(|&f| f == global_idx) {
                    losses.push(cross_entropy(logits.cond_field[i].clone(), local));
                }
            }
        }

        // Condition comparators
        for (i, &cmp_idx) in datum.cond_cmps.iter().enumerate() {
            if i < logits.cond_cmp.len() {
                losses.push(cross_entropy(logits.cond_cmp[i].clone(), cmp_idx));
            }
        }

        // Assignment fields (2× weight — structurally harder than read heads)
        for (i, &global_idx) in datum.asgn_fields.iter().enumerate() {
            if i < logits.assignment.len() {
                if let Some(local) = valid_fields.iter().position(|&f| f == global_idx) {
                    losses.push(cross_entropy(logits.assignment[i].clone(), local) * 2.0f32);
                }
            }
        }

        // Modifier types
        for (i, &mod_idx) in datum.mod_types.iter().enumerate() {
            if i < logits.mod_type.len() {
                losses.push(cross_entropy(logits.mod_type[i].clone(), mod_idx));
            }
        }

        // Modifier fields
        for (i, &global_idx) in datum.mod_fields.iter().enumerate() {
            if i < logits.mod_field.len() {
                if let Some(local) = valid_fields.iter().position(|&f| f == global_idx) {
                    losses.push(cross_entropy(logits.mod_field[i].clone(), local));
                }
            }
        }

        let count = losses.len();
        if count == 0 {
            return (Tensor::zeros([1], &self.device), 0);
        }
        let total = losses.into_iter().reduce(|a, b| a + b).unwrap();
        (total, count)
    }

    pub fn evaluate(&self, data: &[&LamellaDatum], bar: &ProgressBar) -> Metrics {
        let inner = self.model.valid();
        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        // Per-head f32 accumulators (correct_sum, total).
        // tc_* vecs are declared INSIDE the chunk loop so they never grow beyond
        // chunk_size (≤32) — we flush each with stack+sum at end of each chunk.
        // This gives 13 CE syncs + 13×8 = 104 accuracy syncs (was 3000 argmax syncs,
        // or worse: 8 syncs of 395-long chain-reduce that causes growing GPU backlogs).
        let mut s_intent = (0.0f32, 0usize);
        let mut s_entity = (0.0f32, 0usize);
        let mut s_proj   = (0.0f32, 0usize);
        let mut s_cf     = (0.0f32, 0usize);
        let mut s_cc     = (0.0f32, 0usize);
        let mut s_asgn   = (0.0f32, 0usize);
        let mut s_modt   = (0.0f32, 0usize);
        let mut s_modf   = (0.0f32, 0usize);

        let embs = inner.precompute_schema_embs(&self.catalog, &self.device);
        let eval_cache = GatherCache::new(&self.catalog, &self.device);

        for chunk in data.chunks(32) {
            let chunk_tokens: Vec<Vec<String>> = chunk.iter()
                .map(|d| tokenize(&d.nl).0)
                .collect();
            let (full_seqs, pools, _seq_lens, seq_mask) = inner.encode_nl_batch(
                &chunk_tokens, self.config.token_buckets, &self.device,
            );
            let (pool1s, entity_logits) = inner.level1_batch(pools, &embs, &self.device);
            let chunk_slots: Vec<_> = chunk.iter().map(|d| d.slot_counts()).collect();
            let chunk_entity_idxs: Vec<Option<usize>> = chunk.iter().map(|d| Some(d.entity)).collect();
            let chunk_logits = inner.head_scoring_batch(
                pool1s, entity_logits, full_seqs, seq_mask,
                &chunk_slots, &chunk_entity_idxs, &eval_cache, &embs, &self.device,
            );

            let mut chunk_ces: Vec<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = Vec::new();
            // Per-chunk tc_* — reset every iteration; stack+sum at end keeps the
            // GPU queue bounded to ≤chunk_size items between syncs.
            let mut tc_intent: Vec<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = Vec::new();
            let mut tc_entity: Vec<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = Vec::new();
            let mut tc_proj:   Vec<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = Vec::new();
            let mut tc_cf:     Vec<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = Vec::new();
            let mut tc_cc:     Vec<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = Vec::new();
            let mut tc_asgn:   Vec<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = Vec::new();
            let mut tc_modt:   Vec<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = Vec::new();
            let mut tc_modf:   Vec<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = Vec::new();

            for (datum, logits) in chunk.iter().zip(chunk_logits.iter()) {
                let valid_fields = &self.catalog.table_field_indices[datum.entity];
                let mut datum_ces: Vec<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = Vec::new();

                macro_rules! ce_push {
                    ($logit:expr, $target:expr) => {{
                        let ls = activation::log_softmax($logit.clone().unsqueeze::<2>(), 1).squeeze::<1>();
                        datum_ces.push(ls.slice([$target..$target + 1]).neg());
                    }};
                }

                ce_push!(&logits.intent, datum.intent);
                tc_intent.push(tensor_correct(&logits.intent, datum.intent));

                ce_push!(&logits.entity, datum.entity);
                tc_entity.push(tensor_correct(&logits.entity, datum.entity));

                for (i, &global_idx) in datum.proj_fields.iter().enumerate() {
                    if i < logits.projection.len() {
                        if let Some(local) = valid_fields.iter().position(|&f| f == global_idx) {
                            ce_push!(&logits.projection[i], local);
                            tc_proj.push(tensor_correct(&logits.projection[i], local));
                        }
                    }
                }
                for (i, &global_idx) in datum.cond_fields.iter().enumerate() {
                    if i < logits.cond_field.len() {
                        if let Some(local) = valid_fields.iter().position(|&f| f == global_idx) {
                            ce_push!(&logits.cond_field[i], local);
                            tc_cf.push(tensor_correct(&logits.cond_field[i], local));
                        }
                    }
                }
                for (i, &cmp_idx) in datum.cond_cmps.iter().enumerate() {
                    if i < logits.cond_cmp.len() {
                        ce_push!(&logits.cond_cmp[i], cmp_idx);
                        tc_cc.push(tensor_correct(&logits.cond_cmp[i], cmp_idx));
                    }
                }
                for (i, &global_idx) in datum.asgn_fields.iter().enumerate() {
                    if i < logits.assignment.len() {
                        if let Some(local) = valid_fields.iter().position(|&f| f == global_idx) {
                            ce_push!(&logits.assignment[i], local);
                            tc_asgn.push(tensor_correct(&logits.assignment[i], local));
                        }
                    }
                }
                for (i, &mod_idx) in datum.mod_types.iter().enumerate() {
                    if i < logits.mod_type.len() {
                        ce_push!(&logits.mod_type[i], mod_idx);
                        tc_modt.push(tensor_correct(&logits.mod_type[i], mod_idx));
                    }
                }
                for (i, &global_idx) in datum.mod_fields.iter().enumerate() {
                    if i < logits.mod_field.len() {
                        if let Some(local) = valid_fields.iter().position(|&f| f == global_idx) {
                            ce_push!(&logits.mod_field[i], local);
                            tc_modf.push(tensor_correct(&logits.mod_field[i], local));
                        }
                    }
                }

                let n = datum_ces.len();
                if n > 0 {
                    let datum_ce = datum_ces.into_iter().reduce(|a, b| a + b).unwrap() / n as f32;
                    chunk_ces.push(datum_ce);
                    count += 1;
                }
            }

            // CE sync — 1 per chunk (13 total)
            if !chunk_ces.is_empty() {
                let chunk_ce = chunk_ces.into_iter().reduce(|a, b| a + b).unwrap();
                total_loss += chunk_ce.into_scalar().elem::<f32>();
            }

            // Accuracy sync — stack [n,1] then sum → 1 parallel reduce per head per chunk.
            // Using stack+sum instead of chain-reduce: 2 GPU ops vs n-1 sequential adds.
            macro_rules! flush_tc {
                ($tc:expr, $acc:expr) => {{
                    let n = $tc.len();
                    if n > 0 {
                        let s: f32 = Tensor::stack::<2>($tc, 0).sum().into_scalar().elem::<f32>();
                        $acc.0 += s;
                        $acc.1 += n;
                    }
                }};
            }
            flush_tc!(tc_intent, s_intent);
            flush_tc!(tc_entity, s_entity);
            flush_tc!(tc_proj,   s_proj);
            flush_tc!(tc_cf,     s_cf);
            flush_tc!(tc_cc,     s_cc);
            flush_tc!(tc_asgn,   s_asgn);
            flush_tc!(tc_modt,   s_modt);
            flush_tc!(tc_modf,   s_modf);

            bar.inc(1);
        }

        let head_acc = HeadAcc {
            intent:     (s_intent.0.round() as usize, s_intent.1),
            entity:     (s_entity.0.round() as usize, s_entity.1),
            proj:       (s_proj.0.round()   as usize, s_proj.1),
            cond_field: (s_cf.0.round()     as usize, s_cf.1),
            cond_cmp:   (s_cc.0.round()     as usize, s_cc.1),
            assign:     (s_asgn.0.round()   as usize, s_asgn.1),
            mod_type:   (s_modt.0.round()   as usize, s_modt.1),
            mod_field:  (s_modf.0.round()   as usize, s_modf.1),
        };

        let val_loss = if count > 0 { total_loss / count as f32 } else { 0.0 };
        let total = head_acc.intent.1 + head_acc.entity.1 + head_acc.proj.1
            + head_acc.cond_field.1 + head_acc.cond_cmp.1 + head_acc.assign.1
            + head_acc.mod_type.1 + head_acc.mod_field.1;
        let correct = head_acc.intent.0 + head_acc.entity.0 + head_acc.proj.0
            + head_acc.cond_field.0 + head_acc.cond_cmp.0 + head_acc.assign.0
            + head_acc.mod_type.0 + head_acc.mod_field.0;
        let val_acc = if total > 0 { correct as f32 / total as f32 } else { 0.0 };

        Metrics { train_loss: 0.0, val_loss, val_acc, head_acc }
    }

    /// Print per-table assignment accuracy and top misclassifications.
    pub fn diagnose_asgn(&self, data: &[&LamellaDatum]) {
        let inner = self.model.valid();
        let embs = inner.precompute_schema_embs(&self.catalog, &self.device);
        let eval_cache = GatherCache::new(&self.catalog, &self.device);

        // per-table: (correct, total, Vec<(actual_field, predicted_field)> misses)
        let n_tables = self.catalog.tables.len();
        let mut tbl_correct = vec![0usize; n_tables];
        let mut tbl_total   = vec![0usize; n_tables];
        let mut tbl_misses: Vec<Vec<(String, String)>> = vec![Vec::new(); n_tables];

        for chunk in data.chunks(32) {
            let chunk_tokens: Vec<Vec<String>> = chunk.iter()
                .map(|d| tokenize(&d.nl).0)
                .collect();
            let (full_seqs, pools, _seq_lens, seq_mask) = inner.encode_nl_batch(
                &chunk_tokens, self.config.token_buckets, &self.device,
            );

            let (pool1s, entity_logits) = inner.level1_batch(pools, &embs, &self.device);

            let chunk_slots: Vec<_> = chunk.iter().map(|d| d.slot_counts()).collect();
            let chunk_entity_idxs: Vec<Option<usize>> = chunk.iter().map(|d| Some(d.entity)).collect();
            let chunk_logits = inner.head_scoring_batch(
                pool1s, entity_logits, full_seqs, seq_mask,
                &chunk_slots, &chunk_entity_idxs, &eval_cache, &embs, &self.device,
            );

            for (datum, logits) in chunk.iter().zip(chunk_logits.iter()) {
                if datum.asgn_fields.is_empty() { continue; }
                let valid_fields = &self.catalog.table_field_indices[datum.entity];
                let ti = datum.entity;

                for (i, &global_idx) in datum.asgn_fields.iter().enumerate() {
                    if i >= logits.assignment.len() { break; }
                    let Some(local_target) = valid_fields.iter().position(|&f| f == global_idx) else { continue };
                    tbl_total[ti] += 1;

                    // argmax over local candidates
                    use burn::tensor::ElementConversion;
                    let pred_local = logits.assignment[i].clone().argmax(0).into_scalar().elem::<i32>() as usize;
                    if pred_local == local_target {
                        tbl_correct[ti] += 1;
                    } else {
                        let actual_field = self.catalog.fields[global_idx].1.clone();
                        let pred_global  = valid_fields.get(pred_local).copied().unwrap_or(0);
                        let pred_field   = self.catalog.fields[pred_global].1.clone();
                        tbl_misses[ti].push((actual_field, pred_field));
                    }
                }
            }
        }

        println!("\n=== asgn head per-table ===");
        let mut rows: Vec<(usize, usize, usize)> = tbl_total.iter().enumerate()
            .filter(|(_, t)| **t > 0)
            .map(|(i, t)| (i, tbl_correct[i], *t))
            .collect();
        rows.sort_by(|a, b| {
            let pa = a.1 as f32 / a.2 as f32;
            let pb = b.1 as f32 / b.2 as f32;
            pa.partial_cmp(&pb).unwrap()
        });
        for (ti, correct, total) in &rows {
            let pct = *correct as f32 / *total as f32 * 100.0;
            println!("  {:15}  {:2}/{:2}  ({:.0}%)", self.catalog.tables[*ti], correct, total, pct);
            // Show up to 3 misses
            for (actual, pred) in tbl_misses[*ti].iter().take(3) {
                println!("      got {:?}  expected {:?}", pred, actual);
            }
        }
        println!();
    }

    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder.record(self.model.clone().into_record(), path.clone())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")))?;
        Ok(())
    }

    pub fn load(&mut self, path: &PathBuf) -> std::io::Result<()> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(path.clone(), &self.device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")))?;
        self.model = self.model.clone().load_record(record);
        Ok(())
    }
}

// =============================================================================
// Training loop
// =============================================================================

pub fn train_loop<B: AutodiffBackend>(
    ctx: &mut LamellaTrainCtx<B>,
    train_data: &[LamellaDatum],
    val_data: &[LamellaDatum],
    train_config: &TrainConfig,
    weights_path: &str,
) {
    let mut train_indices: Vec<usize> = (0..train_data.len()).collect();

    let bar = ProgressBar::new(train_config.epochs as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] epoch {pos}/{len}  {msg}"
        ).unwrap().progress_chars("=> "),
    );

    let mut best_val_loss = f32::MAX;
    let mut epoch_secs: Vec<f64> = Vec::new();
    let mut best_epoch = 0;
    let mut patience_counter = 0;

    let bs = train_config.batch_size;
    let num_batches = (train_indices.len() + bs - 1) / bs;

    for epoch in 0..train_config.epochs {
        let epoch_start = Instant::now();
        shuffle(&mut train_indices, epoch as u64);

        let mut epoch_loss = 0.0f32;
        let mut epoch_datums = 0usize;
        let batch_bar = ProgressBar::new(num_batches as u64);
        batch_bar.set_style(
            ProgressStyle::with_template(
                "  train [{bar:30.yellow/red}] {pos}/{len} batch  {per_sec}  eta {eta}"
            ).unwrap().progress_chars("=> "),
        );

        let mut batch_times: Vec<f32> = Vec::with_capacity(num_batches);
        for chunk in train_indices.chunks(bs) {
            let batch: Vec<&LamellaDatum> = chunk.iter().map(|&i| &train_data[i]).collect();
            let t0 = Instant::now();
            let batch_loss = ctx.step_batch(&batch);
            batch_times.push(t0.elapsed().as_secs_f32());
            epoch_loss += batch_loss * batch.len() as f32;
            epoch_datums += batch.len();
            batch_bar.inc(1);
        }
        batch_bar.finish_and_clear();
        let train_secs = epoch_start.elapsed().as_secs_f64();

        // Print per-batch timing split to diagnose within-epoch vs cross-epoch slowdown
        let n = batch_times.len();
        if n >= 6 {
            let first3: f32 = batch_times[..3].iter().sum::<f32>() / 3.0;
            let last3:  f32 = batch_times[n-3..].iter().sum::<f32>() / 3.0;
            bar.println(format!(
                "    timing: first3={:.2}s/batch  last3={:.2}s/batch  (growth={:+.2}s)",
                first3, last3, last3 - first3,
            ));
        }

        let train_loss = epoch_loss / epoch_datums as f32;
        let val_refs: Vec<&LamellaDatum> = val_data.iter().collect();
        let eval_bar = ProgressBar::new(((val_refs.len() + 31) / 32) as u64);
        eval_bar.set_style(
            ProgressStyle::with_template(
                "  eval  [{bar:30.green/blue}] {pos}/{len} batch  {per_sec}  eta {eta}"
            ).unwrap().progress_chars("=> "),
        );
        let mut metrics = ctx.evaluate(&val_refs, &eval_bar);
        eval_bar.finish_and_clear();
        metrics.train_loss = train_loss;

        let elapsed = epoch_start.elapsed().as_secs_f64();
        let eval_secs = elapsed - train_secs;
        if epoch > 0 { epoch_secs.push(elapsed); }
        let eta_str = if epoch_secs.is_empty() {
            "warmup".to_string()
        } else {
            let avg = epoch_secs.iter().copied().sum::<f64>() / epoch_secs.len() as f64;
            let remaining = (train_config.epochs - epoch - 1) as f64;
            fmt_duration(avg * remaining)
        };

        bar.set_message(format!(
            "loss={:.4} val={:.4} acc={:.1}%  eta {}",
            train_loss, metrics.val_loss, metrics.val_acc * 100.0, eta_str,
        ));
        bar.inc(1);
        bar.println(format!(
            "  epoch {:>2}  loss={:.4}  val={:.4}  acc={:.1}%  [tr{:.0}s ev{:.0}s]  | {}",
            epoch + 1, train_loss, metrics.val_loss, metrics.val_acc * 100.0,
            train_secs, eval_secs, metrics.head_acc.display(),
        ));

        if metrics.val_loss < best_val_loss {
            best_val_loss = metrics.val_loss;
            best_epoch = epoch;
            patience_counter = 0;

            let weights = PathBuf::from(weights_path);
            if let Some(parent) = weights.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = ctx.save(&weights);
        } else {
            patience_counter += 1;
            if patience_counter >= train_config.patience {
                bar.finish_with_message(format!(
                    "early stop at epoch {} (best={})", epoch, best_epoch
                ));
                return;
            }
        }
    }

    bar.finish_with_message(format!("done (best epoch={})", best_epoch));
}

fn fmt_duration(secs: f64) -> String {
    let s = secs as u64;
    if s < 60 { format!("{s}s") }
    else if s < 3600 { format!("{}m{:02}s", s / 60, s % 60) }
    else { format!("{}h{:02}m", s / 3600, (s % 3600) / 60) }
}

fn shuffle(v: &mut [usize], seed: u64) {
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    for i in (1..v.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
}
