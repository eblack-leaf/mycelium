// trainable.rs — Trainable impls with end-to-end Septa → Hyphae pipeline

use crate::{
    Datum, SpanLabel, SpanType,
    trainer::{HeadAcc, Metrics, Trainable},
};
use burn::{
    lr_scheduler::{cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig}, LrScheduler},
    module::{AutodiffModule, Module},
    optim::{AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer},
    tensor::{activation, backend::AutodiffBackend, backend::Backend, ElementConversion, Tensor},
};
use hyphae::{
    graph::SchemaGraph,
    model::{Hyphae, HyphaeConfig, HyphaeLogits},
    query::QueryNode,
};
use septa::model::{Septa, SeptaConfig};

// =============================================================================
// Basidium — combined Septa + Hyphae for end-to-end gradient flow
// =============================================================================

#[derive(Module, Debug)]
pub struct Basidium<B: Backend> {
    pub septa: Septa<B>,
    pub hyphae: Hyphae<B>,
}

// =============================================================================
// BasidiumTrainCtx — wraps Basidium + SchemaGraph + optimizer for training
// =============================================================================

pub struct BasidiumTrainCtx<B: AutodiffBackend> {
    pub model: Basidium<B>,
    pub schema_graph: SchemaGraph,
    pub hyphae_config: HyphaeConfig,
    pub septa_config: SeptaConfig,
    pub optimizer: OptimizerAdaptor<B>,
    pub lr_scheduler: CosineAnnealingLrScheduler,
    pub device: B::Device,
}

type OptimizerAdaptor<B> = burn::optim::adaptor::OptimizerAdaptor<AdamW, Basidium<B>, B>;

impl<B: AutodiffBackend> BasidiumTrainCtx<B> {
    pub fn new(
        hyphae_config: HyphaeConfig,
        septa_config: SeptaConfig,
        schema_graph: SchemaGraph,
        lr: f64,
        num_iters: usize,
        device: &B::Device,
    ) -> Self {
        let hyphae = Hyphae::new(&hyphae_config, device);
        let septa = Septa::new(&septa_config, device);
        let model = Basidium { septa, hyphae };
        let optimizer = AdamWConfig::new().init();
        let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(lr, num_iters)
            .with_min_lr(lr * 0.01)
            .init()
            .unwrap();
        Self { model, schema_graph, hyphae_config, septa_config, optimizer, lr_scheduler, device: device.clone() }
    }
}

// =============================================================================
// Loss + accuracy helpers
// =============================================================================

/// Cross-entropy loss for a single logit vector and target index.
/// logits: [n], target_idx: index into [0..n).
fn cross_entropy<B: Backend>(logits: Tensor<B, 1>, target_idx: usize) -> Tensor<B, 1> {
    let log_softmax = activation::log_softmax(logits.unsqueeze::<2>(), 1).squeeze::<1>();
    log_softmax.slice([target_idx..target_idx + 1]).neg()
}

/// Find the index of `target` within `candidates` by comparing graph nodes.
fn find_target_in_candidates(
    nodes: &[QueryNode],
    candidates: &[usize],
    target: &QueryNode,
) -> Option<usize> {
    candidates.iter().position(|&c| &nodes[c] == target)
}

/// Compute total loss over all labels. Returns (loss_tensor, num_labels).
fn resolution_loss<B: Backend>(
    logits: &HyphaeLogits<B>,
    labels: &[SpanLabel],
    nodes: &[QueryNode],
    graph: &hyphae::graph::GroundedGraph,
    semantics: &septa::Semantics,
    device: &B::Device,
) -> (Tensor<B, 1>, usize) {
    let mut losses: Vec<Tensor<B, 1>> = Vec::new();

    let assign_logit_map = build_ragged_map(&semantics.assignments, |a| a.field_text.is_some());
    let mod_field_logit_map = build_ragged_map(&semantics.modifiers, |m| m.argument.is_some());

    for label in labels {
        let loss = match (&label.span_type, &label.target) {
            (SpanType::Intent, QueryNode::Operation(_)) => {
                let idx = find_target_in_candidates(nodes, &graph.intent_resolution.candidates, &label.target);
                idx.map(|i| cross_entropy(logits.intent.clone(), i))
            }
            (SpanType::Entity, QueryNode::Table(_)) => {
                let idx = find_target_in_candidates(nodes, &graph.entity_resolution.candidates, &label.target);
                idx.map(|i| cross_entropy(logits.entity.clone(), i))
            }
            (SpanType::Projection, QueryNode::Field { .. }) => {
                let si = label.span_index;
                if si < logits.projection.len() {
                    let res = &graph.projection_resolutions[si];
                    let idx = find_target_in_candidates(nodes, &res.candidates, &label.target);
                    idx.map(|i| cross_entropy(logits.projection[si].clone(), i))
                } else { None }
            }
            (SpanType::Condition, QueryNode::Field { .. }) => {
                let si = label.span_index;
                if si < logits.condition_field.len() {
                    let res = &graph.condition_field_resolutions[si];
                    let idx = find_target_in_candidates(nodes, &res.candidates, &label.target);
                    idx.map(|i| cross_entropy(logits.condition_field[si].clone(), i))
                } else { None }
            }
            (SpanType::Condition, QueryNode::Comparator(_)) => {
                let si = label.span_index;
                if si < logits.condition_cmp.len() {
                    let res = &graph.condition_cmp_resolutions[si];
                    let idx = find_target_in_candidates(nodes, &res.candidates, &label.target);
                    idx.map(|i| cross_entropy(logits.condition_cmp[si].clone(), i))
                } else { None }
            }
            (SpanType::Assignment, QueryNode::Field { .. }) => {
                let si = label.span_index;
                if si < assign_logit_map.len() {
                    assign_logit_map[si].and_then(|li| {
                        if li < logits.assignment.len() {
                            let res = &graph.assignment_resolutions[li];
                            let idx = find_target_in_candidates(nodes, &res.candidates, &label.target);
                            idx.map(|i| cross_entropy(logits.assignment[li].clone(), i))
                        } else { None }
                    })
                } else { None }
            }
            (SpanType::Modifier, QueryNode::Modifier(_)) => {
                let si = label.span_index;
                if si < logits.modifier_type.len() {
                    let res = &graph.modifier_type_resolutions[si];
                    let idx = find_target_in_candidates(nodes, &res.candidates, &label.target);
                    idx.map(|i| cross_entropy(logits.modifier_type[si].clone(), i))
                } else { None }
            }
            (SpanType::Modifier, QueryNode::Field { .. }) => {
                let si = label.span_index;
                if si < mod_field_logit_map.len() {
                    mod_field_logit_map[si].and_then(|li| {
                        if li < logits.modifier_field.len() {
                            let res = &graph.modifier_field_resolutions[li];
                            let idx = find_target_in_candidates(nodes, &res.candidates, &label.target);
                            idx.map(|i| cross_entropy(logits.modifier_field[li].clone(), i))
                        } else { None }
                    })
                } else { None }
            }
            _ => None,
        };

        if let Some(l) = loss {
            losses.push(l);
        }
    }

    let count = losses.len();
    if count == 0 {
        return (Tensor::zeros([1], device), 0);
    }

    let total = losses.into_iter().reduce(|a, b| a + b).unwrap();
    let mean = total / (count as f32);
    (mean, count)
}

/// Count correct predictions per head.
fn count_correct<B: Backend>(
    logits: &HyphaeLogits<B>,
    labels: &[SpanLabel],
    nodes: &[QueryNode],
    graph: &hyphae::graph::GroundedGraph,
    semantics: &septa::Semantics,
    head_acc: &mut HeadAcc,
) {
    let assign_logit_map = build_ragged_map(&semantics.assignments, |a| a.field_text.is_some());
    let mod_field_logit_map = build_ragged_map(&semantics.modifiers, |m| m.argument.is_some());

    let check = |logit_vec: &Tensor<B, 1>, candidates: &[usize], target: &QueryNode| -> bool {
        if let Some(target_idx) = find_target_in_candidates(nodes, candidates, target) {
            let argmax: i32 = logit_vec.clone().argmax(0).into_scalar().elem();
            argmax as usize == target_idx
        } else {
            false
        }
    };

    let score = |pair: &mut (usize, usize), hit: bool| {
        pair.1 += 1;
        if hit { pair.0 += 1; }
    };

    for label in labels {
        match (&label.span_type, &label.target) {
            (SpanType::Intent, QueryNode::Operation(_)) =>
                score(&mut head_acc.intent, check(&logits.intent, &graph.intent_resolution.candidates, &label.target)),
            (SpanType::Entity, QueryNode::Table(_)) =>
                score(&mut head_acc.entity, check(&logits.entity, &graph.entity_resolution.candidates, &label.target)),
            (SpanType::Projection, QueryNode::Field { .. }) if label.span_index < logits.projection.len() =>
                score(&mut head_acc.proj, check(&logits.projection[label.span_index], &graph.projection_resolutions[label.span_index].candidates, &label.target)),
            (SpanType::Condition, QueryNode::Field { .. }) if label.span_index < logits.condition_field.len() =>
                score(&mut head_acc.cond_field, check(&logits.condition_field[label.span_index], &graph.condition_field_resolutions[label.span_index].candidates, &label.target)),
            (SpanType::Condition, QueryNode::Comparator(_)) if label.span_index < logits.condition_cmp.len() =>
                score(&mut head_acc.cond_cmp, check(&logits.condition_cmp[label.span_index], &graph.condition_cmp_resolutions[label.span_index].candidates, &label.target)),
            (SpanType::Assignment, QueryNode::Field { .. }) => {
                let hit = assign_logit_map.get(label.span_index).copied().flatten().map_or(false, |li| {
                    li < logits.assignment.len() && check(&logits.assignment[li], &graph.assignment_resolutions[li].candidates, &label.target)
                });
                score(&mut head_acc.assign, hit);
            }
            (SpanType::Modifier, QueryNode::Modifier(_)) if label.span_index < logits.modifier_type.len() =>
                score(&mut head_acc.mod_type, check(&logits.modifier_type[label.span_index], &graph.modifier_type_resolutions[label.span_index].candidates, &label.target)),
            (SpanType::Modifier, QueryNode::Field { .. }) => {
                let hit = mod_field_logit_map.get(label.span_index).copied().flatten().map_or(false, |li| {
                    li < logits.modifier_field.len() && check(&logits.modifier_field[li], &graph.modifier_field_resolutions[li].candidates, &label.target)
                });
                score(&mut head_acc.mod_field, hit);
            }
            _ => {}
        };
    }
}

fn build_ragged_map<T, F: Fn(&T) -> bool>(items: &[T], pred: F) -> Vec<Option<usize>> {
    let mut map = Vec::new();
    let mut j = 0;
    for item in items {
        if pred(item) {
            map.push(Some(j));
            j += 1;
        } else {
            map.push(None);
        }
    }
    map
}

// =============================================================================
// Trainable for BasidiumTrainCtx
// =============================================================================

impl<B: AutodiffBackend> Trainable for BasidiumTrainCtx<B> {
    fn step_batch(&mut self, batch: &[&Datum]) -> f32 {
        let texts: Vec<&str> = batch.iter().map(|d| d.nl.as_str()).collect();
        let sems: Vec<&septa::Semantics> = batch.iter().map(|d| &d.semantics).collect();
        let all_hiddens = self.model.septa.batch_forward_with_spans(
            &texts, &sems, self.septa_config.vocab_size, &self.device,
        );

        // Per-datum backward with gradient accumulation, single optimizer step per batch.
        let mut accumulator: GradientsAccumulator<Basidium<B>> = GradientsAccumulator::new();
        let mut total_loss_val = 0.0f32;
        let mut n_total = 0usize;

        for i in 0..batch.len() {
            let datum = batch[i];
            let hiddens = &all_hiddens[i];
            let graph = self.schema_graph.inject(&datum.semantics);
            let logits = self.model.hyphae.forward(&graph, hiddens, &self.device);
            let (loss, n) = resolution_loss(
                &logits, &datum.labels, &graph.nodes, &graph, &datum.semantics, &self.device,
            );
            if n == 0 { continue; }

            let loss = loss / (n as f32);
            total_loss_val += loss.clone().inner().into_scalar().elem::<f32>() * n as f32;
            n_total += n;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.model);
            accumulator.accumulate(&self.model, grads);
        }

        if n_total == 0 { return 0.0; }

        let lr = self.lr_scheduler.step();
        self.model = self.optimizer.step(lr, self.model.clone(), accumulator.grads());

        total_loss_val / n_total as f32
    }

    fn evaluate(&self, batch: &[&Datum], bar: &indicatif::ProgressBar) -> Metrics {
        let inner = self.model.valid(); // strip autodiff — no gradient tracking
        let mut total_loss = 0.0f32;
        let mut count = 0usize;
        let mut head_acc = HeadAcc::default();

        for chunk in batch.chunks(32) {
            let texts: Vec<&str> = chunk.iter().map(|d| d.nl.as_str()).collect();
            let sems: Vec<&septa::Semantics> = chunk.iter().map(|d| &d.semantics).collect();
            let all_hiddens = inner.septa.batch_forward_with_spans(
                &texts, &sems, self.septa_config.vocab_size, &self.device,
            );
            for (datum, hiddens) in chunk.iter().zip(all_hiddens.iter()) {
                let graph = self.schema_graph.inject(&datum.semantics);
                let logits = inner.hyphae.forward(&graph, hiddens, &self.device);

                let (loss, n) = resolution_loss(
                    &logits, &datum.labels, &graph.nodes, &graph, &datum.semantics, &self.device,
                );
                if n > 0 {
                    total_loss += loss.into_scalar().elem::<f32>();
                    count += 1;
                }

                count_correct(
                    &logits, &datum.labels, &graph.nodes, &graph, &datum.semantics, &mut head_acc,
                );
            }
            bar.inc(1);
        }

        let val_loss = if count > 0 { total_loss / count as f32 } else { 0.0 };
        let total = head_acc.intent.1 + head_acc.entity.1 + head_acc.proj.1
            + head_acc.cond_field.1 + head_acc.cond_cmp.1 + head_acc.assign.1
            + head_acc.mod_type.1 + head_acc.mod_field.1;
        let correct = head_acc.intent.0 + head_acc.entity.0 + head_acc.proj.0
            + head_acc.cond_field.0 + head_acc.cond_cmp.0 + head_acc.assign.0
            + head_acc.mod_type.0 + head_acc.mod_field.0;
        let val_acc = if total > 0 { correct as f32 / total as f32 } else { 0.0 };

        Metrics {
            train_loss: 0.0, // filled by trainer
            val_loss,
            train_acc: 0.0,
            val_acc,
            f1: val_acc,
            head_acc,
        }
    }

    fn save(&self, path: &std::path::PathBuf) -> std::io::Result<()> {
        use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder.record(self.model.clone().into_record(), path.clone())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")))?;
        Ok(())
    }
}

impl<B: AutodiffBackend> BasidiumTrainCtx<B> {
    pub fn load(&mut self, path: &std::path::PathBuf) -> std::io::Result<()> {
        use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(path.clone(), &self.device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")))?;
        self.model = self.model.clone().load_record(record);
        Ok(())
    }
}
