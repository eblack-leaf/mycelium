use basidium::{trainable::PipelineTrainCtx, trainer::{Trainer, TrainerConfig}, Datum};
use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
use hyphae::{graph::SchemaGraph, model::HyphaeConfig, schema::Schema};
use septa::model::SeptaConfig;
use std::path::Path;

type B = Autodiff<Wgpu>;

fn main() {
    let device = WgpuDevice::default();
    let schema = Schema::from_dir(Path::new("stipe/fixtures/schema/")).unwrap();
    let data = Datum::generate(&schema);
    println!("Generated {} training datums", data.len());

    let hyphae_config = HyphaeConfig::new();
    let septa_config = SeptaConfig::new(12); // 12 BIO tags: B-/I- × 6 span types
    let schema_graph = SchemaGraph::new(schema, hyphae_config.ngram_buckets);

    let ctx: PipelineTrainCtx<B> = PipelineTrainCtx::new(
        hyphae_config, septa_config, schema_graph, 1e-3, &device,
    );
    let trainer_config = TrainerConfig::new();
    let mut trainer = Trainer::new(trainer_config, ctx, "weights/pipeline");

    let result = trainer.train(&data);
    println!(
        "Best epoch: {} — val_loss={:.4} val_acc={:.3}",
        result.best_epoch, result.best_metrics.val_loss, result.best_metrics.val_acc
    );
}
