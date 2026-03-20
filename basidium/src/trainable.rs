// trainable.rs — Trainable impls for Septa and Hyphae models

use crate::{
    Datum,
    trainer::{Metrics, Trainable},
};
use hyphae::model::Hyphae;
use septa::model::Septa;

impl<B: burn::tensor::backend::Backend> Trainable for Septa<B> {
    fn step(&mut self, _batch: &[Datum]) -> f32 {
        todo!()
    }

    fn evaluate(&self, _batch: &[Datum]) -> Metrics {
        todo!()
    }

    fn save(&self, _path: &std::path::PathBuf) -> std::io::Result<()> {
        todo!()
    }
}

impl<B: burn::tensor::backend::Backend> Trainable for Hyphae<B> {
    fn step(&mut self, _batch: &[Datum]) -> f32 {
        todo!()
    }

    fn evaluate(&self, _batch: &[Datum]) -> Metrics {
        todo!()
    }

    fn save(&self, _path: &std::path::PathBuf) -> std::io::Result<()> {
        todo!()
    }
}
