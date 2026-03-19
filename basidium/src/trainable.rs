// trainable.rs — Trainable impls for septa and hyphae models

use crate::{Datum, trainer::{Trainable, Metrics}};
use septa::model::Model as SeptaModel;
use hyphae::model::GnnModel;

impl Trainable for SeptaModel {
    fn step(&mut self, batch: &[Datum]) -> f32 {
        todo!()
    }

    fn evaluate(&self, batch: &[Datum]) -> Metrics {
        todo!()
    }

    fn save(&self, path: &std::path::PathBuf) -> std::io::Result<()> {
        todo!()
    }
}

impl Trainable for GnnModel {
    fn step(&mut self, batch: &[Datum]) -> f32 {
        todo!()
    }

    fn evaluate(&self, batch: &[Datum]) -> Metrics {
        todo!()
    }

    fn save(&self, path: &std::path::PathBuf) -> std::io::Result<()> {
        todo!()
    }
}
