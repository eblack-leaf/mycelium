use std::path::{Path, PathBuf};

use crate::schema::{Schema, SchemaError};

/// A loaded plugin module — schema config + trained spore weights.
pub struct Module {
    pub schema: Schema,
    pub weights_dir: PathBuf,
}

impl Module {
    /// Load a module from a directory containing schema.json and weight files.
    ///
    /// Expected layout:
    ///   schema.json          — fields, phrases, temporal markers
    ///   field_spore.bin      — trained field CNN weights
    ///   phrase_spore.bin     — trained phrase CNN weights
    ///   temporal_spore.bin   — trained temporal CNN weights
    ///   orchestrator.bin     — trained orchestrator model weights
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, ModuleError> {
        let dir = dir.as_ref();
        let schema = Schema::load(dir.join("schema.json")).map_err(ModuleError::Schema)?;

        Ok(Self {
            schema,
            weights_dir: dir.to_path_buf(),
        })
    }

    pub fn weights_path(&self, name: &str) -> PathBuf {
        self.weights_dir.join(format!("{name}.bin"))
    }

    pub fn has_weights(&self, name: &str) -> bool {
        self.weights_path(name).exists()
    }
}

#[derive(Debug)]
pub enum ModuleError {
    Schema(SchemaError),
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Schema(e) => write!(f, "module: {e}"),
        }
    }
}

impl std::error::Error for ModuleError {}
