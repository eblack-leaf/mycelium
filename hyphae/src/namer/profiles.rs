use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// A naming profile — each profile is a separately trained checkpoint
/// of the same NamerModel architecture, trained on style-specific examples.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Profile {
    /// Descriptive, kebab-case, reads like a variable name.
    /// e.g. "target-user-id", "current-record", "selected-item"
    Classic,
    /// Short, 1-2 syllables, abbreviated.
    /// e.g. "tgt", "cur-rec", "sel"
    Terse,
    /// Evocative, slightly unexpected, personality-driven.
    /// e.g. "the-one", "quest-anchor", "chosen-path"
    Creative,
    /// User-defined profile — points to a custom checkpoint directory.
    Custom(String),
}

impl Profile {
    pub fn name(&self) -> &str {
        match self {
            Profile::Classic => "classic",
            Profile::Terse => "terse",
            Profile::Creative => "creative",
            Profile::Custom(name) => name.as_str(),
        }
    }

    /// Directory where this profile's checkpoint and training data live.
    pub fn checkpoint_dir(&self, base: &PathBuf) -> PathBuf {
        base.join("profiles").join(self.name())
    }

    pub fn model_path(&self, base: &PathBuf) -> PathBuf {
        self.checkpoint_dir(base).join("model.bin")
    }

    pub fn training_data_path(&self, base: &PathBuf) -> PathBuf {
        self.checkpoint_dir(base).join("train.jsonl")
    }

    pub fn all_builtin() -> &'static [Profile] {
        &[Profile::Classic, Profile::Terse, Profile::Creative]
    }
}

impl Default for Profile {
    fn default() -> Self {
        Profile::Classic
    }
}

/// A single training example for the namer model.
/// Stored as JSONL in the profile's `train.jsonl`.
#[derive(Debug, Serialize, Deserialize)]
pub struct NamerExample {
    /// The value being named (JSON string, truncated at 128 chars of input encoding)
    pub value: String,
    /// Optional recent query context (the query that produced this value)
    pub context: String,
    /// The human-chosen name in the style of this profile
    pub name: String,
}
