use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Profile {
    /// Descriptive, kebab-case: `target-user`, `result-count`, `alice`
    Classic,
    /// Short, abbreviated: `usr`, `n`, `tgt`
    Terse,
    /// Evocative, unexpected: `wanderer`, `the-chosen`, `echo`
    Creative,
    /// Informal, irreverent: `that-user`, `the-boss`, `magic-num`
    Hacker,
    /// User-supplied checkpoint directory
    Custom(String),
}

impl Profile {
    pub fn name(&self) -> &str {
        match self {
            Profile::Classic    => "classic",
            Profile::Terse      => "terse",
            Profile::Creative   => "creative",
            Profile::Hacker     => "hacker",
            Profile::Custom(n)  => n.as_str(),
        }
    }

    pub fn checkpoint_dir(&self, base: &PathBuf) -> PathBuf {
        base.join("profiles").join(self.name())
    }

    pub fn model_path(&self, base: &PathBuf) -> PathBuf {
        self.checkpoint_dir(base).join("model.bin")
    }

    pub fn vocab_path(&self, base: &PathBuf) -> PathBuf {
        self.checkpoint_dir(base).join("vocab.txt")
    }

    pub fn training_data_path(&self, base: &PathBuf) -> PathBuf {
        base.join(format!("{}.jsonl", self.name()))
    }
}

impl Default for Profile {
    fn default() -> Self { Profile::Classic }
}

/// One training example: a raw value string and the name a human gave it.
/// The name encodes both what the value is AND the profile's style.
/// Context is empty — style comes from which profile's train.jsonl this lives in.
#[derive(Debug, Serialize, Deserialize)]
pub struct NamerExample {
    /// The raw value — whatever was saved: `"user:abc"`, `"Alice"`, `42`, `{...}`
    pub value: String,
    /// The name in the style of this profile
    pub name: String,
}
