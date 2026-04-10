pub mod normalize;

use std::{
    collections::HashMap,
    path::PathBuf,
};
use serde::{Deserialize, Serialize};

use normalize::normalize;

/// First-order Markov transition table over normalized query templates.
/// Counts how often query B follows query A in the user's session history.
/// Persisted to disk as JSON so it accumulates across sessions (online learning).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PatternTable {
    /// transitions[from][to] = count
    transitions: HashMap<String, HashMap<String, u64>>,
    /// Total queries observed — used to age out stale counts in the future
    total_observed: u64,
}

impl PatternTable {
    /// Load from disk, or start fresh if the file doesn't exist.
    pub fn load(path: &PathBuf) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist to disk.
    pub fn save(&self, path: &PathBuf) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            std::fs::write(path, json).ok();
        }
    }

    /// Record that `next_query` was run immediately after `prev_query`.
    pub fn observe(&mut self, prev_query: &str, next_query: &str) {
        let from = normalize(prev_query);
        let to = normalize(next_query);
        *self.transitions
            .entry(from)
            .or_default()
            .entry(to)
            .or_insert(0) += 1;
        self.total_observed += 1;
    }

    /// Given the last query, return the top-k most likely next query templates
    /// sorted by frequency descending.
    pub fn suggest(&self, last_query: &str, k: usize) -> Vec<SuggestedPattern> {
        let key = normalize(last_query);
        let Some(nexts) = self.transitions.get(&key) else {
            return vec![];
        };
        let total: u64 = nexts.values().sum();
        let mut ranked: Vec<_> = nexts
            .iter()
            .map(|(template, &count)| SuggestedPattern {
                template: template.clone(),
                probability: count as f64 / total as f64,
                count,
            })
            .collect();
        ranked.sort_by(|a, b| b.count.cmp(&a.count));
        ranked.truncate(k);
        ranked
    }

    pub fn total_patterns(&self) -> usize {
        self.transitions.len()
    }
}

#[derive(Debug, Clone)]
pub struct SuggestedPattern {
    /// Normalized template, e.g. "UPDATE <table> SET <field> = <str>"
    pub template: String,
    pub probability: f64,
    pub count: u64,
}
