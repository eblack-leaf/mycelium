use std::collections::HashMap;

// ── Input: char-level encoding of the raw value string ───────────────────────

pub const CHAR_VOCAB_SIZE: usize = 96; // printable ASCII 32..=126
pub const CHAR_PAD: usize = 95;
pub const CHAR_MAX_LEN: usize = 96;

pub fn encode_value(s: &str) -> Vec<usize> {
    let mut ids: Vec<usize> = s
        .chars()
        .filter_map(|c| {
            let v = c as usize;
            if (32..=126).contains(&v) { Some(v - 32) } else { None }
        })
        .take(CHAR_MAX_LEN)
        .collect();
    ids.resize(CHAR_MAX_LEN, CHAR_PAD);
    ids
}

// ── Output: word-level vocabulary built from training name words ──────────────

pub const STOP: usize = 0; // reserved — always index 0

/// Word vocabulary derived from training names.
/// Each name is split on '-' to extract constituent words.
/// The model predicts word indices, not characters.
#[derive(Debug, Clone)]
pub struct WordVocab {
    words: Vec<String>,
    index: HashMap<String, usize>,
}

impl WordVocab {
    /// Build from an iterator of name strings.
    /// Index 0 is always STOP.
    pub fn build<'a>(names: impl Iterator<Item = &'a str>) -> Self {
        let mut set: Vec<String> = names
            .flat_map(|n| n.split('-').map(str::to_lowercase))
            .filter(|w| !w.is_empty())
            .collect();
        set.sort();
        set.dedup();

        // STOP is always 0; real words start at 1
        let words: Vec<String> = std::iter::once(String::from("<stop>"))
            .chain(set)
            .collect();
        let index = words.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
        Self { words, index }
    }

    pub fn size(&self) -> usize {
        self.words.len()
    }

    /// Encode a name into at most 2 word indices + stop.
    /// Returns [word1, word2_or_stop] — always length 2.
    pub fn encode_name(&self, name: &str) -> [usize; 2] {
        let parts: Vec<usize> = name
            .split('-')
            .filter(|w| !w.is_empty())
            .map(|w| *self.index.get(&w.to_lowercase()).unwrap_or(&STOP))
            .take(2)
            .collect();
        match parts.as_slice() {
            [a, b] => [*a, *b],
            [a]    => [*a, STOP],
            _      => [STOP, STOP],
        }
    }

    /// Decode word indices back to a hyphen-joined name.
    pub fn decode_name(&self, w1: usize, w2: usize) -> String {
        let first = self.words.get(w1).map(|s| s.as_str()).unwrap_or("");
        if w2 == STOP || w2 == 0 {
            first.to_string()
        } else {
            let second = self.words.get(w2).map(|s| s.as_str()).unwrap_or("");
            format!("{}-{}", first, second)
        }
    }

    /// Persist as a plain text file, one word per line.
    pub fn save(&self, path: &std::path::Path) {
        std::fs::write(path, self.words.join("\n")).ok();
    }

    /// Load from the same plain text file.
    pub fn load(path: &std::path::Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let words: Vec<String> = text.lines().map(str::to_string).collect();
        let index = words.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
        Some(Self { words, index })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_word_vocab() {
        let names = ["target-user", "alice", "admin-role", "count"];
        let vocab = WordVocab::build(names.iter().copied());
        let [w1, w2] = vocab.encode_name("target-user");
        assert_ne!(w1, STOP);
        let decoded = vocab.decode_name(w1, w2);
        assert_eq!(decoded, "target-user");
    }

    #[test]
    fn single_word_name() {
        let names = ["alice", "count"];
        let vocab = WordVocab::build(names.iter().copied());
        let [w1, w2] = vocab.encode_name("alice");
        assert_eq!(w2, STOP);
        assert_eq!(vocab.decode_name(w1, w2), "alice");
    }
}
