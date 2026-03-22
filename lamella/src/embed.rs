use std::collections::HashSet;

/// FNV-1a hash of a string.
pub fn fnv1a(s: &str) -> u64 {
    const PRIME: u64 = 1_099_511_628_211;
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    let mut h = OFFSET;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Extract character 2-gram and 3-gram hash buckets from a name.
pub fn char_ngram_buckets(name: &str, num_buckets: usize) -> Vec<usize> {
    let chars: Vec<char> = name.chars().collect();
    let mut seen = HashSet::new();
    let mut buckets = Vec::new();

    for n in [2usize, 3] {
        for window in chars.windows(n) {
            let s: String = window.iter().collect();
            let bucket = fnv1a(&s) as usize % num_buckets;
            if seen.insert(bucket) {
                buckets.push(bucket);
            }
        }
    }

    if buckets.is_empty() {
        buckets.push(fnv1a(name) as usize % num_buckets);
    }

    buckets
}

/// Whitespace tokenizer that tracks byte offsets for each token.
pub fn tokenize(text: &str) -> (Vec<String>, Vec<(usize, usize)>) {
    let mut tokens = Vec::new();
    let mut ranges = Vec::new();
    let mut i = 0;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        tokens.push(text[start..i].to_lowercase());
        ranges.push((start, i));
    }

    (tokens, ranges)
}
