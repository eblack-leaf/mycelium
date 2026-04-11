/// Evocative 2-word name generator.
///
/// Two styles:
///   Creative — poetic, unexpected: "hollow-thread", "amber-tide", "drifting-lantern"
///   Hacker   — terse, irreverent: "dead-loop", "ghost-wire", "null-gate"
///
/// Names are random; call `generate` whenever a placeholder value needs a name.

use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Creative,
    Hacker,
}

const CREATIVE_ADJ: &[&str] = &[
    "silver", "hollow", "woven", "drifting", "amber", "crimson", "distant",
    "fading", "ancient", "quiet", "fleeting", "wandering", "luminous",
    "broken", "hidden", "sunken", "velvet", "ashen", "gilded", "borrowed",
];

const CREATIVE_NOUN: &[&str] = &[
    "thread", "echo", "lantern", "current", "ember", "threshold", "tide",
    "mirror", "signal", "ridge", "vessel", "glimmer", "compass", "fragment",
    "lattice", "veil", "canopy", "mantle", "cipher", "relic",
];

const HACKER_ADJ: &[&str] = &[
    "raw", "zero", "dark", "ghost", "dead", "live", "dirty", "loose",
    "cold", "fast", "silent", "spare", "sharp", "null", "void",
    "soft", "hard", "deep", "flat", "open",
];

const HACKER_NOUN: &[&str] = &[
    "wire", "bus", "stack", "loop", "flag", "gate", "node", "hook",
    "mask", "tick", "salt", "sink", "core", "edge", "root",
    "pipe", "fork", "trap", "lock", "patch",
];

/// Generate a random 2-word name in the given style.
pub fn generate(style: Style) -> String {
    let mut rng = rand::thread_rng();
    let (adjs, nouns) = match style {
        Style::Creative => (CREATIVE_ADJ, CREATIVE_NOUN),
        Style::Hacker   => (HACKER_ADJ,   HACKER_NOUN),
    };
    let adj  = adjs[rng.gen_range(0..adjs.len())];
    let noun = nouns[rng.gen_range(0..nouns.len())];
    format!("{}-{}", adj, noun)
}

/// Generate a name picking a style at random (equal chance of Creative or Hacker).
pub fn generate_random() -> String {
    let style = if rand::thread_rng().gen_bool(0.5) {
        Style::Creative
    } else {
        Style::Hacker
    };
    generate(style)
}
