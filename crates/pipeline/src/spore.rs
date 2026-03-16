use crate::hint::{AnnotatedInput, Hint, HintKind, Span};
use crate::schema::Schema;

/// A spore scans input and produces positional hints.
///
/// CNN-backed spores use multi-width conv filters over character embeddings.
/// Unlike standard Kim-CNN (which global max-pools and loses positions),
/// spores keep the activation map spatial — each position gets a class score.
///
/// Architecture per spore:
///   char embeddings [vocab_size × embed_dim]
///   → parallel Conv1d per filter width (e.g. 2,3,4,5-gram)
///   → concat filter outputs along channel dim per position
///   → 1×1 conv or linear per position → [seq_len × num_classes] score map
///   → threshold → Activations with (range, class, score)
///
/// This is a 1D fully-convolutional tagger, not a classifier.
/// Typo tolerance comes from partial n-gram overlap: "stokc" still fires
/// bigram "st" and trigram "sto" against filters trained on "stock".
pub trait Spore {
    fn scan(&self, input: &str, schema: &Schema) -> Vec<Hint>;
}

/// Raw activation from a spore's conv output at a position.
#[derive(Debug, Clone)]
pub struct Activation {
    /// Byte range in input where the n-gram pattern fired.
    pub range: std::ops::Range<usize>,
    /// Class index (maps to field/phrase/marker via schema).
    pub class: usize,
    /// Position-wise score from the conv tagger.
    pub score: f32,
}

/// Label for what a spore is tagging.
#[derive(Debug, Clone)]
pub enum SporeKind {
    Field,
    Phrase,
    Temporal,
    Op,
}

/// A CNN-backed spore. Weights loaded from the plugin module.
pub struct CnnSpore {
    pub kind: SporeKind,
    // TODO: Burn model handle — Conv1d banks + position-wise classifier
    // Loaded from weights.bin via Burn's record system
}

impl CnnSpore {
    /// Forward pass through the conv tagger, returning activations above threshold.
    pub fn activate(&self, _input: &str, _threshold: f32) -> Vec<Activation> {
        // TODO:
        // 1. Tokenize input to char indices
        // 2. Embed → [seq_len × embed_dim]
        // 3. Conv1d per filter width → [seq_len × num_filters] each
        // 4. Concat along channels → [seq_len × total_filters]
        // 5. Position-wise linear → [seq_len × num_classes]
        // 6. Threshold + extract spans (merge adjacent positions for same class)
        Vec::new()
    }
}

impl Spore for CnnSpore {
    fn scan(&self, input: &str, schema: &Schema) -> Vec<Hint> {
        self.activate(input, 0.5)
            .into_iter()
            .filter_map(|act| {
                let text = input.get(act.range.clone())?.to_string();
                let kind = match self.kind {
                    SporeKind::Field => HintKind::Field {
                        field: schema.fields.get(act.class)?.name.clone(),
                    },
                    SporeKind::Phrase => {
                        let p = schema.phrases.get(act.class)?;
                        HintKind::Phrase {
                            field: p.field.clone(),
                            op: p.op,
                            value: p.value.clone(),
                        }
                    }
                    SporeKind::Temporal => HintKind::Temporal {
                        marker: schema.temporal_markers.get(act.class)?.clone(),
                    },
                    SporeKind::Op => HintKind::Op {
                        op: schema.op_phrases.get(act.class)?.op,
                    },
                };

                Some(Hint {
                    kind,
                    span: Span {
                        range: act.range,
                        text,
                    },
                    confidence: act.score,
                })
            })
            .collect()
    }
}

/// Numeric literals don't need a model — deterministic scan.
pub struct NumericSpore;

impl Spore for NumericSpore {
    fn scan(&self, input: &str, _schema: &Schema) -> Vec<Hint> {
        let mut hints = Vec::new();
        let bytes = input.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            let start = if bytes[i].is_ascii_digit()
                || (bytes[i] == b'-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit())
            {
                i
            } else {
                i += 1;
                continue;
            };

            let mut end = start + 1;
            let mut has_dot = false;
            while end < bytes.len() {
                if bytes[end].is_ascii_digit() {
                    end += 1;
                } else if bytes[end] == b'.' && !has_dot {
                    has_dot = true;
                    end += 1;
                } else {
                    break;
                }
            }

            if let Ok(value) = input[start..end].parse::<f64>() {
                hints.push(Hint {
                    kind: HintKind::Numeric { value },
                    span: Span {
                        range: start..end,
                        text: input[start..end].to_string(),
                    },
                    confidence: 1.0,
                });
            }

            i = end;
        }

        hints
    }
}

/// Collect hints from all spores into a single annotated input.
pub fn gather(input: &str, schema: &Schema, spores: &[&dyn Spore]) -> AnnotatedInput {
    let mut hints = Vec::new();
    for spore in spores {
        hints.extend(spore.scan(input, schema));
    }
    hints.sort_by_key(|h| h.span.range.start);
    AnnotatedInput {
        original: input.to_string(),
        hints,
    }
}
