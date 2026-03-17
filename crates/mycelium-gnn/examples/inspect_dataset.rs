//! Inspect dataset distribution: candidate counts, confidence ranges, ambiguity stats.
//!
//! Usage:
//!   cargo run --example inspect_dataset

use std::path::Path;
use gnn_burn::training::Dataset;

fn main() {
    let demo_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo");
    let dataset = Dataset::load(&demo_dir.join("dataset.json")).expect("load dataset");
    let n = dataset.samples.len();
    println!("=== Dataset: {} samples ===\n", n);

    // --- Pattern distribution (by which candidate types are present) ---
    let mut patterns: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut n_collections = 0usize;
    let mut n_fields = 0usize;
    let mut n_filters = 0usize;
    let mut n_traversals = 0usize;
    let mut n_modifiers = 0usize;

    // Confidence distributions
    let mut coll_confs = Vec::new();
    let mut field_confs = Vec::new();
    let mut filter_confs = Vec::new();
    let mut trav_confs = Vec::new();
    let mut mod_confs = Vec::new();

    // Ambiguity: how often correct match is NOT the highest-scored
    let mut field_ambiguous = 0usize;
    let mut field_total_scored = 0usize;
    let mut coll_ambiguous = 0usize;
    let mut coll_total_scored = 0usize;

    // Distractor counts
    let mut field_distractor_counts = Vec::new();
    let mut coll_distractor_counts = Vec::new();

    // Schema coverage
    let mut table_hits: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut field_hits: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    for sample in &dataset.samples {
        let ext = &sample.extraction;
        let gt = &sample.ground_truth;

        // Pattern key
        let mut parts = Vec::new();
        if !ext.collections.is_empty() { parts.push("coll"); }
        if !ext.fields.is_empty() { parts.push("field"); }
        if !ext.filters.is_empty() { parts.push("filter"); }
        if !ext.traversals.is_empty() { parts.push("trav"); }
        if !ext.modifiers.is_empty() { parts.push("mod"); }
        let key = parts.join("+");
        *patterns.entry(key).or_default() += 1;

        // Counts
        n_collections += ext.collections.len();
        n_fields += ext.fields.len();
        n_filters += ext.filters.len();
        n_traversals += ext.traversals.len();
        n_modifiers += ext.modifiers.len();

        // Collection confidence + ambiguity
        for (i, coll) in ext.collections.iter().enumerate() {
            coll_confs.push(coll.confidence);
            let target = gt.collection_targets.get(i).copied();
            if let Some(target_id) = target {
                *table_hits.entry(target_id).or_default() += 1;
                let n_matches = coll.schema_matches.len();
                coll_distractor_counts.push(n_matches.saturating_sub(1));
                coll_total_scored += 1;
                // Check if correct match has highest score
                if let Some(correct) = coll.schema_matches.iter()
                    .find(|m| m.schema_node_id == target_id)
                {
                    let max_score = coll.schema_matches.iter()
                        .map(|m| m.score)
                        .fold(0.0f32, f32::max);
                    if correct.score < max_score - 0.001 {
                        coll_ambiguous += 1;
                    }
                }
            }
        }

        // Field confidence + ambiguity
        for (i, f) in ext.fields.iter().enumerate() {
            field_confs.push(f.confidence);
            let target = gt.field_targets.get(i).copied();
            if let Some(target_id) = target {
                *field_hits.entry(target_id).or_default() += 1;
                let n_matches = f.schema_matches.len();
                field_distractor_counts.push(n_matches.saturating_sub(1));
                field_total_scored += 1;
                if let Some(correct) = f.schema_matches.iter()
                    .find(|m| m.schema_node_id == target_id)
                {
                    let max_score = f.schema_matches.iter()
                        .map(|m| m.score)
                        .fold(0.0f32, f32::max);
                    if correct.score < max_score - 0.001 {
                        field_ambiguous += 1;
                    }
                }
            }
        }

        // Filter confidence
        for f in &ext.filters {
            filter_confs.push(f.confidence);
        }
        for t in &ext.traversals {
            trav_confs.push(t.confidence);
        }
        for m in &ext.modifiers {
            mod_confs.push(m.confidence);
        }
    }

    // --- Print pattern distribution ---
    println!("Pattern distribution:");
    let mut sorted_patterns: Vec<_> = patterns.iter().collect();
    sorted_patterns.sort_by(|a, b| b.1.cmp(a.1));
    for (pattern, count) in &sorted_patterns {
        let pct = *count as f64 / n as f64 * 100.0;
        let bar = "#".repeat((*count * 40 / n).max(1));
        println!("  {:<30} {:>5} ({:>5.1}%) {}", pattern, count, pct, bar);
    }

    // --- Candidate counts ---
    println!("\nCandidate totals:");
    println!("  collections: {} ({:.1}/sample)", n_collections, n_collections as f64 / n as f64);
    println!("  fields:      {} ({:.1}/sample)", n_fields, n_fields as f64 / n as f64);
    println!("  filters:     {} ({:.1}/sample)", n_filters, n_filters as f64 / n as f64);
    println!("  traversals:  {} ({:.1}/sample)", n_traversals, n_traversals as f64 / n as f64);
    println!("  modifiers:   {} ({:.1}/sample)", n_modifiers, n_modifiers as f64 / n as f64);

    // --- Confidence distributions ---
    println!("\nConfidence distributions (min / p25 / median / p75 / max):");
    print_dist("  collections", &mut coll_confs);
    print_dist("  fields     ", &mut field_confs);
    print_dist("  filters    ", &mut filter_confs);
    print_dist("  traversals ", &mut trav_confs);
    print_dist("  modifiers  ", &mut mod_confs);

    // --- Ambiguity ---
    println!("\nAmbiguity (correct match NOT highest scored):");
    if coll_total_scored > 0 {
        println!("  collections: {}/{} ({:.1}%)",
            coll_ambiguous, coll_total_scored,
            coll_ambiguous as f64 / coll_total_scored as f64 * 100.0);
    }
    if field_total_scored > 0 {
        println!("  fields:      {}/{} ({:.1}%)",
            field_ambiguous, field_total_scored,
            field_ambiguous as f64 / field_total_scored as f64 * 100.0);
    }

    // --- Distractor counts ---
    println!("\nDistractor counts per candidate (min / p25 / median / p75 / max):");
    let mut cd: Vec<f32> = coll_distractor_counts.iter().map(|&x| x as f32).collect();
    let mut fd: Vec<f32> = field_distractor_counts.iter().map(|&x| x as f32).collect();
    print_dist("  collections", &mut cd);
    print_dist("  fields     ", &mut fd);

    // --- Schema coverage ---
    println!("\nTable coverage (target hit counts):");
    let mut table_sorted: Vec<_> = table_hits.iter().collect();
    table_sorted.sort_by_key(|(&id, _)| id);
    for (&id, &count) in &table_sorted {
        let bar = "#".repeat((count * 30 / n).max(1));
        println!("  table {:>2}: {:>5} {}", id, count, bar);
    }

    let covered_fields = field_hits.len();
    let max_field_hits = field_hits.values().max().copied().unwrap_or(0);
    let min_field_hits = field_hits.values().min().copied().unwrap_or(0);
    println!("\nField coverage: {}/{} fields used (hits: {}..{})",
        covered_fields, covered_fields + field_hits.values().filter(|&&v| v == 0).count(),
        min_field_hits, max_field_hits);
}

fn print_dist(label: &str, values: &mut Vec<f32>) {
    if values.is_empty() {
        println!("{}: (none)", label);
        return;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    let min = values[0];
    let p25 = values[n / 4];
    let median = values[n / 2];
    let p75 = values[3 * n / 4];
    let max = values[n - 1];
    println!("{}: {:.3} / {:.3} / {:.3} / {:.3} / {:.3}  (n={})",
        label, min, p25, median, p75, max, n);
}
