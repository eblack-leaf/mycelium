/// Train a namer profile checkpoint.
///
/// Usage:
///   cargo run -p hyphae --bin train -- <profile> <data_dir>
///
/// Arguments:
///   profile   — classic | terse | creative | hacker | custom:<name>
///   data_dir  — root directory containing profiles/<name>/train.jsonl
///               also where the checkpoint will be written
///
/// Example:
///   cargo run -p hyphae --bin train --release -- classic ./data

use std::path::PathBuf;

use hyphae::namer::{
    profiles::Profile,
    train::{train, TrainConfig},
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: train <profile> <data_dir>");
        eprintln!("  profile:  classic | terse | creative | hacker | custom:<name>");
        eprintln!("  data_dir: directory containing profiles/<name>/train.jsonl");
        std::process::exit(1);
    }

    let profile = parse_profile(&args[1]);
    let base = PathBuf::from(&args[2]);

    let data_path = profile.training_data_path(&base);
    if !data_path.exists() {
        eprintln!("error: training data not found at {:?}", data_path);
        eprintln!("expected JSONL with lines: {{\"value\": \"...\", \"name\": \"...\"}}");
        std::process::exit(1);
    }

    let cfg = TrainConfig::default();
    train(&profile, &base, cfg);
}

fn parse_profile(s: &str) -> Profile {
    match s {
        "classic"  => Profile::Classic,
        "terse"    => Profile::Terse,
        "creative" => Profile::Creative,
        "hacker"   => Profile::Hacker,
        other if other.starts_with("custom:") => {
            Profile::Custom(other["custom:".len()..].to_string())
        }
        other => {
            eprintln!("unknown profile '{}', treating as custom", other);
            Profile::Custom(other.to_string())
        }
    }
}
