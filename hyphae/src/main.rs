use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Commands {
    #[command(subcommand)]
    script: Script,
}
#[derive(Subcommand, Debug, Clone, Copy)]
enum Script {
    Train,
    Inference,
    Evaluate,
    DataStats
}
fn main() {
    let args = Commands::parse();
    match args.script {
        Script::Train => {
            println!("Training...");
        }
        Script::Inference => {
            println!("Inference...");
        }
        Script::Evaluate => {
            println!("Evaluating...");
        }
        Script::DataStats => {
            println!("Data stats...");
        }
    }
}