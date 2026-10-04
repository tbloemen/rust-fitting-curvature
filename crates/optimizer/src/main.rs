use clap::Parser;
use indicatif::MultiProgress;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::cli::Args;
use crate::data::Dataset;
use crate::evaluate::Evaluator;
use crate::pareto::run_pareto;

mod cli;
mod common;
mod data;
mod evaluate;
mod gp;
mod metrics;
mod pareto;
mod resume;
mod search_space;
mod trial_result;

// ─── Main ─────────────────────────────────────────────────────────────────────

/// The four real datasets, in thesis-table order.
const REAL_DATASETS: [&str; 4] = ["mnist", "fashion_mnist", "pbmc", "wordnet_mammals"];

/// Every synthetic dataset `Dataset::load_synthetic` accepts.
///
/// The synthetic suite the thesis reports.
///
/// `crates/analysis/src/cell.rs::SYNTH_TRUTH` must carry a truth for every name
/// here, or Experiment 1 silently drops the dataset; a test there pins it.
const SYNTHETIC_DATASETS: [&str; 4] = ["sphere", "tree", "hyperbolic_shells", "grid"];

fn get_dataset_names(dataset_arg: Option<&str>) -> Vec<String> {
    match dataset_arg {
        Some("all") => REAL_DATASETS
            .iter()
            .chain(SYNTHETIC_DATASETS.iter())
            .map(|s| (*s).to_string())
            .collect(),
        Some("real") => REAL_DATASETS.iter().map(|s| (*s).to_string()).collect(),
        Some("synthetic") => SYNTHETIC_DATASETS
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        Some(name) => vec![name.to_string()],
        None => vec!["mnist".to_string()],
    }
}

fn print_mode_banner(args: &Args, dataset_names: &[String]) {
    match args.mode.as_str() {
        "pareto" => println!(
            "Starting qParEGO multi-objective optimisation: {} datasets × {} trials, 6 objectives, geometry={}, seeds={}",
            dataset_names.len(),
            args.n_trials,
            args.geometry,
            args.n_seeds
        ),
        other => {
            eprintln!(
                "Unknown --mode '{other}'. The only mode is 'pareto'."
            );
            std::process::exit(1);
        }
    }
    println!("Output file: {}", args.output);
}

fn main() {
    let args = Args::parse();

    if let Some(parent) = Path::new(&args.output).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }

    let dataset_names = get_dataset_names(args.dataset.as_deref());
    print_mode_banner(&args, &dataset_names);

    let mp = Arc::new(MultiProgress::new());
    let work = build_work_queue(&args, &dataset_names);
    spawn_workers(&args, &mp, work);
    println!("\nAll sessions complete.");
}

fn build_work_queue(args: &Args, dataset_names: &[String]) -> VecDeque<(String, Arc<Evaluator>)> {
    let mut work: VecDeque<(String, Arc<Evaluator>)> = VecDeque::new();
    for dataset_name in dataset_names {
        println!("Loading dataset: {dataset_name}...");
        let dp = &args.data_path;
        let n = args.n_samples;
        let result: Result<Dataset, String> = match dataset_name.as_str() {
            "mnist" => Dataset::load_mnist(&format!("{dp}/mnist"), n),
            "fashion_mnist" => Dataset::load_fashion_mnist(&format!("{dp}/fashion-mnist"), n),
            "wordnet_mammals" => Dataset::load_wordnet_mammals(&format!("{dp}/wordnet"), n),
            "pbmc" => Dataset::load_pbmc(&format!("{dp}/pbmc"), n),
            name => Dataset::load_synthetic(name, n, 42),
        };
        let dataset = match result {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error loading dataset '{dataset_name}': {e}");
                std::process::exit(1);
            }
        };
        println!(
            "Loaded {} samples with {} features",
            dataset.n_points, dataset.n_features
        );
        let evaluator = Arc::new(Evaluator::new(dataset));
        work.push_back((dataset_name.clone(), evaluator));
    }
    work
}

fn spawn_workers(args: &Args, mp: &Arc<MultiProgress>, work: VecDeque<(String, Arc<Evaluator>)>) {
    let n_threads = args
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
        .max(1);

    // The outer pool processes datasets in parallel (up to n_threads datasets at once).
    // For `pareto`, each dataset job additionally spawns n_threads parallel evaluators
    // per batch round — so with a single dataset all cores stay busy.
    // With multiple datasets the outer and inner parallelism combine; on a typical
    // single-dataset run this is always just n_threads total threads.
    let n_outer = n_threads.min(work.len().max(1));
    println!(
        "Using {n_threads} thread(s) ({n_outer} outer dataset worker(s), batch_size={n_threads} for pareto).",
    );

    let queue = Arc::new(Mutex::new(work));
    let mut handles = Vec::new();

    for _ in 0..n_outer {
        let queue = Arc::clone(&queue);
        let args = args.clone();
        let mp = Arc::clone(mp);
        let h = thread::spawn(move || loop {
            let item = queue.lock().unwrap().pop_front();
            match item {
                None => break,
                Some((dataset_name, evaluator)) => {
                    run_pareto(&dataset_name, &args, &evaluator, &mp, n_threads);
                }
            }
        });
        handles.push(h);
    }

    for h in handles {
        h.join().expect("optimizer thread panicked");
    }
}
