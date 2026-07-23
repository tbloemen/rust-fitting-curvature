use clap::Parser;
use indicatif::MultiProgress;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::bayes::run_bayes;
use crate::cli::Args;
use crate::data::Dataset;
use crate::detect::run_detect;
use crate::evaluate::Evaluator;
use crate::pareto::run_pareto;
use crate::random::run_random;
use crate::scan::run_scan;

mod bayes;
mod cli;
mod common;
mod data;
mod detect;
mod evaluate;
mod gp;
mod metrics;
mod pareto;
mod random;
mod resume;
mod scan;
mod search_space;
mod trial_result;

// ─── Main ─────────────────────────────────────────────────────────────────────

fn get_dataset_names(dataset_arg: &Option<String>) -> Vec<String> {
    match dataset_arg {
        Some(name) if name == "all" => vec![
            "mnist".to_string(),
            "fashion_mnist".to_string(),
            "pbmc".to_string(),
            "wordnet_mammals".to_string(),
            "sphere".to_string(),
            "antipodal_clusters".to_string(),
            "tree".to_string(),
            "hyperbolic_shells".to_string(),
        ],
        Some(name) if name == "real" => vec![
            "mnist".to_string(),
            "fashion_mnist".to_string(),
            "pbmc".to_string(),
            "wordnet_mammals".to_string(),
        ],
        Some(name) => vec![name.clone()],
        None => vec!["mnist".to_string()],
    }
}

fn main() {
    let args = Args::parse();

    if (args.mode == "scan" || args.mode == "bayes") && args.metric.is_none() {
        eprintln!("Error: --metric is required for --mode {}", args.mode);
        std::process::exit(1);
    }
    if args.mode == "pareto" && args.metric.is_some() {
        eprintln!("Note: --metric is ignored for --mode pareto (optimises all objectives).");
    }

    if let Some(parent) = Path::new(&args.output).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }

    let dataset_names = get_dataset_names(&args.dataset);

    match args.mode.as_str() {
        "random" => println!(
            "Starting random search: {} datasets × {} trials, curvature=[{},{}], all metrics, seeds={}",
            dataset_names.len(),
            args.n_trials,
            args.curvature_min,
            args.curvature_max,
            args.n_seeds
        ),
        "scan" => println!(
            "Starting scan: {} datasets × ~{} sweep points, metric={}, geometry={}, seeds={}",
            dataset_names.len(),
            args.scan_steps * 7,
            args.metric.as_deref().unwrap(),
            args.geometry.as_deref().unwrap_or("auto-detect"),
            args.n_seeds
        ),
        "bayes" => println!(
            "Starting Bayesian optimisation: {} datasets × {} trials, metric={}, geometry={}, seeds={}",
            dataset_names.len(),
            args.n_trials,
            args.metric.as_deref().unwrap(),
            args.geometry.as_deref().unwrap_or("auto-detect"),
            args.n_seeds
        ),
        "pareto" => println!(
            "Starting qParEGO multi-objective optimisation: {} datasets × {} trials, 10 objectives, geometry={}, seeds={}",
            dataset_names.len(),
            args.n_trials,
            args.geometry.as_deref().unwrap_or("auto-detect"),
            args.n_seeds
        ),
        "detect" => println!(
            "Starting curvature detection: {} datasets, exporting κ_data diagnostics (no embedding fit).",
            dataset_names.len(),
        ),
        other => {
            eprintln!(
                "Unknown --mode '{}'. Use 'random', 'scan', 'bayes', 'pareto', or 'detect'.",
                other
            );
            std::process::exit(1);
        }
    }
    println!("Output file: {}", args.output);

    let mp = Arc::new(MultiProgress::new());

    // Build unified per-dataset work queue.
    let mut work: VecDeque<(String, Arc<Evaluator>)> = VecDeque::new();
    for dataset_name in &dataset_names {
        println!("Loading dataset: {}...", dataset_name);
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

    let n_threads = args
        .threads
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1);

    // The outer pool processes datasets in parallel (up to n_threads datasets at once).
    // For `bayes`, each dataset job additionally spawns n_threads parallel evaluators
    // per batch round — so with a single dataset all cores stay busy.
    // With multiple datasets the outer and inner parallelism combine; on a typical
    // single-dataset run this is always just n_threads total threads.
    let n_outer = n_threads.min(work.len().max(1));
    println!(
        "Using {} thread(s) ({} outer dataset worker(s), batch_size={} for bayes/pareto).",
        n_threads, n_outer, n_threads,
    );

    let queue = Arc::new(Mutex::new(work));
    let mut handles = Vec::new();

    for _ in 0..n_outer {
        let queue = Arc::clone(&queue);
        let args = args.clone();
        let mp = Arc::clone(&mp);
        let h = thread::spawn(move || loop {
            let item = queue.lock().unwrap().pop_front();
            match item {
                None => break,
                Some((dataset_name, evaluator)) => match args.mode.as_str() {
                    "scan" => run_scan(&dataset_name, &args, evaluator, &mp),
                    "bayes" => run_bayes(&dataset_name, &args, evaluator, &mp, n_threads),
                    "pareto" => run_pareto(&dataset_name, &args, evaluator, &mp, n_threads),
                    "detect" => run_detect(&dataset_name, &args, &evaluator),
                    _ => run_random(&dataset_name, &args, evaluator, &mp),
                },
            }
        });
        handles.push(h);
    }

    for h in handles {
        h.join().expect("optimizer thread panicked");
    }

    println!("\nAll sessions complete.");
}
