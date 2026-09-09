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
mod reference;
mod resume;
mod scan;
mod search_space;
mod trial_result;

// ─── Main ─────────────────────────────────────────────────────────────────────

/// The four real datasets, in thesis-table order.
const REAL_DATASETS: [&str; 4] = ["mnist", "fashion_mnist", "pbmc", "wordnet_mammals"];

/// Every synthetic dataset `Dataset::load_synthetic` accepts.
///
/// The first five are the original suite and are unchanged, so the results
/// already under `results/` remain valid for them. `tree_graph` and the
/// `ball*` family were added for the geometry-matching redesign: the balls are
/// one sampling scheme mapped into three geometries, so within a tier they
/// differ in curvature and nothing else.
///
/// `crates/analysis/src/cell.rs::SYNTH_TRUTH` must carry a truth for every name
/// here, or Experiment 1 silently drops the dataset; a test there pins it.
const SYNTHETIC_DATASETS: [&str; 12] = [
    "sphere",
    "antipodal_clusters",
    "tree",
    "hyperbolic_shells",
    "grid",
    "tree_graph",
    "ball2_euclidean",
    "ball2_spherical",
    "ball2_hyperbolic",
    "ball9_euclidean",
    "ball9_spherical",
    "ball9_hyperbolic",
];

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
            "Starting qParEGO multi-objective optimisation: {} datasets × {} trials, 6 objectives, geometry={}, seeds={}",
            dataset_names.len(),
            args.n_trials,
            args.geometry.as_deref().unwrap_or("auto-detect"),
            args.n_seeds
        ),
        "reference" => println!(
            "Scoring the ground-truth source configuration of {} dataset(s) through the trial metric pipeline (no embedding is fitted)",
            dataset_names.len()
        ),
        "detect" => println!(
            "Starting curvature detection: {} datasets, exporting κ_data diagnostics (no embedding fit).",
            dataset_names.len(),
        ),
        other => {
            eprintln!(
                "Unknown --mode '{other}'. Use 'random', 'scan', 'bayes', 'pareto', 'detect', or 'reference'."
            );
            std::process::exit(1);
        }
    }
    println!("Output file: {}", args.output);
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
    // For `bayes`, each dataset job additionally spawns n_threads parallel evaluators
    // per batch round — so with a single dataset all cores stay busy.
    // With multiple datasets the outer and inner parallelism combine; on a typical
    // single-dataset run this is always just n_threads total threads.
    let n_outer = n_threads.min(work.len().max(1));
    println!(
        "Using {n_threads} thread(s) ({n_outer} outer dataset worker(s), batch_size={n_threads} for bayes/pareto).",
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
                Some((dataset_name, evaluator)) => match args.mode.as_str() {
                    "scan" => run_scan(&dataset_name, &args, &evaluator, &mp),
                    "bayes" => run_bayes(&dataset_name, &args, &evaluator, &mp, n_threads),
                    "pareto" => run_pareto(&dataset_name, &args, &evaluator, &mp, n_threads),
                    "detect" => run_detect(&dataset_name, &args, &evaluator),
                    "reference" => {
                        crate::reference::run_reference(&dataset_name, &args, &evaluator);
                    }
                    _ => run_random(&dataset_name, &args, &evaluator, &mp),
                },
            }
        });
        handles.push(h);
    }

    for h in handles {
        h.join().expect("optimizer thread panicked");
    }
}
