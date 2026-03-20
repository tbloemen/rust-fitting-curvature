use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::data::Dataset;
use crate::evaluate::Evaluator;
use crate::search_space::{OptimizeDirection, SearchSpace};
use crate::tpe::TpeOptimizer;

mod data;
mod evaluate;
mod search_space;
mod tpe;

#[derive(Parser, Debug)]
#[command(name = "fitting-optimizer")]
#[command(about = "Bayesian hyperparameter optimizer for fitting-curvature")]
struct Args {
    #[arg(long, default_value = "./www/public/data")]
    data_path: String,

    #[arg(long)]
    metric: String,

    #[arg(long, default_value = "100")]
    n_trials: usize,

    #[arg(long, default_value = "3")]
    n_seeds: usize,

    #[arg(long, default_value = "5000")]
    n_samples: usize,

    #[arg(long, default_value = "results.jsonl")]
    output: String,

    #[arg(long)]
    dataset: Option<String>,
}

fn metric_direction(name: &str) -> OptimizeDirection {
    match name {
        "davies_bouldin" | "geodesic_distortion_gu2019" | "geodesic_distortion_mse" => {
            OptimizeDirection::Minimize
        }
        "trustworthiness"
        | "continuity"
        | "knn_overlap"
        | "davies_bouldin_ratio"
        | "dunn_index"
        | "class_density_measure"
        | "cluster_density_measure" => OptimizeDirection::Maximize,
        _ => panic!("Unknown metric: {}. See --help for options.", name),
    }
}

#[derive(Debug, Serialize)]
struct TrialResult {
    trial: usize,
    learning_rate: f64,
    perplexity: f64,
    momentum_main: f64,
    n_iterations: i64,
    early_exaggeration_iterations: i64,
    curvature: f64,
    metric_mean: f64,
    metric_std: f64,
    time_ms: u64,
}

fn main() {
    let args = Args::parse();

    let direction = metric_direction(&args.metric);
    let mut space = SearchSpace::default_tsne();
    space.direction = direction;

    println!("Loading dataset...");
    let dataset = if let Some(name) = &args.dataset {
        Dataset::load_synthetic(name, args.n_samples, 42)
    } else {
        match Dataset::load_mnist(&args.data_path, args.n_samples) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error loading MNIST: {}", e);
                eprintln!("Make sure you have MNIST files in: {}", args.data_path);
                eprintln!("Expected files: train-images-idx3-ubyte, train-labels-idx1-ubyte");
                std::process::exit(1);
            }
        }
    };
    println!(
        "Loaded {} samples with {} features",
        dataset.n_points, dataset.n_features
    );

    let evaluator = Evaluator::new(dataset.clone());
    let mut optimizer = TpeOptimizer::new(space);
    let mut rng = fitting_core::synthetic_data::Rng::new(42);

    if Path::new(&args.output).exists() {
        std::fs::remove_file(&args.output).ok();
    }

    println!(
        "Starting optimization: {} trials, metric={} ({}), seeds={}",
        args.n_trials, args.metric, direction, args.n_seeds
    );

    let mp = MultiProgress::new();
    let trial_style = ProgressStyle::with_template(
        "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} trials | {per_sec} | eta {eta} | best: {msg}",
    )
    .unwrap()
    .progress_chars("=>-");
    let seed_style = ProgressStyle::with_template(
        "  {spinner:.yellow} seed {pos}/{len}",
    )
    .unwrap();
    let iter_style = ProgressStyle::with_template(
        "    {spinner:.blue} iter {pos}/{len} | {per_sec}",
    )
    .unwrap();

    let pb_trials = mp.add(ProgressBar::new(args.n_trials as u64));
    pb_trials.set_style(trial_style);
    pb_trials.set_message("n/a");

    let pb_seeds = mp.add(ProgressBar::new(args.n_seeds as u64));
    pb_seeds.set_style(seed_style);

    let pb_iters = mp.add(ProgressBar::new(0));
    pb_iters.set_style(iter_style);

    for trial_idx in 1..=args.n_trials {
        let start = std::time::Instant::now();

        let config = optimizer.suggest(&mut rng);

        pb_seeds.reset();
        let mut metrics = Vec::with_capacity(args.n_seeds);
        for seed_idx in 0..args.n_seeds {
            let seed = 42 + trial_idx as u64 * 100 + seed_idx as u64;
            let m = evaluator.evaluate_with_metric(&config, &args.metric, seed, &pb_iters);
            metrics.push(m);
            pb_seeds.inc(1);
        }

        let mean = metrics.iter().sum::<f64>() / args.n_seeds as f64;
        let variance =
            metrics.iter().map(|&m| (m - mean).powi(2)).sum::<f64>() / args.n_seeds as f64;
        let std = variance.sqrt();

        optimizer.observe(config.clone(), mean);

        let elapsed = start.elapsed().as_millis() as u64;

        let result = TrialResult {
            trial: trial_idx,
            learning_rate: config.learning_rate,
            perplexity: config.perplexity,
            momentum_main: config.momentum_main,
            n_iterations: config.n_iterations,
            early_exaggeration_iterations: config.early_exaggeration_iterations,
            curvature: config.curvature,
            metric_mean: mean,
            metric_std: std,
            time_ms: elapsed,
        };

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.output)
            .unwrap();
        let json = serde_json::to_string(&result).unwrap();
        writeln!(file, "{}", json).ok();

        let best = optimizer.best_trial();
        pb_trials.set_message(format!("{:.4}", best));
        pb_trials.println(format!(
            "Trial {:3} | {} = {:.4} ± {:.4} | best = {:.4} | {}ms | lr={:.2}, perp={:.1}, k={:.1}",
            trial_idx, args.metric, mean, std, best, elapsed,
            config.learning_rate, config.perplexity, config.curvature,
        ));
        pb_trials.inc(1);
    }

    pb_iters.finish_and_clear();
    pb_seeds.finish_and_clear();
    pb_trials.finish_and_clear();

    println!("\n=== Best Configuration ===");
    if let Some(best_config) = optimizer.best_config() {
        println!("learning_rate: {:.4}", best_config.learning_rate);
        println!("perplexity: {:.4}", best_config.perplexity);
        println!("momentum_main: {:.4}", best_config.momentum_main);
        println!("init_scale: {:.6} (auto)", fitting_core::matrices::get_default_init_scale(2));
        println!("n_iterations: {}", best_config.n_iterations);
        println!(
            "early_exaggeration_iterations: {}",
            best_config.early_exaggeration_iterations
        );
        println!("curvature: {:.2}", best_config.curvature);
        println!("init_method: Pca (fixed)");
        println!("Best {}: {:.4}", args.metric, optimizer.best_trial());
    }

    println!("\nResults saved to: {}", args.output);
}
