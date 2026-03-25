use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::thread;

use crate::data::Dataset;
use crate::evaluate::Evaluator;
use crate::gp::GpOptimizer;
use crate::search_space::{OptimizeDirection, SearchSpace};

mod data;
mod evaluate;
mod gp;
mod search_space;

#[derive(Parser, Debug, Clone)]
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

    /// Output file prefix. Each curvature session writes to <output>_k<curvature>.jsonl.
    #[arg(long, default_value = "results")]
    output: String,

    #[arg(long)]
    dataset: Option<String>,

    /// Fixed curvature values to optimise over (comma-separated).
    /// Each value gets its own independent optimisation session run in parallel.
    #[arg(
        long,
        num_args = 1..,
        value_delimiter = ',',
        default_values = ["-2", "-1", "-0.5", "-0.1", "0", "0.1", "0.5", "1", "2"]
    )]
    curvatures: Vec<f64>,
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
    curvature: f64,
    trial: usize,
    learning_rate: f64,
    perplexity: f64,
    momentum_main: f64,
    n_iterations: i64,
    early_exaggeration_iterations: i64,
    scaling_loss: u8,
    centering_weight: f64,
    global_loss_weight: f64,
    norm_loss_weight: f64,
    metric_mean: f64,
    metric_std: f64,
    time_ms: u64,
}

fn output_path(prefix: &str, curvature: f64) -> String {
    format!("{}_k{:.1}.jsonl", prefix, curvature)
}

fn run_session(curvature: f64, args: &Args, evaluator: Arc<Evaluator>, mp: &MultiProgress) {
    let direction = metric_direction(&args.metric);
    let space = SearchSpace { direction };
    let mut optimizer = GpOptimizer::new(space);
    // Each curvature session gets a distinct seed derived from the curvature bits.
    let mut rng =
        fitting_core::synthetic_data::Rng::new(curvature.to_bits() ^ 0xdead_beef_cafe_0000);

    let out_path = output_path(&args.output, curvature);
    if Path::new(&out_path).exists() {
        std::fs::remove_file(&out_path).ok();
    }

    let trial_style = ProgressStyle::with_template(
        "{spinner:.green} k={msg} [{bar:35.cyan/blue}] {pos}/{len} | best: {prefix}",
    )
    .unwrap()
    .progress_chars("=>-");

    let pb = mp.add(ProgressBar::new(args.n_trials as u64));
    pb.set_style(trial_style);
    pb.set_message(format!("{:+.1}", curvature));
    pb.set_prefix("n/a");

    // Hidden bar passed into evaluate — we don't render per-iter bars in parallel mode.
    let pb_iters = ProgressBar::hidden();

    for trial_idx in 1..=args.n_trials {
        let start = std::time::Instant::now();
        let config = optimizer.suggest(&mut rng);

        let mut metrics = Vec::with_capacity(args.n_seeds);
        for seed_idx in 0..args.n_seeds {
            let seed = 42 + trial_idx as u64 * 100 + seed_idx as u64;
            let m = evaluator.evaluate_with_metric(
                &config,
                curvature,
                &args.metric,
                seed,
                &pb_iters,
            );
            metrics.push(m);
        }

        let mean = metrics.iter().sum::<f64>() / args.n_seeds as f64;
        let variance =
            metrics.iter().map(|&m| (m - mean).powi(2)).sum::<f64>() / args.n_seeds as f64;
        let std = variance.sqrt();

        optimizer.observe(config.clone(), mean);
        let elapsed = start.elapsed().as_millis() as u64;

        let result = TrialResult {
            curvature,
            trial: trial_idx,
            learning_rate: config.learning_rate,
            perplexity: config.perplexity,
            momentum_main: config.momentum_main,
            n_iterations: config.n_iterations,
            early_exaggeration_iterations: config.early_exaggeration_iterations,
            scaling_loss: config.scaling_loss,
            centering_weight: config.centering_weight,
            global_loss_weight: config.global_loss_weight,
            norm_loss_weight: config.norm_loss_weight,
            metric_mean: mean,
            metric_std: std,
            time_ms: elapsed,
        };

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out_path)
            .unwrap();
        let json = serde_json::to_string(&result).unwrap();
        writeln!(file, "{}", json).ok();

        let best = optimizer.best_trial();
        pb.set_prefix(format!("{:.4}", best));
        pb.println(format!(
            "k={:+.1} trial {:3} | {} = {:.4} ± {:.4} | best={:.4} | {}ms \
             | lr={:.4} perp={:.2} sl={} cw={:.3} glw={:.2} nw={:.4}",
            curvature,
            trial_idx,
            args.metric,
            mean,
            std,
            best,
            elapsed,
            config.learning_rate,
            config.perplexity,
            config.scaling_loss,
            config.centering_weight,
            config.global_loss_weight,
            config.norm_loss_weight,
        ));
        pb.inc(1);
    }

    pb.finish_with_message(format!("{:+.1} done", curvature));

    if let Some(best_config) = optimizer.best_config() {
        pb.println(format!(
            "\n=== Best for k={:+.1} | {}={:.4} ===\n  \
             lr={:.4}  perp={:.4}  momentum={:.4}\n  \
             n_iter={}  ee_iter={}\n  \
             scaling_loss={}  centering={:.3}  global_loss={:.3}  norm={:.4}",
            curvature,
            args.metric,
            optimizer.best_trial(),
            best_config.learning_rate,
            best_config.perplexity,
            best_config.momentum_main,
            best_config.n_iterations,
            best_config.early_exaggeration_iterations,
            best_config.scaling_loss,
            best_config.centering_weight,
            best_config.global_loss_weight,
            best_config.norm_loss_weight,
        ));
    }
}

fn main() {
    let args = Args::parse();

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

    let direction = metric_direction(&args.metric);
    let evaluator = Arc::new(Evaluator::new(dataset));

    println!(
        "Starting optimization: {} curvature sessions × {} trials, metric={} ({}), seeds={}",
        args.curvatures.len(),
        args.n_trials,
        args.metric,
        direction,
        args.n_seeds
    );
    println!("Curvatures: {:?}", args.curvatures);
    println!("Output prefix: {}", args.output);

    let mp = Arc::new(MultiProgress::new());

    let handles: Vec<_> = args
        .curvatures
        .iter()
        .map(|&k| {
            let args = args.clone();
            let evaluator = Arc::clone(&evaluator);
            let mp = Arc::clone(&mp);
            thread::spawn(move || {
                run_session(k, &args, evaluator, &mp);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("optimizer thread panicked");
    }

    println!(
        "\nAll sessions complete. Results written to {}_k<curvature>.jsonl",
        args.output
    );
}
