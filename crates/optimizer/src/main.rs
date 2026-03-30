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
use crate::search_space::{
    FIXED_EARLY_EXAG_ITERATIONS, FIXED_N_ITERATIONS, OptimizeDirection, SearchSpace, TrialConfig,
};

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

    /// Run mode: "optimize" (default) or "scan".
    /// scan: sweep each parameter individually from a base config and record effects.
    #[arg(long, default_value = "optimize")]
    mode: String,

    /// For --mode scan: path prefix of a previous optimization run to load the base
    /// config from (best trial per curvature). Uses the same prefix format as --output.
    #[arg(long)]
    scan_from: Option<String>,

    /// For --mode scan: number of evenly-spaced values to sweep per continuous parameter.
    #[arg(long, default_value = "12")]
    scan_steps: usize,
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
    metric_name: String,
    curvature: f64,
    trial: usize,
    learning_rate: f64,
    perplexity_ratio: f64,
    momentum_main: f64,
    scaling_loss: u8,
    centering_weight: f64,
    global_loss_weight: f64,
    norm_loss_weight: f64,
    metric_mean: f64,
    metric_std: f64,
    time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    scan_param: Option<String>,
}

fn output_path(prefix: &str, curvature: f64) -> String {
    format!("{}_k{:.1}.jsonl", prefix, curvature)
}

fn write_result(result: &TrialResult, out_path: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)
        .unwrap();
    let json = serde_json::to_string(result).unwrap();
    writeln!(file, "{}", json).ok();
}

fn run_session(curvature: f64, args: &Args, evaluator: Arc<Evaluator>, mp: &MultiProgress) {
    let direction = metric_direction(&args.metric);
    let space = SearchSpace { direction };
    let mut optimizer = GpOptimizer::new(space);
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

    let pb_iters = ProgressBar::hidden();

    for trial_idx in 1..=args.n_trials {
        let start = std::time::Instant::now();
        let config = optimizer.suggest(&mut rng);

        let mut metrics = Vec::with_capacity(args.n_seeds);
        for seed_idx in 0..args.n_seeds {
            let seed = 42 + trial_idx as u64 * 100 + seed_idx as u64;
            let m =
                evaluator.evaluate_with_metric(&config, curvature, &args.metric, seed, &pb_iters);
            metrics.push(m);
        }

        let mean = metrics.iter().sum::<f64>() / args.n_seeds as f64;
        let variance =
            metrics.iter().map(|&m| (m - mean).powi(2)).sum::<f64>() / args.n_seeds as f64;
        let std = variance.sqrt();

        optimizer.observe(config.clone(), mean);
        let elapsed = start.elapsed().as_millis() as u64;

        let result = TrialResult {
            metric_name: args.metric.clone(),
            curvature,
            trial: trial_idx,
            learning_rate: config.learning_rate,
            perplexity_ratio: config.perplexity_ratio,
            momentum_main: config.momentum_main,
            scaling_loss: config.scaling_loss,
            centering_weight: config.centering_weight,
            global_loss_weight: config.global_loss_weight,
            norm_loss_weight: config.norm_loss_weight,
            metric_mean: mean,
            metric_std: std,
            time_ms: elapsed,
            scan_param: None,
        };

        write_result(&result, &out_path);

        let best = optimizer.best_trial();
        let actual_perp = config.perplexity_ratio * evaluator.n_points() as f64;
        pb.set_prefix(format!("{:.4}", best));
        pb.println(format!(
            "k={:+.1} trial {:3} | {} = {:.4} ± {:.4} | best={:.4} | {}ms \
             | lr={:.4} perp={:.1} (ratio={:.4}) sl={} cw={:.3} glw={:.2} nw={:.4}",
            curvature,
            trial_idx,
            args.metric,
            mean,
            std,
            best,
            elapsed,
            config.learning_rate,
            actual_perp,
            config.perplexity_ratio,
            config.scaling_loss,
            config.centering_weight,
            config.global_loss_weight,
            config.norm_loss_weight,
        ));
        pb.inc(1);
    }

    pb.finish_with_message(format!("{:+.1} done", curvature));

    if let Some(best_config) = optimizer.best_config() {
        let best_perp = best_config.perplexity_ratio * evaluator.n_points() as f64;
        pb.println(format!(
            "\n=== Best for k={:+.1} | {}={:.4} ===\n  \
             lr={:.4}  perp={:.1} (ratio={:.4})  momentum={:.4}\n  \
             n_iter={}  ee_iter={}\n  \
             scaling_loss={}  centering={:.3}  global_loss={:.3}  norm={:.4}",
            curvature,
            args.metric,
            optimizer.best_trial(),
            best_config.learning_rate,
            best_perp,
            best_config.perplexity_ratio,
            best_config.momentum_main,
            FIXED_N_ITERATIONS,
            FIXED_EARLY_EXAG_ITERATIONS,
            best_config.scaling_loss,
            best_config.centering_weight,
            best_config.global_loss_weight,
            best_config.norm_loss_weight,
        ));
    }
}

// ─── Scan mode ────────────────────────────────────────────────────────────────

/// Load the best TrialConfig from a JSONL file (by metric_mean).
/// Handles both old format (absolute `perplexity`) and new format (`perplexity_ratio`).
/// `n_points` is used to convert old-format absolute perplexity to a ratio.
fn load_best_config_from_jsonl(path: &str, n_points: usize) -> Option<TrialConfig> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut best_val = f64::NEG_INFINITY;
    let mut best: Option<TrialConfig> = None;

    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        let metric = v["metric_mean"].as_f64().unwrap_or(f64::NEG_INFINITY);
        if metric > best_val {
            best_val = metric;
            // Prefer new-format perplexity_ratio; fall back to old absolute perplexity / n_points.
            let perplexity_ratio = v["perplexity_ratio"]
                .as_f64()
                .unwrap_or_else(|| v["perplexity"].as_f64().unwrap_or(15.0) / n_points as f64);
            best = Some(TrialConfig {
                learning_rate: v["learning_rate"].as_f64().unwrap_or(10.0),
                perplexity_ratio,
                momentum_main: v["momentum_main"].as_f64().unwrap_or(0.8),
                scaling_loss: v["scaling_loss"].as_u64().unwrap_or(0) as u8,
                centering_weight: v["centering_weight"].as_f64().unwrap_or(0.0),
                global_loss_weight: v["global_loss_weight"].as_f64().unwrap_or(0.0),
                norm_loss_weight: v["norm_loss_weight"].as_f64().unwrap_or(0.0),
            });
        }
    }
    best
}

/// Generate `n` evenly-spaced values on [lo, hi] in log space (if log=true) or linear.
fn sweep_values(lo: f64, hi: f64, n: usize, log: bool) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.0
            };
            if log {
                (lo.ln() + t * (hi.ln() - lo.ln())).exp()
            } else {
                lo + t * (hi - lo)
            }
        })
        .collect()
}

fn run_scan(curvature: f64, args: &Args, evaluator: Arc<Evaluator>, mp: &MultiProgress) {
    let n_points = evaluator.n_points();
    // Default config (perplexity_ratio 0.003 ≈ perplexity 15 for n=5000)
    let default_config = TrialConfig {
        learning_rate: 10.0,
        perplexity_ratio: 0.003,
        momentum_main: 0.85,
        scaling_loss: 3,
        centering_weight: 0.5,
        global_loss_weight: 0.0,
        norm_loss_weight: 0.0,
    };

    // Load base config from previous optimization run.
    let base = if let Some(prefix) = &args.scan_from {
        let path = output_path(prefix, curvature);
        match load_best_config_from_jsonl(&path, n_points) {
            Some(c) => {
                eprintln!(
                    "scan k={}: loaded base config from {} (best trial)",
                    curvature, path
                );
                c
            }
            None => {
                eprintln!(
                    "scan k={}: could not load from {}, using default config",
                    curvature, path
                );
                default_config
            }
        }
    } else {
        eprintln!(
            "scan k={}: no --scan-from provided, using default config",
            curvature
        );
        default_config
    };

    let n = args.scan_steps;

    // (param_name, values to sweep)
    let params: Vec<(&str, Vec<f64>)> = vec![
        ("learning_rate", sweep_values(0.5, 300.0, n, true)),
        ("perplexity_ratio", sweep_values(0.0004, 0.01, n, true)),
        ("momentum_main", sweep_values(0.70, 0.95, n, false)),
        ("scaling_loss", vec![0.0, 1.0, 2.0, 3.0, 4.0]),
        ("centering_weight", sweep_values(0.0, 2.0, n, false)),
        ("global_loss_weight", sweep_values(0.0, 2.0, n, false)),
        ("norm_loss_weight", sweep_values(0.0, 0.02, n, false)),
    ];

    let total = params.iter().map(|(_, v)| v.len()).sum::<usize>() as u64;

    let pb = mp.add(ProgressBar::new(total));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} scan k={msg} [{bar:35.cyan/blue}] {pos}/{len} {wide_msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(format!("{:+.1}", curvature));

    let out_path = format!("{}_k{:.1}_scan.jsonl", args.output, curvature);
    if Path::new(&out_path).exists() {
        std::fs::remove_file(&out_path).ok();
    }

    let pb_iters = ProgressBar::hidden();
    let mut trial_idx = 0usize;

    for (param_name, values) in &params {
        for &val in values {
            trial_idx += 1;
            let mut config = base.clone();
            match *param_name {
                "learning_rate" => config.learning_rate = val,
                "perplexity_ratio" => config.perplexity_ratio = val,
                "momentum_main" => config.momentum_main = val,
                "scaling_loss" => config.scaling_loss = val as u8,
                "centering_weight" => config.centering_weight = val,
                "global_loss_weight" => config.global_loss_weight = val,
                "norm_loss_weight" => config.norm_loss_weight = val,
                _ => unreachable!(),
            }

            let start = std::time::Instant::now();
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
            let elapsed = start.elapsed().as_millis() as u64;

            let result = TrialResult {
                metric_name: args.metric.clone(),
                curvature,
                trial: trial_idx,
                learning_rate: config.learning_rate,
                perplexity_ratio: config.perplexity_ratio,
                momentum_main: config.momentum_main,
                scaling_loss: config.scaling_loss,
                centering_weight: config.centering_weight,
                global_loss_weight: config.global_loss_weight,
                norm_loss_weight: config.norm_loss_weight,
                metric_mean: mean,
                metric_std: std,
                time_ms: elapsed,
                scan_param: Some(param_name.to_string()),
            };

            write_result(&result, &out_path);

            pb.set_message(format!(
                "{:+.1} | {}={:.4} → {:.4} ± {:.4}",
                curvature, param_name, val, mean, std
            ));
            pb.inc(1);
        }
    }

    pb.finish_with_message(format!("{:+.1} scan done", curvature));
}

// ─── main ─────────────────────────────────────────────────────────────────────

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

    let mode = args.mode.as_str();
    match mode {
        "optimize" => {
            println!(
                "Starting optimization: {} curvature sessions × {} trials, metric={} ({}), seeds={}",
                args.curvatures.len(),
                args.n_trials,
                args.metric,
                direction,
                args.n_seeds
            );
            println!(
                "Fixed: n_iterations={}, early_exag_iterations={}",
                FIXED_N_ITERATIONS, FIXED_EARLY_EXAG_ITERATIONS
            );
            println!("Curvatures: {:?}", args.curvatures);
            println!("Output prefix: {}", args.output);
        }
        "scan" => {
            let total_per_k = args.scan_steps * 6 + 5; // 6 continuous + 1 discrete (5 values)
            println!(
                "Starting scan: {} curvatures × ~{} sweep points, metric={}, seeds={}",
                args.curvatures.len(),
                total_per_k,
                args.metric,
                args.n_seeds
            );
            println!("Curvatures: {:?}", args.curvatures);
            println!("Output prefix: {} (suffix: _scan.jsonl)", args.output);
        }
        other => {
            eprintln!("Unknown --mode '{}'. Use 'optimize' or 'scan'.", other);
            std::process::exit(1);
        }
    }

    let mp = Arc::new(MultiProgress::new());

    let handles: Vec<_> = args
        .curvatures
        .iter()
        .map(|&k| {
            let args = args.clone();
            let evaluator = Arc::clone(&evaluator);
            let mp = Arc::clone(&mp);
            thread::spawn(move || match args.mode.as_str() {
                "scan" => run_scan(k, &args, evaluator, &mp),
                _ => run_session(k, &args, evaluator, &mp),
            })
        })
        .collect();

    for h in handles {
        h.join().expect("optimizer thread panicked");
    }

    println!("\nAll sessions complete.");
    match args.mode.as_str() {
        "scan" => println!("Scan results: {}_k<curvature>_scan.jsonl", args.output),
        _ => println!("Optimization results: {}_k<curvature>.jsonl", args.output),
    }
}
