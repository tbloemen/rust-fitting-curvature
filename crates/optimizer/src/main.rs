use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::thread;

use crate::data::Dataset;
use crate::evaluate::{AllMetrics, Evaluator};
use crate::search_space::TrialConfig;

mod data;
mod evaluate;
mod search_space;

#[derive(Parser, Debug, Clone)]
#[command(name = "fitting-optimizer")]
#[command(about = "Hyperparameter search for fitting-curvature")]
struct Args {
    #[arg(long, default_value = "./www/public/data")]
    data_path: String,

    #[arg(long, default_value = "1000")]
    n_trials: usize,

    #[arg(long, default_value = "3")]
    n_seeds: usize,

    #[arg(long, default_value = "1000")]
    n_samples: usize,

    /// Output file prefix. Each session writes to <output>_<dataset>_k<curvature>.jsonl.
    #[arg(long, default_value = "results/results")]
    output: String,

    #[arg(long)]
    dataset: Option<String>,

    /// Curvature values to run over (comma-separated).
    #[arg(
        long,
        num_args = 1..,
        value_delimiter = ',',
        default_values = ["-2", "-1", "-0.5", "-0.1", "0", "0.1", "0.5", "1", "2"]
    )]
    curvatures: Vec<f64>,

    /// Run mode: "random" (default) or "scan".
    /// random: sample random configs and compute all metrics.
    /// scan: sweep each parameter individually from a base config.
    #[arg(long, default_value = "random")]
    mode: String,

    /// For --mode scan: metric to evaluate (e.g. davies_bouldin_ratio).
    #[arg(long)]
    metric: Option<String>,

    /// For --mode scan: path prefix of a previous run to load the best config from.
    #[arg(long)]
    scan_from: Option<String>,

    /// For --mode scan: number of evenly-spaced values to sweep per continuous parameter.
    #[arg(long, default_value = "12")]
    scan_steps: usize,
}

// ─── Trial result ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct TrialResult {
    dataset_name: String,
    n_samples: usize,
    n_seeds: usize,
    curvature: f64,

    learning_rate: f64,
    perplexity_ratio: f64,
    momentum_main: f64,
    centering_weight: f64,
    global_loss_weight: f64,
    norm_loss_weight: f64,

    trustworthiness: Option<f64>,
    continuity: Option<f64>,
    knn_overlap: Option<f64>,
    geodesic_distortion_gu2019: Option<f64>,
    geodesic_distortion_mse: Option<f64>,
    davies_bouldin_ratio: Option<f64>,
    dunn_index: Option<f64>,
    class_density_measure: Option<f64>,
    cluster_density_measure: Option<f64>,

    time_ms: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    scan_param: Option<String>,
}

impl TrialResult {
    fn new(
        config: &TrialConfig,
        dataset_name: &str,
        n_samples: usize,
        n_seeds: usize,
        curvature: f64,
        time_ms: u64,
    ) -> Self {
        Self {
            dataset_name: dataset_name.to_string(),
            n_samples,
            n_seeds,
            curvature,
            learning_rate: config.learning_rate,
            perplexity_ratio: config.perplexity_ratio,
            momentum_main: config.momentum_main,
            centering_weight: config.centering_weight,
            global_loss_weight: config.global_loss_weight,
            norm_loss_weight: config.norm_loss_weight,
            trustworthiness: None,
            continuity: None,
            knn_overlap: None,
            geodesic_distortion_gu2019: None,
            geodesic_distortion_mse: None,
            davies_bouldin_ratio: None,
            dunn_index: None,
            class_density_measure: None,
            cluster_density_measure: None,
            time_ms,
            scan_param: None,
        }
    }

    fn with_all_metrics(mut self, m: &AggregatedMetrics) -> Self {
        self.trustworthiness = Some(m.trustworthiness);
        self.continuity = Some(m.continuity);
        self.knn_overlap = Some(m.knn_overlap);
        self.geodesic_distortion_gu2019 = Some(m.geodesic_distortion_gu2019);
        self.geodesic_distortion_mse = Some(m.geodesic_distortion_mse);
        self.davies_bouldin_ratio = Some(m.davies_bouldin_ratio);
        self.dunn_index = Some(m.dunn_index);
        self.class_density_measure = Some(m.class_density_measure);
        self.cluster_density_measure = Some(m.cluster_density_measure);
        self
    }
}

fn output_path(prefix: &str, dataset_name: &str, curvature: f64) -> String {
    format!("{}_{}_k{:.1}.jsonl", prefix, dataset_name, curvature).replace('.', "_")
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

// ─── Shared evaluation helpers ────────────────────────────────────────────────

fn trial_seed(trial_idx: usize, seed_idx: usize) -> u64 {
    42 + trial_idx as u64 * 100 + seed_idx as u64
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

fn eval_single_metric(
    evaluator: &Evaluator,
    config: &TrialConfig,
    curvature: f64,
    metric: &str,
    n_seeds: usize,
    trial_idx: usize,
    pb_iters: &ProgressBar,
) -> (f64, f64) {
    let values: Vec<f64> = (0..n_seeds)
        .map(|si| {
            evaluator.evaluate_with_metric(
                config,
                curvature,
                metric,
                trial_seed(trial_idx, si),
                pb_iters,
            )
        })
        .collect();
    mean_std(&values)
}

struct AggregatedMetrics {
    trustworthiness: f64,
    continuity: f64,
    knn_overlap: f64,
    geodesic_distortion_gu2019: f64,
    geodesic_distortion_mse: f64,
    davies_bouldin_ratio: f64,
    dunn_index: f64,
    class_density_measure: f64,
    cluster_density_measure: f64,
}

fn eval_all_metrics(
    evaluator: &Evaluator,
    config: &TrialConfig,
    curvature: f64,
    n_seeds: usize,
    trial_idx: usize,
    pb_iters: &ProgressBar,
) -> AggregatedMetrics {
    let samples: Vec<AllMetrics> = (0..n_seeds)
        .map(|si| {
            evaluator.compute_all_metrics(config, curvature, trial_seed(trial_idx, si), pb_iters)
        })
        .collect();

    let avg = |f: fn(&AllMetrics) -> f64| -> f64 {
        samples.iter().map(f).sum::<f64>() / samples.len() as f64
    };

    AggregatedMetrics {
        trustworthiness: avg(|m| m.trustworthiness),
        continuity: avg(|m| m.continuity),
        knn_overlap: avg(|m| m.knn_overlap),
        geodesic_distortion_gu2019: avg(|m| m.geodesic_distortion_gu2019),
        geodesic_distortion_mse: avg(|m| m.geodesic_distortion_mse),
        davies_bouldin_ratio: avg(|m| m.davies_bouldin_ratio),
        dunn_index: avg(|m| m.dunn_index),
        class_density_measure: avg(|m| m.class_density_measure),
        cluster_density_measure: avg(|m| m.cluster_density_measure),
    }
}

fn make_progress_bar(mp: &MultiProgress, total: u64, template: &str) -> ProgressBar {
    let pb = mp.add(ProgressBar::new(total));
    pb.set_style(
        ProgressStyle::with_template(template)
            .unwrap()
            .progress_chars("=>-"),
    );
    pb
}

// ─── Random search ────────────────────────────────────────────────────────────

fn run_random(
    curvature: f64,
    dataset_name: &str,
    args: &Args,
    evaluator: Arc<Evaluator>,
    mp: &MultiProgress,
) {
    let mut rng =
        fitting_core::synthetic_data::Rng::new(curvature.to_bits() ^ 0xdead_beef_cafe_1111);
    let out_path = output_path(&args.output, dataset_name, curvature);

    let pb = make_progress_bar(
        mp,
        args.n_trials as u64,
        "{spinner:.green} {msg} [{bar:35.cyan/blue}] {pos}/{len} | {wide_msg}",
    );
    pb.set_message(format!("k={:+.1} dataset={}", curvature, dataset_name));
    let pb_iters = ProgressBar::hidden();

    for trial_idx in 1..=args.n_trials {
        let start = std::time::Instant::now();
        let config = TrialConfig::random(&mut rng);
        let agg = eval_all_metrics(
            &evaluator,
            &config,
            curvature,
            args.n_seeds,
            trial_idx,
            &pb_iters,
        );
        let elapsed = start.elapsed().as_millis() as u64;

        let result = TrialResult::new(
            &config,
            dataset_name,
            args.n_samples,
            args.n_seeds,
            curvature,
            elapsed,
        )
        .with_all_metrics(&agg);
        write_result(&result, &out_path);

        pb.set_message(format!(
            "k={:+.1} trial {:4} | db_ratio={:.4} | trust={:.4} | {}ms",
            curvature, trial_idx, agg.davies_bouldin_ratio, agg.trustworthiness, elapsed
        ));
        pb.inc(1);
    }

    pb.finish_with_message(format!("k={:+.1} dataset={} done", curvature, dataset_name));
}

// ─── Scan ─────────────────────────────────────────────────────────────────────

fn load_best_config_from_jsonl(path: &str, n_points: usize) -> Option<TrialConfig> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut best_val = f64::NEG_INFINITY;
    let mut best: Option<TrialConfig> = None;

    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        let metric = v["metric_mean"].as_f64().unwrap_or(f64::NEG_INFINITY);
        if metric > best_val {
            best_val = metric;
            let perplexity_ratio = v["perplexity_ratio"]
                .as_f64()
                .unwrap_or_else(|| v["perplexity"].as_f64().unwrap_or(15.0) / n_points as f64);
            best = Some(TrialConfig {
                learning_rate: v["learning_rate"].as_f64().unwrap_or(10.0),
                perplexity_ratio,
                momentum_main: v["momentum_main"].as_f64().unwrap_or(0.8),
                centering_weight: v["centering_weight"].as_f64().unwrap_or(0.0),
                global_loss_weight: v["global_loss_weight"].as_f64().unwrap_or(0.0),
                norm_loss_weight: v["norm_loss_weight"].as_f64().unwrap_or(0.0),
            });
        }
    }
    best
}

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

fn apply_param(config: &mut TrialConfig, param: &str, val: f64) {
    match param {
        "learning_rate" => config.learning_rate = val,
        "perplexity_ratio" => config.perplexity_ratio = val,
        "momentum_main" => config.momentum_main = val,
        "centering_weight" => config.centering_weight = val,
        "global_loss_weight" => config.global_loss_weight = val,
        "norm_loss_weight" => config.norm_loss_weight = val,
        _ => unreachable!(),
    }
}

fn run_scan(
    curvature: f64,
    dataset_name: &str,
    args: &Args,
    evaluator: Arc<Evaluator>,
    mp: &MultiProgress,
) {
    let metric = args.metric.as_deref().unwrap();
    let n_points = evaluator.n_points();

    let default_config = TrialConfig {
        learning_rate: 10.0,
        perplexity_ratio: 0.003,
        momentum_main: 0.85,
        centering_weight: 0.5,
        global_loss_weight: 0.0,
        norm_loss_weight: 0.0,
    };

    let base = if let Some(prefix) = &args.scan_from {
        let path = output_path(prefix, dataset_name, curvature);
        match load_best_config_from_jsonl(&path, n_points) {
            Some(c) => {
                eprintln!("scan k={}: loaded base config from {}", curvature, path);
                c
            }
            None => {
                eprintln!(
                    "scan k={}: could not load from {}, using default",
                    curvature, path
                );
                default_config
            }
        }
    } else {
        eprintln!("scan k={}: no --scan-from, using default config", curvature);
        default_config
    };

    let n = args.scan_steps;
    let params: Vec<(&str, Vec<f64>)> = vec![
        ("learning_rate", sweep_values(0.5, 50.0, n, true)),
        ("perplexity_ratio", sweep_values(0.0004, 0.03, n, true)),
        ("momentum_main", sweep_values(0.60, 1.0, n, false)),
        ("centering_weight", sweep_values(0.0, 2.0, n, false)),
        ("global_loss_weight", sweep_values(0.0, 1.0, n, false)),
        ("norm_loss_weight", sweep_values(0.0, 0.02, n, false)),
    ];

    let total = params.iter().map(|(_, v)| v.len()).sum::<usize>() as u64;
    let pb = make_progress_bar(
        mp,
        total,
        "{spinner:.green} scan k={msg} [{bar:35.cyan/blue}] {pos}/{len} {wide_msg}",
    );
    pb.set_message(format!("{:+.1}", curvature));

    let out_path = format!(
        "{}_{}_k{:.1}_scan.jsonl",
        args.output, dataset_name, curvature
    )
    .replace('.', "_");
    if Path::new(&out_path).exists() {
        std::fs::remove_file(&out_path).ok();
    }

    let pb_iters = ProgressBar::hidden();
    let mut trial_idx = 0usize;

    for (param_name, values) in &params {
        for &val in values {
            trial_idx += 1;
            let mut config = base.clone();
            apply_param(&mut config, param_name, val);

            let start = std::time::Instant::now();
            let (mean, std) = eval_single_metric(
                &evaluator,
                &config,
                curvature,
                metric,
                args.n_seeds,
                trial_idx,
                &pb_iters,
            );
            let elapsed = start.elapsed().as_millis() as u64;

            let mut result = TrialResult::new(
                &config,
                dataset_name,
                args.n_samples,
                args.n_seeds,
                curvature,
                elapsed,
            );
            result.scan_param = Some(param_name.to_string());
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

// ─── Main ─────────────────────────────────────────────────────────────────────

fn get_dataset_names(dataset_arg: &Option<String>) -> Vec<String> {
    match dataset_arg {
        Some(name) if name == "all" => vec![
            "gaussian_blob".to_string(),
            "concentric_circles".to_string(),
            "tree".to_string(),
            "grid".to_string(),
        ],
        Some(name) => vec![name.clone()],
        None => vec!["mnist".to_string()],
    }
}

fn main() {
    let args = Args::parse();

    if args.mode == "scan" && args.metric.is_none() {
        eprintln!("Error: --metric is required for --mode scan");
        std::process::exit(1);
    }

    if let Some(parent) = Path::new(&args.output).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }

    let dataset_names = get_dataset_names(&args.dataset);

    match args.mode.as_str() {
        "random" => println!(
            "Starting random search: {} datasets × {} curvatures × {} trials, all metrics, seeds={}",
            dataset_names.len(),
            args.curvatures.len(),
            args.n_trials,
            args.n_seeds
        ),
        "scan" => println!(
            "Starting scan: {} datasets × {} curvatures × ~{} sweep points, metric={}, seeds={}",
            dataset_names.len(),
            args.curvatures.len(),
            args.scan_steps * 5,
            args.metric.as_deref().unwrap(),
            args.n_seeds
        ),
        other => {
            eprintln!("Unknown --mode '{}'. Use 'random' or 'scan'.", other);
            std::process::exit(1);
        }
    }
    println!("Curvatures: {:?}", args.curvatures);
    println!("Output prefix: {}", args.output);

    let mp = Arc::new(MultiProgress::new());
    let mut handles = Vec::new();

    for dataset_name in &dataset_names {
        println!("Loading dataset: {}...", dataset_name);
        let dataset = if dataset_name == "mnist" {
            match Dataset::load_mnist(&args.data_path, args.n_samples) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Error loading MNIST: {e}");
                    eprintln!("Make sure you have MNIST files in: {}", args.data_path);
                    std::process::exit(1);
                }
            }
        } else {
            Dataset::load_synthetic(dataset_name, args.n_samples, 42)
        };
        println!(
            "Loaded {} samples with {} features",
            dataset.n_points, dataset.n_features
        );

        let evaluator = Arc::new(Evaluator::new(dataset));

        for &k in &args.curvatures {
            let args = args.clone();
            let dataset_name = dataset_name.clone();
            let evaluator = Arc::clone(&evaluator);
            let mp = Arc::clone(&mp);
            let h = thread::spawn(move || match args.mode.as_str() {
                "scan" => run_scan(k, &dataset_name, &args, evaluator, &mp),
                _ => run_random(k, &dataset_name, &args, evaluator, &mp),
            });
            handles.push(h);
        }
    }

    for h in handles {
        h.join().expect("optimizer thread panicked");
    }

    println!("\nAll sessions complete.");
}
