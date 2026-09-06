use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "fitting-optimizer")]
#[command(about = "Hyperparameter search for fitting-curvature")]
pub(crate) struct Args {
    #[arg(long, default_value = "./www/public/data")]
    pub(crate) data_path: String,

    #[arg(long, default_value = "1000")]
    pub(crate) n_trials: usize,

    #[arg(long, default_value = "3")]
    pub(crate) n_seeds: usize,

    #[arg(long, default_value = "1000")]
    pub(crate) n_samples: usize,

    /// Output file. All results (all datasets, all curvatures) are appended here.
    #[arg(long, default_value = "results/results.jsonl")]
    pub(crate) output: String,

    /// Dataset to run. Use "all" for all datasets, "real" for real datasets only
    /// (mnist, fashion_mnist, pbmc, wordnet_mammals), or a single dataset name.
    #[arg(long)]
    pub(crate) dataset: Option<String>,

    /// Run mode: "random" (default), "bayes", "scan", "pareto" or "detect".
    /// random: sample random configs with continuous curvature, compute all metrics.
    /// bayes:  Bayesian optimisation over all 7 hyperparameters (requires --metric).
    ///         Geometry sign is detected automatically unless --geometry is given.
    /// scan:   sweep each parameter individually from a base config (requires --metric).
    /// pareto: qParEGO multi-objective optimisation over 10 objectives (no --metric needed).
    /// detect: run curvature detection on the data distances; one JSONL line per
    ///         dataset, carrying κ_data plus all three Wilson arms and the δ(k)
    ///         diagnostics. This is what `fitting-analysis`'s `exp1` reads.
    #[arg(long, default_value = "random")]
    pub(crate) mode: String,

    /// Force a specific geometry for --mode bayes and --mode scan.
    /// Values: "hyperbolic" (k<0), "euclidean" (k=0), "spherical" (k>0).
    /// If omitted, the geometry is inferred automatically via curvature detection.
    #[arg(long)]
    pub(crate) geometry: Option<String>,

    /// For --mode random: lower bound of the continuous curvature range.
    #[arg(long, default_value = "-5.0")]
    pub(crate) curvature_min: f64,

    /// For --mode random: upper bound of the continuous curvature range.
    #[arg(long, default_value = "5.0")]
    pub(crate) curvature_max: f64,

    /// For --mode bayes or scan: metric to optimise (e.g. trustworthiness).
    #[arg(long)]
    pub(crate) metric: Option<String>,

    /// For --mode scan: results file to load the best prior config from as a base.
    #[arg(long)]
    pub(crate) scan_from: Option<String>,

    /// For --mode scan: number of evenly-spaced values to sweep per parameter.
    #[arg(long, default_value = "12")]
    pub(crate) scan_steps: usize,

    /// For --mode bayes: results file to warm-start the GP from.
    #[arg(long, default_value = "results/results.jsonl")]
    pub(crate) warm_start: Option<String>,

    /// Number of worker threads. Defaults to the number of logical CPUs.
    #[arg(long)]
    pub(crate) threads: Option<usize>,

    /// Experiment variant controlling which loss weights are optimized vs fixed to 0.
    /// Values: all_off, centering_only, global_only, norm_only, all_free (default).
    /// In all variants: lr, perplexity, early_exaggeration_factor are always optimized;
    /// momentum_main is always fixed at 0.8; scaling_loss_type is always MeanDistance.
    #[arg(long, default_value = "all_free")]
    pub(crate) experiment: String,

    /// For --mode pareto: resume from an existing --output JSONL. Trials already
    /// recorded there are replayed (their suggestions are re-derived but the
    /// expensive embedding evaluation is skipped and the stored metrics reused),
    /// so the search continues bit-identically from where it stopped. If the file
    /// is absent or empty the run simply starts fresh — so the same flag can be
    /// passed unconditionally to every job in a checkpointed/chained sweep.
    #[arg(long, default_value = "false")]
    pub(crate) resume: bool,
}
