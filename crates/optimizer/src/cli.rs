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
    /// (mnist, `fashion_mnist`, pbmc, `wordnet_mammals`), or a single dataset name.
    #[arg(long)]
    pub(crate) dataset: Option<String>,

    /// Run mode. The only mode is "pareto": qParEGO multi-objective
    /// optimisation over the 6 objectives. Kept as a flag so existing job
    /// scripts that pass `--mode pareto` keep working.
    #[arg(long, default_value = "pareto")]
    pub(crate) mode: String,

    /// Embedding geometry: "hyperbolic" (K<0), "euclidean" (K=0) or
    /// "spherical" (K>0).
    #[arg(long, value_parser = ["hyperbolic", "euclidean", "spherical"])]
    pub(crate) geometry: String,

    /// With `--curvature-max`: the larger absolute value of the two is the
    /// upper bound of the curvature magnitude the search may choose.
    #[arg(long, default_value = "-5.0")]
    pub(crate) curvature_min: f64,

    /// See `--curvature-min`.
    #[arg(long, default_value = "5.0")]
    pub(crate) curvature_max: f64,

    /// Number of worker threads. Defaults to the number of logical CPUs.
    #[arg(long)]
    pub(crate) threads: Option<usize>,

    /// Experiment variant controlling which loss weights are optimized vs fixed to 0.
    /// Values: `all_off`, `centering_only`, `global_only`, `norm_only`, `all_free` (default).
    /// In all variants: lr, perplexity, `early_exaggeration_factor` are always optimized;
    /// `momentum_main` is always fixed at 0.8; `scaling_loss_type` is always `MeanDistance`.
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
