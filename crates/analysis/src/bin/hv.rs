//! Hypervolume analysis of the qParEGO sweeps.
//!
//! Rust port of `hv_stats.py` + `hv_aggregate.py` + `pareto_utils.front_entries`.
//!
//! This runs **locally**; there is no cluster stage. The Python version needed a
//! 22-way SLURM array and a hand-provisioned numpy venv to get through the table
//! in a couple of hours. The whole 176-cell table now takes ~1s at `--n-mc 2e6`
//! and ~12s at the default 2e7 on a 16-core desktop, so the job that used to be
//! submitted is now shorter than the queue wait.
//!
//! Three subcommands:
//!
//! * **`stats`** — stage 1. For every experiment cell (one results `.jsonl` file
//!   = one (setting, dataset, N, geometry) run) compute the dominated
//!   hypervolume of its 10-objective Pareto front in the oriented unit box
//!   `[0, 1]^10` by Monte-Carlo integration. One JSON object per cell.
//!
//!   The per-cell RNG seed is derived deterministically from the cell name and
//!   `--seed` (`crc32(stem) ^ seed`), so a cell's estimate does not depend on
//!   `--jobs`, on how the work happened to be scheduled, or on which other cells
//!   were in the same run — re-running one cell reproduces it exactly.
//!
//! * **`aggregate`** — stage 2. ΔH over the `all_off` baseline plus the
//!   cross-dataset Wilcoxon signed-rank test.
//!
//! * **`front`** — recompute a cell's Pareto front in the optimizer's
//!   `*_pareto_*.json` schema, for the cells whose sweep predates front writing.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use clap::{Parser, Subcommand};
use serde::Serialize;

use fitting_analysis::aggregate::{self, HvRecord};
use fitting_analysis::hypervolume::{cell_hypervolume, cell_seed};
use fitting_analysis::objectives::OBJECTIVES;
use fitting_analysis::{pareto_front_records, parse_cell_stem, trial_records, Cell, TrialRecord};

#[derive(Parser, Debug)]
#[command(name = "hv", about = "Hypervolume analysis of the qParEGO sweeps")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Stage 1: per-cell Monte-Carlo hypervolume.
    Stats(StatsArgs),
    /// Stage 2: ΔH over the baseline + cross-dataset Wilcoxon test.
    Aggregate(AggregateArgs),
    /// Recompute Pareto fronts in the optimizer's `*_pareto_*.json` schema.
    Front(FrontArgs),
}

#[derive(Parser, Debug)]
struct StatsArgs {
    /// Directory of `*.jsonl` result files.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Output JSONL path (one line per cell).
    #[arg(long)]
    out: PathBuf,

    /// Monte-Carlo samples per cell. The default costs ~12s for the full table
    /// on 16 cores and puts the per-cell standard error near 1e-4, an order of
    /// magnitude below the smallest ΔH the aggregate stage calls significant.
    #[arg(long, default_value = "20000000")]
    n_mc: u64,

    /// Base seed; combined with each cell's name.
    #[arg(long, default_value = "0")]
    seed: u64,

    /// Worker threads. Defaults to the number of logical CPUs.
    #[arg(long)]
    jobs: Option<usize>,
}

#[derive(Parser, Debug)]
struct AggregateArgs {
    /// Stage-1 HV JSONL file(s) / shard files.
    #[arg(required = true)]
    tables: Vec<PathBuf>,

    /// Optional path to write the per-dataset ΔH rows as CSV.
    #[arg(long)]
    csv: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct FrontArgs {
    /// Directory of `*.jsonl` result files.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Where to write the front JSON (defaults to --results-dir).
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Overwrite fronts the optimizer already wrote instead of skipping them.
    #[arg(long, default_value = "false")]
    force: bool,
}

fn main() -> std::io::Result<()> {
    match Args::parse().command {
        Command::Stats(a) => run_stats(a),
        Command::Aggregate(a) => run_aggregate(a),
        Command::Front(a) => run_front(a),
    }
}

// ─── Cell discovery ───────────────────────────────────────────────────────────

/// Every `(path, Cell)` for trial-results JSONL files under *results_dir*.
///
/// Front files (`*_pareto_*.json`) and anything whose stem doesn't parse as a
/// cell are skipped. Sorted by stem so shard assignment is stable.
fn discover_cells(results_dir: &Path) -> std::io::Result<Vec<(PathBuf, Cell)>> {
    let mut out: Vec<(PathBuf, Cell)> = Vec::new();
    for entry in std::fs::read_dir(results_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some(cell) = parse_cell_stem(stem) {
            out.push((path.clone(), cell));
        }
    }
    out.sort_by(|a, b| a.0.file_stem().cmp(&b.0.file_stem()));
    Ok(out)
}

// ─── Stage 1: per-cell hypervolume ────────────────────────────────────────────

/// A full HV record for one cell: identity + counts + HV and its MC error.
#[derive(Serialize)]
struct CellRecord {
    stem: String,
    setting: String,
    dataset: String,
    n: usize,
    geometry: String,
    n_mc: u64,
    n_trials: usize,
    n_front: usize,
    hv: f64,
    hv_se: f64,
}

fn run_stats(args: StatsArgs) -> std::io::Result<()> {
    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        eprintln!("No result cells found under {}", args.results_dir.display());
        std::process::exit(1);
    }

    let jobs = args
        .jobs
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
        .max(1);

    eprintln!("{} cells, n_mc={}, jobs={}", cells.len(), args.n_mc, jobs);

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    // Written as each cell finishes rather than at the end, so an interrupted
    // run still leaves the cells it did complete on disk. Line order is
    // therefore completion order; every consumer keys by `stem`.
    let out = Mutex::new(std::io::BufWriter::new(std::fs::File::create(&args.out)?));
    let next = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((path, cell)) = cells.get(i) else {
                    return;
                };
                let stem = path.file_stem().unwrap().to_str().unwrap();
                let records = trial_records(path);
                let summary = cell_hypervolume(&records, args.n_mc, cell_seed(stem, args.seed));
                let rec = CellRecord {
                    stem: stem.to_string(),
                    setting: cell.setting.clone(),
                    dataset: cell.dataset.clone(),
                    n: cell.n,
                    geometry: cell.geometry.clone(),
                    n_mc: args.n_mc,
                    n_trials: summary.n_trials,
                    n_front: summary.n_front,
                    hv: summary.hv,
                    hv_se: summary.hv_se,
                };
                let line = serde_json::to_string(&rec).expect("CellRecord serialises");
                let mut w = out.lock().unwrap();
                let _ = writeln!(w, "{line}");
                let _ = w.flush();
                eprintln!(
                    "  {}: hv={:.5} ± {:.5} (front={})",
                    rec.stem, rec.hv, rec.hv_se, rec.n_front
                );
            });
        }
    });

    eprintln!("wrote {}", args.out.display());
    Ok(())
}

// ─── Stage 2: ΔH + Wilcoxon ───────────────────────────────────────────────────

fn run_aggregate(args: AggregateArgs) -> std::io::Result<()> {
    let table: Vec<HvRecord> = aggregate::load_hv_table(&args.tables)?;
    if table.is_empty() {
        eprintln!("empty HV table");
        std::process::exit(1);
    }
    let rows = aggregate::compute_deltas(&table);
    let summaries = aggregate::summarise(&rows);

    let hdr = format!(
        "{:>5} {:<11} {:<15} {:>4} {:>10} {:>10} {:>5} {:>11}",
        "N", "geometry", "setting", "#ds", "mean ΔH", "median ΔH", "#pos", "wilcoxon p"
    );
    // Dashes must match the *display* width, and the header has multi-byte ΔH.
    let width = hdr.chars().count();
    println!("{hdr}");
    println!("{}", "-".repeat(width));
    for s in &summaries {
        let fmt = |v: Option<f64>| match v {
            None => "n/a".to_string(),
            Some(x) => format!("{x:+.5}"),
        };
        let p = match s.wilcoxon_p {
            None => "n/a".to_string(),
            Some(x) => format!("{x:.4}"),
        };
        let star = if s.wilcoxon_p.is_some_and(|x| x < 0.05) {
            " *"
        } else {
            ""
        };
        println!(
            "{:>5} {:<11} {:<15} {:>4} {:>10} {:>10} {:>5} {:>11}{}",
            s.n,
            s.geometry,
            s.setting,
            s.n_datasets,
            fmt(s.mean_delta),
            fmt(s.median_delta),
            s.n_positive,
            p,
            star
        );
    }

    if let Some(csv_path) = args.csv {
        let mut rows = rows;
        rows.sort_by(|a, b| {
            (a.n, &a.geometry, &a.setting, &a.dataset).cmp(&(
                b.n,
                &b.geometry,
                &b.setting,
                &b.dataset,
            ))
        });
        let mut f = std::io::BufWriter::new(std::fs::File::create(&csv_path)?);
        writeln!(f, "n,geometry,setting,dataset,hv,hv_baseline,delta_hv")?;
        let num = |v: Option<f64>| v.map(|x| x.to_string()).unwrap_or_default();
        for r in &rows {
            writeln!(
                f,
                "{},{},{},{},{},{},{}",
                r.n,
                r.geometry,
                r.setting,
                r.dataset,
                r.hv,
                num(r.hv_baseline),
                num(r.delta_hv)
            )?;
        }
        eprintln!("\nwrote per-dataset ΔH rows to {}", csv_path.display());
    }
    Ok(())
}

// ─── Front recomputation ──────────────────────────────────────────────────────

/// One entry of a `*_pareto_*.json` file, matching `pareto.rs::write_pareto_front`.
#[derive(Serialize)]
struct FrontEntry {
    n_samples: Option<usize>,
    learning_rate: Option<f64>,
    perplexity_ratio: Option<f64>,
    momentum_main: Option<f64>,
    centering_weight: Option<f64>,
    global_loss_weight: Option<f64>,
    norm_loss_weight: Option<f64>,
    early_exaggeration_factor: Option<f64>,
    /// Euclidean trials carry no curvature; the optimizer still writes a (fixed)
    /// magnitude, so a missing field becomes 0.0 rather than null.
    curvature_magnitude: f64,
    r_max: Option<f64>,
    r_rms: Option<f64>,
    metrics: serde_json::Map<String, serde_json::Value>,
}

fn front_entry(r: &TrialRecord) -> FrontEntry {
    let mut metrics = serde_json::Map::new();
    for name in OBJECTIVES {
        let v = match r.objective(name) {
            Some(x) => serde_json::Number::from_f64(x)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            None => serde_json::Value::Null,
        };
        metrics.insert(name.to_string(), v);
    }
    FrontEntry {
        n_samples: r.n_samples,
        learning_rate: r.learning_rate,
        perplexity_ratio: r.perplexity_ratio,
        momentum_main: r.momentum_main,
        centering_weight: r.centering_weight,
        global_loss_weight: r.global_loss_weight,
        norm_loss_weight: r.norm_loss_weight,
        early_exaggeration_factor: r.early_exaggeration_factor,
        curvature_magnitude: r.curvature_magnitude.unwrap_or(0.0),
        r_max: r.r_max,
        r_rms: r.r_rms,
        metrics,
    }
}

fn run_front(args: FrontArgs) -> std::io::Result<()> {
    let out_dir = args.out_dir.unwrap_or_else(|| args.results_dir.clone());
    std::fs::create_dir_all(&out_dir)?;
    let cells = discover_cells(&args.results_dir)?;
    let (mut written, mut skipped) = (0usize, 0usize);

    for (path, cell) in &cells {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        // The optimizer names fronts `<stem>_pareto_<dataset>_<geometry>.json`;
        // analyze_hyperparams.py recovers dataset + geometry from that suffix.
        let front_path = out_dir.join(format!(
            "{stem}_pareto_{}_{}.json",
            cell.dataset, cell.geometry
        ));
        if front_path.exists() && !args.force {
            skipped += 1;
            continue;
        }
        let records = trial_records(path);
        if records.is_empty() {
            continue;
        }
        let front = pareto_front_records(&records);
        let entries: Vec<FrontEntry> = front.iter().map(front_entry).collect();
        std::fs::write(&front_path, serde_json::to_string_pretty(&entries)?)?;
        println!("wrote {} ({} entries)", front_path.display(), entries.len());
        written += 1;
    }
    eprintln!("{written} fronts written, {skipped} already present");
    Ok(())
}
