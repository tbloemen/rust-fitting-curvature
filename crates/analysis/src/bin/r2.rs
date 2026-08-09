//! R2-indicator analysis of the qParEGO sweeps.
//!
//! This runs **locally**; there is no cluster stage. The whole table is a few
//! seconds of work: the indicator is an exact mean over an enumerated weight
//! set, so there is no sampling budget to trade against precision and no seed to
//! carry around.
//!
//! Four subcommands:
//!
//! * **`stats`** — stage 1. For every experiment cell (one results `.jsonl` file
//!   = one (setting, dataset, N, geometry) run) compute the R2 indicator of its
//!   10-objective Pareto front under each preference region, plus the front
//!   point each region recommends. One JSON object per cell.
//!
//! * **`aggregate`** — stage 2. ΔR2 over the `all_off` baseline per region, plus
//!   the Friedman test over the (dataset, geometry, N) blocks with Holm-adjusted
//!   post-hoc comparisons against the control.
//!
//! * **`recommend`** — the per-preference configuration table: for each cell and
//!   region, the recommended front point's hyperparameters and all ten of its
//!   oriented objective values.
//!
//! * **`front`** — recompute a cell's Pareto front in the optimizer's
//!   `*_pareto_*.json` schema, for the cells whose sweep predates front writing.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use clap::{Parser, Subcommand};
use serde::Serialize;

use fitting_analysis::aggregate::{self, CellRecord};
use fitting_analysis::objectives::OBJECTIVES;
use fitting_analysis::r2::{cell_summary, oriented_objectives, Weights};
use fitting_analysis::{pareto_front_records, parse_cell_stem, trial_records, Cell, TrialRecord};

/// Hyperparameters reported for a recommended configuration.
const PARAMS: [&str; 8] = [
    "learning_rate",
    "perplexity_ratio",
    "early_exaggeration_factor",
    "centering_weight",
    "global_loss_weight",
    "norm_loss_weight",
    "curvature_magnitude",
    "r_rms",
];

#[derive(Parser, Debug)]
#[command(name = "r2", about = "R2-indicator analysis of the qParEGO sweeps")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Stage 1: per-cell R2 indicator under each preference region.
    Stats(StatsArgs),
    /// Stage 2: ΔR2 over the baseline + cross-dataset Wilcoxon test.
    Aggregate(AggregateArgs),
    /// Recommended configuration per (cell, preference region).
    Recommend(RecommendArgs),
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

    /// Worker threads. Defaults to the number of logical CPUs.
    #[arg(long)]
    jobs: Option<usize>,
}

#[derive(Parser, Debug)]
struct AggregateArgs {
    /// Stage-1 JSONL file(s).
    #[arg(required = true)]
    tables: Vec<PathBuf>,

    /// Optional path to write the per-dataset ΔR2 rows as CSV.
    #[arg(long)]
    csv: Option<PathBuf>,

    /// Restrict the printed tables to one preference region.
    #[arg(long)]
    region: Option<String>,

    /// Settings to include as Friedman treatments. Must contain `all_off` and
    /// at least three entries; blocks missing any of them are dropped.
    #[arg(long, value_delimiter = ',')]
    settings: Option<Vec<String>>,

    /// Print the per-(N, geometry, setting) descriptive ΔR2 table as well.
    #[arg(long, default_value = "false")]
    descriptive: bool,
}

#[derive(Parser, Debug)]
struct RecommendArgs {
    /// Directory of `*.jsonl` result files.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Output CSV path.
    #[arg(long)]
    csv: PathBuf,

    /// Worker threads. Defaults to the number of logical CPUs.
    #[arg(long)]
    jobs: Option<usize>,
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
        Command::Recommend(a) => run_recommend(a),
        Command::Front(a) => run_front(a),
    }
}

// ─── Cell discovery ───────────────────────────────────────────────────────────

/// Every `(path, Cell)` for trial-results JSONL files under *results_dir*.
///
/// Front files (`*_pareto_*.json`) and anything whose stem doesn't parse as a
/// cell are skipped. Sorted by stem so the output order is stable.
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

fn worker_count(jobs: Option<usize>) -> usize {
    jobs.or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
        .max(1)
}

// ─── Stage 1: per-cell R2 ─────────────────────────────────────────────────────

fn run_stats(args: StatsArgs) -> std::io::Result<()> {
    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        eprintln!("No result cells found under {}", args.results_dir.display());
        std::process::exit(1);
    }

    let weights = Weights::new();
    let jobs = worker_count(args.jobs);
    eprintln!(
        "{} cells, {} weight vectors, {} regions, jobs={}",
        cells.len(),
        weights.vectors.len(),
        weights.regions.len(),
        jobs
    );

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
                let summary = cell_summary(&records, &weights);
                let rec = CellRecord {
                    stem: stem.to_string(),
                    setting: cell.setting.clone(),
                    dataset: cell.dataset.clone(),
                    n: cell.n,
                    geometry: cell.geometry.clone(),
                    n_trials: summary.n_trials,
                    n_front: summary.n_front,
                    r2: summary.r2.clone(),
                };
                let line = serde_json::to_string(&rec).expect("CellRecord serialises");
                let all = summary.r2.get("all").copied().unwrap_or(f64::NAN);
                let mut w = out.lock().unwrap();
                let _ = writeln!(w, "{line}");
                let _ = w.flush();
                eprintln!("  {}: R2(all)={:.5} (front={})", rec.stem, all, rec.n_front);
            });
        }
    });

    eprintln!("wrote {}", args.out.display());
    Ok(())
}

// ─── Stage 2: ΔR2 + Wilcoxon ──────────────────────────────────────────────────

fn run_aggregate(args: AggregateArgs) -> std::io::Result<()> {
    let table: Vec<CellRecord> = aggregate::load_table(&args.tables)?;
    if table.is_empty() {
        eprintln!("empty indicator table");
        std::process::exit(1);
    }
    if let Some(region) = &args.region {
        if !aggregate::regions(&table).iter().any(|r| r == region) {
            eprintln!(
                "no region {region:?} in the table; have {:?}",
                aggregate::regions(&table)
            );
            std::process::exit(1);
        }
    }

    let rows = aggregate::compute_deltas(&table);
    let keep_region = |region: &str| args.region.as_ref().map_or(true, |r| r == region);

    // ── Friedman + Holm, the test the thesis reports ──────────────────────────
    let settings = args
        .settings
        .clone()
        .unwrap_or_else(|| aggregate::settings(&rows));
    if settings.len() < 3 || !settings.iter().any(|s| s == aggregate::BASELINE) {
        eprintln!(
            "--settings must list at least three settings including {}; got {:?}",
            aggregate::BASELINE,
            settings
        );
        std::process::exit(1);
    }
    let tests = aggregate::rank_tests(&rows, &settings);

    println!(
        "Friedman over (dataset, geometry, N) blocks; Holm-adjusted post-hoc vs {}.",
        aggregate::BASELINE
    );
    println!("treatments: {}\n", settings.join(", "));

    let hdr = format!(
        "{:<18} {:<11} {:>7} {:>5} {:>9} {:>9}   {}",
        "region",
        "geometry",
        "#blocks",
        "#drop",
        "chi2",
        "p",
        settings
            .iter()
            .map(|s| format!("{s:>16}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let width = hdr.chars().count();
    println!("{hdr}");
    println!("{}", "-".repeat(width));
    for t in &tests {
        if !keep_region(&t.region) {
            continue;
        }
        let cells: Vec<String> = t
            .mean_ranks
            .iter()
            .zip(&t.holm_p)
            .map(|(rank, p)| match p {
                // The control has no p against itself; show its rank alone.
                None => format!("{rank:>10.2}      "),
                Some(p) => {
                    let star = if *p < 0.05 { "*" } else { " " };
                    format!("{rank:>7.2} {p:>7.4}{star}")
                }
            })
            .collect();
        println!(
            "{:<18} {:<11} {:>7} {:>5} {:>9.3} {:>9.4}   {}",
            t.region,
            t.geometry,
            t.n_blocks,
            t.dropped_blocks,
            t.statistic,
            t.p,
            cells.join(" ")
        );
    }
    println!("\nper cell: mean rank (1 = best) and Holm-adjusted p vs the control; * is p < 0.05.");

    // ── Descriptive ΔR2, opt-in ───────────────────────────────────────────────
    if args.descriptive {
        let summaries = aggregate::summarise(&rows);
        let hdr = format!(
            "{:>5} {:<11} {:<15} {:<18} {:>4} {:>11} {:>11} {:>5}",
            "N", "geometry", "setting", "region", "#ds", "mean ΔR2", "median ΔR2", "#pos"
        );
        // Dashes must match the *display* width, and the header has multi-byte ΔR2.
        let width = hdr.chars().count();
        println!("\n{hdr}");
        println!("{}", "-".repeat(width));
        for s in &summaries {
            if !keep_region(&s.region) {
                continue;
            }
            let fmt = |v: Option<f64>| match v {
                None => "n/a".to_string(),
                Some(x) => format!("{x:+.5}"),
            };
            println!(
                "{:>5} {:<11} {:<15} {:<18} {:>4} {:>11} {:>11} {:>5}",
                s.n,
                s.geometry,
                s.setting,
                s.region,
                s.n_datasets,
                fmt(s.mean_delta),
                fmt(s.median_delta),
                s.n_positive,
            );
        }
    }

    if let Some(csv_path) = args.csv {
        let mut rows = rows;
        rows.sort_by(|a, b| {
            (a.n, &a.geometry, &a.setting, &a.region, &a.dataset).cmp(&(
                b.n,
                &b.geometry,
                &b.setting,
                &b.region,
                &b.dataset,
            ))
        });
        let mut f = std::io::BufWriter::new(std::fs::File::create(&csv_path)?);
        writeln!(f, "n,geometry,setting,region,dataset,r2,r2_baseline,delta_r2")?;
        let num = |v: Option<f64>| v.map(|x| x.to_string()).unwrap_or_default();
        for r in &rows {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{}",
                r.n,
                r.geometry,
                r.setting,
                r.region,
                r.dataset,
                r.r2,
                num(r.r2_baseline),
                num(r.delta_r2)
            )?;
        }
        eprintln!("\nwrote per-dataset ΔR2 rows to {}", csv_path.display());
    }
    Ok(())
}

// ─── Recommended configurations ───────────────────────────────────────────────

/// One row of the recommendation table.
struct RecRow {
    stem: String,
    setting: String,
    dataset: String,
    n: usize,
    geometry: String,
    region: String,
    share: f64,
    params: Vec<Option<f64>>,
    objectives: Vec<f64>,
}

fn run_recommend(args: RecommendArgs) -> std::io::Result<()> {
    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        eprintln!("No result cells found under {}", args.results_dir.display());
        std::process::exit(1);
    }

    let weights = Weights::new();
    let jobs = worker_count(args.jobs);
    eprintln!("{} cells, jobs={}", cells.len(), jobs);

    let rows: Mutex<Vec<RecRow>> = Mutex::new(Vec::new());
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
                let summary = cell_summary(&records, &weights);
                let mut local = Vec::new();
                for (region, rec) in &summary.recommended {
                    let Some(&record_idx) = summary.front.get(rec.front_index) else {
                        continue;
                    };
                    let record = &records[record_idx];
                    let objectives = oriented_objectives(record);
                    local.push(RecRow {
                        stem: stem.to_string(),
                        setting: cell.setting.clone(),
                        dataset: cell.dataset.clone(),
                        n: cell.n,
                        geometry: cell.geometry.clone(),
                        region: region.clone(),
                        share: rec.share,
                        params: PARAMS.iter().map(|p| record.param(p)).collect(),
                        objectives: OBJECTIVES
                            .iter()
                            .map(|o| objectives.get(*o).copied().unwrap_or(0.0))
                            .collect(),
                    });
                }
                rows.lock().unwrap().extend(local);
            });
        }
    });

    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| {
        (a.n, &a.geometry, &a.dataset, &a.setting, &a.region).cmp(&(
            b.n,
            &b.geometry,
            &b.dataset,
            &b.setting,
            &b.region,
        ))
    });

    let mut f = std::io::BufWriter::new(std::fs::File::create(&args.csv)?);
    write!(f, "stem,setting,dataset,n,geometry,region,share")?;
    for p in PARAMS {
        write!(f, ",{p}")?;
    }
    for o in OBJECTIVES {
        write!(f, ",{o}")?;
    }
    writeln!(f)?;

    for r in &rows {
        write!(
            f,
            "{},{},{},{},{},{},{}",
            r.stem, r.setting, r.dataset, r.n, r.geometry, r.region, r.share
        )?;
        for v in &r.params {
            match v {
                Some(x) => write!(f, ",{x}")?,
                None => write!(f, ",")?,
            }
        }
        for v in &r.objectives {
            write!(f, ",{v}")?;
        }
        writeln!(f)?;
    }
    eprintln!("wrote {} rows to {}", rows.len(), args.csv.display());
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
