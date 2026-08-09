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
//!
//! Every subcommand's output is a **file**; nothing is written to stdout. The
//! one thing that reaches the terminal is a failure, rendered once by `main`
//! returning `Err`.

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use serde::Serialize;

use fitting_analysis::aggregate::{self, CellRecord, DeltaRow, GroupSummary, RankTest};
use fitting_analysis::objectives::OBJECTIVES;
use fitting_analysis::r2::{cell_summary, oriented_objectives, Weights};
use fitting_analysis::{
    pareto_front_records, parse_cell_stem, trial_records, Cell, Error, IoContext, Result,
    TrialRecord,
};

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
    /// Stage 2: ΔR2 over the baseline + cross-dataset rank test.
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
}

#[derive(Parser, Debug)]
struct AggregateArgs {
    /// Stage-1 JSONL file(s).
    #[arg(required = true)]
    tables: Vec<PathBuf>,

    /// Friedman + Holm results, one row per (region, geometry, setting).
    #[arg(long, default_value = "r2_tests.csv")]
    tests: PathBuf,

    /// Optional path to write the per-dataset ΔR2 rows as CSV.
    #[arg(long)]
    csv: Option<PathBuf>,

    /// Optional path for the descriptive per-(N, geometry, setting) ΔR2 table.
    #[arg(long)]
    descriptive: Option<PathBuf>,

    /// Restrict every written table to one preference region.
    #[arg(long)]
    region: Option<String>,

    /// Settings to include as Friedman treatments. Must contain `all_off` and
    /// at least three entries; blocks missing any of them are dropped.
    #[arg(long, value_delimiter = ',')]
    settings: Option<Vec<String>>,
}

#[derive(Parser, Debug)]
struct RecommendArgs {
    /// Directory of `*.jsonl` result files.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Output CSV path.
    #[arg(long)]
    csv: PathBuf,
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

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Stats(a) => run_stats(a),
        Command::Aggregate(a) => run_aggregate(a),
        Command::Recommend(a) => run_recommend(a),
        Command::Front(a) => run_front(a),
    }
}

// ─── Cell discovery ───────────────────────────────────────────────────────────

/// One results file and the experiment cell its name encodes.
struct CellFile {
    path: PathBuf,
    /// The file stem, which every downstream table keys by. Carried along
    /// because `parse_cell_stem` already proved it is valid UTF-8.
    stem: String,
    cell: Cell,
}

/// Every trial-results JSONL under *results_dir*, with its parsed cell.
///
/// Front files (`*_pareto_*.json`) and anything whose stem doesn't parse as a
/// cell are skipped. Sorted by stem so the output order is stable.
fn discover_cells(results_dir: &Path) -> Result<Vec<CellFile>> {
    let mut out: Vec<CellFile> = Vec::new();
    for entry in std::fs::read_dir(results_dir).at(results_dir)? {
        let path = entry.at(results_dir)?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some(cell) = parse_cell_stem(stem) {
            out.push(CellFile {
                stem: stem.to_string(),
                path: path.clone(),
                cell,
            });
        }
    }
    out.sort_by(|a, b| a.stem.cmp(&b.stem));
    Ok(out)
}

// ─── Stage 1: per-cell R2 ─────────────────────────────────────────────────────

fn run_stats(args: StatsArgs) -> Result<()> {
    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }

    let weights = Weights::new();

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).at(parent)?;
        }
    }

    // Cells are visited in `discover_cells` order, so the file is sorted by
    // stem and byte-identical across runs.
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out).at(&args.out)?);

    for cf in &cells {
        let records = trial_records(&cf.path)?;
        let summary = cell_summary(&records, &weights);
        let rec = CellRecord {
            stem: cf.stem.clone(),
            setting: cf.cell.setting.clone(),
            dataset: cf.cell.dataset.clone(),
            n: cf.cell.n,
            geometry: cf.cell.geometry.clone(),
            n_trials: summary.n_trials,
            n_front: summary.n_front,
            r2: summary.r2.clone(),
        };
        let line = serde_json::to_string(&rec).map_err(Error::Serialize)?;
        writeln!(out, "{line}").at(&args.out)?;
    }
    out.flush().at(&args.out)?;
    Ok(())
}

// ─── Stage 2: ΔR2 + the rank test ─────────────────────────────────────────────

fn run_aggregate(args: AggregateArgs) -> Result<()> {
    let table: Vec<CellRecord> = aggregate::load_table(&args.tables)?;
    if table.is_empty() {
        let first = args.tables.first().cloned().unwrap_or_default();
        return Err(Error::NoCells(first));
    }
    if let Some(region) = &args.region {
        let available = aggregate::regions(&table);
        if !available.iter().any(|r| r == region) {
            return Err(Error::UnknownRegion {
                region: region.clone(),
                available,
            });
        }
    }

    let rows = aggregate::compute_deltas(&table);
    // `map_or(true, ..)` rather than `is_none_or`: the crate's MSRV is 1.81.
    let keep_region = |region: &str| args.region.as_deref().map_or(true, |r| r == region);

    // ── Friedman + Holm, the test the thesis reports ──────────────────────────
    let settings = args
        .settings
        .clone()
        .unwrap_or_else(|| aggregate::settings(&rows));
    if settings.len() < 3 {
        return Err(Error::TooFewSettings(settings));
    }
    if !settings.iter().any(|s| s == aggregate::BASELINE) {
        return Err(Error::MissingBaseline {
            baseline: aggregate::BASELINE,
            settings,
        });
    }
    let tests = aggregate::rank_tests(&rows, &settings);
    write_tests_csv(&args.tests, &tests, &keep_region)?;

    if let Some(path) = &args.descriptive {
        let summaries = aggregate::summarise(&rows);
        write_descriptive_csv(path, &summaries, &keep_region)?;
    }

    if let Some(path) = &args.csv {
        write_delta_csv(path, rows, &keep_region)?;
    }
    Ok(())
}

/// A CSV writer over *path*, with the parent directory created.
fn csv_writer(path: &Path) -> Result<std::io::BufWriter<std::fs::File>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).at(parent)?;
        }
    }
    Ok(std::io::BufWriter::new(
        std::fs::File::create(path).at(path)?,
    ))
}

/// An optional float as a CSV field: empty rather than `None`.
fn num(v: Option<f64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

/// The Friedman/Holm table: one row per (region, geometry, setting).
///
/// Flattened per setting rather than per test because `mean_ranks` and `holm_p`
/// are parallel to `settings`, and a CSV cell holding a list is no use to a
/// spreadsheet. The omnibus columns repeat across a test's rows.
fn write_tests_csv(
    path: &Path,
    tests: &[RankTest],
    keep_region: &impl Fn(&str) -> bool,
) -> Result<()> {
    let mut f = csv_writer(path)?;
    writeln!(
        f,
        "region,geometry,n_blocks,dropped_blocks,statistic,p,setting,mean_rank,holm_p"
    )
    .at(path)?;
    for t in tests {
        if !keep_region(&t.region) {
            continue;
        }
        for (i, setting) in t.settings.iter().enumerate() {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{}",
                t.region,
                t.geometry,
                t.n_blocks,
                t.dropped_blocks,
                t.statistic,
                t.p,
                setting,
                num(t.mean_ranks.get(i).copied()),
                num(t.holm_p.get(i).copied().flatten()),
            )
            .at(path)?;
        }
    }
    f.flush().at(path)
}

/// The descriptive mean/median ΔR2 table per (N, geometry, setting, region).
fn write_descriptive_csv(
    path: &Path,
    summaries: &[GroupSummary],
    keep_region: &impl Fn(&str) -> bool,
) -> Result<()> {
    let mut f = csv_writer(path)?;
    writeln!(
        f,
        "n,geometry,setting,region,n_datasets,mean_delta_r2,median_delta_r2,n_positive"
    )
    .at(path)?;
    for s in summaries {
        if !keep_region(&s.region) {
            continue;
        }
        writeln!(
            f,
            "{},{},{},{},{},{},{},{}",
            s.n,
            s.geometry,
            s.setting,
            s.region,
            s.n_datasets,
            num(s.mean_delta),
            num(s.median_delta),
            s.n_positive
        )
        .at(path)?;
    }
    f.flush().at(path)
}

/// The per-dataset ΔR2 rows the figures and the thesis tables read.
fn write_delta_csv(
    path: &Path,
    mut rows: Vec<DeltaRow>,
    keep_region: &impl Fn(&str) -> bool,
) -> Result<()> {
    rows.sort_by(|a, b| {
        (a.n, &a.geometry, &a.setting, &a.region, &a.dataset).cmp(&(
            b.n,
            &b.geometry,
            &b.setting,
            &b.region,
            &b.dataset,
        ))
    });
    let mut f = csv_writer(path)?;
    writeln!(
        f,
        "n,geometry,setting,region,dataset,r2,r2_baseline,delta_r2"
    )
    .at(path)?;
    for r in &rows {
        if !keep_region(&r.region) {
            continue;
        }
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
        )
        .at(path)?;
    }
    f.flush().at(path)
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

fn run_recommend(args: RecommendArgs) -> Result<()> {
    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }

    let weights = Weights::new();

    let mut rows: Vec<RecRow> = Vec::new();
    for cf in &cells {
        let records = trial_records(&cf.path)?;
        let summary = cell_summary(&records, &weights);
        for (region, rec) in &summary.recommended {
            let Some(&record_idx) = summary.front.get(rec.front_index) else {
                continue;
            };
            let record = &records[record_idx];
            let objectives = oriented_objectives(record);
            rows.push(RecRow {
                stem: cf.stem.clone(),
                setting: cf.cell.setting.clone(),
                dataset: cf.cell.dataset.clone(),
                n: cf.cell.n,
                geometry: cf.cell.geometry.clone(),
                region: region.clone(),
                share: rec.share,
                params: PARAMS.iter().map(|p| record.param(p)).collect(),
                objectives: OBJECTIVES
                    .iter()
                    .map(|o| objectives.get(*o).copied().unwrap_or(0.0))
                    .collect(),
            });
        }
    }

    rows.sort_by(|a, b| {
        (a.n, &a.geometry, &a.dataset, &a.setting, &a.region).cmp(&(
            b.n,
            &b.geometry,
            &b.dataset,
            &b.setting,
            &b.region,
        ))
    });

    let path = &args.csv;
    let mut f = csv_writer(path)?;
    write!(f, "stem,setting,dataset,n,geometry,region,share").at(path)?;
    for p in PARAMS {
        write!(f, ",{p}").at(path)?;
    }
    for o in OBJECTIVES {
        write!(f, ",{o}").at(path)?;
    }
    writeln!(f).at(path)?;

    for r in &rows {
        write!(
            f,
            "{},{},{},{},{},{},{}",
            r.stem, r.setting, r.dataset, r.n, r.geometry, r.region, r.share
        )
        .at(path)?;
        for v in &r.params {
            match v {
                Some(x) => write!(f, ",{x}").at(path)?,
                None => write!(f, ",").at(path)?,
            }
        }
        for v in &r.objectives {
            write!(f, ",{v}").at(path)?;
        }
        writeln!(f).at(path)?;
    }
    f.flush().at(path)
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

fn run_front(args: FrontArgs) -> Result<()> {
    let out_dir = args.out_dir.unwrap_or_else(|| args.results_dir.clone());
    std::fs::create_dir_all(&out_dir).at(&out_dir)?;
    let cells = discover_cells(&args.results_dir)?;

    for cf in &cells {
        // The optimizer names fronts `<stem>_pareto_<dataset>_<geometry>.json`;
        // analyze_hyperparams.py recovers dataset + geometry from that suffix.
        let front_path = out_dir.join(format!(
            "{}_pareto_{}_{}.json",
            cf.stem, cf.cell.dataset, cf.cell.geometry
        ));
        if front_path.exists() && !args.force {
            continue;
        }
        let records = trial_records(&cf.path)?;
        if records.is_empty() {
            continue;
        }
        let front = pareto_front_records(&records);
        let entries: Vec<FrontEntry> = front.iter().map(front_entry).collect();
        let json = serde_json::to_string_pretty(&entries).map_err(Error::Serialize)?;
        std::fs::write(&front_path, json).at(&front_path)?;
    }
    Ok(())
}
