//! R2-indicator analysis of the qParEGO sweeps.
//!
//! Five subcommands, each writing a **file** — JSONL throughout, the same format
//! the sweeps themselves are written in. Nothing goes to stdout; the one thing
//! that reaches the terminal is a failure, rendered once by `main` returning
//! `Err`.
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
//! * **`compare`** — the parameter-free cross-check. The binary additive
//!   ε-indicator between each setting's front and the `all_off` baseline's,
//!   in both directions, plus an optional table of how well its verdict agrees
//!   with ΔR2. No preference regions here: having no parameters is the point.
//!
//! * **`recommend`** — the per-preference configuration table: for each cell and
//!   region, the recommended front point's hyperparameters and all ten of its
//!   oriented objective values.
//!
//! * **`front`** — recompute a cell's Pareto front in the optimizer's
//!   `*_pareto_*.json` schema, for the cells whose sweep predates front writing.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde::Serialize;

use fitting_analysis::aggregate::{self, CellRecord};
use fitting_analysis::cell::{discover_cells, CellFile};
use fitting_analysis::indicators::epsilon_pair;
use fitting_analysis::objectives::{oriented_matrix, OBJECTIVES};
use fitting_analysis::r2::{cell_summary, oriented_objectives, Weights};
use fitting_analysis::stats;
use fitting_analysis::{
    pareto_front_records, trial_records, write_jsonl, Error, IoContext, Result, TrialRecord,
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
    /// Parameter-free cross-check: the binary additive ε-indicator vs the baseline.
    Compare(CompareArgs),
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

    /// Friedman + Holm results, one JSON object per (region, geometry) test.
    #[arg(long, default_value = "r2_tests.jsonl")]
    tests: PathBuf,

    /// Optional path to write the per-dataset ΔR2 rows as JSONL.
    #[arg(long)]
    deltas: Option<PathBuf>,

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
struct CompareArgs {
    /// Directory of `*.jsonl` result files.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Output JSONL path (one line per (dataset, geometry, N, setting)).
    #[arg(long, default_value = "r2_epsilon.jsonl")]
    out: PathBuf,

    /// Settings to compare against the baseline. The default is the four
    /// loss-weight settings of Experiment 2; `rms_anchored` is excluded because
    /// it fixes a different gauge and only exists for hyperbolic.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "centering_only,global_only,norm_only,all_free"
    )]
    settings: Vec<String>,

    /// Stage-1 R2 table, joined against the ε rows for `--agreement`.
    #[arg(long)]
    r2_table: Option<PathBuf>,

    /// Optional path for the ΔR2 / Δε agreement table.
    #[arg(long, requires = "r2_table")]
    agreement: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct RecommendArgs {
    /// Directory of `*.jsonl` result files.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Output JSONL path (one line per (cell, preference region)).
    #[arg(long)]
    out: PathBuf,
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
    #[arg(long)]
    force: bool,
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Stats(a) => run_stats(a),
        Command::Aggregate(a) => run_aggregate(a),
        Command::Compare(a) => run_compare(a),
        Command::Recommend(a) => run_recommend(a),
        Command::Front(a) => run_front(a),
    }
}

// ─── Stage 1: per-cell R2 ─────────────────────────────────────────────────────

fn run_stats(args: StatsArgs) -> Result<()> {
    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }

    let weights = Weights::new();

    // Cells are visited in `discover_cells` order
    let mut rows = Vec::with_capacity(cells.len());
    for cf in &cells {
        let records = trial_records(&cf.path)?;
        let summary = cell_summary(&records, &weights);
        rows.push(CellRecord {
            stem: cf.stem.clone(),
            setting: cf.cell.setting.clone(),
            dataset: cf.cell.dataset.clone(),
            n: cf.cell.n,
            geometry: cf.cell.geometry.clone(),
            n_trials: summary.n_trials,
            n_front: summary.n_front,
            r2: summary.r2.clone(),
        });
    }
    write_jsonl(&args.out, &rows)
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

    let mut rows = aggregate::compute_deltas(&table);
    let keep_region = |region: &str| args.region.as_deref().is_none_or(|r| r == region);

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
    // One JSON object per (region, geometry) test: `settings`, `mean_ranks` and
    // `holm_p` stay parallel arrays inside it, the shape `rank_tests` produces.
    let tests = aggregate::rank_tests(&rows, &settings);
    write_jsonl(&args.tests, tests.iter().filter(|t| keep_region(&t.region)))?;

    // Descriptive mean/median ΔR2 per (N, geometry, setting, region).
    if let Some(path) = &args.descriptive {
        let summaries = aggregate::summarise(&rows);
        write_jsonl(path, summaries.iter().filter(|s| keep_region(&s.region)))?;
    }

    // The per-dataset ΔR2 rows the thesis tables read.
    if let Some(path) = &args.deltas {
        rows.sort_by(|a, b| {
            (a.n, &a.geometry, &a.setting, &a.region, &a.dataset).cmp(&(
                b.n,
                &b.geometry,
                &b.setting,
                &b.region,
                &b.dataset,
            ))
        });
        write_jsonl(path, rows.iter().filter(|r| keep_region(&r.region)))?;
    }
    Ok(())
}

// ─── The parameter-free cross-check ───────────────────────────────────────────

/// One (dataset, geometry, N, setting) ε comparison against the baseline.
///
/// Front sizes are reported next to the indicator because the ε-indicator says
/// nothing about cardinality: a front of 12 points and one of 230 can score the
/// same, and the reader has to be able to see which case they are looking at.
#[derive(Serialize)]
struct EpsilonRow {
    dataset: String,
    geometry: String,
    n: usize,
    setting: String,
    n_front_setting: usize,
    n_front_baseline: usize,
    /// `I_ε+(setting, baseline)`.
    eps_setting_vs_baseline: f64,
    /// `I_ε+(baseline, setting)`.
    eps_baseline_vs_setting: f64,
    /// `eps_baseline_vs_setting − eps_setting_vs_baseline`; positive = improvement.
    delta_eps: f64,
    setting_covers_baseline: bool,
    baseline_covers_setting: bool,
}

/// How often the ε verdict and the ΔR2 verdict point the same way, per
/// preference region. Δε carries no region, so it is the same column joined
/// against each region's ΔR2 — divergence under a narrow region is the
/// interesting case, which is why this is not collapsed to one row.
#[derive(Serialize)]
struct AgreementRow {
    region: String,
    /// A single geometry, or `"*"` when pooled, matching `RankTest`.
    geometry: String,
    /// Pairs with both indicators finite; the basis for the correlation.
    n_pairs: usize,
    /// Pairs where both are additionally non-zero; the basis for the concordance.
    n_signed: usize,
    /// Fraction of `n_signed` pairs where `sign(ΔR2) == sign(Δε)`.
    sign_concordance: Option<f64>,
    spearman_rho: Option<f64>,
    spearman_p: Option<f64>,
}

fn run_compare(args: CompareArgs) -> Result<()> {
    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }

    // (N, geometry, dataset) → setting → cell. Sorted keys throughout, so the
    // output is sorted by (n, geometry, dataset, setting) without a final sort
    // and is byte-identical across runs.
    type BlockKey = (usize, String, String);
    let mut blocks: BTreeMap<BlockKey, BTreeMap<&str, &CellFile>> = BTreeMap::new();
    for cf in &cells {
        blocks
            .entry((cf.cell.n, cf.cell.geometry.clone(), cf.cell.dataset.clone()))
            .or_default()
            .insert(cf.cell.setting.as_str(), cf);
    }

    // Comparing the baseline with itself is not a comparison; drop it silently
    // so `--settings` can be pasted from the `aggregate` invocation.
    let wanted: Vec<&str> = args
        .settings
        .iter()
        .map(String::as_str)
        .filter(|s| *s != aggregate::BASELINE)
        .collect();

    let mut rows: Vec<EpsilonRow> = Vec::new();
    for ((n, geometry, dataset), by_setting) in &blocks {
        let present: Vec<&str> = wanted
            .iter()
            .copied()
            .filter(|s| by_setting.contains_key(s))
            .collect();
        if present.is_empty() {
            continue;
        }
        let Some(base_cf) = by_setting.get(aggregate::BASELINE) else {
            return Err(Error::NoBaselineCell {
                baseline: aggregate::BASELINE,
                n: *n,
                geometry: geometry.clone(),
                dataset: dataset.clone(),
            });
        };
        // One cell of records at a time: the front is all that outlives the load.
        let baseline = oriented_matrix(&pareto_front_records(&trial_records(&base_cf.path)?));

        for setting in present {
            let cf = by_setting[setting];
            let front = oriented_matrix(&pareto_front_records(&trial_records(&cf.path)?));
            // An empty front means an empty cell; there is nothing to compare.
            let Some(eps) = epsilon_pair(&front, &baseline) else {
                continue;
            };
            rows.push(EpsilonRow {
                dataset: dataset.clone(),
                geometry: geometry.clone(),
                n: *n,
                setting: setting.to_string(),
                n_front_setting: front.len(),
                n_front_baseline: baseline.len(),
                eps_setting_vs_baseline: eps.setting_vs_baseline,
                eps_baseline_vs_setting: eps.baseline_vs_setting,
                delta_eps: eps.delta,
                setting_covers_baseline: eps.setting_covers_baseline(),
                baseline_covers_setting: eps.baseline_covers_setting(),
            });
        }
    }
    if rows.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }
    write_jsonl(&args.out, &rows)?;

    if let (Some(agreement), Some(table)) = (&args.agreement, &args.r2_table) {
        let r2_table: Vec<CellRecord> = aggregate::load_table(std::slice::from_ref(table))?;
        if r2_table.is_empty() {
            return Err(Error::NoCells(table.clone()));
        }
        write_jsonl(agreement, agreement_rows(&rows, &r2_table))?;
    }
    Ok(())
}

/// Join Δε onto the per-region ΔR2 rows and correlate the two, per
/// (region, geometry) and pooled.
fn agreement_rows(eps: &[EpsilonRow], table: &[CellRecord]) -> Vec<AgreementRow> {
    let by_cell: BTreeMap<(usize, &str, &str, &str), f64> = eps
        .iter()
        .map(|e| {
            (
                (
                    e.n,
                    e.geometry.as_str(),
                    e.dataset.as_str(),
                    e.setting.as_str(),
                ),
                e.delta_eps,
            )
        })
        .collect();

    // region → geometry → paired (ΔR2, Δε).
    let mut paired: BTreeMap<String, BTreeMap<String, Vec<(f64, f64)>>> = BTreeMap::new();
    for row in aggregate::compute_deltas(table) {
        let Some(delta_r2) = row.delta_r2.filter(|d| d.is_finite()) else {
            continue;
        };
        let key = (
            row.n,
            row.geometry.as_str(),
            row.dataset.as_str(),
            row.setting.as_str(),
        );
        let Some(&delta_eps) = by_cell.get(&key).filter(|d| d.is_finite()) else {
            continue;
        };
        paired
            .entry(row.region.clone())
            .or_default()
            .entry(row.geometry.clone())
            .or_default()
            .push((delta_r2, delta_eps));
    }

    let mut out = Vec::new();
    for (region, by_geometry) in &paired {
        // "*" first, then one scope per geometry — the shape `rank_tests` uses.
        let pooled: Vec<(f64, f64)> = by_geometry.values().flatten().copied().collect();
        out.push(agreement_row(region, "*", &pooled));
        for (geometry, pairs) in by_geometry {
            out.push(agreement_row(region, geometry, pairs));
        }
    }
    out
}

fn agreement_row(region: &str, geometry: &str, pairs: &[(f64, f64)]) -> AgreementRow {
    // Sign concordance only asks about pairs that actually have a sign; a ΔR2 of
    // exactly 0 (the setting tied with the baseline) agrees with nothing.
    let signed: Vec<(f64, f64)> = pairs
        .iter()
        .copied()
        .filter(|(a, b)| *a != 0.0 && *b != 0.0)
        .collect();
    let concordant = signed
        .iter()
        .filter(|(a, b)| a.is_sign_positive() == b.is_sign_positive())
        .count();
    let (x, y): (Vec<f64>, Vec<f64>) = pairs.iter().copied().unzip();
    let spearman = stats::spearman(&x, &y);
    AgreementRow {
        region: region.to_string(),
        geometry: geometry.to_string(),
        n_pairs: pairs.len(),
        n_signed: signed.len(),
        sign_concordance: (!signed.is_empty()).then(|| concordant as f64 / signed.len() as f64),
        spearman_rho: spearman.map(|(rho, _)| rho),
        spearman_p: spearman.map(|(_, p)| p),
    }
}

// ─── Recommended configurations ───────────────────────────────────────────────

/// One row of the recommendation table. `params` and `objectives` are
/// name-keyed; a param the trial does not have stays present as `null`, so every
/// record has the same keys.
#[derive(Serialize)]
struct RecRow {
    stem: String,
    setting: String,
    dataset: String,
    n: usize,
    geometry: String,
    region: String,
    share: f64,
    params: BTreeMap<&'static str, Option<f64>>,
    objectives: BTreeMap<&'static str, f64>,
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
                params: PARAMS.iter().map(|p| (*p, record.param(p))).collect(),
                objectives: OBJECTIVES
                    .iter()
                    .map(|o| (*o, objectives.get(*o).copied().unwrap_or(0.0)))
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

    write_jsonl(&args.out, &rows)
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
