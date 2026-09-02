//! Thesis results figures (Experiments 2–5) from the qParEGO sweeps.
//!
//! Port of `analyze_experiments.py`. Produces the figures the results chapter
//! marks with `// TODO: figure:` in `docs/thesis/sections/5results.typ`:
//!
//! * **Exp 2** (`ablation-results`) — stacked Pareto fronts, one panel per
//!   (dataset, geometry), one curve per loss-weight setting.
//! * **Exp 3** (`curvature-magnitude-results`) — median Pareto-front
//!   dimensionless curvature κ = |K|·R_rms² against the data-intrinsic κ_data
//!   (from `results/kappa_data.jsonl`), synthetic vs real markers; plus the
//!   unanchored-vs-`rms_anchored` κ overlay (skipped with a notice if no
//!   rms_anchored runs exist).
//! * **Exp 4** (`manifold-projection-gap`) — two figures. ρ_man-proj(κ): per
//!   cell, the Spearman correlation between each metric's manifold and
//!   2D-projected variants over the cell's trials, against the cell's median κ.
//!   And the projection gap: per Pareto-front trial, `oriented(manifold) −
//!   oriented(2D)` against that trial's own κ, all datasets pooled, with a
//!   Spearman per geometry in each panel. Both are one panel per metric,
//!   hyperbolic vs spherical.
//! * **Exp 5** (`comparison-real-results`) — the 4×3 trustworthiness-vs-stress
//!   Pareto-front grid with convex envelope, and the hyperparameter marginal
//!   histograms of the `all_off` fronts.
//!
//! κ uses **R_rms** (`r_rms`), not R_max — the thesis definition.

pub mod exp2;
pub mod exp3;
pub mod exp4;
pub mod exp5;
pub mod r2_bars;

use std::collections::BTreeMap;
use std::path::Path;

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use crate::cell::{discover_cells, Cell, SYNTH_TRUTH};
use crate::error::{Error, IoContext, Result};
use crate::pareto::pareto_front_records;
use crate::records::{load_jsonl, trial_records, TrialRecord};
use crate::stats::{median, quantile};

/// What a `draw` implementation returns. Boxed rather than [`crate::Error`]
/// because plotters' error type is generic over the backend; [`save`] is the
/// single point where that becomes an [`Error::Plot`].
pub type Res = std::result::Result<(), Box<dyn std::error::Error>>;

/// Turn any backend error into [`Error::Plot`].
fn plot_err<T, E: std::fmt::Display>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| Error::Plot(e.to_string()))
}

/// Every (setting, dataset, N, geometry) cell mapped to its trial records.
pub type CellMap = BTreeMap<Cell, Vec<TrialRecord>>;

// ─── Palette (Okabe-Ito, colourblind-safe, fixed order — never cycled) ────────

pub const OK_BLACK: RGBColor = RGBColor(0, 0, 0);
pub const OK_ORANGE: RGBColor = RGBColor(0xE6, 0x9F, 0x00);
pub const OK_SKYBLUE: RGBColor = RGBColor(0x56, 0xB4, 0xE9);
pub const OK_GREEN: RGBColor = RGBColor(0x00, 0x9E, 0x73);
pub const OK_BLUE: RGBColor = RGBColor(0x00, 0x72, 0xB2);
pub const OK_VERMILLION: RGBColor = RGBColor(0xD5, 0x5E, 0x00);
pub const OK_GREY: RGBColor = RGBColor(0x99, 0x99, 0x99);

pub const SETTING_ORDER: [&str; 5] = [
    "all_off",
    "centering_only",
    "global_only",
    "norm_only",
    "all_free",
];

pub fn setting_color(setting: &str) -> RGBColor {
    match setting {
        "all_off" => OK_BLACK,
        "centering_only" => OK_ORANGE,
        "global_only" => OK_SKYBLUE,
        "norm_only" => OK_GREEN,
        "all_free" => OK_VERMILLION,
        _ => OK_GREY,
    }
}

pub fn geometry_color(geometry: &str) -> RGBColor {
    match geometry {
        "euclidean" => OK_GREY,
        "hyperbolic" => OK_BLUE,
        "spherical" => OK_VERMILLION,
        _ => OK_BLACK,
    }
}

pub const REAL_DATASETS: [&str; 4] = ["mnist", "fashion_mnist", "pbmc", "wordnet_mammals"];

/// The synthetic datasets, *derived* from [`SYNTH_TRUTH`] rather than written
/// out again: the two lists desynced once already (`grid` was added to the sweep
/// grid and to `SYNTH_TRUTH`, but not here, so every figure that iterates
/// [`all_datasets`] silently dropped it), and a hand-kept copy would desync
/// again the next time a generator is added.
pub const SYNTH_DATASETS: [&str; SYNTH_TRUTH.len()] = synth_datasets();

const fn synth_datasets() -> [&'static str; SYNTH_TRUTH.len()] {
    let mut out = [""; SYNTH_TRUTH.len()];
    let mut i = 0;
    while i < SYNTH_TRUTH.len() {
        out[i] = SYNTH_TRUTH[i].0;
        i += 1;
    }
    out
}

pub const CURVED: [&str; 2] = ["hyperbolic", "spherical"];

pub use crate::cell::GEOMETRIES;
/// The five paired metrics (2D vs manifold); Exp 4 plots one panel per metric,
/// in table order.
pub use crate::objectives::METRIC_PAIRS;

/// All datasets, real first — the order Exp 3 iterates in.
pub fn all_datasets() -> Vec<&'static str> {
    REAL_DATASETS.into_iter().chain(SYNTH_DATASETS).collect()
}

/// Trustworthiness (local, ↑) vs normalised stress (global, ↓): the local/global
/// cross-section the thesis uses for the front cross-sections.
///
/// Axis labels stay inside Latin-1 + Greek: the bitmap backend renders through
/// whatever the system resolves "sans-serif" to, and arrows (U+2190/2192) and
/// geometric shapes come out as tofu on this machine. Greek does resolve, so κ
/// and ρ are safe.
pub const X_METRIC: &str = "trustworthiness";
pub const Y_METRIC: &str = "normalized_stress";
pub const X_LABEL: &str = "trustworthiness (higher is better)";
pub const Y_LABEL: &str = "normalised stress (lower is better)";

// ─── Rendering scaffolding ────────────────────────────────────────────────────

/// A figure that can be rendered to any backend, so one definition writes both
/// the SVG (for the thesis) and the PNG (for a quick look).
pub trait Figure {
    fn name(&self) -> String;
    fn size(&self) -> (u32, u32);
    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static;
}

/// Render *fig* to `<out_dir>/<name>.svg` and `.png`.
pub fn save<F: Figure>(fig: &F, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir).at(out_dir)?;
    let name = fig.name();
    let size = fig.size();

    let svg_path = out_dir.join(format!("{name}.svg"));
    {
        let root = SVGBackend::new(&svg_path, size).into_drawing_area();
        plot_err(root.fill(&WHITE))?;
        plot_err(fig.draw(&root))?;
        plot_err(root.present())?;
    }
    let png_path = out_dir.join(format!("{name}.png"));
    {
        let root = BitMapBackend::new(&png_path, size).into_drawing_area();
        plot_err(root.fill(&WHITE))?;
        plot_err(fig.draw(&root))?;
        plot_err(root.present())?;
    }
    Ok(())
}

/// The mesh style shared by every panel: a faint grid, no top/right spines.
///
/// A macro rather than a function because `configure_mesh` hands back a builder
/// whose type names every coordinate parameter; the caller keeps chaining
/// `.x_desc(...)` etc. onto the result.
#[macro_export]
macro_rules! style_mesh {
    ($binding:expr) => {
        $binding
            .light_line_style(plotters::style::TRANSPARENT)
            .bold_line_style(plotters::style::RGBColor(120, 120, 120).mix(0.25))
            .axis_style(plotters::style::RGBColor(80, 80, 80))
            .label_style(
                ("sans-serif", 13)
                    .into_font()
                    .color(&plotters::style::RGBColor(60, 60, 60)),
            )
            .axis_desc_style(
                ("sans-serif", 14)
                    .into_font()
                    .color(&plotters::style::RGBColor(30, 30, 30)),
            )
    };
}

/// One legend entry: label, colour, and the line/marker style that identifies
/// the series (Exp 4 overlays two sample sizes in one colour, so the swatch has
/// to carry the dash and marker distinction too).
pub struct LegendEntry {
    pub label: String,
    pub color: RGBColor,
    /// `(dash, gap)` in pixels, or `None` for a solid line. Exp 2 gives every
    /// setting its own pattern, so the swatch carries the pattern itself rather
    /// than a dashed/not-dashed flag.
    pub dash: Option<(i32, i32)>,
    pub triangle: bool,
}

impl LegendEntry {
    pub fn new(label: impl Into<String>, color: RGBColor) -> Self {
        Self {
            label: label.into(),
            color,
            dash: None,
            triangle: false,
        }
    }

    /// Draw this entry's line with the given `(dash, gap)` pattern.
    pub fn with_dash(mut self, dash: i32, gap: i32) -> Self {
        self.dash = Some((dash, gap));
        self
    }

    /// Mark this entry as the dashed/triangle series (Exp 4's second N).
    pub fn secondary(mut self) -> Self {
        self.dash = Some((8, 6));
        self.triangle = true;
        self
    }
}

/// Draw a horizontal legend strip: a swatch plus a label per entry.
pub fn draw_legend<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    entries: &[LegendEntry],
) -> Res
where
    DB::ErrorType: 'static,
{
    if entries.is_empty() {
        return Ok(());
    }
    let (w, h) = area.dim_in_pixel();
    let font = ("sans-serif", 15).into_font().color(&OK_BLACK);
    // Lay the entries out in equal slots: swatch, then text.
    let slot = w as i32 / entries.len() as i32;
    let y = h as i32 / 2;
    for (i, e) in entries.iter().enumerate() {
        let x0 = i as i32 * slot + 12;
        const SWATCH: i32 = 26;
        match e.dash {
            // Tile the pattern across the swatch, clipped to its width.
            Some((dash, gap)) if dash > 0 && gap > 0 => {
                let mut a = x0;
                while a < x0 + SWATCH {
                    let b = (a + dash).min(x0 + SWATCH);
                    area.draw(&PathElement::new(
                        vec![(a, y), (b, y)],
                        e.color.stroke_width(3),
                    ))?;
                    a = b + gap;
                }
            }
            _ => area.draw(&PathElement::new(
                vec![(x0, y), (x0 + SWATCH, y)],
                e.color.stroke_width(3),
            ))?,
        }
        if e.triangle {
            area.draw(&TriangleMarker::new((x0 + 13, y), 5, e.color.filled()))?;
        } else {
            area.draw(&Circle::new((x0 + 13, y), 4, e.color.filled()))?;
        }
        area.draw(&Text::new(
            e.label.clone(),
            (x0 + 34, y),
            font.clone().pos(Pos::new(HPos::Left, VPos::Center)),
        ))?;
    }
    Ok(())
}

// ─── Loading & shared data helpers ────────────────────────────────────────────

/// Map every (setting, dataset, n, geometry) to its list of trial records.
pub fn load_all_cells(results_dir: &Path) -> Result<CellMap> {
    let mut cells = CellMap::new();
    for cf in discover_cells(results_dir)? {
        cells.insert(cf.cell, trial_records(&cf.path)?);
    }
    Ok(cells)
}

/// One `kappa_data` row: the data-intrinsic curvature under each fitted geometry.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct KappaData {
    pub dataset: String,
    pub n_samples: usize,
    #[serde(default)]
    pub hyp_kappa: Option<f64>,
    #[serde(default)]
    pub sph_kappa: Option<f64>,
}

impl KappaData {
    /// The κ_data to compare against for an embedding of the given geometry.
    pub fn for_geometry(&self, geometry: &str) -> Option<f64> {
        match geometry {
            "hyperbolic" => self.hyp_kappa,
            "spherical" => self.sph_kappa,
            _ => None,
        }
    }
}

/// κ_data records keyed by dataset for sample size *n*.
///
/// Prefers `kappa_data_n{n}.jsonl` and falls back to the unsuffixed
/// `kappa_data.jsonl` (which the local n=1000 run writes), trusting the latter
/// only for the N it was actually run at. An **absent** table is not an error —
/// the κ_data export is a separate optimizer run, and Exp 3 skips its scatter
/// when it has not been done — but a table that is there and will not parse is.
pub fn load_kappa_data(results_dir: &Path, n: usize) -> Result<BTreeMap<String, KappaData>> {
    for name in [format!("kappa_data_n{n}.jsonl"), "kappa_data.jsonl".into()] {
        let rows: Vec<KappaData> = match load_jsonl(results_dir.join(&name)) {
            Ok(rows) => rows,
            Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
                continue
            }
            Err(e) => return Err(e),
        };
        let rows: BTreeMap<String, KappaData> = rows
            .into_iter()
            .filter(|r| r.n_samples == n)
            .map(|r| (r.dataset.clone(), r))
            .collect();
        if !rows.is_empty() {
            return Ok(rows);
        }
    }
    Ok(BTreeMap::new())
}

/// Median κ over the 10-objective Pareto front of *records*.
pub fn median_front_kappa(records: &[TrialRecord]) -> Option<f64> {
    let front = pareto_front_records(records);
    let ks: Vec<f64> = front.iter().filter_map(|r| r.kappa()).collect();
    median(&ks)
}

/// Finite (x, y) pairs of two raw metric columns over *records*.
pub fn finite_xy(records: &[TrialRecord], xm: &str, ym: &str) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for r in records {
        let (Some(x), Some(y)) = (r.objective(xm), r.objective(ym)) else {
            continue;
        };
        if x.is_finite() && y.is_finite() {
            xs.push(x);
            ys.push(y);
        }
    }
    (xs, ys)
}

/// Median of y in log-spaced x bins, for bins with ≥ `min_per_bin` points — the
/// trend curve through a cloud of independent cells.
///
/// `min_per_bin` is the guard against reading a trend off two or three cells: a
/// median over that few flips on a single point, and on a log axis the sparse
/// tail bins are exactly where that happens. Callers set it from how many cells
/// they expect per bin, never below 2.
pub fn binned_median(
    x: &[f64],
    y: &[f64],
    n_bins: usize,
    min_per_bin: usize,
) -> (Vec<f64>, Vec<f64>) {
    let min_per_bin = min_per_bin.max(2);
    if x.len() < min_per_bin {
        return (Vec::new(), Vec::new());
    }
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (lo, hi) = (x_min.log10(), x_max.log10());
    if hi <= lo {
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        return (vec![mean], median(y).into_iter().collect());
    }
    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| 10f64.powf(lo + (hi - lo) * i as f64 / n_bins as f64))
        .collect();
    let mut centres = Vec::new();
    let mut meds = Vec::new();
    for w in edges.windows(2) {
        let (a, b) = (w[0], w[1]);
        let vals: Vec<f64> = x
            .iter()
            .zip(y)
            .filter(|(xv, _)| **xv >= a && **xv <= b)
            .map(|(_, yv)| *yv)
            .collect();
        // `median` is None only on an empty slice, which the length test rules
        // out; `if let` keeps that a fact of the code rather than an unwrap.
        if let (true, Some(m)) = (vals.len() >= min_per_bin, median(&vals)) {
            centres.push((a * b).sqrt());
            meds.push(m);
        }
    }
    (centres, meds)
}

/// Indices of the lower convex hull chain, sorted by x ascending.
///
/// The envelope of the (trustworthiness ↑, stress ↓) trade-off: the lower
/// boundary in stress as trustworthiness grows.
pub fn convex_lower_hull(x: &[f64], y: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut hull: Vec<usize> = Vec::new();
    for i in order {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            // Cross product of (b − a) × (i − a); a right turn pops.
            let cross = (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a]);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(i);
    }
    hull
}

/// A padded axis range covering *values*; `None` when there is nothing to plot.
pub fn padded_range(values: &[f64], frac: f64) -> Option<(f64, f64)> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    let lo = finite.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // A degenerate range still needs a non-zero span for the axis to build.
    //
    // "Degenerate" has to be judged *relatively*: values that agree to all but
    // the last ulp satisfy `hi > lo`, so an exact test leaves a span of ~1e-16
    // and the panel magnifies float rounding noise to full width. Exp 3's
    // hyperbolic κ_data hit exactly this once every dataset pinned at the
    // Wilson cap, where hyp_kappa is the constant HYPERBOLIC_KAPPA_MIN.
    let span = hi - lo;
    let pad = if span > 1e-12 * lo.abs().max(hi.abs()).max(1.0) {
        span * frac
    } else {
        lo.abs().max(1.0) * 0.05
    };
    Some((lo - pad, hi + pad))
}

/// A padded axis range that ignores outliers: [`padded_range`] over the values
/// inside Tukey's fences, `[q1 − 1.5·IQR, q3 + 1.5·IQR]`.
///
/// A single diverged front point used to set a whole Exp 2 panel's scale and
/// squash the informative knee into a sliver. The fence only bites when there
/// really is a far tail, so a well-behaved panel comes out identical to
/// [`padded_range`]; too few points (a 3-point front) or a degenerate IQR fall
/// back to it outright. Callers must handle the points now outside the range —
/// plotters clips them silently.
pub fn robust_range(values: &[f64], frac: f64) -> Option<(f64, f64)> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 8 {
        return padded_range(values, frac);
    }
    let (q1, q3) = (quantile(&finite, 0.25)?, quantile(&finite, 0.75)?);
    let iqr = q3 - q1;
    if iqr <= 0.0 {
        return padded_range(values, frac);
    }
    let (lo, hi) = (q1 - 1.5 * iqr, q3 + 1.5 * iqr);
    let kept: Vec<f64> = finite
        .into_iter()
        .filter(|v| *v >= lo && *v <= hi)
        .collect();
    padded_range(&kept, frac)
}

/// A padded *log* axis range: same idea, in decades, with non-positive values
/// dropped because they cannot be placed on a log axis.
pub fn padded_log_range(values: &[f64], frac: f64) -> Option<(f64, f64)> {
    let logs: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .map(|v| v.log10())
        .collect();
    let (lo, hi) = padded_range(&logs, frac)?;
    Some((10f64.powf(lo), 10f64.powf(hi)))
}

/// Widen a log range outward to whole decades.
///
/// Only for *axis* bounds, never for histogram bins. plotters derives log ticks
/// from the range endpoints, so an endpoint like 9.87e-5 scatters the ticks off
/// the decades; snapping puts them on exact powers of ten. Data spanning less
/// than a decade is left alone — snapping it would squash every point into a
/// sliver of the panel.
pub fn snap_to_decades((lo, hi): (f64, f64)) -> (f64, f64) {
    if !(lo > 0.0 && hi > 0.0) || hi / lo < 10.0 {
        return (lo, hi);
    }
    (
        10f64.powf(lo.log10().floor()),
        10f64.powf(hi.log10().ceil()),
    )
}

/// Tick label for a log axis.
///
/// plotters accumulates float error while walking decades, handing the default
/// formatter values like 9.999999999e-5; anything within a fraction of a percent
/// of a power of ten is printed as that power.
pub fn log_tick(v: &f64) -> String {
    if *v <= 0.0 || !v.is_finite() {
        return String::new();
    }
    let exp = v.log10();
    let rounded = exp.round();
    if (exp - rounded).abs() < 0.01 {
        let k = rounded as i32;
        return if (-4..=5).contains(&k) {
            // Plain decimals read better than exponents in the common range.
            let decimals = (-k).max(0) as usize;
            format!("{:.*}", decimals, 10f64.powi(k))
        } else {
            format!("1e{k}")
        };
    }
    format!("{v:.3}")
}
