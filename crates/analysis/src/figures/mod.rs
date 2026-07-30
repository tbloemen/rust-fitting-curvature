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
//! * **Exp 4** (`manifold-projection-gap`) — ρ_man-proj(κ): per cell, the
//!   Spearman correlation between each metric's manifold and 2D-projected
//!   variants over the cell's trials, plotted against the cell's median κ, one
//!   panel per metric, hyperbolic vs spherical, both N overlaid.
//! * **Exp 5** (`comparison-real-results`) — the 4×3 trustworthiness-vs-stress
//!   Pareto-front grid with convex envelope, and the hyperparameter marginal
//!   histograms of the `all_off` fronts.
//!
//! κ uses **R_rms** (`r_rms`), not R_max — the thesis definition.

pub mod exp2;
pub mod exp3;
pub mod exp4;
pub mod exp5;

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use crate::cell::Cell;
use crate::pareto::pareto_front_records;
use crate::records::{trial_records, TrialRecord};
use crate::stats::median;

pub type Res = Result<(), Box<dyn Error>>;

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
pub const SYNTH_DATASETS: [&str; 4] = ["sphere", "antipodal_clusters", "tree", "hyperbolic_shells"];
pub const GEOMETRIES: [&str; 3] = ["euclidean", "hyperbolic", "spherical"];
pub const CURVED: [&str; 2] = ["hyperbolic", "spherical"];

/// All datasets, real first — the order Exp 3 iterates in.
pub fn all_datasets() -> Vec<&'static str> {
    REAL_DATASETS.into_iter().chain(SYNTH_DATASETS).collect()
}

/// The five paired metrics (2D vs manifold) used in Exp 4.
pub const PAIRED_METRICS: [&str; 5] = [
    "trustworthiness",
    "continuity",
    "normalized_stress",
    "shepard_goodness",
    "neighborhood_hit",
];

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
pub fn save<F: Figure>(fig: &F, out_dir: &Path) -> Res {
    std::fs::create_dir_all(out_dir)?;
    let name = fig.name();
    let size = fig.size();

    let svg_path = out_dir.join(format!("{name}.svg"));
    {
        let root = SVGBackend::new(&svg_path, size).into_drawing_area();
        root.fill(&WHITE)?;
        fig.draw(&root)?;
        root.present()?;
    }
    let png_path = out_dir.join(format!("{name}.png"));
    {
        let root = BitMapBackend::new(&png_path, size).into_drawing_area();
        root.fill(&WHITE)?;
        fig.draw(&root)?;
        root.present()?;
    }
    println!("  wrote {}/{name}.svg (+.png)", out_dir.display());
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
    pub dashed: bool,
    pub triangle: bool,
}

impl LegendEntry {
    pub fn new(label: impl Into<String>, color: RGBColor) -> Self {
        Self {
            label: label.into(),
            color,
            dashed: false,
            triangle: false,
        }
    }

    /// Mark this entry as the dashed/triangle series.
    pub fn secondary(mut self) -> Self {
        self.dashed = true;
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
        if e.dashed {
            for seg in 0..3 {
                let a = x0 + seg * 10;
                area.draw(&PathElement::new(
                    vec![(a, y), (a + 6, y)],
                    e.color.stroke_width(3),
                ))?;
            }
        } else {
            area.draw(&PathElement::new(
                vec![(x0, y), (x0 + 26, y)],
                e.color.stroke_width(3),
            ))?;
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
pub fn load_all_cells(results_dir: &Path) -> std::io::Result<CellMap> {
    let mut cells = CellMap::new();
    let mut paths: Vec<_> = std::fs::read_dir(results_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect();
    paths.sort();
    for path in paths {
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some(cell) = crate::parse_cell_stem(stem) {
            cells.insert(cell, trial_records(&path));
        }
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
/// only for the N it was actually run at.
pub fn load_kappa_data(results_dir: &Path, n: usize) -> BTreeMap<String, KappaData> {
    for name in [format!("kappa_data_n{n}.jsonl"), "kappa_data.jsonl".into()] {
        let path = results_dir.join(&name);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let rows: Vec<KappaData> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<KappaData>(l).ok())
            .filter(|r| r.n_samples == n)
            .collect();
        if !rows.is_empty() {
            return rows.into_iter().map(|r| (r.dataset.clone(), r)).collect();
        }
    }
    BTreeMap::new()
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

/// Median of y in log-spaced x bins, for bins with ≥2 points — the trend curve
/// through a cloud of independent cells.
pub fn binned_median(x: &[f64], y: &[f64], n_bins: usize) -> (Vec<f64>, Vec<f64>) {
    if x.len() < 2 {
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
        if vals.len() >= 2 {
            centres.push((a * b).sqrt());
            meds.push(median(&vals).unwrap());
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
    let pad = if hi > lo {
        (hi - lo) * frac
    } else {
        lo.abs().max(1.0) * 0.05
    };
    Some((lo - pad, hi + pad))
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
