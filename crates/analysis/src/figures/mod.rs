//! Thesis results figures (Experiments 1–4) from the qParEGO sweeps.
//!
//! One module per research question of `docs/thesis/sections/4methods.typ`,
//! numbered as the results chapter numbers them:
//!
//! | module | question | thesis anchor |
//! |---|---|---|
//! | [`exp1`] | Does matching dataset and embedding curvature improve quality? | `synthetic-grid-results` |
//! | [`exp2`] | What effect does curvature have on established metrics? | `metric-results` |
//! | [`exp3`] | Can curvature be tuned as its own hyperparameter? | `curvature-tuning-results` |
//! | [`exp4`] | Can auxiliary global loss terms improve the visualisations? | `ablation-results` |
//!
//! * **Exp 1** — the matched-minus-mismatched R2 gain, one group per synthetic
//!   dataset, one bar per mismatched embedding geometry, plus the ε-indicator
//!   between the same fronts. The figure companions to
//!   `@tab:geometry-match-r2`, read back from the same JSONL, so they and the
//!   thesis table cannot disagree.
//! * **Exp 2** — the metric-response trends: every `[0,1]` metric overlaid
//!   against κ and against |K|, and one panel per unbounded metric on its own
//!   y axis ([`exp2::MetricTrend`], [`exp2::UnboundedTrend`]); and the
//!   Spearman metric-dependence heatmap
//!   ([`exp2_dependence::MetricDependence`]); and the curved-versus-Euclidean
//!   R2 gain under every preference region, datasets by regions, read back
//!   from the stage-1 table ([`exp2_region_gain::RegionGain`]).
//! * **Exp 3** — a skeleton; see the module doc.
//! * **Exp 4** — stacked Pareto fronts, one panel per (dataset, geometry) and
//!   one curve per loss-weight setting; the R2 levels of the same settings
//!   as grouped bar charts under `<out-dir>/experiment_4`; the per-setting
//!   R2 gain over `all_off` as a dot plot, datasets by geometry
//!   ([`exp4_gain_dots::GainDots`]); and the ε-indicator against `all_off`
//!   in the same layout ([`exp4_epsilon_dots::EpsilonDots`]).
//!
//! κ uses **`R_rms`** (`r_rms`), not `R_max` — the thesis definition.
//!
//! **Some κ and log-axis helpers below still have no caller and are kept on
//! purpose.** `KappaData`, [`load_kappa_data`], [`median_front_kappa`],
//! [`binned_median`], [`convex_lower_hull`], [`snap_to_decades`] and
//! [`all_datasets`] were written for figures Exp 2 and Exp 3 will need again.
//! Do not sweep them out as dead code. The rest of that list now has one:
//! Exp 2's κ axis reads [`padded_log_range`], [`log_tick`], [`CURVED`] and the
//! two halves of the binner, [`BinScale::edges`] and [`binned_median_on`].

pub mod dot_panels;
pub mod exp1;
pub mod exp2;
pub mod exp2_dependence;
pub mod exp2_dumbbell;
pub mod exp2_region_gain;
pub mod exp3;
pub mod exp4;
pub mod exp4_epsilon_dots;
pub mod exp4_gain_dots;

use fitting_core::cast::{count_to_f64, to_i32};
use std::collections::BTreeMap;
use std::path::Path;

use plotters::coord::types::RangedCoordf64;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use crate::cell::{discover_cells, Cell, SYNTH_TRUTH};
use crate::error::{Error, IoContext, Result};
use crate::objectives::{resolve_space, ObjectiveSpace};
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

#[must_use]
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

#[must_use]
pub fn geometry_color(geometry: &str) -> RGBColor {
    match geometry {
        "euclidean" => OK_GREY,
        "hyperbolic" => OK_BLUE,
        "spherical" => OK_VERMILLION,
        _ => OK_BLACK,
    }
}

/// One Okabe-Ito colour per objective, indexed by position in [`OBJECTIVES`].
///
/// Six objectives, six colours — `OK_GREY` is left out on purpose, so it stays
/// available as the "not one of these" fallback the way it is for a geometry.
/// Adding a seventh objective would need a seventh distinguishable colour
/// before this list can grow; `metric_colors_are_distinct` fails first.
pub const METRIC_PALETTE: [RGBColor; 6] = [
    OK_BLACK,
    OK_ORANGE,
    OK_SKYBLUE,
    OK_GREEN,
    OK_BLUE,
    OK_VERMILLION,
];

/// Dash pattern per objective, in the same order — so a figure overlaying all
/// six survives a greyscale print, as Exp 4's settings do. `None` is solid.
pub const METRIC_DASH: [Option<(i32, i32)>; 6] = [
    None,
    Some((7, 4)),
    Some((2, 3)),
    Some((11, 4)),
    Some((7, 3)),
    Some((2, 2)),
];

/// The colour a metric is drawn in, by wire name; `OK_GREY` for anything that
/// is not an objective.
#[must_use]
pub fn metric_color(name: &str) -> RGBColor {
    metric_slot(name).map_or(OK_GREY, |i| METRIC_PALETTE[i])
}

/// The `(dash, gap)` pattern a metric is drawn with, or `None` for solid.
#[must_use]
pub fn metric_dash(name: &str) -> Option<(i32, i32)> {
    metric_slot(name).and_then(|i| METRIC_DASH[i])
}

/// Position of *name* in [`OBJECTIVES`], if it is one.
///
/// The two style tables are as long as [`OBJECTIVES`] — pinned by
/// `metric_style_tables_cover_the_objectives` — so the index needs no modulo,
/// and a seventh objective is a compile-time-length mismatch away from being
/// caught rather than a silently reused colour.
fn metric_slot(name: &str) -> Option<usize> {
    OBJECTIVES.iter().position(|m| m.name() == name)
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
/// The five metrics that have both a 2D and a manifold reading, in table order.
/// A *diagnostic* table — no longer the objective list, which is
/// [`OBJECTIVES`]. No figure reads it at present; the manifold-vs-projection
/// pair that did was deleted with the old `exp4.rs`.
pub use crate::objectives::METRIC_PAIRS;
/// The six objectives and the three preference families, in region order;
/// `r2_bars` labels its axes from these.
pub use crate::objectives::{FAMILIES, OBJECTIVES};

/// All datasets, real first.
#[must_use]
pub fn all_datasets() -> Vec<&'static str> {
    REAL_DATASETS.into_iter().chain(SYNTH_DATASETS).collect()
}

/// Trustworthiness (local, ↑) vs normalised stress (global, ↓): the local/global
/// cross-section the thesis uses for the front cross-sections.
///
/// Axis labels stay inside Latin-1 + Greek: SVG text is rendered by the viewer through
/// whatever the system resolves "sans-serif" to, and arrows (U+2190/2192) and
/// geometric shapes come out as tofu on this machine. Greek does resolve, so κ
/// and ρ are safe.
pub const X_METRIC: &str = "trustworthiness";
pub const Y_METRIC: &str = "normalized_stress";
pub const X_LABEL: &str = "trustworthiness (higher is better)";
pub const Y_LABEL: &str = "normalised stress (lower is better)";

// ─── Rendering scaffolding ────────────────────────────────────────────────────

/// A figure that can be rendered to any backend; [`save`] writes the SVG the
/// thesis embeds.
pub trait Figure {
    fn name(&self) -> String;
    fn size(&self) -> (u32, u32);
    /// # Errors
    ///
    /// Returns drawing backend errors.
    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static;
}

/// Render *fig* to `<out_dir>/<name>.svg`.
///
/// SVG only: it is what the thesis embeds, and a PNG twin of every figure
/// doubled the file count of an already crowded `plots/` for a quick look
/// any SVG viewer gives anyway.
///
/// # Errors
///
/// Returns errors from directory creation, backend creation, or drawing.
pub fn save<F: Figure>(fig: &F, out_dir: &Path, space: ObjectiveSpace) -> Result<()> {
    std::fs::create_dir_all(out_dir).at(out_dir)?;
    // The objective-space tag is applied here rather than by each `name()`, so
    // no figure can forget it. Two runs over differently-scored sweeps then
    // write two sets of files instead of one overwriting the other — the same
    // rule the JSONL tables follow.
    let name = format!("{}_{}", fig.name(), space.tag());
    let svg_path = out_dir.join(format!("{name}.svg"));
    let root = SVGBackend::new(&svg_path, fig.size()).into_drawing_area();
    plot_err(root.fill(&WHITE))?;
    plot_err(fig.draw(&root))?;
    plot_err(root.present())?;
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

/// The backend x of a point given as a fraction of the plotting area's width.
///
/// The Exp 1 bar charts and the Exp 2 region-gain heatmap label their columns
/// in a strip split off *below* the chart, which has no coordinate system of
/// its own, so a group or column centre has to be placed by pixel. *`plot_px`*
/// is the chart's horizontal pixel range; the strip's own draw calls are
/// relative to its top-left, so callers subtract the strip's origin from what
/// this returns.
pub(crate) fn plot_x(plot_px: &std::ops::Range<i32>, fraction: f64) -> i32 {
    plot_px.start + to_i32((f64::from(plot_px.end - plot_px.start) * fraction).round())
}

/// One legend entry: label, colour, and the line style that identifies the
/// series.
pub struct LegendEntry {
    pub label: String,
    pub color: RGBColor,
    /// `(dash, gap)` in pixels, or `None` for a solid line. Exp 4 gives every
    /// setting its own pattern, so the swatch carries the pattern itself rather
    /// than a dashed/not-dashed flag.
    pub dash: Option<(i32, i32)>,
}

impl LegendEntry {
    pub fn new(label: impl Into<String>, color: RGBColor) -> Self {
        Self {
            label: label.into(),
            color,
            dash: None,
        }
    }

    /// Draw this entry's line with the given `(dash, gap)` pattern.
    #[must_use]
    pub fn with_dash(mut self, dash: i32, gap: i32) -> Self {
        self.dash = Some((dash, gap));
        self
    }
}

/// Draw a horizontal legend strip: a swatch plus a label per entry.
///
/// # Panics
///
/// Panics if the legend has more than `i32::MAX` entries.
///
/// # Errors
///
/// Returns drawing backend errors.
pub fn draw_legend<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    entries: &[LegendEntry],
) -> Res
where
    DB::ErrorType: 'static,
{
    const SWATCH: i32 = 26;

    if entries.is_empty() {
        return Ok(());
    }
    let (width, height) = area.dim_in_pixel();
    let font = ("sans-serif", 15).into_font().color(&OK_BLACK);
    // Lay the entries out in equal slots: swatch, then text.
    let slot =
        i32::try_from(width).unwrap_or(i32::MAX) / i32::try_from(entries.len()).unwrap_or(i32::MAX);
    let vertical_centre = i32::try_from(height).unwrap_or(i32::MAX) / 2;
    for (i, e) in entries.iter().enumerate() {
        let x0 = i32::try_from(i).expect("legend has fewer than 2^31 entries") * slot + 12;
        match e.dash {
            // Tile the pattern across the swatch, clipped to its width.
            Some((dash, gap)) if dash > 0 && gap > 0 => {
                let mut dash_start = x0;
                while dash_start < x0 + SWATCH {
                    let dash_end = (dash_start + dash).min(x0 + SWATCH);
                    area.draw(&PathElement::new(
                        vec![(dash_start, vertical_centre), (dash_end, vertical_centre)],
                        e.color.stroke_width(3),
                    ))?;
                    dash_start = dash_end + gap;
                }
            }
            _ => area.draw(&PathElement::new(
                vec![(x0, vertical_centre), (x0 + SWATCH, vertical_centre)],
                e.color.stroke_width(3),
            ))?,
        }
        area.draw(&Circle::new(
            (x0 + 13, vertical_centre),
            4,
            e.color.filled(),
        ))?;
        area.draw(&Text::new(
            e.label.clone(),
            (x0 + 34, vertical_centre),
            font.clone().pos(Pos::new(HPos::Left, VPos::Center)),
        ))?;
    }
    Ok(())
}

/// [`draw_legend`], wrapped onto *cols* columns.
///
/// A smaller swatch and font than the single-row form: what makes this exist is
/// Exp 2's half-width panel, where six entries laid out in one row give each
/// about 60 px and the labels collide. Rows are as tall as the area divides
/// into, so the caller sizes the strip by `rows * ~18 px`.
///
/// # Panics
///
/// Panics if the legend has more than `i32::MAX` rows or columns.
///
/// # Errors
///
/// Returns drawing backend errors.
pub fn draw_legend_grid<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    entries: &[LegendEntry],
    cols: usize,
) -> Res
where
    DB::ErrorType: 'static,
{
    const SWATCH: i32 = 18;

    if entries.is_empty() || cols == 0 {
        return Ok(());
    }
    let (width, height) = area.dim_in_pixel();
    let font = ("sans-serif", 12).into_font().color(&OK_BLACK);
    let rows = entries.len().div_ceil(cols);
    let slot = i32::try_from(width).unwrap_or(i32::MAX) / i32::try_from(cols).unwrap_or(i32::MAX);
    let row_height =
        i32::try_from(height).unwrap_or(i32::MAX) / i32::try_from(rows).unwrap_or(i32::MAX);
    for (i, e) in entries.iter().enumerate() {
        let col = i32::try_from(i % cols).expect("a legend has fewer than 2^31 columns");
        let row = i32::try_from(i / cols).expect("a legend has fewer than 2^31 rows");
        let x0 = col * slot + 6;
        let y = row * row_height + row_height / 2;
        match e.dash {
            // Tile the pattern across the swatch, clipped to its width.
            Some((dash, gap)) if dash > 0 && gap > 0 => {
                let mut dash_start = x0;
                while dash_start < x0 + SWATCH {
                    let dash_end = (dash_start + dash).min(x0 + SWATCH);
                    area.draw(&PathElement::new(
                        vec![(dash_start, y), (dash_end, y)],
                        e.color.stroke_width(3),
                    ))?;
                    dash_start = dash_end + gap;
                }
            }
            _ => area.draw(&PathElement::new(
                vec![(x0, y), (x0 + SWATCH, y)],
                e.color.stroke_width(3),
            ))?,
        }
        area.draw(&Text::new(
            e.label.clone(),
            (x0 + SWATCH + 5, y),
            font.clone().pos(Pos::new(HPos::Left, VPos::Center)),
        ))?;
    }
    Ok(())
}

// ─── Loading & shared data helpers ────────────────────────────────────────────

/// Map every (setting, dataset, n, geometry) to its list of trial records.
///
/// # Errors
///
/// Propagates errors from [`discover_cells`] and [`trial_records`].
/// Map every (setting, dataset, n, geometry) to its trial records, with the
/// objective space they were all written in.
///
/// *forced* is the `--objectives` override; `None` reads the space off the
/// sweeps. The space comes back with the cells because every figure that
/// reduces a cell to its Pareto front needs it — the front is a different set
/// in each space, not a rescaling of one.
///
/// # Errors
///
/// Propagates errors from [`discover_cells`], [`trial_records`] and
/// [`resolve_space`].
pub fn load_all_cells(
    results_dir: &Path,
    forced: Option<ObjectiveSpace>,
) -> Result<(CellMap, ObjectiveSpace)> {
    let found = discover_cells(results_dir)?;
    let space = resolve_space(&found, forced)?;
    let mut cells = CellMap::new();
    for cf in found {
        cells.insert(cf.cell, trial_records(&cf.path)?);
    }
    Ok((cells, space))
}

/// Every cell reduced to its Pareto front in *space*.
///
/// The corpus a thesis figure describes is the front, not the sweep, so the
/// figures that pool trials (`exp2`, `exp2_dependence`) take this rather than
/// the sweep `CellMap`. Reduced once, here, so the front and the space it was
/// found in are decided in one place.
#[must_use]
pub fn front_cells(cells: &CellMap, space: ObjectiveSpace) -> CellMap {
    cells
        .iter()
        .map(|(cell, records)| (cell.clone(), pareto_front_records(records, space)))
        .collect()
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
    /// The `κ_data` to compare against for an embedding of the given geometry.
    #[must_use]
    pub fn for_geometry(&self, geometry: &str) -> Option<f64> {
        match geometry {
            "hyperbolic" => self.hyp_kappa,
            "spherical" => self.sph_kappa,
            _ => None,
        }
    }
}

/// `κ_data` records keyed by dataset for sample size *n*.
///
/// Prefers `kappa_data_n{n}.jsonl` and falls back to the unsuffixed
/// `kappa_data.jsonl` (which the local n=1000 run writes), trusting the latter
/// only for the N it was actually run at. An **absent** table is not an error —
/// the `κ_data` export is a separate optimizer run, so a figure that needs it
/// is skipped when it has not been done — but a table that is there and will
/// not parse is.
///
/// # Errors
///
/// Propagates errors from [`load_jsonl`] (file I/O or deserialization).
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

/// Median κ over the Pareto front of *records* in *space*.
#[must_use]
pub fn median_front_kappa(records: &[TrialRecord], space: ObjectiveSpace) -> Option<f64> {
    let front = pareto_front_records(records, space);
    let ks: Vec<f64> = front
        .iter()
        .filter_map(super::records::TrialRecord::kappa)
        .collect();
    median(&ks)
}

/// Finite (x, y) pairs of two raw metric columns over *records*.
#[must_use]
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

/// How a binned axis is spaced: which edges [`BinScale::edges`] lays down, and
/// where in a bin [`binned_median_on`] draws its median.
///
/// The two have to agree. A geometric bin centre on a linear axis puts the
/// point in the wrong place, and on a bin whose lower edge is near zero it
/// collapses the centre onto that edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinScale {
    /// Equal ratios. Non-positive values cannot be placed and are dropped.
    Log,
    /// Equal widths.
    Linear,
}

impl BinScale {
    /// `n_bins + 1` edges spanning the usable values of *x*, or `None` when
    /// they do not span a range at all — none usable, or every one of them
    /// equal.
    ///
    /// Split out of [`binned_median`] so that several *y* series can be binned
    /// on **one** set of edges. Deriving the edges per series is wrong the
    /// moment two series drop different rows (a metric a diverged trial did not
    /// record): the two polylines are then sampled at different x positions and
    /// cannot be read against each other, which is exactly what Exp 2 overlays
    /// them to do.
    #[must_use]
    pub fn edges(self, x: &[f64], n_bins: usize) -> Option<Vec<f64>> {
        let usable: Vec<f64> = x
            .iter()
            .copied()
            .filter(|v| v.is_finite() && (self == BinScale::Linear || *v > 0.0))
            .collect();
        // Ruled out first so neither fold below can return an infinity, which
        // is what would leave `hi` a NaN and the comparison undecidable.
        let (Some(&first), Some(&last)) = (usable.first(), usable.last()) else {
            return None;
        };
        let x_min = usable.iter().copied().fold(first, f64::min);
        let x_max = usable.iter().copied().fold(last, f64::max);
        let (lo, hi) = match self {
            BinScale::Log => (x_min.log10(), x_max.log10()),
            BinScale::Linear => (x_min, x_max),
        };
        if hi <= lo {
            return None;
        }
        Some(
            (0..=n_bins)
                .map(|i| {
                    let t = lo + (hi - lo) * count_to_f64(i) / count_to_f64(n_bins);
                    match self {
                        BinScale::Log => 10f64.powf(t),
                        BinScale::Linear => t,
                    }
                })
                .collect(),
        )
    }

    /// Where in a bin its median is drawn: the geometric mean of the edges on a
    /// log axis, the arithmetic mean on a linear one. Either way it is the
    /// point that sits mid-bin *as the axis renders it*.
    fn centre(self, a: f64, b: f64) -> f64 {
        match self {
            BinScale::Log => (a * b).sqrt(),
            BinScale::Linear => (a + b) / 2.0,
        }
    }
}

/// Median of *y* per bin of *edges*, for bins holding at least `min_per_bin`
/// points, drawn at the bin centre *scale* defines.
///
/// *scale* must be the one *edges* came from — see [`BinScale`].
///
/// Bins are closed at both ends, so a point sitting exactly on an interior edge
/// counts in the two bins that share it. That is deliberate — the alternative
/// drops a point from the last bin — and it is invisible on continuous data.
///
/// `x` and `y` are read pairwise, so they must be the same length; a shorter
/// one simply truncates the pairing.
#[must_use]
pub fn binned_median_on(
    edges: &[f64],
    scale: BinScale,
    x: &[f64],
    y: &[f64],
    min_per_bin: usize,
) -> (Vec<f64>, Vec<f64>) {
    let min_per_bin = min_per_bin.max(2);
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
            centres.push(scale.centre(a, b));
            meds.push(m);
        }
    }
    (centres, meds)
}

/// One bin's drawn quantiles, at the x position [`BinScale::centre`] puts them
/// at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinBand {
    /// Where in the bin the three values are drawn.
    pub centre: f64,
    /// The bin's median — the trend line's point.
    pub median: f64,
    /// First quartile; the band's lower edge.
    pub lo: f64,
    /// Third quartile; the band's upper edge.
    pub hi: f64,
}

/// Median and interquartile range of *y* per bin of *edges*, **one entry per
/// bin**: `None` where the bin holds fewer than `min_per_bin` points.
///
/// [`binned_median_on`] with a band around the line, and with the skipped bins
/// still in the result. That last part is the difference that matters: a caller
/// drawing a polyline needs to tell an unsampled bin from a sampled one, so it
/// can break the curve there rather than run a straight segment across a range
/// nothing was measured in. Bin membership, the centre and the `min_per_bin`
/// floor are all [`binned_median_on`]'s.
///
/// **Quartiles, not mean ± SD.** These bins hold bounded, skewed, often bimodal
/// readings — a collapsed embedding piles every metric at one end — so a
/// symmetric band around a mean both misplaces its centre and, on a `[0, 1]`
/// metric, runs off the axis. `stats::quantile` at `q = 0.5` is `stats::median`
/// exactly, so the line is the one [`binned_median_on`] would draw.
#[must_use]
pub fn binned_band_on(
    edges: &[f64],
    scale: BinScale,
    x: &[f64],
    y: &[f64],
    min_per_bin: usize,
) -> Vec<Option<BinBand>> {
    let min_per_bin = min_per_bin.max(2);
    edges
        .windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            let vals: Vec<f64> = x
                .iter()
                .zip(y)
                .filter(|(xv, _)| **xv >= a && **xv <= b)
                .map(|(_, yv)| *yv)
                .collect();
            if vals.len() < min_per_bin {
                return None;
            }
            // Every `quantile` here is `Some`: the slice is non-empty and the
            // three q are in range. `?` keeps that a fact of the code.
            Some(BinBand {
                centre: scale.centre(a, b),
                median: quantile(&vals, 0.5)?,
                lo: quantile(&vals, 0.25)?,
                hi: quantile(&vals, 0.75)?,
            })
        })
        .collect()
}

/// Median of y in log-spaced x bins, for bins with ≥ `min_per_bin` points — the
/// trend curve through a cloud of independent cells.
///
/// [`BinScale::edges`] followed by [`binned_median_on`]. Callers overlaying more
/// than one series must call those two directly, so every series lands on the
/// same edges.
///
/// `min_per_bin` is the guard against reading a trend off two or three cells: a
/// median over that few flips on a single point, and on a log axis the sparse
/// tail bins are exactly where that happens. Callers set it from how many cells
/// they expect per bin, never below 2.
#[must_use]
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
    let Some(edges) = BinScale::Log.edges(x, n_bins) else {
        // One x value (or none on a log axis): there is no axis to spread the
        // set over, so report it as the single point it is.
        let mean = x.iter().sum::<f64>() / count_to_f64(x.len());
        return (vec![mean], median(y).into_iter().collect());
    };
    binned_median_on(&edges, BinScale::Log, x, y, min_per_bin)
}

/// Indices of the lower convex hull chain, sorted by x ascending.
///
/// The envelope of the (trustworthiness ↑, stress ↓) trade-off: the lower
/// boundary in stress as trustworthiness grows.
#[must_use]
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
    let lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // A degenerate range still needs a non-zero span for the axis to build.
    //
    // "Degenerate" has to be judged *relatively*: values that agree to all but
    // the last ulp satisfy `hi > lo`, so an exact test leaves a span of ~1e-16
    // and the panel magnifies float rounding noise to full width. Exp 3's
    // The deleted κ_data scatter hit exactly this once every hyperbolic
    // dataset pinned at the Wilson cap, where hyp_kappa is the constant
    // HYPERBOLIC_KAPPA_MIN.
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
/// A single diverged front point used to set a whole Exp 4 panel's scale and
/// squash the informative knee into a sliver. The fence only bites when there
/// really is a far tail, so a well-behaved panel comes out identical to
/// [`padded_range`]; too few points (a 3-point front) or a degenerate IQR fall
/// back to it outright. Callers must handle the points now outside the range —
/// plotters clips them silently.
#[must_use]
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
#[must_use]
pub fn padded_log_range(values: &[f64], frac: f64) -> Option<(f64, f64)> {
    let logs: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .map(|v| v.log10())
        .collect();
    let (lo, hi) = padded_range(&logs, frac)?;
    Some((10f64.powf(lo), 10f64.powf(hi)))
}

/// A padded log range over the values above Tukey's *lower* fence in decades —
/// [`robust_range`]'s rule, applied at one end only.
///
/// For a non-negative ratio metric the far tail is one-sided: a collapsed
/// embedding drives `dunn_index` and `davies_bouldin_ratio` to exactly zero and
/// `cluster_density_measure` to ~1e-34, thirty decades under the body, while a
/// high reading is the metric doing what it measures. A two-sided fence would
/// cut the high end along with the floor — Exp 2's `cluster_density_measure`
/// rises two decades in its last κ bin, and that bin is the finding. So the
/// upper bound is the largest value, padded; only the floor is fenced. Values
/// at or below zero cannot be placed on a log axis and are dropped first.
#[must_use]
pub fn log_range_above_floor(values: &[f64], frac: f64) -> Option<(f64, f64)> {
    let logs: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .map(|v| v.log10())
        .collect();
    let (lo, hi) = if logs.len() < 8 {
        padded_range(&logs, frac)?
    } else {
        let (q1, q3) = (quantile(&logs, 0.25)?, quantile(&logs, 0.75)?);
        let iqr = q3 - q1;
        let floor = if iqr > 0.0 {
            q1 - 1.5 * iqr
        } else {
            f64::NEG_INFINITY
        };
        let kept: Vec<f64> = logs.into_iter().filter(|v| *v >= floor).collect();
        padded_range(&kept, frac)?
    };
    Some((10f64.powf(lo), 10f64.powf(hi)))
}

/// Widen a log range outward to whole decades.
///
/// Only for *axis* bounds, never for histogram bins. plotters derives log ticks
/// from the range endpoints, so an endpoint like 9.87e-5 scatters the ticks off
/// the decades; snapping puts them on exact powers of ten. Data spanning less
/// than a decade is left alone — snapping it would squash every point into a
/// sliver of the panel.
#[must_use]
pub fn snap_to_decades((lo, hi): (f64, f64)) -> (f64, f64) {
    if !(lo > 0.0 && hi > 0.0) || hi / lo < 10.0 {
        return (lo, hi);
    }
    (
        10f64.powf(lo.log10().floor()),
        10f64.powf(hi.log10().ceil()),
    )
}

/// A linear axis whose ticks are computed so the last one is not lost.
///
/// plotters' own f64 key points walk the axis by *adding* the step: with
/// `2.6 + 0.2 + 0.2 + 0.2 = 3.2000000000000006` the accumulated value overshoots
/// the exact `3.2` it compares against by more than `f64::EPSILON`, and the
/// tick at the right end of the axis is dropped. Exp 2's spherical panel lost
/// its `3.2` that way. Here every tick is an integer multiple of the step, so
/// the k-th tick is `k · step` however many precede it.
///
/// The step is the smallest of `1, 2, 5 × 10^e` that fits at most `max` ticks
/// inside the range, the same 1-2-5 ladder plotters climbs. It is its own
/// [`Ranged`] coordinate rather than a `with_key_points` binding because
/// plotters 0.3.7 gives that combinator no `ValueFormatter` over an f64 range,
/// so `configure_mesh` will not accept it. Label the axis with
/// [`LinearTicks::label`], which prints exactly the decimals the step needs —
/// `k · step` is `3.2000000000000004`, and the default formatter would say so.
#[derive(Debug, Clone)]
pub struct LinearTicks {
    range: std::ops::Range<f64>,
    ticks: Vec<f64>,
    /// Decimals the step needs: one for `0.2`, none for `10`.
    decimals: usize,
}

impl LinearTicks {
    /// Up to `max` ticks inside `(lo, hi)`; none if the range is not finite or
    /// has no width — the axis still builds, unlabelled.
    #[must_use]
    pub fn new((lo, hi): (f64, f64), max: usize) -> Self {
        let mut out = Self {
            range: lo..hi,
            ticks: Vec::new(),
            decimals: 0,
        };
        let span = hi - lo;
        if !(span.is_finite() && span > 0.0) || max == 0 {
            return out;
        }
        let exponent = (span / count_to_f64(max)).log10().floor();
        // Climb the ladder until the tick count fits. `10` is the next decade's
        // `1`, and the coarsest step that can be needed: one decade up, at most
        // `max` ticks always fit.
        for mantissa in [1.0, 2.0, 5.0, 10.0] {
            let step = mantissa * 10f64.powf(exponent);
            let k_lo = (lo / step).ceil();
            let k_hi = (hi / step).floor();
            if k_hi - k_lo + 1.0 <= count_to_f64(max) {
                out.decimals = usize::try_from(-to_i32(step.log10().floor())).unwrap_or(0);
                let mut k = k_lo;
                while k <= k_hi {
                    // `ceil` of a slightly negative lower bound is -0.0, and
                    // `-0.0 * step` would print as "-0".
                    let v = k * step;
                    out.ticks.push(if v == 0.0 { 0.0 } else { v });
                    k += 1.0;
                }
                return out;
            }
        }
        out
    }

    /// The tick positions, in axis order.
    #[must_use]
    pub fn ticks(&self) -> &[f64] {
        &self.ticks
    }

    /// The tick's label, at the step's own precision.
    #[must_use]
    pub fn label(&self, v: &f64) -> String {
        format!("{:.*}", self.decimals, v)
    }
}

impl Ranged for LinearTicks {
    type FormatOption = plotters::coord::ranged1d::DefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        RangedCoordf64::from(self.range.clone()).map(value, limit)
    }

    fn key_points<Hint: plotters::coord::ranged1d::KeyPointHint>(&self, _hint: Hint) -> Vec<f64> {
        self.ticks.clone()
    }

    fn range(&self) -> std::ops::Range<f64> {
        self.range.clone()
    }
}

/// A log axis whose ticks are chosen to fit, not one per decade.
///
/// plotters labels a log axis at powers of ten and nowhere else unless it is
/// asked for ten times as many ticks as there are decades — so a y axis
/// spanning 1.2 decades, which is what `dunn_index`'s bin medians do, gets one
/// label. This climbs the same ladder the other way: the `1, 2, 5 × 10^k` marks
/// inside the range if at most `max` of them fit, else the decades, else every
/// second, third, … decade. Like [`LinearTicks`] it is its own [`Ranged`]
/// coordinate; the mapping is the one plotters' `LogCoord` does, linear in
/// `ln`. Label it with [`LogTicks::label`].
#[derive(Debug, Clone)]
pub struct LogTicks {
    range: std::ops::Range<f64>,
    ticks: Vec<f64>,
}

impl LogTicks {
    /// Up to `max` ticks inside `(lo, hi)`; none when the range is not a
    /// positive, finite, non-empty one — the axis still builds, unlabelled.
    #[must_use]
    pub fn new((lo, hi): (f64, f64), max: usize) -> Self {
        let mut out = Self {
            range: lo..hi,
            ticks: Vec::new(),
        };
        if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && hi > lo) || max == 0 {
            return out;
        }
        let k_lo = to_i32(lo.log10().floor());
        let k_hi = to_i32(hi.log10().ceil());
        let inside = |v: f64| v >= lo && v <= hi;
        let fine: Vec<f64> = (k_lo..=k_hi)
            .flat_map(|k| [1.0, 2.0, 5.0].map(|m| m * 10f64.powi(k)))
            .filter(|v| inside(*v))
            .collect();
        if fine.len() <= max {
            out.ticks = fine;
            return out;
        }
        let decades: Vec<f64> = (k_lo..=k_hi)
            .map(|k| 10f64.powi(k))
            .filter(|v| inside(*v))
            .collect();
        // Every `stride`-th decade, the smallest stride that fits.
        let stride = decades.len().div_ceil(max).max(1);
        out.ticks = decades.into_iter().step_by(stride).collect();
        out
    }

    /// The tick positions, in axis order.
    #[must_use]
    pub fn ticks(&self) -> &[f64] {
        &self.ticks
    }

    /// The tick's label: plain decimals in the common range (`0.002`, `200`),
    /// `2e7` outside it. Same range rule as [`log_tick`].
    #[must_use]
    pub fn label(&self, v: &f64) -> String {
        if !(*v > 0.0 && v.is_finite()) {
            return String::new();
        }
        let k = to_i32(v.log10().floor());
        if (-4..=5).contains(&k) {
            let decimals = usize::try_from((-k).max(0)).unwrap_or(0);
            format!("{v:.decimals$}")
        } else {
            let mantissa = v / 10f64.powi(k);
            format!("{}e{k}", mantissa.round())
        }
    }
}

impl Ranged for LogTicks {
    type FormatOption = plotters::coord::ranged1d::DefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        // What `LogCoord<f64>` does: a linear axis over `ln`. Callers never
        // map a non-positive value — it has no place on this axis.
        RangedCoordf64::from(self.range.start.ln()..self.range.end.ln()).map(&value.ln(), limit)
    }

    fn key_points<Hint: plotters::coord::ranged1d::KeyPointHint>(&self, _hint: Hint) -> Vec<f64> {
        self.ticks.clone()
    }

    fn range(&self) -> std::ops::Range<f64> {
        self.range.clone()
    }
}

/// Tick label for a log axis.
///
/// plotters accumulates float error while walking decades, handing the default
/// formatter values like 9.999999999e-5; anything within a fraction of a percent
/// of a power of ten is printed as that power.
///
/// # Panics
///
/// Panics if the rounded exponent fails to convert to `usize`.
#[must_use]
pub fn log_tick(v: &f64) -> String {
    if *v <= 0.0 || !v.is_finite() {
        return String::new();
    }
    let exp = v.log10();
    let rounded = exp.round();
    if (exp - rounded).abs() < 0.01 {
        let k = to_i32(rounded);
        return if (-4..=5).contains(&k) {
            // Plain decimals read better than exponents in the common range.
            let decimals =
                usize::try_from((-k).max(0)).expect("a rounded log10 exponent is a small integer");
            format!("{:.*}", decimals, 10f64.powi(k))
        } else {
            format!("1e{k}")
        };
    }
    format!("{v:.3}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_ticks_fill_a_short_range_and_thin_a_long_one() {
        // 1.2 decades: the 1-2-5 marks fit.
        let t = LogTicks::new((5e-4, 8e-3), 5);
        assert_eq!(t.ticks(), &[5e-4, 1e-3, 2e-3, 5e-3]);
        assert_eq!(t.label(&2e-3), "0.002");
        // 1.3 decades above one: `200`, not `200.000`.
        let t = LogTicks::new((20.0, 400.0), 5);
        assert_eq!(t.ticks(), &[20.0, 50.0, 100.0, 200.0]);
        assert_eq!(t.label(&200.0), "200");
        // Seven decades: every second decade, as plotters would.
        let t = LogTicks::new((10.0, 1e8), 5);
        assert_eq!(t.ticks(), &[10.0, 1e3, 1e5, 1e7]);
        assert_eq!(t.label(&1e7), "1e7");
        assert_eq!(LogTicks::new((0.0, 1.0), 5).ticks(), &[] as &[f64]);
    }

    /// The split has to be behaviour-preserving on the input `binned_median`
    /// was written for: positive, spanning several decades.
    #[test]
    fn the_split_binner_reproduces_binned_median() {
        let x: Vec<f64> = (0..400)
            .map(|i| 10f64.powf(-3.0 + f64::from(i) / 80.0))
            .collect();
        let y: Vec<f64> = x.iter().map(|v| v.log10() * 0.1 + 0.5).collect();

        let (want_x, want_y) = binned_median(&x, &y, 10, 4);
        let edges = BinScale::Log
            .edges(&x, 10)
            .expect("five decades span a log range");
        let (got_x, got_y) = binned_median_on(&edges, BinScale::Log, &x, &y, 4);

        assert_eq!(want_x, got_x);
        assert_eq!(want_y, got_y);
        assert!(!got_x.is_empty(), "400 points over 10 bins fill every bin");
    }

    /// The property the split exists for: two series that dropped different
    /// rows still land on the same bin centres, so their polylines are sampled
    /// at the same x and can be overlaid.
    #[test]
    fn two_series_of_different_length_share_bin_centres() {
        let x: Vec<f64> = (0..400)
            .map(|i| 10f64.powf(-3.0 + f64::from(i) / 80.0))
            .collect();
        let y: Vec<f64> = x.iter().map(|_| 0.5).collect();
        let edges = BinScale::Log
            .edges(&x, 8)
            .expect("five decades span a log range");

        // The second series is missing every third trial, as a metric a
        // diverged trial did not record would be.
        let thinned: Vec<(f64, f64)> = x
            .iter()
            .zip(&y)
            .enumerate()
            .filter(|(i, _)| i % 3 != 0)
            .map(|(_, (a, b))| (*a, *b))
            .collect();
        let tx: Vec<f64> = thinned.iter().map(|p| p.0).collect();
        let ty: Vec<f64> = thinned.iter().map(|p| p.1).collect();

        let (full_centres, _) = binned_median_on(&edges, BinScale::Log, &x, &y, 4);
        let (thin_centres, _) = binned_median_on(&edges, BinScale::Log, &tx, &ty, 4);
        assert_eq!(full_centres, thin_centres);

        // …which is exactly what deriving the edges per series would break.
        let (_, own_edges_y) = binned_median(&tx, &ty, 8, 4);
        let own = BinScale::Log
            .edges(&tx, 8)
            .expect("the thinned series still spans decades");
        assert_ne!(
            own, edges,
            "the thinned series has its own min, hence its own edges"
        );
        assert_eq!(own_edges_y.len(), thin_centres.len());
    }

    /// The band's line is the curve `binned_median_on` draws, and its edges
    /// bracket that line — on the same edges, so the two can be read as one
    /// figure.
    #[test]
    fn a_band_brackets_the_median_on_the_same_centres() {
        // Ten points per bin, spread over the unit interval so the quartiles
        // are strictly inside it.
        let x: Vec<f64> = (0..100).map(|i| 1.0 + f64::from(i) / 10.0).collect();
        let y: Vec<f64> = (0..100).map(|i| f64::from(i % 10) / 10.0).collect();
        let edges = BinScale::Linear.edges(&x, 5).expect("x spans a range");

        let bands = binned_band_on(&edges, BinScale::Linear, &x, &y, 4);
        let (centres, meds) = binned_median_on(&edges, BinScale::Linear, &x, &y, 4);

        assert_eq!(bands.len(), 5, "one entry per bin, drawn or not");
        let drawn: Vec<BinBand> = bands.into_iter().flatten().collect();
        assert_eq!(drawn.len(), centres.len(), "the same bins clear the floor");
        for (band, (c, m)) in drawn.iter().zip(centres.iter().zip(&meds)) {
            assert!((band.centre - c).abs() < f64::EPSILON);
            assert!((band.median - m).abs() < f64::EPSILON);
            assert!(
                band.lo <= band.median && band.median <= band.hi,
                "Q1 {} <= median {} <= Q3 {}",
                band.lo,
                band.median,
                band.hi
            );
            assert!(band.lo < band.hi, "a spread bin has a band with width");
        }
    }

    /// A bin under the floor is `None` **at its own index**, which is what lets
    /// a caller break its curve there rather than join the bins either side.
    #[test]
    fn a_sparse_bin_is_a_hole_at_its_own_index() {
        // Bins 0 and 2 hold plenty; bin 1 holds two points.
        let mut x: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.015).collect();
        x.extend([0.4, 0.45]);
        x.extend((0..20).map(|i| 0.7 + f64::from(i) * 0.014));
        let y: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
        let edges: Vec<f64> = (0..=3).map(|i| f64::from(i) / 3.0).collect();

        let bands = binned_band_on(&edges, BinScale::Linear, &x, &y, 5);
        assert_eq!(bands.len(), 3);
        assert!(bands[0].is_some());
        assert!(bands[1].is_none(), "two points do not make a quartile");
        assert!(bands[2].is_some());
    }

    #[test]
    fn log_bin_edges_rejects_a_range_it_cannot_place() {
        assert!(BinScale::Log.edges(&[], 4).is_none(), "nothing to span");
        assert!(
            BinScale::Log.edges(&[-1.0, 0.0], 4).is_none(),
            "no positive value"
        );
        assert!(
            BinScale::Log.edges(&[3.0, 3.0, 3.0], 4).is_none(),
            "one x, no axis"
        );
        assert!(BinScale::Log.edges(&[1.0, 100.0], 4).is_some());
    }

    /// The linear arm spaces its edges by width, keeps the non-positive values
    /// a log axis has to drop, and draws each median at the arithmetic centre.
    #[test]
    fn linear_bins_are_equal_width_and_centred_arithmetically() {
        let edges = BinScale::Linear
            .edges(&[0.0, 4.0], 4)
            .expect("two distinct values span a linear range");
        assert_eq!(edges, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        // Zero and negatives are values on a linear axis and not on a log
        // one, which is why the two arms disagree about the same input: Log
        // drops the 0.0 and is left with a single point to span.
        assert!(BinScale::Log.edges(&[0.0, 4.0], 4).is_none());
        assert!(BinScale::Log.edges(&[0.0, 1.0, 4.0], 4).is_some());
        assert!(BinScale::Linear.edges(&[-2.0, 2.0], 4).is_some());
        assert!(BinScale::Log.edges(&[-2.0, 0.0], 4).is_none());

        let x = [0.5, 0.6, 0.7, 3.5, 3.6, 3.7];
        let y = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let (centres, meds) = binned_median_on(&edges, BinScale::Linear, &x, &y, 3);
        assert_eq!(centres, vec![0.5, 3.5], "arithmetic, not geometric");
        assert_eq!(meds, vec![2.0, 20.0]);

        // The same bin under the log rule would be drawn somewhere else.
        assert!((BinScale::Log.centre(1.0, 4.0) - 2.0).abs() < 1e-12);
        assert!((BinScale::Linear.centre(1.0, 4.0) - 2.5).abs() < 1e-12);
    }

    /// The case plotters gets wrong: four 0.2 steps from 2.6 land on 3.2, which
    /// its accumulating walk overshoots and drops. The spherical Exp 2 axis.
    #[test]
    fn linear_ticks_keep_the_last_tick() {
        let t = LinearTicks::new((2.46, 3.27), 4);
        let labels: Vec<String> = t.ticks().iter().map(|v| t.label(v)).collect();
        assert_eq!(labels, ["2.6", "2.8", "3.0", "3.2"]);

        // Hyperbolic linear panel: whole tens, no decimals, and the padded
        // lower bound just below zero must not produce a "-0".
        let t = LinearTicks::new((-1.1, 37.0), 4);
        let labels: Vec<String> = t.ticks().iter().map(|v| t.label(v)).collect();
        assert_eq!(labels, ["0", "10", "20", "30"]);

        // Never more than asked for, and a degenerate range builds unlabelled.
        for hi in [3.0, 3.21, 3.5, 4.0, 9.9] {
            assert!(
                LinearTicks::new((2.46, hi), 4).ticks().len() <= 4,
                "hi={hi}"
            );
        }
        assert!(LinearTicks::new((1.0, 1.0), 4).ticks().is_empty());
    }

    /// Two metrics sharing a colour would be unreadable, and the style tables
    /// are indexed by position, so they must be as long as the objective list.
    #[test]
    fn metric_style_tables_cover_the_objectives() {
        assert_eq!(METRIC_PALETTE.len(), OBJECTIVES.len());
        assert_eq!(METRIC_DASH.len(), OBJECTIVES.len());
    }

    #[test]
    fn metric_colors_are_distinct() {
        let rgb = |c: RGBColor| (c.0, c.1, c.2);
        for (i, a) in OBJECTIVES.iter().enumerate() {
            assert_ne!(
                rgb(metric_color(a.name())),
                rgb(OK_GREY),
                "{} is an objective and must not fall through to the grey default",
                a.name()
            );
            for b in &OBJECTIVES[i + 1..] {
                assert_ne!(
                    rgb(metric_color(a.name())),
                    rgb(metric_color(b.name())),
                    "{} and {} share a colour",
                    a.name(),
                    b.name()
                );
            }
        }
    }
}
