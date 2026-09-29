//! Experiment 4 — how far the 2D projection's reading of a metric drifts from
//! the manifold's, against curvature.
//!
//! One panel per [`METRIC_PAIRS`] row: the oriented gap
//! `oriented(manifold) − oriented(2D)` of every Pareto-front trial, against
//! that trial's own κ, all datasets and settings pooled. Restored from
//! `b43c731`, which deleted it once the argument for dropping the manifold
//! objectives was settled.
//!
//! κ is kept per trial rather than summarised per cell: it is a swept
//! hyperparameter (`curvature_magnitude`), so it varies more *within* a cell
//! than between cells — on the hyperbolic arm the mean within-cell spread of
//! log₁₀κ is ~2.5 decades against ~1.7–2.1 between cell medians.
//!
//! `distance_consistency` has no panel: it has no manifold twin (see its
//! registry entry), so it is not in [`METRIC_PAIRS`].

use fitting_core::cast::count_to_f64;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use super::exp2::PANEL;
use super::{
    binned_median, draw_legend, geometry_color, log_tick, padded_range, CellMap, Figure,
    LegendEntry, LinearTicks, ObjectiveSpace, Res, CURVED, METRIC_PAIRS, OK_BLACK,
};
use crate::objectives::{oriented, N_METRIC_PAIRS};
use crate::pareto::pareto_front_records;
use crate::stats::{quantile, spearman};
use crate::style_mesh;

/// Fraction of each panel's points kept outside the y range, split evenly
/// between the tails. plotters clips silently, so the range is a *choice about
/// what to hide*: the gap distributions are spikes at 0 with long one-sided
/// tails (`shepard_goodness` runs to −0.997), and a full-range axis squeezes
/// the body of every panel into a few pixels.
const Y_TAIL: f64 = 0.005;

/// Points a κ bin needs before its median joins the trend line. High because
/// these are *front points*, not cells — tens of thousands of them per
/// geometry.
const GAP_MIN_PER_BIN: usize = 30;
const GAP_N_BINS: usize = 12;

/// Fraction of the data's y span added above it, as a clear band for the two ρ
/// annotations.
const HEAD_ROOM: f64 = 0.22;

/// How far, in decades, a κ may sit outside the x frame before the frame grows
/// by a decade to hold it. 0.01 decades is 2.3 % in κ and under a pixel at
/// panel width.
const DECADE_SLACK: f64 = 0.01;

/// The grid the panels are set in: two [`PANEL`]-sized columns, so the figure
/// is exactly the 740 px the other full-width thesis figures are and sits at
/// the A4 text width. One cell is left over for [`draw_key`].
const GRID_COLS: usize = 2;
const GRID_ROWS: usize = (N_METRIC_PAIRS + 1).div_ceil(GRID_COLS);

/// Most points *drawn* per geometry per panel.
///
/// Only the scatter is thinned — ρ and the binned median are computed over
/// every point, and the panel says which n the ρ is over. Without this the SVG
/// carries one `<circle>` per point per panel: 136k of them, 12 MB, which no
/// thesis build wants to embed. The sample is a fixed stride through the points
/// in cell order, so it is deterministic and spread over every cell rather than
/// favouring the ones discovered first.
const DRAW_CAP: usize = 2500;

/// One Pareto-front trial: its κ and its per-metric projection gap.
struct GapPoint {
    kappa: f64,
    /// `None` where either variant of that metric is missing or non-finite.
    /// **Not** [`oriented`]'s 0.0 substitution: that maps an absent value
    /// to "worst possible", which here would manufacture a gap of ±1 out of a
    /// missing column rather than dropping the point.
    gaps: [Option<f64>; N_METRIC_PAIRS],
}

/// Experiment 4 — the *size* of the manifold-vs-projection disagreement
/// against curvature, one point per Pareto-front trial.
///
/// It asks how far apart the two readings of the same configuration are, and
/// whether that distance grows with κ. Three deliberate choices:
///
/// - **Front points, not all trials.** The front is the set a practitioner
///   would actually choose from, and it is where a wrong reading costs
///   something. It also drops the diverged tail, whose gaps are noise.
/// - **All datasets and settings pooled**, so a panel is a population of front
///   points rather than a population of cells. The scatter is therefore dense
///   (thousands of points per geometry) and drawn with heavy alpha.
/// - **The gap is oriented**, `oriented(manifold) − oriented(2D)`, so positive
///   always means the 2D reading is the *pessimistic* one, on all five metrics
///   including `normalized_stress` where the raw value runs the other way.
///
/// Euclidean cells are excluded ([`CURVED`]): there the projection is the
/// identity, so the gap is 0 by construction, and their κ is exactly 0, which
/// has no place on a log axis.
///
/// κ is [`TrialRecord::kappa`](crate::records::TrialRecord::kappa), gauged by
/// `r_gyration`, so only sweeps that log it have points: every file under
/// `results/` predates it and yields an empty figure, which is then not
/// written. Point `--results-dir` at `results-rgyr/`.
///
/// [`ProjGap::zoomed`] restricts the figure to κ above a floor. That is a
/// restriction of the **population**, not just of the axis: ρ, the trend line,
/// the y range and the reported n all describe the points that survive it, so
/// the panel never quotes a statistic over marks the reader cannot see.
pub struct ProjGap {
    n: usize,
    /// Front points per geometry, in [`CURVED`] order.
    points: [Vec<GapPoint>; CURVED.len()],
    /// Lowest κ kept. `0.0` for the full figure.
    ///
    /// Why a zoom is worth its own figure rather than a tighter axis: the low-κ
    /// end of the hyperbolic arm is a dense spike at κ ≈ 2e-7 of embeddings
    /// **collapsed to a point** — `r_max == r_rms` and a coordinate extent of
    /// 4.5e-4 whatever `|K|` is, so their metrics are chance-level noise. They
    /// are legitimate front points (a collapsed embedding can still win one
    /// objective) and the full figure keeps them, but they sit three decades
    /// left of every other point and dominate its shape.
    kappa_min: f64,
}

impl ProjGap {
    pub fn new(cells: &CellMap, n: usize, space: ObjectiveSpace) -> Self {
        let mut points: [Vec<GapPoint>; CURVED.len()] = Default::default();
        for (key, recs) in cells {
            if key.n != n {
                continue;
            }
            let Some(slot) = CURVED.iter().position(|g| *g == key.geometry) else {
                continue;
            };
            for r in pareto_front_records(recs, space) {
                // A non-positive or absent κ cannot be placed on the log axis.
                let Some(kappa) = r.kappa().filter(|k| k.is_finite() && *k > 0.0) else {
                    continue;
                };
                let mut gaps = [None; N_METRIC_PAIRS];
                for (slot, (proj, man)) in gaps.iter_mut().zip(METRIC_PAIRS.iter()) {
                    let (Some(pv), Some(mv)) = (r.metrics.get(*proj), r.metrics.get(*man)) else {
                        continue;
                    };
                    if pv.is_finite() && mv.is_finite() {
                        *slot = Some(oriented(*man, Some(mv)) - oriented(*proj, Some(pv)));
                    }
                }
                points[slot].push(GapPoint { kappa, gaps });
            }
        }
        Self {
            n,
            points,
            kappa_min: 0.0,
        }
    }

    /// The same figure restricted to `κ >= kappa_min`, written to its own file.
    #[must_use]
    pub fn zoomed(self, kappa_min: f64) -> Self {
        Self { kappa_min, ..self }
    }

    /// (κ, gap) for one metric and one geometry, dropping the points that metric
    /// has no pair for and the ones below [`ProjGap::kappa_min`].
    fn xy(&self, m_idx: usize, g_idx: usize) -> (Vec<f64>, Vec<f64>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for p in &self.points[g_idx] {
            if p.kappa < self.kappa_min {
                continue;
            }
            if let Some(g) = p.gaps[m_idx] {
                xs.push(p.kappa);
                ys.push(g);
            }
        }
        (xs, ys)
    }

    /// Every kept κ, across geometries — the x range, and the emptiness test.
    fn kappas(&self) -> Vec<f64> {
        self.points
            .iter()
            .flat_map(|ps| ps.iter().map(|p| p.kappa))
            .filter(|k| *k >= self.kappa_min)
            .collect()
    }

    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.kappas().is_empty()
    }
}

impl Figure for ProjGap {
    fn name(&self) -> String {
        let base = format!("exp4_gap_vs_kappa_N{}", self.n);
        if self.kappa_min <= 0.0 {
            return base;
        }
        // The floor goes in the filename, so a zoom at another threshold lands
        // beside this one instead of overwriting it. `log_tick` renders the
        // decades as plain decimals, whose '.' would read as an extension.
        format!(
            "{base}_from_k{}",
            log_tick(&self.kappa_min).replace('.', "p")
        )
    }

    fn size(&self) -> (u32, u32) {
        let rows = u32::try_from(GRID_ROWS).expect("a small grid");
        (2 * PANEL.0, rows * PANEL.1)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        // No title: the caption carries N, the κ floor and the sign convention,
        // as for every other thesis figure. The one cell the metrics leave free
        // holds the legend and the sign note instead.
        let cells = root.split_evenly((GRID_ROWS, GRID_COLS));
        draw_key(&cells[METRIC_PAIRS.len()])?;

        // A shared x range across panels: every panel plots the same points, so
        // a per-panel range would only differ through the metric's own missing
        // values and would make the panels silently incomparable.
        let (x_lo, x_hi) = decade_frame(&self.kappas()).unwrap_or((1e-3, 1e1));
        // The zoom's floor is exact — padding it back below the threshold would
        // leave a strip of axis the figure promises to have excluded.
        let x_lo = x_lo.max(self.kappa_min);

        for (m_idx, pair) in METRIC_PAIRS.iter().enumerate() {
            let left_column = m_idx % GRID_COLS == 0;
            // The y range is per panel: the five metrics' gaps differ by an
            // order of magnitude in spread (neighborhood_hit's 99th percentile
            // is +0.02, normalized_stress's +0.59), so one shared range would
            // flatten four panels to a line.
            let pooled: Vec<f64> = (0..CURVED.len())
                .flat_map(|g| self.xy(m_idx, g).1)
                .collect();
            let (y_lo, y_data_hi) = tail_range(&pooled, Y_TAIL).unwrap_or((-0.5, 0.5));
            // Headroom for the two ρ annotations. Reserving a band is the only
            // placement that holds for every panel: the clouds sit at different
            // heights, and both corners are occupied in at least one of them.
            let y_hi = y_data_hi + (y_data_hi - y_lo) * HEAD_ROOM;

            // Own ticks rather than plotters': its float walk labels the zero
            // line "-0.0" on half the panels — see `LinearTicks`.
            let yt = LinearTicks::new((y_lo, y_hi), 6);
            let mut chart = ChartBuilder::on(&cells[m_idx])
                .margin(8)
                .caption(pair.0.name(), ("sans-serif", 14).into_font())
                .x_label_area_size(40)
                .y_label_area_size(if left_column { 62 } else { 48 })
                .build_cartesian_2d((x_lo..x_hi).log_scale(), yt.clone())?;

            style_mesh!(chart.configure_mesh())
                .x_desc("κ (front point)")
                .y_desc(if left_column {
                    "gap (manifold - 2D)"
                } else {
                    ""
                })
                // One label per decade collides at this panel width; plotters
                // thins the ticks to fit the count it is given.
                .x_labels(5)
                .x_label_formatter(&log_tick)
                .y_label_formatter(&|v| yt.label(v))
                .draw()?;

            // Reference line at "the two readings agree".
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x_lo, 0.0), (x_hi, 0.0)],
                RGBColor(187, 187, 187).stroke_width(1),
            )))?;

            for (g_idx, geometry) in CURVED.iter().enumerate() {
                let color = geometry_color(geometry);
                let (xs, ys) = self.xy(m_idx, g_idx);
                if xs.is_empty() {
                    continue;
                }
                let stride = xs.len().div_ceil(DRAW_CAP).max(1);
                chart.draw_series(
                    xs.iter()
                        .zip(&ys)
                        .step_by(stride)
                        .map(|(x, y)| Circle::new((*x, *y), 1, color.mix(0.35).filled())),
                )?;

                let (bx, by) = binned_median(&xs, &ys, GAP_N_BINS, GAP_MIN_PER_BIN);
                if !bx.is_empty() {
                    let line: Vec<(f64, f64)> =
                        bx.iter().copied().zip(by.iter().copied()).collect();
                    chart.draw_series(LineSeries::new(line, color.mix(0.95).stroke_width(3)))?;
                }

                // ρ is over *every* point of the series, including the ones the
                // y range clips — it is the statistic the question asks for, not
                // a description of what the panel happens to show.
                let label = match spearman(&xs, &ys) {
                    Some((rho, _)) if rho.is_finite() => {
                        format!("{geometry} ρ={rho:+.2} (n={})", xs.len())
                    }
                    // Reported rather than omitted: a blank corner would read as
                    // "no correlation" instead of "not computable".
                    _ => format!("{geometry} ρ=n/a (n={})", xs.len()),
                };
                let y_text = y_hi - (y_hi - y_lo) * (0.02 + 0.09 * count_to_f64(g_idx));
                chart.plotting_area().draw(&Text::new(
                    label,
                    (x_lo * 1.6, y_text),
                    ("sans-serif", 12)
                        .into_font()
                        .color(&color)
                        .pos(Pos::new(HPos::Left, VPos::Top)),
                ))?;
            }
        }
        Ok(())
    }
}

/// Whole decades enclosing *values*, counting a value within [`DECADE_SLACK`]
/// of a decade as on it.
///
/// Not `snap_to_decades(padded_log_range(..))`: the collapsed-embedding spike
/// sits at κ = 9.99e-8, 0.01 % below 1e-7, and padding then flooring put the
/// frame's left edge a whole empty decade further out, at 1e-8. The slack moves
/// those points by a fraction of a pixel onto the frame instead.
fn decade_frame(values: &[f64]) -> Option<(f64, f64)> {
    let logs = values
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .map(|v| v.log10());
    let (lo, hi) = logs.fold(None, |acc: Option<(f64, f64)>, l| {
        Some(acc.map_or((l, l), |(a, b)| (a.min(l), b.max(l))))
    })?;
    let lo = (lo + DECADE_SLACK).floor();
    let hi = (hi - DECADE_SLACK).ceil().max(lo + 1.0);
    Some((10f64.powf(lo), 10f64.powf(hi)))
}

/// The legend cell: the geometry key, and the sign convention the dropped
/// title used to state.
fn draw_key<DB: DrawingBackend>(area: &DrawingArea<DB, Shift>) -> Res
where
    DB::ErrorType: 'static,
{
    let (w, h) = area.dim_in_pixel();
    let (w, h) = (i32::try_from(w).unwrap_or(0), i32::try_from(h).unwrap_or(0));
    let entries: Vec<LegendEntry> = CURVED
        .iter()
        .map(|g| LegendEntry::new((*g).to_string(), geometry_color(g)))
        .collect();
    let strip = area.margin(h / 2 - 30, 0, 20, 0);
    let (key, _) = strip.split_vertically(28);
    draw_legend(&key, &entries)?;
    let font = ("sans-serif", 13)
        .into_font()
        .color(&OK_BLACK)
        .pos(Pos::new(HPos::Center, VPos::Top));
    for (i, line) in [
        "one point per Pareto-front trial",
        "gap > 0: the 2D reading is worse",
    ]
    .iter()
    .enumerate()
    {
        area.draw(&Text::new(
            *line,
            (w / 2, h / 2 + 12 + 20 * i32::try_from(i).unwrap_or(0)),
            font.clone(),
        ))?;
    }
    Ok(())
}

/// A padded range covering all but the outer `tail` of *values* on each side.
///
/// Not [`super::robust_range`]: Tukey's fences are derived from the IQR, and these gap
/// distributions are spikes at zero — most panels have an IQR near 1e-3, so the
/// fences would clip the entire informative tail. A flat quantile cut hides a
/// known, stated fraction instead.
fn tail_range(values: &[f64], tail: f64) -> Option<(f64, f64)> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 20 {
        return padded_range(&finite, 0.08);
    }
    let (lo, hi) = (quantile(&finite, tail)?, quantile(&finite, 1.0 - tail)?);
    // Zero is the reference line the whole figure is read against; a range that
    // excluded it would put the "they agree" line off the panel.
    padded_range(&[lo.min(0.0), hi.max(0.0)], 0.08)
}
