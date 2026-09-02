//! Experiment 4 — how far the 2D projection's verdict drifts from the
//! manifold's, and whether the drift grows with curvature.
//!
//! Two figures, asking the two halves of that question at two different units:
//!
//! - [`RhoManProj`] — does the projection **reorder** hyperparameter
//!   configurations? One Spearman per *cell*, against the cell's median κ.
//! - [`ProjGap`] — how far apart are the two readings of the *same*
//!   configuration? One gap per *Pareto-front trial*, against that trial's own
//!   κ, all datasets pooled.
//!
//! The unit is the substantive difference between them. κ is a swept
//! hyperparameter (`curvature_magnitude`), so it varies more *within* a cell
//! than between cells — on the hyperbolic arm the mean within-cell spread of
//! log₁₀κ is ~2.5 decades against ~1.7–2.1 between cell medians. A cell-median κ
//! therefore summarises a distribution that already covers the whole range of
//! interest, which is why [`RhoManProj`]'s trend is weak and why [`ProjGap`]
//! keeps κ per trial.
//!
//! ## ρ_man-proj(κ)
//!
//! One point per (dataset, geometry, setting) cell at a single N: x = median κ
//! over the cell's trials, y = Spearman(manifold variant, 2D variant) over the
//! trials. Cells are independent, so they are scattered and overlaid with a
//! binned-median trend — never connected, which would imply an ordering.
//!
//! Two things this figure deliberately does *not* do:
//!
//! - **It plots one N, not both.** Overlaying N=1000 and N=5000 doubled the
//!   marks without adding a comparison anyone reads off this figure: ρ is a
//!   within-cell rank correlation, so the two sizes are separate populations,
//!   not a trend in N. The driver picks the largest N asked for.
//! - **Euclidean *embeddings* are excluded** ([`CURVED`]), because there the
//!   projection is the identity and ρ ≡ 1 by construction. Euclidean-*by-
//!   construction datasets* (`grid`) fitted under a curved model stay in: their
//!   fitted κ is small but nonzero, the projection is still nonlinear, and
//!   their ρ spans 0.66–0.998. They are the low-κ end of the trend, not a
//!   tautology — and the cut that would remove a near-isometric projection is
//!   κ, which is already the x axis.

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use super::*;
use crate::objectives::oriented_value;
use crate::pareto::pareto_front_records;
use crate::stats::{median, quantile, spearman};
use crate::style_mesh;

/// A cell needs at least this many usable trials for its ρ to mean anything.
const MIN_TRIALS: usize = 10;
const N_BINS: usize = 6;

/// Cells a bin needs before its median is drawn.
///
/// At the old floor of 2 the trend was not robust: on the spherical
/// `normalized_stress` panel the upper bin read 0.055 over 3 cells and 0.692
/// over 5 — the same bin, flipped by two cells. Five is what the populated bins
/// of the N=5000 panels carry (hyperbolic 5–17 cells, spherical 5–29), so it
/// drops the coin-flip bins without emptying the curve.
const MIN_PER_BIN: usize = 5;

pub struct RhoManProj<'a> {
    cells: &'a CellMap,
    n: usize,
}

impl<'a> RhoManProj<'a> {
    pub fn new(cells: &'a CellMap, n: usize) -> Self {
        Self { cells, n }
    }

    /// (median κ, ρ_man-proj) for every cell of one geometry at this N.
    ///
    /// *pair* is one row of [`METRIC_PAIRS`]: the projected and manifold
    /// objective names whose ranks are correlated against each other.
    fn points(&self, pair: (&str, &str), geometry: &str) -> Vec<(f64, f64)> {
        let (projected, manifold) = pair;
        let mut pts = Vec::new();
        for (key, recs) in self.cells {
            if key.geometry != geometry || key.n != self.n {
                continue;
            }
            let (mut proj, mut man, mut ks) = (Vec::new(), Vec::new(), Vec::new());
            for r in recs {
                let (Some(pv), Some(mv), Some(kv)) =
                    (r.objective(projected), r.objective(manifold), r.kappa())
                else {
                    continue;
                };
                if pv.is_finite() && mv.is_finite() {
                    proj.push(pv);
                    man.push(mv);
                    ks.push(kv);
                }
            }
            if proj.len() < MIN_TRIALS {
                continue;
            }
            if let (Some((rho, _)), Some(mk)) = (spearman(&man, &proj), median(&ks)) {
                if rho.is_finite() {
                    pts.push((mk, rho));
                }
            }
        }
        pts
    }

    pub fn has_data(&self) -> bool {
        METRIC_PAIRS
            .iter()
            .any(|p| CURVED.iter().any(|g| !self.points(*p, g).is_empty()))
    }
}

impl Figure for RhoManProj<'_> {
    fn name(&self) -> String {
        format!("exp4_rho_man_proj_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (400 * METRIC_PAIRS.len() as u32, 480)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let root = root.titled(
            &format!(
                // No "≥" here: the bitmap backend's "sans-serif" has no U+2265
                // and renders it as tofu. Same caveat as the arrows.
                "Experiment 4 — manifold vs projected rank agreement vs curvature, N={} \
                 (points = cells, lines = binned median over {MIN_PER_BIN}+ cells)",
                self.n
            ),
            ("sans-serif", 20).into_font().color(&OK_BLACK),
        )?;
        let (legend, grid) = root.split_vertically(32);

        let entries: Vec<LegendEntry> = CURVED
            .iter()
            .map(|g| LegendEntry::new((*g).to_string(), geometry_color(g)))
            .collect();
        draw_legend(&legend, &entries)?;

        let panels = grid.split_evenly((1, METRIC_PAIRS.len()));
        for (m_idx, pair) in METRIC_PAIRS.iter().enumerate() {
            // A shared x range across geometries keeps the panels comparable.
            let all_x: Vec<f64> = CURVED
                .iter()
                .flat_map(|g| self.points(*pair, g).into_iter().map(|p| p.0))
                .collect();
            let (x_lo, x_hi) =
                snap_to_decades(padded_log_range(&all_x, 0.08).unwrap_or((1e-3, 1e1)));

            let mut chart = ChartBuilder::on(&panels[m_idx])
                .margin(10)
                .caption(pair.0, ("sans-serif", 15).into_font())
                .x_label_area_size(52)
                .y_label_area_size(if m_idx == 0 { 66 } else { 34 })
                .build_cartesian_2d((x_lo..x_hi).log_scale(), -0.8f64..1.1f64)?;

            style_mesh!(chart.configure_mesh())
                .x_desc("median κ (cell)")
                .y_desc(if m_idx == 0 { "ρ_man-proj" } else { "" })
                .x_label_formatter(&log_tick)
                .draw()?;

            // Reference line at perfect agreement.
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x_lo, 1.0), (x_hi, 1.0)],
                RGBColor(187, 187, 187).stroke_width(1),
            )))?;

            for geometry in CURVED {
                let color = geometry_color(geometry);
                let pts = self.points(*pair, geometry);
                if pts.is_empty() {
                    continue;
                }
                // Faint cell cloud ...
                chart.draw_series(
                    pts.iter()
                        .map(|(x, y)| Circle::new((*x, *y), 3, color.mix(0.25).filled())),
                )?;

                // ... plus the binned-median trend. A geometry whose κ barely
                // varies (spherical sits inside [2.47, 3.00] — see the crate's
                // "One κ, one gauge") collapses to one or two bins; that is a
                // true reading of the data, so it is drawn, not suppressed.
                let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
                let ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
                let (bx, by) = binned_median(&xs, &ys, N_BINS, MIN_PER_BIN);
                if bx.is_empty() {
                    continue;
                }
                let line: Vec<(f64, f64)> = bx.iter().copied().zip(by.iter().copied()).collect();
                chart.draw_series(LineSeries::new(
                    line.clone(),
                    color.mix(0.95).stroke_width(3),
                ))?;
                chart.draw_series(
                    line.iter()
                        .map(|(x, y)| Circle::new((*x, *y), 5, color.filled())),
                )?;
            }
        }
        Ok(())
    }
}

// ─── The projection gap against κ, per front point ────────────────────────────

/// Fraction of each panel's points kept outside the y range, split evenly
/// between the tails. plotters clips silently, so the range is a *choice about
/// what to hide*: the gap distributions are spikes at 0 with long one-sided
/// tails (`shepard_goodness` runs to −0.997), and a full-range axis squeezes
/// the body of every panel into a few pixels.
const Y_TAIL: f64 = 0.005;

/// Points a κ bin needs before its median joins the trend line. Far above
/// [`MIN_PER_BIN`] because these are *front points*, not cells — tens of
/// thousands of them per geometry.
const GAP_MIN_PER_BIN: usize = 30;
const GAP_N_BINS: usize = 12;

/// Fraction of the data's y span added above it, as a clear band for the two ρ
/// annotations.
const HEAD_ROOM: f64 = 0.16;

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
    /// **Not** [`oriented_value`]'s 0.0 substitution: that maps an absent value
    /// to "worst possible", which here would manufacture a gap of ±1 out of a
    /// missing column rather than dropping the point.
    gaps: [Option<f64>; METRIC_PAIRS.len()],
}

/// Experiment 4, second figure — the *size* of the manifold-vs-projection
/// disagreement against curvature, one point per Pareto-front trial.
///
/// [`RhoManProj`] asks whether the projection **reorders** configurations;
/// this asks how far apart the two readings of the same configuration are, and
/// whether that distance grows with κ. Three deliberate choices:
///
/// - **Front points, not all trials.** The front is the set a practitioner
///   would actually choose from, and it is where a wrong reading costs
///   something. It also drops the diverged tail, whose gaps are noise.
/// - **All datasets and settings pooled**, so a panel is a population of front
///   points rather than a population of cells. The scatter is therefore dense
///   (~17k hyperbolic, ~8-11k spherical per N) and drawn with heavy alpha.
/// - **The gap is oriented**, `oriented(manifold) − oriented(2D)`, so positive
///   always means the 2D reading is the *pessimistic* one, on all five metrics
///   including `normalized_stress` where the raw value runs the other way.
///
/// Euclidean cells are excluded for the same reason as in [`RhoManProj`], with
/// one addition: their κ is exactly 0, which has no place on a log axis.
///
/// **The spherical ρ is reported but should not be read as a curvature trend.**
/// Spherical κ spans [2.42, 4.93] — a factor of two, against eight decades on
/// the hyperbolic arm — because `|K|·R_rms²` has a floor at π²/4 on a compact
/// manifold (see the crate's "One κ, one gauge"). Over that range ρ describes
/// configuration *shape*, not how curved the space is.
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
    pub fn new(cells: &CellMap, n: usize) -> Self {
        let mut points: [Vec<GapPoint>; CURVED.len()] = Default::default();
        for (key, recs) in cells {
            if key.n != n {
                continue;
            }
            let Some(slot) = CURVED.iter().position(|g| *g == key.geometry) else {
                continue;
            };
            for r in pareto_front_records(recs) {
                // A non-positive or absent κ cannot be placed on the log axis.
                let Some(kappa) = r.kappa().filter(|k| k.is_finite() && *k > 0.0) else {
                    continue;
                };
                let mut gaps = [None; METRIC_PAIRS.len()];
                for (slot, (proj, man)) in gaps.iter_mut().zip(METRIC_PAIRS) {
                    let (Some(pv), Some(mv)) = (r.objective(proj), r.objective(man)) else {
                        continue;
                    };
                    if pv.is_finite() && mv.is_finite() {
                        *slot =
                            Some(oriented_value(man, Some(mv)) - oriented_value(proj, Some(pv)));
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
        (400 * METRIC_PAIRS.len() as u32, 480)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        // No "≥" in the title: the bitmap backend's "sans-serif" renders U+2265
        // as tofu, the same caveat the other figure carries.
        let scope = if self.kappa_min > 0.0 {
            format!(", kappa from {}", log_tick(&self.kappa_min))
        } else {
            String::new()
        };
        let root = root.titled(
            &format!(
                "Experiment 4 — projection gap vs curvature, N={}{scope}, Pareto-front trials, all \
                 datasets pooled (positive = the 2D metric reads worse than the manifold one)",
                self.n
            ),
            ("sans-serif", 20).into_font().color(&OK_BLACK),
        )?;
        let (legend, grid) = root.split_vertically(32);

        let entries: Vec<LegendEntry> = CURVED
            .iter()
            .map(|g| LegendEntry::new((*g).to_string(), geometry_color(g)))
            .collect();
        draw_legend(&legend, &entries)?;

        // A shared x range across panels: every panel plots the same points, so
        // a per-panel range would only differ through the metric's own missing
        // values and would make the panels silently incomparable.
        //
        // A small padding fraction on purpose: κ spans nine decades on the full
        // figure, and `snap_to_decades` rounds outward on top of it, so the 0.08
        // the other figures use buys two entirely empty decades.
        let (x_lo, x_hi) =
            snap_to_decades(padded_log_range(&self.kappas(), 0.02).unwrap_or((1e-3, 1e1)));
        // The zoom's floor is exact — padding it back below the threshold would
        // leave a strip of axis the figure promises to have excluded.
        let x_lo = x_lo.max(self.kappa_min);

        let panels = grid.split_evenly((1, METRIC_PAIRS.len()));
        for (m_idx, pair) in METRIC_PAIRS.iter().enumerate() {
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

            let mut chart = ChartBuilder::on(&panels[m_idx])
                .margin(10)
                .caption(pair.0, ("sans-serif", 15).into_font())
                .x_label_area_size(52)
                .y_label_area_size(if m_idx == 0 { 66 } else { 44 })
                .build_cartesian_2d((x_lo..x_hi).log_scale(), y_lo..y_hi)?;

            style_mesh!(chart.configure_mesh())
                .x_desc("κ (front point)")
                .y_desc(if m_idx == 0 {
                    "oriented gap (manifold - 2D)"
                } else {
                    ""
                })
                // One label per decade collides at this panel width; plotters
                // thins the ticks to fit the count it is given.
                .x_labels(5)
                .x_label_formatter(&log_tick)
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
                let y_text = y_hi - (y_hi - y_lo) * (0.02 + 0.07 * g_idx as f64);
                chart.plotting_area().draw(&Text::new(
                    label,
                    (x_lo * 1.6, y_text),
                    ("sans-serif", 13)
                        .into_font()
                        .color(&color)
                        .pos(Pos::new(HPos::Left, VPos::Top)),
                ))?;
            }
        }
        Ok(())
    }
}

/// A padded range covering all but the outer `tail` of *values* on each side.
///
/// Not [`robust_range`]: Tukey's fences are derived from the IQR, and these gap
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
