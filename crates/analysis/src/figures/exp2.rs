//! Experiment 2 (`metric-results`) — what effect does curvature have on the
//! established visualisation metrics?
//!
//! [`MetricVsKappa`] is the figure this module draws: **κ on x, the metric
//! reading on y, every metric bounded in `[0, 1]` overlaid on one pair of
//! axes, one figure per embedding geometry.** Two figures, not three — see
//! *No Euclidean panel* below.
//!
//! ### What is drawn, and from which trials
//!
//! The crate's own rule is that a panel must say what population it draws from,
//! because a front-point distribution describes the searched corpus rather than
//! an unbiased sample. This one draws **every trial of the `all_off` cells** at
//! the given N and geometry, pooled over all datasets:
//!
//! * *all trials*, not front points — the question is how a metric responds to
//!   κ, not what the search chose to keep;
//! * *`all_off` only*, so the auxiliary loss weights are pinned at zero and do
//!   not vary along with κ. Pooling every setting would mix two effects.
//!
//! Each curve is the **median of its metric in log-spaced κ bins**, drawn only
//! where a bin holds at least [`MIN_PER_BIN`] trials. Every curve is binned on
//! one shared set of edges ([`super::log_bin_edges`] once, then
//! [`super::binned_median_on`] per metric), so the overlaid lines are sampled at
//! the same κ positions and can be read against each other — which is the whole
//! reason they share a panel.
//!
//! ### Which metrics count as "0 to 1"
//!
//! Exactly [`super::OBJECTIVES`]: `QualityMetric::is_objective` *means* bounded
//! in `[0, 1]`, and the registry is the only place that fact lives. The five
//! `_manifold` twins are equally bounded and are left out **on purpose** — the
//! thesis judges the 2-D visualisation, and `crates/analysis/CLAUDE.md` records
//! the figure that was wrong for exactly this reason, putting a manifold and a
//! projected reading of one metric on a single axis. Adding them is a change to
//! the list this module iterates, and nothing else.
//!
//! Orientation comes off the registry too: `normalized_stress` is the one
//! minimised metric, so it is drawn as `1 - stress` and labelled that way.
//!
//! A metric with no column in the loaded sweeps is dropped from the panel and
//! from the legend rather than drawn at zero. That matters today: every file
//! under `results/` is a legacy `obj10` sweep with no `distance_consistency`,
//! and `objectives::oriented` — which maps *absent* to the worst case, correctly,
//! for the R2 indicator — would draw it as a flat line along the bottom. Hence
//! [`reading`] rather than `oriented`.
//!
//! ### No Euclidean panel
//!
//! `κ = |K|·R_rms²` and Euclidean space has `K = 0` exactly, so every Euclidean
//! trial sits at `κ = 0`: one column of points, no axis, nothing to read. The
//! panel is not drawn. A flat reference, if one is wanted, is a different figure
//! — the same metrics' *distributions* at κ = 0 — not this one with another
//! argument.
//!
//! The spherical panel **is** drawn, and its κ window is a factor of ~1.4 rather
//! than the hyperbolic seven decades. That is not a plotting failure: it is the
//! pole mismatch between `Sphere::center` and `lift_pca_to_manifold` documented
//! in `crates/analysis/CLAUDE.md` § *One κ, one gauge*, which pins spherical κ
//! at `mean(θ²)` and makes it blind to `|K|`. Reading the two panels side by
//! side is how that shows.
//!
//! ### Layout
//!
//! Each panel is half of Exp 1's canvas ([`PANEL`]), so two of them occupy the
//! width one Exp 1 figure does and sit side by side at the A4 text width. Only
//! the first carries the legend; it gets an extra [`LEGEND_STRIP`] of canvas
//! **at the top** for it, which leaves the two plot areas the same height — so
//! set the pair **bottom-aligned** and the axes line up.
//!
//! ### Not this module's other two figures
//!
//! [`MetricPanels`] below is still a skeleton. The results chapter asks for two
//! further figures under `<metric-results>` — a dataset-by-metric panel grouped
//! by metric family over synthetic *and* real datasets, and the Spearman
//! metric-dependence heatmap of `<metric-dependence>`, asking whether the
//! metrics order the same visualisations the same way. Neither is drawn yet.

use plotters::coord::Shift;
use plotters::prelude::*;

use fitting_core::metrics::Metric;

use super::{
    binned_median_on, draw_legend_grid, log_bin_edges, log_tick, metric_color, metric_dash,
    padded_log_range, snap_to_decades, CellMap, Figure, LegendEntry, ObjectiveSpace, Res, CURVED,
    OBJECTIVES, OK_BLACK,
};
use crate::objectives::is_minimized_metric;
use crate::records::TrialRecord;
use crate::style_mesh;

/// The loss-weight setting the panels draw from: the baseline, where every
/// auxiliary weight is zero, so κ is the only thing varying with the metric.
const SETTING: &str = "all_off";

/// Log-spaced κ bins across the panel's whole range.
const N_BINS: usize = 24;

/// Trials a bin needs before its median is drawn. The pooled corpus is ~9,000
/// trials per (geometry, N) over 24 bins, so a bin at this floor is already a
/// sparse one — which on a log axis is exactly the tail, where a median over a
/// handful of points flips on one of them.
const MIN_PER_BIN: usize = 20;

/// Half of `exp1::MatchedGain::size()` (740 x 450), so two of these occupy the
/// width one Exp 1 figure does and the pair sits side by side at the A4 text
/// width.
///
/// plotters font sizes are absolute canvas units, so halving the canvas does
/// *not* halve the text: at ~80 mm on the page `style_mesh!`'s 13 px tick
/// labels land at roughly 8 pt, against Exp 1's ~6 pt. That is what a
/// half-width figure wants — it is read at half the size.
const PANEL: (u32, u32) = (370, 225);

/// Extra canvas, on top, for the one panel that carries the legend. Two rows of
/// entries at the 12 px [`draw_legend_grid`] font.
const LEGEND_STRIP: u32 = 40;

/// One metric's trend curve.
struct Series {
    label: String,
    color: RGBColor,
    dash: Option<(i32, i32)>,
    /// `(κ, median reading)` at each bin centre that cleared [`MIN_PER_BIN`].
    points: Vec<(f64, f64)>,
}

/// Every `[0, 1]`-bounded metric against κ, for one embedding geometry.
pub struct MetricVsKappa {
    geometry: &'static str,
    n: usize,
    series: Vec<Series>,
    x_range: (f64, f64),
    /// Whether this panel draws the shared legend. Only the first does.
    legend: bool,
}

impl MetricVsKappa {
    /// One panel per curved geometry, in [`CURVED`] order; a geometry with no
    /// binnable trials is absent from the result rather than returned empty,
    /// the same contract `exp4::R2Bars::panels` has.
    #[must_use]
    pub fn panels(cells: &CellMap, n: usize) -> Vec<MetricVsKappa> {
        let mut out: Vec<MetricVsKappa> = Vec::new();
        for geometry in CURVED {
            // κ is the x axis, so a trial without one is not a point on this
            // figure at all. The `> 0` is the log axis' requirement, not a
            // quality filter: κ = 0 is Euclidean, which has no panel here.
            let kept: Vec<(f64, &TrialRecord)> = cells
                .iter()
                .filter(|(cell, _)| {
                    cell.setting == SETTING && cell.n == n && cell.geometry == geometry
                })
                .flat_map(|(_, records)| records.iter())
                .filter_map(|r| {
                    r.kappa()
                        .filter(|k| k.is_finite() && *k > 0.0)
                        .map(|k| (k, r))
                })
                .collect();

            let kappas: Vec<f64> = kept.iter().map(|(k, _)| *k).collect();
            let Some(edges) = log_bin_edges(&kappas, N_BINS) else {
                continue;
            };

            let series: Vec<Series> = OBJECTIVES
                .iter()
                .filter_map(|&metric| Series::build(metric, &edges, &kept))
                .collect();
            if series.is_empty() {
                continue;
            }

            // Pad in decades, then widen to whole ones so the ticks land on
            // powers of ten. `snap_to_decades` leaves a sub-decade span alone,
            // which is what keeps the spherical panel's narrow window narrow
            // instead of stretching it across a full decade it does not fill.
            let x_range = snap_to_decades(
                padded_log_range(&kappas, 0.03).unwrap_or((edges[0], edges[edges.len() - 1])),
            );

            out.push(MetricVsKappa {
                geometry,
                n,
                series,
                x_range,
                legend: out.is_empty(),
            });
        }
        out
    }

    /// Always true for a panel [`MetricVsKappa::panels`] returned; kept so the
    /// driver reads the same as every other figure's.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.series.is_empty()
    }
}

impl Series {
    /// One metric's binned trend, or `None` when the sweeps do not carry it or
    /// no bin cleared [`MIN_PER_BIN`].
    fn build(metric: Metric, edges: &[f64], kept: &[(f64, &TrialRecord)]) -> Option<Series> {
        let (xs, ys): (Vec<f64>, Vec<f64>) = kept
            .iter()
            .filter_map(|(k, r)| reading(metric, r).map(|v| (*k, v)))
            .unzip();
        let (centres, medians) = binned_median_on(edges, &xs, &ys, MIN_PER_BIN);
        if centres.is_empty() {
            return None;
        }
        Some(Series {
            label: label(metric),
            color: metric_color(metric.name()),
            dash: metric_dash(metric.name()),
            points: centres.into_iter().zip(medians).collect(),
        })
    }

    fn legend_entry(&self) -> LegendEntry {
        let entry = LegendEntry::new(self.label.clone(), self.color);
        match self.dash {
            Some((dash, gap)) => entry.with_dash(dash, gap),
            None => entry,
        }
    }
}

/// One metric's reading off a trial, oriented so higher is better.
///
/// **Not `objectives::oriented`**, which substitutes the worst case for a
/// reading that is absent. That is right for the R2 indicator, where a trial
/// that did not measure an objective must score badly rather than vanish, and
/// wrong for a trend curve, where it would draw a metric the sweeps never wrote
/// as a flat line at zero. Here absent stays absent and the metric is dropped.
fn reading(metric: Metric, record: &TrialRecord) -> Option<f64> {
    let v = record.metrics.get(metric)?;
    if !v.is_finite() {
        return None;
    }
    Some(if is_minimized_metric(metric) {
        1.0 - v
    } else {
        v
    })
}

/// The series label: the registry's abbreviation, marked as flipped where the
/// metric is minimised. ASCII hyphen, not U+2212 — the bitmap backend renders
/// anything outside Latin-1 + Greek as tofu.
fn label(metric: Metric) -> String {
    if is_minimized_metric(metric) {
        format!("1-{}", metric.name())
    } else {
        metric.name().to_string()
    }
}

impl Figure for MetricVsKappa {
    fn name(&self) -> String {
        format!("exp2_metric_vs_kappa_{}_N{}", self.geometry, self.n)
    }

    fn size(&self) -> (u32, u32) {
        let (w, h) = PANEL;
        (w, h + if self.legend { LEGEND_STRIP } else { 0 })
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        // The strip comes off the top, so both panels' plot areas are PANEL.1
        // tall and line up when the pair is set bottom-aligned.
        let plot = if self.legend {
            let (strip, plot) = root.split_vertically(LEGEND_STRIP);
            let entries: Vec<LegendEntry> = self.series.iter().map(Series::legend_entry).collect();
            draw_legend_grid(&strip, &entries, 3)?;
            plot
        } else {
            root.clone()
        };

        // The geometry is the one identifying thing drawn: two of these are
        // read side by side and have to be tellable apart. N, setting and
        // objective space stay in the filename, as Exp 1's do.
        let mut builder = ChartBuilder::on(&plot);
        builder
            .margin(6)
            .margin_right(12)
            .caption(
                self.geometry,
                ("sans-serif", 14)
                    .into_font()
                    .style(FontStyle::Bold)
                    .color(&OK_BLACK),
            )
            .x_label_area_size(34)
            .y_label_area_size(46);

        let (lo, hi) = self.x_range;
        // **A log axis needs a decade to label.** plotters derives a log
        // scale's key points from its endpoints and finds none inside a window
        // narrower than one decade, so the spherical panel — whose κ is pinned
        // to a factor of ~1.4 by the wrong-pole gauge — came out with no x
        // ticks at all. Below a decade the axis is linear, which is also the
        // honest rendering: nothing about that window is multiplicative.
        if hi / lo >= 10.0 {
            let mut chart = builder.build_cartesian_2d((lo..hi).log_scale(), 0.0f64..1.0f64)?;
            style_mesh!(chart.configure_mesh())
                .x_desc(X_DESC)
                .y_desc(Y_DESC)
                .x_label_formatter(&log_tick)
                // A hint, not a count: plotters walks decades on a log axis and
                // rounds to whole ones. Ten decades at this width put "0.0001"
                // next to "0.001", which collides once the figure is set at half
                // the A4 text width; asking for fewer thins them to every other
                // decade instead.
                .x_labels(5)
                .y_labels(5)
                .draw()?;
            self.draw_curves(&mut chart)?;
        } else {
            let mut chart = builder.build_cartesian_2d(lo..hi, 0.0f64..1.0f64)?;
            style_mesh!(chart.configure_mesh())
                .x_desc(X_DESC)
                .y_desc(Y_DESC)
                .x_labels(4)
                .y_labels(5)
                .draw()?;
            self.draw_curves(&mut chart)?;
        }
        Ok(())
    }
}

/// The κ axis title. Greek resolves in both backends; arrows and geometric
/// shapes do not, so the orientation is stated in words on the y axis.
const X_DESC: &str = "κ";
/// Every series is oriented so higher is better — `normalized_stress` is drawn
/// as `1 - stress` — and the axis says so, because that is the one thing about
/// this figure a reader cannot recover from the curves.
const Y_DESC: &str = "higher is better";

impl MetricVsKappa {
    /// Draw every series onto *chart*. Generic over the x coordinate so the log
    /// and linear branches of [`MetricVsKappa::draw`] share one body.
    fn draw_curves<DB, X, Y>(&self, chart: &mut ChartContext<DB, Cartesian2d<X, Y>>) -> Res
    where
        DB: DrawingBackend,
        DB::ErrorType: 'static,
        X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
        Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    {
        for series in &self.series {
            let color = series.color;
            match series.dash {
                Some((dash, gap)) => {
                    chart.draw_series(DashedLineSeries::new(
                        series.points.iter().copied(),
                        dash,
                        gap,
                        color.stroke_width(2),
                    ))?;
                }
                None => {
                    chart.draw_series(LineSeries::new(
                        series.points.iter().copied(),
                        color.stroke_width(2),
                    ))?;
                }
            }
        }
        Ok(())
    }
}

// ─── Skeletons ───────────────────────────────────────────────────────────────

/// Metric readings across the three embedding geometries, one panel per metric.
///
/// **A skeleton: nothing is drawn yet**, so [`MetricPanels::has_data`] returns
/// `false` and the driver's "figures with no data are skipped" rule keeps an
/// empty SVG off disk. See the module doc for what the results chapter asks of
/// it — it is not [`MetricVsKappa`], which answers a different question about
/// the same section.
pub struct MetricPanels<'a> {
    #[expect(dead_code, reason = "read once the figure is drawn")]
    cells: &'a CellMap,
    n: usize,
    #[expect(dead_code, reason = "read once the figure is drawn")]
    space: ObjectiveSpace,
}

impl<'a> MetricPanels<'a> {
    #[must_use]
    pub fn new(cells: &'a CellMap, n: usize, space: ObjectiveSpace) -> Self {
        Self { cells, n, space }
    }

    /// Always `false` while this is a skeleton.
    #[must_use]
    pub fn has_data(&self) -> bool {
        false
    }
}

impl Figure for MetricPanels<'_> {
    fn name(&self) -> String {
        format!("exp2_metric_panels_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (1500, 1000)
    }

    fn draw<DB: DrawingBackend>(&self, _root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        Ok(())
    }
}
