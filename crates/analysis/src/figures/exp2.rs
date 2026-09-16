//! Experiment 2 (`metric-results`) — what effect does curvature have on the
//! established visualisation metrics?
//!
//! [`MetricTrend`] is the figure this module draws: **curvature on x, the
//! metric reading on y, every metric bounded in `[0, 1]` overlaid on one pair
//! of axes, one figure per embedding geometry.** Two figures per x axis, not
//! three — see *No Euclidean panel* below.
//!
//! Two figures accompany it, both **one panel per metric** and neither carrying
//! a legend: [`MetricSpread`] draws one bounded metric's curve alone with its
//! interquartile band, which is how the spread behind a flat-looking median
//! gets reported (see *How wide the spread is*), and [`UnboundedTrend`] is the
//! same idea for the metrics that are not bounded at all (see *The unbounded
//! metrics*).
//!
//! ### Two x axes
//!
//! "Curvature" is two different numbers here, and the figure is drawn once
//! against each ([`XAxis`]):
//!
//! * **`|K|`**, the `curvature_magnitude` hyperparameter — what the optimiser
//!   *chose*, log-searched over `1e-6..5` (`config/params.json`);
//! * **`κ = |K|·R_rms²`**, the dimensionless embedding curvature — what the
//!   embedding *ended up at* once its spread is folded in
//!   (`TrialRecord::kappa`, thesis `@eq:kappa`).
//!
//! Neither reads for the other. κ is the quantity the thesis argues is
//! comparable across embeddings; `|K|` is the knob a practitioner turns. And on
//! the sphere κ is blind to `|K|` altogether — `crates/analysis/CLAUDE.md`
//! § *One κ, one gauge* — so the `|K|` panel is the only one of the two on
//! which the spherical trials actually move. The two share every other choice
//! below and a filename that differs only in its slug, `metric_vs_kappa` or
//! `metric_vs_curvature`.
//!
//! ### What is drawn, and from which trials
//!
//! The crate's own rule is that a panel must say what population it draws from.
//! This one draws **the Pareto front of every `all_off` cell** at the given N
//! and geometry, pooled over all datasets:
//!
//! * *front points*, not every trial — the thesis compares visualisation
//!   corpora, and a corpus *is* the non-dominated set, so a trial the search
//!   discarded is not part of what the figure describes. The reduction is
//!   not done here: `bin/figures.rs` passes a `CellMap` already reduced to
//!   fronts in the scoring space, so the front and the space are decided in
//!   one place and this module only pools. (It drew every trial until
//!   2026-09-14, on the argument that the response of a metric to κ is a
//!   property of the trial population; the methods chapter says otherwise.)
//! * *`all_off` only*, so the auxiliary loss weights are pinned at zero and do
//!   not vary along with κ. Pooling every setting would mix two effects.
//!
//! Each metric is drawn as **the median of its reading in x bins**, in the
//! metric's colour and dash, wherever a bin holds at least [`MIN_PER_BIN`]
//! points — and **nothing else on this figure**. Every curve is binned on one
//! shared set of edges ([`super::BinScale::edges`] once, then
//! [`super::binned_band_on`] per metric), so the overlaid curves are sampled at
//! the same x positions and can be read against each other — which is the whole
//! reason they share a panel. (It was a per-front-point scatter briefly in
//! 2026-09: five metrics times a couple of thousand points is a cloud, not a
//! trend.)
//!
//! **Five medians, no bands.** The spread is real and large, but five shaded
//! ranges on one pair of axes overlap into a wash that hides the very curves
//! the panel exists to compare. It is reported instead one metric at a time, by
//! [`MetricSpread`] — see below.
//!
//! ### How wide the spread is
//!
//! [`MetricSpread`] is the same curve, alone, with the **interquartile range of
//! each bin** shaded around it: one panel per (metric, geometry, x axis, scale).
//! That is far more panels than a results chapter can carry, which is the point
//! — the thesis shows one and the appendix holds the rest, so every metric's
//! spread is reported without any of it being curated away here. It is built
//! from the same [`Rendering`] as the overlay, so a panel taken out of the
//! appendix carries the identical curve rather than a re-binning of the same
//! trials, and [`Trend`] is the unit both figures are made of.
//!
//! **Median and quartiles, not mean ± SD.** The readings in a bin are bounded
//! in `[0, 1]`, skewed, and often bimodal — a collapsed embedding piles every
//! metric at one end — so the two differ in kind rather than in detail. On the
//! pooled hyperbolic front at N = 1000 the κ ≈ 2e-7 collapse bin reads
//! `neighborhood_hit` mean 0.199 against median 0.018, with an IQR of
//! `[0.018, 0.438]` that is asymmetric by a factor of 20; `trustworthiness`
//! runs mean 0.707 against median 0.593 three bins along. A ±1 SD band leaves
//! `[0, 1]` in 14 of the 95 bin-metric pairs a panel set draws — below zero on
//! that collapse bin, above one at the high-κ end — where it sits clipped
//! against the frame claiming spread the metric cannot have. The quartiles
//! stay inside the axis and keep the asymmetry a symmetric band hides.
//!
//! A bin under the floor **breaks** the curve rather than being joined across.
//! On the spherical κ panel only a handful of [`N_BINS`] bins clear it, and a
//! line drawn straight over the rest would assert a trend across κ nothing was
//! measured at. A run of a single bin is drawn as a dot and a vertical bar, so
//! an isolated bin between two breaks is not simply lost.
//!
//! ### Two axis scales, two files
//!
//! The bins and the axis are one choice ([`super::BinScale`]), because a
//! geometric bin centre drawn on a linear axis lands in the wrong place. A
//! geometry whose x spans a decade or more is rendered **twice**: once
//! logarithmic, which is the natural reading of a quantity covering seven
//! decades, and once linear, which shows where the trials actually sit — 64% of
//! the hyperbolic corpus is below κ = 1.8, and the log axis deliberately
//! flattens that. Neither substitutes for the other, so both are written and
//! the linear one carries a `_linear` in its filename. [`UnboundedTrend`] makes
//! the same choice the same way.
//!
//! Below a decade there is only the linear rendering, and it carries no suffix:
//! it is that geometry's only figure, and a suffix would imply a log companion
//! that does not exist. That is the spherical κ panel — plotters finds no key
//! points inside a sub-decade log range and drew no x ticks at all. The
//! spherical `|K|` panel spans the searched decades like the hyperbolic ones
//! and gets the pair; the rule is data-driven, not per geometry.
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
//! ### The unbounded metrics
//!
//! The registry holds three more projected-space metrics — `dunn_index`,
//! `davies_bouldin_ratio`, `cluster_density_measure` — that are ratios,
//! unbounded above, and so not objectives. [`UnboundedTrend`] draws them from
//! the same trials, on the same bins and the same x axes, under the same
//! log/`_linear` rule, and differs from [`MetricTrend`] in exactly the ways
//! their unboundedness forces:
//!
//! * **A bare median, with no band.** The curve is the same binned median
//!   ([`super::binned_median_on`] on the same edges), but a quartile band around
//!   it would be meaningless on an axis spanning eight decades of a ratio, and
//!   the y range below is fenced against exactly the values such a band would
//!   stretch it over.
//! * **One panel per metric.** Their bin medians sit at ~0.9, ~0.001 and
//!   ~10 → 10⁸ respectively; no single axis reads all three, and the y-axis
//!   label naming the metric is the only legend needed. The set is the
//!   registry's `Space::Projected && !is_objective()` ([`unbounded_metrics`]),
//!   not a list here.
//! * **The y axis is the metric's own**, and its scale is decided by the same
//!   rule as x ([`natural_scale`] over the medians): logarithmic where they
//!   span a decade, linear otherwise. Both are this crate's own tick
//!   coordinates ([`super::LinearTicks`], [`super::LogTicks`]) because the range is data:
//!   plotters drops the last linear tick and labels a 1.2-decade log axis once.
//! * **A log y axis breaks the curve at bins that are off its scale**
//!   ([`YAxis::of`]). A collapsed embedding — every hyperbolic trial below
//!   κ ≈ 1e-5 — drives all three ratios to zero, or to ~1e-34 for
//!   `cluster_density_measure`, thirty decades under the body of the curve.
//!   The range is fenced at the *low* end only (Tukey, in decades —
//!   [`super::log_range_above_floor`]); the high end is the metric doing what
//!   it measures, and the two-decade rise of `cluster_density_measure` in its
//!   last κ bin is a finding, not a tail.
//! * The curve is drawn in the geometry's colour, solid, since there is one
//!   per panel. Filenames put the metric where the other figure puts
//!   `metric`: `exp2_dunn_index_vs_kappa_hyperbolic_linear_N1000_obj10`.
//!
//! ### No Euclidean panel
//!
//! `κ = |K|·R_rms²` and Euclidean space has `K = 0` exactly, so every Euclidean
//! trial sits at `κ = 0`: one column of points, no axis, nothing to read. The
//! panel is not drawn. A flat reference, if one is wanted, is a different figure
//! — the same metrics' *distributions* at κ = 0 — not this one with another
//! argument.
//!
//! The spherical κ panel **is** drawn, and its κ window is a factor of ~1.4
//! rather than the hyperbolic seven decades. That is not a plotting failure: it
//! is the pole mismatch between `Sphere::center` and `lift_pca_to_manifold`
//! documented in `crates/analysis/CLAUDE.md` § *One κ, one gauge*, which pins
//! spherical κ at `mean(θ²)` and makes it blind to `|K|`. Reading it beside the
//! spherical `|K|` panel is how that shows.
//!
//! ### Layout
//!
//! Each panel is half of Exp 1's canvas ([`PANEL`]), so two of them occupy the
//! width one Exp 1 figure does and sit side by side at the A4 text width. No
//! panel carries the legend: it is its own file, [`MetricLegend`], a portrait
//! canvas exactly [`PANEL`] tall, meant to be set at the **left** of a pair so
//! the row reads legend, hyperbolic, spherical. Every canvas in the row is the
//! same height, so the pair lines up without any alignment trick. One legend
//! serves both x axes — the curves are the same metrics in the same colours and
//! dashes whichever way the trials are binned — so it carries no axis slug. The
//! per-metric figures need none of it: their y axis names the metric.
//!
//! ### Not this module's other figures
//!
//! The Spearman metric-dependence heatmap of `<metric-dependence>` — whether
//! the metrics order the same visualisations the same way — is
//! [`super::exp2_dependence::MetricDependence`], drawn from the same
//! population ([`SETTING`], front points) and reading metrics through the same
//! [`reading`], so the two figures cannot disagree about orientation or about
//! which trials count. The dataset-by-metric comparison of the three corpora
//! that `<metric-results>` asks for is
//! [`super::exp2_region_gain::RegionGain`]: the curved-versus-Euclidean R2
//! gain under every preference region, drawn from the stage-1 table rather
//! than from the trials, because a corpus is a front and the region-restricted
//! indicator is the crate's one principled way to reduce a front to a number
//! per metric.

use plotters::coord::Shift;
use plotters::prelude::*;

use fitting_core::metrics::{Metric, Space, ALL};

use super::{
    binned_band_on, binned_median_on, draw_legend_grid, geometry_color, log_tick, metric_color,
    metric_dash, padded_log_range, padded_range, BinBand, BinScale, CellMap, Figure, LegendEntry,
    LinearTicks, LogTicks, Res, CURVED, OBJECTIVES, OK_BLACK,
};
use crate::objectives::is_minimized_metric;
use crate::records::TrialRecord;
use crate::style_mesh;

/// The loss-weight setting the panels draw from: the baseline, where every
/// auxiliary weight is zero, so curvature is the only thing varying with the
/// metric. `exp2_dependence` draws from the same one.
pub(super) const SETTING: &str = "all_off";

/// Bins across a panel's whole range, spaced as the axis is. Both figures bin
/// on these.
///
/// Sized for the population, which is a **pooled front** — ~1,600 points per
/// (geometry, N) at N = 1000, not the ~9,000 trials the search visited. At 30
/// bins the hyperbolic κ axis leaves five bins under [`MIN_PER_BIN`] (7 to 13
/// points), so the curve breaks four times in a range that is densely sampled;
/// at 20 one bin is thin (13 points, the gap just above the collapse spike) and
/// at 15 none is. 20 keeps the resolution while costing one break.
const N_BINS: usize = 20;

/// Which curvature a [`MetricTrend`] panel puts on x — see *Two x axes* in the
/// module doc. It decides the value read off each trial, the axis title and
/// the filename slug, and nothing else: binning, scale choice and drawing are
/// the same code for both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XAxis {
    /// `κ = |K|·R_rms²`, the embedding curvature ([`TrialRecord::kappa`]).
    Kappa,
    /// `|K|`, the `curvature_magnitude` hyperparameter the search chose.
    Curvature,
}

impl XAxis {
    /// The x value of one trial, or `None` where the record does not carry it.
    pub(super) fn value(self, record: &TrialRecord) -> Option<f64> {
        match self {
            Self::Kappa => record.kappa(),
            // The same `|curvature|` fallback `kappa()` uses. Only Euclidean
            // sweeps omit `curvature_magnitude`, and Euclidean has no panel
            // here, so it is robustness rather than a path any panel takes.
            Self::Curvature => record
                .curvature_magnitude
                .or_else(|| record.curvature.map(f64::abs)),
        }
    }

    /// The axis title. Greek resolves in both backends; arrows and geometric
    /// shapes do not, so the orientation is stated in words on the y axis, and
    /// the magnitude bars are ASCII.
    fn desc(self) -> &'static str {
        match self {
            Self::Kappa => "κ",
            Self::Curvature => "|K|",
        }
    }

    /// The filename slug: `exp2_metric_vs_<tag>_…`.
    fn tag(self) -> &'static str {
        match self {
            Self::Kappa => "kappa",
            Self::Curvature => "curvature",
        }
    }
}

/// Points a bin needs before it is drawn, in either figure. The pooled front
/// is ~1,600 points per (geometry, N) at N = 1000 over [`N_BINS`] bins, so a
/// bin at this floor is already a sparse one — which on a log axis is exactly
/// the tail, where a median over a handful of points flips on one of them, and
/// where a quartile is barely defined at all.
///
/// **This floor, not emptiness, is what breaks a curve.** On the hyperbolic κ
/// panel exactly one bin is thin (13 points) and none is empty; on the
/// spherical κ panel 13 of 20 are thin and one is genuinely empty, which is why
/// that panel reads as two islands.
const MIN_PER_BIN: usize = 20;

/// Opacity of a [`MetricSpread`] band. One band on the panel, nothing to stack
/// with and nothing to hide, so it is drawn firmly enough to read as the
/// figure's subject rather than as a tint behind the curve.
const BAND_ALPHA: f64 = 0.22;

/// The y axis of a [`MetricTrend`] panel: the unit interval with a little
/// room past each end, so a band edge sitting exactly at 0 or 1 —
/// trustworthiness on a good embedding does — is drawn whole rather than
/// flush against the frame. The ticks stay on `0.0, 0.2, …, 1.0`; the padding
/// is smaller than a step.
const Y_RANGE: (f64, f64) = (-0.03, 1.03);

/// Every trial the panels at one N and geometry draw from — the [`SETTING`]
/// cells, pooled over datasets — paired with its *x* value.
///
/// A trial without an x value is not a point on this figure at all. The `> 0`
/// is the log axis' requirement, not a quality filter: κ = 0 is Euclidean,
/// which has no panel here, and |K| is searched from 1e-6 up.
fn pooled_trials<'a>(
    cells: &'a CellMap,
    n: usize,
    geometry: &str,
    x: XAxis,
) -> Vec<(f64, &'a TrialRecord)> {
    cells
        .iter()
        .filter(|(cell, _)| cell.setting == SETTING && cell.n == n && cell.geometry == geometry)
        .flat_map(|(_, records)| records.iter())
        .filter_map(|r| {
            x.value(r)
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| (v, r))
        })
        .collect()
}

/// How an axis over *values* is naturally spaced: logarithmic when the
/// positive values span a decade or more, linear otherwise.
///
/// **A log axis needs a decade to label.** plotters derives a log scale's key
/// points from its endpoints and finds none inside a window narrower than one
/// decade, so the spherical κ panel — pinned to a factor of ~1.4 by the
/// wrong-pole gauge — came out with no x ticks at all. Below a decade the axis
/// is linear, which is also the honest rendering: nothing about that window is
/// multiplicative.
///
/// The same rule serves both axes: x over the trials' κ or |K|, and, for
/// [`UnboundedTrend`], y over one metric's bin medians.
pub(super) fn natural_scale(values: &[f64]) -> BinScale {
    match padded_log_range(values, 0.0) {
        Some((lo, hi)) if hi / lo >= 10.0 => BinScale::Log,
        _ => BinScale::Linear,
    }
}

/// The x-axis scales a geometry is rendered at, given its natural one.
///
/// Where the natural axis is logarithmic, the linear rendering of the same
/// trials is drawn as well, as a second file. Seven decades of x compressed
/// onto equal-width bins is a different reading of the same corpus — it shows
/// where the trials actually *are*, which the log axis deliberately flattens —
/// and neither is a substitute for the other.
pub(super) fn renderings(natural: BinScale) -> &'static [BinScale] {
    match natural {
        BinScale::Log => &[BinScale::Log, BinScale::Linear],
        BinScale::Linear => &[BinScale::Linear],
    }
}

/// The x range of a panel drawn on *scale*: the span of the *drawn* bin
/// centres, padded a little; `None` when there are none.
///
/// The axis spans the drawn bins, not the trials: a curve stops at the last
/// bin that cleared [`MIN_PER_BIN`], and the sparse tail beyond it — there are
/// trials there, just too few per bin to take a median of — would otherwise
/// read as empty axis. Padded in the axis' own metric so the end points do not
/// sit on the frame; not snapped to whole decades, which would reopen exactly
/// that gap. plotters puts a log axis' ticks on powers of ten whatever the
/// endpoints are.
fn x_range_of(drawn: &[f64], scale: BinScale) -> Option<(f64, f64)> {
    match scale {
        BinScale::Log => padded_log_range(drawn, 0.03),
        BinScale::Linear => padded_range(drawn, 0.03),
    }
}

/// Half of `exp1::MatchedGain::size()` (740 x 450), so two of these occupy the
/// width one Exp 1 figure does and the pair sits side by side at the A4 text
/// width.
///
/// plotters font sizes are absolute canvas units, so halving the canvas does
/// *not* halve the text: at ~80 mm on the page `style_mesh!`'s 13 px tick
/// labels land at roughly 8 pt, against Exp 1's ~6 pt. That is what a
/// half-width figure wants — it is read at half the size.
pub(super) const PANEL: (u32, u32) = (370, 300);

/// Width of the [`MetricLegend`] canvas. One column of the metrics' full wire
/// names at the 12 px [`draw_legend_grid`] font: `1-normalized_stress` is the
/// longest at ~125 px of text, plus the 18 px swatch and its gutters. Narrower
/// than [`PANEL`] is tall, so the canvas is portrait.
const LEGEND_WIDTH: u32 = 130;

/// Height of one legend row. [`draw_legend_grid`] divides whatever area it is
/// given evenly between its rows, so the legend hands it a block of exactly
/// this many pixels per entry and centres that block on the canvas, rather
/// than letting five entries drift apart over [`PANEL`]'s full height.
const LEGEND_ROW: u32 = 20;

/// One panel's worth of binning context: the pooled trials at one (geometry,
/// N) paired with their x values, the bins they are laid on, and how that axis
/// is spaced.
///
/// Every Exp 2 trend figure draws from exactly this — same population, same
/// edges, same log/`_linear` pair — so it is built once ([`Rendering::all`])
/// and each `panels` constructor iterates it. That is what makes
/// [`MetricSpread`]'s appendix panel the same curve as the one
/// [`MetricTrend`] overlays, rather than a second binning of the same trials.
struct Rendering<'a> {
    geometry: &'static str,
    kept: Vec<(f64, &'a TrialRecord)>,
    edges: Vec<f64>,
    scale: BinScale,
    /// Whether this is the linear rendering of a geometry whose natural axis is
    /// logarithmic, which is the only thing the filename has to distinguish.
    alternate: bool,
}

impl<'a> Rendering<'a> {
    /// Every (curved geometry, axis scale) a figure at this N and x axis is
    /// drawn for, in [`CURVED`] order. A geometry whose x cannot be binned at
    /// all is absent.
    fn all(cells: &'a CellMap, n: usize, x: XAxis) -> Vec<Rendering<'a>> {
        let mut out = Vec::new();
        for geometry in CURVED {
            let kept = pooled_trials(cells, n, geometry, x);
            let xs: Vec<f64> = kept.iter().map(|(v, _)| *v).collect();
            let natural = natural_scale(&xs);
            for &scale in renderings(natural) {
                let Some(edges) = scale.edges(&xs, N_BINS) else {
                    continue;
                };
                out.push(Rendering {
                    geometry,
                    kept: kept.clone(),
                    edges,
                    scale,
                    alternate: scale != natural,
                });
            }
        }
        out
    }

    /// One metric's binned curve on these edges, or `None` when the sweeps do
    /// not carry it or no bin cleared [`MIN_PER_BIN`].
    fn trend(&self, metric: Metric) -> Option<Trend> {
        Trend::of(&Series::build(metric, &self.kept)?, &self.edges, self.scale)
    }

    /// The x range of a panel whose curves reached the bin centres *drawn*,
    /// falling back to the bins' own span.
    fn x_range(&self, drawn: &[f64]) -> (f64, f64) {
        let span = (self.edges[0], self.edges[self.edges.len() - 1]);
        x_range_of(drawn, self.scale).unwrap_or(span)
    }
}

/// One metric's points: every front point's `(x, reading)`, unbinned. Every
/// figure builds one per metric and bins it its own way.
struct Series {
    metric: Metric,
    label: String,
    color: RGBColor,
    dash: Option<(i32, i32)>,
    points: Vec<(f64, f64)>,
}

/// One metric's drawn curve on a [`MetricTrend`] panel: its median line and
/// the interquartile band around it, as runs of consecutive bins.
///
/// More than one run where a bin inside the range fell under [`MIN_PER_BIN`].
/// The curve breaks there rather than running a straight segment across it:
/// on the spherical κ panel only seven of [`N_BINS`] bins clear the floor, and
/// a line joining them would assert a trend over κ nothing was measured at.
struct Trend {
    metric: Metric,
    label: String,
    color: RGBColor,
    dash: Option<(i32, i32)>,
    runs: Vec<Vec<BinBand>>,
}

/// Every `[0, 1]`-bounded metric against one curvature ([`XAxis`]), for one
/// embedding geometry.
pub struct MetricTrend {
    x: XAxis,
    geometry: &'static str,
    n: usize,
    trends: Vec<Trend>,
    x_range: (f64, f64),
    /// How the x axis is spaced, and how the bins behind `trends` were laid
    /// down — the two are the same choice, so they are one field.
    scale: BinScale,
    /// Whether this is the linear rendering of a geometry whose natural axis is
    /// logarithmic, which is the only thing the filename has to distinguish.
    alternate: bool,
}

impl MetricTrend {
    /// One panel per curved geometry, in [`CURVED`] order, with *x* on the x
    /// axis; a geometry with no binnable trials is absent from the result
    /// rather than returned empty, the same contract `exp4::R2Bars::panels`
    /// has.
    #[must_use]
    pub fn panels(cells: &CellMap, n: usize, x: XAxis) -> Vec<MetricTrend> {
        Rendering::all(cells, n, x)
            .iter()
            .filter_map(|r| Self::panel(x, n, r))
            .collect()
    }

    /// One panel at one axis scale, or `None` when nothing bins. Every metric
    /// is binned on the rendering's one set of edges, so the overlaid curves
    /// are sampled at the same x and can be read against each other — which is
    /// the whole reason they share a panel.
    fn panel(x: XAxis, n: usize, r: &Rendering) -> Option<MetricTrend> {
        let trends: Vec<Trend> = OBJECTIVES.iter().filter_map(|&m| r.trend(m)).collect();
        if trends.is_empty() {
            return None;
        }

        // The span of the bins that were drawn, not of every trial: a metric
        // absent from the sweeps drops its rows, and a bin no curve reaches
        // would be empty axis.
        let drawn: Vec<f64> = trends
            .iter()
            .flat_map(|t| t.runs.iter().flatten().map(|b| b.centre))
            .collect();

        Some(MetricTrend {
            x,
            geometry: r.geometry,
            n,
            trends,
            x_range: r.x_range(&drawn),
            scale: r.scale,
            alternate: r.alternate,
        })
    }

    /// Always true for a panel [`MetricTrend::panels`] returned; kept so the
    /// driver reads the same as every other figure's.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.trends.is_empty()
    }
}

impl Series {
    /// One metric's `(x, reading)` per front point that carries it, or `None`
    /// when the sweeps do not carry the metric at all.
    fn build(metric: Metric, kept: &[(f64, &TrialRecord)]) -> Option<Series> {
        let points: Vec<(f64, f64)> = kept
            .iter()
            .filter_map(|(x, r)| reading(metric, r).map(|v| (*x, v)))
            .collect();
        if points.is_empty() {
            return None;
        }
        Some(Series {
            metric,
            label: label(metric),
            color: metric_color(metric.name()),
            dash: metric_dash(metric.name()),
            points,
        })
    }

    /// The same points binned for an [`UnboundedTrend`] curve: `(bin centre,
    /// median)` at each bin that cleared [`MIN_PER_BIN`], in x order.
    fn binned(&self, edges: &[f64], scale: BinScale) -> Vec<(f64, f64)> {
        let (xs, ys): (Vec<f64>, Vec<f64>) = self.points.iter().copied().unzip();
        let (centres, medians) = binned_median_on(edges, scale, &xs, &ys, MIN_PER_BIN);
        centres.into_iter().zip(medians).collect()
    }

    /// The same points binned for a [`MetricTrend`] curve: the median and
    /// interquartile range of each bin, `None` where the bin is under
    /// [`MIN_PER_BIN`].
    fn band(&self, edges: &[f64], scale: BinScale) -> Vec<Option<BinBand>> {
        let (xs, ys): (Vec<f64>, Vec<f64>) = self.points.iter().copied().unzip();
        binned_band_on(edges, scale, &xs, &ys, MIN_PER_BIN)
    }
}

impl Trend {
    /// One metric's curve over *edges*, or `None` when no bin cleared
    /// [`MIN_PER_BIN`].
    fn of(series: &Series, edges: &[f64], scale: BinScale) -> Option<Trend> {
        let mut runs: Vec<Vec<BinBand>> = Vec::new();
        let mut open = false;
        for bin in series.band(edges, scale) {
            match bin {
                Some(b) => {
                    if !open {
                        runs.push(Vec::new());
                        open = true;
                    }
                    runs.last_mut().expect("a run was just opened").push(b);
                }
                None => open = false,
            }
        }
        if runs.is_empty() {
            return None;
        }
        Some(Trend {
            metric: series.metric,
            label: series.label.clone(),
            color: series.color,
            dash: series.dash,
            runs,
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
pub(super) fn reading(metric: Metric, record: &TrialRecord) -> Option<f64> {
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

/// The series label: the metric's wire name, marked as flipped where the metric
/// is minimised. The full name rather than `Metric::short()` — a thesis figure
/// is read once and slowly, and `1-normalized_stress` says what `1-stress` only
/// implies. It is what sizes [`LEGEND_COLS`].
///
/// ASCII hyphen, not U+2212 — the bitmap backend renders anything outside
/// Latin-1 + Greek as tofu.
pub(super) fn label(metric: Metric) -> String {
    flipped(metric, metric.name())
}

/// [`label`] on the registry abbreviation (`1-stress`), for an axis that has to
/// fit nine of them — the dependence heatmap's.
pub(super) fn short_label(metric: Metric) -> String {
    flipped(metric, metric.short())
}

/// *name*, prefixed `1-` where the metric is minimised.
fn flipped(metric: Metric, name: &str) -> String {
    if is_minimized_metric(metric) {
        format!("1-{name}")
    } else {
        name.to_string()
    }
}

impl Figure for MetricTrend {
    fn name(&self) -> String {
        // Only the *alternate* rendering is marked. The spherical κ panel is
        // linear too, but it is that geometry's only figure on that axis — a
        // suffix there would imply a log companion that does not exist.
        let axis = if self.alternate { "_linear" } else { "" };
        format!(
            "exp2_metric_vs_{}_{}{axis}_N{}",
            self.x.tag(),
            self.geometry,
            self.n
        )
    }

    fn size(&self) -> (u32, u32) {
        PANEL
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        draw_bounded(self, root)
    }
}

/// The axis scaffolding the two `[0, 1]`-metric figures share: the curvature on
/// x at this panel's scale, the unit interval on y.
///
/// [`MetricTrend`] overlays every metric on it, [`MetricSpread`] draws one of
/// them with its band, and the two have to agree down to the tick positions —
/// an appendix panel is read beside the overlay it was taken out of. Splitting
/// the axes from the curves is what guarantees that; the alternative is the
/// same forty lines twice, drifting apart on the next change.
trait BoundedPanel {
    /// Which curvature is on x — the axis title and the tick format.
    fn x_axis(&self) -> XAxis;
    /// Drawn as the panel caption: two of these are read side by side.
    fn geometry(&self) -> &str;
    fn x_range(&self) -> (f64, f64);
    /// How x is spaced, which is also how the bins behind the curves were laid
    /// down, so a curve sits mid-bin as the axis renders it.
    fn scale(&self) -> BinScale;
    /// What the y axis is called. Owned because one figure names a metric.
    fn y_desc(&self) -> String;
    /// Draw the curves themselves onto the built chart.
    fn draw_body<DB, X, Y>(&self, chart: &mut ChartContext<DB, Cartesian2d<X, Y>>) -> Res
    where
        DB: DrawingBackend,
        DB::ErrorType: 'static,
        X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
        Y: plotters::coord::ranged1d::Ranged<ValueType = f64>;
}

/// Build *panel*'s axes on *root* and hand it the chart to draw on.
fn draw_bounded<P: BoundedPanel, DB: DrawingBackend>(
    panel: &P,
    root: &DrawingArea<DB, Shift>,
) -> Res
where
    DB::ErrorType: 'static,
{
    // The geometry is the one identifying thing drawn: two of these are read
    // side by side and have to be tellable apart. N, setting and objective
    // space stay in the filename, as Exp 1's do.
    let yt = LinearTicks::new(Y_RANGE, 6);
    let y_desc = panel.y_desc();
    let x = panel.x_axis();
    let mut builder = ChartBuilder::on(root);
    builder
        .margin(6)
        .margin_right(12)
        .caption(
            panel.geometry(),
            ("sans-serif", 14)
                .into_font()
                .style(FontStyle::Bold)
                .color(&OK_BLACK),
        )
        .x_label_area_size(34)
        .y_label_area_size(46);

    let (lo, hi) = panel.x_range();
    if panel.scale() == BinScale::Log {
        let mut chart = builder.build_cartesian_2d((lo..hi).log_scale(), yt.clone())?;
        style_mesh!(chart.configure_mesh())
            .x_desc(x.desc())
            .y_desc(&y_desc)
            .x_label_formatter(&log_tick)
            .y_label_formatter(&|v| yt.label(v))
            // A hint, not a count: plotters walks decades on a log axis and
            // rounds to whole ones. Ten decades at this width put "0.0001"
            // next to "0.001", which collides once the figure is set at half
            // the A4 text width; asking for fewer thins them to every other
            // decade instead.
            .x_labels(5)
            .draw()?;
        panel.draw_body(&mut chart)?;
    } else {
        // Own ticks rather than plotters': its float walk drops the tick at the
        // right end of the axis — see `LinearTicks`.
        let ticks = LinearTicks::new((lo, hi), 4);
        let mut chart = builder.build_cartesian_2d(ticks.clone(), yt.clone())?;
        style_mesh!(chart.configure_mesh())
            .x_desc(x.desc())
            .y_desc(&y_desc)
            .x_label_formatter(&|v| ticks.label(v))
            .y_label_formatter(&|v| yt.label(v))
            .draw()?;
        panel.draw_body(&mut chart)?;
    }
    Ok(())
}

impl BoundedPanel for MetricTrend {
    fn x_axis(&self) -> XAxis {
        self.x
    }

    fn geometry(&self) -> &str {
        self.geometry
    }

    fn x_range(&self) -> (f64, f64) {
        self.x_range
    }

    fn scale(&self) -> BinScale {
        self.scale
    }

    fn y_desc(&self) -> String {
        Y_DESC.to_string()
    }

    /// Every metric's median curve, and **only** the medians: five bands on one
    /// panel is what [`MetricSpread`] exists to avoid. The spread of each of
    /// these curves is that figure, one metric at a time.
    fn draw_body<DB, X, Y>(&self, chart: &mut ChartContext<DB, Cartesian2d<X, Y>>) -> Res
    where
        DB: DrawingBackend,
        DB::ErrorType: 'static,
        X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
        Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    {
        for trend in &self.trends {
            for run in &trend.runs {
                draw_median(chart, run, trend.color, trend.dash)?;
            }
        }
        Ok(())
    }
}

/// Every series is oriented so higher is better — `normalized_stress` is drawn
/// as `1 - stress` — and the axis says so, because that is the one thing about
/// this figure a reader cannot recover from the curves.
const Y_DESC: &str = "metric";

/// One run's interquartile band: a polygon up the third quartile and back along
/// the first, filled and unoutlined.
///
/// A run of one bin has no width to fill, so it is drawn as a vertical bar at
/// the bin centre instead — an isolated bin between two breaks is otherwise
/// simply lost, the same reason [`draw_runs`] marks one.
fn draw_band<DB, X, Y>(
    chart: &mut ChartContext<DB, Cartesian2d<X, Y>>,
    run: &[BinBand],
    color: RGBColor,
    alpha: f64,
) -> Res
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
    X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
{
    let fill = color.mix(alpha);
    if let [b] = run {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(b.centre, b.lo), (b.centre, b.hi)],
            fill.stroke_width(3),
        )))?;
        return Ok(());
    }
    let mut outline: Vec<(f64, f64)> = run.iter().map(|b| (b.centre, b.hi)).collect();
    outline.extend(run.iter().rev().map(|b| (b.centre, b.lo)));
    chart.draw_series(std::iter::once(Polygon::new(outline, fill.filled())))?;
    Ok(())
}

/// One run's median curve, dashed where the metric's style table says so — the
/// figure overlays five of them and has to survive a greyscale print.
fn draw_median<DB, X, Y>(
    chart: &mut ChartContext<DB, Cartesian2d<X, Y>>,
    run: &[BinBand],
    color: RGBColor,
    dash: Option<(i32, i32)>,
) -> Res
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
    X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
{
    // A polyline through one point draws nothing.
    if let [b] = run {
        chart.draw_series(std::iter::once(Circle::new(
            (b.centre, b.median),
            3,
            color.filled(),
        )))?;
        return Ok(());
    }
    let points = run.iter().map(|b| (b.centre, b.median));
    match dash {
        Some((dash, gap)) => {
            chart.draw_series(DashedLineSeries::new(
                points,
                dash,
                gap,
                color.stroke_width(2),
            ))?;
        }
        None => {
            chart.draw_series(LineSeries::new(points, color.stroke_width(2)))?;
        }
    }
    Ok(())
}

/// The legend shared by every [`MetricTrend`] panel at one N, on both x axes,
/// as its own portrait file: [`LEGEND_WIDTH`] wide, [`PANEL`] tall, one entry
/// per row.
///
/// It is the union of the panels' curves, in [`OBJECTIVES`] order — the order
/// they are drawn in — so a metric that bins on one geometry and not another is
/// still listed, and a metric the sweeps never wrote is not.
pub struct MetricLegend {
    n: usize,
    entries: Vec<LegendEntry>,
}

impl MetricLegend {
    /// The legend for *panels*, which should be everything
    /// [`MetricTrend::panels`] returned at *n*, over every [`XAxis`].
    #[must_use]
    pub fn from_panels(panels: &[MetricTrend], n: usize) -> Self {
        let entries = OBJECTIVES
            .iter()
            .filter_map(|&metric| {
                panels
                    .iter()
                    .flat_map(|p| p.trends.iter())
                    .find(|t| t.metric == metric)
                    .map(Trend::legend_entry)
            })
            .collect();
        Self { n, entries }
    }

    /// False when no panel had a curve to name — in which case there are no
    /// panels either, and a legend on its own would label nothing.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.entries.is_empty()
    }
}

impl Figure for MetricLegend {
    fn name(&self) -> String {
        format!("exp2_metric_legend_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (LEGEND_WIDTH, PANEL.1)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        // A block of LEGEND_ROW per entry, centred vertically; the grid
        // helper would otherwise space five rows over the full panel height.
        let rows = u32::try_from(self.entries.len()).unwrap_or(u32::MAX);
        let block = rows.saturating_mul(LEGEND_ROW).min(PANEL.1);
        let pad = (PANEL.1 - block) / 2;
        draw_legend_grid(&root.margin(pad, pad, 0, 0), &self.entries, 1)
    }
}

// ─── One metric at a time ────────────────────────────────────────────────────

/// One `[0, 1]`-bounded metric against one curvature ([`XAxis`]), for one
/// embedding geometry: the same median curve [`MetricTrend`] overlays, alone,
/// with its **interquartile band** around it.
///
/// This is the figure that says how wide the spread behind a flat-looking
/// median is. It is drawn for every (metric, geometry, x axis, scale), which is
/// far more panels than a results chapter can carry — the thesis shows one and
/// the appendix holds the rest, which is why the set is complete rather than
/// curated here.
///
/// Everything about it is [`MetricTrend`]'s: the population, the bins
/// ([`Rendering`], so the curve is identical to the overlaid one rather than a
/// second binning), the `[0, 1]` y axis and the metric's own colour and dash.
/// What differs is that there is one curve, it carries the band, and the y axis
/// names the metric instead of saying "metric" — with no legend needed.
pub struct MetricSpread {
    x: XAxis,
    geometry: &'static str,
    n: usize,
    trend: Trend,
    x_range: (f64, f64),
    scale: BinScale,
    alternate: bool,
}

impl MetricSpread {
    /// One panel per (curved geometry, x-axis rendering, objective), in that
    /// nesting order — the same order [`UnboundedTrend::panels`] uses, so the
    /// per-metric panels of one geometry come out together whichever figure
    /// wrote them. A combination with no drawable bin is absent.
    #[must_use]
    pub fn panels(cells: &CellMap, n: usize, x: XAxis) -> Vec<MetricSpread> {
        let mut out = Vec::new();
        for r in Rendering::all(cells, n, x) {
            for &metric in OBJECTIVES {
                let Some(trend) = r.trend(metric) else {
                    continue;
                };
                let drawn: Vec<f64> = trend.runs.iter().flatten().map(|b| b.centre).collect();
                out.push(MetricSpread {
                    x,
                    geometry: r.geometry,
                    n,
                    x_range: r.x_range(&drawn),
                    scale: r.scale,
                    alternate: r.alternate,
                    trend,
                });
            }
        }
        out
    }

    /// Always true for a panel [`MetricSpread::panels`] returned; kept so the
    /// driver reads the same as every other figure's.
    #[must_use]
    pub fn has_data(&self) -> bool {
        self.trend.runs.iter().any(|r| !r.is_empty())
    }
}

impl BoundedPanel for MetricSpread {
    fn x_axis(&self) -> XAxis {
        self.x
    }

    fn geometry(&self) -> &str {
        self.geometry
    }

    fn x_range(&self) -> (f64, f64) {
        self.x_range
    }

    fn scale(&self) -> BinScale {
        self.scale
    }

    /// The metric's own name, oriented — the panel has one curve, so naming it
    /// on the axis is the whole legend.
    fn y_desc(&self) -> String {
        self.trend.label.clone()
    }

    /// The band under the median, which is the point of this figure.
    fn draw_body<DB, X, Y>(&self, chart: &mut ChartContext<DB, Cartesian2d<X, Y>>) -> Res
    where
        DB: DrawingBackend,
        DB::ErrorType: 'static,
        X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
        Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    {
        for run in &self.trend.runs {
            draw_band(chart, run, self.trend.color, BAND_ALPHA)?;
        }
        for run in &self.trend.runs {
            draw_median(chart, run, self.trend.color, self.trend.dash)?;
        }
        Ok(())
    }
}

impl Figure for MetricSpread {
    /// The metric where [`MetricTrend`] writes `metric`, which is also
    /// [`UnboundedTrend`]'s scheme: the two metric sets are disjoint, so the
    /// two figures' directories together hold one panel per projected metric
    /// under one naming rule. Which directory is the driver's choice.
    fn name(&self) -> String {
        let axis = if self.alternate { "_linear" } else { "" };
        format!(
            "exp2_{}_vs_{}_{}{axis}_N{}",
            self.trend.metric.name(),
            self.x.tag(),
            self.geometry,
            self.n
        )
    }

    fn size(&self) -> (u32, u32) {
        PANEL
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        draw_bounded(self, root)
    }
}

// ─── Unbounded metrics ───────────────────────────────────────────────────────

/// Every projected-space metric the registry does *not* bound in `[0, 1]` —
/// the complement of [`OBJECTIVES`] on the same surface, read off the registry
/// the same way. `unbounded_metrics_are_the_three_ratios` pins what that is
/// today.
fn unbounded_metrics() -> impl Iterator<Item = Metric> {
    ALL.iter()
        .copied()
        .filter(|m| m.space() == Space::Projected && !m.is_objective())
}

/// Fraction of the y span kept clear above and below the drawn medians.
const Y_PAD: f64 = 0.05;

/// One unbounded metric against one curvature ([`XAxis`]), for one embedding
/// geometry — see *The unbounded metrics* in the module doc. Same population,
/// bins and x axis as [`MetricTrend`]; what differs is that the median carries
/// no band and the y axis is the metric's own.
pub struct UnboundedTrend {
    x: XAxis,
    geometry: &'static str,
    n: usize,
    metric: Metric,
    /// Runs of consecutive drawn bins, `(x, median)` each. More than one run
    /// only where a bin inside the curve fell outside `y_range`, so the line
    /// breaks there rather than jumping across.
    runs: Vec<Vec<(f64, f64)>>,
    x_range: (f64, f64),
    x_scale: BinScale,
    alternate: bool,
    y_range: (f64, f64),
    y_scale: BinScale,
}

impl UnboundedTrend {
    /// One panel per (curved geometry, x-axis rendering, unbounded metric), in
    /// that nesting order; a combination with no drawable bins is absent.
    #[must_use]
    pub fn panels(cells: &CellMap, n: usize, x: XAxis) -> Vec<UnboundedTrend> {
        let mut out = Vec::new();
        for r in Rendering::all(cells, n, x) {
            for metric in unbounded_metrics() {
                let Some(series) = Series::build(metric, &r.kept) else {
                    continue;
                };
                let Some(y) = YAxis::of(&series.binned(&r.edges, r.scale)) else {
                    continue;
                };
                let drawn: Vec<f64> = y.runs.iter().flatten().map(|p| p.0).collect();
                out.push(UnboundedTrend {
                    x,
                    geometry: r.geometry,
                    n,
                    metric,
                    x_range: r.x_range(&drawn),
                    x_scale: r.scale,
                    alternate: r.alternate,
                    runs: y.runs,
                    y_range: y.range,
                    y_scale: y.scale,
                });
            }
        }
        out
    }

    /// Always true for a panel [`UnboundedTrend::panels`] returned; kept so
    /// the driver reads the same as every other figure's.
    #[must_use]
    pub fn has_data(&self) -> bool {
        self.runs.iter().any(|r| !r.is_empty())
    }
}

/// The y axis of one [`UnboundedTrend`] panel, decided from its bin medians.
struct YAxis {
    scale: BinScale,
    range: (f64, f64),
    /// The medians that sit inside `range`, as runs of consecutive bins.
    runs: Vec<Vec<(f64, f64)>>,
}

impl YAxis {
    /// *points* are one metric's `(x, median)` per drawn bin, in x order.
    /// `None` when none of them can be placed.
    ///
    /// The scale is [`natural_scale`] over the medians, the same rule as x. On
    /// a linear axis every median is drawn and the range is padded around them,
    /// floored at zero — the three metrics are non-negative by construction,
    /// and an axis dipping below zero would say otherwise.
    ///
    /// A log axis is different. Its range is [`super::log_range_above_floor`]
    /// — Tukey's lower fence in decades, the top left open — and a median
    /// outside it is **not drawn**, the line breaking at that bin. Two things
    /// make that necessary rather than cosmetic. A median of zero has no place
    /// on a log axis at all; and `cluster_density_measure` reads ~1e-34 on a
    /// collapsed embedding, which is where every hyperbolic trial below
    /// κ ≈ 1e-5 sits, so the low-κ bins would otherwise stretch the axis over
    /// thirty-odd empty decades and flatten the three the metric actually moves
    /// in. The break is the honest rendering: the metric is not "low" there,
    /// it is off the scale. The fence is one-sided because the failure is:
    /// a collapsed embedding drives these ratios to zero, never up, and the
    /// high end of the curve is the signal.
    fn of(points: &[(f64, f64)]) -> Option<Self> {
        let ys: Vec<f64> = points.iter().map(|p| p.1).collect();
        let scale = natural_scale(&ys);
        let range = match scale {
            BinScale::Log => super::log_range_above_floor(&ys, Y_PAD)?,
            BinScale::Linear => {
                let (lo, hi) = padded_range(&ys, Y_PAD)?;
                (lo.max(0.0), hi)
            }
        };
        let inside = |y: f64| match scale {
            BinScale::Log => y > 0.0 && y >= range.0 && y <= range.1,
            BinScale::Linear => true,
        };

        let mut runs: Vec<Vec<(f64, f64)>> = Vec::new();
        let mut open = false;
        for &p in points {
            if inside(p.1) {
                if !open {
                    runs.push(Vec::new());
                    open = true;
                }
                runs.last_mut().expect("a run was just opened").push(p);
            } else {
                open = false;
            }
        }
        if runs.is_empty() {
            return None;
        }
        Some(Self { scale, range, runs })
    }
}

impl Figure for UnboundedTrend {
    fn name(&self) -> String {
        // The `_linear` marks the alternate x rendering, exactly as
        // `MetricTrend::name` does; the y scale is read off the axis.
        let axis = if self.alternate { "_linear" } else { "" };
        format!(
            "exp2_{}_vs_{}_{}{axis}_N{}",
            self.metric.name(),
            self.x.tag(),
            self.geometry,
            self.n
        )
    }

    fn size(&self) -> (u32, u32) {
        PANEL
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let mut builder = ChartBuilder::on(root);
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
            // Wider than `MetricTrend`'s: a y tick here can be `100000` or
            // `0.0002`, not `0.8`.
            .y_label_area_size(58);

        let (xlo, xhi) = self.x_range;
        let y_desc = label(self.metric);
        let color = geometry_color(self.geometry);
        // Four axis pairings, one chart type each; the curve itself is drawn
        // by one generic body. The x ticks are `MetricTrend`'s; y is always
        // this crate's own ticks, log or linear, because the range is data
        // and plotters loses the last linear tick and labels a short log axis
        // once — see `LinearTicks` and `LogTicks`.
        match (self.x_scale, self.y_scale) {
            (BinScale::Log, BinScale::Log) => {
                let yt = LogTicks::new(self.y_range, 5);
                let mut chart = builder.build_cartesian_2d((xlo..xhi).log_scale(), yt.clone())?;
                style_mesh!(chart.configure_mesh())
                    .x_desc(self.x.desc())
                    .y_desc(y_desc)
                    .x_label_formatter(&log_tick)
                    .y_label_formatter(&|v| yt.label(v))
                    .x_labels(5)
                    .draw()?;
                draw_runs(&mut chart, &self.runs, color)?;
            }
            (BinScale::Log, BinScale::Linear) => {
                let yt = LinearTicks::new(self.y_range, 4);
                let mut chart = builder.build_cartesian_2d((xlo..xhi).log_scale(), yt.clone())?;
                style_mesh!(chart.configure_mesh())
                    .x_desc(self.x.desc())
                    .y_desc(y_desc)
                    .x_label_formatter(&log_tick)
                    .y_label_formatter(&|v| yt.label(v))
                    .x_labels(5)
                    .draw()?;
                draw_runs(&mut chart, &self.runs, color)?;
            }
            (BinScale::Linear, BinScale::Log) => {
                let xt = LinearTicks::new(self.x_range, 4);
                let yt = LogTicks::new(self.y_range, 5);
                let mut chart = builder.build_cartesian_2d(xt.clone(), yt.clone())?;
                style_mesh!(chart.configure_mesh())
                    .x_desc(self.x.desc())
                    .y_desc(y_desc)
                    .x_label_formatter(&|v| xt.label(v))
                    .y_label_formatter(&|v| yt.label(v))
                    .draw()?;
                draw_runs(&mut chart, &self.runs, color)?;
            }
            (BinScale::Linear, BinScale::Linear) => {
                let xt = LinearTicks::new(self.x_range, 4);
                let yt = LinearTicks::new(self.y_range, 4);
                let mut chart = builder.build_cartesian_2d(xt.clone(), yt.clone())?;
                style_mesh!(chart.configure_mesh())
                    .x_desc(self.x.desc())
                    .y_desc(y_desc)
                    .x_label_formatter(&|v| xt.label(v))
                    .y_label_formatter(&|v| yt.label(v))
                    .draw()?;
                draw_runs(&mut chart, &self.runs, color)?;
            }
        }
        Ok(())
    }
}

/// Draw each run as its own solid polyline, so a break between runs is a gap.
fn draw_runs<DB, X, Y>(
    chart: &mut ChartContext<DB, Cartesian2d<X, Y>>,
    runs: &[Vec<(f64, f64)>],
    color: RGBColor,
) -> Res
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
    X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
{
    for run in runs {
        // A polyline through one point draws nothing; mark it so an isolated
        // bin between two breaks is not simply lost.
        if let [p] = run.as_slice() {
            chart.draw_series(std::iter::once(Circle::new(*p, 3, color.filled())))?;
            continue;
        }
        chart.draw_series(LineSeries::new(run.iter().copied(), color.stroke_width(2)))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use fitting_core::cast::count_to_f64;
    use fitting_core::metrics::{CLUSTER_DENSITY_MEASURE, DAVIES_BOULDIN_RATIO, DUNN_INDEX};

    /// One `all_off` hyperbolic cell at N = 1000 whose κ spans four decades,
    /// with every objective reading present, so both bounded figures bin it.
    fn one_cell() -> CellMap {
        let rows: Vec<TrialRecord> = (0..400)
            .map(|i| {
                let t = count_to_f64(i) / 400.0;
                // |K| over four decades with r_rms = 1, so κ = |K|.
                let k = 10f64.powf(-4.0 + 4.0 * t);
                let body = format!(
                    r#"{{"curvature_magnitude":{k},"r_rms":1.0,
                       "trustworthiness":{t},"continuity":{t},
                       "normalized_stress":{t},"shepard_goodness":{t},
                       "neighborhood_hit":{t}}}"#
                );
                serde_json::from_str(&body).expect("trial fixture")
            })
            .collect();
        let mut cells = CellMap::new();
        cells.insert(Cell::new(SETTING, "d", 1000, "hyperbolic"), rows);
        cells
    }

    /// The appendix figure draws exactly the curves the overlay does — one
    /// panel per (metric, geometry, rendering), carrying the same bins — so a
    /// panel pulled out of the appendix is the overlay's curve, not a second
    /// binning of the same trials.
    #[test]
    fn a_spread_panel_is_one_of_the_overlays_curves() {
        let cells = one_cell();
        let overlays = MetricTrend::panels(&cells, 1000, XAxis::Kappa);
        let spreads = MetricSpread::panels(&cells, 1000, XAxis::Kappa);
        assert!(!overlays.is_empty(), "the fixture bins");

        // One panel per curve of every overlay panel, and the same renderings.
        let curves: usize = overlays.iter().map(|p| p.trends.len()).sum();
        assert_eq!(spreads.len(), curves);

        for overlay in &overlays {
            for trend in &overlay.trends {
                let spread = spreads
                    .iter()
                    .find(|s| {
                        s.trend.metric == trend.metric
                            && s.geometry == overlay.geometry
                            && s.scale == overlay.scale
                            && s.alternate == overlay.alternate
                    })
                    .expect("every overlaid curve has its own panel");
                let medians = |runs: &Vec<Vec<BinBand>>| -> Vec<(f64, f64)> {
                    runs.iter()
                        .flatten()
                        .map(|b| (b.centre, b.median))
                        .collect()
                };
                assert_eq!(medians(&spread.trend.runs), medians(&trend.runs));
                assert!(spread.has_data());
                // The two figures are told apart by filename, not by content.
                assert_ne!(spread.name(), overlay.name());
            }
        }
    }

    #[test]
    fn unbounded_metrics_are_the_three_ratios() {
        let got: Vec<Metric> = unbounded_metrics().collect();
        assert_eq!(
            got,
            vec![DAVIES_BOULDIN_RATIO, DUNN_INDEX, CLUSTER_DENSITY_MEASURE]
        );
    }

    #[test]
    fn linear_y_axis_draws_everything_and_floors_at_zero() {
        // Sub-decade medians, one of them zero: linear, all drawn, one run.
        let points = vec![(1.0, 0.0), (2.0, 0.8), (3.0, 0.9), (4.0, 0.95)];
        let y = YAxis::of(&points).expect("drawable");
        assert_eq!(y.scale, BinScale::Linear);
        assert_eq!(y.runs, vec![points]);
        assert!(
            y.range.0.abs() < f64::EPSILON,
            "floored at zero, got {}",
            y.range.0
        );
        assert!(y.range.1 > 0.95);
    }

    #[test]
    fn log_y_axis_breaks_the_curve_at_off_scale_bins() {
        // A body over three decades with a collapsed bin at each end and one
        // in the middle: log, the ~1e-34 and zero medians are outside the
        // Tukey fence and the line breaks there.
        // A high spike at the end is signal, not tail, and stays.
        let mut points: Vec<(f64, f64)> = vec![(0.0, 1e-34), (1.0, 0.0)];
        points.extend((2..12).map(|i| (f64::from(i), 10f64.powi(i % 4 + 1))));
        points.push((12.0, 1e-30));
        points.extend((13..20).map(|i| (f64::from(i), 10f64.powi(i % 4 + 1))));
        points.push((20.0, 1e9));
        let y = YAxis::of(&points).expect("drawable");
        assert_eq!(y.scale, BinScale::Log);
        assert_eq!(y.runs.len(), 2);
        assert_eq!(y.runs[0].len(), 10);
        assert_eq!(y.runs[1].len(), 8);
        assert!(y.range.0 > 1.0 && y.range.0 <= 10.0);
        assert!(y.range.1 >= 1e9 && y.range.1 < 1e11);
    }

    #[test]
    fn nothing_placeable_is_no_axis() {
        assert!(YAxis::of(&[]).is_none());
    }
}
