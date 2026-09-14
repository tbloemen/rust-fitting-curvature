//! Experiment 3 (`curvature-tuning-results`) — can curvature be tuned as its
//! own hyperparameter?
//!
//! [`KappaLanding`] is the figure this module draws: **`|K|` on x, κ on y, one
//! panel per curved embedding geometry.** `curvature_magnitude` is a searched
//! hyperparameter, so given that freedom, where does the search leave κ?
//! κ = `|K|·R_rms²` (`@eq:kappa`) is the quantity the thesis argues is
//! comparable across embeddings — rescaling an embedding changes `K` without
//! changing κ (`<curvature-tuning-results>`, *Separating Curvature from
//! Embedding Scale*) — and this panel is the one place the two are set against
//! each other.
//!
//! ### What is drawn, and from which trials
//!
//! **The Pareto front of every cell** at the given N and geometry, pooled over
//! datasets and over the loss settings of [`super::SETTING_ORDER`]. Front
//! points rather than every trial, because the question is where the search
//! *keeps* κ, not how κ responds to `|K|` across the searched corpus — that
//! response is what `exp2::MetricTrend` draws its trials for, and it is a
//! different population. Fronts are reduced in the objective space the sweeps
//! were written in ([`pareto_front_records`]).
//!
//! `rms_anchored` is left out ([`EXCLUDED`]), the same way `exp4::R2Bars` leaves
//! it out: it pins `R_rms` with a fixed, strong scaling loss so that κ *is*
//! `|K|`, and its points lie on a straight line through a cloud they are not
//! part of. In a single-colour panel that line would be unreadable as a
//! separate population; if it is ever wanted, it is its own series.
//!
//! Every point is one front trial, `(|K|, κ)`, drawn faintly; the line over
//! them is the **median κ in `|K|` bins** ([`N_BINS`] log bins, drawn where a
//! bin holds at least [`MIN_PER_BIN`] points) — the same binning
//! `exp2::MetricTrend` uses, on the same helpers.
//!
//! ### Axes
//!
//! `|K|` is log-searched over `1e-6..5` (`config/params.json`), so x is always
//! logarithmic; a linear axis would put five of the six searched decades in
//! the first pixel. κ takes whichever scale is natural to its span
//! (`exp2::natural_scale`, the rule every Exp 2 axis follows): **log on the
//! hyperbolic panel**, where κ runs from the collapse floor at ~2e-7 to ~40,
//! and **linear on the spherical one**, where κ sits in a factor-of-two window
//! that a log axis cannot label.
//!
//! Each panel is also rendered **linear on both axes**, as a second file
//! suffixed `_linear` — the same rule `exp2::MetricTrend` follows for a
//! geometry whose natural axis is logarithmic ([`renderings`]). The log
//! rendering is the natural reading of six decades of `|K|`; the linear one
//! shows where the front actually sits — most of it below `|K| = 1`, and the
//! `|K| = 5` search boundary as a column. On it the bins are linear too
//! ([`super::BinScale`]: bins and axis are one choice), and κ is linear even
//! on the hyperbolic panel, since a log κ over a linear `|K|` would still
//! flatten what the rendering exists to show.
//!
//! On the log y axis the range is *not* fenced against outliers the way
//! `exp2::UnboundedTrend` fences its ratios. The κ ≈ 2e-7 population is the
//! collapse floor — trials whose embedding shrank to a point — and it is on the
//! front, at every `|K|` from ~1e-2 up. Fencing it off would hide the very
//! thing the raw scatter is drawn to show under the median line.
//!
//! ### Two hazards
//!
//! Both are documented in `crates/analysis/CLAUDE.md` § *One κ, one gauge*, and
//! both are visible on these panels rather than corrected by them:
//!
//! * the κ ≈ 2e-7 band is *collapsed* embeddings, not near-flat space;
//! * on the sphere `R_rms` is measured from the wrong pole, so κ there is
//!   `mean(θ²)`, bounded in `[0, π²]` and blind to `|K|` by construction. The
//!   spherical panel is a near-flat band at κ ≈ 2.5–3 across six decades of
//!   `|K|`, and that flatness is the finding, not a reason to drop the panel.
//!
//! ### No Euclidean panel
//!
//! `K = 0` exactly, so every Euclidean trial sits at κ = 0: one column of
//! points, no axis. Two panels, hyperbolic and spherical, as in Exp 2.
//!
//! ### Layout
//!
//! Each panel is `exp2::PANEL`, so the pair sits side by side at the A4 text
//! width like the Exp 2 rows. The geometry is the only identifying thing drawn;
//! N and the objective space stay in the filename,
//! `exp3_kappa_vs_curvature_<geometry>[_linear]_N<n>_<space>`, under
//! `<out-dir>/experiment_3`. One series per panel, so there is no legend.

use plotters::coord::Shift;
use plotters::prelude::*;

use super::exp2::{natural_scale, renderings, XAxis, PANEL};
use super::{
    binned_median_on, geometry_color, log_tick, padded_log_range, padded_range, BinScale, CellMap,
    Figure, LinearTicks, LogTicks, ObjectiveSpace, Res, CURVED, OK_BLACK, SETTING_ORDER,
};
use crate::pareto::pareto_front_records;
use crate::style_mesh;

/// Settings whose fronts are not pooled — see *What is drawn* in the module
/// doc. Everything else in [`SETTING_ORDER`] is.
const EXCLUDED: [&str; 1] = ["rms_anchored"];

/// Log bins across the panel's `|K|` range for the median line.
const N_BINS: usize = 30;

/// Front points a bin needs before its median is drawn. A pooled front is a few
/// thousand points per (geometry, N), so this is the sparse-tail guard
/// `exp2::MIN_PER_BIN` is, at the same value.
const MIN_PER_BIN: usize = 20;

/// Fraction of the y span kept clear above and below the points.
const Y_PAD: f64 = 0.05;

/// Opacity of one front point. Thousands overlap, so a point is faint and the
/// cloud's density is what reads; at 0.18 the dense bands went solid and the
/// median line in the same colour disappeared into them.
const POINT_ALPHA: f64 = 0.06;

/// Width of the median line.
const TREND_WIDTH: u32 = 3;

/// Where the search leaves κ when curvature is a free hyperparameter: the
/// pooled Pareto fronts of one curved geometry at one N, as `(|K|, κ)`.
pub struct KappaLanding {
    geometry: &'static str,
    n: usize,
    /// Every front point, `(|K|, κ)`.
    points: Vec<(f64, f64)>,
    /// `(bin centre, median κ)` at each `|K|` bin that cleared [`MIN_PER_BIN`].
    trend: Vec<(f64, f64)>,
    x_range: (f64, f64),
    y_range: (f64, f64),
    /// How the x axis is spaced, and how the bins behind `trend` were laid
    /// down — one choice, as in `exp2::MetricTrend`.
    x_scale: BinScale,
    y_scale: BinScale,
    /// Whether this is the linear rendering of a panel whose natural axes are
    /// logarithmic, which is the only thing the filename has to distinguish.
    alternate: bool,
}

impl KappaLanding {
    /// One panel per curved geometry, in [`CURVED`] order, at every rendering
    /// [`renderings`] gives it — log, then its `_linear` companion; a geometry
    /// with no front point that can be placed is absent from the result rather
    /// than returned empty, the same contract `exp2::MetricTrend::panels` has.
    #[must_use]
    pub fn panels(cells: &CellMap, n: usize, space: ObjectiveSpace) -> Vec<KappaLanding> {
        let mut out = Vec::new();
        for geometry in CURVED {
            let points = front_points(cells, n, geometry, space);
            for &scale in renderings(BinScale::Log) {
                if let Some(panel) = Self::panel(geometry, n, &points, scale) {
                    out.push(panel);
                }
            }
        }
        out
    }

    /// One panel at one x scale, or `None` when nothing can be placed.
    ///
    /// On the log rendering y follows [`natural_scale`] over κ: log on the
    /// hyperbolic panel, linear on the spherical one. On the linear rendering
    /// both axes are linear — the point of it is to show where the front
    /// actually sits, and a log κ axis over a linear `|K|` axis would still
    /// flatten that.
    fn panel(
        geometry: &'static str,
        n: usize,
        points: &[(f64, f64)],
        x_scale: BinScale,
    ) -> Option<KappaLanding> {
        let xs: Vec<f64> = points.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = points.iter().map(|p| p.1).collect();

        // The range is the points', not the trend's: the scatter is drawn in
        // full, and the sparse tail beyond the last binnable bin is part of it.
        let (x_range, y_scale) = match x_scale {
            BinScale::Log => (padded_log_range(&xs, 0.03)?, natural_scale(&ys)),
            BinScale::Linear => {
                let (lo, hi) = padded_range(&xs, 0.03)?;
                ((lo.max(0.0), hi), BinScale::Linear)
            }
        };
        let y_range = match y_scale {
            BinScale::Log => padded_log_range(&ys, Y_PAD)?,
            BinScale::Linear => {
                let (lo, hi) = padded_range(&ys, Y_PAD)?;
                (lo.max(0.0), hi)
            }
        };

        // No bins is not "no panel": the scatter still says where the front
        // sits, it just has no summary line over it.
        let trend = x_scale
            .edges(&xs, N_BINS)
            .map(|edges| {
                let (centres, medians) = binned_median_on(&edges, x_scale, &xs, &ys, MIN_PER_BIN);
                centres.into_iter().zip(medians).collect()
            })
            .unwrap_or_default();

        Some(KappaLanding {
            geometry,
            n,
            points: points.to_vec(),
            trend,
            x_range,
            y_range,
            x_scale,
            y_scale,
            alternate: x_scale != BinScale::Log,
        })
    }

    /// Always true for a panel [`KappaLanding::panels`] returned; kept so the
    /// driver reads the same as every other figure's.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.points.is_empty()
    }
}

/// Every front point the panel at one N and geometry draws — the fronts of the
/// pooled cells, each reduced in *space* — as `(|K|, κ)`.
///
/// A trial without both values is not a point on this figure. The `> 0` is the
/// log x axis' requirement (and, on the hyperbolic panel, the log y axis'),
/// not a quality filter: `|K|` is searched from 1e-6 up, and κ = 0 is
/// Euclidean, which has no panel here.
fn front_points(
    cells: &CellMap,
    n: usize,
    geometry: &str,
    space: ObjectiveSpace,
) -> Vec<(f64, f64)> {
    cells
        .iter()
        .filter(|(cell, _)| {
            cell.n == n
                && cell.geometry == geometry
                && SETTING_ORDER.contains(&cell.setting.as_str())
                && !EXCLUDED.contains(&cell.setting.as_str())
        })
        .flat_map(|(_, records)| pareto_front_records(records, space))
        .filter_map(|r| {
            let x = XAxis::Curvature.value(&r)?;
            let y = r.kappa()?;
            (x.is_finite() && x > 0.0 && y.is_finite() && y > 0.0).then_some((x, y))
        })
        .collect()
}

impl Figure for KappaLanding {
    fn name(&self) -> String {
        // Only the alternate rendering is marked, as `exp2::MetricTrend::name`
        // does; the spherical log panel has a linear y axis and no suffix.
        let axis = if self.alternate { "_linear" } else { "" };
        format!(
            "exp3_kappa_vs_curvature_{}{axis}_N{}",
            self.geometry, self.n
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
            // A κ tick can be `0.0001` on the hyperbolic panel, so the wider
            // `UnboundedTrend` gutter rather than `MetricTrend`'s.
            .y_label_area_size(58);

        let (xlo, xhi) = self.x_range;
        let color = geometry_color(self.geometry);
        // Three axis pairings — a log x carries y on its natural scale, a
        // linear x is linear on both. y is always this crate's own ticks, log
        // or linear, because the range is data and plotters loses the last
        // linear tick and labels a short log axis once; a linear x is too.
        match (self.x_scale, self.y_scale) {
            (BinScale::Log, BinScale::Log) => {
                let yt = LogTicks::new(self.y_range, 5);
                let mut chart = builder.build_cartesian_2d((xlo..xhi).log_scale(), yt.clone())?;
                style_mesh!(chart.configure_mesh())
                    .x_desc(X_DESC)
                    .y_desc(Y_DESC)
                    .x_label_formatter(&log_tick)
                    .y_label_formatter(&|v| yt.label(v))
                    .x_labels(5)
                    .draw()?;
                self.draw_marks(&mut chart, color)?;
            }
            (BinScale::Log, BinScale::Linear) => {
                let yt = LinearTicks::new(self.y_range, 4);
                let mut chart = builder.build_cartesian_2d((xlo..xhi).log_scale(), yt.clone())?;
                style_mesh!(chart.configure_mesh())
                    .x_desc(X_DESC)
                    .y_desc(Y_DESC)
                    .x_label_formatter(&log_tick)
                    .y_label_formatter(&|v| yt.label(v))
                    .x_labels(5)
                    .draw()?;
                self.draw_marks(&mut chart, color)?;
            }
            (BinScale::Linear, _) => {
                let xt = LinearTicks::new(self.x_range, 4);
                let yt = LinearTicks::new(self.y_range, 4);
                let mut chart = builder.build_cartesian_2d(xt.clone(), yt.clone())?;
                style_mesh!(chart.configure_mesh())
                    .x_desc(X_DESC)
                    .y_desc(Y_DESC)
                    .x_label_formatter(&|v| xt.label(v))
                    .y_label_formatter(&|v| yt.label(v))
                    .draw()?;
                self.draw_marks(&mut chart, color)?;
            }
        }
        Ok(())
    }
}

/// ASCII bars, not U+2223: the bitmap backend renders anything outside
/// Latin-1 + Greek as tofu. κ itself is fine.
const X_DESC: &str = "|K|";
const Y_DESC: &str = "κ";

impl KappaLanding {
    /// The scatter first, the median line on top of it. Generic over the y
    /// coordinate so the two arms of [`KappaLanding::draw`] share one body.
    fn draw_marks<DB, X, Y>(
        &self,
        chart: &mut ChartContext<DB, Cartesian2d<X, Y>>,
        color: RGBColor,
    ) -> Res
    where
        DB: DrawingBackend,
        DB::ErrorType: 'static,
        X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
        Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    {
        chart.draw_series(
            self.points
                .iter()
                .map(|&p| Circle::new(p, 2, color.mix(POINT_ALPHA).filled())),
        )?;
        chart.draw_series(LineSeries::new(
            self.trend.iter().copied(),
            color.stroke_width(TREND_WIDTH),
        ))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::objectives::{is_minimized_metric, OBJECTIVES};
    use crate::records::TrialRecord;

    const SPACE: ObjectiveSpace = ObjectiveSpace::Current6;

    /// A hyperbolic trial at `|K| = k` with `r_rms` as given (a JSON value, so
    /// `null` can be written), scoring `v` on every objective —
    /// oriented so that a larger `v` dominates. Through the deserialiser
    /// because `SpreadDiagnostics` has no public constructor, and this is the
    /// same path a results line takes.
    fn trial(k: f64, r_rms: &str, v: f64) -> TrialRecord {
        let mut fields = vec![
            format!("\"curvature_magnitude\": {k}"),
            format!("\"r_rms\": {r_rms}"),
        ];
        for &m in OBJECTIVES {
            let raw = if is_minimized_metric(m) { 1.0 - v } else { v };
            fields.push(format!("\"{}\": {raw}", m.name()));
        }
        serde_json::from_str(&format!("{{{}}}", fields.join(", "))).expect("a valid trial line")
    }

    fn cell(setting: &str) -> Cell {
        Cell::new(setting, "tree", 1000, "hyperbolic")
    }

    #[test]
    fn only_front_trials_become_points() {
        let mut cells = CellMap::new();
        // The 0.9 trial dominates the 0.3 one, so the front is just it.
        cells.insert(
            cell("all_off"),
            vec![trial(0.1, "2.0", 0.3), trial(0.01, "3.0", 0.9)],
        );
        let points = front_points(&cells, 1000, "hyperbolic", SPACE);
        assert_eq!(points.len(), 1);
        let (x, y) = points[0];
        assert!((x - 0.01).abs() < 1e-12);
        assert!((y - 0.01 * 9.0).abs() < 1e-12, "κ = |K|·R_rms², got {y}");
    }

    #[test]
    fn rms_anchored_and_other_cells_contribute_nothing() {
        let mut cells = CellMap::new();
        cells.insert(cell("rms_anchored"), vec![trial(0.1, "1.0", 0.9)]);
        cells.insert(
            Cell::new("all_off", "tree", 5000, "hyperbolic"),
            vec![trial(0.1, "1.0", 0.9)],
        );
        cells.insert(
            Cell::new("all_off", "tree", 1000, "spherical"),
            vec![trial(0.1, "1.0", 0.9)],
        );
        assert!(front_points(&cells, 1000, "hyperbolic", SPACE).is_empty());
        // The same cell under a pooled setting does count.
        cells.insert(cell("all_free"), vec![trial(0.1, "1.0", 0.9)]);
        assert_eq!(front_points(&cells, 1000, "hyperbolic", SPACE).len(), 1);
    }

    #[test]
    fn a_trial_without_a_usable_kappa_is_not_a_point() {
        let mut cells = CellMap::new();
        // Three trials scoring identically, so every one is on the front; only
        // the one with a positive R_rms lands on the figure. (A non-finite
        // reading cannot be written in JSON at all; the accessor reads it as
        // absent, which is the `null` case.)
        cells.insert(
            cell("all_off"),
            vec![
                trial(0.1, "null", 0.5),
                trial(0.1, "0.0", 0.5),
                trial(0.1, "2.0", 0.5),
            ],
        );
        let points = front_points(&cells, 1000, "hyperbolic", SPACE);
        assert_eq!(points.len(), 1);
        assert!((points[0].0 - 0.1).abs() < 1e-12);
        assert!((points[0].1 - 0.4).abs() < 1e-12);
    }
}
