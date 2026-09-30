//! Experiment 1 — where the `all_off` corpora land in κ.
//!
//! RQ1 compares the `all_off` corpora of the three geometries, and a curved
//! corpus is only as curved as its embeddings: `|K|` is the hyperparameter the
//! search chose, `κ = |K|·R_g²` is the curvature the visualisation *ended up
//! at*. This figure is the distribution of that κ.
//!
//! ### What is drawn
//!
//! **A histogram of κ over the Pareto-front points of every `all_off` cell**,
//! pooled over datasets, one panel per curved geometry. Front points, because
//! the thesis compares corpora and a corpus is the front — the same population
//! as `exp2::MetricTrend` (`exp2::SETTING`, reduced by
//! [`super::front_cells`]). κ is [`TrialRecord::kappa`], so only files that
//! carry `r_gyration` (`results-rgyr/`) place anything; from `results/` there
//! is no panel.
//!
//! Each bar is a bin of front points; the dashed line is their median κ, and
//! the corner carries the point count and that median.
//!
//! ### Decisions
//!
//! * **No Euclidean panel.** `K = 0` exactly, so every Euclidean point sits at
//!   κ = 0: one bar, no axis. Same rule as Exp 2 and Exp 3.
//! * **The collapse spike is kept.** The κ ≈ 2e-7 band is embeddings shrunk to
//!   a point (`crates/analysis/CLAUDE.md` § *One κ, one gauge*), and it is on
//!   the front. Where the corpora land is the question, and that is part of
//!   the answer; the median is pulled left by it, which the caption has to say.
//! * **Log and `_linear`**, by the `exp2::renderings` rule: a geometry whose κ
//!   spans a decade or more is drawn with log bins on a log axis, and again
//!   with linear bins on a linear axis, where the spike folds into the first
//!   bar and the bulk is visible at its true width. Bins and axis are one
//!   choice ([`BinScale`]).
//!
//! Files `exp1_kappa_hist_<geometry>[_linear]_N<n>_<space>` under
//! `<out-dir>/experiment_1`, each `exp2::PANEL`, so the pair sits side by side
//! like Exp 3's. Nothing identifying is drawn but the geometry.

use fitting_core::cast::count_to_f64;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use super::exp2::{natural_scale, renderings, PANEL, SETTING};
use super::{
    geometry_color, log_tick, padded_log_range, padded_range, BinScale, CellMap, Figure,
    LinearTicks, Res, CURVED, OK_BLACK,
};
use crate::records::TrialRecord;
use crate::stats::median;
use crate::style_mesh;

/// Bins per panel. A pooled `all_off` front is a few thousand points per
/// geometry at N=5000, so 30 bins keep the populated ones well filled.
const N_BINS: usize = 30;

/// Headroom above the tallest bar, as a fraction of it, so the corner label
/// does not sit on a bar.
const Y_HEADROOM: f64 = 0.15;

/// Opacity of a bar's fill; the outline is drawn solid.
const BAR_ALPHA: f64 = 0.55;

/// ASCII, not U+00B7 or U+2223: the viewer's sans-serif has no guaranteed
/// glyph outside Latin-1 + Greek. κ is fine.
const X_DESC: &str = "κ = |K| R_g²";
const Y_DESC: &str = "front points";

/// The κ distribution of the pooled `all_off` fronts of one curved geometry at
/// one N, at one rendering.
pub struct KappaHistogram {
    geometry: &'static str,
    n: usize,
    /// `N_BINS + 1` edges, in the spacing of `scale`.
    edges: Vec<f64>,
    /// Front points per bin; `counts.len() == edges.len() - 1`.
    counts: Vec<usize>,
    median: f64,
    scale: BinScale,
    /// Whether this is the linear rendering of a geometry whose natural axis
    /// is logarithmic — the only thing the filename has to distinguish.
    alternate: bool,
}

impl KappaHistogram {
    /// One panel per curved geometry in [`CURVED`] order, at every rendering
    /// [`renderings`] gives it; a geometry with no placeable front point is
    /// absent rather than returned empty.
    ///
    /// *fronts* must already be reduced to the front
    /// ([`super::front_cells`]); nothing here recomputes it.
    #[must_use]
    pub fn panels(fronts: &CellMap, n: usize) -> Vec<KappaHistogram> {
        let mut out = Vec::new();
        for geometry in CURVED {
            let kappas = front_kappas(fronts, n, geometry);
            let Some(median) = median(&kappas) else {
                continue;
            };
            let natural = natural_scale(&kappas);
            for &scale in renderings(natural) {
                let Some(edges) = scale.edges(&kappas, N_BINS) else {
                    continue;
                };
                out.push(KappaHistogram {
                    geometry,
                    n,
                    counts: bin_counts(&edges, scale, &kappas),
                    edges,
                    median,
                    scale,
                    alternate: scale != natural,
                });
            }
        }
        out
    }

    /// Front points drawn, over all bins.
    #[must_use]
    pub fn total(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Always true for a panel [`KappaHistogram::panels`] returned; kept so
    /// the driver reads the same as every other figure's.
    #[must_use]
    pub fn has_data(&self) -> bool {
        self.total() > 0
    }
}

/// κ of every `all_off` front point at one N and geometry, pooled over
/// datasets. A point without a finite, positive κ is not on the figure: absent
/// is a file without `r_gyration`, zero is Euclidean.
fn front_kappas(fronts: &CellMap, n: usize, geometry: &str) -> Vec<f64> {
    fronts
        .iter()
        .filter(|(cell, _)| cell.n == n && cell.setting == SETTING && cell.geometry == geometry)
        .flat_map(|(_, records)| records.iter().filter_map(TrialRecord::kappa))
        .filter(|k| k.is_finite() && *k > 0.0)
        .collect()
}

/// Count of *values* per bin of *edges*: half-open `[a, b)` bins, the last
/// closed. *edges* must come from [`BinScale::edges`] over these same *values*,
/// so every value is in range by construction; an index is therefore clamped
/// into the bins rather than a value dropped, because `10^log10(max)` can
/// round to just below `max` and would otherwise lose the maximum. Non-finite
/// values and, on a log scale, non-positive ones are skipped, as `edges`
/// skips them. The index is found in the axis' own metric.
fn bin_counts(edges: &[f64], scale: BinScale, values: &[f64]) -> Vec<usize> {
    let n_bins = edges.len() - 1;
    let mut counts = vec![0; n_bins];
    let t = |v: f64| match scale {
        BinScale::Log => v.log10(),
        BinScale::Linear => v,
    };
    let (lo, hi) = (t(edges[0]), t(edges[n_bins]));
    for x in values.iter().map(|&v| t(v)).filter(|x| x.is_finite()) {
        let frac = (x - lo) / (hi - lo) * count_to_f64(n_bins);
        // `as` saturates a negative `frac` to 0, and `min` holds the top.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let i = (frac.floor() as usize).min(n_bins - 1);
        counts[i] += 1;
    }
    counts
}

impl Figure for KappaHistogram {
    fn name(&self) -> String {
        let axis = if self.alternate { "_linear" } else { "" };
        format!("exp1_kappa_hist_{}{axis}_N{}", self.geometry, self.n)
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
            .y_label_area_size(52);

        let max = self.counts.iter().copied().max().unwrap_or(0);
        let y_hi = (count_to_f64(max) * (1.0 + Y_HEADROOM)).max(1.0);
        let yt = LinearTicks::new((0.0, y_hi), 5);
        let (e_lo, e_hi) = (self.edges[0], self.edges[self.edges.len() - 1]);

        match self.scale {
            BinScale::Log => {
                let (xlo, xhi) = padded_log_range(&[e_lo, e_hi], 0.02).ok_or_else(no_range)?;
                let mut chart = builder.build_cartesian_2d((xlo..xhi).log_scale(), yt.clone())?;
                style_mesh!(chart.configure_mesh())
                    .disable_x_mesh()
                    .x_desc(X_DESC)
                    .y_desc(Y_DESC)
                    .x_label_formatter(&log_tick)
                    .y_label_formatter(&|v| yt.label(v))
                    .x_labels(5)
                    .draw()?;
                self.draw_marks(&mut chart, (xhi, y_hi))?;
            }
            BinScale::Linear => {
                let (lo, hi) = padded_range(&[e_lo, e_hi], 0.02).ok_or_else(no_range)?;
                let xt = LinearTicks::new((lo.max(0.0), hi), 5);
                let mut chart = builder.build_cartesian_2d(xt.clone(), yt.clone())?;
                style_mesh!(chart.configure_mesh())
                    .disable_x_mesh()
                    .x_desc(X_DESC)
                    .y_desc(Y_DESC)
                    .x_label_formatter(&|v| xt.label(v))
                    .y_label_formatter(&|v| yt.label(v))
                    .draw()?;
                self.draw_marks(&mut chart, (hi, y_hi))?;
            }
        }
        Ok(())
    }
}

/// The edges always span a positive range ([`BinScale::edges`] returns `None`
/// otherwise), so this is unreachable; it exists to keep `draw` panic-free.
fn no_range() -> crate::Error {
    crate::Error::Plot("κ histogram: empty axis range".into())
}

impl KappaHistogram {
    /// Bars, the median line, and the corner label at *corner* (the top-right
    /// of the data range). Generic over the x coordinate so both renderings
    /// share one body.
    fn draw_marks<DB, X, Y>(
        &self,
        chart: &mut ChartContext<DB, Cartesian2d<X, Y>>,
        corner: (f64, f64),
    ) -> Res
    where
        DB: DrawingBackend,
        DB::ErrorType: 'static,
        X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
        Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    {
        let color = geometry_color(self.geometry);
        let bars = || {
            self.edges
                .windows(2)
                .zip(&self.counts)
                .filter(|(_, c)| **c > 0)
                .map(|(e, &c)| [(e[0], 0.0), (e[1], count_to_f64(c))])
        };
        chart.draw_series(bars().map(|r| Rectangle::new(r, color.mix(BAR_ALPHA).filled())))?;
        chart.draw_series(bars().map(|r| Rectangle::new(r, color.stroke_width(1))))?;

        chart.draw_series(DashedLineSeries::new(
            [(self.median, 0.0), (self.median, corner.1)],
            5,
            4,
            OK_BLACK.stroke_width(2),
        ))?;

        let label = format!(
            "n = {}\nmedian κ = {}",
            self.total(),
            fmt_kappa(self.median)
        );
        let font = ("sans-serif", 12)
            .into_font()
            .color(&OK_BLACK)
            .pos(Pos::new(HPos::Right, VPos::Top));
        // Anchored at the data corner and offset in pixels, so the label keeps
        // its inset whatever the axis scale.
        chart.draw_series(label.lines().enumerate().map(|(i, line)| {
            let dy = 4 + 15 * i32::try_from(i).unwrap_or(0);
            EmptyElement::at(corner) + Text::new(line.to_string(), (-6, dy), font.clone())
        }))?;
        Ok(())
    }
}

/// κ to three significant figures, in scientific notation below 0.01 so the
/// collapse-floor median reads as `2.0e-7` rather than `0.000`.
fn fmt_kappa(k: f64) -> String {
    if k != 0.0 && k.abs() < 0.01 {
        format!("{k:.1e}")
    } else {
        format!("{k:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;

    /// A front point at `|K| = k` with `r_gyration` as given (a JSON value, so
    /// `null` can be written), through the deserialiser as a results line is.
    fn trial(k: f64, r_gyration: &str) -> TrialRecord {
        serde_json::from_str(&format!(
            "{{\"curvature_magnitude\": {k}, \"r_gyration\": {r_gyration}}}"
        ))
        .expect("a valid trial line")
    }

    fn cell(setting: &str, dataset: &str, geometry: &str) -> Cell {
        Cell::new(setting, dataset, 5000, geometry)
    }

    #[test]
    fn only_all_off_curved_points_at_this_n_are_counted() {
        let mut fronts = CellMap::new();
        fronts.insert(
            cell("all_off", "tree", "hyperbolic"),
            vec![trial(0.1, "2.0")],
        );
        fronts.insert(
            cell("all_off", "grid", "hyperbolic"),
            vec![trial(1.0, "3.0")],
        );
        fronts.insert(
            cell("all_free", "tree", "hyperbolic"),
            vec![trial(0.1, "1.0")],
        );
        fronts.insert(
            cell("all_off", "tree", "euclidean"),
            vec![trial(0.0, "1.0")],
        );
        fronts.insert(
            Cell::new("all_off", "tree", 1000, "hyperbolic"),
            vec![trial(0.1, "1.0")],
        );
        let mut ks = front_kappas(&fronts, 5000, "hyperbolic");
        ks.sort_by(f64::total_cmp);
        assert_eq!(ks.len(), 2, "pooled over datasets, nothing else");
        assert!((ks[0] - 0.4).abs() < 1e-12);
        assert!((ks[1] - 9.0).abs() < 1e-12);
        assert!(front_kappas(&fronts, 5000, "euclidean").is_empty());
    }

    #[test]
    fn a_point_without_a_usable_kappa_is_not_counted() {
        let mut fronts = CellMap::new();
        fronts.insert(
            cell("all_off", "tree", "spherical"),
            vec![trial(0.1, "null"), trial(0.1, "0.0"), trial(0.1, "2.0")],
        );
        assert_eq!(front_kappas(&fronts, 5000, "spherical"), vec![0.4]);
    }

    #[test]
    fn every_value_lands_in_a_bin_including_the_maximum() {
        let values = [1e-7, 2e-7, 1e-3, 0.5, 1.0, 40.0];
        for scale in [BinScale::Log, BinScale::Linear] {
            let edges = scale.edges(&values, N_BINS).unwrap();
            let counts = bin_counts(&edges, scale, &values);
            assert_eq!(counts.len(), N_BINS);
            assert_eq!(counts.iter().sum::<usize>(), values.len(), "{scale:?}");
            assert!(counts[N_BINS - 1] >= 1, "the maximum is in the last bin");
        }
    }

    #[test]
    fn a_wide_geometry_gets_a_log_and_a_linear_panel() {
        let mut fronts = CellMap::new();
        fronts.insert(
            cell("all_off", "tree", "hyperbolic"),
            vec![trial(1e-6, "0.5"), trial(0.1, "2.0"), trial(5.0, "2.5")],
        );
        let panels = KappaHistogram::panels(&fronts, 5000);
        let names: Vec<String> = panels.iter().map(Figure::name).collect();
        assert_eq!(
            names,
            [
                "exp1_kappa_hist_hyperbolic_N5000",
                "exp1_kappa_hist_hyperbolic_linear_N5000"
            ]
        );
        assert!(panels.iter().all(|p| p.total() == 3));
        assert!(KappaHistogram::panels(&fronts, 1000).is_empty());
    }
}
