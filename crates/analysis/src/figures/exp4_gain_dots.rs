//! Experiment 4 (`ablation-results`) — which loss setting improves the front,
//! and for which dataset and geometry?
//!
//! [`GainDots`] is the figure this module draws: **one row per dataset, the R2
//! gain of a loss setting over `all_off` along x, one panel per embedding
//! geometry, one marker per setting on its own sub-row.** The stacked fronts
//! show the settings as curves that mostly coincide, and the R2 bar charts
//! show one (dataset, geometry) at a time; neither answers the chapter's
//! question — *does a setting help, and does that depend on the data or on
//! the geometry?* — without the reader carrying numbers between panels. Here
//! the four settings of one cell sit stacked a few pixels apart on one row,
//! so "which one moved the front" is the marker furthest from the zero rule,
//! and the same row in the neighbouring panel is the same dataset under
//! another curvature.
//!
//! ### The numbers are the table's
//!
//! The rows of `results/r2_delta_<space>.jsonl` under the `all` region — the
//! same `delta_r2` the Typst table prints — at one N, times [`dot_panels::SCALE`]. `ΔR2 =
//! R2(all_off) − R2(setting)`, so a marker **right** of zero is a setting
//! that served the uniform preference better than the baseline; R2 is a cost.
//! `all_off` is the zero rule rather than a marker, and `rms_anchored` is not
//! drawn ([`dot_panels::EXCLUDED`]) for the reason `exp4::R2Bars` gives: it fixes the
//! curvature gauge for Experiment 3 rather than ablating a loss term.
//!
//! ### The axis is two log axes back to back
//!
//! Most gains are within a unit or two of zero; a handful (`hyperbolic_shells`
//! under every geometry, the norm loss on the Euclidean real datasets) are
//! ten to forty. On a linear axis the majority collapses onto the zero rule
//! and the figure becomes a picture of one dataset; a plain log axis has no
//! zero and no sign, and both are what the figure reads. [`MirroredLogAxis`]
//! is an ordinary `log10` axis of the gain's magnitude, from [`RESOLUTION`]
//! outward, drawn once to the right of zero for improvements and once,
//! mirrored, to the left for regressions, with a narrow gutter between the
//! two `1` ticks that holds the zero rule. Every tick is a value on a log
//! axis the reader already knows how to read; nothing else is scaled. The
//! three panels share the axis so a marker's position is comparable across
//! geometries.
//!
//! ### The linear twin
//!
//! The figure is drawn twice, log and linear, as two files — the second
//! suffixed `_linear`, the rule the Exp 2 panels follow. The linear one
//! ([`Scale::Linear`]) is an ordinary axis over `±max|ΔR2|`, with nothing
//! to explain and nothing on the rule but the exact zeros; it shows how far
//! the large gains stand from everything else, which the log axis flattens
//! by design, at the price of the small gains sitting a few pixels from
//! the rule. Same rows, marks and stems; only the axis differs.
//!
//! ### The gutter
//!
//! A gain under [`RESOLUTION`] in magnitude — one unit of the ×1000 scale —
//! has no position on either log axis and is **drawn on the zero rule**, in
//! the gutter. That is a statement, not a loss: the Typst table prints the
//! real datasets' levels (100–200) to zero decimals, so such a gain is one
//! the table cannot show and the stacked fronts draw as coincident curves.
//! The gutter is not a scale, and it is not a significance test — the
//! cross-dataset Wilcoxon of `r2 aggregate` is that.
//!
//! ### Rows, sub-rows, marks
//!
//! Datasets synthetic first, in the order the results chapter introduces
//! them, then the real ones, as the region-gain heatmap orders them; a
//! dataset absent from every geometry at this N is not a row. Inside a row
//! the settings are stacked in [`dot_panels::SETTINGS`] order, the three single terms
//! above `all_free`, so whether the combination matches its best component
//! or exceeds it is read down the sub-rows. Each marker has its setting's
//! colour (`setting_color`) and its own shape, outlined in black, so the
//! four stay apart in greyscale; a stem from the zero rule carries the sign
//! out to the marker. A setting the sweep did not run for
//! that geometry — `norm_only` on the sphere — or whose weight is inert
//! there — `centering_only` off the hyperboloid, `all_free` on the sphere
//! (`cell::setting_applies`) — is written `n/a` in grey on its sub-row, so
//! the absence is not read as a zero. The layout is [`dot_panels`]'s,
//! shared with the ε figure.

use plotters::coord::ranged1d::{DefaultFormatting, KeyPointHint, Ranged};
use plotters::coord::types::RangedCoordf64;
use plotters::coord::Shift;
use plotters::prelude::*;

use super::dot_panels::{
    canvas, collect_rows, draw_marker, draw_na, draw_panels, draw_settings_legend, Row, SCALE,
    SETTINGS,
};
use super::{setting_color, Figure, LinearTicks, Res};
use crate::aggregate::DeltaRow;
use crate::r2::REGION_ALL;

/// Where each log axis starts, in scaled units: a gain under this in
/// magnitude is drawn on the zero rule.
const RESOLUTION: f64 = 1.0;

/// Half-width of the gutter between the two log axes, in decades. Enough
/// for the two `1` tick labels to clear each other and the zero rule to
/// read as a rule rather than a tick.
const GUTTER_DECADES: f64 = 0.2;

/// Which x axis a rendering carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// [`MirroredLogAxis`].
    Log,
    /// A plain linear axis over `±max|ΔR2|`.
    Linear,
}

impl Scale {
    pub const ALL: [Self; 2] = [Self::Log, Self::Linear];
}

/// The x axis of one rendering: either scale behind one `Ranged`.
#[derive(Debug, Clone)]
pub enum GainAxis {
    Log(MirroredLogAxis),
    Linear(LinearTicks),
}

impl GainAxis {
    /// The axis of *scale* covering `±max_abs`.
    #[must_use]
    pub fn new(scale: Scale, max_abs: f64) -> Self {
        match scale {
            Scale::Log => Self::Log(MirroredLogAxis::symmetric(max_abs)),
            Scale::Linear => {
                // Padded so the outermost marker clears the frame; at least
                // a unit wide so an all-zero figure still has an axis.
                let hi = max_abs.max(RESOLUTION) * 1.08;
                Self::Linear(LinearTicks::new((-hi, hi), 5))
            }
        }
    }

    /// The tick's label.
    #[must_use]
    pub fn label(&self, v: &f64) -> String {
        match self {
            Self::Log(_) => MirroredLogAxis::label(v),
            Self::Linear(t) => t.label(v),
        }
    }
}

impl Ranged for GainAxis {
    type FormatOption = DefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        match self {
            Self::Log(a) => a.map(value, limit),
            Self::Linear(a) => a.map(value, limit),
        }
    }

    fn key_points<Hint: KeyPointHint>(&self, hint: Hint) -> Vec<f64> {
        match self {
            Self::Log(a) => a.key_points(hint),
            Self::Linear(a) => a.key_points(hint),
        }
    }

    fn range(&self) -> std::ops::Range<f64> {
        match self {
            Self::Log(a) => a.range(),
            Self::Linear(a) => a.range(),
        }
    }
}

/// The per-setting R2 gain dot plot at one N.
pub struct GainDots {
    n: usize,
    scale: Scale,
    rows: Vec<Row<f64>>,
    axis: GainAxis,
}

impl GainDots {
    /// The figure over the stage-2 rows at N on *scale*. Rows are the
    /// datasets with at least one drawn gain, in chapter order.
    #[must_use]
    pub fn new(rows: &[DeltaRow], n: usize, scale: Scale) -> Self {
        let out = collect_rows(|dataset, geometry, setting| {
            rows.iter()
                .find(|r| {
                    r.n == n
                        && r.region == REGION_ALL
                        && r.dataset == dataset
                        && r.geometry == geometry
                        && r.setting == setting
                })
                .and_then(|r| r.delta_r2)
                .map(|d| d * SCALE)
                .filter(|d| d.is_finite())
        });
        let max_abs = out
            .iter()
            .flat_map(Row::drawn)
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        Self {
            n,
            scale,
            rows: out,
            axis: GainAxis::new(scale, max_abs),
        }
    }

    /// True when there is at least one row to draw.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.rows.is_empty()
    }

    /// The rows, top to bottom.
    #[must_use]
    pub fn rows(&self) -> &[Row<f64>] {
        &self.rows
    }

    /// The shared x axis.
    #[must_use]
    pub fn axis(&self) -> &GainAxis {
        &self.axis
    }
}

/// Two `log10` axes of the gain's magnitude back to back — one per sign —
/// from [`RESOLUTION`] outward, separated by a gutter of
/// 2·[`GUTTER_DECADES`] that holds the zero rule ([`MirroredLogAxis::t`]).
/// Ticks at `±3·10^k` and `±10^k` inside the range, labelled by value.
#[derive(Debug, Clone)]
pub struct MirroredLogAxis {
    hi: f64,
    ticks: Vec<f64>,
}

impl MirroredLogAxis {
    /// The transform, in decades from the zero rule:
    /// `±(GUTTER_DECADES + log10(|v|/RESOLUTION))` on the axes, `0` for a
    /// magnitude under [`RESOLUTION`], which has no place on either.
    fn t(v: f64) -> f64 {
        let u = v / RESOLUTION;
        if u.abs() < 1.0 {
            0.0
        } else {
            u.signum() * (GUTTER_DECADES + u.abs().log10())
        }
    }

    /// The axes covering `±max_abs`, padded by a tenth of a decade so the
    /// outermost marker clears the frame, and at least one decade long so
    /// the axis is an axis when every gain is small.
    #[must_use]
    pub fn symmetric(max_abs: f64) -> Self {
        let hi = max_abs.max(10.0 * RESOLUTION) * 10f64.powf(0.1);
        let mut ticks = Vec::new();
        let mut decade = RESOLUTION;
        while decade <= hi {
            for m in [1.0, 3.0] {
                let t = m * decade;
                if t <= hi {
                    ticks.push(t);
                    ticks.push(-t);
                }
            }
            decade *= 10.0;
        }
        ticks.sort_by(f64::total_cmp);
        Self { hi, ticks }
    }

    /// The tick's label: the value, as an integer.
    #[must_use]
    pub fn label(v: &f64) -> String {
        format!("{v:.0}")
    }

    /// The end of the axis; the start is its negative.
    #[must_use]
    pub fn hi(&self) -> f64 {
        self.hi
    }

    /// The tick positions, in axis order.
    #[must_use]
    pub fn ticks(&self) -> &[f64] {
        &self.ticks
    }
}

impl Ranged for MirroredLogAxis {
    type FormatOption = DefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        let t = Self::t(self.hi);
        RangedCoordf64::from(-t..t).map(&Self::t(*value), limit)
    }

    fn key_points<Hint: KeyPointHint>(&self, _hint: Hint) -> Vec<f64> {
        self.ticks.clone()
    }

    fn range(&self) -> std::ops::Range<f64> {
        -self.hi..self.hi
    }
}

impl Figure for GainDots {
    fn name(&self) -> String {
        let suffix = match self.scale {
            Scale::Log => "",
            Scale::Linear => "_linear",
        };
        format!("exp4_gain_dots{suffix}_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        canvas(self.rows.len(), 1)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        draw_panels(
            root,
            &self.rows,
            1,
            &self.axis,
            &|v| self.axis.label(v),
            Some(0.0),
            "\u{394}R2 \u{d7} 1000",
            draw_settings_legend,
            |col, chart, area, origin, row| {
                // A stem from the zero rule in data coordinates, the marker
                // by pixel on the panel area.
                for ((setting, shape, off), value) in SETTINGS.iter().zip(row.panel(col)) {
                    let y = row.centre() + off;
                    let Some(v) = value else {
                        let (px, py) = chart.backend_coord(&(0.0, y));
                        draw_na(area, (px - origin.0, py - origin.1))?;
                        continue;
                    };
                    let color = setting_color(setting);
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(0.0, y), (*v, y)],
                        color.mix(0.55).stroke_width(2),
                    )))?;
                    let (px, py) = chart.backend_coord(&(*v, y));
                    draw_marker(area, (px - origin.0, py - origin.1), *shape, color, true)?;
                }
                Ok(())
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fitting_core::cast::to_i32;

    fn delta(dataset: &str, geometry: &str, setting: &str, region: &str, d: f64) -> DeltaRow {
        DeltaRow {
            space: "obj5".into(),
            n: 5000,
            geometry: geometry.into(),
            setting: setting.into(),
            dataset: dataset.into(),
            region: region.into(),
            r2: 0.1 - d,
            r2_baseline: Some(0.1),
            delta_r2: Some(d),
        }
    }

    #[test]
    fn rows_follow_chapter_order_and_drop_the_baseline_and_rms() {
        let rows = vec![
            delta("mnist", "euclidean", "norm_only", "all", 0.02),
            delta("mnist", "euclidean", "all_off", "all", 0.0),
            delta("grid", "hyperbolic", "rms_anchored", "all", 0.01),
            delta("grid", "hyperbolic", "all_free", "all", 0.0025),
            // Another region and another N are not this figure's.
            delta("tree", "spherical", "global_only", "trustworthiness", 0.5),
            DeltaRow {
                n: 1000,
                ..delta("tree", "spherical", "global_only", "all", 0.5)
            },
        ];
        let fig = GainDots::new(&rows, 5000, Scale::Log);
        let datasets: Vec<&str> = fig.rows().iter().map(Row::dataset).collect();
        assert_eq!(
            datasets,
            ["grid", "mnist"],
            "synthetic first, absent skipped"
        );
        assert_eq!(fig.rows()[0].value("hyperbolic", "all_free"), Some(2.5));
        assert_eq!(fig.rows()[1].value("euclidean", "norm_only"), Some(20.0));
        // Neither the baseline nor the gauge setting is a marker.
        assert_eq!(fig.rows()[1].value("euclidean", "all_off"), None);
        assert_eq!(fig.rows()[0].value("hyperbolic", "rms_anchored"), None);
    }

    #[test]
    fn missing_setting_is_none_not_zero() {
        let rows = vec![delta("sphere", "spherical", "global_only", "all", 0.001)];
        let fig = GainDots::new(&rows, 5000, Scale::Linear);
        assert_eq!(fig.rows()[0].value("spherical", "norm_only"), None);
        assert_eq!(fig.rows()[0].value("spherical", "global_only"), Some(1.0));
        assert_eq!(fig.rows()[0].value("euclidean", "global_only"), None);
    }

    #[test]
    fn inert_setting_is_none_even_when_the_table_has_a_row() {
        let rows = vec![
            delta("sphere", "spherical", "all_free", "all", 0.001),
            delta("sphere", "euclidean", "centering_only", "all", 0.001),
            delta("sphere", "hyperbolic", "centering_only", "all", 0.001),
        ];
        let fig = GainDots::new(&rows, 5000, Scale::Linear);
        assert_eq!(fig.rows()[0].value("spherical", "all_free"), None);
        assert_eq!(fig.rows()[0].value("euclidean", "centering_only"), None);
        assert_eq!(
            fig.rows()[0].value("hyperbolic", "centering_only"),
            Some(1.0)
        );
    }

    #[test]
    fn mirrored_log_axis_is_log_on_each_side_with_a_gutter_at_zero() {
        let axis = MirroredLogAxis::symmetric(37.0);
        assert!(axis.hi() > 37.0);
        let px = |v: f64| axis.map(&v, (0, 1000));
        assert_eq!(px(0.0), 500);
        // Plotters rounds to a pixel, so mirror images may differ by one.
        assert!((px(-20.0) - (1000 - px(20.0))).abs() <= 1);
        assert!(px(1.0) < px(3.0) && px(3.0) < px(10.0) && px(10.0) < px(30.0));
        // Decades are evenly spaced: an ordinary log axis on each side.
        let decade = px(10.0) - px(1.0);
        assert!((px(100.0) - px(10.0) - decade).abs() <= 1);
        // The gutter: `GUTTER_DECADES` from the rule to the `1` tick, and a
        // gain under one unit sits on the rule.
        let gutter = to_i32(GUTTER_DECADES * f64::from(decade));
        assert!((px(1.0) - px(0.0) - gutter).abs() <= 1);
        assert_eq!(px(0.5), px(0.0));
        assert_eq!(px(-0.99), px(0.0));
        assert_eq!(
            axis.ticks(),
            &[-30.0, -10.0, -3.0, -1.0, 1.0, 3.0, 10.0, 30.0],
            "each axis starts at its own `1`; zero is the rule, not a tick"
        );
    }

    #[test]
    fn the_linear_twin_is_linear_symmetric_and_named_apart() {
        let rows = vec![delta("grid", "euclidean", "all_free", "all", 0.02)];
        let log = GainDots::new(&rows, 5000, Scale::Log);
        let lin = GainDots::new(&rows, 5000, Scale::Linear);
        assert_eq!(log.name(), "exp4_gain_dots_N5000");
        assert_eq!(lin.name(), "exp4_gain_dots_linear_N5000");
        let axis = lin.axis();
        let px = |v: f64| axis.map(&v, (0, 1000));
        assert_eq!(px(0.0), 500);
        assert!((px(10.0) - px(5.0)) - (px(5.0) - px(0.0)) <= 1);
        assert!(axis.range().end > 20.0 && axis.range().start < -20.0);
        assert_eq!(axis.label(&20.0), "20");
    }

    #[test]
    fn a_flat_figure_still_has_a_decade_of_axis() {
        let axis = MirroredLogAxis::symmetric(0.0);
        assert!(axis.hi() > 10.0 * RESOLUTION);
        assert_eq!(axis.ticks(), &[-10.0, -3.0, -1.0, 1.0, 3.0, 10.0]);
    }
}
