//! Experiment 4 (`ablation-results`) — did the auxiliary loss terms buy
//! global structure with local structure?
//!
//! [`TradeoffScatter`] is the figure this module draws: **one panel per
//! embedding geometry, the R2 gain over `all_off` under the local-structure
//! preference along x and under the global-distance preference along y, one
//! marker per (dataset, setting).** The gain dot plot says whether a setting
//! helped under the *uniform* preference; a setting that trades
//! trustworthiness for stress can come out flat there, and that trade is the
//! hypothesis the chapter has to test. Here the two families sit on
//! orthogonal axes, so the trade is a **quadrant**: upper-right is a setting
//! that improved both families, upper-left bought global structure at a local
//! cost — the suspected trade — lower-right the reverse, lower-left harmed
//! both. The anti-diagonal `y = −x` separates a net gain from a net trade.
//!
//! ### The numbers are the table's
//!
//! The rows of `results/r2_delta_<space>.jsonl` under the two **family
//! regions** ([`LOCAL_REGION`], [`GLOBAL_REGION`]) at one N, times
//! [`dot_panels::SCALE`]. A family region is the slice of the weight simplex
//! putting at least half its mass on that family (`r2::build_regions`), so
//! the `structure` gain is how much better the front serves a reader who
//! weights trustworthiness and continuity, and the `distance` gain the same
//! for stress and Shepard goodness. `ΔR2 = R2(all_off) − R2(setting)`, so a
//! marker right of the vertical rule and above the horizontal one is better
//! than the baseline on that family; R2 is a cost. A (dataset, geometry,
//! setting) is a marker only when *both* rows exist — a setting the sweep
//! did not run for a geometry (`norm_only` on the sphere) is simply absent,
//! which the caption states rather than the panel.
//!
//! ### The axes are the gain dot plot's
//!
//! Both axes are one [`GainAxis`] — the mirrored log axis of
//! `exp4_gain_dots`, or its linear twin under [`Scale::Linear`] — covering
//! `±max|ΔR2|` over both families and all three panels, so a marker's
//! position is comparable across panels and the anti-diagonal is the same
//! line in every one. The log axis is what keeps the figure readable: the
//! median gain is a unit or two, the largest a hundred, and a linear axis
//! piles everything but `hyperbolic_shells` onto the origin. Under the log
//! axis a gain under `RESOLUTION` on one family sits on that family's rule,
//! the gutter rule of the gain dot plot, and a marker in the gutter on both
//! is a setting that moved neither family.
//!
//! ### Marks and labels
//!
//! A marker carries its setting's colour and shape, as every Exp 4 figure
//! does ([`dot_panels::SETTINGS`]). The dataset is written beside the marker,
//! on the side away from the origin so the labels grow outward rather than
//! into the dense centre, and only for markers at least [`LABEL_THRESHOLD`]
//! from the origin on some axis: the cloud of markers inside that radius is
//! the set of settings that moved nothing worth naming, and naming each of
//! them would bury the ones that did. A dataset whose markers coincide —
//! `norm_only` and `all_free` do on most datasets — is named once for the
//! cluster ([`place_labels`]), a label another marker would sit on flips
//! to the other side, and labels that would overlap are pushed apart with
//! a leader back to the marker. The quadrant readings are the caption's to state, as
//! the reading direction is on every other figure.
//!
//! The figure is drawn twice, log and `_linear`, the rule the other Exp 4
//! dot plots follow.

use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use fitting_core::cast::to_i32;

use super::dot_panels::{
    draw_marker, draw_settings_legend, CAPTION, DESC_STRIP, DOT, EXCLUDED, LEGEND_ROW, MARGIN,
    PANEL_GAP, SCALE, SETTINGS, WIDTH, X_LABEL_AREA,
};
use super::exp1::dataset_label;
use super::exp4_gain_dots::{GainAxis, Scale};
use super::{
    plot_x, setting_color, Figure, Res, GEOMETRIES, OK_BLACK, OK_GREY, REAL_DATASETS,
    SYNTH_DATASETS,
};
use crate::aggregate::DeltaRow;
use crate::style_mesh;

/// The preference region on x: the family of trustworthiness and continuity
/// (`Family::Structure`; the test pins the name).
pub const LOCAL_REGION: &str = "structure";
/// The preference region on y: the family of stress and Shepard goodness
/// (`Family::Distance`).
pub const GLOBAL_REGION: &str = "distance";

/// A marker is labelled with its dataset when its gain on either family is
/// at least this, in scaled units: three units of the ×1000 scale, a gain
/// the Typst table prints as a whole digit.
const LABEL_THRESHOLD: f64 = 3.0;

/// Room left of the first panel: the y tick labels (`-100`) and the y axis
/// description, rotated. Wider than the tick labels alone need, because the
/// description has to fit beside them.
const Y_LABEL_AREA: u32 = 58;

/// One marker: the gains of *setting* on *dataset* under `GEOMETRIES[col]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub dataset: String,
    /// Panel index into [`GEOMETRIES`].
    pub col: usize,
    /// Index into [`SETTINGS`].
    pub setting: usize,
    /// Gain under [`LOCAL_REGION`], scaled.
    pub local: f64,
    /// Gain under [`GLOBAL_REGION`], scaled.
    pub global: f64,
}

/// The local-versus-global R2 gain scatter at one N.
pub struct TradeoffScatter {
    n: usize,
    scale: Scale,
    points: Vec<Point>,
    axis: GainAxis,
}

impl TradeoffScatter {
    /// The figure over the stage-2 rows at N on *scale*. A point is a
    /// (dataset, geometry, setting) with a finite gain under both family
    /// regions, datasets in chapter order.
    #[must_use]
    pub fn new(rows: &[DeltaRow], n: usize, scale: Scale) -> Self {
        let gain = |dataset: &str, geometry: &str, setting: &str, region: &str| {
            rows.iter()
                .find(|r| {
                    r.n == n
                        && r.region == region
                        && r.dataset == dataset
                        && r.geometry == geometry
                        && r.setting == setting
                })
                .and_then(|r| r.delta_r2)
                .map(|d| d * SCALE)
                .filter(|d| d.is_finite())
        };
        let mut points = Vec::new();
        for dataset in SYNTH_DATASETS.iter().chain(REAL_DATASETS.iter()) {
            for (col, geometry) in GEOMETRIES.iter().enumerate() {
                for (s, (setting, _, _)) in SETTINGS.iter().enumerate() {
                    if EXCLUDED.contains(setting) {
                        continue;
                    }
                    let (Some(local), Some(global)) = (
                        gain(dataset, geometry, setting, LOCAL_REGION),
                        gain(dataset, geometry, setting, GLOBAL_REGION),
                    ) else {
                        continue;
                    };
                    points.push(Point {
                        dataset: (*dataset).to_string(),
                        col,
                        setting: s,
                        local,
                        global,
                    });
                }
            }
        }
        let max_abs = points
            .iter()
            .flat_map(|p| [p.local, p.global])
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        Self {
            n,
            scale,
            points,
            axis: GainAxis::new(scale, max_abs),
        }
    }

    /// True when there is at least one marker to draw.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.points.is_empty()
    }

    /// The markers, in chapter, geometry, setting order.
    #[must_use]
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// The `(local, global)` gains of one marker, if drawn.
    #[must_use]
    pub fn point(&self, dataset: &str, geometry: &str, setting: &str) -> Option<(f64, f64)> {
        let col = GEOMETRIES.iter().position(|g| *g == geometry)?;
        let s = SETTINGS.iter().position(|(x, _, _)| *x == setting)?;
        self.points
            .iter()
            .find(|p| p.dataset == dataset && p.col == col && p.setting == s)
            .map(|p| (p.local, p.global))
    }

    /// The shared axis of both x and y.
    #[must_use]
    pub fn axis(&self) -> &GainAxis {
        &self.axis
    }

    /// Side length of one panel's plotting area: the three areas share the
    /// width left of the label column, and each is square so the diagonal
    /// is a diagonal.
    fn plot_side() -> u32 {
        let inner = WIDTH.saturating_sub(2 * MARGIN + Y_LABEL_AREA);
        let each = inner / u32::try_from(GEOMETRIES.len()).unwrap_or(1);
        each.saturating_sub(2 * PANEL_GAP)
    }

    /// The labels and markers of panel *col*, by pixel on *area* (whose
    /// backend origin is *origin*): labels first, so no label is drawn over
    /// a marker.
    fn draw_marks<DB: DrawingBackend>(
        &self,
        col: usize,
        chart: &ChartContext<'_, DB, Cartesian2d<GainAxis, GainAxis>>,
        area: &DrawingArea<DB, Shift>,
        origin: (i32, i32),
    ) -> Res
    where
        DB::ErrorType: 'static,
    {
        let label_font = ("sans-serif", 10).into_font().color(&RGBColor(40, 40, 40));
        // Labels first, so no label is drawn over a marker.
        let anchors: Vec<Anchor> = self
            .points
            .iter()
            .filter(|p| p.col == col)
            .filter(|p| p.local.abs() >= LABEL_THRESHOLD || p.global.abs() >= LABEL_THRESHOLD)
            .map(|p| {
                let (px, py) = chart.backend_coord(&(p.local, p.global));
                Anchor {
                    dataset: p.dataset.as_str(),
                    px: px - origin.0,
                    py: py - origin.1,
                    right: p.local >= 0.0,
                }
            })
            .collect();
        for label in place_labels(&anchors) {
            if let Some(line) = label.leader() {
                area.draw(&PathElement::new(line.to_vec(), OK_GREY.stroke_width(1)))?;
            }
            let (at, hpos) = label.text_at();
            area.draw(&Text::new(
                dataset_label(label.anchor.dataset).to_string(),
                at,
                label_font.clone().pos(Pos::new(hpos, VPos::Center)),
            ))?;
        }
        for p in self.points.iter().filter(|p| p.col == col) {
            let (setting, shape, _) = SETTINGS[p.setting];
            let (px, py) = chart.backend_coord(&(p.local, p.global));
            draw_marker(
                area,
                (px - origin.0, py - origin.1),
                shape,
                setting_color(setting),
                true,
            )?;
        }
        Ok(())
    }
}

impl Figure for TradeoffScatter {
    fn name(&self) -> String {
        let suffix = match self.scale {
            Scale::Log => "",
            Scale::Linear => "_linear",
        };
        format!("exp4_tradeoff{suffix}_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (
            WIDTH,
            LEGEND_ROW + CAPTION + Self::plot_side() + X_LABEL_AREA + DESC_STRIP + 2 * MARGIN,
        )
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let (legend, rest) = root.split_vertically(LEGEND_ROW);
        draw_settings_legend(&legend)?;
        let (_, rest_h) = rest.dim_in_pixel();
        let (panels, desc) = rest.split_vertically(rest_h.saturating_sub(DESC_STRIP));

        let (width, _) = panels.dim_in_pixel();
        let inner = width.saturating_sub(2 * MARGIN + Y_LABEL_AREA);
        let each = inner / u32::try_from(GEOMETRIES.len()).unwrap_or(1);
        let first = MARGIN + Y_LABEL_AREA + each;
        let areas = panels.split_by_breakpoints([first, first + each], [] as [u32; 0]);

        let (lo, hi) = (self.axis.range().start, self.axis.range().end);
        let rule = RGBColor(90, 90, 90);

        for (col, (geometry, area)) in GEOMETRIES.iter().zip(&areas).enumerate() {
            let mut builder = ChartBuilder::on(area);
            builder
                .margin_top(CAPTION)
                .margin_bottom(MARGIN)
                .margin_right(PANEL_GAP)
                .x_label_area_size(X_LABEL_AREA);
            if col == 0 {
                builder.margin_left(MARGIN).y_label_area_size(Y_LABEL_AREA);
            } else {
                builder.margin_left(PANEL_GAP).y_label_area_size(0);
            }
            let mut chart = builder.build_cartesian_2d(self.axis.clone(), self.axis.clone())?;
            let tick = |v: &f64| self.axis.label(v);
            let mut mesh = chart.configure_mesh();
            style_mesh!(mesh)
                .x_label_formatter(&tick)
                .y_label_formatter(&tick);
            if col == 0 {
                mesh.y_desc(format!("\u{394}R2 \u{d7} 1000, W_{GLOBAL_REGION}"));
            } else {
                mesh.disable_y_axis();
            }
            mesh.draw()?;

            // The anti-diagonal, then the two zero rules, under the marks.
            // The axis transform is odd, so `(v, −v)` maps onto one straight
            // pixel line on either scale.
            chart.draw_series(DashedLineSeries::new(
                vec![(lo, hi), (hi, lo)],
                6,
                4,
                OK_GREY.mix(0.7).stroke_width(1),
            ))?;
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(0.0, lo), (0.0, hi)],
                rule.stroke_width(1),
            )))?;
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(lo, 0.0), (hi, 0.0)],
                rule.stroke_width(1),
            )))?;

            let origin = (
                area.get_pixel_range().0.start,
                area.get_pixel_range().1.start,
            );
            let plot_px = chart.plotting_area().get_pixel_range();
            area.draw(&Text::new(
                *geometry,
                (
                    plot_x(&plot_px.0, 0.5) - origin.0,
                    to_i32(f64::from(CAPTION) / 2.0),
                ),
                ("sans-serif", 15)
                    .into_font()
                    .style(FontStyle::Bold)
                    .color(&OK_BLACK)
                    .pos(Pos::new(HPos::Center, VPos::Center)),
            ))?;

            self.draw_marks(col, &chart, area, origin)?;
        }

        // The x axis label once, centred under the three panels.
        let (dw, dh) = desc.dim_in_pixel();
        let x = plot_x(
            &(to_i32(f64::from(MARGIN + Y_LABEL_AREA))..to_i32(f64::from(dw - MARGIN))),
            0.5,
        );
        desc.draw(&Text::new(
            format!("\u{394}R2 \u{d7} 1000, W_{LOCAL_REGION}"),
            (x, to_i32(f64::from(dh) / 2.0)),
            ("sans-serif", 13)
                .into_font()
                .color(&RGBColor(30, 30, 30))
                .pos(Pos::new(HPos::Center, VPos::Center)),
        ))?;
        Ok(())
    }
}

/// A labelled marker, by pixel on the panel area.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Anchor<'a> {
    dataset: &'a str,
    px: i32,
    py: i32,
    /// Whether the label goes right of the marker (a non-negative local gain).
    right: bool,
}

/// A placed label: the marker it names, and where the text sits.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Label<'a> {
    anchor: Anchor<'a>,
    /// Text baseline centre, after pushing apart.
    py: i32,
    /// The side the text is on, after flipping away from other markers.
    right: bool,
}

/// Markers of one dataset closer than this, in pixels, share one label.
const CLUSTER_PX: i32 = 14;
/// The line height of a label: two labels on one side closer than this in y
/// are pushed apart.
const LINE_PX: i32 = 11;
/// Two labels on one side whose anchors are closer than this in x can
/// overlap in text; further apart they cannot, whatever their y.
const OVERLAP_PX: i32 = 70;
/// Text advance per character of the 10 px label font, roughly.
const CHAR_PX: i32 = 6;
/// The gap between a marker's edge and its label.
const LABEL_GAP: i32 = DOT + 3;

/// Where each label goes.
///
/// **One per cluster** of a dataset's markers — `norm_only` and `all_free`
/// coincide on most datasets, and two copies of the name over one spot is
/// what made the first render unreadable — at the cluster's outermost
/// marker on the label's side. The label goes on the side away from the
/// origin, unless another marker sits where the text would go and the other
/// side is clear, in which case it **flips**. Then labels on one side are
/// **pushed apart** in y, top down, wherever two would overlap; a pushed
/// label gets a leader line back to its marker ([`leader`]).
fn place_labels<'a>(anchors: &[Anchor<'a>]) -> Vec<Label<'a>> {
    // Single-linkage clusters within a dataset.
    let mut cluster: Vec<usize> = (0..anchors.len()).collect();
    for i in 0..anchors.len() {
        for j in 0..i {
            let (a, b) = (anchors[i], anchors[j]);
            if a.dataset == b.dataset
                && (a.px - b.px).abs() <= CLUSTER_PX
                && (a.py - b.py).abs() <= CLUSTER_PX
            {
                let (ci, cj) = (cluster[i], cluster[j]);
                for c in &mut cluster {
                    if *c == ci {
                        *c = cj;
                    }
                }
            }
        }
    }
    let mut labels: Vec<Label<'a>> = Vec::new();
    for c in 0..anchors.len() {
        let members: Vec<Anchor<'a>> = (0..anchors.len())
            .filter(|&i| cluster[i] == c)
            .map(|i| anchors[i])
            .collect();
        let Some(first) = members.first() else {
            continue;
        };
        let right = first.right;
        // The outermost marker on the label's side, so the label clears
        // every marker of the cluster; the text is centred on the cluster.
        let outer = members
            .iter()
            .copied()
            .max_by_key(|a| if right { a.px } else { -a.px })
            .unwrap_or(*first);
        let py =
            members.iter().map(|a| a.py).sum::<i32>() / i32::try_from(members.len()).unwrap_or(1);
        let anchor = Anchor { py, ..outer };
        let width = CHAR_PX * i32::try_from(dataset_label(anchor.dataset).len()).unwrap_or(0);
        let blocked = |right: bool| {
            let (x0, x1) = if right {
                (anchor.px + LABEL_GAP, anchor.px + LABEL_GAP + width)
            } else {
                (anchor.px - LABEL_GAP - width, anchor.px - LABEL_GAP)
            };
            anchors.iter().any(|o| {
                o.dataset != anchor.dataset
                    && o.px + DOT >= x0
                    && o.px - DOT <= x1
                    && (o.py - anchor.py).abs() <= DOT + LINE_PX / 2
            })
        };
        let right = if blocked(right) && !blocked(!right) {
            !right
        } else {
            right
        };
        labels.push(Label { anchor, py, right });
    }
    // Push apart, top down, per side.
    labels.sort_by_key(|l| (l.right, l.py));
    for i in 1..labels.len() {
        let prev = labels[i - 1];
        let cur = &mut labels[i];
        if cur.right == prev.right
            && (cur.anchor.px - prev.anchor.px).abs() < OVERLAP_PX
            && cur.py - prev.py < LINE_PX
        {
            cur.py = prev.py + LINE_PX;
        }
    }
    labels
}

impl Label<'_> {
    /// The text's anchor pixel and its horizontal alignment.
    fn text_at(&self) -> ((i32, i32), HPos) {
        if self.right {
            ((self.anchor.px + LABEL_GAP, self.py), HPos::Left)
        } else {
            ((self.anchor.px - LABEL_GAP, self.py), HPos::Right)
        }
    }

    /// A leader from the marker's edge to the text's start, when the text
    /// was pushed off the marker's line; `None` when it sits beside it.
    fn leader(&self) -> Option<[(i32, i32); 2]> {
        if (self.py - self.anchor.py).abs() < LINE_PX / 2 {
            return None;
        }
        let side = if self.right { 1 } else { -1 };
        Some([
            (self.anchor.px + side * (DOT + 1), self.anchor.py),
            (self.anchor.px + side * (LABEL_GAP - 1), self.py),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objectives::{families, ObjectiveSpace};
    use fitting_core::metrics::Family;

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
    fn the_regions_are_the_two_families_of_the_projected_space() {
        let names: Vec<&str> = families(ObjectiveSpace::Projected5)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(LOCAL_REGION, Family::Structure.name());
        assert_eq!(GLOBAL_REGION, Family::Distance.name());
        assert!(names.contains(&LOCAL_REGION));
        assert!(names.contains(&GLOBAL_REGION));
    }

    #[test]
    fn a_point_needs_both_families_and_skips_baseline_and_rms() {
        let rows = vec![
            delta("mnist", "euclidean", "norm_only", LOCAL_REGION, -0.002),
            delta("mnist", "euclidean", "norm_only", GLOBAL_REGION, 0.02),
            // Only one family: no marker.
            delta("mnist", "euclidean", "all_free", LOCAL_REGION, 0.01),
            // The baseline and the gauge setting are never markers.
            delta("mnist", "euclidean", "all_off", LOCAL_REGION, 0.0),
            delta("mnist", "euclidean", "all_off", GLOBAL_REGION, 0.0),
            delta("tree", "hyperbolic", "rms_anchored", LOCAL_REGION, 0.01),
            delta("tree", "hyperbolic", "rms_anchored", GLOBAL_REGION, 0.01),
            // Another N is not this figure's.
            DeltaRow {
                n: 1000,
                ..delta("grid", "euclidean", "all_free", LOCAL_REGION, 0.5)
            },
            DeltaRow {
                n: 1000,
                ..delta("grid", "euclidean", "all_free", GLOBAL_REGION, 0.5)
            },
        ];
        let fig = TradeoffScatter::new(&rows, 5000, Scale::Log);
        assert_eq!(fig.points().len(), 1);
        assert_eq!(
            fig.point("mnist", "euclidean", "norm_only"),
            Some((-2.0, 20.0))
        );
        assert_eq!(fig.point("mnist", "euclidean", "all_free"), None);
        assert_eq!(fig.point("mnist", "euclidean", "all_off"), None);
        assert_eq!(fig.point("tree", "hyperbolic", "rms_anchored"), None);
        // The axis covers the larger of the two families.
        assert!(fig.axis().range().end > 20.0 && fig.axis().range().start < -20.0);
    }

    #[test]
    fn points_follow_chapter_order_and_names_carry_the_scale() {
        let mut rows = Vec::new();
        for ds in ["mnist", "grid"] {
            rows.push(delta(ds, "spherical", "all_free", LOCAL_REGION, 0.001));
            rows.push(delta(ds, "spherical", "all_free", GLOBAL_REGION, 0.001));
        }
        let log = TradeoffScatter::new(&rows, 5000, Scale::Log);
        let lin = TradeoffScatter::new(&rows, 5000, Scale::Linear);
        let datasets: Vec<&str> = log.points().iter().map(|p| p.dataset.as_str()).collect();
        assert_eq!(datasets, ["grid", "mnist"], "synthetic first");
        assert_eq!(log.name(), "exp4_tradeoff_N5000");
        assert_eq!(lin.name(), "exp4_tradeoff_linear_N5000");
        assert!(log.has_data());
        assert!(!TradeoffScatter::new(&[], 5000, Scale::Log).has_data());
    }

    #[test]
    fn labels_are_one_per_cluster_flip_off_markers_and_push_apart() {
        let a = |dataset: &'static str, px, py, right| Anchor {
            dataset,
            px,
            py,
            right,
        };
        // Two coincident markers of one dataset: one label, at the outer one.
        let labels = place_labels(&[a("mnist", 100, 50, true), a("mnist", 104, 52, true)]);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].anchor.px, 104);
        assert_eq!(labels[0].py, 51);
        assert!(labels[0].leader().is_none());

        // Another dataset on the same spot is its own label, pushed down a
        // line with a leader back to its marker.
        let labels = place_labels(&[a("mnist", 100, 50, true), a("pbmc", 100, 53, true)]);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].py, 50);
        assert_eq!(labels[1].py, 50 + LINE_PX);
        assert!(labels[1].leader().is_some());

        // A marker where the text would go flips the label to the free side.
        let labels = place_labels(&[a("grid", 100, 50, true), a("pbmc", 130, 50, true)]);
        let grid = labels.iter().find(|l| l.anchor.dataset == "grid").unwrap();
        assert!(!grid.right, "flipped left, away from pbmc");
        let pbmc = labels.iter().find(|l| l.anchor.dataset == "pbmc").unwrap();
        assert!(pbmc.right, "nothing to its right");
        // Far apart in x, nothing moves.
        let labels = place_labels(&[a("grid", 20, 50, true), a("pbmc", 180, 50, true)]);
        assert!(labels.iter().all(|l| l.py == 50 && l.right));
    }

    #[test]
    fn the_canvas_is_one_square_panel_row() {
        let fig = TradeoffScatter::new(&[], 5000, Scale::Log);
        let (w, h) = fig.size();
        assert_eq!(w, WIDTH);
        assert!(h > TradeoffScatter::plot_side());
        assert!(TradeoffScatter::plot_side() > 150, "panels are legible");
    }
}
