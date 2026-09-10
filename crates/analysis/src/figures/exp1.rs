//! Experiment 1 — the matched-minus-mismatched R2 gain, by dataset.
//!
//! The companion to `@tab:geometry-match-r2`, which `scripts/exp1_r2_typst.py`
//! renders from the same JSONL. That table is long-format — one row per
//! (dataset, geometry), 36 of them — and its own docstring concedes where the
//! answer lives: "whether the matched geometry's ΔR2 beats the mismatched
//! one's", a subtraction the reader has to carry out by eye, down a column,
//! across three rows at a time. Here that subtraction *is* the drawn quantity.
//!
//! One group per dataset of known intrinsic geometry, two bars in it:
//!
//! ```text
//! gain(g) = R2(g) − R2(truth)     for each embedding geometry g ≠ truth
//! ```
//!
//! R2 is a cost (distance to the ideal point, `@eq:r2`), so the subtraction
//! runs mismatched-minus-matched and a **positive** bar means the matched
//! geometry's front was the better one — the same reading direction as the
//! table's ΔR2 and `@eq:r2-gain`.
//!
//! **The Euclidean comparison is always in the picture, on both sides of it.**
//! For a curved-truth dataset one of the two bars is `vs euclidean`, and it is
//! numerically the matched row's ΔR2 from the table. For a flat-truth dataset
//! (`grid`, `ball2_euclidean`, `ball9_euclidean`) Euclidean *is* the matched
//! arm, so it is the reference the two curved bars are measured from. Nothing
//! about the flat arm is special-cased; it is simply one of the three.
//!
//! **No error bars.** Each cell of `results/` is a single sweep — the three
//! seeds `--n-seeds 3` asks for are averaged *inside* every trial, not across
//! repeated optimisations — and `discover_cells` rejects two files mapping to
//! one cell. There is no run-to-run spread to draw, and the figure does not
//! manufacture one from the within-run trial scatter, which measures something
//! else.
//!
//! Read back from the stage-2 JSONL rather than recomputed from the sweeps, for
//! the reason `r2_bars` is: the chart and the thesis table must not be able to
//! disagree.

use fitting_core::cast::{count_to_f64, to_i32};
use std::collections::BTreeMap;
use std::path::Path;

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use super::{
    draw_legend, geometry_color, Figure, LegendEntry, ObjectiveSpace, Res, OK_BLACK, SYNTH_DATASETS,
};
use crate::cell::truth_of;
use crate::error::{Error, Result};
use crate::records::load_jsonl;
use crate::style_mesh;

/// Every gain is multiplied by this, exactly as the Typst tables scale R2: the
/// values sit around 1e-3, so the unscaled axis would spend its digits on
/// leading zeros.
const SCALE: f64 = 1000.0;

/// Bar order inside a dataset's group, the truth arm removed.
///
/// Curvature-sign order, so the flat arm sits between the two curved ones —
/// the `GEOMETRY_ORDER` of `scripts/exp1_common.py`, kept identical so a group
/// here reads left-to-right the way the table's arms read top-to-bottom.
const ARM_ORDER: [&str; 3] = ["spherical", "euclidean", "hyperbolic"];

/// Bars per group: the three geometries less the matched one.
const ARMS: usize = ARM_ORDER.len() - 1;

/// Height reserved under the plot for the two-line group labels.
const LABEL_AREA: i32 = 42;

/// Fraction of a group's width left empty at each end, so neighbouring groups'
/// bars do not touch.
const GROUP_PAD: f64 = 0.10;
/// Fraction of a bar's slot left empty, as the gap between bars of one group.
const BAR_GAP: f64 = 0.15;

/// Fraction of the drawn span left clear past each extreme, for the value
/// labels. Both ends, because the bars are signed.
const HEAD_ROOM: f64 = 0.18;

/// Shorter dataset labels, so twelve of them fit under one axis.
///
/// The same abbreviations `DATASET_LABEL` in `scripts/exp1_common.py` uses, so
/// a group here is findable as a row there. A dataset with no entry falls
/// through to its own name, which is only ever a little wider.
fn dataset_label(dataset: &str) -> &str {
    match dataset {
        "hyperbolic_shells" => "hyp. shells",
        "tree" => "tree (layout)",
        "tree_graph" => "tree (metric)",
        "ball2_euclidean" => "ball E2",
        "ball2_spherical" => "ball S2",
        "ball2_hyperbolic" => "ball H2",
        "ball9_euclidean" => "ball E9",
        "ball9_spherical" => "ball S9",
        "ball9_hyperbolic" => "ball H9",
        other => other,
    }
}

/// The three-letter arm abbreviations of the table.
fn geometry_label(geometry: &str) -> &str {
    match geometry {
        "euclidean" => "euc",
        "hyperbolic" => "hyp",
        "spherical" => "sph",
        other => other,
    }
}

// ─── Input ───────────────────────────────────────────────────────────────────

/// One line of `results/exp1_geometry_match.jsonl`, reduced to what a gain
/// needs.
///
/// Deliberately *not* the `Exp1Row` of `bin/exp1.rs`: that one carries the
/// whole Wilson block, and its `truth: &'static str` cannot be deserialized.
/// Same treatment as [`super::KappaData`], which reads `kappa_data.jsonl` the
/// same way. `truth` is not read from the file at all — [`truth_of`] is the
/// authority on it.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Exp1Row {
    pub dataset: String,
    pub n: usize,
    pub setting: String,
    /// The objective space the R2 values were computed in. Rows written before
    /// the two spaces were separated are legacy ones.
    #[serde(default = "legacy_tag")]
    pub space: String,
    pub geometry: String,
    /// Preference region name → R2 indicator of that cell's front.
    pub r2: BTreeMap<String, f64>,
}

/// The space an untagged row was written in — see [`Exp1Row::space`].
fn legacy_tag() -> String {
    ObjectiveSpace::Legacy10.tag().to_string()
}

/// Load the rows written by the `exp1` binary.
///
/// An **absent** table is not an error: it is a separate `exp1` run, and this
/// figure is skipped without it exactly as Exp 3 skips its scatter without a
/// `κ_data` export. A table that is there and will not parse still fails.
///
/// # Errors
///
/// Returns `Err` if the file is present but malformed. A missing file returns
/// an empty `Vec`.
pub fn load_rows(path: &Path) -> Result<Vec<Exp1Row>> {
    match load_jsonl(path) {
        Ok(rows) => Ok(rows),
        Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            Ok(Vec::new())
        }
        Err(e) => Err(e),
    }
}

// ─── Assembly ────────────────────────────────────────────────────────────────

/// One dataset's group of bars.
struct Group {
    dataset: String,
    /// The geometry this dataset is built to have — the bars' reference.
    truth: &'static str,
    /// The mismatched geometries, in [`ARM_ORDER`] with `truth` removed. Fixed
    /// length [`ARMS`] so every group's bars sit at the same offsets.
    arms: [&'static str; ARMS],
    /// `R2(arms[i]) − R2(truth)`, or `None` where that cell is absent from the
    /// table — a hole in the group, never a zero-height bar.
    gains: [Option<f64>; ARMS],
}

/// The matched-minus-mismatched gains of every synthetic dataset, at one sample
/// size and under one preference region.
pub struct MatchedGain {
    n: usize,
    region: String,
    /// The loss-weight setting the rows came from, named in the title.
    setting: String,
    /// The objective space the rows were scored in, as `ObjectiveSpace::tag`.
    /// Not drawn — [`MatchedGain::space`] exists so the caller can check it
    /// against the space it will tag the filename with, since nothing on the
    /// image would reveal a mismatch.
    space: String,
    groups: Vec<Group>,
}

impl MatchedGain {
    /// Build the figure from the `exp1` table.
    ///
    /// Rows are filtered to *n* and to a single setting — the lexicographically
    /// smallest present, which is what `prepare()` in
    /// `scripts/exp1_common.py` picks, so the figure and the table select the
    /// same rows out of a concatenated file. A dataset whose *matched* arm is
    /// missing is dropped: there is nothing to reference its bars against.
    #[must_use]
    pub fn new(rows: &[Exp1Row], n: usize, region: &str) -> Self {
        let setting = rows
            .iter()
            .filter(|r| r.n == n)
            .map(|r| r.setting.as_str())
            .min()
            .unwrap_or_default()
            .to_string();
        // One table is written in one space (`bin/exp1.rs` resolves it once per
        // run), so taking the first row's is enough to name it.
        let space = rows.iter().find(|r| r.n == n).map_or_else(
            || ObjectiveSpace::Legacy10.tag().to_string(),
            |r| r.space.clone(),
        );

        // (dataset, geometry) → this region's R2.
        let mut r2: BTreeMap<(&str, &str), f64> = BTreeMap::new();
        for row in rows.iter().filter(|r| r.n == n && r.setting == setting) {
            if let Some(&v) = row.r2.get(region) {
                if v.is_finite() {
                    r2.insert((&row.dataset, &row.geometry), v);
                }
            }
        }

        let groups = SYNTH_DATASETS
            .iter()
            .filter_map(|&dataset| {
                let truth = truth_of(dataset)?;
                let matched = *r2.get(&(dataset, truth))?;
                let arms = mismatched(truth);
                let gains = arms.map(|g| r2.get(&(dataset, g)).map(|v| v - matched));
                // A group with both cells missing is an empty slot pair, which
                // would draw as a labelled gap. Drop it.
                if gains.iter().all(Option::is_none) {
                    return None;
                }
                Some(Group {
                    dataset: dataset.to_string(),
                    truth,
                    arms,
                    gains,
                })
            })
            .collect();

        Self {
            n,
            region: region.to_string(),
            setting,
            space,
            groups,
        }
    }

    /// The objective space tag the rows carry.
    ///
    /// The figure is named for the space [`super::save`] is given, not for this
    /// one; they differ only if a caller points `--exp1` at a table from the
    /// other space, which would mislabel the file. The binary compares them.
    #[must_use]
    pub fn space(&self) -> &str {
        &self.space
    }

    /// True when at least one dataset has a matched arm and something to
    /// compare it against.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.groups.is_empty()
    }

    /// Every geometry that appears as a mismatched arm, in [`ARM_ORDER`] — the
    /// legend.
    fn arms_present(&self) -> Vec<&'static str> {
        ARM_ORDER
            .into_iter()
            .filter(|g| {
                self.groups.iter().any(|group| {
                    group
                        .arms
                        .iter()
                        .zip(&group.gains)
                        .any(|(arm, gain)| arm == g && gain.is_some())
                })
            })
            .collect()
    }

    /// The y range, scaled, always containing zero and with head room at both
    /// ends for the value labels.
    fn y_range(&self) -> (f64, f64) {
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        for v in self
            .groups
            .iter()
            .flat_map(|g| g.gains.iter().flatten())
            .map(|v| v * SCALE)
        {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        // An all-zero (or single-sided, all-tiny) figure still needs a span for
        // the axis to build.
        let span = if hi - lo > 0.0 { hi - lo } else { 1.0 };
        (lo - span * HEAD_ROOM, hi + span * HEAD_ROOM)
    }
}

/// The two geometries that are not *truth*, in [`ARM_ORDER`].
fn mismatched(truth: &str) -> [&'static str; ARMS] {
    let mut out = [""; ARMS];
    let mut i = 0;
    for g in ARM_ORDER {
        if g != truth {
            out[i] = g;
            i += 1;
        }
    }
    // `truth` always comes from `SYNTH_TRUTH`, whose values are exactly the
    // three of `ARM_ORDER`, so `i == ARMS` here. A truth outside that set would
    // leave an empty arm, which `arms_present` and the draw loop both skip.
    out
}

/// A gain at the scale the axis is in: decimals drop as the magnitude grows, so
/// the labels stay three or four significant figures wide, as `r2_bars` does it.
fn fixed(scaled: f64) -> String {
    let decimals = if scaled.abs() < 10.0 {
        2
    } else {
        usize::from(scaled.abs() < 100.0)
    };
    format!("{scaled:+.decimals$}")
}

// ─── Drawing ─────────────────────────────────────────────────────────────────

impl Figure for MatchedGain {
    /// The setting, region and sample size, in the name.
    ///
    /// Nothing identifying is drawn *on* the figure — the caption states it —
    /// so the filename is what distinguishes two renders, and it has to carry
    /// every input that changes the bars. [`super::save`] appends the objective
    /// space, which is the fourth. The region takes its `W` prefix from the
    /// thesis notation, which also keeps `all_off` + `all` from reading as one
    /// token.
    fn name(&self) -> String {
        format!(
            "exp1_matched_gain_{}_W{}_N{}",
            self.setting, self.region, self.n
        )
    }

    fn size(&self) -> (u32, u32) {
        // Sized by group count, as `r2_bars` is: each group carries two bars
        // and a two-line label, and the value labels must not collide.
        (
            220 + 130 * u32::try_from(self.groups.len()).expect("a small number of groups"),
            560,
        )
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let root = root.titled(
            "R2 gain of the matched geometry over each mismatched one",
            ("sans-serif", 18).into_font().color(&OK_BLACK),
        )?;
        let (legend, body) = root.split_vertically(30);

        let entries: Vec<LegendEntry> = self
            .arms_present()
            .into_iter()
            .map(|g| LegendEntry::new(format!("vs {g}"), geometry_color(g)))
            .collect();
        draw_legend(&legend, &entries)?;

        // The group labels are drawn by hand rather than as x tick labels: the
        // x axis is continuous (a group is the interval [i, i+1], so the bars
        // inside it can be placed at any fraction), and plotters picks its own
        // key points on a continuous axis — they would not land on the group
        // centres. Splitting the strip off first keeps the plot's own geometry
        // untouched.
        let (plot_area, label_area) = body.split_vertically(
            i32::try_from(body.dim_in_pixel().1).unwrap_or(i32::MAX) - LABEL_AREA,
        );

        let n_groups = self.groups.len();
        let (y_lo, y_hi) = self.y_range();

        let mut chart = ChartBuilder::on(&plot_area)
            .margin(10)
            .x_label_area_size(0)
            .y_label_area_size(64)
            .build_cartesian_2d(0f64..count_to_f64(n_groups), y_lo..y_hi)?;

        style_mesh!(chart.configure_mesh())
            .disable_x_mesh()
            .disable_x_axis()
            .y_desc("R2 gain (units of 1e-3)")
            .draw()?;

        // Alternating bands, so a bar is read against its own group rather than
        // against its neighbour across a group boundary.
        chart.draw_series((0..n_groups).filter(|g| g % 2 == 1).map(|g| {
            Rectangle::new(
                [(count_to_f64(g), y_lo), (count_to_f64(g) + 1.0, y_hi)],
                RGBColor(246, 246, 246).filled(),
            )
        }))?;

        // The baseline the bars are measured from. Signed bars, so it has to be
        // drawn rather than left implicit at the axis edge.
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(0.0, 0.0), (count_to_f64(n_groups), 0.0)],
            OK_BLACK.stroke_width(1),
        )))?;

        let slot = (1.0 - 2.0 * GROUP_PAD) / count_to_f64(ARMS);
        for arm in 0..ARMS {
            // One series per *slot*, not per geometry: a slot holds a different
            // geometry depending on the group's truth, so each bar carries its
            // own colour.
            let bars: Vec<(f64, f64, f64, &'static str)> = self
                .groups
                .iter()
                .enumerate()
                .filter_map(|(g, group)| {
                    let value = group.gains[arm]? * SCALE;
                    let x0 = count_to_f64(g)
                        + GROUP_PAD
                        + count_to_f64(arm) * slot
                        + slot * BAR_GAP / 2.0;
                    Some((x0, x0 + slot * (1.0 - BAR_GAP), value, group.arms[arm]))
                })
                .collect();

            chart.draw_series(bars.iter().map(|(x0, x1, v, geometry)| {
                Rectangle::new(
                    [(*x0, 0.0), (*x1, *v)],
                    geometry_color(geometry).mix(0.9).filled(),
                )
            }))?;

            // The value on its bar's far end, reading upwards. Horizontal text
            // at this bar width would overlap its neighbour, and the sign plus
            // the last digit is what the chart is read for.
            //
            // The anchor is stated in the *text's* frame whatever the rotation
            // (plotters' `text_anchor` doc), and `Rotate270` maps the text's
            // left edge to the bottom of the strip on screen. So a bar above
            // zero anchors `Left` and grows upward off its top; one below zero
            // anchors `Right` and grows downward off its bottom.
            let pad = (y_hi - y_lo) * 0.012;
            chart.draw_series(bars.iter().map(|(x0, x1, v, _)| {
                let (offset, h_pos) = if *v < 0.0 {
                    (-pad, HPos::Right)
                } else {
                    (pad, HPos::Left)
                };
                Text::new(
                    fixed(*v),
                    ((x0 + x1) / 2.0, v + offset),
                    ("sans-serif", 11)
                        .into_font()
                        .color(&RGBColor(60, 60, 60))
                        .transform(FontTransform::Rotate270)
                        .pos(Pos::new(h_pos, VPos::Center)),
                )
            }))?;
        }

        let (plot_px, _) = chart.plotting_area().get_pixel_range();
        self.draw_group_labels(&label_area, plot_px)
    }
}

impl MatchedGain {
    /// The group labels, drawn into the strip split off below the plot.
    ///
    /// *`plot_px`* is the plotting area's horizontal pixel range. Both it and
    /// the strip's range are absolute backend coordinates, so their difference
    /// is the offset into the strip, whose own draw calls are relative to its
    /// top-left.
    fn draw_group_labels<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        plot_px: std::ops::Range<i32>,
    ) -> Res
    where
        DB::ErrorType: 'static,
    {
        let strip_x0 = area.get_pixel_range().0.start;
        let width = f64::from(plot_px.end - plot_px.start);
        let n_groups = self.groups.len();
        for (g, group) in self.groups.iter().enumerate() {
            let centre = plot_px.start
                + to_i32((width * (count_to_f64(g) + 0.5) / count_to_f64(n_groups)).round());
            area.draw(&Text::new(
                dataset_label(&group.dataset).to_string(),
                (centre - strip_x0, 12),
                ("sans-serif", 14)
                    .into_font()
                    .color(&OK_BLACK)
                    .pos(Pos::new(HPos::Center, VPos::Center)),
            ))?;
            // Which arm the bars are measured *from*, without which a group of
            // two bars does not say what it is a gain over.
            area.draw(&Text::new(
                format!("matched: {}", geometry_label(group.truth)),
                (centre - strip_x0, 29),
                ("sans-serif", 11)
                    .into_font()
                    .color(&geometry_color(group.truth))
                    .pos(Pos::new(HPos::Center, VPos::Center)),
            ))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One row of the table, with a single region.
    fn row(dataset: &str, geometry: &str, r2: f64) -> Exp1Row {
        Exp1Row {
            dataset: dataset.to_string(),
            n: 1000,
            setting: "all_off".to_string(),
            space: ObjectiveSpace::Legacy10.tag().to_string(),
            geometry: geometry.to_string(),
            r2: [("all".to_string(), r2)].into_iter().collect(),
        }
    }

    /// All three arms of one dataset.
    fn arms(dataset: &str, euc: f64, hyp: f64, sph: f64) -> Vec<Exp1Row> {
        vec![
            row(dataset, "euclidean", euc),
            row(dataset, "hyperbolic", hyp),
            row(dataset, "spherical", sph),
        ]
    }

    /// The gain of one group's named arm.
    fn gain(fig: &MatchedGain, dataset: &str, geometry: &str) -> Option<f64> {
        let group = fig.groups.iter().find(|g| g.dataset == dataset)?;
        let i = group.arms.iter().position(|a| *a == geometry)?;
        group.gains[i]
    }

    /// R2 is a cost, so the subtraction runs mismatched-minus-matched: a
    /// mismatched arm that scored *worse* (larger R2) is a positive bar. The
    /// one thing the figure must not get backwards.
    #[test]
    fn a_worse_mismatched_arm_is_a_positive_gain() {
        // `tree` is hyperbolic by construction; give the matched arm the best
        // (smallest) R2 and the flat arm the worst.
        let fig = MatchedGain::new(&arms("tree", 0.06, 0.04, 0.05), 1000, "all");

        let euc = gain(&fig, "tree", "euclidean").expect("euclidean arm present");
        let sph = gain(&fig, "tree", "spherical").expect("spherical arm present");
        assert!((euc - 0.02).abs() < 1e-12, "got {euc}");
        assert!((sph - 0.01).abs() < 1e-12, "got {sph}");
    }

    /// And the other way round: a matched arm that lost reads as a negative
    /// bar rather than as a missing one.
    #[test]
    fn a_better_mismatched_arm_is_a_negative_gain() {
        let fig = MatchedGain::new(&arms("tree", 0.03, 0.04, 0.05), 1000, "all");

        let euc = gain(&fig, "tree", "euclidean").expect("euclidean arm present");
        assert!(euc < 0.0, "got {euc}");
    }

    /// The Euclidean comparison is in the picture for every dataset — as one of
    /// the two bars when the truth is curved, and as the reference the two
    /// curved bars are measured from when it is flat.
    #[test]
    fn the_euclidean_comparison_is_always_present() {
        let mut rows = arms("tree", 0.06, 0.04, 0.05); // hyperbolic truth
        rows.extend(arms("grid", 0.04, 0.06, 0.05)); // euclidean truth
        let fig = MatchedGain::new(&rows, 1000, "all");

        // Curved truth: euclidean is a bar.
        assert!(gain(&fig, "tree", "euclidean").is_some());

        // Flat truth: euclidean is the reference, so both bars are curved and
        // both are measured off it.
        let grid = fig
            .groups
            .iter()
            .find(|g| g.dataset == "grid")
            .expect("grid group");
        assert_eq!(grid.truth, "euclidean");
        assert_eq!(grid.arms, ["spherical", "hyperbolic"]);
        let hyp = gain(&fig, "grid", "hyperbolic").expect("hyperbolic arm present");
        assert!((hyp - 0.02).abs() < 1e-12, "got {hyp}");
    }

    /// Groups come out in `SYNTH_TRUTH` order whatever order the JSONL is in,
    /// so the figure's x axis matches the table's row order.
    #[test]
    fn groups_follow_the_ground_truth_order() {
        let mut rows = arms("tree", 0.06, 0.04, 0.05);
        rows.extend(arms("grid", 0.04, 0.06, 0.05));
        rows.extend(arms("sphere", 0.06, 0.05, 0.04));
        let fig = MatchedGain::new(&rows, 1000, "all");

        let order: Vec<&str> = fig.groups.iter().map(|g| g.dataset.as_str()).collect();
        assert_eq!(order, ["grid", "sphere", "tree"]);
    }

    /// Without the matched arm there is nothing to reference against, so the
    /// dataset is dropped rather than referenced to whichever arm survives.
    #[test]
    fn a_dataset_missing_its_matched_arm_is_dropped() {
        let rows = vec![
            row("tree", "euclidean", 0.06),
            row("tree", "spherical", 0.05),
        ];
        let fig = MatchedGain::new(&rows, 1000, "all");

        assert!(!fig.has_data());
    }

    /// A missing mismatched arm is a hole in the group, never a zero-height bar
    /// — which would read as "no difference".
    #[test]
    fn a_missing_mismatched_arm_is_a_hole_not_a_zero() {
        let rows = vec![
            row("tree", "hyperbolic", 0.04),
            row("tree", "euclidean", 0.06),
        ];
        let fig = MatchedGain::new(&rows, 1000, "all");

        assert!(gain(&fig, "tree", "euclidean").is_some());
        assert_eq!(gain(&fig, "tree", "spherical"), None);
    }

    /// A real dataset has no ground truth, so it has no matched arm and cannot
    /// appear — the question is only asked of the synthetic suite.
    #[test]
    fn real_datasets_are_not_plotted() {
        let fig = MatchedGain::new(&arms("mnist", 0.06, 0.04, 0.05), 1000, "all");

        assert!(!fig.has_data());
    }

    /// Asking for a region the table does not carry yields no data, rather than
    /// a figure of NaN bars.
    #[test]
    fn an_unknown_region_yields_no_data() {
        let fig = MatchedGain::new(&arms("tree", 0.06, 0.04, 0.05), 1000, "no_such_region");

        assert!(!fig.has_data());
    }

    /// Rows of another sample size belong to another figure.
    #[test]
    fn rows_of_another_n_are_excluded() {
        let fig = MatchedGain::new(&arms("tree", 0.06, 0.04, 0.05), 5000, "all");

        assert!(!fig.has_data());
    }

    /// The y range always contains zero: a bar's length is its value, so the
    /// baseline it is measured from has to be on the axis even when every bar
    /// falls on one side of it.
    #[test]
    fn the_axis_always_contains_zero() {
        let fig = MatchedGain::new(&arms("tree", 0.06, 0.04, 0.05), 1000, "all");
        let (lo, hi) = fig.y_range();
        assert!(lo < 0.0 && hi > 0.0, "range ({lo}, {hi}) excludes zero");

        // Same when the matched arm lost outright and every bar is negative.
        let fig = MatchedGain::new(&arms("tree", 0.02, 0.04, 0.03), 1000, "all");
        let (lo, hi) = fig.y_range();
        assert!(lo < 0.0 && hi > 0.0, "range ({lo}, {hi}) excludes zero");
    }

    /// Every dataset the ground-truth map names must be labelled: an
    /// unabbreviated name renders at full width under a 130px group and
    /// overlaps its neighbours instead of erroring.
    #[test]
    fn every_dataset_has_a_short_label() {
        for dataset in SYNTH_DATASETS {
            assert!(
                dataset_label(dataset).len() <= "tree (metric)".len(),
                "`{dataset}` has no short label"
            );
        }
    }
}
