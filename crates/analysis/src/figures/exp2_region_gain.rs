//! Experiment 2 (`metric-results`) — which established metrics carry the
//! curvature gain?
//!
//! [`RegionGain`] is the figure this module draws: **a heatmap of the R2 gain
//! of a curved corpus over the Euclidean one, datasets down the side and the
//! preference regions of the methods chapter across, one panel per curved
//! geometry.** Experiment 1 reports that both curved geometries beat the flat
//! baseline under `W_all` and then says in the same breath that an overall
//! indicator cannot tell which metrics that came from. The regions are the
//! instrument that can: concentrating the weight set on one metric — or, in
//! the legacy space, on one *surface* — asks the same fronts what an analyst
//! with that priority would have seen. The panel therefore decomposes the
//! `W_all` bar of `exp1_matched_gain` column by column.
//!
//! ### The numbers are the table's
//!
//! The panel plots the **stage-1 table** (`results/r2_local_<space>.jsonl`,
//! one [`CellRecord`] per cell with its R2 under every region) rather than
//! re-scoring the sweeps, the same rule [`super::exp4::R2Bars`] follows: the
//! figure and the Typst tables cannot then disagree by a rounding. Each cell is
//!
//! `gain = 1000 · (R2(euclidean) − R2(curved))`
//!
//! under the region of its column. R2 is a cost, so the subtraction runs
//! Euclidean-minus-curved and a **positive** cell means the curved corpus
//! served that preference better — the reading direction of `ΔR2` in the
//! results chapter and of `exp1::MatchedGain`, and the easy thing to get
//! backwards. The `×1000` is the tables' own scale.
//!
//! ### Population
//!
//! The `all_off` cells only ([`SETTING`]), so curvature is the only thing that
//! differs between a row's Euclidean and curved fronts, at one N. Every dataset
//! with both cells at that N is a row — synthetic first, in the order the
//! results chapter introduces them, then the real ones — and a dataset missing
//! its Euclidean cell is left out rather than drawn grey, because it has no
//! reference to be measured from. A region absent from either record reads as
//! `n/a`; that is the one grey a cell can show.
//!
//! ### Columns
//!
//! Exactly [`region_labels`] of the objective space, in the order
//! `r2::build_regions` emits them, labelled as the R2 bar chart labels its
//! axis. In the legacy space that is `W_all`, one `W_<metric>` per metric
//! pair and the two surface regions `W_man` / `W_proj`; in the current space
//! `W_all`, the three families and one per objective. Nothing here names a
//! region, so an `obj6` table draws its own ten columns without a code change.
//!
//! ### The projected-only variant
//!
//! In the legacy space a metric-pair region cannot say whether a gain sits on
//! the page or on the manifold — its half-mass rule counts both readings —
//! and the `W_proj` column says only that the surface as a whole gained
//! little. [`Columns::Projected`] draws the same panel over the regions of
//! the projected surface alone ([`projected_region_labels`]): `W_proj` in the
//! place of `W_all`, then one region per metric holding the vectors of that
//! surface with at least half their mass on the metric's projected reading.
//! It is the per-metric decomposition of the `W_proj` column, and the panel an
//! analyst who only ever sees the page should read. The current space has no
//! manifold readings to exclude, so the variant is not drawn there; its full
//! panel already is this one.
//!
//! ### One colour scale per N
//!
//! The fill is the same diverging ramp the dependence heatmap uses
//! ([`super::exp2_dependence::diverging_color`]), centred on zero, but the
//! gains are not in `[-1, 1]`, so a panel divides by a **scale shared by both
//! geometries at that N** — the largest absolute gain either panel carries.
//! The hyperbolic and spherical panels of one N can then be read against each
//! other, which is what setting them side by side invites; the colourbar
//! ([`RegionGainColorbar`]) is written once per N with that scale on its
//! axis. Every value is also printed in its cell, so the panel reads without
//! it, and the sign is the finding: a column of small numbers beside a column
//! of large ones is the decomposition.

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use std::collections::BTreeMap;

use fitting_core::cast::{count_to_f64, to_i32};

use super::exp2::SETTING;
use super::exp2_dependence::{diverging_color, text_on, CategoryAxis, TITLE_STRIP};
use super::{
    Figure, LinearTicks, ObjectiveSpace, Res, CURVED, OK_BLACK, REAL_DATASETS, SYNTH_DATASETS,
};
use crate::aggregate::CellRecord;
use crate::r2::{projected_region_labels, region_labels};
use crate::style_mesh;

/// The reference geometry every gain is measured from.
const BASELINE_GEOMETRY: &str = "euclidean";

/// The tables' own scale for R2 and ΔR2.
const SCALE: f64 = 1000.0;

/// Canvas of one heatmap: eight regions across at ~60 px each behind a label
/// area wide enough for `hyperbolic_shells`, eight datasets down.
const PANEL: (u32, u32) = (620, 460);

/// Canvas of the colourbar: as wide as a panel, one strip tall.
const COLORBAR: (u32, u32) = (PANEL.0, 64);

/// Which preference regions a [`RegionGain`] panel puts across.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Columns {
    /// Every region of the space, as [`region_labels`] lists them.
    Full,
    /// The projected surface and its per-metric regions
    /// ([`projected_region_labels`]); legacy space only.
    Projected,
}

impl Columns {
    fn labels(self, space: ObjectiveSpace) -> Vec<(String, String)> {
        match self {
            Self::Full => region_labels(space),
            Self::Projected => projected_region_labels(space),
        }
    }

    /// The filename infix telling the two variants apart; the full panel
    /// carries none, as the only one until the other existed.
    fn slug(self) -> &'static str {
        match self {
            Self::Full => "",
            Self::Projected => "projected_",
        }
    }

    fn title_suffix(self) -> &'static str {
        match self {
            Self::Full => "",
            Self::Projected => ", projected readings only",
        }
    }
}

/// The per-region R2 gain of one curved geometry over the Euclidean corpus.
pub struct RegionGain {
    geometry: &'static str,
    n: usize,
    columns: Columns,
    /// Row labels, in draw order.
    datasets: Vec<String>,
    /// `(region name, axis label)`, in column order.
    regions: Vec<(String, String)>,
    /// `gain[row][col]`, already at [`SCALE`]; `None` where a region is missing.
    gain: Vec<Vec<Option<f64>>>,
    /// The absolute gain the ramp saturates at — shared by every panel of one N.
    scale: f64,
}

impl RegionGain {
    /// One panel per curved geometry with at least one complete row at this
    /// N, in [`CURVED`] order, all on one colour scale. Empty when the space
    /// has no such *columns* — the projected variant outside the legacy space.
    #[must_use]
    pub fn panels(
        rows: &[CellRecord],
        n: usize,
        space: ObjectiveSpace,
        columns: Columns,
    ) -> Vec<RegionGain> {
        let regions = columns.labels(space);
        if regions.is_empty() {
            return Vec::new();
        }
        // (dataset, geometry) → record, over the baseline setting at this N and
        // in this space. A record from the other space is not a rescaling of
        // this one and is skipped rather than differenced.
        let by_cell: BTreeMap<(&str, &str), &CellRecord> = rows
            .iter()
            .filter(|r| r.setting == SETTING && r.n == n && r.space == space.tag())
            .map(|r| ((r.dataset.as_str(), r.geometry.as_str()), r))
            .collect();

        let mut panels: Vec<RegionGain> = CURVED
            .iter()
            .filter_map(|&geometry| {
                let mut datasets = Vec::new();
                let mut gain = Vec::new();
                for &dataset in SYNTH_DATASETS.iter().chain(REAL_DATASETS.iter()) {
                    let (Some(flat), Some(curved)) = (
                        by_cell.get(&(dataset, BASELINE_GEOMETRY)),
                        by_cell.get(&(dataset, geometry)),
                    ) else {
                        continue;
                    };
                    datasets.push(dataset.to_string());
                    gain.push(
                        regions
                            .iter()
                            .map(|(name, _)| {
                                Some(SCALE * (flat.r2.get(name)? - curved.r2.get(name)?))
                            })
                            .collect(),
                    );
                }
                (!datasets.is_empty()).then(|| RegionGain {
                    geometry,
                    n,
                    columns,
                    datasets,
                    regions: regions.clone(),
                    gain,
                    scale: 0.0,
                })
            })
            .collect();

        let scale = panels
            .iter()
            .flat_map(|p| p.gain.iter().flatten().flatten())
            .fold(0.0f64, |acc, g| acc.max(g.abs()));
        for p in &mut panels {
            p.scale = scale;
        }
        panels
    }

    /// The absolute gain the fill saturates at.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The datasets down the side, in draw order.
    #[must_use]
    pub fn datasets(&self) -> &[String] {
        &self.datasets
    }

    /// The regions across, as `(name, axis label)`, in draw order.
    #[must_use]
    pub fn regions(&self) -> &[(String, String)] {
        &self.regions
    }

    /// The drawn value at `(row, col)`, at the tables' `×1000` scale.
    #[must_use]
    pub fn gain_at(&self, row: usize, col: usize) -> Option<f64> {
        self.gain[row][col]
    }

    /// The ramp position of a gain: `[-1, 1]` at the shared scale, or the
    /// grey `None`. A zero scale — every gain exactly zero — maps to white.
    fn ramp(&self, gain: Option<f64>) -> Option<f64> {
        gain.map(|g| {
            if self.scale > 0.0 {
                g / self.scale
            } else {
                0.0
            }
        })
    }
}

impl Figure for RegionGain {
    fn name(&self) -> String {
        format!(
            "exp2_region_gain_{}{}_N{}",
            self.columns.slug(),
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
        // The title strip is the one place the panel says which pair of
        // corpora it differences; N, the setting and the space are in the
        // filename, as on every Exp 2 figure.
        let (title, plot) = root.split_vertically(TITLE_STRIP);
        let (width, _) = title.dim_in_pixel();
        let centre = to_i32(f64::from(width) / 2.0);
        title.draw(&Text::new(
            format!(
                "{} vs {BASELINE_GEOMETRY}{}",
                self.geometry,
                self.columns.title_suffix()
            ),
            (centre, 12),
            ("sans-serif", 15)
                .into_font()
                .style(FontStyle::Bold)
                .color(&OK_BLACK)
                .pos(Pos::new(HPos::Center, VPos::Center)),
        ))?;

        let columns: Vec<String> = self.regions.iter().map(|(_, l)| l.clone()).collect();
        let x_axis = CategoryAxis::new(&columns, false);
        let y_axis = CategoryAxis::new(&self.datasets, true);
        let mut chart = ChartBuilder::on(&plot)
            .margin(6)
            // Room for the last column's label, which is centred on a cell
            // edge and would otherwise run off the canvas.
            .margin_right(28)
            .x_label_area_size(30)
            .y_label_area_size(110)
            .build_cartesian_2d(x_axis.clone(), y_axis.clone())?;
        style_mesh!(chart.configure_mesh())
            .disable_mesh()
            .x_label_formatter(&|v| x_axis.label(v))
            .y_label_formatter(&|v| y_axis.label(v))
            .draw()?;

        for (i, row) in self.gain.iter().enumerate() {
            for (j, &gain) in row.iter().enumerate() {
                let t = self.ramp(gain);
                let (x0, x1) = (count_to_f64(j), count_to_f64(j) + 1.0);
                let (y0, y1) = (count_to_f64(i), count_to_f64(i) + 1.0);
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x0, y0), (x1, y1)],
                    diverging_color(t).filled(),
                )))?;
                // A thin white seam between cells, so the grid reads without
                // mesh lines.
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x0, y0), (x1, y1)],
                    WHITE.stroke_width(1),
                )))?;

                let value = gain.map_or_else(
                    || "n/a".to_string(),
                    // A gain that rounds to zero prints as `0.0`, not `-0.0`:
                    // the sign of a rounding is not a finding.
                    |g| format!("{:.1}", if g.abs() < 0.05 { 0.0 } else { g }),
                );
                chart.draw_series(std::iter::once(
                    EmptyElement::at((x0 + 0.5, y0 + 0.5))
                        + Text::new(
                            value,
                            (0, 0),
                            ("sans-serif", 12)
                                .into_font()
                                .color(&text_on(t))
                                .pos(Pos::new(HPos::Center, VPos::Center)),
                        ),
                ))?;
            }
        }
        Ok(())
    }
}

/// The gain scale the [`RegionGain`] panels of one N are filled on, as its
/// own file — one per N, since the scale is the data's.
pub struct RegionGainColorbar {
    n: usize,
    columns: Columns,
    scale: f64,
}

impl RegionGainColorbar {
    /// The colourbar for *panels*, all of which share one scale; `None` if
    /// there are none to share it.
    #[must_use]
    pub fn from_panels(panels: &[RegionGain], n: usize) -> Option<Self> {
        let first = panels.first()?;
        // A zero scale has no axis to draw; the panels are then all white
        // and say so themselves.
        (first.scale > 0.0).then_some(Self {
            n,
            columns: first.columns,
            scale: first.scale,
        })
    }
}

impl Figure for RegionGainColorbar {
    fn name(&self) -> String {
        format!(
            "exp2_region_gain_{}colorbar_N{}",
            self.columns.slug(),
            self.n
        )
    }

    fn size(&self) -> (u32, u32) {
        COLORBAR
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        const STEPS: usize = 200;

        let ticks = LinearTicks::new((-self.scale, self.scale), 7);
        let mut chart = ChartBuilder::on(root)
            .margin(4)
            .margin_left(30)
            .margin_right(30)
            .x_label_area_size(38)
            .y_label_area_size(0)
            .build_cartesian_2d(ticks.clone(), 0.0f64..1.0f64)?;
        style_mesh!(chart.configure_mesh())
            .disable_mesh()
            .disable_y_axis()
            .x_desc(format!("R2 gain over {BASELINE_GEOMETRY} (x1000)"))
            .x_label_formatter(&|v| ticks.label(v))
            .draw()?;
        let step = 2.0 * self.scale / count_to_f64(STEPS);
        chart.draw_series((0..STEPS).map(|s| {
            let x0 = -self.scale + count_to_f64(s) * step;
            let mid = (x0 + step / 2.0) / self.scale;
            Rectangle::new(
                [(x0, 0.0), (x0 + step, 1.0)],
                diverging_color(Some(mid)).filled(),
            )
        }))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r2::REGION_ALL;

    fn record(
        dataset: &str,
        geometry: &str,
        space: ObjectiveSpace,
        r2: &[(&str, f64)],
    ) -> CellRecord {
        CellRecord {
            stem: format!("all_off_{dataset}_n5000_{geometry}"),
            space: space.tag().to_string(),
            setting: SETTING.to_string(),
            dataset: dataset.to_string(),
            n: 5000,
            geometry: geometry.to_string(),
            n_trials: 1,
            n_front: 1,
            r2: r2.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
        }
    }

    /// Every region of *space* at one level.
    fn flat(space: ObjectiveSpace, level: f64) -> Vec<(String, f64)> {
        region_labels(space)
            .into_iter()
            .map(|(name, _)| (name, level))
            .collect()
    }

    /// `(dataset, geometry, r2 by region)` rows.
    type Fixture<'a> = (&'a str, &'a str, &'a [(String, f64)]);

    fn records_at(space: ObjectiveSpace, rows: &[Fixture<'_>]) -> Vec<CellRecord> {
        rows.iter()
            .map(|(d, g, r2)| {
                let r2: Vec<(&str, f64)> = r2.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                record(d, g, space, &r2)
            })
            .collect()
    }

    #[test]
    fn gain_is_euclidean_minus_curved_at_the_tables_scale() {
        let space = ObjectiveSpace::Legacy10;
        let rows = records_at(
            space,
            &[
                ("tree", "euclidean", &flat(space, 0.10)),
                ("tree", "hyperbolic", &flat(space, 0.08)),
            ],
        );
        let panels = RegionGain::panels(&rows, 5000, space, Columns::Full);
        assert_eq!(panels.len(), 1, "only the hyperbolic arm has a row");
        let p = &panels[0];
        assert_eq!(p.geometry, "hyperbolic");
        assert_eq!(p.datasets(), ["tree".to_string()]);
        let all = p
            .regions()
            .iter()
            .position(|(name, _)| name == REGION_ALL)
            .expect("W_all column");
        let g = p.gain_at(0, all).expect("gain defined");
        assert!(
            (g - 20.0).abs() < 1e-9,
            "1000 * (0.10 - 0.08) = 20, got {g}"
        );
    }

    #[test]
    fn columns_are_the_space_regions_in_report_order() {
        for space in [ObjectiveSpace::Legacy10, ObjectiveSpace::Current6] {
            let rows = records_at(
                space,
                &[
                    ("grid", "euclidean", &flat(space, 0.1)),
                    ("grid", "spherical", &flat(space, 0.1)),
                ],
            );
            let panels = RegionGain::panels(&rows, 5000, space, Columns::Full);
            assert_eq!(panels[0].regions(), region_labels(space).as_slice());
        }
    }

    #[test]
    fn a_dataset_without_a_euclidean_cell_has_no_row() {
        let space = ObjectiveSpace::Legacy10;
        let rows = records_at(
            space,
            &[
                ("tree", "hyperbolic", &flat(space, 0.08)),
                ("sphere", "euclidean", &flat(space, 0.10)),
                ("sphere", "hyperbolic", &flat(space, 0.12)),
            ],
        );
        let panels = RegionGain::panels(&rows, 5000, space, Columns::Full);
        assert_eq!(panels.len(), 1);
        assert_eq!(panels[0].datasets(), ["sphere".to_string()]);
        // And nothing at all when no dataset has both cells.
        assert!(RegionGain::panels(&rows[..1], 5000, space, Columns::Full).is_empty());
    }

    #[test]
    fn both_geometries_share_the_largest_absolute_gain() {
        let space = ObjectiveSpace::Legacy10;
        let rows = records_at(
            space,
            &[
                ("tree", "euclidean", &flat(space, 0.10)),
                ("tree", "hyperbolic", &flat(space, 0.05)),
                ("tree", "spherical", &flat(space, 0.11)),
            ],
        );
        let panels = RegionGain::panels(&rows, 5000, space, Columns::Full);
        assert_eq!(panels.len(), 2);
        for p in &panels {
            assert!((p.scale() - 50.0).abs() < 1e-9, "scale {}", p.scale());
        }
        let bar = RegionGainColorbar::from_panels(&panels, 5000).expect("a scale to draw");
        assert!((bar.scale - 50.0).abs() < 1e-9);
    }

    #[test]
    fn projected_columns_are_the_surface_and_its_metrics_in_the_legacy_space_only() {
        let space = ObjectiveSpace::Legacy10;
        let rows = records_at(
            space,
            &[
                ("tree", "euclidean", &flat(space, 0.10)),
                ("tree", "hyperbolic", &flat(space, 0.08)),
            ],
        );
        let panels = RegionGain::panels(&rows, 5000, space, Columns::Projected);
        assert_eq!(panels.len(), 1);
        let names: Vec<&str> = panels[0]
            .regions()
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert_eq!(names[0], crate::r2::REGION_PROJECTED);
        assert_eq!(names.len(), 6, "the surface and one region per metric pair");
        assert!(
            names[1..].iter().all(|n| n.starts_with("projected:")),
            "{names:?}"
        );
        assert_eq!(
            panels[0].name(),
            "exp2_region_gain_projected_hyperbolic_N5000"
        );
        let bar = RegionGainColorbar::from_panels(&panels, 5000).expect("scale");
        assert_eq!(bar.name(), "exp2_region_gain_projected_colorbar_N5000");

        // The current space has no manifold readings to exclude.
        let space = ObjectiveSpace::Current6;
        let rows = records_at(
            space,
            &[
                ("tree", "euclidean", &flat(space, 0.10)),
                ("tree", "hyperbolic", &flat(space, 0.08)),
            ],
        );
        assert!(RegionGain::panels(&rows, 5000, space, Columns::Projected).is_empty());
    }

    #[test]
    fn other_settings_spaces_and_sizes_are_ignored() {
        let space = ObjectiveSpace::Legacy10;
        let mut rows = records_at(
            space,
            &[
                ("tree", "euclidean", &flat(space, 0.10)),
                ("tree", "hyperbolic", &flat(space, 0.08)),
            ],
        );
        rows[1].setting = "all_free".to_string();
        assert!(RegionGain::panels(&rows, 5000, space, Columns::Full).is_empty());
        rows[1].setting = SETTING.to_string();
        rows[1].n = 1000;
        assert!(RegionGain::panels(&rows, 5000, space, Columns::Full).is_empty());
        rows[1].n = 5000;
        assert!(
            RegionGain::panels(&rows, 5000, ObjectiveSpace::Current6, Columns::Full).is_empty()
        );
    }
}
