//! Experiment 2 (`metric-dependence`) — the same correlations as
//! [`MetricDependence`], drawn so that the *geometries* can be compared.
//!
//! [`DependenceDumbbell`] is the figure this module draws: **one row per
//! metric pair, Spearman ρ along x, one marker per embedding geometry on the
//! row, with every dataset's own ρ as a faint dot behind the marker.** The
//! per-geometry heatmaps answer "which metrics rank the corpus alike?" one
//! geometry at a time, and leave the question the results chapter actually
//! asks — does the answer change with curvature? — to the reader's eye,
//! jumping between three panels for every cell. Here the three ρ of one pair
//! sit on one row a few pixels apart: a pair whose markers straddle the zero
//! rule changes sign with the geometry, a pair whose markers stack changes
//! nothing, and the length of the run between them is how much curvature
//! moves the dependence.
//!
//! ### The rows are the pairs, grouped by family
//!
//! Every unordered pair of the metrics the heatmaps carry (28 for the eight
//! `obj10` metrics; the diagonal and the mirror half of the matrix are not
//! rows). The rows come in four blocks, each under its own header: the pairs
//! **within a family** first — the two structure metrics, the two distance
//! metrics, then the label-aware set — and then the three **between-family**
//! blocks. Within a block the rows are sorted by the Euclidean ρ, descending
//! (by the mean of the available geometries where the flat corpus is
//! missing), so each block reads from the pairs that agree most to the ones
//! that disagree, and a marker far from its neighbours is far *because of
//! the geometry*, not because the row order put it there. The grouping is
//! what lets the figure say something about families: a between-family block
//! whose rows all move the same way under curvature is a family-level
//! statement, a block where each row does its own thing is not.
//!
//! ### Three sub-rows, in the order of K
//!
//! The geometries are placed in curvature order, hyperbolic above the row's
//! centre line, Euclidean on it, spherical below ([`ORDER`]), each on its own
//! sub-row so the three whiskers do not overprint. A thin grey connector runs
//! hyperbolic → Euclidean → spherical through the three medians: a straight
//! slant is a dependence monotone in K, a `<` or `>` is one that moves with
//! curvature *either way* — curvature against flatness rather than the sign
//! of the curvature, the distinction the region-gain figure draws too.
//! Behind each median sit the **per-dataset ρ** it summarises, as small faint
//! dots on the same sub-row — the spread the heatmaps print as a min–max
//! range in small type under each median. Dots rather than a whisker because
//! with eight datasets the min–max is one outlying dataset wide (on most
//! pairs it spans nearly all of `[-1, 1]`) and says nothing about where the
//! other seven sit; the dots show whether a median is the centre of a cluster
//! or the middle of a scatter. A sign change whose dots overlap the other
//! geometry's is not a finding, and this is the figure where that shows.
//!
//! Colours and markers are the geometry palette (`geometry_color`), with the
//! Euclidean reference as an outlined square so the three stay apart in
//! greyscale. The legend strip above the plot names each geometry with the
//! number of datasets behind its ρ; N, the setting and the space stay in the
//! filename, as on every Exp 2 figure. The population, orientation and
//! missing-value rules are [`MetricDependence`]'s: this module only re-reads
//! the panels it is handed.

use plotters::coord::ranged1d::{DefaultFormatting, KeyPointHint, Ranged};
use plotters::coord::types::RangedCoordf64;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use fitting_core::cast::{count_to_f64, to_i32};
use fitting_core::metrics::{Family, Metric};

use super::exp2::short_label;
use super::exp2_dependence::MetricDependence;
use super::{geometry_color, plot_x, Figure, LinearTicks, Res, OK_BLACK, OK_GREY};
use crate::style_mesh;

/// The geometries in curvature order — K < 0, K = 0, K > 0 — top to bottom
/// on a row, with each one's offset from the row's centre line in slots.
const ORDER: [(&str, f64); 3] = [("hyperbolic", -0.3), ("euclidean", 0.0), ("spherical", 0.3)];

/// Canvas width: the Exp 1 figure width, one full text width in the thesis.
const WIDTH: u32 = 740;

/// Pixels per pair row (one slot).
const PAIR: u32 = 20;

/// Slots a block header takes, including the gap above it.
const HEADER: f64 = 1.1;

/// The strip above the plot carrying the legend.
const LEGEND_STRIP: u32 = 28;

/// Room for the x axis and its description.
const X_LABEL_AREA: u32 = 46;

/// Room for the row labels and the block headers, which share the area and
/// are both right-aligned to the axis: the longest header, `structure ×
/// class separation` at 13 px bold (~145 px), plus [`HEADER_GAP`], leaving
/// it a few pixels short of the canvas edge.
const Y_LABEL_AREA: u32 = 160;

const MARGIN: u32 = 8;

/// Gap between a block header's end and the axis.
const HEADER_GAP: i32 = 10;

/// Marker radius of a curved geometry's dot; the Euclidean square is drawn
/// to the same size.
const DOT: i32 = 4;

/// Radius of one dataset's own ρ, drawn faintly behind the median.
const SAMPLE_DOT: i32 = 2;

/// Which pairs a block holds, in draw order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Block {
    Within,
    StructureDistance,
    StructureClass,
    DistanceClass,
}

impl Block {
    fn of(a: Family, b: Family) -> Self {
        use Family::{ClassSeparation, Distance, Structure};
        match (a, b) {
            (Structure, Structure) | (Distance, Distance) | (ClassSeparation, ClassSeparation) => {
                Self::Within
            }
            (Structure, Distance) | (Distance, Structure) => Self::StructureDistance,
            (Structure, ClassSeparation) | (ClassSeparation, Structure) => Self::StructureClass,
            (Distance, ClassSeparation) | (ClassSeparation, Distance) => Self::DistanceClass,
        }
    }

    /// The header text. `×` is Latin-1, so it renders (see
    /// `figures/mod.rs` on the font); an en dash would not.
    fn title(self) -> &'static str {
        match self {
            Self::Within => "within a family",
            Self::StructureDistance => "structure \u{d7} distance",
            Self::StructureClass => "structure \u{d7} class separation",
            Self::DistanceClass => "distance \u{d7} class separation",
        }
    }

    const ALL: [Self; 4] = [
        Self::Within,
        Self::StructureDistance,
        Self::StructureClass,
        Self::DistanceClass,
    ];
}

/// The draw order of the families inside the `Within` block.
fn family_rank(f: Family) -> u8 {
    match f {
        Family::Structure => 0,
        Family::Distance => 1,
        Family::ClassSeparation => 2,
    }
}

/// One metric pair: its row.
#[derive(Debug, Clone)]
pub struct Pair {
    label: String,
    block: Block,
    family: u8,
    /// `(median, per-dataset ρ)` per geometry of [`ORDER`]; `None` where
    /// that geometry has no panel or no ρ for the pair.
    values: [Option<(f64, Vec<f64>)>; 3],
    /// Row centre, in slots from the top.
    centre: f64,
}

impl Pair {
    /// The ρ the block is sorted on: the flat corpus, or the mean of what is
    /// there when it is missing.
    fn reference(&self) -> f64 {
        let euclid = ORDER
            .iter()
            .position(|(g, _)| *g == "euclidean")
            .expect("euclidean is in ORDER");
        if let Some((m, _)) = &self.values[euclid] {
            return *m;
        }
        let present: Vec<f64> = self.values.iter().flatten().map(|(m, _)| *m).collect();
        if present.is_empty() {
            0.0
        } else {
            present.iter().sum::<f64>() / count_to_f64(present.len())
        }
    }
}

/// The metric-pair dumbbell plot over the per-geometry [`MetricDependence`]
/// panels of one N.
pub struct DependenceDumbbell {
    n: usize,
    rows: Vec<Pair>,
    /// `(block, header centre in slots)`, for the blocks that have rows.
    headers: Vec<(Block, f64)>,
    /// Slots from the top of the plot to its bottom.
    total: f64,
    /// `(geometry, datasets)` for every geometry with a panel, in [`ORDER`].
    present: Vec<(&'static str, usize)>,
}

impl DependenceDumbbell {
    /// The figure over *panels* — the geometries' heatmaps of one N, in any
    /// order. `None` if no panel has a pair to draw.
    #[must_use]
    pub fn from_panels(panels: &[MetricDependence], n: usize) -> Option<Self> {
        let by_geometry: Vec<Option<&MetricDependence>> = ORDER
            .iter()
            .map(|(g, _)| panels.iter().find(|p| p.geometry() == *g))
            .collect();
        let present: Vec<(&'static str, usize)> = ORDER
            .iter()
            .zip(&by_geometry)
            .filter_map(|((g, _), p)| p.map(|p| (*g, p.population().0)))
            .collect();

        // The metrics, in the first panel's order and then any the others
        // add; every panel of one run carries the same set, so this is a
        // guard, not a path.
        let mut metrics: Vec<Metric> = Vec::new();
        for p in by_geometry.iter().flatten() {
            for &m in p.metrics() {
                if !metrics.contains(&m) {
                    metrics.push(m);
                }
            }
        }

        let mut pairs: Vec<Pair> = Vec::new();
        for i in 0..metrics.len() {
            for j in i + 1..metrics.len() {
                let (first, second) = (metrics[i], metrics[j]);
                let mut values: [Option<(f64, Vec<f64>)>; 3] = [None, None, None];
                for (slot, panel) in by_geometry.iter().enumerate() {
                    let Some(panel) = panel else { continue };
                    let (Some(row), Some(col)) = (
                        panel.metrics().iter().position(|m| *m == first),
                        panel.metrics().iter().position(|m| *m == second),
                    ) else {
                        continue;
                    };
                    if let Some(med) = panel.median_at(row, col) {
                        values[slot] = Some((med, panel.samples_at(row, col).to_vec()));
                    }
                }
                if values.iter().all(Option::is_none) {
                    continue;
                }
                let block = Block::of(first.family(), second.family());
                pairs.push(Pair {
                    label: format!("{} / {}", short_label(first), short_label(second)),
                    block,
                    // Between-family blocks hold one family pair each, so the
                    // rank only orders the `Within` block.
                    family: family_rank(first.family()).max(family_rank(second.family())),
                    values,
                    centre: 0.0,
                });
            }
        }
        if pairs.is_empty() {
            return None;
        }

        // Blocks in order; inside a block by family, then by the reference
        // ρ descending. Both keys are finite, so `total_cmp` is a total order.
        pairs.sort_by(|x, y| {
            (x.block, x.family)
                .cmp(&(y.block, y.family))
                .then_with(|| y.reference().total_cmp(&x.reference()))
        });

        // Lay the rows out: a header slot before each block, one slot per row.
        let mut headers = Vec::new();
        let mut y = 0.0;
        let mut rows: Vec<Pair> = Vec::new();
        for block in Block::ALL {
            let members: Vec<&Pair> = pairs.iter().filter(|p| p.block == block).collect();
            if members.is_empty() {
                continue;
            }
            headers.push((block, y + HEADER / 2.0));
            y += HEADER;
            for m in members {
                let mut m = m.clone();
                m.centre = y + 0.5;
                rows.push(m);
                y += 1.0;
            }
        }
        Some(Self {
            n,
            rows,
            headers,
            total: y,
            present,
        })
    }

    /// The pair rows, top to bottom.
    #[must_use]
    pub fn rows(&self) -> &[Pair] {
        &self.rows
    }

    /// The blocks drawn, top to bottom.
    #[must_use]
    pub fn blocks(&self) -> Vec<Block> {
        self.headers.iter().map(|(b, _)| *b).collect()
    }
}

impl Pair {
    /// The row's label, `a / b`.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The block this pair is drawn in.
    #[must_use]
    pub fn block(&self) -> Block {
        self.block
    }

    /// The median ρ of *geometry*, if drawn.
    #[must_use]
    pub fn median(&self, geometry: &str) -> Option<f64> {
        ORDER
            .iter()
            .position(|(g, _)| *g == geometry)
            .and_then(|i| self.values[i].as_ref().map(|(m, _)| *m))
    }
}

/// The y axis: slots from the top, one per row plus the headers, reversed so
/// slot 0 is at the top, with a key point at each row's centre.
#[derive(Clone)]
pub(super) struct SlotAxis {
    pub(super) total: f64,
    pub(super) centres: Vec<f64>,
}

impl Ranged for SlotAxis {
    type FormatOption = DefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        RangedCoordf64::from(self.total..0.0).map(value, limit)
    }

    fn key_points<Hint: KeyPointHint>(&self, _hint: Hint) -> Vec<f64> {
        self.centres.clone()
    }

    fn range(&self) -> std::ops::Range<f64> {
        self.total..0.0
    }
}

impl Figure for DependenceDumbbell {
    fn name(&self) -> String {
        format!("exp2_dependence_dumbbell_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        let slots = to_i32((self.total * f64::from(PAIR)).ceil());
        let plot = u32::try_from(slots).unwrap_or(0);
        (WIDTH, LEGEND_STRIP + MARGIN + plot + X_LABEL_AREA + MARGIN)
    }

    #[allow(clippy::too_many_lines)]
    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let (legend, plot) = root.split_vertically(LEGEND_STRIP);
        self.draw_legend(&legend)?;

        let y_axis = SlotAxis {
            total: self.total,
            centres: self.rows.iter().map(|r| r.centre).collect(),
        };
        let ticks = LinearTicks::new((-1.0, 1.0), 5);
        let mut chart = ChartBuilder::on(&plot)
            .margin(MARGIN)
            .margin_right(16)
            .x_label_area_size(X_LABEL_AREA)
            .y_label_area_size(Y_LABEL_AREA)
            .build_cartesian_2d(ticks.clone(), y_axis.clone())?;
        let labels: Vec<&str> = self.rows.iter().map(|r| r.label.as_str()).collect();
        let y_label = |v: &f64| {
            self.rows
                .iter()
                .position(|r| (r.centre - v).abs() < 1e-9)
                .map_or_else(String::new, |i| labels[i].to_string())
        };
        style_mesh!(chart.configure_mesh())
            .disable_y_mesh()
            .x_desc("Spearman \u{3c1}: median over datasets (large), each dataset (small)")
            .x_label_formatter(&|v| ticks.label(v))
            .y_label_formatter(&y_label)
            .draw()?;

        // Alternate rows shaded, so a row's three sub-rows read as one.
        for (i, row) in self.rows.iter().enumerate() {
            if i % 2 == 1 {
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(-1.0, row.centre - 0.5), (1.0, row.centre + 0.5)],
                    RGBColor(240, 240, 240).filled(),
                )))?;
            }
        }
        // The zero rule, over the shading and under the marks.
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(0.0, 0.0), (0.0, self.total)],
            RGBColor(90, 90, 90).stroke_width(1),
        )))?;

        for row in &self.rows {
            // Connector through the medians, in K order.
            let path: Vec<(f64, f64)> = ORDER
                .iter()
                .zip(&row.values)
                .filter_map(|((_, off), v)| v.as_ref().map(|(m, _)| (*m, row.centre + off)))
                .collect();
            if path.len() > 1 {
                chart.draw_series(std::iter::once(PathElement::new(
                    path,
                    RGBColor(120, 120, 120).stroke_width(1),
                )))?;
            }
            // The datasets' own ρ as faint dots, then the median marker.
            for ((geometry, off), v) in ORDER.iter().zip(&row.values) {
                let Some((m, samples)) = v else { continue };
                let (m, y) = (*m, row.centre + off);
                let color = geometry_color(geometry);
                chart.draw_series(
                    samples
                        .iter()
                        .map(|&r| Circle::new((r, y), SAMPLE_DOT, color.mix(0.35).filled())),
                )?;
                if *geometry == "euclidean" {
                    chart.draw_series(std::iter::once(
                        EmptyElement::at((m, y))
                            + Rectangle::new([(-DOT, -DOT), (DOT, DOT)], color.filled())
                            + Rectangle::new([(-DOT, -DOT), (DOT, DOT)], OK_BLACK.stroke_width(1)),
                    ))?;
                } else {
                    chart.draw_series(std::iter::once(Circle::new((m, y), DOT, color.filled())))?;
                }
            }
        }

        // Block headers in the label area at their slot, right-aligned to
        // the axis like the row labels under them.
        let plot_px = chart.plotting_area().get_pixel_range();
        let area_origin = (
            plot.get_pixel_range().0.start,
            plot.get_pixel_range().1.start,
        );
        let rule_x0 = to_i32(f64::from(MARGIN)) + 2;
        let rule_x1 = plot_px.0.end - area_origin.0;
        let header_x = plot_px.0.start - area_origin.0 - HEADER_GAP;
        for (block, centre) in &self.headers {
            let y = plot_x(&plot_px.1, centre / self.total);
            plot.draw(&Text::new(
                block.title(),
                (header_x, y - area_origin.1),
                ("sans-serif", 13)
                    .into_font()
                    .style(FontStyle::Bold)
                    .color(&OK_BLACK)
                    .pos(Pos::new(HPos::Right, VPos::Center)),
            ))?;
            // A hairline under the header across the plot, closing the block
            // above and opening this one.
            let rule_y = plot_x(&plot_px.1, (centre + HEADER / 2.0) / self.total) - area_origin.1;
            plot.draw(&PathElement::new(
                vec![(rule_x0, rule_y), (rule_x1, rule_y)],
                OK_GREY.mix(0.5).stroke_width(1),
            ))?;
        }
        Ok(())
    }
}

impl DependenceDumbbell {
    /// The legend strip: one marker and name per present geometry, with the
    /// datasets behind it.
    fn draw_legend<DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let (width, height) = area.dim_in_pixel();
        let font = ("sans-serif", 14).into_font().color(&OK_BLACK);
        // Equal slots across the strip, from the left margin: the strip is
        // its own row above the plot, not aligned to the axis.
        let inner = f64::from(width) - 2.0 * f64::from(MARGIN);
        let slot = to_i32(inner / count_to_f64(self.present.len().max(1)));
        let cy = to_i32(f64::from(height) / 2.0);
        for (i, (geometry, datasets)) in self.present.iter().enumerate() {
            let x0 = to_i32(f64::from(MARGIN)) + DOT + to_i32(count_to_f64(i)) * slot;
            let color = geometry_color(geometry);
            if *geometry == "euclidean" {
                area.draw(&Rectangle::new(
                    [(x0 - DOT, cy - DOT), (x0 + DOT, cy + DOT)],
                    color.filled(),
                ))?;
                area.draw(&Rectangle::new(
                    [(x0 - DOT, cy - DOT), (x0 + DOT, cy + DOT)],
                    OK_BLACK.stroke_width(1),
                ))?;
            } else {
                area.draw(&Circle::new((x0, cy), DOT, color.filled()))?;
            }
            let noun = if *datasets == 1 {
                "dataset"
            } else {
                "datasets"
            };
            area.draw(&Text::new(
                format!("{geometry} ({datasets} {noun})"),
                (x0 + 12, cy),
                font.clone().pos(Pos::new(HPos::Left, VPos::Center)),
            ))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::figures::exp2::SETTING;
    use crate::figures::exp2_dependence::MIN_TRIALS;
    use crate::figures::CellMap;
    use crate::records::TrialRecord;

    fn trial(fields: &[(&str, f64)]) -> TrialRecord {
        let body: Vec<String> = fields
            .iter()
            .map(|(k, v)| format!("\"{k}\":{v:?}"))
            .collect();
        serde_json::from_str(&format!("{{{}}}", body.join(","))).expect("trial fixture")
    }

    /// `MIN_TRIALS` complete trials; `sign` flips the direction in which
    /// `davies_bouldin_ratio` follows `normalized_stress`, so one geometry
    /// can carry the opposite dependence from another.
    fn cell(sign: f64) -> Vec<TrialRecord> {
        (0..MIN_TRIALS)
            .map(|i| {
                let t = count_to_f64(i) / count_to_f64(MIN_TRIALS);
                trial(&[
                    ("trustworthiness", t),
                    ("continuity", t * t),
                    ("normalized_stress", t),
                    ("shepard_goodness", 1.0 - t),
                    ("neighborhood_hit", (t * 7.0).sin()),
                    ("dunn_index", t * 3.0),
                    ("davies_bouldin_ratio", sign * t + 0.1 * (t * 5.0).cos()),
                    ("cluster_density_measure", (t * 3.0).cos()),
                ])
            })
            .collect()
    }

    fn panels(geometries: &[(&str, f64)]) -> Vec<MetricDependence> {
        let mut cells = CellMap::new();
        for (g, sign) in geometries {
            cells.insert(Cell::new(SETTING, "a", 1000, g), cell(*sign));
            cells.insert(Cell::new(SETTING, "b", 1000, g), cell(*sign * 0.8));
        }
        MetricDependence::panels(&cells, 1000)
    }

    #[test]
    fn every_pair_once_grouped_by_block_in_order() {
        let panels = panels(&[("euclidean", 1.0), ("hyperbolic", -1.0), ("spherical", 1.0)]);
        let fig = DependenceDumbbell::from_panels(&panels, 1000).expect("rows");
        // Eight metrics: 28 unordered pairs, no diagonal.
        assert_eq!(fig.rows().len(), 28);
        let mut labels: Vec<&str> = fig.rows().iter().map(Pair::label).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), 28);
        assert_eq!(fig.blocks(), Block::ALL.to_vec());
        // Blocks are contiguous and in order.
        let blocks: Vec<Block> = fig.rows().iter().map(Pair::block).collect();
        let mut sorted = blocks.clone();
        sorted.sort();
        assert_eq!(blocks, sorted);
        // 1 + 1 + 6 within, 4 + 8 + 8 between.
        let count = |b: Block| blocks.iter().filter(|x| **x == b).count();
        assert_eq!(count(Block::Within), 8);
        assert_eq!(count(Block::StructureDistance), 4);
        assert_eq!(count(Block::StructureClass), 8);
        assert_eq!(count(Block::DistanceClass), 8);
    }

    #[test]
    fn rows_sort_by_euclidean_rho_within_a_block() {
        let panels = panels(&[("euclidean", 1.0), ("hyperbolic", -1.0)]);
        let fig = DependenceDumbbell::from_panels(&panels, 1000).expect("rows");
        for block in [Block::StructureClass, Block::DistanceClass] {
            let rhos: Vec<f64> = fig
                .rows()
                .iter()
                .filter(|r| r.block() == block)
                .map(|r| r.median("euclidean").expect("euclidean drawn"))
                .collect();
            assert!(
                rhos.windows(2).all(|w| w[0] >= w[1]),
                "{block:?} not descending: {rhos:?}"
            );
        }
        // The geometry with no panel has no marker.
        assert!(fig.rows().iter().all(|r| r.median("spherical").is_none()));
        assert_eq!(fig.present, vec![("hyperbolic", 2), ("euclidean", 2)]);
    }

    #[test]
    fn a_sign_flip_is_two_markers_either_side_of_zero() {
        let panels = panels(&[("euclidean", 1.0), ("hyperbolic", -1.0)]);
        let fig = DependenceDumbbell::from_panels(&panels, 1000).expect("rows");
        let row = fig
            .rows()
            .iter()
            .find(|r| r.label() == "1-stress / db")
            .expect("the stress-db pair");
        let e = row.median("euclidean").unwrap();
        let h = row.median("hyperbolic").unwrap();
        assert!(e * h < 0.0, "euclidean {e}, hyperbolic {h}");
    }

    #[test]
    fn height_follows_the_rows() {
        let panels = panels(&[("euclidean", 1.0)]);
        let fig = DependenceDumbbell::from_panels(&panels, 1000).expect("rows");
        let (w, h) = fig.size();
        assert_eq!(w, WIDTH);
        let slots = 28.0 + 4.0 * HEADER;
        assert!(f64::from(h) >= slots * f64::from(PAIR));
        assert_eq!(fig.name(), "exp2_dependence_dumbbell_N1000");
        assert!(DependenceDumbbell::from_panels(&[], 1000).is_none());
    }
}
