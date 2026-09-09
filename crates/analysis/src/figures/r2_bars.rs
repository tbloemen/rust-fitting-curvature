//! The R2 indicator as bar charts — one chart per (dataset, geometry).
//!
//! Same table `scripts/r2_delta_typst.py` renders, read the other way round: the
//! table is meant to be scanned *down* a column, this is meant to be read
//! *across* the priorities. One chart per (dataset, geometry, N); x groups are
//! the eight preference regions of `@preference-regions`; the bars inside a
//! group are the loss-weight settings, so "which setting wins under this
//! priority, and does the winner change with the priority" is a single glance
//! rather than a row-by-row comparison.
//!
//! Four choices worth stating:
//!
//! - **The level `r2`, not the gain `delta_r2`.** So `all_off` is a bar like any
//!   other setting rather than the zero line, and the regions are comparable
//!   against each other — `W_shep` is a costlier region to serve than `W_trust`
//!   whatever the setting does. The price is resolution: the settings of one
//!   region typically differ by well under a percent of the level, which is why
//!   every bar carries its value as a label.
//! - **The y axis starts at zero**, because a bar's length is its value. A
//!   truncated axis would turn those sub-percent differences into dramatic
//!   steps, which is the one thing this chart must not do.
//! - **R2 is a cost** (distance to the ideal point, @eq:r2), so a *shorter* bar
//!   is the better front. That runs opposite to every quality metric and the
//!   title says so.
//! - **Everything is scaled by [`SCALE`]**, exactly as the table is, so a chart
//!   and the table row it comes from carry the same digits.

use fitting_core::cast::{count_to_f64, to_i32};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use super::{
    draw_legend, setting_color, Figure, LegendEntry, Res, FAMILIES, OBJECTIVES, OK_BLACK,
    SETTING_ORDER,
};
use crate::aggregate::DeltaRow;
use crate::error::{Error, Result};
use crate::records::load_jsonl;
use crate::style_mesh;

/// Every entry is multiplied by this, the way the Typst table multiplies it:
/// R2 sits around 0.1, so the unscaled axis would spend three of its four digits
/// on leading zeros.
const SCALE: f64 = 1000.0;

/// Settings that are not drawn.
///
/// `rms_anchored` fixes the curvature gauge for Experiment 3 rather than
/// ablating a loss term, and it exists for hyperbolic only — so it is not one of
/// the alternatives this chart compares, and its (often much larger) level sets
/// the y scale for bars it does not belong beside. The Typst table drops it for
/// the same reason.
const EXCLUDED: [&str; 1] = ["rms_anchored"];

/// Height reserved under the plot for the group labels.
const LABEL_AREA: i32 = 26;

/// Fraction of a group's width left empty at each end, so neighbouring groups'
/// bars do not touch.
const GROUP_PAD: f64 = 0.10;
/// Fraction of a bar's slot left empty, as the gap between bars of one group.
const BAR_GAP: f64 = 0.15;

/// Fraction of the tallest bar left clear above it, for the value labels.
const HEAD_ROOM: f64 = 0.16;

/// Load the R2 rows written by `r2 aggregate --deltas`.
///
/// An **absent** table is not an error: it is a separate `r2` run, and the bar
/// charts are skipped without it exactly as Exp 3 skips its scatter without a
/// `κ_data` export. A table that is there and will not parse still fails.
///
/// # Errors
///
/// Returns `Err` if the file is present but malformed. A missing file returns
/// an empty `Vec`.
pub fn load_deltas(path: &Path) -> Result<Vec<DeltaRow>> {
    match load_jsonl(path) {
        Ok(rows) => Ok(rows),
        Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            Ok(Vec::new())
        }
        Err(e) => Err(e),
    }
}

/// Short axis labels for the preference regions, in report order.
///
/// Derived from [`FAMILIES`] and [`OBJECTIVES`] rather than written out again,
/// so the order and membership cannot desync from `r2::build_regions`, which
/// builds the regions in exactly this order.
fn regions() -> Vec<(&'static str, String)> {
    let mut out = vec![("all", "W_all".to_string())];
    out.extend(
        FAMILIES
            .iter()
            .map(|(family, _)| (*family, format!("W_{}", short_metric(family)))),
    );
    out.extend(
        OBJECTIVES
            .iter()
            .map(|o| (o.name(), format!("W_{}", o.short()))),
    );
    out
}

/// Abbreviations for the family region labels; the full names do not fit.
///
/// Only the families are listed here — a metric carries its own abbreviation
/// on `QualityMetric::short`, so the objective regions no longer need an arm.
/// The fallthrough is a hazard rather than a convenience: an unabbreviated name
/// renders at full width and overlaps its neighbours, so every region
/// [`regions`] can emit needs an arm here or a `short()`.
/// `every_region_label_is_abbreviated` in `test_r2.rs` pins that.
fn short_metric(metric: &str) -> &str {
    match metric {
        "structure" => "struct",
        "distance" => "dist",
        "class_separation" => "class",
        other => other,
    }
}

/// A level at the scale the axis is in: decimals drop as the magnitude grows,
/// the way the Typst table's `fixed` does it, so the labels stay three or four
/// significant figures wide.
fn fixed(scaled: f64) -> String {
    let decimals = if scaled.abs() < 10.0 {
        2
    } else {
        usize::from(scaled.abs() < 100.0)
    };
    format!("{scaled:.decimals$}")
}

/// One preference region's column of bars.
struct Group {
    label: String,
    /// R2 per setting, in [`R2Bars::settings`] order; `None` where that setting
    /// has no row for this region (an incomplete run, not a zero).
    values: Vec<Option<f64>>,
}

/// The R2 levels of one (dataset, geometry) cell block at one N.
pub struct R2Bars {
    dataset: String,
    geometry: String,
    n: usize,
    /// Bar order within every group.
    settings: Vec<String>,
    groups: Vec<Group>,
}

impl R2Bars {
    /// One chart per (dataset, geometry) present at this N.
    #[must_use]
    pub fn panels(rows: &[DeltaRow], n: usize) -> Vec<R2Bars> {
        // (dataset, geometry) → setting → region → row.
        type Block<'a> = BTreeMap<&'a str, BTreeMap<&'a str, &'a DeltaRow>>;
        let mut blocks: BTreeMap<(&str, &str), Block> = BTreeMap::new();
        for r in rows.iter().filter(|r| r.n == n) {
            blocks
                .entry((&r.dataset, &r.geometry))
                .or_default()
                .entry(&r.setting)
                .or_default()
                .insert(&r.region, r);
        }

        blocks
            .into_iter()
            .map(|((dataset, geometry), by_setting)| {
                let settings = bar_order(&by_setting);
                let groups = regions()
                    .into_iter()
                    .filter_map(|(region, label)| {
                        let values: Vec<Option<f64>> = settings
                            .iter()
                            .map(|s| {
                                by_setting[s.as_str()]
                                    .get(region)
                                    .map(|r| r.r2)
                                    .filter(|v| v.is_finite())
                            })
                            .collect();
                        if values.iter().all(Option::is_none) {
                            return None;
                        }
                        Some(Group { label, values })
                    })
                    .collect();
                R2Bars {
                    dataset: dataset.to_string(),
                    geometry: geometry.to_string(),
                    n,
                    settings,
                    groups,
                }
            })
            .filter(R2Bars::has_data)
            .collect()
    }

    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.settings.is_empty() && !self.groups.is_empty()
    }

    /// The tallest bar, scaled; `0.0` when there is nothing to draw.
    fn scaled_max(&self) -> f64 {
        self.groups
            .iter()
            .flat_map(|g| g.values.iter().flatten().map(|v| v * SCALE))
            .fold(0.0f64, f64::max)
    }
}

/// The settings to draw, in [`SETTING_ORDER`], with anything present but
/// unlisted appended rather than dropped — except [`EXCLUDED`].
fn bar_order(by_setting: &BTreeMap<&str, BTreeMap<&str, &DeltaRow>>) -> Vec<String> {
    let present: BTreeSet<&str> = by_setting
        .keys()
        .copied()
        .filter(|s| !EXCLUDED.contains(s))
        .collect();
    let mut out: Vec<String> = SETTING_ORDER
        .iter()
        .filter(|s| present.contains(*s))
        .map(|s| (*s).to_string())
        .collect();
    out.extend(
        present
            .iter()
            .filter(|s| !SETTING_ORDER.contains(s))
            .map(|s| (*s).to_string()),
    );
    out
}

impl Figure for R2Bars {
    fn name(&self) -> String {
        format!("r2_bars_{}_{}_N{}", self.dataset, self.geometry, self.n)
    }

    fn size(&self) -> (u32, u32) {
        // Wide enough that a bar can carry its own value: the settings of one
        // region differ in the third digit, so the labels are the chart's
        // resolution and they must not collide.
        (
            200 + 150 * u32::try_from(self.groups.len()).expect("a small number of groups"),
            520,
        )
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let root = root.titled(
            &format!(
                "R2 indicator by preference region — {} / {}, N={} (a cost: shorter is better)",
                self.dataset, self.geometry, self.n
            ),
            ("sans-serif", 18).into_font().color(&OK_BLACK),
        )?;
        let (legend, body) = root.split_vertically(32);

        let entries: Vec<LegendEntry> = self
            .settings
            .iter()
            .map(|s| LegendEntry::new(s.clone(), setting_color(s)))
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
        // Zero-based, always: the bar's length is the value it reports.
        let y_hi = self.scaled_max() * (1.0 + HEAD_ROOM);
        let y_hi = if y_hi > 0.0 { y_hi } else { 1.0 };

        let mut chart = ChartBuilder::on(&plot_area)
            .margin(10)
            .x_label_area_size(0)
            .y_label_area_size(64)
            .build_cartesian_2d(0f64..count_to_f64(n_groups), 0f64..y_hi)?;

        style_mesh!(chart.configure_mesh())
            .disable_x_mesh()
            .disable_x_axis()
            .y_desc("R2 (units of 1e-3)")
            .draw()?;

        // Alternating bands, so a bar is read against its own group rather than
        // against its neighbour across a group boundary.
        chart.draw_series((0..n_groups).filter(|g| g % 2 == 1).map(|g| {
            Rectangle::new(
                [(count_to_f64(g), 0.0), (count_to_f64(g) + 1.0, y_hi)],
                RGBColor(246, 246, 246).filled(),
            )
        }))?;

        let slot = (1.0 - 2.0 * GROUP_PAD) / count_to_f64(self.settings.len());
        for (i, setting) in self.settings.iter().enumerate() {
            let color = setting_color(setting);
            let bars: Vec<(f64, f64, f64)> = self
                .groups
                .iter()
                .enumerate()
                .filter_map(|(g, group)| {
                    let value = group.values[i]? * SCALE;
                    let x0 =
                        count_to_f64(g) + GROUP_PAD + count_to_f64(i) * slot + slot * BAR_GAP / 2.0;
                    Some((x0, x0 + slot * (1.0 - BAR_GAP), value))
                })
                .collect();

            chart.draw_series(bars.iter().map(|(x0, x1, v)| {
                Rectangle::new([(*x0, 0.0), (*x1, *v)], color.mix(0.9).filled())
            }))?;

            // The value on top of its bar, reading upwards. Horizontal text at
            // this bar width would overlap its neighbour, and the differences
            // the chart is read for live in the last digit.
            //
            // The anchor is stated in the *text's* frame whatever the rotation
            // (plotters' `text_anchor` doc), and `Rotate270` maps the text's
            // left edge to the bottom of the strip on screen — so `Left` is the
            // end that sits on the bar and `Center` centres the strip on it.
            chart.draw_series(bars.iter().map(|(x0, x1, v)| {
                Text::new(
                    fixed(*v),
                    ((x0 + x1) / 2.0, v + y_hi * 0.012),
                    ("sans-serif", 11)
                        .into_font()
                        .color(&RGBColor(60, 60, 60))
                        .transform(FontTransform::Rotate270)
                        .pos(Pos::new(HPos::Left, VPos::Center)),
                )
            }))?;
        }

        // Group labels, placed from the plot's pixel range. Both ranges are
        // absolute backend coordinates, so the difference is the offset into the
        // label strip, whose own draw calls are relative to its top-left.
        let (plot_px, _) = chart.plotting_area().get_pixel_range();
        let strip_x0 = label_area.get_pixel_range().0.start;
        let width = f64::from(plot_px.end - plot_px.start);
        for (g, group) in self.groups.iter().enumerate() {
            let centre = plot_px.start
                + to_i32((width * (count_to_f64(g) + 0.5) / count_to_f64(n_groups)).round());
            label_area.draw(&Text::new(
                group.label.clone(),
                (centre - strip_x0, 13),
                ("sans-serif", 14)
                    .into_font()
                    .color(&OK_BLACK)
                    .pos(Pos::new(HPos::Center, VPos::Center)),
            ))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unabbreviated label renders at full width and overlaps its
    /// neighbours instead of erroring, so every region needs a short form —
    /// whichever route produced it. Objective regions take theirs from
    /// `QualityMetric::short`; the families still come from [`short_metric`],
    /// whose fallthrough is the hazard this guards.
    #[test]
    fn every_region_label_is_abbreviated() {
        for (name, label) in regions() {
            if name == "all" {
                continue;
            }
            assert_ne!(
                label,
                format!("W_{name}"),
                "region `{name}` has no abbreviation, so it renders as `{label}`"
            );
        }
    }

    /// The bar chart's x axis is `r2::build_regions`' output order; a region
    /// added to one and not the other silently mislabels every bar after it.
    #[test]
    fn labels_match_the_regions_the_indicator_builds() {
        let built: Vec<String> = crate::r2::Weights::new()
            .regions
            .iter()
            .map(|r| r.name.clone())
            .collect();
        let labelled: Vec<String> = regions().into_iter().map(|(n, _)| n.to_string()).collect();
        assert_eq!(labelled, built);
    }
}
