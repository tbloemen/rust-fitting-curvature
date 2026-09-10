//! Experiment 2 (`metric-results`) — what effect does curvature have on the
//! established visualisation metrics?
//!
//! **A skeleton: nothing is drawn yet.** The module is wired into
//! `bin/figures.rs` so `--exp 2` has a slot, and [`MetricPanels::has_data`]
//! returns `false`, so the driver skips it and no empty file is written.
//!
//! The results chapter asks for two figures here
//! (`docs/thesis/sections/5results.typ`, under `<metric-results>`):
//!
//! * a dataset-by-metric panel comparing the Euclidean, hyperbolic and
//!   spherical corpora under `all_off`, grouped by metric family —
//!   trustworthiness/continuity, SNS/Shepard goodness, neighbourhood
//!   hit/distance consistency — over synthetic *and* real datasets;
//! * the Spearman metric-dependence heatmap of `<metric-dependence>`, within
//!   each dataset and geometry, asking whether the metrics order the same
//!   visualisations the same way.
//!
//! Two things the thesis TODOs insist on before either is inserted: state the
//! **population** each panel draws from (all trials, front points, or
//! preference-selected configurations — a front-point distribution describes
//! the searched corpus, not an unbiased sample), and orient every metric
//! consistently, which [`crate::objectives::is_minimized`] reads off the
//! registry.
//!
//! This is not the deleted `exp3.rs`. That one plotted the front's κ against
//! the data-intrinsic `κ_data` and belongs to `<curvature-magnitude-results>`
//! in the exploratory chapter, not to a results-chapter question.

use plotters::coord::Shift;
use plotters::prelude::*;

use super::{CellMap, Figure, ObjectiveSpace, Res};

/// Metric readings across the three embedding geometries, one panel per metric.
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

    /// Always `false` while this is a skeleton, so the driver's
    /// "figures with no data are skipped" rule keeps an empty SVG off disk.
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
