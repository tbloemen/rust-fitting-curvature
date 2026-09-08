//! The qParEGO objectives and their orientation into `[0, 1]`-higher-is-better.

use std::sync::LazyLock;

use fitting_core::metrics::{Direction, Metric};

use crate::records::TrialRecord;

/// The qParEGO objectives, in the order the optimizer writes them.
///
/// This *is* `fitting_core::metrics::OBJECTIVES`, which
/// `optimizer::pareto::default_pareto_metrics` also returns. The two used to be
/// separate lists kept aligned by hand across crates — the alignment this
/// module's doc comment used to ask readers to maintain.
///
/// Every one is measured **after projection to 2D** and is bounded in `[0, 1]`
/// by construction. Both properties are load-bearing:
///
/// - *Projected only.* The manifold (pre-projection, geodesic) variants used to
///   occupy half this list. They are still measured and still present on
///   [`TrialRecord`] — [`METRIC_PAIRS`] and `figures/exp4.rs` read them — but
///   they no longer steer the search.
/// - *Bounded only.* [`oriented_value`] clamps to `[0, 1]` and the R2 ideal
///   point is pinned at `(1, …, 1)`, so an unbounded objective would be
///   silently truncated rather than measured. That is why `dunn_index`,
///   `davies_bouldin_ratio` and `cluster_density_measure` are absent: all three
///   are ratios whose upper tails over `results/` reach 3.0e10, 2.9e11 and 2e24.
///   Admitting one would require estimated ideal/nadir bounds (Karl et al.,
///   *MOHPO — An Overview*, §3.3.3; Grodzevich & Romanko 2006 §4.2), which no
///   longer applies to a set whose limits are all known a priori.
///
/// Both rules are now checked rather than described: `QualityMetric::space` and
/// `is_objective` carry them per metric, and `test_registry.rs` asserts them in
/// both directions.
///
/// Row order is grouped by [`FAMILIES`] and is otherwise a free choice — the
/// indicators are invariant under a relabelling of the axes (the weight simplex
/// is enumerated symmetrically) and nothing on disk is positional, since
/// [`oriented_row`] resolves values by name and every output table is
/// name-keyed. The grouping itself is not free: [`FAMILIES`] indexes into this
/// list by position, and `objectives_are_grouped_by_family` in the core
/// registry tests pins the contiguity.
pub const OBJECTIVES: &[Metric] = fitting_core::metrics::OBJECTIVES;

/// Number of objectives; the dimension of the oriented objective space.
///
/// A `const`, because `r2.rs`, `indicators.rs` and `pareto.rs` all use it as an
/// array length. `<[T]>::len` is const-evaluable, so this still derives from
/// the registry rather than restating its size.
pub const N_OBJECTIVES: usize = OBJECTIVES.len();

/// The three preference families, as `(name, indices into [`OBJECTIVES`])`.
///
/// These replace the `manifold` / `projected` regions, which became vacuous
/// once every objective was projected. The split is the standard DR-quality
/// taxonomy: what the projection preserves of the *neighbourhood* structure, of
/// the *distances*, and of the *labels*.
///
/// `neighborhood_hit` sits in `class_separation`, not `structure`: it is the
/// fraction of a point's k nearest neighbours in the embedding sharing its
/// label, so it reads `labels` and never touches the high-dimensional data. Its
/// resemblance to trustworthiness/continuity is that it is a k-NN statistic at
/// the same `k`, which is a computational similarity, not a semantic one.
///
/// **`class_separation` currently holds a single objective**, since
/// `class_density_measure` was dropped, so its region is *identical* to the
/// per-objective `neighborhood_hit` region — both admit exactly the vectors
/// putting at least half the mass on that one axis. It is reported anyway, so
/// the family row survives if a second bounded class-separation metric is added
/// back. Tests that contrast a family against its members exempt it for that
/// reason.
///
/// Membership is a **slice, not a fixed-size array**: the `[usize; 2]` this used
/// to be silently outlived the objective set it indexed into, leaving
/// `("class_separation", [4, 5])` reading index 5 of a 5-element row.
/// Variable arity removes the failure mode rather than the one instance.
///
/// Indices rather than names because [`OBJECTIVES`] is ordered by family, so
/// they are contiguous and there is nothing to look up.
/// `families_partition_the_objectives` pins that.
pub const FAMILIES: [(&str, &[usize]); 3] = [
    ("structure", &[0, 1]),
    ("distance", &[2, 3]),
    ("class_separation", &[4]),
];

/// The metrics that have both a projected and a manifold reading, as
/// `(projected, manifold)`.
///
/// **This is a diagnostic table, not the objective list.** It used to generate
/// [`OBJECTIVES`] by interleaving; it no longer does, and the two are now
/// independent — an objective with no manifold variant would correctly not
/// appear here.
///
/// Its remaining consumer is `figures/exp4.rs`, which plots one panel per row
/// to compare the manifold and projected readings of the same metric. That
/// figure is the evidence for dropping the manifold objectives, so it outlives
/// them.
///
/// Derived by matching `QualityMetric::base` over the registry, so a metric
/// that gains or loses a twin gains or loses a panel with no edit here.
pub static METRIC_PAIRS: LazyLock<Vec<(Metric, Metric)>> =
    LazyLock::new(|| Metric::dual_pairs().collect());

/// Length of [`METRIC_PAIRS`], as a `const` because `figures/exp4.rs` uses it
/// as an array length and `QualityMetric`'s methods are not const-callable.
/// The one number here that is restated rather than derived;
/// `metric_pairs_has_the_declared_length` pins it.
pub const N_METRIC_PAIRS: usize = 5;

/// True when *name* is an objective that is minimised.
///
/// Was a `MINIMIZE: [&str; 1]` constant listing `normalized_stress`; the
/// registry states each metric's own orientation, and
/// `only_normalized_stress_is_minimized` pins that this is still the only one.
pub fn is_minimized(name: &str) -> bool {
    Metric::by_name(name).is_some_and(|m| m.direction() == Direction::Minimize)
}

/// Map one raw metric value into `[0, 1]` with higher = better.
///
/// A missing / null / non-finite value is the worst case (0.0), matching the
/// optimizer's `metrics_to_vec` substitution so diverged trials score as bad
/// rather than being dropped silently.
pub fn oriented_value(name: &str, v: Option<f64>) -> f64 {
    let Some(x) = v else { return 0.0 };
    if !x.is_finite() {
        return 0.0;
    }
    let x = if is_minimized(name) { 1.0 - x } else { x };
    // Every objective is bounded in [0, 1] by construction; clamp defensively.
    x.clamp(0.0, 1.0)
}

/// One record's oriented objective vector.
pub fn oriented_row(r: &TrialRecord) -> [f64; N_OBJECTIVES] {
    let mut row = [0.0; N_OBJECTIVES];
    for (slot, metric) in row.iter_mut().zip(OBJECTIVES) {
        *slot = oriented_value(metric.name(), r.objective(metric.name()));
    }
    row
}

/// The `(n, 6)` oriented-objective matrix for *records* (higher = better).
pub fn oriented_matrix(records: &[TrialRecord]) -> Vec<[f64; N_OBJECTIVES]> {
    records.iter().map(oriented_row).collect()
}
