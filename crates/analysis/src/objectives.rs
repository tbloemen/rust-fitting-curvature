//! The 6 qParEGO objectives and their orientation into `[0, 1]`-higher-is-better.

use crate::records::TrialRecord;

/// The 6 qParEGO objectives, in the order written by the optimizer
/// (`default_pareto_metrics` in `crates/optimizer/src/pareto.rs`).
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
///   `davies_bouldin_ratio` and `cluster_density_measure` are absent while
///   `class_density_measure` is present: the first three are ratios whose upper
///   tails over `results/` reach 3.0e10, 2.9e11 and 2e24. Admitting one would
///   require estimated ideal/nadir bounds (Karl et al., *MOHPO — An Overview*,
///   §3.3.3; Grodzevich & Romanko 2006 §4.2), which no longer applies to a set
///   whose limits are all known a priori.
///
/// Row order is grouped by [`FAMILIES`] and is otherwise a free choice — the
/// indicators are invariant under a relabelling of the axes (the weight simplex
/// is enumerated symmetrically) and nothing on disk is positional, since
/// [`oriented_row`] resolves values by name and every output table is
/// name-keyed. What it *does* have to match is the optimizer's
/// `default_pareto_metrics`; that alignment is by hand across crates.
pub const OBJECTIVES: [&str; 6] = [
    // structure
    "trustworthiness",
    "continuity",
    // distance preservation
    "normalized_stress",
    "shepard_goodness",
    // class separation
    "neighborhood_hit",
    "class_density_measure",
];

/// Number of objectives; the dimension of the oriented objective space.
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
/// Indices rather than names because [`OBJECTIVES`] is ordered by family, so
/// they are contiguous and there is nothing to look up.
/// `families_partition_the_objectives` pins that.
pub const FAMILIES: [(&str, [usize; 2]); 3] = [
    ("structure", [0, 1]),
    ("distance", [2, 3]),
    ("class_separation", [4, 5]),
];

/// The five metrics that have both a projected and a manifold variant, as
/// `(projected, manifold)` column names.
///
/// **This is a diagnostic table, not the objective list.** It used to generate
/// [`OBJECTIVES`] by interleaving; it no longer does, and the two are now
/// independent — `class_density_measure` is an objective with no manifold
/// variant and correctly does not appear here.
///
/// Its remaining consumer is `figures/exp4.rs`, which plots one panel per row
/// to compare the manifold and projected readings of the same metric. That
/// figure is the evidence for dropping the manifold objectives, so it outlives
/// them. Row order is the Exp 4 panel order.
pub const METRIC_PAIRS: [(&str, &str); 5] = [
    ("trustworthiness", "trustworthiness_manifold"),
    ("continuity", "continuity_manifold"),
    ("normalized_stress", "normalized_stress_manifold"),
    ("shepard_goodness", "shepard_goodness_manifold"),
    ("neighborhood_hit", "neighborhood_hit_manifold"),
];

/// Objectives where lower is better; oriented as `1 - value`.
pub const MINIMIZE: [&str; 1] = ["normalized_stress"];

/// True when *name* is an objective that is minimised.
pub fn is_minimized(name: &str) -> bool {
    MINIMIZE.contains(&name)
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
    for (slot, name) in row.iter_mut().zip(OBJECTIVES) {
        *slot = oriented_value(name, r.objective(name));
    }
    row
}

/// The `(n, 6)` oriented-objective matrix for *records* (higher = better).
pub fn oriented_matrix(records: &[TrialRecord]) -> Vec<[f64; N_OBJECTIVES]> {
    records.iter().map(oriented_row).collect()
}
