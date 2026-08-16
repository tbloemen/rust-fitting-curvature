//! The 10 qParEGO objectives and their orientation into `[0, 1]`-higher-is-better.

use crate::records::TrialRecord;

/// The five quality metrics with their `(projected, manifold)` objective names.
///
/// The only place the pairing is written down: [`OBJECTIVES`], [`METRICS`] and
/// the objective indices are all derived from this table, so there is no second
/// order to keep in sync.
///
/// Row order is the objective-space axis order and the Exp 4 panel order. It is
/// a free choice — the indicators are invariant under a relabelling of the axes
/// (the weight simplex is enumerated symmetrically) and nothing on disk is
/// positional, since [`oriented_row`] resolves values by name and every output
/// table is name-keyed. It only has to stay consistent within one build, which
/// deriving the rest from here is what guarantees.
pub const METRIC_PAIRS: [(&str, &str); 5] = [
    ("trustworthiness", "trustworthiness_manifold"),
    ("continuity", "continuity_manifold"),
    ("normalized_stress", "normalized_stress_manifold"),
    ("shepard_goodness", "shepard_goodness_manifold"),
    ("neighborhood_hit", "neighborhood_hit_manifold"),
];

/// Number of paired quality metrics.
pub const N_METRICS: usize = METRIC_PAIRS.len();

/// Number of objectives; the dimension of the oriented objective space.
pub const N_OBJECTIVES: usize = 2 * N_METRICS;

const fn flatten_pairs() -> [&'static str; N_OBJECTIVES] {
    let mut out = [""; N_OBJECTIVES];
    let mut i = 0;
    while i < N_METRICS {
        out[2 * i] = METRIC_PAIRS[i].0;
        out[2 * i + 1] = METRIC_PAIRS[i].1;
        i += 1;
    }
    out
}

const fn projected_names() -> [&'static str; N_METRICS] {
    let mut out = [""; N_METRICS];
    let mut i = 0;
    while i < N_METRICS {
        out[i] = METRIC_PAIRS[i].0;
        i += 1;
    }
    out
}

/// The 10 qParEGO objectives, in the order written by the optimizer
/// (`default_pareto_metrics` in `crates/optimizer/src/pareto.rs`).
pub const OBJECTIVES: [&str; N_OBJECTIVES] = flatten_pairs();

/// The five quality metrics, i.e. the projected half of [`METRIC_PAIRS`].
pub const METRICS: [&str; N_METRICS] = projected_names();

/// The `(projected, manifold)` objective indices of metric `i` — the inverse of
/// the interleaving [`flatten_pairs`] performs.
pub fn metric_objectives(i: usize) -> (usize, usize) {
    (2 * i, 2 * i + 1)
}

/// True for objectives measured on the embedding manifold. Same interleaving:
/// [`flatten_pairs`] puts the manifold name of every pair at an odd index.
pub fn is_manifold(j: usize) -> bool {
    j % 2 == 1
}

/// Objectives where lower is better; oriented as `1 - value`.
pub const MINIMIZE: [&str; 2] = ["normalized_stress", "normalized_stress_manifold"];

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
    // Metrics are bounded in [0, 1] by construction; clamp defensively.
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

/// The `(n, 10)` oriented-objective matrix for *records* (higher = better).
pub fn oriented_matrix(records: &[TrialRecord]) -> Vec<[f64; N_OBJECTIVES]> {
    records.iter().map(oriented_row).collect()
}
