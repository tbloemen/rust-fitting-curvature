//! The 10 qParEGO objectives and their orientation into `[0, 1]`-higher-is-better.

use crate::records::TrialRecord;

/// The 10 qParEGO objectives, in the order written by the optimizer.
pub const OBJECTIVES: [&str; 10] = [
    "trustworthiness",
    "trustworthiness_manifold",
    "continuity",
    "continuity_manifold",
    "normalized_stress",
    "normalized_stress_manifold",
    "shepard_goodness",
    "shepard_goodness_manifold",
    "neighborhood_hit",
    "neighborhood_hit_manifold",
];

/// Number of objectives; the dimension of the oriented objective space.
pub const N_OBJECTIVES: usize = OBJECTIVES.len();

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
