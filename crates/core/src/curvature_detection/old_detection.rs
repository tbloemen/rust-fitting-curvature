use crate::curvature_detection::gromov::four_distinct;
use crate::curvature_detection::gromov::median_pairwise_distance;
use crate::curvature_detection::gromov::quad_delta;
// ── Global percentile estimate ────────────────────────────────────────────────

/// Estimate Gromov 4-point hyperbolicity from the distance matrix.
///
/// Samples random 4-tuples, computes δ = (S_max − S_mid) / 2 for each
/// (where S are the three distance-pair sums), and returns the 90th
/// percentile of δ normalised by the median pairwise distance.
///
/// Hyperbolic spaces have small normalised δ (bounded by log(2)/R for
/// curvature −1 and typical distance R), while Euclidean/spherical
/// spaces produce larger values.
pub fn gromov_hyperbolicity(distances: &[f64], n: usize, n_samples: usize) -> f64 {
    if n < 4 {
        return 0.0;
    }

    // Simple deterministic PRNG for reproducible sampling (raw u64 stream;
    // `four_distinct` reduces modulo the range).
    let mut state: u64 = 0xdeadbeef;
    let mut next = || -> usize {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as usize
    };

    let mut deltas = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let [a, b, c, d] = four_distinct(n, &mut next);
        deltas.push(quad_delta(distances, n, a, b, c, d) / 2.0);
    }

    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // 90th percentile δ.
    let p90_idx = (deltas.len() as f64 * 0.90) as usize;
    let delta_90 = deltas[p90_idx.min(deltas.len() - 1)];

    let median_d = median_pairwise_distance(distances, n);
    if median_d < 1e-12 {
        return 0.0;
    }
    delta_90 / median_d
}
