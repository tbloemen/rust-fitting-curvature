//! The radius of gyration.
//!
//! The one function here with no `impl QualityMetric` beside it: it is a
//! *spread diagnostic*, not a quality metric, so it is absent from
//! [`ALL`](super::ALL) and reached through
//! [`SpreadDiagnostics`](crate::spread::SpreadDiagnostics) instead. It lives
//! among the metrics because it is the same shape of function — a distance
//! matrix in, a scalar out.

/// Radius of gyration from a full `n × n` pairwise distance matrix — the
/// embedding's spread, measured without an origin.
///
/// `R_g² = (1 / 2n²) ΣᵢΣⱼ d²ᵢⱼ`, which in flat space is *exactly* the mean
/// squared distance to the centroid. The `2n²` divisor is load-bearing: `dist`
/// is the full matrix, so every pair appears twice, and an `n(n−1)` divisor (or
/// a missing factor of two) still yields plausible-looking numbers while
/// quietly breaking the identity. `test_gyration_matches_centroid_rms` pins it.
///
/// This exists because `r_max`/`r_rms` are measured from a *fixed* pole. That is
/// meaningful on the hyperboloid, which `Hyperboloid::center` re-centres on the
/// origin every iteration, and vacuous on the sphere: `Sphere::center` is a
/// no-op, and `lift_pca_to_manifold` writes the constrained coordinate to the
/// last ambient slot while `Sphere::distances_from_origin` reads the first — so
/// PCA init lands every point ~90° from the pole κ is gauged against, and
/// `|K|·r_rms²` sits at `π²/4` however curved the space actually is.
#[must_use]
pub fn gyration_radius(dist: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = dist.iter().map(|d| d * d).sum();
    (sum_sq / (2.0 * (n * n) as f64)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::compute_euclidean_distance_matrix;

    /// The flat-space identity the `2n²` divisor exists for: in Euclidean space
    /// the gyration radius *is* the RMS distance to the centroid. A wrong
    /// divisor scales the result by a constant, which no eyeball check on a κ
    /// column would catch.
    #[test]
    fn test_gyration_matches_centroid_rms() {
        const N: usize = 37;
        const D: usize = 3;

        // Deterministic, spread over a few orders of magnitude so a constant
        // factor cannot hide in the noise.
        let mut points = vec![0.0f64; N * D];
        for i in 0..N {
            for d in 0..D {
                let t = (i * D + d) as f64;
                points[i * D + d] = (t * 0.7).sin() * (1.0 + t * 0.31);
            }
        }

        let dist = compute_euclidean_distance_matrix(&points, N, D);
        let got = gyration_radius(&dist, N);

        // Direct definition: RMS distance from the centroid.
        let mut centroid = [0.0f64; D];
        for i in 0..N {
            for (d, c) in centroid.iter_mut().enumerate() {
                *c += points[i * D + d];
            }
        }
        for c in &mut centroid {
            *c /= N as f64;
        }
        let want = {
            let sum_sq: f64 = (0..N)
                .map(|i| {
                    (0..D)
                        .map(|d| (points[i * D + d] - centroid[d]).powi(2))
                        .sum::<f64>()
                })
                .sum();
            (sum_sq / N as f64).sqrt()
        };

        assert!(
            (got - want).abs() < 1e-12 * want.max(1.0),
            "gyration radius {got} != centroid RMS {want}"
        );
    }

    /// A configuration collapsed to a point has zero spread, whatever the
    /// manifold's own radius is — the case `r_rms` reports as the `π²/4` floor
    /// on the sphere rather than as zero.
    #[test]
    fn test_gyration_of_collapsed_configuration_is_zero() {
        let dist = vec![0.0; 16 * 16];
        assert_eq!(gyration_radius(&dist, 16), 0.0);
    }
}
