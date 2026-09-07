use fitting_core::curvature_detection::{detect_geometry, GeometryVerdict};
use fitting_core::embedding::EmbeddingState;
use fitting_core::manifolds::create_manifold;
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::metrics::{
    class_density_measure, cluster_density_measure, continuity, davies_bouldin_ratio, dunn_index,
    knn_overlap, neighborhood_hit, normalized_stress, shepard_goodness, trustworthiness,
};
use fitting_core::visualisation::{project_to_2d, SphericalProjection};
use indicatif::ProgressBar;

use crate::data::Dataset;
use crate::metrics::AllMetrics;
use crate::search_space::TrialConfig;

pub struct Evaluator {
    dataset: Dataset,
    high_dim_dist: Vec<f64>,
    n_samples: usize,
}

impl Evaluator {
    pub fn new(dataset: Dataset) -> Self {
        let n = dataset.n_points;
        let high_dim_dist = if dataset.precomputed_distances.is_empty() {
            compute_euclidean_distance_matrix(&dataset.x, n, dataset.n_features)
        } else {
            dataset.precomputed_distances.clone()
        };
        Self {
            n_samples: n,
            dataset,
            high_dim_dist,
        }
    }

    pub fn n_points(&self) -> usize {
        self.n_samples
    }

    /// The precomputed high-dimensional pairwise distance matrix (flat,
    /// row-major, `n_points × n_points`). Used by `--mode detect` to run the
    /// curvature-detection fits directly on the data distances.
    pub fn distances(&self) -> &[f64] {
        &self.high_dim_dist
    }

    /// Detect the best-fitting geometry for this dataset.  Returns only
    /// the [`GeometryVerdict`] (geometry label + curvature) — the caller
    /// acts on the decision, not the detector's diagnostic internals.
    pub fn infer_geometry(&self) -> GeometryVerdict {
        // Fit the curvature models at the embedding target dimension (2-D).
        detect_geometry(&self.high_dim_dist, self.n_samples, 2)
    }

    pub fn compute_all_metrics(
        &self,
        config: &TrialConfig,
        curvature_sign: f64,
        seed: u64,
        pb_iters: &ProgressBar,
    ) -> AllMetrics {
        let n = self.n_samples;
        let training_config = config.to_training_config(n, curvature_sign, seed);

        pb_iters.reset();
        pb_iters.set_length(training_config.n_iterations as u64);

        let mut state = if self.dataset.precomputed_distances.is_empty() {
            EmbeddingState::new(&self.dataset.x, self.dataset.n_features, &training_config)
        } else {
            EmbeddingState::from_distances(&self.dataset.precomputed_distances, n, &training_config)
        }
        .with_loss_tracking(false);
        while !state.is_done() {
            state.step();
            pb_iters.inc(1);
        }

        metrics_from_embedding(
            &self.high_dim_dist,
            &self.dataset.labels,
            &state.points,
            n,
            state.ambient_dim,
            training_config.curvature,
        )
    }

    pub fn evaluate_with_metric(
        &self,
        config: &TrialConfig,
        curvature_sign: f64,
        metric: &str,
        seed: u64,
        pb_iters: &ProgressBar,
    ) -> f64 {
        let n = self.n_samples;
        let training_config = config.to_training_config(n, curvature_sign, seed);

        pb_iters.reset();
        pb_iters.set_length(training_config.n_iterations as u64);

        let mut state = if self.dataset.precomputed_distances.is_empty() {
            EmbeddingState::new(&self.dataset.x, self.dataset.n_features, &training_config)
        } else {
            EmbeddingState::from_distances(&self.dataset.precomputed_distances, n, &training_config)
        }
        .with_loss_tracking(false);
        while !state.is_done() {
            state.step();
            pb_iters.inc(1);
        }

        let projected = project_to_2d(
            &state.points,
            n,
            state.ambient_dim,
            training_config.curvature,
            SphericalProjection::AzimuthalEquidistant,
        );

        let k = (30_f64.min(n as f64 * 0.1)).round() as usize;

        // Lazily compute distance matrices only when needed.
        let dist_2d = || compute_euclidean_distance_matrix(&projected.coords, n, 2);
        let manifold_dist = || state.embedded_distances();

        match metric {
            "trustworthiness" => trustworthiness(&self.high_dim_dist, &dist_2d(), n, k),
            "trustworthiness_manifold" => {
                trustworthiness(&self.high_dim_dist, &manifold_dist(), n, k)
            }
            "continuity" => continuity(&self.high_dim_dist, &dist_2d(), n, k),
            "continuity_manifold" => continuity(&self.high_dim_dist, &manifold_dist(), n, k),
            "knn_overlap" => knn_overlap(&self.high_dim_dist, &dist_2d(), n, k),
            "knn_overlap_manifold" => knn_overlap(&self.high_dim_dist, &manifold_dist(), n, k),
            "neighborhood_hit" => neighborhood_hit(&dist_2d(), &self.dataset.labels, n, k),
            "neighborhood_hit_manifold" => {
                neighborhood_hit(&manifold_dist(), &self.dataset.labels, n, k)
            }
            "normalized_stress" => normalized_stress(&self.high_dim_dist, &dist_2d(), n),
            "normalized_stress_manifold" => {
                normalized_stress(&self.high_dim_dist, &manifold_dist(), n)
            }
            "shepard_goodness" => shepard_goodness(&self.high_dim_dist, &dist_2d(), n),
            "shepard_goodness_manifold" => {
                shepard_goodness(&self.high_dim_dist, &manifold_dist(), n)
            }
            "dunn_index" => dunn_index(&dist_2d(), &self.dataset.labels, n),
            "davies_bouldin_ratio" => davies_bouldin_ratio(
                &self.high_dim_dist,
                &projected.coords,
                &self.dataset.labels,
                n,
            ),
            "class_density_measure" => {
                class_density_measure(&projected.coords, &self.dataset.labels, n)
            }
            "cluster_density_measure" => {
                cluster_density_measure(&projected.coords, &self.dataset.labels, n)
            }
            _ => panic!(
                "Unknown metric: {metric}. Options: trustworthiness[_manifold], \
                 continuity[_manifold], knn_overlap[_manifold], neighborhood_hit[_manifold], \
                 normalized_stress[_manifold], shepard_goodness[_manifold], \
                 davies_bouldin_ratio, dunn_index, class_density_measure, cluster_density_measure"
            ),
        }
    }
}

/// Score a configuration on every metric, given only its coordinates.
///
/// Split out of [`Evaluator::compute_all_metrics`], which is now this function
/// plus the t-SNE run that produces `points`. The split was introduced so a
/// configuration that did *not* come from t-SNE — the Wilson reconstruction of
/// the former `--mode wilson-mds` — went through exactly this code rather than
/// a parallel copy. That mode is gone and `compute_all_metrics` is now the only
/// caller; the split is kept because it is the seam any future
/// score-these-coordinates path should re-use, for the same reason: two
/// embeddings scored by two pieces of code produce numbers that only look
/// comparable.
///
/// `points` is row-major `n × ambient_dim` on the manifold of the given
/// `curvature`, matching `EmbeddingState::points` / `Reconstruction::points`.
/// Both distance matrices are derived here rather than passed in, because
/// `EmbeddingState` derives them the same way — from
/// `create_manifold(curvature)` — so deriving them once here keeps the two
/// callers from drifting.
pub fn metrics_from_embedding(
    high_dim_dist: &[f64],
    labels: &[u32],
    points: &[f64],
    n: usize,
    ambient_dim: usize,
    curvature: f64,
) -> AllMetrics {
    let manifold = create_manifold(curvature);

    let projected = project_to_2d(
        points,
        n,
        ambient_dim,
        curvature,
        SphericalProjection::AzimuthalEquidistant,
    );

    let k = (30_f64.min(n as f64 * 0.1)).round() as usize;

    // Before-projection distances: manifold geodesic.
    let manifold_dist = manifold.pairwise_distances(points, n, ambient_dim);
    // After-projection distances: Euclidean in 2D projected space.
    let dist_2d = compute_euclidean_distance_matrix(&projected.coords, n, 2);

    let origin_dist = manifold.distances_from_origin(points, n, ambient_dim);
    let r_max = origin_dist.iter().cloned().fold(0.0_f64, f64::max);
    let r_rms = {
        let sum_sq: f64 = origin_dist.iter().map(|d| d * d).sum();
        (sum_sq / origin_dist.len() as f64).sqrt()
    };

    let r_gyration = gyration_radius(&manifold_dist, n);

    AllMetrics {
        trustworthiness: trustworthiness(high_dim_dist, &dist_2d, n, k),
        trustworthiness_manifold: trustworthiness(high_dim_dist, &manifold_dist, n, k),
        continuity: continuity(high_dim_dist, &dist_2d, n, k),
        continuity_manifold: continuity(high_dim_dist, &manifold_dist, n, k),
        knn_overlap: knn_overlap(high_dim_dist, &dist_2d, n, k),
        knn_overlap_manifold: knn_overlap(high_dim_dist, &manifold_dist, n, k),
        neighborhood_hit: neighborhood_hit(&dist_2d, labels, n, k),
        neighborhood_hit_manifold: neighborhood_hit(&manifold_dist, labels, n, k),
        normalized_stress: normalized_stress(high_dim_dist, &dist_2d, n),
        normalized_stress_manifold: normalized_stress(high_dim_dist, &manifold_dist, n),
        shepard_goodness: shepard_goodness(high_dim_dist, &dist_2d, n),
        shepard_goodness_manifold: shepard_goodness(high_dim_dist, &manifold_dist, n),
        davies_bouldin_ratio: davies_bouldin_ratio(high_dim_dist, &projected.coords, labels, n),
        dunn_index: dunn_index(&dist_2d, labels, n),
        class_density_measure: class_density_measure(&projected.coords, labels, n),
        cluster_density_measure: cluster_density_measure(&projected.coords, labels, n),
        r_max,
        r_rms,
        r_gyration,
    }
}

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
        for c in centroid.iter_mut() {
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
