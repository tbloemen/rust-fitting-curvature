//! Tests for embedding quality metrics.
//! Ported from Python `test/test_metrics.py`

use fitting_core::context::EmbeddingContext;
use fitting_core::metrics::*;
use fitting_core::metrics::{Metric, MetricValue, MetricValues};
use fitting_core::spread::SpreadDiagnostics;
use fitting_core::synthetic_data::Rng;
use fitting_core::visualisation::SphericalProjection;

// ---------------------------------------------------------------------------
// Helpers shared by multiple tests
// ---------------------------------------------------------------------------

/// Build 2D point clusters: `n_clusters` groups of `per_cluster` points,
/// placed on a wide grid so clusters are well-separated.
fn make_clustered_2d(
    n_clusters: usize,
    per_cluster: usize,
    spread: f64,
    seed: u64,
) -> (Vec<f64>, Vec<u32>) {
    let mut rng = Rng::new(seed);
    let n = n_clusters * per_cluster;
    let mut pts = vec![0.0f64; n * 2];
    let mut labels = vec![0u32; n];
    for c in 0..n_clusters {
        let cx = (c as f64) * 20.0;
        for i in 0..per_cluster {
            let idx = c * per_cluster + i;
            pts[idx * 2] = cx + rng.normal() * spread;
            pts[idx * 2 + 1] = rng.normal() * spread;
            labels[idx] = c as u32;
        }
    }
    (pts, labels)
}

fn make_distance_matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let val = rng.uniform() * 5.0 + 0.1;
            d[i * n + j] = val;
            d[j * n + i] = val;
        }
    }
    d
}

#[test]
fn test_cluster_density_measure_separated() {
    // Two well-separated clusters should have high ClDM
    let n = 100;
    let mut pts = vec![0.0; n * 2];
    let mut labels = vec![0u32; n];

    // Cluster 0 around (-5, 0), Cluster 1 around (5, 0)
    let mut rng = Rng::new(42);
    for i in 0..n / 2 {
        pts[i * 2] = -5.0 + rng.normal() * 0.5;
        pts[i * 2 + 1] = rng.normal() * 0.5;
        labels[i] = 0;
    }
    for i in n / 2..n {
        pts[i * 2] = 5.0 + rng.normal() * 0.5;
        pts[i * 2 + 1] = rng.normal() * 0.5;
        labels[i] = 1;
    }

    let cldm = cluster_density_measure(&pts, &labels, n);
    assert!(
        cldm > 1.0,
        "Separated clusters should have high ClDM, got {cldm}"
    );
}

#[test]
fn test_cluster_density_measure_overlapping() {
    // Two overlapping clusters should have lower ClDM
    let n = 100;
    let mut rng = Rng::new(42);
    let pts: Vec<f64> = (0..n * 2).map(|_| rng.normal()).collect();
    let labels: Vec<u32> = (0..n).map(|i| u32::from(i >= n / 2)).collect();

    let cldm = cluster_density_measure(&pts, &labels, n);
    // Not a strict bound, but should be relatively low
    assert!(cldm >= 0.0);
}

#[test]
fn test_dunn_index_well_separated() {
    let n = 60;
    let mut d = vec![0.0; n * n];
    let mut labels = vec![0u32; n];

    // 3 clusters of 20 points each
    // Within cluster: small distances (0.1-0.5)
    // Between clusters: large distances (5-10)
    let mut rng = Rng::new(42);
    for c in 0..3 {
        for i in 0..20 {
            labels[c * 20 + i] = c as u32;
            for j in (i + 1)..20 {
                let dist = 0.1 + rng.uniform() * 0.4;
                d[(c * 20 + i) * n + (c * 20 + j)] = dist;
                d[(c * 20 + j) * n + (c * 20 + i)] = dist;
            }
        }
    }
    // Between-cluster distances
    for ci in 0..3 {
        for cj in (ci + 1)..3 {
            for i in 0..20 {
                for j in 0..20 {
                    let dist = 5.0 + rng.uniform() * 5.0;
                    d[(ci * 20 + i) * n + (cj * 20 + j)] = dist;
                    d[(cj * 20 + j) * n + (ci * 20 + i)] = dist;
                }
            }
        }
    }

    let di = dunn_index(&d, &labels, n);
    assert!(
        di > 1.0,
        "Well-separated clusters should have Dunn > 1, got {di}"
    );
}

// ---------------------------------------------------------------------------
// Normalized stress
// ---------------------------------------------------------------------------

#[test]
fn test_normalized_stress_zero_for_identical() {
    let d = make_distance_matrix(20, 42);
    let stress = normalized_stress(&d, &d, 20);
    assert!(
        stress.abs() < 1e-10,
        "Identical matrices should give 0 stress, got {stress}"
    );
}

#[test]
fn test_normalized_stress_positive_for_different() {
    let d1 = make_distance_matrix(20, 42);
    let d2 = make_distance_matrix(20, 99);
    let stress = normalized_stress(&d1, &d2, 20);
    assert!(
        stress > 0.0,
        "Different matrices should give positive stress"
    );
}

#[test]
fn test_normalized_stress_range() {
    let d1 = make_distance_matrix(30, 1);
    let d2 = make_distance_matrix(30, 2);
    let stress = normalized_stress(&d1, &d2, 30);
    assert!(
        (0.0..=1.0).contains(&stress),
        "Stress out of [0,1]: {stress}"
    );
}

// ---------------------------------------------------------------------------
// Neighborhood hit
// ---------------------------------------------------------------------------

#[test]
fn test_neighborhood_hit_perfect_separation() {
    let (pts, labels) = make_clustered_2d(3, 20, 0.3, 42);
    let n = pts.len() / 2;
    let d = euclidean_dist_2d(&pts, n);
    let nh = neighborhood_hit(&d, &labels, n, 7);
    assert!(
        nh > 0.95,
        "Well-separated clusters should have NH > 0.95, got {nh}"
    );
}

#[test]
fn test_neighborhood_hit_random_labels() {
    // Random labels on random points: NH should be around 1/num_classes
    let mut rng = Rng::new(42);
    let n = 120;
    let num_classes = 3usize;
    let pts: Vec<f64> = (0..n * 2).map(|_| rng.normal()).collect();
    let labels: Vec<u32> = (0..n).map(|i| (i % num_classes) as u32).collect();
    let d = euclidean_dist_2d(&pts, n);
    let nh = neighborhood_hit(&d, &labels, n, 7);
    // With random points + balanced labels, expected NH ≈ 1/3
    assert!(nh < 0.6, "Random labels should give low NH, got {nh}");
}

#[test]
fn test_neighborhood_hit_range() {
    let (pts, labels) = make_clustered_2d(2, 15, 1.0, 7);
    let n = pts.len() / 2;
    let d = euclidean_dist_2d(&pts, n);
    let nh = neighborhood_hit(&d, &labels, n, 5);
    assert!(
        (0.0..=1.0).contains(&nh),
        "Neighborhood hit out of [0,1]: {nh}"
    );
}

// ---------------------------------------------------------------------------
// Distance consistency
// ---------------------------------------------------------------------------

#[test]
fn test_distance_consistency_perfect_separation() {
    let (pts, labels) = make_clustered_2d(3, 20, 0.3, 42);
    let dsc = distance_consistency(&pts, &labels, pts.len() / 2);
    assert!(
        (dsc - 1.0).abs() < 1e-12,
        "clusters 20 apart with spread 0.3 must all sit nearest their own centroid, got {dsc}"
    );
}

#[test]
fn test_distance_consistency_random_labels() {
    // Random labels over one blob: the class centroids all land near the blob's
    // centre, so membership is close to a coin toss between them.
    let mut rng = Rng::new(42);
    let n = 120;
    let pts: Vec<f64> = (0..n * 2).map(|_| rng.normal()).collect();
    let labels: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
    let dsc = distance_consistency(&pts, &labels, n);
    assert!(dsc < 0.6, "random labels should give low DSC, got {dsc}");
}

#[test]
fn test_distance_consistency_range() {
    let (pts, labels) = make_clustered_2d(2, 15, 1.0, 7);
    let dsc = distance_consistency(&pts, &labels, pts.len() / 2);
    assert!((0.0..=1.0).contains(&dsc), "DSC out of [0,1]: {dsc}");
}

#[test]
fn test_distance_consistency_is_global_where_neighborhood_hit_is_local() {
    // Two classes as concentric rings sharing a centre. Neither interleaves —
    // every point's nearest neighbours are its own class, so NH is perfect —
    // but the two centroids coincide, so the classes occupy the same region of
    // the plane and DSC sees no separation. This is the case that motivates
    // measuring both, and the non-convexity DSC is documented to mishandle.
    let n_per = 60;
    let n = n_per * 2;
    let mut pts = vec![0.0f64; n * 2];
    let mut labels = vec![0u32; n];
    for i in 0..n_per {
        let theta = std::f64::consts::TAU * i as f64 / n_per as f64;
        for (ring, r) in [(0usize, 1.0f64), (1, 4.0)] {
            let idx = ring * n_per + i;
            pts[idx * 2] = r * theta.cos();
            pts[idx * 2 + 1] = r * theta.sin();
            labels[idx] = ring as u32;
        }
    }

    let d = euclidean_dist_2d(&pts, n);
    let nh = neighborhood_hit(&d, &labels, n, 5);
    let dsc = distance_consistency(&pts, &labels, n);
    assert!(
        nh > 0.99,
        "the rings do not interleave, so NH ≈ 1: got {nh}"
    );
    assert!(
        dsc < 0.55,
        "coincident centroids leave DSC near chance: got {dsc}"
    );
}

#[test]
fn test_distance_consistency_single_class_is_one() {
    // The minimum over an empty set of rival centroids is infinite, so every
    // point trivially qualifies — the same convention `neighborhood_hit` uses
    // for a degenerate input.
    let (pts, _) = make_clustered_2d(2, 10, 1.0, 3);
    let n = pts.len() / 2;
    let labels = vec![0u32; n];
    assert_eq!(distance_consistency(&pts, &labels, n), 1.0);
}

// ---------------------------------------------------------------------------
// Shepard goodness
// ---------------------------------------------------------------------------

#[test]
fn test_shepard_goodness_perfect() {
    let d = make_distance_matrix(25, 42);
    let sg = shepard_goodness(&d, &d, 25);
    assert!(
        (sg - 1.0).abs() < 1e-10,
        "Identical matrices should give shepard_goodness = 1, got {sg}"
    );
}

#[test]
fn test_shepard_goodness_range() {
    let d1 = make_distance_matrix(25, 42);
    let d2 = make_distance_matrix(25, 99);
    let sg = shepard_goodness(&d1, &d2, 25);
    assert!(
        (0.0..=1.0).contains(&sg),
        "Shepard goodness out of [0,1]: {sg}"
    );
}

#[test]
fn test_shepard_goodness_lower_for_uncorrelated() {
    // Reversed rank order should give poor Shepard goodness
    let n = 20;
    // Make d1 increasing, d2 decreasing (perfectly anti-correlated ranks)
    let mut d1 = vec![0.0f64; n * n];
    let mut d2 = vec![0.0f64; n * n];
    let mut val = 1.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            d1[i * n + j] = val;
            d1[j * n + i] = val;
            d2[i * n + j] = 1000.0 - val;
            d2[j * n + i] = 1000.0 - val;
            val += 1.0;
        }
    }
    let sg = shepard_goodness(&d1, &d2, n);
    // Anti-correlated ranks → raw r_s = -1, which (r_s + 1) / 2 maps to 0, the
    // bottom of the normalised scale.
    assert!(
        sg < 1e-10,
        "Anti-correlated distances should give shepard goodness 0, got {sg}"
    );
}

#[test]
fn test_shepard_goodness_collapsed_embedding_is_no_skill() {
    // Every pairwise distance identical: the embedding carries no rank
    // information whatsoever, so the score must be the no-skill value of the
    // normalised scale, 0.5 — the image of r_s = 0 under (r_s + 1) / 2, and
    // strictly worse than any embedding with real rank agreement.
    let n = 30;
    let d_high = make_distance_matrix(n, 7);
    let collapsed = vec![0.25f64; n * n];
    let sg = shepard_goodness(&d_high, &collapsed, n);
    assert!(
        (sg - 0.5).abs() < 1e-12,
        "Collapsed embedding should score 0.5, got {sg}"
    );
    // ...and it must not beat a genuinely correlated embedding.
    assert!(
        sg < shepard_goodness(&d_high, &d_high, n),
        "Collapsed embedding scored no worse than a perfect one"
    );
}

#[test]
fn test_shepard_goodness_ties_match_reference_spearman() {
    // Heavily tied integer distances on both sides (the wordnet_mammals shape:
    // BFS hop counts take a handful of distinct values). Compare against
    // r_s computed independently as Pearson on fractional ranks.
    let n = 24;
    let mut d1 = vec![0.0f64; n * n];
    let mut d2 = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let a = ((i + j) % 4) as f64 + 1.0;
            let b = ((i * j) % 3) as f64 + 1.0;
            d1[i * n + j] = a;
            d1[j * n + i] = a;
            d2[i * n + j] = b;
            d2[j * n + i] = b;
        }
    }
    let sg = shepard_goodness(&d1, &d2, n);

    let mut u1 = Vec::new();
    let mut u2 = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            u1.push(d1[i * n + j]);
            u2.push(d2[i * n + j]);
        }
    }
    let expected = (pearson_on_fractional_ranks(&u1, &u2) + 1.0) / 2.0;
    assert!(
        (sg - expected).abs() < 1e-10,
        "Tied inputs should give r_s = {expected}, got {sg}"
    );
}

#[test]
fn test_shepard_goodness_tie_order_is_irrelevant() {
    // Permuting points must not change the score. Under ordinal ranking it
    // does, because ties are then resolved by array position.
    let n = 20;
    let mut d1 = vec![0.0f64; n * n];
    let mut d2 = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let a = ((i + j) % 3) as f64;
            let b = ((i + 2 * j) % 5) as f64;
            d1[i * n + j] = a;
            d1[j * n + i] = a;
            d2[i * n + j] = b;
            d2[j * n + i] = b;
        }
    }
    let sg = shepard_goodness(&d1, &d2, n);

    // Reverse the point order in both matrices: same distance multiset, same
    // pairing, different array layout.
    let perm: Vec<usize> = (0..n).rev().collect();
    let mut p1 = vec![0.0f64; n * n];
    let mut p2 = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            p1[i * n + j] = d1[perm[i] * n + perm[j]];
            p2[i * n + j] = d2[perm[i] * n + perm[j]];
        }
    }
    let sg_perm = shepard_goodness(&p1, &p2, n);
    assert!(
        (sg - sg_perm).abs() < 1e-12,
        "Score changed under point relabelling: {sg} vs {sg_perm}"
    );
}

/// Reference `r_s`: the Pearson correlation coefficient of the rank variables
/// `R[X]`, `R[Y]`, with ties given fractional (average) ranks.
fn pearson_on_fractional_ranks(a: &[f64], b: &[f64]) -> f64 {
    fn fractional_ranks(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&x, &y| v[x].total_cmp(&v[y]));
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0;
            for &k in &idx[i..=j] {
                r[k] = avg;
            }
            i = j + 1;
        }
        r
    }
    let (ra, rb) = (fractional_ranks(a), fractional_ranks(b));
    let m = ra.len() as f64;
    let ma = ra.iter().sum::<f64>() / m;
    let mb = rb.iter().sum::<f64>() / m;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in ra.iter().zip(rb.iter()) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    cov / (va * vb).sqrt()
}

// ---------------------------------------------------------------------------
// Before vs after projection distinction
// ---------------------------------------------------------------------------

#[test]
fn test_euclidean_dist_2d_is_symmetric() {
    let (pts, _) = make_clustered_2d(2, 10, 1.0, 42);
    let n = pts.len() / 2;
    let d = euclidean_dist_2d(&pts, n);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (d[i * n + j] - d[j * n + i]).abs() < 1e-12,
                "Distance matrix not symmetric at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_before_after_projection_differ() {
    // Simulate before/after: use two different distance matrices for the
    // same set of points (analogous to manifold geodesic vs. projected 2D).
    // Metrics computed on them should generally differ, demonstrating that
    // both variants are worth storing.
    let (pts, labels) = make_clustered_2d(3, 15, 0.5, 42);
    let n = pts.len() / 2;

    // "before": distances in some ambient space (here: a scaled version)
    let d_before = euclidean_dist_2d(&pts, n);
    // "after": distances in a distorted 2D space (simulate projection distortion)
    let pts_distorted: Vec<f64> = pts
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v * 2.0 } else { v * 0.5 })
        .collect();
    let d_after = euclidean_dist_2d(&pts_distorted, n);

    let t_before = trustworthiness(&d_before, &d_before, n, 7);
    let _t_after = trustworthiness(&d_before, &d_after, n, 7);
    let nh_before = neighborhood_hit(&d_before, &labels, n, 7);
    let nh_after = neighborhood_hit(&d_after, &labels, n, 7);
    let sg_before = shepard_goodness(&d_before, &d_before, n);
    let sg_after = shepard_goodness(&d_before, &d_after, n);

    // Before (self-comparison) should be perfect / better
    assert!(
        (t_before - 1.0).abs() < 1e-10,
        "Trustworthiness before should be 1.0, got {t_before}"
    );
    assert!(
        sg_before > sg_after,
        "Shepard goodness before ({sg_before}) should exceed after ({sg_after})"
    );
    // NH may be high in both cases since clusters are well-separated
    let _ = nh_before;
    let _ = nh_after;
}

#[test]
fn test_davies_bouldin_separated() {
    // Create well-separated clusters with a distance matrix
    let n = 40;
    let mut d = vec![0.0; n * n];
    let mut labels = vec![0u32; n];
    let mut rng = Rng::new(42);

    for c in 0..2 {
        for i in 0..20 {
            labels[c * 20 + i] = c as u32;
            for j in (i + 1)..20 {
                let dist = 0.1 + rng.uniform() * 0.3;
                d[(c * 20 + i) * n + (c * 20 + j)] = dist;
                d[(c * 20 + j) * n + (c * 20 + i)] = dist;
            }
        }
    }
    for i in 0..20 {
        for j in 0..20 {
            let dist = 8.0 + rng.uniform() * 2.0;
            d[i * n + (20 + j)] = dist;
            d[(20 + j) * n + i] = dist;
        }
    }

    let db = davies_bouldin(&d, &labels, n);
    assert!(db > 0.0, "DB should be positive");
    assert!(
        db < 1.0,
        "Well-separated clusters should have low DB, got {db}"
    );
}

// ---------------------------------------------------------------------------
// MetricValues::compute
// ---------------------------------------------------------------------------

/// A metric that must have been measured.
fn measured(m: &MetricValues, metric: Metric) -> f64 {
    m.get(metric)
        .unwrap_or_else(|| panic!("{metric} was not measured: {:?}", m.reading(metric)))
}

/// A flat context over 2-D points, which is what these tests need: at
/// curvature 0 the "manifold" is the plane itself, so the manifold and
/// projected readings are the same geometry and any difference between them is
/// the projection's rescaling rather than curvature.
fn flat_context<'a>(
    high_dim_dist: &'a [f64],
    pts_2d: &'a [f64],
    labels: Option<&'a [u32]>,
    k: usize,
) -> EmbeddingContext<'a> {
    let n = pts_2d.len() / 2;
    EmbeddingContext::new(
        high_dim_dist,
        pts_2d,
        labels,
        n,
        2,
        0.0,
        k,
        SphericalProjection::AzimuthalEquidistant,
    )
}

#[test]
fn test_compute_without_labels_leaves_label_metrics_absent() {
    let (pts_2d, _) = make_clustered_2d(2, 12, 1.0, 42);
    let d = make_distance_matrix(pts_2d.len() / 2, 42);
    let m = MetricValues::compute(&flat_context(&d, &pts_2d, None, 5));

    for metric in [
        NEIGHBORHOOD_HIT,
        NEIGHBORHOOD_HIT_MANIFOLD,
        CLUSTER_DENSITY_MEASURE,
        DAVIES_BOULDIN_RATIO,
        DUNN_INDEX,
        DISTANCE_CONSISTENCY,
    ] {
        assert_eq!(m.get(metric), None, "{} needs labels", metric.name());
    }
    // The label-free metrics are unaffected.
    assert!(m.get(TRUSTWORTHINESS).is_some());
    assert!(m.get(NORMALIZED_STRESS).is_some());
}

#[test]
fn test_compute_with_labels_scores_the_label_metrics() {
    let (pts_2d, labels) = make_clustered_2d(3, 10, 0.5, 42);
    let d = euclidean_dist_2d(&pts_2d, pts_2d.len() / 2);
    let m = MetricValues::compute(&flat_context(&d, &pts_2d, Some(&labels), 7));

    for metric in [
        NEIGHBORHOOD_HIT,
        NEIGHBORHOOD_HIT_MANIFOLD,
        CLUSTER_DENSITY_MEASURE,
        DAVIES_BOULDIN_RATIO,
        DUNN_INDEX,
        DISTANCE_CONSISTENCY,
    ] {
        assert!(m.get(metric).is_some(), "{} unscored", metric.name());
    }
}

#[test]
fn test_compute_perfect_embedding_scores() {
    // High-dimensional distances *are* the embedding's own distances, so the
    // manifold reading is a self-comparison and every metric must be at its
    // best: 1.0 for the maximised ones, 0 for stress.
    let (pts_2d, _) = make_clustered_2d(4, 8, 1.0, 42);
    let n = pts_2d.len() / 2;
    let d = euclidean_dist_2d(&pts_2d, n);
    let m = MetricValues::compute(&flat_context(&d, &pts_2d, None, 5));

    assert!(
        (measured(&m, TRUSTWORTHINESS_MANIFOLD) - 1.0).abs() < 1e-10,
        "trustworthiness_manifold {}",
        measured(&m, TRUSTWORTHINESS_MANIFOLD)
    );
    assert!(
        (measured(&m, CONTINUITY_MANIFOLD) - 1.0).abs() < 1e-10,
        "continuity_manifold {}",
        measured(&m, CONTINUITY_MANIFOLD)
    );
    assert!(
        measured(&m, NORMALIZED_STRESS_MANIFOLD).abs() < 1e-10,
        "normalized_stress_manifold {}",
        measured(&m, NORMALIZED_STRESS_MANIFOLD)
    );
    assert!(
        (measured(&m, SHEPARD_GOODNESS_MANIFOLD) - 1.0).abs() < 1e-10,
        "shepard_goodness_manifold {}",
        measured(&m, SHEPARD_GOODNESS_MANIFOLD)
    );
}

#[test]
fn test_compute_all_values_in_range() {
    let (pts_2d, labels) = make_clustered_2d(3, 10, 0.5, 99);
    let d_high = make_distance_matrix(pts_2d.len() / 2, 1);
    let m = MetricValues::compute(&flat_context(&d_high, &pts_2d, Some(&labels), 5));

    for metric in [
        TRUSTWORTHINESS,
        TRUSTWORTHINESS_MANIFOLD,
        CONTINUITY,
        CONTINUITY_MANIFOLD,
        SHEPARD_GOODNESS,
        SHEPARD_GOODNESS_MANIFOLD,
        NEIGHBORHOOD_HIT,
        NEIGHBORHOOD_HIT_MANIFOLD,
    ] {
        let v = m
            .get(metric)
            .unwrap_or_else(|| panic!("{} absent", metric.name()));
        assert!(
            (0.0..=1.0).contains(&v),
            "{} = {v} is outside [0, 1], which every objective is assumed to be",
            metric.name()
        );
    }
    assert!(measured(&m, NORMALIZED_STRESS) >= 0.0);
    assert!(measured(&m, NORMALIZED_STRESS_MANIFOLD) >= 0.0);
}

/// The spread diagnostics are measured from the same context as the metrics,
/// but they are not metrics — they have no direction, no family and no
/// objective status, which is why they are their own type.
#[test]
fn test_spread_diagnostics_measure_the_configurations_extent() {
    let (pts_2d, _) = make_clustered_2d(3, 10, 1.0, 7);
    let d = euclidean_dist_2d(&pts_2d, pts_2d.len() / 2);
    let spread = SpreadDiagnostics::compute(&flat_context(&d, &pts_2d, None, 5));

    let r_max = spread.r_max().expect("r_max");
    let r_rms = spread.r_rms().expect("r_rms");
    let r_gyration = spread.r_gyration().expect("r_gyration");
    assert!(r_max >= r_rms, "r_max {r_max} < r_rms {r_rms}");
    assert!(
        r_gyration > 0.0,
        "a spread-out configuration has nonzero gyration"
    );
}

/// `r_gyration` *is* the RMS distance to the centroid in flat space, so on a
/// Euclidean context it must agree with `r_rms` measured from an origin the
/// data is centred on. They separate only under curvature.
#[test]
fn test_a_collapsed_configuration_has_no_spread() {
    let pts_2d = vec![0.0; 24];
    let d = euclidean_dist_2d(&pts_2d, 12);
    let spread = SpreadDiagnostics::compute(&flat_context(&d, &pts_2d, None, 3));

    assert_eq!(spread.r_max(), Some(0.0));
    assert_eq!(spread.r_rms(), Some(0.0));
    assert_eq!(spread.r_gyration(), Some(0.0));
}

/// Absent is absent: nothing measured reads back as `None` rather than as a
/// zero that would look like a genuinely collapsed embedding.
#[test]
fn test_missing_spread_is_absent_not_zero() {
    let missing = SpreadDiagnostics::MISSING;
    assert_eq!(missing.r_max(), None);
    assert_eq!(missing.r_rms(), None);
    assert_eq!(missing.r_gyration(), None);
}

#[test]
fn test_normalized_stress_scale_invariant() {
    // SNS must be invariant to uniform scaling of the embedded distances.
    // If it is not, the manifold and 2D variants will differ for Euclidean
    // embeddings where project_to_2d rescales coords for display.
    let n = 20;
    let d_high = make_distance_matrix(n, 1);
    let d_embed = make_distance_matrix(n, 2);
    let scale = 7.3_f64;
    let d_embed_scaled: Vec<f64> = d_embed.iter().map(|&v| v * scale).collect();
    let s1 = normalized_stress(&d_high, &d_embed, n);
    let s2 = normalized_stress(&d_high, &d_embed_scaled, n);
    assert!(
        (s1 - s2).abs() < 1e-10,
        "normalized_stress must be scale-invariant: {s1} != {s2}"
    );
}

// ---------------------------------------------------------------------------
// The divergence gate
// ---------------------------------------------------------------------------

/// A context whose embedding blew up: one coordinate is non-finite, so every
/// distance involving that point is `inf` or `NaN`.
fn diverged_context(pts_2d: &mut [f64], high_dim: &[f64]) -> Vec<f64> {
    pts_2d[0] = f64::INFINITY;
    high_dim.to_vec()
}

/// **This is the assertion the gate exists for.**
///
/// On a diverged embedding, only the four metrics that *sum* distances notice.
/// The ones that *compare* fall into their degenerate branches (`f64::max`
/// ignores NaN; `if d > max_intra` is false for NaN) and report `0.0`, and the
/// ones that *rank* sort NaN to a defined position via `total_cmp` and return a
/// confident, plausible score — `trustworthiness` reads ~0.93 on a blown-up
/// embedding in `results/`, and `metrics_to_vec` then hands the GP a 0.93.
///
/// Every one of them must now report `Diverged` instead.
#[test]
fn a_diverged_embedding_measures_nothing() {
    let (mut pts_2d, labels) = make_clustered_2d(3, 10, 1.0, 42);
    let n = pts_2d.len() / 2;
    let high_dim = make_distance_matrix(n, 42);
    let high_dim = diverged_context(&mut pts_2d, &high_dim);

    let m = MetricValues::compute(&flat_context(&high_dim, &pts_2d, Some(&labels), 5));

    for metric in fitting_core::metrics::ALL {
        assert_eq!(
            m.reading(*metric),
            MetricValue::Diverged,
            "{metric} reported {:?} from a diverged embedding",
            m.reading(*metric)
        );
        assert_eq!(m.get(*metric), None);
    }

    // ...and the gate is load-bearing, not belt-and-braces: called directly on
    // the same distances, trustworthiness still returns a finite, respectable
    // number. That is what every pre-gate sweep recorded, and what
    // `metrics_to_vec` passed to the GP as a real score.
    let dist_2d = euclidean_dist_2d(&pts_2d, n);
    let ungated = trustworthiness(&high_dim, &dist_2d, n, 5);
    assert!(
        ungated.is_finite() && (0.0..=1.0).contains(&ungated),
        "the gate would be pointless if the raw function already failed here, \
         but it returned {ungated}"
    );
}

/// The same for the spread diagnostics, where `r_max` is the one that lied
/// loudest: `fold(0.0, f64::max)` ignores NaN, so a blown-up embedding used to
/// report a measured-looking `r_max: 0.0` beside `r_rms: null`.
#[test]
fn a_diverged_embedding_has_no_measurable_spread() {
    let (mut pts_2d, _) = make_clustered_2d(3, 10, 1.0, 42);
    let n = pts_2d.len() / 2;
    let high_dim = make_distance_matrix(n, 42);
    let high_dim = diverged_context(&mut pts_2d, &high_dim);

    let spread = SpreadDiagnostics::compute(&flat_context(&high_dim, &pts_2d, None, 5));
    assert_eq!(spread.r_max(), None, "r_max reported a number from garbage");
    assert_eq!(spread.r_rms(), None);
    assert_eq!(spread.r_gyration(), None);
}

/// The gate must not fire on a sound embedding, or every trial becomes
/// `Diverged` and the sweep measures nothing at all.
#[test]
fn a_sound_embedding_is_not_gated() {
    let (pts_2d, labels) = make_clustered_2d(3, 10, 1.0, 42);
    let n = pts_2d.len() / 2;
    let high_dim = make_distance_matrix(n, 42);
    let m = MetricValues::compute(&flat_context(&high_dim, &pts_2d, Some(&labels), 5));

    for metric in fitting_core::metrics::ALL {
        assert!(
            m.reading(*metric).is_measured(),
            "{metric} was gated on a sound embedding: {:?}",
            m.reading(*metric)
        );
    }
}

/// Absence keeps its reason: no labels is `NotApplicable`, which is a different
/// fact from a diverged reading even though both serialise as `null`.
#[test]
fn no_labels_is_not_applicable_rather_than_diverged() {
    let (pts_2d, _) = make_clustered_2d(3, 10, 1.0, 42);
    let n = pts_2d.len() / 2;
    let high_dim = make_distance_matrix(n, 42);
    let m = MetricValues::compute(&flat_context(&high_dim, &pts_2d, None, 5));

    assert_eq!(m.reading(NEIGHBORHOOD_HIT), MetricValue::NotApplicable);
    assert_eq!(m.reading(DUNN_INDEX), MetricValue::NotApplicable);
    assert!(m.reading(TRUSTWORTHINESS).is_measured());
}
