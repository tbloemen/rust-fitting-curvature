//! Tests for embedding quality metrics.
//! Ported from Python test/test_metrics.py

use fitting_core::metrics::*;
use fitting_core::synthetic_data::Rng;

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
fn test_knn_overlap_perfect() {
    // If both distance matrices are identical, knn_overlap should be 1.0
    let d = make_distance_matrix(30, 42);
    let overlap = knn_overlap(&d, &d, 30, 5);
    assert!(
        (overlap - 1.0).abs() < 1e-10,
        "Perfect overlap should be 1.0, got {overlap}"
    );
}

#[test]
fn test_knn_overlap_range() {
    let d1 = make_distance_matrix(30, 42);
    let d2 = make_distance_matrix(30, 99);
    let overlap = knn_overlap(&d1, &d2, 30, 5);
    assert!(
        (0.0..=1.0).contains(&overlap),
        "Overlap out of range: {overlap}"
    );
}

#[test]
fn test_geodesic_distortion_zero_for_identical() {
    let d = make_distance_matrix(20, 42);
    let distortion = geodesic_distortion_gu2019(&d, &d, 20);
    assert!(
        distortion.abs() < 1e-10,
        "Distortion should be 0 for identical matrices, got {distortion}"
    );
}

#[test]
fn test_geodesic_distortion_mse_zero_for_identical() {
    let d = make_distance_matrix(20, 42);
    let distortion = geodesic_distortion_mse(&d, &d, 20);
    assert!(
        distortion.abs() < 1e-10,
        "MSE distortion should be 0 for identical, got {distortion}"
    );
}

#[test]
fn test_geodesic_distortion_positive_for_different() {
    let d1 = make_distance_matrix(20, 42);
    let d2 = make_distance_matrix(20, 99);
    let distortion = geodesic_distortion_gu2019(&d1, &d2, 20);
    assert!(
        distortion > 0.0,
        "Distortion should be positive for different matrices"
    );
}

#[test]
fn test_radial_distribution() {
    // Points at unit circle should have low CV (uniform radii)
    let n = 100;
    let mut pts = vec![0.0; n * 2];
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        pts[i * 2] = angle.cos();
        pts[i * 2 + 1] = angle.sin();
    }
    let cv = radial_distribution(&pts, n);
    assert!(
        cv < 0.01,
        "Points on circle should have near-zero CV, got {cv}"
    );
}

#[test]
fn test_radial_distribution_spread() {
    let mut rng = Rng::new(42);
    let n = 200;
    let pts: Vec<f64> = (0..n * 2).map(|_| rng.normal()).collect();
    let cv = radial_distribution(&pts, n);
    // Gaussian points should have moderate CV
    assert!(cv > 0.0 && cv < 2.0, "Unexpected CV: {cv}");
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
    let labels: Vec<u32> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();

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
