//! Tests for synthetic dataset generators.
//! Ported from Python test/test_synthetic.py

use fitting_core::synthetic_data::*;

// ---------------------------------------------------------------------------
// Euclidean generators
// ---------------------------------------------------------------------------

#[test]
fn test_uniform_grid_shape() {
    let data = generate_uniform_grid(100, 42);
    assert_eq!(data.n_points, 100);
    assert_eq!(data.ambient_dim, 2);
    assert_eq!(data.x.len(), 100 * 2);
    assert_eq!(data.labels.len(), 100);
    assert_eq!(data.distances.len(), 100 * 100);
}

#[test]
fn test_uniform_grid_labels() {
    let data = generate_uniform_grid(100, 42);
    for &l in &data.labels {
        assert!(l <= 3, "Label out of range: {l}");
    }
}

#[test]
fn test_gaussian_blob_shape() {
    let data = generate_gaussian_blob(100, 42);
    assert_eq!(data.n_points, 100);
    assert_eq!(data.ambient_dim, 2);
    assert_eq!(data.distances.len(), 100 * 100);
}

#[test]
fn test_gaussian_blob_labels() {
    let data = generate_gaussian_blob(100, 42);
    for &l in &data.labels {
        assert!(l <= 1);
    }
}

#[test]
fn test_concentric_circles_shape() {
    let data = generate_concentric_circles(100, 42);
    assert_eq!(data.n_points, 100);
    assert_eq!(data.ambient_dim, 2);
    assert_eq!(data.distances.len(), 100 * 100);
}

#[test]
fn test_concentric_circles_labels() {
    let data = generate_concentric_circles(100, 42);
    let has_0 = data.labels.contains(&0);
    let has_1 = data.labels.contains(&1);
    assert!(has_0 && has_1, "Should have labels 0 and 1");
}

// ---------------------------------------------------------------------------
// Spherical generators
// ---------------------------------------------------------------------------

#[test]
fn test_uniform_sphere_shape() {
    let data = generate_uniform_sphere(50, 42);
    assert_eq!(data.n_points, 50);
    assert_eq!(data.ambient_dim, 3);
    assert_eq!(data.distances.len(), 50 * 50);
}

#[test]
fn test_uniform_sphere_on_unit_sphere() {
    let data = generate_uniform_sphere(50, 42);
    for i in 0..50 {
        let mut norm_sq = 0.0;
        for d in 0..3 {
            norm_sq += data.x[i * 3 + d].powi(2);
        }
        assert!(
            (norm_sq - 1.0).abs() < 1e-5,
            "Point {i} not on unit sphere: norm_sq={norm_sq}"
        );
    }
}

#[test]
fn test_spherical_distances_match_arccos() {
    let data = generate_uniform_sphere(30, 42);
    let d = &data.distances;
    let n = 30;

    for i in 0..n {
        for j in (i + 1)..n {
            let mut dot = 0.0;
            for k in 0..3 {
                dot += data.x[i * 3 + k] * data.x[j * 3 + k];
            }
            let expected = dot.clamp(-1.0, 1.0).acos();
            let diff = (d[i * n + j] - expected).abs();
            assert!(
                diff < 1e-5,
                "Distance mismatch at ({i},{j}): {} vs {expected}",
                d[i * n + j]
            );
        }
    }
}

#[test]
fn test_spherical_distances_symmetric() {
    let data = generate_uniform_sphere(30, 42);
    let d = &data.distances;
    let n = 30;
    for i in 0..n {
        for j in 0..n {
            let diff = (d[i * n + j] - d[j * n + i]).abs();
            assert!(diff < 1e-5);
        }
    }
}

#[test]
fn test_spherical_distances_diagonal_zero() {
    let data = generate_uniform_sphere(30, 42);
    let d = &data.distances;
    for i in 0..30 {
        assert!(d[i * 30 + i].abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// Hyperbolic generators
// ---------------------------------------------------------------------------

#[test]
fn test_uniform_hyperbolic_shape() {
    let data = generate_uniform_hyperbolic(50, 42, 3.0);
    assert_eq!(data.n_points, 50);
    assert_eq!(data.ambient_dim, 3);
    assert_eq!(data.distances.len(), 50 * 50);
}

#[test]
fn test_hyperbolic_points_on_hyperboloid() {
    let data = generate_uniform_hyperbolic(50, 42, 3.0);
    for i in 0..50 {
        let x0 = data.x[i * 3];
        let x1 = data.x[i * 3 + 1];
        let x2 = data.x[i * 3 + 2];
        let lorentz = -x0 * x0 + x1 * x1 + x2 * x2;
        assert!(
            (lorentz + 1.0).abs() < 1e-4,
            "Point {i} not on hyperboloid: lorentz={lorentz}"
        );
        assert!(x0 > 0.0, "Point {i} has x0 <= 0");
    }
}

#[test]
fn test_hyperbolic_distances_match_acosh() {
    let data = generate_uniform_hyperbolic(30, 42, 3.0);
    let d = &data.distances;
    let n = 30;

    for i in 0..n {
        for j in (i + 1)..n {
            // Lorentzian inner: -x0*y0 + x1*y1 + x2*y2
            let inner = -data.x[i * 3] * data.x[j * 3]
                + data.x[i * 3 + 1] * data.x[j * 3 + 1]
                + data.x[i * 3 + 2] * data.x[j * 3 + 2];
            let expected = (-inner).max(1.0).acosh();
            let diff = (d[i * n + j] - expected).abs();
            assert!(diff < 1e-4, "Distance mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_hyperbolic_distances_symmetric() {
    let data = generate_uniform_hyperbolic(30, 42, 3.0);
    let d = &data.distances;
    let n = 30;
    for i in 0..n {
        for j in 0..n {
            let diff = (d[i * n + j] - d[j * n + i]).abs();
            assert!(diff < 1e-5);
        }
    }
}

#[test]
fn test_tree_structured_shape() {
    let data = generate_tree_structured(50, 42);
    assert_eq!(data.n_points, 50);
    assert_eq!(data.ambient_dim, 3);
    assert_eq!(data.distances.len(), 50 * 50);
}

#[test]
fn test_tree_on_hyperboloid() {
    let data = generate_tree_structured(50, 42);
    for i in 0..50 {
        let x0 = data.x[i * 3];
        let x1 = data.x[i * 3 + 1];
        let x2 = data.x[i * 3 + 2];
        let lorentz = -x0 * x0 + x1 * x1 + x2 * x2;
        assert!((lorentz + 1.0).abs() < 1e-4, "Point {i}: lorentz={lorentz}");
    }
}

#[test]
fn test_hyperbolic_shells_shape() {
    let data = generate_hyperbolic_shells(60, 42);
    assert_eq!(data.n_points, 60);
    assert_eq!(data.ambient_dim, 3);
    assert_eq!(data.distances.len(), 60 * 60);
}

#[test]
fn test_shells_on_hyperboloid() {
    let data = generate_hyperbolic_shells(60, 42);
    for i in 0..60 {
        let x0 = data.x[i * 3];
        let x1 = data.x[i * 3 + 1];
        let x2 = data.x[i * 3 + 2];
        let lorentz = -x0 * x0 + x1 * x1 + x2 * x2;
        assert!((lorentz + 1.0).abs() < 1e-4, "Point {i}: lorentz={lorentz}");
    }
}

// ---------------------------------------------------------------------------
// HD spherical generators
// ---------------------------------------------------------------------------

#[test]
fn test_hd_sphere_shape() {
    let data = generate_hd_sphere(100, 10, 42);
    assert_eq!(data.n_points, 100);
    assert_eq!(data.ambient_dim, 10);
    assert_eq!(data.x.len(), 100 * 10);
    assert_eq!(data.labels.len(), 100);
    assert_eq!(data.distances.len(), 100 * 100);
}

#[test]
fn test_hd_sphere_on_unit_sphere() {
    let data = generate_hd_sphere(50, 10, 42);
    for i in 0..50 {
        let norm_sq: f64 = (0..10).map(|k| data.x[i * 10 + k].powi(2)).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "Point {i} not on unit sphere: norm_sq={norm_sq}"
        );
    }
}

#[test]
fn test_hd_sphere_distances_match_arccos() {
    let data = generate_hd_sphere(30, 8, 42);
    let n = 30;
    let dim = 8;
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f64 = (0..dim)
                .map(|k| data.x[i * dim + k] * data.x[j * dim + k])
                .sum();
            let expected = dot.clamp(-1.0, 1.0).acos();
            let diff = (data.distances[i * n + j] - expected).abs();
            assert!(diff < 1e-10, "Distance mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_hd_sphere_distances_symmetric() {
    let data = generate_hd_sphere(30, 5, 42);
    let n = 30;
    for i in 0..n {
        for j in 0..n {
            let diff = (data.distances[i * n + j] - data.distances[j * n + i]).abs();
            assert!(diff < 1e-10);
        }
    }
}

#[test]
fn test_hd_sphere_labels_binary() {
    let data = generate_hd_sphere(100, 10, 42);
    for &l in &data.labels {
        assert!(l <= 1, "Label out of range: {l}");
    }
    assert!(data.labels.contains(&0) && data.labels.contains(&1));
}

#[test]
fn test_hd_sphere_dim3_matches_structure() {
    // With dim=3 we should get S^2, same as generate_uniform_sphere in structure
    let data = generate_hd_sphere(50, 3, 42);
    assert_eq!(data.ambient_dim, 3);
    for i in 0..50 {
        let norm_sq: f64 = (0..3).map(|k| data.x[i * 3 + k].powi(2)).sum();
        assert!((norm_sq - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_hd_antipodal_clusters_shape() {
    let data = generate_hd_antipodal_clusters(100, 10, 42);
    assert_eq!(data.n_points, 100);
    assert_eq!(data.ambient_dim, 10);
    assert_eq!(data.distances.len(), 100 * 100);
}

#[test]
fn test_hd_antipodal_clusters_on_unit_sphere() {
    let data = generate_hd_antipodal_clusters(50, 10, 42);
    for i in 0..50 {
        let norm_sq: f64 = (0..10).map(|k| data.x[i * 10 + k].powi(2)).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "Point {i} not on unit sphere"
        );
    }
}

#[test]
fn test_hd_antipodal_clusters_separation() {
    // North cluster (label 0) should have positive x[0]; south (label 1) negative x[0].
    // With kappa=5 this holds for the vast majority of points.
    let data = generate_hd_antipodal_clusters(200, 10, 42);
    let mut north_pos = 0usize;
    let mut south_neg = 0usize;
    for i in 0..200 {
        if data.labels[i] == 0 && data.x[i * 10] > 0.0 {
            north_pos += 1;
        }
        if data.labels[i] == 1 && data.x[i * 10] < 0.0 {
            south_neg += 1;
        }
    }
    // At kappa=5 in 10D the concentration is strong; expect >90% in correct hemisphere.
    assert!(
        north_pos > 90,
        "North cluster not concentrated: {north_pos}/100"
    );
    assert!(
        south_neg > 90,
        "South cluster not concentrated: {south_neg}/100"
    );
}

#[test]
fn test_hd_antipodal_clusters_labels_binary() {
    let data = generate_hd_antipodal_clusters(100, 10, 42);
    for &l in &data.labels {
        assert!(l <= 1);
    }
}

// ---------------------------------------------------------------------------
// HD hyperbolic generators
// ---------------------------------------------------------------------------

/// Check the hyperboloid constraint -x0^2 + x1^2 + ... + x(dim-1)^2 = -1
fn check_on_hyperboloid(x: &[f64], n: usize, dim: usize, tol: f64) {
    for i in 0..n {
        let lorentz = -x[i * dim].powi(2) + (1..dim).map(|k| x[i * dim + k].powi(2)).sum::<f64>();
        assert!(
            (lorentz + 1.0).abs() < tol,
            "Point {i} not on hyperboloid: lorentz={lorentz:.6}"
        );
        assert!(x[i * dim] > 0.0, "Point {i} has x0 <= 0");
    }
}

#[test]
fn test_hd_tree_shape() {
    let data = generate_hd_tree(100, 10, 42);
    assert_eq!(data.n_points, 100);
    assert_eq!(data.ambient_dim, 10);
    assert_eq!(data.x.len(), 100 * 10);
    assert_eq!(data.labels.len(), 100);
    assert_eq!(data.distances.len(), 100 * 100);
}

#[test]
fn test_hd_tree_on_hyperboloid() {
    let data = generate_hd_tree(80, 10, 42);
    check_on_hyperboloid(&data.x, 80, 10, 1e-6);
}

#[test]
fn test_hd_tree_distances_match_acosh() {
    let data = generate_hd_tree(30, 6, 42);
    let n = 30;
    let dim = 6;
    for i in 0..n {
        for j in (i + 1)..n {
            let inner = -data.x[i * dim] * data.x[j * dim]
                + (1..dim)
                    .map(|k| data.x[i * dim + k] * data.x[j * dim + k])
                    .sum::<f64>();
            let expected = (-inner).max(1.0).acosh();
            let diff = (data.distances[i * n + j] - expected).abs();
            assert!(diff < 1e-6, "Distance mismatch at ({i},{j})");
        }
    }
}

#[test]
fn test_hd_tree_distances_symmetric() {
    let data = generate_hd_tree(30, 8, 42);
    let n = 30;
    for i in 0..n {
        for j in 0..n {
            let diff = (data.distances[i * n + j] - data.distances[j * n + i]).abs();
            assert!(diff < 1e-10);
        }
    }
}

#[test]
fn test_hd_tree_labels_range() {
    let data = generate_hd_tree(100, 10, 42);
    for &l in &data.labels {
        assert!(l <= 4, "Label out of range: {l}");
    }
}

#[test]
fn test_hd_tree_dim3_on_hyperboloid() {
    // dim=3 reduces to H^2 in R^3, same as generate_tree_structured
    let data = generate_hd_tree(50, 3, 42);
    assert_eq!(data.ambient_dim, 3);
    check_on_hyperboloid(&data.x, 50, 3, 1e-6);
}

#[test]
fn test_hd_hyperbolic_shells_shape() {
    let data = generate_hd_hyperbolic_shells(90, 10, 42);
    assert_eq!(data.n_points, 90);
    assert_eq!(data.ambient_dim, 10);
    assert_eq!(data.x.len(), 90 * 10);
    assert_eq!(data.labels.len(), 90);
    assert_eq!(data.distances.len(), 90 * 90);
}

#[test]
fn test_hd_hyperbolic_shells_on_hyperboloid() {
    let data = generate_hd_hyperbolic_shells(90, 10, 42);
    check_on_hyperboloid(&data.x, 90, 10, 1e-6);
}

#[test]
fn test_hd_hyperbolic_shells_labels() {
    let data = generate_hd_hyperbolic_shells(90, 10, 42);
    for &l in &data.labels {
        assert!(l <= 2, "Label out of range: {l}");
    }
    assert!(data.labels.contains(&0));
    assert!(data.labels.contains(&1));
    assert!(data.labels.contains(&2));
}

#[test]
fn test_hd_hyperbolic_shells_distances_symmetric() {
    let data = generate_hd_hyperbolic_shells(30, 6, 42);
    let n = 30;
    for i in 0..n {
        for j in 0..n {
            let diff = (data.distances[i * n + j] - data.distances[j * n + i]).abs();
            assert!(diff < 1e-10);
        }
    }
}

#[test]
fn test_hd_hyperbolic_shells_radial_ordering() {
    // Shell 2 (outer) should have larger mean distance from origin than shell 0 (inner).
    let data = generate_hd_hyperbolic_shells(90, 10, 42);
    let dim = 10;
    // Hyperbolic distance from origin = acosh(x0) for a point on the hyperboloid
    let mean_dist = |shell: u32| {
        let pts: Vec<f64> = (0..90)
            .filter(|&i| data.labels[i] == shell)
            .map(|i| data.x[i * dim].acosh())
            .collect();
        pts.iter().sum::<f64>() / pts.len() as f64
    };
    let d0 = mean_dist(0);
    let d1 = mean_dist(1);
    let d2 = mean_dist(2);
    assert!(
        d0 < d1,
        "Shell 0 should be closer than shell 1: {d0:.3} vs {d1:.3}"
    );
    assert!(
        d1 < d2,
        "Shell 1 should be closer than shell 2: {d1:.3} vs {d2:.3}"
    );
}

#[test]
fn test_hd_generators_deterministic() {
    let a = generate_hd_sphere(20, 5, 99);
    let b = generate_hd_sphere(20, 5, 99);
    assert_eq!(a.x, b.x);

    let a = generate_hd_tree(20, 5, 99);
    let b = generate_hd_tree(20, 5, 99);
    assert_eq!(a.x, b.x);
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

#[test]
fn test_load_all_datasets() {
    for &name in DATASET_NAMES {
        let result = load_synthetic(name, 50, 42);
        assert!(result.is_ok(), "Failed to load {name}: {:?}", result.err());
        let data = result.unwrap();
        assert_eq!(data.n_points, 50);
        assert_eq!(data.labels.len(), 50);
    }
}

#[test]
fn test_load_unknown_dataset() {
    let result = load_synthetic("nonexistent", 50, 42);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

#[test]
fn test_labels_non_negative() {
    for &name in DATASET_NAMES {
        let data = load_synthetic(name, 100, 42).unwrap();
        for &l in &data.labels {
            assert!(l < 100, "Unreasonable label value: {l}");
        }
    }
}
