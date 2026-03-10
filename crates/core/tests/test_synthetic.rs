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
    let data = generate_uniform_hyperbolic(50, 42);
    assert_eq!(data.n_points, 50);
    assert_eq!(data.ambient_dim, 3);
    assert_eq!(data.distances.len(), 50 * 50);
}

#[test]
fn test_hyperbolic_points_on_hyperboloid() {
    let data = generate_uniform_hyperbolic(50, 42);
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
    let data = generate_uniform_hyperbolic(30, 42);
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
    let data = generate_uniform_hyperbolic(30, 42);
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
