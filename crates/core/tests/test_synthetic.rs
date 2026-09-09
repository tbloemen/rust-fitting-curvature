//! Tests for synthetic dataset generators.
//! Ported from Python `test/test_synthetic.py`

use fitting_core::cast::count_to_f64;
use fitting_core::graph::RootedTree;
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
        pts.iter().sum::<f64>() / count_to_f64(pts.len())
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
// Tree metric (hierarchy benchmark)
// ---------------------------------------------------------------------------

#[test]
fn test_tree_graph_shape() {
    let data = generate_tree_graph(100, 2, 3);
    assert_eq!(data.n_points, 100);
    // Graph data carries no coordinates, exactly like `wordnet_mammals`.
    assert_eq!(data.ambient_dim, 0);
    assert!(data.x.is_empty());
    assert_eq!(data.labels.len(), 100);
    assert_eq!(data.distances.len(), 100 * 100);
}

#[test]
fn test_tree_graph_is_a_tree_metric() {
    let n = 127;
    let tree = RootedTree::complete(n, 2);
    let data = generate_tree_graph(n, 2, 3);

    for u in 0..n {
        for v in 0..n {
            let d = data.distances[u * n + v];
            // Integral hop counts, and equal to the closed form
            // depth(u) + depth(v) - 2*depth(lca(u,v)).
            assert!(
                (d - d.round()).abs() < 1e-12,
                "d({u},{v}) = {d} is not integral"
            );
            assert!(
                (d - f64::from(tree.hop_distance(u, v))).abs() < 1e-12,
                "d({u},{v}) = {d} disagrees with the lca formula"
            );
        }
    }
}

#[test]
fn test_tree_graph_is_zero_hyperbolic() {
    // A tree metric satisfies the four-point condition with delta = 0: of the
    // three pairwise sums, the largest two are equal. This is the property that
    // makes the tree the reference hyperbolic case.
    let n = 63;
    let data = generate_tree_graph(n, 2, 3);
    let d = |i: usize, j: usize| data.distances[i * n + j];

    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    let mut sums = [d(i, j) + d(k, l), d(i, k) + d(j, l), d(i, l) + d(j, k)];
                    sums.sort_by(f64::total_cmp);
                    assert!(
                        (sums[2] - sums[1]).abs() < 1e-9,
                        "four-point condition violated on ({i},{j},{k},{l}): {sums:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_tree_graph_labels() {
    let data = generate_tree_graph(1000, 2, 3);
    // 8 branches at depth 3, plus the 7-node trunk (depths 0-2) as label 0.
    let mut counts = [0usize; 9];
    for &l in &data.labels {
        assert!(l <= 8, "unexpected branch label {l}");
        counts[l as usize] += 1;
    }
    assert_eq!(counts[0], 7, "the trunk is depths 0-2 of a binary tree");
    for (branch, &count) in counts.iter().enumerate().skip(1) {
        assert!(count > 0, "branch {branch} is empty");
    }
}

#[test]
fn test_tree_graph_depths_are_recoverable() {
    // Depth is deliberately not folded into the labels; it is regenerated from
    // the same deterministic tree instead of being stored.
    let depths = RootedTree::complete(15, 2).depths().to_vec();
    assert_eq!(depths, vec![0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]);
}

// ---------------------------------------------------------------------------
// Matched geodesic balls
// ---------------------------------------------------------------------------

const BALL_SIGNS: [f64; 3] = [0.0, 1.0, -1.0];

#[test]
fn test_matched_ball_shape() {
    for sign in BALL_SIGNS {
        for m in [2usize, 9] {
            let data = generate_matched_ball(60, m, sign, 2.5, 42);
            let ambient = if sign == 0.0 { m } else { m + 1 };
            assert_eq!(data.n_points, 60);
            assert_eq!(data.ambient_dim, ambient, "sign {sign}, m {m}");
            assert_eq!(data.x.len(), 60 * ambient);
            assert_eq!(data.labels.len(), 60);
            assert_eq!(data.distances.len(), 60 * 60);
        }
    }
}

#[test]
fn test_matched_ball_constraints() {
    let m = 9;
    let sphere = generate_matched_ball(80, m, 1.0, 2.5, 42);
    for i in 0..80 {
        let norm_sq: f64 = (0..=m).map(|k| sphere.x[i * (m + 1) + k].powi(2)).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "sphere point {i} has |x|^2 = {norm_sq}"
        );
    }

    let hyp = generate_matched_ball(80, m, -1.0, 2.5, 42);
    check_on_hyperboloid(&hyp.x, 80, m + 1, 1e-6);
}

#[test]
fn test_matched_ball_labels_identical_across_geometries() {
    // The whole point of the family: same seed, same draw, so the label vectors
    // must agree bit for bit. If this drifts, the controlled comparison is void.
    for m in [2usize, 9] {
        let euc = generate_matched_ball(200, m, 0.0, 2.5, 7);
        let sph = generate_matched_ball(200, m, 1.0, 2.5, 7);
        let hyp = generate_matched_ball(200, m, -1.0, 2.5, 7);
        assert_eq!(euc.labels, sph.labels, "m {m}");
        assert_eq!(euc.labels, hyp.labels, "m {m}");
        // Eight classes, all populated at n = 200.
        for class in 0..8u32 {
            assert!(
                euc.labels.contains(&class),
                "class {class} missing at m {m}"
            );
        }
    }
}

/// Geodesic radius from each manifold's own origin, read off the coordinates
/// directly — the closed form of what `Manifold::distances_from_origin`
/// computes, without its `1e-7` clamp (which would floor every radius at
/// ~4.5e-4 and hide a genuine zero).
fn ball_radii(data: &DataPoints, sign: f64) -> Vec<f64> {
    let a = data.ambient_dim;
    (0..data.n_points)
        .map(|i| {
            let row = &data.x[i * a..(i + 1) * a];
            if sign > 0.0 {
                (-row[0]).clamp(-1.0, 1.0).acos()
            } else if sign < 0.0 {
                row[0].max(1.0).acosh()
            } else {
                row.iter().map(|v| v * v).sum::<f64>().sqrt()
            }
        })
        .collect()
}

#[test]
fn test_matched_ball_radii_match() {
    let extent = 2.5;
    for m in [2usize, 9] {
        let radii: Vec<Vec<f64>> = BALL_SIGNS
            .iter()
            .map(|&sign| ball_radii(&generate_matched_ball(150, m, sign, extent, 11), sign))
            .collect();

        for (i, &euclidean_r) in radii[0].iter().enumerate() {
            for curved in &radii[1..] {
                assert!(
                    (curved[i] - euclidean_r).abs() < 1e-10,
                    "point {i} at m {m}: radius {} vs {euclidean_r}",
                    curved[i]
                );
            }
            assert!(
                euclidean_r <= extent + 1e-12,
                "point {i} at m {m} is outside the ball: {euclidean_r}"
            );
        }
    }
}

#[test]
#[expect(
    clippy::float_cmp,
    reason = "the generators write a literal 0.0 on the diagonal; an epsilon here would accept a diagonal that is merely small"
)]
fn test_matched_ball_distances_agree() {
    let n = 40;
    let m = 2;
    for sign in BALL_SIGNS {
        let data = generate_matched_ball(n, m, sign, 2.5, 3);
        let a = data.ambient_dim;
        for i in 0..n {
            // The generators fill only i < j and leave the diagonal at exactly
            // zero. Recomputing d(i,i) is not a fair check: acos is vertical at
            // 1, so the ~1e-16 slack in |d| turns into ~1e-8 of angle.
            assert_eq!(data.distances[i * n + i], 0.0, "sign {sign}, diagonal {i}");
            for j in 0..n {
                if i == j {
                    continue;
                }
                assert!(
                    (data.distances[i * n + j] - data.distances[j * n + i]).abs() < 1e-12,
                    "sign {sign}, asymmetric at ({i},{j})"
                );
                let xi = &data.x[i * a..(i + 1) * a];
                let xj = &data.x[j * a..(j + 1) * a];
                let expected = if sign > 0.0 {
                    let dot: f64 = xi.iter().zip(xj).map(|(a, b)| a * b).sum();
                    dot.clamp(-1.0, 1.0).acos()
                } else if sign < 0.0 {
                    let inner = -xi[0] * xj[0]
                        + xi[1..]
                            .iter()
                            .zip(&xj[1..])
                            .map(|(a, b)| a * b)
                            .sum::<f64>();
                    (-inner).max(1.0).acosh()
                } else {
                    xi.iter()
                        .zip(xj)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt()
                };
                assert!(
                    (data.distances[i * n + j] - expected).abs() < 1e-9,
                    "sign {sign}, pair ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn test_matched_ball_deterministic() {
    for sign in BALL_SIGNS {
        let a = generate_matched_ball(50, 2, sign, 2.5, 99);
        let b = generate_matched_ball(50, 2, sign, 2.5, 99);
        assert_eq!(a.x, b.x);
        assert_eq!(a.labels, b.labels);
    }
}

#[test]
#[should_panic(expected = "extent must be <= PI")]
fn test_matched_ball_sphere_extent_panics() {
    let _ = generate_matched_ball(10, 2, 1.0, 4.0, 42);
}

#[test]
fn test_matched_ball_hyperbolic_accepts_a_wide_extent() {
    // Only the sphere is capped; the hyperbolic arm has no such bound.
    let data = generate_matched_ball(10, 2, -1.0, 5.0, 42);
    check_on_hyperboloid(&data.x, 10, 3, 1e-6);
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

/// Every synthetic dataset must ship its own intrinsic distance matrix.
///
/// Consumers choose between `EmbeddingState::from_distances` and
/// `EmbeddingState::new` on exactly this test (`optimizer::Evaluator::new`,
/// `web::EmbeddingRunner::from_synthetic`). It is load-bearing in two ways: a
/// graph dataset like `tree_graph` has *no* coordinates, so falling through to
/// the coordinate path embeds n zero-length vectors and collapses the result to
/// a single point; and a curved generator embedded from its ambient coordinates
/// would be fitted to chordal rather than geodesic distances, so the viewer
/// would not show the data the sweeps are computed from.
#[test]
fn test_every_synthetic_ships_intrinsic_distances() {
    for &name in DATASET_NAMES {
        let data = load_synthetic(name, 40, 42).unwrap();
        assert_eq!(
            data.distances.len(),
            40 * 40,
            "{name} ships no distance matrix; every consumer would fall back to \
             coordinates, which is wrong for a curved manifold and degenerate for a graph"
        );
        assert!(
            !data.x.is_empty() || data.ambient_dim == 0,
            "{name} has an ambient dimension but no coordinates"
        );
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
