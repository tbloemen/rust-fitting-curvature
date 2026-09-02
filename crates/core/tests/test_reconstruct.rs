//! Round-trip tests for `curvature_detection::reconstruct`.
//!
//! The claim under test: a dataset that lies exactly on a constant-curvature
//! manifold is recovered from its distance matrix alone, with the discarded
//! eigenvalue mass — the Wilson residual — going to zero. If that holds, the
//! reconstruction is measuring the model rather than an artefact of it.

use fitting_core::curvature_detection::{
    fit_euclidean, fit_hyperbolic, fit_spherical, reconstruct_euclidean, reconstruct_hyperbolic,
    reconstruct_spherical,
};
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::synthetic_data::{
    generate_uniform_grid, generate_uniform_hyperbolic, generate_uniform_sphere,
};

const DIM: usize = 2;

/// Largest absolute disagreement between two distance matrices.
fn max_distance_error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// RMS pairwise distance, for scaling the tolerance to the data.
fn d_rms(d: &[f64], n: usize) -> f64 {
    let sum: f64 = d.iter().map(|v| v * v).sum();
    (sum / (n * (n - 1)) as f64).sqrt()
}

#[test]
fn sphere_round_trips_at_its_own_radius() {
    let data = generate_uniform_sphere(120, 42);
    let n = data.n_points;
    // `generate_uniform_sphere` produces geodesic distances on the unit sphere.
    let rec = reconstruct_spherical(&data.distances, n, DIM, 1.0);

    assert_eq!(rec.ambient_dim, DIM + 1);
    assert!((rec.curvature - 1.0).abs() < 1e-12);

    // Exact data on S² leaves nothing outside the signature block.
    let gauge = n as f64 * d_rms(&data.distances, n).powi(2);
    assert!(
        rec.discarded_mass / gauge < 1e-10,
        "S² should reconstruct exactly, discarded {} (gauge {gauge})",
        rec.discarded_mass
    );

    let err = max_distance_error(&data.distances, &rec.pairwise_distances(n));
    assert!(err < 1e-6, "S² distances off by {err}");
}

#[test]
fn hyperboloid_round_trips_at_its_own_radius() {
    let data = generate_uniform_hyperbolic(120, 42, 3.0);
    let n = data.n_points;
    let rec = reconstruct_hyperbolic(&data.distances, n, DIM, 1.0);

    assert_eq!(rec.ambient_dim, DIM + 1);
    assert!((rec.curvature + 1.0).abs() < 1e-12);

    let gauge = n as f64 * d_rms(&data.distances, n).powi(2);
    assert!(
        rec.discarded_mass / gauge < 1e-9,
        "H² should reconstruct exactly, discarded {} (gauge {gauge})",
        rec.discarded_mass
    );

    let err = max_distance_error(&data.distances, &rec.pairwise_distances(n));
    assert!(err < 1e-5, "H² distances off by {err}");
}

#[test]
fn grid_round_trips_flat() {
    let data = generate_uniform_grid(120, 42);
    let n = data.n_points;
    let d = compute_euclidean_distance_matrix(&data.x, n, data.ambient_dim);
    let rec = reconstruct_euclidean(&d, n, DIM);

    assert_eq!(rec.ambient_dim, DIM);
    assert_eq!(rec.curvature, 0.0);

    // A 2-D grid spans exactly 2 dimensions, so classical MDS is exact.
    let gauge = n as f64 * d_rms(&d, n).powi(2);
    assert!(
        rec.discarded_mass / gauge < 1e-10,
        "a planar grid should reconstruct exactly, discarded {}",
        rec.discarded_mass
    );

    let err = max_distance_error(&d, &rec.pairwise_distances(n));
    assert!(err < 1e-8, "flat distances off by {err}");
}

/// Every point must satisfy its manifold's defining constraint, or the metric
/// code downstream (which assumes it) reads garbage.
#[test]
fn reconstructions_satisfy_the_manifold_constraint() {
    let n = 80;

    let sphere = generate_uniform_sphere(n, 7);
    let rec = reconstruct_spherical(&sphere.distances, n, DIM, 1.0);
    for i in 0..n {
        let row = &rec.points[i * rec.ambient_dim..(i + 1) * rec.ambient_dim];
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "point {i} off the sphere: {norm}");
    }

    let hyp = generate_uniform_hyperbolic(n, 7, 3.0);
    let rec = reconstruct_hyperbolic(&hyp.distances, n, DIM, 1.0);
    for i in 0..n {
        let row = &rec.points[i * rec.ambient_dim..(i + 1) * rec.ambient_dim];
        let q = -row[0] * row[0] + row[1..].iter().map(|v| v * v).sum::<f64>();
        assert!((q + 1.0).abs() < 1e-8, "point {i} off the hyperboloid: {q}");
        assert!(row[0] > 0.0, "point {i} landed on the lower sheet");
    }
}

/// The headline invariant: `discarded_mass` is the Wilson residual, so the fit
/// and the reconstruction are two readings of one procedure.
#[test]
fn discarded_mass_equals_the_wilson_residual() {
    let cases: Vec<(&str, Vec<f64>, usize)> = vec![
        ("sphere", generate_uniform_sphere(100, 3).distances, 100),
        (
            "hyperbolic",
            generate_uniform_hyperbolic(100, 3, 3.0).distances,
            100,
        ),
        ("grid", {
            let g = generate_uniform_grid(100, 3);
            compute_euclidean_distance_matrix(&g.x, 100, g.ambient_dim)
        }, 100),
    ];

    for (name, d, n) in cases {
        let sph = fit_spherical(&d, n, DIM);
        let rec = reconstruct_spherical(&d, n, DIM, sph.radius);
        assert!(
            (rec.discarded_mass - sph.residual).abs() <= 1e-8 * (1.0 + sph.residual.abs()),
            "{name}: spherical discarded {} vs residual {}",
            rec.discarded_mass,
            sph.residual
        );

        let hyp = fit_hyperbolic(&d, n, DIM);
        let rec = reconstruct_hyperbolic(&d, n, DIM, hyp.radius);
        assert!(
            (rec.discarded_mass - hyp.residual).abs() <= 1e-8 * (1.0 + hyp.residual.abs()),
            "{name}: hyperbolic discarded {} vs residual {}",
            rec.discarded_mass,
            hyp.residual
        );

        let euc = fit_euclidean(&d, n, DIM);
        let rec = reconstruct_euclidean(&d, n, DIM);
        assert!(
            (rec.discarded_mass - euc.residual).abs() <= 1e-8 * (1.0 + euc.residual.abs()),
            "{name}: euclidean discarded {} vs residual {}",
            rec.discarded_mass,
            euc.residual
        );
    }
}

/// Reconstructing a mismatched model must still produce usable coordinates —
/// this is the off-diagonal of the experiment grid, so it cannot panic or
/// return NaN just because the data does not fit.
#[test]
fn mismatched_models_still_produce_finite_coordinates() {
    let n = 80;
    let sphere = generate_uniform_sphere(n, 5);

    for rec in [
        reconstruct_hyperbolic(&sphere.distances, n, DIM, 4.0),
        reconstruct_euclidean(&sphere.distances, n, DIM),
    ] {
        assert!(rec.points.iter().all(|v| v.is_finite()), "non-finite coords");
        assert!(rec.discarded_mass.is_finite() && rec.discarded_mass > 0.0);
        assert!(rec.pairwise_distances(n).iter().all(|v| v.is_finite()));
        assert!(rec.distances_from_origin(n).iter().all(|v| v.is_finite()));
    }
}
