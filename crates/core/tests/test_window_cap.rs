//! The hyperbolic search window's flat-ward cap, now gauged by `R_rms`.
//!
//! `HYPERBOLIC_KAPPA_MIN` bounds `κ = |K|·R_rms²`, the same gauge the fit
//! reports, which makes the cap a fixed point of `r = R_rms(r)/√κ_min` rather
//! than a closed form. These tests pin that it lands where it claims to and
//! that the search still terminates inside the window.

use fitting_core::curvature_detection::{
    fit_hyperbolic, reconstruct_hyperbolic, HYPERBOLIC_KAPPA_MIN,
};
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::synthetic_data::{
    generate_hd_hyperbolic_shells, generate_hd_tree, generate_uniform_grid,
    generate_uniform_hyperbolic,
};

const DIM: usize = 2;

fn grid_distances(n: usize, seed: u64) -> Vec<f64> {
    let g = generate_uniform_grid(n, seed);
    compute_euclidean_distance_matrix(&g.x, n, g.ambient_dim)
}

/// A pinned fit sits exactly at the cap, and the cap is where the
/// reconstruction's κ equals `HYPERBOLIC_KAPPA_MIN`. That equality is the whole
/// content of the new gauge: under the old input gauge a pinned arm reported
/// `κ = 0.01` in the *input* units and something else entirely in these.
#[test]
fn a_pinned_fit_reports_kappa_at_the_floor() {
    let n = 150;
    let cases: Vec<(&str, Vec<f64>)> = vec![
        ("grid", grid_distances(n, 42)),
        ("tree", generate_hd_tree(n, 10, 42).distances),
        (
            "hyperbolic_shells",
            generate_hd_hyperbolic_shells(n, 10, 42).distances,
        ),
    ];

    for (name, d) in cases {
        let fit = fit_hyperbolic(&d, n, DIM);
        if !fit.at_upper_bound {
            continue;
        }
        let kappa = reconstruct_hyperbolic(&d, n, DIM, fit.radius).kappa(n);
        let rel = (kappa - HYPERBOLIC_KAPPA_MIN).abs() / HYPERBOLIC_KAPPA_MIN;
        assert!(
            rel < 0.05,
            "{name}: pinned at r={}, kappa={kappa} but the floor is {HYPERBOLIC_KAPPA_MIN}",
            fit.radius
        );
    }
}

/// The fit must land inside its own window, pinned or not.
#[test]
fn the_fitted_radius_stays_within_the_window() {
    let n = 120;
    for (name, d) in [
        ("grid", grid_distances(n, 7)),
        (
            "hyperbolic",
            generate_uniform_hyperbolic(n, 7, 3.0).distances,
        ),
    ] {
        let d_max = d.iter().cloned().fold(0.0_f64, f64::max);
        let fit = fit_hyperbolic(&d, n, DIM);
        assert!(
            fit.radius >= d_max / 20.0 - 1e-9,
            "{name}: r* {} below the overflow floor",
            fit.radius
        );
        assert!(fit.radius.is_finite(), "{name}: r* not finite");
        // Beyond the cap the reconstruction is flatter than the floor allows.
        let kappa = reconstruct_hyperbolic(&d, n, DIM, fit.radius).kappa(n);
        assert!(
            kappa >= HYPERBOLIC_KAPPA_MIN * 0.95,
            "{name}: r* {} implies kappa {kappa}, under the floor",
            fit.radius
        );
    }
}

/// Genuinely hyperbolic data should not need the cap — the residual has an
/// interior minimum well short of the flat end.
#[test]
fn exact_hyperbolic_data_is_not_pinned() {
    let n = 150;
    let d = generate_uniform_hyperbolic(n, 42, 3.0).distances;
    let fit = fit_hyperbolic(&d, n, DIM);
    assert!(
        !fit.at_upper_bound,
        "H² data pinned at the flat-ward cap (r*={})",
        fit.radius
    );
    // The generator uses unit radius; the fit should be in that neighbourhood.
    assert!(
        fit.radius > 0.2 && fit.radius < 5.0,
        "H² fit radius {} is far from the generator's 1.0",
        fit.radius
    );
}

/// The cap must be reproducible: the fixed point is seeded deterministically
/// and `eigen_symmetric` is deterministic, so two calls agree exactly.
#[test]
fn the_window_cap_is_deterministic() {
    let n = 100;
    let d = grid_distances(n, 3);
    let a = fit_hyperbolic(&d, n, DIM);
    let b = fit_hyperbolic(&d, n, DIM);
    assert_eq!(a.radius, b.radius);
    assert_eq!(a.residual, b.residual);
    assert_eq!(a.at_upper_bound, b.at_upper_bound);
}
