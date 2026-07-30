//! Monte-Carlo hypervolume: agreement with exactly-computable volumes,
//! determinism, and the seed derivation the sharding relies on.

use fitting_analysis::hypervolume::{cell_seed, monte_carlo_hypervolume};
use fitting_analysis::objectives::N_OBJECTIVES;

const N_MC: u64 = 2_000_000;

/// Assert the estimate is within *k* standard errors of the exact volume.
fn assert_within_se(hv: f64, se: f64, exact: f64, k: f64) {
    let tol = k * se.max(1e-12);
    assert!(
        (hv - exact).abs() <= tol,
        "hv = {hv} ± {se}, exact = {exact} (off by {:.2} SE)",
        (hv - exact).abs() / se
    );
}

#[test]
fn single_point_volume_is_the_product_of_its_coordinates() {
    // A lone point dominates the box [0, f] — volume ∏ f_d. Put the slack in one
    // dimension so the volume is 0.5 and the relative MC error is small.
    let mut f = [1.0; N_OBJECTIVES];
    f[4] = 0.5;
    let (hv, se) = monte_carlo_hypervolume(&[f], N_MC, 12345);
    assert_within_se(hv, se, 0.5, 5.0);
}

#[test]
fn two_point_union_matches_inclusion_exclusion() {
    // f1 covers 0.2 in the last dim, f2 covers 0.2 in the first; their
    // intersection is 0.2·0.2. Union = 0.2 + 0.2 − 0.04 = 0.36.
    let mut f1 = [1.0; N_OBJECTIVES];
    f1[N_OBJECTIVES - 1] = 0.2;
    let mut f2 = [1.0; N_OBJECTIVES];
    f2[0] = 0.2;
    let (hv, se) = monte_carlo_hypervolume(&[f1, f2], N_MC, 7);
    assert_within_se(hv, se, 0.36, 5.0);
}

#[test]
fn dominated_points_do_not_change_the_volume() {
    // Adding a point inside the region another already dominates is a no-op, so
    // passing the whole trial set and passing just its front agree exactly (same
    // seed ⇒ same sample stream ⇒ bit-identical estimate).
    let mut f = [0.8; N_OBJECTIVES];
    f[0] = 0.9;
    let inside = [0.4; N_OBJECTIVES];
    let (a, _) = monte_carlo_hypervolume(&[f], N_MC, 99);
    let (b, _) = monte_carlo_hypervolume(&[f, inside], N_MC, 99);
    assert_eq!(a, b);
}

#[test]
fn full_box_and_empty_set_are_the_extremes() {
    let (hv, se) = monte_carlo_hypervolume(&[[1.0; N_OBJECTIVES]], 10_000, 1);
    assert_eq!(hv, 1.0);
    assert_eq!(se, 0.0);

    assert_eq!(monte_carlo_hypervolume(&[], N_MC, 1), (0.0, 0.0));
    // A point on the reference point itself dominates nothing.
    assert_eq!(
        monte_carlo_hypervolume(&[[0.0; N_OBJECTIVES]], 10_000, 1).0,
        0.0
    );
}

#[test]
fn estimate_is_deterministic_in_the_seed() {
    let f = [[0.7; N_OBJECTIVES]];
    let a = monte_carlo_hypervolume(&f, 200_000, 42);
    let b = monte_carlo_hypervolume(&f, 200_000, 42);
    let c = monte_carlo_hypervolume(&f, 200_000, 43);
    assert_eq!(a, b, "same seed must reproduce the estimate exactly");
    assert_ne!(a.0, c.0, "a different seed must draw a different sample");
}

#[test]
fn front_point_order_does_not_change_the_estimate() {
    // The implementation sorts the front internally for early-exit speed; that
    // must not leak into the result.
    let mut f1 = [0.9; N_OBJECTIVES];
    f1[0] = 0.3;
    let mut f2 = [0.4; N_OBJECTIVES];
    f2[0] = 1.0;
    let a = monte_carlo_hypervolume(&[f1, f2], 500_000, 5);
    let b = monte_carlo_hypervolume(&[f2, f1], 500_000, 5);
    assert_eq!(a, b);
}

#[test]
fn standard_error_shrinks_as_the_sample_grows() {
    let mut f = [1.0; N_OBJECTIVES];
    f[0] = 0.5;
    let (_, se_small) = monte_carlo_hypervolume(&[f], 100_000, 3);
    let (_, se_big) = monte_carlo_hypervolume(&[f], 1_600_000, 3);
    // 16× the samples ⇒ ~4× tighter; allow slack for the estimate of p itself.
    assert!(se_big < se_small / 3.0, "{se_big} vs {se_small}");
}

#[test]
fn cell_seed_matches_pythons_crc32_derivation() {
    // zlib.crc32(b"all_off_mnist_hyperbolic") == 334314840 (verified in Python).
    assert_eq!(cell_seed("all_off_mnist_hyperbolic", 0), 334_314_840);
    // The base seed XORs into the low 32 bits.
    assert_eq!(cell_seed("all_off_mnist_hyperbolic", 1), 334_314_840 ^ 1);
    // Different cells get different streams.
    assert_ne!(
        cell_seed("all_off_mnist_hyperbolic", 0),
        cell_seed("all_off_mnist_euclidean", 0)
    );
}
