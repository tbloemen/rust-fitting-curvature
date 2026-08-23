//! The statistics ported from scipy, checked against scipy's own output.
//!
//! Reference values were produced with scipy 1.17.1 using the same defaults the
//! Python analysis called with (`spearmanr(x, y)`).

use fitting_analysis::stats::{
    chi2_sf, friedman, holm_against_control, mean, median, normal_sf, pearson, quantile, rankdata,
    spearman, student_t_sf,
};

fn assert_close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} != {b} (tol {tol})");
}

#[test]
fn rankdata_averages_ties() {
    assert_eq!(rankdata(&[10.0, 20.0, 30.0]), vec![1.0, 2.0, 3.0]);
    // Two-way tie at the bottom shares ranks 1 and 2 → 1.5 each.
    assert_eq!(rankdata(&[5.0, 5.0, 9.0]), vec![1.5, 1.5, 3.0]);
    // Three-way tie over ranks 2,3,4 → 3.0 each.
    assert_eq!(rankdata(&[1.0, 7.0, 7.0, 7.0]), vec![1.0, 3.0, 3.0, 3.0]);
}

#[test]
fn pearson_matches_hand_values() {
    assert_close(
        pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap(),
        1.0,
        1e-12,
    );
    assert_close(
        pearson(&[1.0, 2.0, 3.0], &[6.0, 4.0, 2.0]).unwrap(),
        -1.0,
        1e-12,
    );
    // A constant series has no correlation to report.
    assert!(pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
}

#[test]
fn spearman_matches_scipy() {
    // scipy: SignificanceResult(statistic=0.8214285714285715, pvalue=0.0234488083456915)
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let y = [2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 6.0];
    let (rho, p) = spearman(&x, &y).unwrap();
    assert_close(rho, 0.8214285714285715, 1e-12);
    assert_close(p, 0.0234488083456915, 1e-9);
}

#[test]
fn spearman_handles_ties_like_scipy() {
    // scipy: statistic=-0.9461247469114745, pvalue=0.0003753118737904405
    let x = [0.1, 0.5, 0.2, 0.9, 0.4, 0.4, 0.7, 0.3];
    let y = [1.0, 0.2, 0.8, 0.1, 0.55, 0.5, 0.3, 0.9];
    let (rho, p) = spearman(&x, &y).unwrap();
    assert_close(rho, -0.9461247469114745, 1e-12);
    assert_close(p, 0.0003753118737904405, 1e-9);
}

#[test]
fn spearman_perfect_anticorrelation_is_significant() {
    // scipy gives rho = -1 (to float noise) and p = 1.4e-24 here; the exact
    // p depends on that noise, so only the magnitude is meaningful.
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [5.0, 4.0, 3.0, 2.0, 1.0];
    let (rho, p) = spearman(&x, &y).unwrap();
    assert_close(rho, -1.0, 1e-12);
    assert!(p < 1e-20, "p = {p}");
}

#[test]
fn spearman_needs_three_points() {
    assert!(spearman(&[1.0, 2.0], &[2.0, 1.0]).is_none());
}

#[test]
fn normal_tail_matches_known_quantiles() {
    assert_close(normal_sf(0.0), 0.5, 1e-7);
    assert_close(normal_sf(1.0), 0.15865525393145707, 1e-7);
    assert_close(normal_sf(1.959963984540054), 0.025, 1e-7);
    assert_close(normal_sf(-1.0), 0.8413447460685429, 1e-7);
}

#[test]
fn student_t_tail_matches_known_quantiles() {
    // The 97.5th percentile of t with 3 dof is 3.182446305284263.
    assert_close(student_t_sf(3.182446305284263, 3.0), 0.025, 1e-9);
    assert_close(student_t_sf(0.0, 5.0), 0.5, 1e-12);
    // Large dof converges to the normal.
    assert_close(student_t_sf(1.959963984540054, 1e7), 0.025, 1e-5);
}

#[test]
fn mean_and_median() {
    assert_close(mean(&[1.0, 2.0, 6.0]).unwrap(), 3.0, 1e-12);
    assert_close(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0, 1e-12);
    assert_close(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5, 1e-12);
    assert!(median(&[]).is_none());
}

/// Reference values from `numpy.quantile(..., method="linear")`, the default
/// scipy and numpy both use.
#[test]
fn quantile_interpolates_like_numpy() {
    let even = [1.0, 2.0, 3.0, 4.0];
    assert_close(quantile(&even, 0.25).unwrap(), 1.75, 1e-12);
    assert_close(quantile(&even, 0.5).unwrap(), 2.5, 1e-12);
    assert_close(quantile(&even, 0.75).unwrap(), 3.25, 1e-12);

    let odd = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_close(quantile(&odd, 0.25).unwrap(), 2.0, 1e-12);
    assert_close(quantile(&odd, 0.75).unwrap(), 4.0, 1e-12);

    // Unsorted input, and a q that lands between two order statistics.
    assert_close(
        quantile(&[50.0, 10.0, 30.0, 20.0, 40.0], 0.1).unwrap(),
        14.0,
        1e-12,
    );

    // The endpoints are the extremes; one value is its own every quantile.
    assert_close(quantile(&[3.0, 1.0, 2.0], 0.0).unwrap(), 1.0, 1e-12);
    assert_close(quantile(&[3.0, 1.0, 2.0], 1.0).unwrap(), 3.0, 1e-12);
    assert_close(quantile(&[7.0], 0.42).unwrap(), 7.0, 1e-12);

    assert!(quantile(&[], 0.5).is_none());
    assert!(quantile(&[1.0, 2.0], 1.5).is_none());
}

// ─── χ² tail ─────────────────────────────────────────────────────────────────

/// Closed form for even degrees of freedom:
/// `sf(x) = e^(−x/2) · Σ_{j<dof/2} (x/2)^j / j!`.
fn chi2_sf_even(x: f64, dof: usize) -> f64 {
    let h = x / 2.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for j in 1..dof / 2 {
        term *= h / j as f64;
        sum += term;
    }
    (-h).exp() * sum
}

#[test]
fn chi2_tail_matches_the_even_dof_closed_form() {
    // Checked against the analytic form rather than pinned to scipy, so the test
    // is independent of the library it is meant to replace.
    for &dof in &[2usize, 4, 6, 10, 20] {
        for &x in &[0.25f64, 1.0, 3.5, 8.0, 15.0, 40.0] {
            assert_close(chi2_sf(x, dof as f64), chi2_sf_even(x, dof), 1e-10);
        }
    }
}

#[test]
fn chi2_tail_matches_the_normal_at_one_dof() {
    // With dof = 1, X = Z² so P(X > x) = 2·P(Z > √x).
    for &x in &[0.1f64, 1.0, 3.841_458_820_694_124, 10.0] {
        assert_close(chi2_sf(x, 1.0), 2.0 * normal_sf(x.sqrt()), 1e-7);
    }
    // The 5% critical value of χ²₁.
    assert_close(chi2_sf(3.841_458_820_694_124, 1.0), 0.05, 1e-7);
}

#[test]
fn chi2_tail_covers_the_degenerate_ends() {
    assert_eq!(chi2_sf(0.0, 3.0), 1.0);
    assert_eq!(chi2_sf(-1.0, 3.0), 1.0);
    assert!(chi2_sf(1.0, 0.0).is_nan());
    assert!(chi2_sf(1e6, 2.0) < 1e-300);
}

// ─── Friedman ────────────────────────────────────────────────────────────────

#[test]
fn friedman_matches_the_hand_computed_statistic() {
    // Four blocks that rank the three treatments identically: rank sums 4, 8, 12.
    //   ssbn = 224,  χ² = 12/(3·4·4)·224 − 3·4·4 = 8,  p = e^(−4) at dof 2.
    let blocks = vec![
        vec![3.0, 2.0, 1.0],
        vec![3.0, 2.0, 1.0],
        vec![3.0, 2.0, 1.0],
        vec![3.0, 2.0, 1.0],
    ];
    let f = friedman(&blocks).unwrap();
    assert_close(f.statistic, 8.0, 1e-12);
    assert_close(f.p, (-4.0f64).exp(), 1e-10);
    assert_eq!(f.n_blocks, 4);
    assert_eq!(f.k, 3);
    // Larger is better, so the best treatment gets rank 1.
    assert_close(f.mean_ranks[0], 1.0, 1e-12);
    assert_close(f.mean_ranks[2], 3.0, 1e-12);
}

#[test]
fn friedman_applies_the_tie_correction() {
    // Two blocks, each tying the first two treatments:
    //   rank sums 3, 3, 6;  ssbn = 54;  ties Σ(t³−t) = 12
    //   c = 1 − 12/(3·8·2) = 0.75;  χ² = (27 − 24)/0.75 = 4;  p = e^(−2).
    let blocks = vec![vec![1.0, 1.0, 0.0], vec![1.0, 1.0, 0.0]];
    let f = friedman(&blocks).unwrap();
    assert_close(f.statistic, 4.0, 1e-12);
    assert_close(f.p, (-2.0f64).exp(), 1e-10);
    assert_close(f.mean_ranks[0], 1.5, 1e-12);
    assert_close(f.mean_ranks[1], 1.5, 1e-12);
    assert_close(f.mean_ranks[2], 3.0, 1e-12);
}

#[test]
fn friedman_is_invariant_to_monotone_rescaling() {
    // It is a rank test: only the within-block ordering may matter.
    let blocks: Vec<Vec<f64>> = vec![
        vec![0.1, 0.5, 0.3],
        vec![9.0, 2.0, 4.0],
        vec![1.0, 3.0, 2.0],
    ];
    let scaled: Vec<Vec<f64>> = blocks
        .iter()
        .map(|b| b.iter().map(|v: &f64| (v * 3.0).exp()).collect())
        .collect();
    let a = friedman(&blocks).unwrap();
    let b = friedman(&scaled).unwrap();
    assert_close(a.statistic, b.statistic, 1e-9);
    assert_eq!(a.mean_ranks, b.mean_ranks);
}

#[test]
fn friedman_rejects_input_it_cannot_test() {
    // Fewer than two blocks.
    assert!(friedman(&[vec![1.0, 2.0, 3.0]]).is_none());
    // Fewer than three treatments: that is a sign test, not a Friedman test.
    assert!(friedman(&[vec![1.0, 2.0], vec![2.0, 1.0]]).is_none());
    // Ragged blocks.
    assert!(friedman(&[vec![1.0, 2.0, 3.0], vec![1.0, 2.0]]).is_none());
    // Non-finite values.
    assert!(friedman(&[vec![1.0, f64::NAN, 3.0], vec![1.0, 2.0, 3.0]]).is_none());
    // Everything tied everywhere: the tie correction would divide by zero.
    assert!(friedman(&[vec![1.0; 3], vec![1.0; 3]]).is_none());
}

// ─── Holm post-hoc ───────────────────────────────────────────────────────────

#[test]
fn holm_against_control_matches_the_hand_computed_adjustment() {
    // Mean ranks 1, 2, 3 over 4 blocks: SE = sqrt(3·4/(6·4)) = sqrt(1/2).
    //   z₁ = √2   → p = 0.157299…,  z₂ = 2√2 → p = 0.004678…
    //   Holm (m = 2): the smaller is doubled, the larger is left alone.
    let blocks = vec![
        vec![3.0, 2.0, 1.0],
        vec![3.0, 2.0, 1.0],
        vec![3.0, 2.0, 1.0],
        vec![3.0, 2.0, 1.0],
    ];
    let f = friedman(&blocks).unwrap();
    let p = holm_against_control(&f, 0);

    assert!(p[0].is_none(), "the control has no p against itself");
    let raw_1 = 2.0 * normal_sf(std::f64::consts::SQRT_2);
    let raw_2 = 2.0 * normal_sf(2.0 * std::f64::consts::SQRT_2);
    assert_close(p[2].unwrap(), 2.0 * raw_2, 1e-9);
    assert_close(p[1].unwrap(), raw_1, 1e-9);
    // Holm is strictly more conservative than the uncorrected comparison.
    assert!(p[2].unwrap() > raw_2);
}

#[test]
fn holm_adjusted_p_values_are_monotone_and_bounded() {
    let blocks = vec![
        vec![0.9, 0.5, 0.4, 0.1],
        vec![0.8, 0.6, 0.3, 0.2],
        vec![0.7, 0.5, 0.4, 0.3],
        vec![0.9, 0.4, 0.5, 0.2],
        vec![0.6, 0.55, 0.3, 0.25],
    ];
    let f = friedman(&blocks).unwrap();
    let p = holm_against_control(&f, 0);

    let mut adjusted: Vec<(f64, f64)> = (1..f.k)
        .map(|i| ((f.mean_ranks[i] - f.mean_ranks[0]).abs(), p[i].unwrap()))
        .collect();
    // Every adjusted value is a probability.
    for (_, v) in &adjusted {
        assert!((0.0..=1.0).contains(v), "adjusted p {v} out of range");
    }
    // A treatment further from the control in mean rank cannot get a larger
    // adjusted p than a nearer one.
    adjusted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for w in adjusted.windows(2) {
        assert!(
            w[0].1 >= w[1].1 - 1e-12,
            "monotonicity broken: {:?} then {:?}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn holm_handles_a_control_index_out_of_range() {
    let blocks = vec![vec![3.0, 2.0, 1.0], vec![1.0, 2.0, 3.0]];
    let f = friedman(&blocks).unwrap();
    assert!(holm_against_control(&f, 9).iter().all(|p| p.is_none()));
}
