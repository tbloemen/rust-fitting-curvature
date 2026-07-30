//! The statistics ported from scipy, checked against scipy's own output.
//!
//! Reference values were produced with scipy 1.17.1 using the same defaults the
//! Python analysis called with (`spearmanr(x, y)`, `wilcoxon(d)`).

use fitting_analysis::stats::{
    mean, median, normal_sf, pearson, rankdata, spearman, student_t_sf, wilcoxon, WilcoxonMethod,
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
fn wilcoxon_matches_scipy_exact() {
    // scipy: WilcoxonResult(statistic=3.0, pvalue=0.0390625)
    let d = [0.01, -0.002, 0.03, 0.005, -0.001, 0.02, 0.008, 0.015];
    let (t, p, method) = wilcoxon(&d).unwrap();
    assert_eq!(method, WilcoxonMethod::Exact);
    assert_close(t, 3.0, 1e-12);
    assert_close(p, 0.0390625, 1e-12);

    // scipy: statistic=2.0, pvalue=0.09375
    let d2 = [0.01, 0.02, 0.03, -0.005, 0.004, 0.006];
    let (t, p, _) = wilcoxon(&d2).unwrap();
    assert_close(t, 2.0, 1e-12);
    assert_close(p, 0.09375, 1e-12);

    // scipy: statistic=3.0, pvalue=0.625
    let d3 = [0.1, 0.2, -0.3, 0.4];
    let (t, p, _) = wilcoxon(&d3).unwrap();
    assert_close(t, 3.0, 1e-12);
    assert_close(p, 0.625, 1e-12);
}

#[test]
fn wilcoxon_discards_zeros() {
    // scipy's default zero_method="wilcox" drops zero differences before
    // ranking, so these two inputs must give the same answer.
    let with_zeros = [0.1, 0.2, -0.3, 0.4, 0.0, 0.0];
    let without = [0.1, 0.2, -0.3, 0.4];
    let (t1, p1, _) = wilcoxon(&with_zeros).unwrap();
    let (t2, p2, _) = wilcoxon(&without).unwrap();
    assert_close(t1, t2, 1e-12);
    assert_close(p1, p2, 1e-12);
}

#[test]
fn wilcoxon_all_zero_is_undefined() {
    assert!(wilcoxon(&[0.0, 0.0, 0.0]).is_none());
    assert!(wilcoxon(&[]).is_none());
}

#[test]
fn wilcoxon_all_positive_is_the_smallest_possible_p() {
    // Every difference positive → W⁻ = 0 → two-sided p = 2/2ⁿ.
    let d = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let (t, p, _) = wilcoxon(&d).unwrap();
    assert_close(t, 0.0, 1e-12);
    assert_close(p, 2.0 / 64.0, 1e-12);
}

#[test]
fn wilcoxon_falls_back_to_asymptotic_on_ties() {
    // Tied |d| values rule out the exact distribution (scipy's "auto" does the
    // same); the tie-corrected normal approximation takes over.
    let d = [0.1, -0.1, 0.2, 0.3, 0.3, -0.4, 0.5, 0.6];
    let (_, p, method) = wilcoxon(&d).unwrap();
    assert_eq!(method, WilcoxonMethod::Asymptotic);
    assert!((0.0..=1.0).contains(&p), "p = {p}");
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
