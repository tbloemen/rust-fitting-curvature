//! The statistics ported from scipy, checked against scipy's own output.
//!
//! Reference values were produced with scipy 1.17.1 using the same defaults the
//! Python analysis called with (`spearmanr(x, y)`).

use fitting_analysis::stats::{
    mean, median, pearson, quantile, rankdata, spearman, spearman_matrix, student_t_sf,
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
    assert_close(rho, 0.821_428_571_428_571_5, 1e-12);
    assert_close(p, 0.023_448_808_345_691_5, 1e-9);
}

#[test]
fn spearman_handles_ties_like_scipy() {
    // scipy: statistic=-0.9461247469114745, pvalue=0.0003753118737904405
    let x = [0.1, 0.5, 0.2, 0.9, 0.4, 0.4, 0.7, 0.3];
    let y = [1.0, 0.2, 0.8, 0.1, 0.55, 0.5, 0.3, 0.9];
    let (rho, p) = spearman(&x, &y).unwrap();
    assert_close(rho, -0.946_124_746_911_474_5, 1e-12);
    assert_close(p, 0.000_375_311_873_790_440_5, 1e-9);
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
fn spearman_matrix_is_symmetric_with_unit_diagonal() {
    // Column 0 against column 1 is the `spearman_matches_scipy` pair; column 2
    // is column 0 reversed, so its ρ against column 0 is exactly -1.
    let columns = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        vec![2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 6.0],
        vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ];
    let m = spearman_matrix(&columns);
    assert_eq!(m.len(), 3);
    for (i, row) in m.iter().enumerate() {
        assert_eq!(row.len(), 3);
        assert_eq!(row[i], Some(1.0));
        for j in 0..3 {
            assert_eq!(row[j], m[j][i], "ρ[{i}][{j}] != ρ[{j}][{i}]");
        }
    }
    assert_close(m[0][1].unwrap(), 0.821_428_571_428_571_5, 1e-12);
    assert_close(m[0][2].unwrap(), -1.0, 1e-12);
    assert_close(m[1][2].unwrap(), -0.821_428_571_428_571_5, 1e-12);
}

#[test]
fn spearman_matrix_marks_a_constant_column_undefined() {
    let columns = vec![vec![1.0, 2.0, 3.0], vec![5.0, 5.0, 5.0]];
    let m = spearman_matrix(&columns);
    assert_eq!(m[0][0], Some(1.0));
    assert_eq!(m[1][1], Some(1.0));
    assert_eq!(m[0][1], None);
    assert_eq!(m[1][0], None);
    assert!(spearman_matrix(&[]).is_empty());
}

#[test]
fn student_t_tail_matches_known_quantiles() {
    // The 97.5th percentile of t with 3 dof is 3.182446305284263.
    assert_close(student_t_sf(3.182_446_305_284_263, 3.0), 0.025, 1e-9);
    assert_close(student_t_sf(0.0, 5.0), 0.5, 1e-12);
    // Large dof converges to the normal.
    assert_close(student_t_sf(1.959_963_984_540_054, 1e7), 0.025, 1e-5);
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
