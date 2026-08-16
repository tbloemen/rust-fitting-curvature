//! The binary additive ε-indicator: hand-computed values, the asymmetry that
//! forces both directions, and the compliance property `Δε > 0` rests on.

use fitting_analysis::indicators::{epsilon_additive, epsilon_pair};
use fitting_analysis::objectives::N_OBJECTIVES;

/// A front point that scores *v* on every objective.
fn flat(v: f64) -> [f64; N_OBJECTIVES] {
    [v; N_OBJECTIVES]
}

/// `epsilon_additive` on two non-empty fronts, unwrapped.
fn eps(a: &[[f64; N_OBJECTIVES]], b: &[[f64; N_OBJECTIVES]]) -> f64 {
    epsilon_additive(a, b).expect("both fronts non-empty")
}

fn close(x: f64, y: f64) -> bool {
    (x - y).abs() < 1e-12
}

// ─── Hand-computed values ─────────────────────────────────────────────────────

#[test]
fn one_front_that_covers_the_other_scores_negative() {
    // A is 0.3 better on every objective, so it covers B with 0.3 to spare and B
    // must be raised by 0.3 to cover A.
    let a = [flat(0.9)];
    let b = [flat(0.6)];
    assert!(close(eps(&a, &b), -0.3), "{}", eps(&a, &b));
    assert!(close(eps(&b, &a), 0.3), "{}", eps(&b, &a));

    let pair = epsilon_pair(&a, &b).unwrap();
    assert!(close(pair.delta, 0.6), "{}", pair.delta);
    assert!(pair.setting_covers_baseline());
    assert!(!pair.baseline_covers_setting());
}

#[test]
fn crossing_fronts_are_covered_by_neither_side() {
    // Two specialists at 0.2 with one objective pushed to 0.95, against a
    // generalist at 0.5.
    let mut a1 = flat(0.2);
    a1[0] = 0.95;
    let mut a2 = flat(0.2);
    a2[1] = 0.95;
    let a = [a1, a2];
    let b = [flat(0.5)];

    // Covering b needs +0.3 on the eight objectives where a sits at 0.2.
    assert!(close(eps(&a, &b), 0.30), "{}", eps(&a, &b));
    // Covering either specialist needs +0.45 on its strong objective.
    assert!(close(eps(&b, &a), 0.45), "{}", eps(&b, &a));

    let pair = epsilon_pair(&a, &b).unwrap();
    assert!(close(pair.delta, 0.15), "{}", pair.delta);
    assert!(!pair.setting_covers_baseline());
    assert!(!pair.baseline_covers_setting());
}

#[test]
fn identical_fronts_score_zero_in_both_directions() {
    let a = [flat(0.7), flat(0.4)];
    let pair = epsilon_pair(&a, &a).unwrap();
    assert_eq!(pair.setting_vs_baseline, 0.0);
    assert_eq!(pair.baseline_vs_setting, 0.0);
    assert_eq!(pair.delta, 0.0);
    // Weak coverage both ways is the correct reading of a tie.
    assert!(pair.setting_covers_baseline());
    assert!(pair.baseline_covers_setting());
}

// ─── Properties ───────────────────────────────────────────────────────────────

#[test]
fn the_indicator_is_asymmetric() {
    // Why both directions are computed and reported. Anyone tempted to collapse
    // this to one call breaks here.
    let mut a1 = flat(0.2);
    a1[0] = 0.95;
    let a = [a1];
    let b = [flat(0.5)];
    assert_ne!(eps(&a, &b), eps(&b, &a));
}

#[test]
fn an_empty_front_has_no_indicator() {
    // Not zero: with no point to shift there is no ε, and with nothing to cover
    // the maximum is over an empty set.
    let a = [flat(0.5)];
    assert!(epsilon_additive(&a, &[]).is_none());
    assert!(epsilon_additive(&[], &a).is_none());
    assert!(epsilon_additive(&[], &[]).is_none());
    assert!(epsilon_pair(&[], &a).is_none());
    assert!(epsilon_pair(&a, &[]).is_none());
}

#[test]
fn the_indicator_is_pareto_compliant() {
    // The property that makes Δε > 0 mean something: improving a front on every
    // objective can only improve the indicator in both directions. Unlike R2
    // this holds strictly, which is the reason for reporting it alongside.
    let worse = [0.3, 0.4, 0.5, 0.2, 0.6, 0.7, 0.1, 0.4, 0.5, 0.3];
    let better = [0.4, 0.5, 0.6, 0.3, 0.8, 0.7, 0.2, 0.6, 0.5, 0.9];
    let reference = [flat(0.55), flat(0.45)];

    // Against a fixed reference, the dominating front needs no more of a shift…
    assert!(eps(&[better], &reference) <= eps(&[worse], &reference));
    // …and is harder for the reference to cover.
    assert!(eps(&reference, &[better]) >= eps(&reference, &[worse]));
}

#[test]
fn dominated_points_do_not_change_the_indicator() {
    // Why reducing each cell to its Pareto front before comparing is exact
    // rather than an approximation — the same argument `test_r2.rs` makes for R2.
    let mut front = flat(0.8);
    front[0] = 0.9;
    let inside = flat(0.4);
    let b = [flat(0.6), flat(0.5)];

    assert_eq!(eps(&[front], &b), eps(&[front, inside], &b));
    assert_eq!(eps(&b, &[front]), eps(&b, &[front, inside]));
}

#[test]
fn the_indicator_is_order_independent() {
    let mut a = flat(0.5);
    a[3] = 0.9;
    let mut b = flat(0.6);
    b[7] = 0.2;
    let reference = [flat(0.45)];
    assert_eq!(eps(&[a, b], &reference), eps(&[b, a], &reference));
    assert_eq!(eps(&reference, &[a, b]), eps(&reference, &[b, a]));
}

#[test]
fn coverage_means_the_signed_summary_is_non_negative() {
    // The invariant the output table is checked against: a setting that covers
    // the baseline never reports a negative Δε.
    let a = [flat(0.9), flat(0.75)];
    let b = [flat(0.6)];
    let pair = epsilon_pair(&a, &b).unwrap();
    assert!(pair.setting_covers_baseline());
    assert!(pair.delta >= 0.0);
}
