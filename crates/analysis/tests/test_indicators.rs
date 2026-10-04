//! The binary additive ε-indicator: hand-computed values, the asymmetry that
//! forces both directions, and the compliance property `Δε > 0` rests on.

use fitting_analysis::indicators::{epsilon_additive, epsilon_pair};
use fitting_analysis::objectives::{ObjectiveSpace, Row};
use fitting_core::cast::count_to_f64;
use std::cmp::Ordering;

/// A front point that scores *v* on every objective.
/// The space these fixtures are built in. The ε-indicator is defined for any
/// dimension, so one is enough — `epsilon_is_dimension_agnostic` covers the
/// other.
const SPACE: ObjectiveSpace = ObjectiveSpace::Current6;

fn flat(v: f64) -> Row {
    vec![v; SPACE.len()]
}

/// A front point whose objectives all differ, so a test cannot pass by symmetry.
/// Derived from the arity rather than written out, so growing the objective set
/// does not turn every literal row into a size mismatch.
fn ramp(base: f64) -> Row {
    (0..SPACE.len())
        .map(|j| base + 0.03 * count_to_f64(j % 4))
        .collect()
}

/// `epsilon_additive` on two non-empty fronts, unwrapped.
fn eps(a: &[Row], b: &[Row]) -> f64 {
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
    assert_eq!(
        pair.setting_vs_baseline.partial_cmp(&0.0),
        Some(Ordering::Equal)
    );
    assert_eq!(
        pair.baseline_vs_setting.partial_cmp(&0.0),
        Some(Ordering::Equal)
    );
    assert_eq!(pair.delta.partial_cmp(&0.0), Some(Ordering::Equal));
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
    assert_ne!(eps(&a, &b).partial_cmp(&eps(&b, &a)), Some(Ordering::Equal));
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
    let worse = ramp(0.3);
    let better: Row = worse.iter().map(|v| v + 0.1).collect();
    let reference = [flat(0.55), flat(0.45)];

    // Against a fixed reference, the dominating front needs no more of a shift…
    let (b1, w1) = ([better.clone()], [worse.clone()]);
    assert!(eps(&b1, &reference) <= eps(&w1, &reference));
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

    assert_eq!(
        eps(&[front.clone()], &b).partial_cmp(&eps(&[front.clone(), inside.clone()], &b)),
        Some(Ordering::Equal)
    );
    assert_eq!(
        eps(&b, &[front.clone()]).partial_cmp(&eps(&b, &[front, inside])),
        Some(Ordering::Equal)
    );
}

#[test]
fn the_indicator_is_order_independent() {
    let mut a = flat(0.5);
    a[3] = 0.9;
    let mut b = flat(0.6);
    b[4] = 0.2;
    let reference = [flat(0.45)];
    let ab = [a.clone(), b.clone()];
    let ba = [b, a];
    assert_eq!(
        eps(&ab, &reference).partial_cmp(&eps(&ba, &reference)),
        Some(Ordering::Equal)
    );
    assert_eq!(
        eps(&reference, &ab).partial_cmp(&eps(&reference, &ba)),
        Some(Ordering::Equal)
    );
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
