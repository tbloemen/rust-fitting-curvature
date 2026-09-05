//! Tests for the pieces `bin/exp1.rs` depends on.
//!
//! The binary itself is thin glue over `cell_summary` and `front_utilities`,
//! both of which have their own tests. What is new here, and so
//! what is worth pinning, is the ground-truth map and the properties the
//! Experiment 1 table's two headline numbers rely on.

use fitting_analysis::cell::{truth_of, GEOMETRIES, SYNTH_TRUTH};
use fitting_analysis::objectives::N_OBJECTIVES;
use fitting_analysis::r2::{front_utilities, r2, Weights};
use fitting_analysis::TrialRecord;

// ─── κ on the embedding gauge ────────────────────────────────────────────────

/// Euclidean sweeps write `curvature: 0.0` and omit `curvature_magnitude`
/// entirely, so `kappa()` has to read `|K|` off `curvature` or every Euclidean
/// cell reports no κ. `K = 0` makes `κ = 0` exactly — a value, not a gap.
#[test]
fn kappa_falls_back_to_curvature_for_euclidean() {
    let record: TrialRecord = serde_json::from_str(
        r#"{"geometry":"euclidean","curvature":0.0,"r_rms":7.869024392227243}"#,
    )
    .expect("euclidean trial shape");

    assert_eq!(
        record.curvature_magnitude, None,
        "fixture must mimic a real euclidean trial"
    );
    assert_eq!(record.kappa(), Some(0.0));
}

/// The fallback must not shadow a recorded magnitude, which is what every
/// curved trial carries.
#[test]
fn kappa_prefers_the_recorded_magnitude() {
    let record: TrialRecord = serde_json::from_str(
        r#"{"curvature":-0.1587415548667954,
             "curvature_magnitude":0.1587415548667954,
             "r_rms":4.583735477895421}"#,
    )
    .expect("hyperbolic trial shape");

    let expected = 0.1587415548667954 * 4.583735477895421_f64.powi(2);
    let got = record.kappa().expect("curved trial has kappa");
    assert!((got - expected).abs() < 1e-12, "got {got}, want {expected}");
}

/// `|K|` from a signed curvature, so the sign convention cannot leak into κ.
#[test]
fn kappa_is_unsigned() {
    let neg: TrialRecord = serde_json::from_str(r#"{"curvature":-0.25,"r_rms":2.0}"#).unwrap();
    let pos: TrialRecord = serde_json::from_str(r#"{"curvature":0.25,"r_rms":2.0}"#).unwrap();
    assert_eq!(neg.kappa(), Some(1.0));
    assert_eq!(pos.kappa(), Some(1.0));
}

/// A record with no curvature at all is still missing, not zero — the fallback
/// must not manufacture a κ for a row that never recorded one.
#[test]
fn kappa_is_none_without_any_curvature() {
    let no_curvature: TrialRecord = serde_json::from_str(r#"{"r_rms":3.0}"#).unwrap();
    assert_eq!(no_curvature.kappa(), None);

    let no_radius: TrialRecord = serde_json::from_str(r#"{"curvature":0.0}"#).unwrap();
    assert_eq!(no_radius.kappa(), None);
}

// ─── The ground-truth map ────────────────────────────────────────────────────

#[test]
fn every_synthetic_dataset_has_a_valid_truth() {
    for (dataset, truth) in SYNTH_TRUTH {
        assert!(
            GEOMETRIES.contains(&truth),
            "{dataset} claims geometry {truth}, which is not one of {GEOMETRIES:?}"
        );
        assert_eq!(truth_of(dataset), Some(truth));
    }
}

#[test]
fn truth_covers_the_optimizer_synthetic_set() {
    // `crates/optimizer/src/data.rs::load_synthetic` accepts exactly these five
    // names. Experiment 1 filters cells by `truth_of(..).is_some()`, so a name
    // missing here would silently drop a dataset from the table rather than
    // failing.
    for dataset in [
        "sphere",
        "antipodal_clusters",
        "tree",
        "hyperbolic_shells",
        "grid",
    ] {
        assert!(
            truth_of(dataset).is_some(),
            "{dataset} is an optimizer synthetic dataset with no ground truth"
        );
    }
    // Real datasets must stay absent: their geometry is the question.
    for dataset in ["mnist", "fashion_mnist", "pbmc", "wordnet_mammals"] {
        assert_eq!(truth_of(dataset), None, "{dataset} must not carry a truth");
    }
    // A curved dataset for each sign, so the table always has a matched row to
    // compare against the Euclidean baseline.
    let truths: Vec<&str> = SYNTH_TRUTH.iter().map(|(_, t)| *t).collect();
    for geometry in GEOMETRIES {
        assert!(
            truths.contains(&geometry),
            "no synthetic dataset is {geometry}"
        );
    }
}

// ─── Properties the table's headline numbers depend on ───────────────────────

fn point(fill: f64) -> [f64; N_OBJECTIVES] {
    [fill; N_OBJECTIVES]
}

/// Build a small spread-out front so the indicator has something to work with.
fn sample_front() -> Vec<[f64; N_OBJECTIVES]> {
    let mut a = point(0.5);
    a[0] = 0.9;
    let mut b = point(0.5);
    b[1] = 0.9;
    let mut c = point(0.6);
    c[2] = 0.7;
    vec![a, b, c]
}

/// `ΔR2^W = R2(front) − R2(front ∪ {w})` is the table's fair Wilson number
/// precisely because it cannot be negative: adding a point to a set can only
/// lower a cost. If this ever failed, a negative entry would read as "the
/// closed-form solution hurt the front", which is not a thing that can happen.
#[test]
fn adding_a_point_never_raises_r2() {
    let weights = Weights::new();
    let front = sample_front();
    let base = front_utilities(&front, &weights.vectors);

    for fill in [0.0, 0.3, 0.55, 0.95] {
        let mut augmented = front.clone();
        augmented.push(point(fill));
        let after = front_utilities(&augmented, &weights.vectors);

        for region in &weights.regions {
            let before = r2(&base, region);
            let now = r2(&after, region);
            assert!(
                now <= before + 1e-12,
                "region {}: adding a {fill}-point raised R2 from {before} to {now}",
                region.name
            );
        }
    }
}

/// The counterpart caveat, and the reason the singleton's R2 is not tabulated:
/// a one-point set is scored strictly worse than a front containing it, so
/// putting the two side by side would compare cardinality, not quality.
#[test]
fn a_singleton_never_beats_a_front_containing_it() {
    let weights = Weights::new();
    let front = sample_front();
    let singleton = vec![front[0]];

    let full = front_utilities(&front, &weights.vectors);
    let one = front_utilities(&singleton, &weights.vectors);

    for region in &weights.regions {
        assert!(
            r2(&one, region) >= r2(&full, region) - 1e-12,
            "region {}: a singleton scored better than a superset",
            region.name
        );
    }
}

/// ΔR2 is formed baseline-minus-row because R2 is a cost, so a *lower* R2 for
/// the curved arm must come out as a *positive* ΔR2. This is the sign
/// convention `aggregate.rs` uses and the easiest thing in the analysis to get
/// backwards.
#[test]
fn delta_r2_is_positive_when_the_row_beats_the_baseline() {
    let weights = Weights::new();

    // A better front: every objective higher, so its R2 is lower.
    let good = vec![point(0.9)];
    let poor = vec![point(0.4)];

    let good_u = front_utilities(&good, &weights.vectors);
    let poor_u = front_utilities(&poor, &weights.vectors);

    for region in &weights.regions {
        let r2_good = r2(&good_u, region);
        let r2_poor = r2(&poor_u, region);
        assert!(r2_good < r2_poor, "R2 should be lower for the better front");
        // baseline (poor, standing in for euclidean) minus row (good)
        assert!(
            r2_poor - r2_good > 0.0,
            "region {}: an improvement must give a positive delta",
            region.name
        );
    }
}
