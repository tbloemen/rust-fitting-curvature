//! The invariants the metric registry cannot get from the compiler.
//!
//! The trait shape gives up exhaustive matching: a metric type that exists but
//! is missing from `ALL` is simply never measured, and nothing errors. `ALL` is
//! therefore the one hand-maintained list left in the design, and these tests
//! are what guard it.

use fitting_core::metrics::{Direction, Family, Metric, Space, ALL, OBJECTIVES};

/// The wire names, in the order `MetricValues` serialises them.
///
/// This list is the JSONL column order of every results file written before the
/// registry existed, taken from `TrialResult`'s field order. A metric dropped
/// from `ALL`, added to it, or moved within it changes the schema of every new
/// results line — so that has to be a deliberate edit here, not a silent
/// consequence of editing `ALL`.
const EXPECTED: [&str; 13] = [
    "trustworthiness",
    "trustworthiness_manifold",
    "continuity",
    "continuity_manifold",
    "neighborhood_hit",
    "neighborhood_hit_manifold",
    "normalized_stress",
    "normalized_stress_manifold",
    "shepard_goodness",
    "shepard_goodness_manifold",
    "davies_bouldin_ratio",
    "dunn_index",
    "cluster_density_measure",
];

#[test]
fn all_holds_exactly_the_expected_metrics_in_order() {
    let got: Vec<&str> = ALL.iter().map(|m| m.name()).collect();
    assert_eq!(got, EXPECTED);
    assert_eq!(Metric::COUNT, EXPECTED.len());
}

#[test]
fn names_are_unique_and_round_trip() {
    for m in ALL {
        assert_eq!(
            Metric::by_name(m.name()),
            Some(*m),
            "{} does not parse back to itself",
            m.name()
        );
    }
    let mut names: Vec<&str> = ALL.iter().map(|m| m.name()).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(names.len(), before, "duplicate wire name in ALL");
}

#[test]
fn index_round_trips_through_all() {
    for (i, m) in ALL.iter().enumerate() {
        assert_eq!(m.index(), i, "{} reports the wrong slot", m.name());
    }
}

#[test]
fn an_unknown_name_does_not_parse() {
    // The retired columns still present throughout `results/`. They must read
    // as "not a metric" so deserialisation skips them rather than erroring.
    for retired in ["knn_overlap", "knn_overlap_manifold", "class_density_measure"] {
        assert_eq!(Metric::by_name(retired), None, "{retired} still parses");
    }
}

#[test]
fn objectives_are_bounded_projected_metrics() {
    for m in OBJECTIVES {
        assert!(ALL.contains(m), "{} is not in ALL", m.name());
        assert!(
            m.is_objective(),
            "{} is an objective but not bounded in [0, 1]",
            m.name()
        );
        assert_eq!(
            m.space(),
            Space::Projected,
            "{} is an objective but not read on the projection",
            m.name()
        );
    }
}

#[test]
fn every_bounded_projected_metric_is_an_objective() {
    // The converse of the test above: the objective set is not a hand-picked
    // subset of the eligible metrics, it *is* the eligible metrics. Marking a
    // new metric `is_objective` without adding it to `OBJECTIVES` — or the
    // reverse — is what this catches.
    for m in ALL.iter().filter(|m| m.is_objective()) {
        assert!(
            OBJECTIVES.contains(m),
            "{} claims to be an objective but is not in OBJECTIVES",
            m.name()
        );
    }
}

#[test]
fn objectives_are_grouped_by_family() {
    // `fitting_analysis::objectives::FAMILIES` indexes into OBJECTIVES by
    // position and needs each family contiguous. An interleaving here would
    // regroup every preference region silently.
    let families: Vec<Family> = OBJECTIVES.iter().map(|m| m.family()).collect();
    let mut seen = Vec::new();
    for f in families {
        if seen.last() != Some(&f) {
            assert!(!seen.contains(&f), "family {:?} is not contiguous", f);
            seen.push(f);
        }
    }
}

#[test]
fn only_normalized_stress_is_minimized() {
    for m in ALL {
        let want = if m.base() == "normalized_stress" {
            Direction::Minimize
        } else {
            Direction::Maximize
        };
        assert_eq!(m.direction(), want, "{} is oriented the wrong way", m.name());
    }
}

/// The spread diagnostics are not metrics, so `--metric r_max` is rejected
/// for the honest reason: there is no such metric.
///
/// They were briefly registry entries, which forced a `Family::Spread` that
/// emitted no region, a `Space::Ambient` meaning "neither", a `direction()`
/// with no meaning, and a `Metric::optimizable()` filter whose only job was to
/// take them back out again. They live in `fitting_core::spread` now.
#[test]
fn the_spread_diagnostics_are_not_metrics() {
    for d in ["r_max", "r_rms", "r_gyration"] {
        assert_eq!(Metric::by_name(d), None, "{d} still parses as a metric");
        assert!(
            !Metric::valid_names().contains(d),
            "{d} is offered to --metric"
        );
    }
    assert_eq!(ALL.len(), 13);
    assert!(Metric::valid_names().contains("trustworthiness"));
}

#[test]
fn dual_pairs_are_the_metrics_with_two_readings() {
    let pairs: Vec<(&str, &str)> = Metric::dual_pairs()
        .map(|(p, m)| (p.name(), m.name()))
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("trustworthiness", "trustworthiness_manifold"),
            ("continuity", "continuity_manifold"),
            ("neighborhood_hit", "neighborhood_hit_manifold"),
            ("normalized_stress", "normalized_stress_manifold"),
            ("shepard_goodness", "shepard_goodness_manifold"),
        ]
    );
    // A twin differs from its base reading in exactly one respect.
    for (p, m) in Metric::dual_pairs() {
        assert_eq!(p.base(), m.base());
        assert_eq!(p.family(), m.family());
        assert_eq!(p.direction(), m.direction());
        assert_ne!(p.space(), m.space());
    }
}

#[test]
fn only_the_paired_metrics_report_a_twin() {
    // `crates/web` suffixes `_2d` exactly for the metrics with a twin, and the
    // browser's metrics panel reads the rest by their bare wire name. Getting
    // this set wrong renames a key the panel is looking for, and the row simply
    // disappears from the UI with no error anywhere.
    let twinned: Vec<&str> = ALL.iter().filter(|m| m.has_twin()).map(|m| m.name()).collect();
    assert_eq!(
        twinned,
        vec![
            "trustworthiness",
            "trustworthiness_manifold",
            "continuity",
            "continuity_manifold",
            "neighborhood_hit",
            "neighborhood_hit_manifold",
            "normalized_stress",
            "normalized_stress_manifold",
            "shepard_goodness",
            "shepard_goodness_manifold",
        ]
    );
    for solo in ["davies_bouldin_ratio", "dunn_index", "cluster_density_measure"] {
        assert!(
            !Metric::by_name(solo).unwrap().has_twin(),
            "{solo} claims a twin"
        );
    }
}
