//! Cell parsing, objective orientation, and Pareto non-domination.

use fitting_analysis::objectives::{oriented_row, oriented_value, ObjectiveSpace, OBJECTIVES};
use fitting_analysis::pareto::{slice_front_2d, step_polyline};
use fitting_analysis::{
    pareto_front_mask, pareto_front_records, parse_cell_stem, parse_cell_stem_variant, TrialRecord,
    Variant,
};
use fitting_core::metrics::{Direction, MetricValue, MetricValues};
use std::cmp::Ordering;

/// The manifold columns are populated too. They are no longer objectives, so
/// `oriented_row` ignores them — which is part of what the row test checks.
/// A record scoring *v* on every objective, oriented so that a larger `v` is a
/// better record: `normalized_stress` is minimised, so it gets `1 - v`.
///
/// Built through `MetricValues` rather than as a struct literal, since the
/// metric columns are one flattened block now.
fn metrics_at(v: f64) -> MetricValues {
    let mut m = MetricValues::MISSING;
    for metric in fitting_core::metrics::ALL {
        m.set(
            *metric,
            MetricValue::measured(match metric.direction() {
                Direction::Minimize => 1.0 - v,
                Direction::Maximize => v,
            }),
        );
    }
    m
}

fn record_at(v: f64) -> TrialRecord {
    TrialRecord {
        metrics: metrics_at(v),
        ..Default::default()
    }
}

// ─── Cell stems ───────────────────────────────────────────────────────────────

#[test]
fn parses_plain_stems() {
    let c = parse_cell_stem("all_off_mnist_hyperbolic").unwrap();
    assert_eq!(
        (
            c.setting.as_str(),
            c.dataset.as_str(),
            c.n,
            c.geometry.as_str()
        ),
        ("all_off", "mnist", 1000, "hyperbolic")
    );

    let c = parse_cell_stem("all_free_fashion_mnist_n5000_euclidean").unwrap();
    assert_eq!(
        (
            c.setting.as_str(),
            c.dataset.as_str(),
            c.n,
            c.geometry.as_str()
        ),
        ("all_free", "fashion_mnist", 5000, "euclidean")
    );
}

#[test]
fn dataset_names_that_contain_a_geometry_still_split_correctly() {
    let c = parse_cell_stem("centering_only_hyperbolic_shells_hyperbolic").unwrap();
    assert_eq!(c.dataset, "hyperbolic_shells");
    assert_eq!(c.geometry, "hyperbolic");
    assert_eq!(c.n, 1000);

    let c = parse_cell_stem("norm_only_sphere_n5000_spherical").unwrap();
    assert_eq!(c.dataset, "sphere");
    assert_eq!(c.geometry, "spherical");
    assert_eq!(c.n, 5000);
}

/// The `_rgyr` re-run's stems must parse to the *same* cell as the originals —
/// that is what lets `--results-dir results-rgyr` render every existing figure
/// unchanged — while still reporting the marker so the two can be told apart.
#[test]
fn variant_marker_is_stripped_but_reported() {
    let plain = parse_cell_stem("all_off_sphere_n5000_spherical").unwrap();
    let (marked, variant) = parse_cell_stem_variant("all_off_sphere_n5000_spherical_rgyr").unwrap();

    assert_eq!(marked, plain, "a variant must not change the cell identity");
    assert_eq!(variant, Some(Variant::Rgyr));
    assert_eq!(Variant::Rgyr.suffix(), "rgyr");
    assert_eq!(Variant::from_suffix("rgyr"), Some(Variant::Rgyr));
    assert_eq!(Variant::from_suffix("nope"), None);
    assert_eq!(
        parse_cell_stem_variant("all_off_sphere_n5000_spherical")
            .unwrap()
            .1,
        None
    );

    // The marker sits after the geometry, so the geometry anchor has to survive
    // it — without the strip this stem parses as nothing at all.
    assert_eq!(marked.geometry, "spherical");
    assert_eq!(marked.dataset, "sphere");
    assert_eq!(marked.n, 5000);
}

#[test]
fn rejects_front_files_and_junk() {
    // Front files carry a second geometry token and must never be read as cells.
    assert!(parse_cell_stem("all_off_mnist_hyperbolic_pareto_mnist_hyperbolic").is_none());
    assert!(parse_cell_stem("kappa_data").is_none());
    assert!(parse_cell_stem("results").is_none());
    assert!(parse_cell_stem("all_off_mnist_toroidal").is_none());
    assert!(parse_cell_stem("all_off__hyperbolic").is_none());
}

// ─── Orientation ──────────────────────────────────────────────────────────────

#[test]
fn orientation_flips_minimised_objectives() {
    assert_eq!(
        oriented_value("trustworthiness", Some(0.8)).partial_cmp(&0.8),
        Some(Ordering::Equal)
    );
    assert_eq!(
        oriented_value("normalized_stress", Some(0.3)).partial_cmp(&0.7),
        Some(Ordering::Equal)
    );

    // The manifold reading of a minimised metric is minimised too, and is now
    // flipped alongside its twin. It used to be left alone, because `MINIMIZE`
    // listed objective names and `normalized_stress_manifold` is not an
    // objective — but the manifold-vs-projection figure deleted in `b43c731`
    // did look it up, to difference the two readings of one configuration.
    // Under the old
    // orientation that panel
    // subtracted an unoriented manifold stress from an oriented projected one,
    // so an embedding whose two readings agreed at 0.1 plotted a gap of −0.8
    // instead of ~0. Orientation is a property of the metric, not of whether it
    // happens to be searched.
    assert_eq!(
        oriented_value("normalized_stress_manifold", Some(0.3)).partial_cmp(&0.7),
        Some(Ordering::Equal)
    );

    // A name that is not a metric at all is not flipped — nothing to consult.
    assert_eq!(
        oriented_value("not_a_metric", Some(0.3)).partial_cmp(&0.3),
        Some(Ordering::Equal)
    );
}

#[test]
fn missing_and_non_finite_values_score_worst() {
    // Matches the optimizer's metrics_to_vec substitution: a diverged trial is
    // scored as bad rather than dropped.
    assert_eq!(
        oriented_value("trustworthiness", None).partial_cmp(&0.0),
        Some(Ordering::Equal)
    );
    assert_eq!(
        oriented_value("trustworthiness", Some(f64::NAN)).partial_cmp(&0.0),
        Some(Ordering::Equal)
    );
    assert_eq!(
        oriented_value("trustworthiness", Some(f64::INFINITY)).partial_cmp(&0.0),
        Some(Ordering::Equal)
    );
    assert_eq!(
        oriented_value("normalized_stress", None).partial_cmp(&0.0),
        Some(Ordering::Equal)
    );
    assert_eq!(
        oriented_value("normalized_stress", Some(f64::NAN)).partial_cmp(&0.0),
        Some(Ordering::Equal)
    );
}

#[test]
fn orientation_clamps_out_of_range_values() {
    assert_eq!(
        oriented_value("trustworthiness", Some(1.4)).partial_cmp(&1.0),
        Some(Ordering::Equal)
    );
    assert_eq!(
        oriented_value("trustworthiness", Some(-0.2)).partial_cmp(&0.0),
        Some(Ordering::Equal)
    );
    // Stress above 1 orients below 0 and clamps up.
    assert_eq!(
        oriented_value("normalized_stress", Some(2.0)).partial_cmp(&0.0),
        Some(Ordering::Equal)
    );
}

#[test]
fn oriented_row_covers_all_six_objectives_in_order() {
    let space = ObjectiveSpace::Current6;
    assert_eq!(OBJECTIVES.len(), space.len());
    let row = oriented_row(&record_at(0.6), space);
    assert_eq!(row, vec![0.6; space.len()]);
    // An empty record is the all-zeros worst case.
    assert_eq!(
        oriented_row(&TrialRecord::default(), space),
        vec![0.0; space.len()]
    );
}

/// The legacy space is ten wide and interleaved: a metric's projected reading
/// then its manifold one. A record whose manifold columns read lower must show
/// that at the odd indices, which is what makes the two halves separable — the
/// whole reason the space existed.
#[test]
fn the_legacy_space_interleaves_projected_and_manifold_readings() {
    let space = ObjectiveSpace::Legacy10;
    assert_eq!(space.len(), 10);
    for (j, metric) in space.metrics().iter().enumerate() {
        let manifold = metric.name().ends_with("_manifold");
        assert_eq!(
            manifold,
            j % 2 == 1,
            "objective {j} is `{}`, which breaks the (projected, manifold) interleaving \
             that `build_regions` indexes by parity",
            metric.name()
        );
    }
}

// ─── Non-domination ───────────────────────────────────────────────────────────

/// A constant row of the current space's width.
fn flat(v: f64) -> fitting_analysis::objectives::Row {
    vec![v; ObjectiveSpace::Current6.len()]
}

#[test]
fn front_keeps_only_non_dominated_rows() {
    let a = flat(0.9);
    let b = flat(0.5); // dominated by a in every objective
    let mut c = flat(0.5);
    c[0] = 1.0; // better than a in one objective → non-dominated
    assert_eq!(pareto_front_mask(&[a, b, c]), vec![true, false, true]);
}

#[test]
fn identical_rows_are_all_kept() {
    // No row strictly dominates an identical one, so duplicates survive.
    let a = flat(0.7);
    assert_eq!(
        pareto_front_mask(&[a.clone(), a.clone(), a]),
        vec![true, true, true]
    );
}

#[test]
fn weak_domination_needs_a_strict_improvement() {
    let a = flat(0.5);
    let mut b = flat(0.5);
    b[3] = 0.6;
    // b is >= a everywhere and strictly better in one → a is dropped.
    assert_eq!(pareto_front_mask(&[a, b]), vec![false, true]);
}

#[test]
fn empty_input_gives_empty_front() {
    assert!(pareto_front_mask(&[]).is_empty());
    assert!(pareto_front_records(&[], ObjectiveSpace::Current6).is_empty());
}

#[test]
fn front_records_round_trip_through_orientation() {
    // 0.9 dominates 0.5 on every objective once stress is oriented, so only the
    // better record survives — this is the path cell_summary uses.
    let records = vec![record_at(0.5), record_at(0.9), record_at(0.7)];
    let front = pareto_front_records(&records, ObjectiveSpace::Current6);
    assert_eq!(front.len(), 1);
    assert_eq!(front[0].objective("trustworthiness"), Some(0.9));
}

// ─── 2D front cross-section ───────────────────────────────────────────────────

#[test]
fn slice_front_2d_takes_the_best_tradeoffs() {
    // x = trustworthiness (higher better), y = stress (lower better).
    // (0.9, 0.1) and (0.5, 0.05) are both non-dominated; (0.5, 0.5) is dominated
    // by both and (0.4, 0.6) by everything.
    let x = [0.9, 0.5, 0.5, 0.4];
    let y = [0.1, 0.05, 0.5, 0.6];
    let idx = slice_front_2d(&x, &y, true, false);
    assert_eq!(idx, vec![1, 0], "front must be sorted by x ascending");
}

#[test]
fn slice_front_2d_is_not_the_worst_case_boundary() {
    // Guards the bug in the Python `_slice_front` this replaces, whose
    // domination test was inverted: with one clearly-best and one clearly-worst
    // point, the front is the best one.
    let x = [0.9, 0.5];
    let y = [0.1, 0.5];
    assert_eq!(slice_front_2d(&x, &y, true, false), vec![0]);
}

#[test]
fn slice_front_2d_respects_axis_orientation() {
    // Same points, but now both axes are "higher is better": the front flips.
    let x = [0.9, 0.5];
    let y = [0.1, 0.5];
    assert_eq!(slice_front_2d(&x, &y, true, true), vec![1, 0]);
}

// ─── Attainment staircase ─────────────────────────────────────────────────────

#[test]
fn step_polyline_risers_come_before_treads() {
    // Asking for more trustworthiness than (0.9, 0.30) gives costs the next
    // point's stress immediately, so the riser sits at the earlier x.
    let front = [(0.9, 0.30), (0.95, 0.35), (0.97, 0.40)];
    assert_eq!(
        step_polyline(&front),
        vec![
            (0.9, 0.30),
            (0.9, 0.35),
            (0.95, 0.35),
            (0.95, 0.40),
            (0.97, 0.40),
        ]
    );
}

#[test]
fn step_polyline_passes_through_degenerate_fronts() {
    assert!(step_polyline(&[]).is_empty());
    assert_eq!(step_polyline(&[(0.9, 0.3)]), vec![(0.9, 0.3)]);
}
