//! Cell parsing, objective orientation, and Pareto non-domination.

use fitting_analysis::objectives::{oriented_row, oriented_value, N_OBJECTIVES, OBJECTIVES};
use fitting_analysis::pareto::{slice_front_2d, step_polyline};
use fitting_analysis::{pareto_front_mask, pareto_front_records, parse_cell_stem, TrialRecord};

/// A record whose 10 objectives are all *v* except normalised stress, which is
/// set so that its oriented value is also *v* (stress is minimised).
fn record_at(v: f64) -> TrialRecord {
    TrialRecord {
        trustworthiness: Some(v),
        trustworthiness_manifold: Some(v),
        continuity: Some(v),
        continuity_manifold: Some(v),
        normalized_stress: Some(1.0 - v),
        normalized_stress_manifold: Some(1.0 - v),
        shepard_goodness: Some(v),
        shepard_goodness_manifold: Some(v),
        neighborhood_hit: Some(v),
        neighborhood_hit_manifold: Some(v),
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
    assert_eq!(oriented_value("trustworthiness", Some(0.8)), 0.8);
    assert_eq!(oriented_value("normalized_stress", Some(0.3)), 0.7);
    assert_eq!(oriented_value("normalized_stress_manifold", Some(0.3)), 0.7);
}

#[test]
fn missing_and_non_finite_values_score_worst() {
    // Matches the optimizer's metrics_to_vec substitution: a diverged trial is
    // scored as bad rather than dropped.
    assert_eq!(oriented_value("trustworthiness", None), 0.0);
    assert_eq!(oriented_value("trustworthiness", Some(f64::NAN)), 0.0);
    assert_eq!(oriented_value("trustworthiness", Some(f64::INFINITY)), 0.0);
    assert_eq!(oriented_value("normalized_stress", None), 0.0);
    assert_eq!(oriented_value("normalized_stress", Some(f64::NAN)), 0.0);
}

#[test]
fn orientation_clamps_out_of_range_values() {
    assert_eq!(oriented_value("trustworthiness", Some(1.4)), 1.0);
    assert_eq!(oriented_value("trustworthiness", Some(-0.2)), 0.0);
    // Stress above 1 orients below 0 and clamps up.
    assert_eq!(oriented_value("normalized_stress", Some(2.0)), 0.0);
}

#[test]
fn oriented_row_covers_all_ten_objectives_in_order() {
    assert_eq!(OBJECTIVES.len(), N_OBJECTIVES);
    let row = oriented_row(&record_at(0.6));
    assert_eq!(row, [0.6; N_OBJECTIVES]);
    // An empty record is the all-zeros worst case.
    assert_eq!(oriented_row(&TrialRecord::default()), [0.0; N_OBJECTIVES]);
}

// ─── Non-domination ───────────────────────────────────────────────────────────

#[test]
fn front_keeps_only_non_dominated_rows() {
    let a = [0.9; N_OBJECTIVES];
    let b = [0.5; N_OBJECTIVES]; // dominated by a in every objective
    let mut c = [0.5; N_OBJECTIVES];
    c[0] = 1.0; // better than a in one objective → non-dominated
    assert_eq!(pareto_front_mask(&[a, b, c]), vec![true, false, true]);
}

#[test]
fn identical_rows_are_all_kept() {
    // No row strictly dominates an identical one, so duplicates survive.
    let a = [0.7; N_OBJECTIVES];
    assert_eq!(pareto_front_mask(&[a, a, a]), vec![true, true, true]);
}

#[test]
fn weak_domination_needs_a_strict_improvement() {
    let a = [0.5; N_OBJECTIVES];
    let mut b = [0.5; N_OBJECTIVES];
    b[3] = 0.6;
    // b is >= a everywhere and strictly better in one → a is dropped.
    assert_eq!(pareto_front_mask(&[a, b]), vec![false, true]);
}

#[test]
fn empty_input_gives_empty_front() {
    assert!(pareto_front_mask(&[]).is_empty());
    assert!(pareto_front_records(&[]).is_empty());
}

#[test]
fn front_records_round_trip_through_orientation() {
    // 0.9 dominates 0.5 on every objective once stress is oriented, so only the
    // better record survives — this is the path cell_summary uses.
    let records = vec![record_at(0.5), record_at(0.9), record_at(0.7)];
    let front = pareto_front_records(&records);
    assert_eq!(front.len(), 1);
    assert_eq!(front[0].trustworthiness, Some(0.9));
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
