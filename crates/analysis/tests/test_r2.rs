//! The R2 indicator: weight-simplex enumeration, the preference regions, the
//! compliance property the ΔR2 claim rests on, and the recommendation.

use fitting_analysis::objectives::{N_OBJECTIVES, OBJECTIVES};
use fitting_analysis::r2::{
    cell_summary, front_utilities, metric_objectives, r2, recommendation, Weights, METRICS,
    REGION_ALL, REGION_MANIFOLD, REGION_PROJECTED, S,
};
use fitting_analysis::TrialRecord;

/// A front point that scores *v* on every objective.
fn flat(v: f64) -> [f64; N_OBJECTIVES] {
    [v; N_OBJECTIVES]
}

/// R2 of a front under a named region.
fn score(front: &[[f64; N_OBJECTIVES]], w: &Weights, region: &str) -> f64 {
    let u = front_utilities(front, &w.vectors);
    r2(&u, w.region(region).expect("region exists"))
}

// ─── The weight simplex ──────────────────────────────────────────────────────

#[test]
fn simplex_has_the_expected_size_and_every_vector_sums_to_one() {
    let w = Weights::new();
    // C(S + k - 1, k - 1) = C(14, 9) = 2002 for k = 10, S = 5.
    assert_eq!(w.vectors.len(), 2002);
    assert_eq!(w.counts.len(), w.vectors.len());

    for (counts, lambda) in w.counts.iter().zip(&w.vectors) {
        let total: u32 = counts.iter().map(|&c| u32::from(c)).sum();
        assert_eq!(total, S as u32, "counts {counts:?} must sum to S");
        let sum: f64 = lambda.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "λ {lambda:?} sums to {sum}");
    }
}

#[test]
fn simplex_vectors_are_distinct() {
    let w = Weights::new();
    let mut seen: Vec<[u8; N_OBJECTIVES]> = w.counts.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), w.counts.len());
}

// ─── Preference regions ──────────────────────────────────────────────────────

#[test]
fn metric_layout_matches_the_objective_order() {
    // The regions are built from index arithmetic on OBJECTIVES; if that array
    // is ever reordered, this is the test that catches it.
    for (i, metric) in METRICS.iter().enumerate() {
        let (proj, man) = metric_objectives(i);
        assert_eq!(OBJECTIVES[proj], *metric);
        assert_eq!(OBJECTIVES[man], format!("{metric}_manifold"));
    }
}

#[test]
fn region_sizes_match_the_combinatorics() {
    let w = Weights::new();

    assert_eq!(w.region(REGION_ALL).unwrap().indices.len(), 2002);

    // Supported on 5 of the 10 objectives: compositions of 5 into 5 parts,
    // C(9, 4) = 126.
    assert_eq!(w.region(REGION_MANIFOLD).unwrap().indices.len(), 126);
    assert_eq!(w.region(REGION_PROJECTED).unwrap().indices.len(), 126);

    // At least 3 of 5 units on the metric's two objectives:
    //   t=3: 4·C(9,7)=144, t=4: 5·C(8,7)=40, t=5: 6·C(7,7)=6  ⇒ 190.
    for metric in METRICS {
        assert_eq!(
            w.region(metric).unwrap().indices.len(),
            190,
            "region {metric}"
        );
    }
}

#[test]
fn surface_regions_are_supported_on_their_own_objectives() {
    let w = Weights::new();
    for &i in &w.region(REGION_MANIFOLD).unwrap().indices {
        for j in (0..N_OBJECTIVES).step_by(2) {
            assert_eq!(w.counts[i][j], 0, "projected objective {j} carries weight");
        }
    }
    for &i in &w.region(REGION_PROJECTED).unwrap().indices {
        for j in (1..N_OBJECTIVES).step_by(2) {
            assert_eq!(w.counts[i][j], 0, "manifold objective {j} carries weight");
        }
    }
}

#[test]
fn a_metric_region_ignores_the_other_surfaces_objectives() {
    // Two fronts identical on the manifold objectives and different on the
    // projected ones must score the same under W_manifold.
    let w = Weights::new();
    let mut a = flat(0.6);
    let mut b = flat(0.6);
    for j in (0..N_OBJECTIVES).step_by(2) {
        a[j] = 0.1;
        b[j] = 0.9;
    }
    assert_eq!(
        score(&[a], &w, REGION_MANIFOLD),
        score(&[b], &w, REGION_MANIFOLD)
    );
    // ...and differently under the projected region, or the regions would be
    // measuring the same thing.
    assert_ne!(
        score(&[a], &w, REGION_PROJECTED),
        score(&[b], &w, REGION_PROJECTED)
    );
}

// ─── The indicator ───────────────────────────────────────────────────────────

#[test]
fn the_ideal_point_scores_zero_and_the_nadir_scores_worst() {
    let w = Weights::new();
    assert_eq!(score(&[flat(1.0)], &w, REGION_ALL), 0.0);

    // A front at the origin gives max_j λ_j per weight vector, which is what an
    // empty front is defined to score too.
    let nadir = score(&[flat(0.0)], &w, REGION_ALL);
    let empty = score(&[], &w, REGION_ALL);
    assert_eq!(nadir, empty);
    assert!(nadir > 0.0);
}

#[test]
fn the_indicator_is_weakly_pareto_compliant() {
    // The property ΔR2 > 0 rests on: a dominating front can never score worse.
    let w = Weights::new();
    let worse = [0.3, 0.4, 0.5, 0.2, 0.6, 0.7, 0.1, 0.4, 0.5, 0.3];
    let better = [0.4, 0.5, 0.5, 0.3, 0.8, 0.7, 0.2, 0.6, 0.5, 0.9];
    for region in Weights::new().region_names() {
        let b = score(&[better], &w, region);
        let a = score(&[worse], &w, region);
        assert!(b <= a, "region {region}: dominating front scored {b} > {a}");
    }
}

#[test]
fn dominated_points_do_not_change_the_indicator() {
    // Why reducing to the Pareto front first is exact rather than an
    // approximation.
    let w = Weights::new();
    let mut front = flat(0.8);
    front[0] = 0.9;
    let inside = flat(0.4);
    assert_eq!(
        score(&[front], &w, REGION_ALL),
        score(&[front, inside], &w, REGION_ALL)
    );
}

#[test]
fn adding_a_non_dominated_point_can_only_help() {
    let w = Weights::new();
    let mut a = flat(0.2);
    a[0] = 0.95;
    let mut b = flat(0.2);
    b[1] = 0.95;
    let both = score(&[a, b], &w, REGION_ALL);
    assert!(both <= score(&[a], &w, REGION_ALL));
    assert!(both <= score(&[b], &w, REGION_ALL));
}

#[test]
fn the_indicator_is_order_independent() {
    let w = Weights::new();
    let mut a = flat(0.5);
    a[3] = 0.9;
    let mut b = flat(0.6);
    b[7] = 0.2;
    assert_eq!(
        score(&[a, b], &w, REGION_ALL),
        score(&[b, a], &w, REGION_ALL)
    );
}

// ─── Recommendations ─────────────────────────────────────────────────────────

#[test]
fn a_metric_region_recommends_the_point_that_is_good_at_that_metric() {
    let w = Weights::new();
    // Point 0 is strong on trustworthiness (objectives 0, 1) and weak elsewhere;
    // point 1 is the mirror image on neighbourhood hit (objectives 8, 9).
    let mut trust = flat(0.2);
    trust[0] = 0.95;
    trust[1] = 0.95;
    let mut hit = flat(0.2);
    hit[8] = 0.95;
    hit[9] = 0.95;
    let front = [trust, hit];
    let u = front_utilities(&front, &w.vectors);

    let rec = recommendation(&u, w.region("trustworthiness").unwrap()).unwrap();
    assert_eq!(rec.front_index, 0);
    assert!(rec.share > 0.5, "share was {}", rec.share);

    let rec = recommendation(&u, w.region("neighborhood_hit").unwrap()).unwrap();
    assert_eq!(rec.front_index, 1);
}

#[test]
fn recommendation_ties_resolve_to_the_lowest_front_index() {
    let w = Weights::new();
    let front = [flat(0.5), flat(0.5)];
    let u = front_utilities(&front, &w.vectors);
    let rec = recommendation(&u, w.region(REGION_ALL).unwrap()).unwrap();
    assert_eq!(rec.front_index, 0);
    assert_eq!(rec.share, 1.0);
}

// ─── Cell summary ────────────────────────────────────────────────────────────

/// A record scoring *v* on every maximised objective; stress is stored raw, so
/// `1 - v` there gives an oriented value of *v* as well.
fn record(v: f64) -> TrialRecord {
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

#[test]
fn cell_summary_indexes_the_front_back_into_the_records() {
    let w = Weights::new();
    // Record 1 dominates both others, so the front is exactly [1].
    let records = vec![record(0.3), record(0.9), record(0.5)];
    let summary = cell_summary(&records, &w);

    assert_eq!(summary.n_trials, 3);
    assert_eq!(summary.n_front, 1);
    assert_eq!(summary.front, vec![1]);

    // Every region scores it and recommends it, and `front_index` is an index
    // into `front`, not into `records` — the recommendation table depends on it.
    for region in w.region_names() {
        assert!(summary.r2.contains_key(region), "region {region} missing");
        let rec = &summary.recommended[region];
        assert_eq!(rec.front_index, 0);
        assert_eq!(summary.front[rec.front_index], 1);
    }
}

#[test]
fn a_diverged_trial_scores_worst_rather_than_vanishing() {
    let w = Weights::new();
    let mut diverged = record(0.9);
    diverged.trustworthiness = None;
    diverged.continuity_manifold = Some(f64::NAN);

    let good = cell_summary(&[record(0.9)], &w);
    let bad = cell_summary(&[diverged], &w);
    assert!(bad.r2[REGION_ALL] > good.r2[REGION_ALL]);
}
