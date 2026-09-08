//! The R2 indicator: weight-simplex enumeration, the preference regions, the
//! compliance property the ΔR2 claim rests on, and the recommendation.

use fitting_analysis::objectives::{
    FAMILIES, METRIC_PAIRS, N_METRIC_PAIRS, N_OBJECTIVES, OBJECTIVES,
};
use fitting_analysis::r2::{
    cell_summary, front_utilities, r2, recommendation, Weights, REGION_ALL,
};
use fitting_analysis::TrialRecord;
use fitting_core::metrics::{Direction, MetricValue, MetricValues, CONTINUITY, TRUSTWORTHINESS};

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
    // C(s + k - 1, k - 1) = C(9, 4) = 126 for k = 5, s = 5.
    assert_eq!(w.s, Weights::DEFAULT_S);
    assert_eq!(w.vectors.len(), 126);
    assert_eq!(w.counts.len(), w.vectors.len());

    for (counts, lambda) in w.counts.iter().zip(&w.vectors) {
        let total: u32 = counts.iter().map(|&c| u32::from(c)).sum();
        assert_eq!(total, w.s as u32, "counts {counts:?} must sum to s");
        let sum: f64 = lambda.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "λ {lambda:?} sums to {sum}");
    }
}

#[test]
fn resolution_flows_through_the_enumeration_and_the_regions() {
    let w = Weights::with_resolution(2);
    // C(2 + 4, 4) = 15, and "at least half the mass" is now a count of 1.
    assert_eq!(w.s, 2);
    assert_eq!(w.vectors.len(), 15);
    for counts in &w.counts {
        let total: u32 = counts.iter().map(|&c| u32::from(c)).sum();
        assert_eq!(total, 2, "counts {counts:?} must sum to s");
    }
    // The families partition the objectives, so every vector puts at least one
    // of its two units on some family's objectives and the family regions have
    // to cover the whole simplex.
    let covered: std::collections::BTreeSet<usize> = FAMILIES
        .iter()
        .flat_map(|(f, _)| w.region(f).expect("family region exists").indices.clone())
        .collect();
    assert_eq!(covered.len(), w.vectors.len());
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
fn every_pair_follows_the_manifold_naming_convention() {
    // METRIC_PAIRS is derived by matching `QualityMetric::base`, so the pairing
    // itself can no longer be wrong. What is still worth pinning is the
    // *naming*: `_manifold` is the suffix that ties a pair's second member to
    // its JSONL column, and Exp 4's panel captions read the first member's.
    for (projected, manifold) in METRIC_PAIRS.iter() {
        assert_eq!(manifold.name(), format!("{}_manifold", projected.name()));
    }
    assert_eq!(METRIC_PAIRS.len(), N_METRIC_PAIRS);
}

#[test]
fn metric_pairs_has_the_declared_length() {
    // `N_METRIC_PAIRS` is a `const` because `figures/exp4.rs` uses it as an
    // array length, so it is the one count restated rather than derived.
    assert_eq!(METRIC_PAIRS.len(), N_METRIC_PAIRS);
}

#[test]
fn every_objective_resolves_on_a_record() {
    // `oriented_row` still resolves objectives by *name*, so a metric whose
    // wire name does not round-trip reads as permanently missing — silently
    // worst-case rather than an error.
    let r = record(0.5);
    for metric in OBJECTIVES {
        assert!(
            r.objective(metric.name()).is_some(),
            "{metric} does not resolve on a fully-populated record"
        );
    }
}

#[test]
fn region_sizes_match_the_combinatorics() {
    let w = Weights::new();

    assert_eq!(w.region(REGION_ALL).unwrap().indices.len(), 126);

    // At least 3 of 5 units on the family's own objectives, the remaining
    // objectives taking the rest. For a two-objective family out of five:
    //   t=3: 4·C(4,2)=24, t=4: 5·C(3,2)=15, t=5: 6·C(2,2)=6  ⇒ 45.
    // A one-objective family is the single-objective count below, 15 — which
    // is exactly why `class_separation` currently duplicates the
    // `neighborhood_hit` region.
    for (family, members) in FAMILIES {
        let want = if members.len() == 2 { 45 } else { 15 };
        assert_eq!(
            w.region(family).unwrap().indices.len(),
            want,
            "region {family}"
        );
    }

    // At least 3 of 5 units on one objective, the other four taking the rest:
    //   l=3: C(5,3)=10, l=4: C(4,3)=4, l=5: 1  ⇒ 15.
    for objective in OBJECTIVES {
        assert_eq!(
            w.region(objective.name()).unwrap().indices.len(),
            15,
            "region {objective}"
        );
    }
}

#[test]
fn families_partition_the_objectives() {
    // FAMILIES indexes into OBJECTIVES by position, so a reordering of either
    // silently regroups the regions. Every objective must belong to exactly one
    // family.
    let mut seen: Vec<usize> = FAMILIES
        .iter()
        .flat_map(|(_, idx)| idx.iter().copied())
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..N_OBJECTIVES).collect::<Vec<_>>());
}

#[test]
fn a_family_region_puts_at_least_half_its_mass_on_its_own_objectives() {
    let w = Weights::new();
    let half = w.s.div_ceil(2) as u8;
    for (family, members) in FAMILIES {
        for &i in &w.region(family).unwrap().indices {
            let c = &w.counts[i];
            let mass: u8 = members.iter().map(|&j| c[j]).sum();
            assert!(
                mass >= half,
                "region {family} admits {c:?}, which puts {mass} of {} units on it",
                w.s
            );
        }
    }
}

#[test]
fn a_family_region_penalises_its_own_objectives_hardest() {
    // The regions have to actually express different preferences, or reporting
    // them separately says nothing. Degrading a family's own objectives must
    // cost more under that family than degrading someone else's by the same
    // amount. (R2 is a cost, so "worse" is larger.)
    let w = Weights::new();
    let degrade = |members: &[usize]| {
        let mut p = flat(0.9);
        for &j in members {
            p[j] = 0.1;
        }
        p
    };
    for (family, own) in FAMILIES {
        let hurt_own = score(&[degrade(own)], &w, family);
        for (other_family, other) in FAMILIES {
            if other_family == family {
                continue;
            }
            let hurt_other = score(&[degrade(other)], &w, family);
            assert!(
                hurt_own > hurt_other,
                "under {family}, degrading {other_family} ({hurt_other}) cost at \
                 least as much as degrading {family} itself ({hurt_own})"
            );
        }
    }
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
    let worse = [0.3, 0.4, 0.5, 0.2, 0.6];
    let better = [0.4, 0.5, 0.5, 0.3, 0.8];
    for region in &w.regions {
        let (region, b, a) = (
            region.name.as_str(),
            score(&[better], &w, &region.name),
            score(&[worse], &w, &region.name),
        );
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
    b[4] = 0.2;
    assert_eq!(
        score(&[a, b], &w, REGION_ALL),
        score(&[b, a], &w, REGION_ALL)
    );
}

// ─── Recommendations ─────────────────────────────────────────────────────────

#[test]
fn a_metric_region_recommends_the_point_that_is_good_at_that_metric() {
    let w = Weights::new();
    // Point 0 is strong on trustworthiness (objective 0) and weak elsewhere;
    // point 1 is the mirror image on neighbourhood hit (objective 4).
    let mut trust = flat(0.2);
    trust[0] = 0.95;
    let mut hit = flat(0.2);
    hit[4] = 0.95;
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

fn record(v: f64) -> TrialRecord {
    TrialRecord {
        metrics: metrics_at(v),
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
    for region in &w.regions {
        let region = region.name.as_str();
        assert!(summary.r2.contains_key(region), "region {region} missing");
        let rec = &summary.recommended[region];
        assert_eq!(rec.front_index, 0);
        assert_eq!(summary.front[rec.front_index], 1);
    }
}

#[test]
fn a_diverged_trial_scores_worst_rather_than_vanishing() {
    let w = Weights::new();
    // The two ways a reading can carry no number. Both orient to the worst
    // case, which is what keeps a diverged trial from scoring well.
    let mut diverged = record(0.9);
    diverged.metrics.set(TRUSTWORTHINESS, MetricValue::Diverged);
    diverged.metrics.set(CONTINUITY, MetricValue::Absent);

    let good = cell_summary(&[record(0.9)], &w);
    let bad = cell_summary(&[diverged], &w);
    assert!(bad.r2[REGION_ALL] > good.r2[REGION_ALL]);
}
