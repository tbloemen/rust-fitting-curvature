//! The R2 indicator: weight-simplex enumeration, the preference regions, the
//! compliance property the ΔR2 claim rests on, and the recommendation.

use fitting_analysis::objectives::{
    ObjectiveSpace, Row, FAMILIES, METRIC_PAIRS, N_METRIC_PAIRS, OBJECTIVES,
};
use fitting_analysis::r2::{
    cell_summary, front_utilities, r2, recommendation, Weights, REGION_ALL,
};
use fitting_analysis::TrialRecord;
use fitting_core::cast::count_to_f64;
use fitting_core::metrics::{Direction, MetricValue, MetricValues, CONTINUITY, TRUSTWORTHINESS};
use std::cmp::Ordering;

/// The space these fixtures are built in: the one the optimizer searches today.
/// The legacy space has its own tests below, since its region set is different.
const SPACE: ObjectiveSpace = ObjectiveSpace::Current6;

/// A front point that scores *v* on every objective.
fn flat(v: f64) -> Row {
    vec![v; SPACE.len()]
}

/// A front point whose objectives all differ, so a test cannot pass by symmetry.
///
/// Derived from the arity rather than written out: a literal row is one more
/// place the objective set has to be edited when it grows, and the compiler
/// reports it as a size mismatch a long way from the reason.
fn ramp(base: f64) -> Row {
    (0..SPACE.len())
        .map(|j| base + 0.03 * count_to_f64(j % 4))
        .collect()
}

/// `n` choose `k`, exactly, for the region combinatorics below.
fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    (0..k.min(n - k)).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

/// Count vectors of `k` non-negative integers summing to `s`: `C(s + k - 1, k - 1)`.
fn compositions(s: usize, k: usize) -> usize {
    if k == 0 {
        return usize::from(s == 0);
    }
    binom(s + k - 1, k - 1)
}

/// Size of the "at least half the mass on `m` of the `N_OBJECTIVES` axes" region.
///
/// Split on `t`, the mass the region's own axes carry: the members take one of
/// `compositions(t, m)` arrangements and the rest take one of
/// `compositions(s - t, N_OBJECTIVES - m)`.
fn region_size(m: usize, s: usize) -> usize {
    let half = s.div_ceil(2);
    (half..=s)
        .map(|t| compositions(t, m) * compositions(s - t, SPACE.len() - m))
        .sum()
}

/// R2 of a front under a named region.
fn score(front: &[Row], w: &Weights, region: &str) -> f64 {
    let u = front_utilities(front, &w.vectors);
    r2(&u, w.region(region).expect("region exists"))
}

// ─── The weight simplex ──────────────────────────────────────────────────────

#[test]
fn simplex_has_the_expected_size_and_every_vector_sums_to_one() {
    let w = Weights::new(SPACE);
    // C(s + k - 1, k - 1): the number of ways to split s units over k axes.
    assert_eq!(w.s, Weights::DEFAULT_S);
    assert_eq!(w.vectors.len(), compositions(w.s, SPACE.len()));
    assert_eq!(w.counts.len(), w.vectors.len());

    for (counts, lambda) in w.counts.iter().zip(&w.vectors) {
        let total: u32 = counts.iter().map(|&c| u32::from(c)).sum();
        assert_eq!(
            total,
            u32::try_from(w.s).expect("s is a small resolution"),
            "counts {counts:?} must sum to s"
        );
        let sum: f64 = lambda.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "λ {lambda:?} sums to {sum}");
    }
}

#[test]
fn resolution_flows_through_the_enumeration_and_the_regions() {
    let w = Weights::with_resolution(SPACE, 2);
    // Two units instead of five, so "at least half the mass" is now a count of 1.
    assert_eq!(w.s, 2);
    assert_eq!(w.vectors.len(), compositions(2, SPACE.len()));
    assert_ne!(w.vectors.len(), Weights::new(SPACE).vectors.len());
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
    let w = Weights::new(SPACE);
    let mut seen: Vec<Vec<u8>> = w.counts.clone();
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
    // its JSONL column, and a panel caption over such a pair reads the first
    // member's.
    for (projected, manifold) in METRIC_PAIRS.iter() {
        assert_eq!(manifold.name(), format!("{}_manifold", projected.name()));
    }
    assert_eq!(METRIC_PAIRS.len(), N_METRIC_PAIRS);
}

#[test]
fn metric_pairs_has_the_declared_length() {
    // `N_METRIC_PAIRS` is a `const` because it is used as an array length, so
    // it is the one count restated rather than derived.
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
    let w = Weights::new(SPACE);

    assert_eq!(w.region(REGION_ALL).unwrap().indices.len(), w.vectors.len());

    // At least half the mass on the family's own objectives, the remaining
    // objectives taking the rest — `region_size` is that split, summed over how
    // much mass the family carries. A family and a single objective use the
    // same rule, so a one-member family scores exactly the region of its lone
    // objective; that is what `class_separation` did while `neighborhood_hit`
    // was its only member.
    for (family, members) in FAMILIES {
        assert_eq!(
            w.region(family).unwrap().indices.len(),
            region_size(members.len(), w.s),
            "region {family}"
        );
    }

    for objective in OBJECTIVES {
        assert_eq!(
            w.region(objective.name()).unwrap().indices.len(),
            region_size(1, w.s),
            "region {objective}"
        );
    }

    // The rule has to actually be selective: a region that admitted everything,
    // or nothing, would satisfy the equalities above just as well.
    assert!(region_size(1, w.s) > 0);
    assert!(region_size(FAMILIES[0].1.len(), w.s) < w.vectors.len());
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
    assert_eq!(seen, (0..SPACE.len()).collect::<Vec<_>>());
}

#[test]
fn a_family_region_puts_at_least_half_its_mass_on_its_own_objectives() {
    let w = Weights::new(SPACE);
    let half = u8::try_from(w.s.div_ceil(2)).expect("s is at most 255");
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
    let w = Weights::new(SPACE);
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
    let w = Weights::new(SPACE);
    assert_eq!(
        score(&[flat(1.0)], &w, REGION_ALL).partial_cmp(&0.0),
        Some(Ordering::Equal)
    );

    // A front at the origin gives max_j λ_j per weight vector, which is what an
    // empty front is defined to score too.
    let nadir = score(&[flat(0.0)], &w, REGION_ALL);
    let empty = score(&[], &w, REGION_ALL);
    assert_eq!(nadir.partial_cmp(&empty), Some(Ordering::Equal));
    assert!(nadir > 0.0);
}

#[test]
fn the_indicator_is_weakly_pareto_compliant() {
    // The property ΔR2 > 0 rests on: a dominating front can never score worse.
    let w = Weights::new(SPACE);
    let worse = ramp(0.3);
    let better: Row = worse.iter().map(|v| v + 0.1).collect();
    for region in &w.regions {
        let (region, b, a) = (
            region.name.as_str(),
            score(std::slice::from_ref(&better), &w, &region.name),
            score(std::slice::from_ref(&worse), &w, &region.name),
        );
        assert!(b <= a, "region {region}: dominating front scored {b} > {a}");
    }
}

#[test]
fn dominated_points_do_not_change_the_indicator() {
    // Why reducing to the Pareto front first is exact rather than an
    // approximation.
    let w = Weights::new(SPACE);
    let mut front = flat(0.8);
    front[0] = 0.9;
    let inside = flat(0.4);
    assert_eq!(
        score(&[front.clone()], &w, REGION_ALL).partial_cmp(&score(
            &[front, inside],
            &w,
            REGION_ALL
        )),
        Some(Ordering::Equal)
    );
}

#[test]
fn adding_a_non_dominated_point_can_only_help() {
    let w = Weights::new(SPACE);
    let mut a = flat(0.2);
    a[0] = 0.95;
    let mut b = flat(0.2);
    b[1] = 0.95;
    let both = score(&[a.clone(), b.clone()], &w, REGION_ALL);
    assert!(both <= score(&[a], &w, REGION_ALL));
    assert!(both <= score(&[b], &w, REGION_ALL));
}

#[test]
fn the_indicator_is_order_independent() {
    let w = Weights::new(SPACE);
    let mut a = flat(0.5);
    a[3] = 0.9;
    let mut b = flat(0.6);
    b[4] = 0.2;
    let ab = [a.clone(), b.clone()];
    let ba = [b, a];
    assert_eq!(
        score(&ab, &w, REGION_ALL).partial_cmp(&score(&ba, &w, REGION_ALL)),
        Some(Ordering::Equal)
    );
}

// ─── Recommendations ─────────────────────────────────────────────────────────

#[test]
fn a_metric_region_recommends_the_point_that_is_good_at_that_metric() {
    let w = Weights::new(SPACE);
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
    let w = Weights::new(SPACE);
    let front = [flat(0.5), flat(0.5)];
    let u = front_utilities(&front, &w.vectors);
    let rec = recommendation(&u, w.region(REGION_ALL).unwrap()).unwrap();
    assert_eq!(rec.front_index, 0);
    assert_eq!(rec.share.partial_cmp(&1.0), Some(Ordering::Equal));
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
    let w = Weights::new(SPACE);
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
    let w = Weights::new(SPACE);
    // The two ways a reading can carry no number. Both orient to the worst
    // case, which is what keeps a diverged trial from scoring well.
    let mut diverged = record(0.9);
    diverged.metrics.set(TRUSTWORTHINESS, MetricValue::Diverged);
    diverged.metrics.set(CONTINUITY, MetricValue::Absent);

    let good = cell_summary(&[record(0.9)], &w);
    let bad = cell_summary(&[diverged], &w);
    assert!(bad.r2[REGION_ALL] > good.r2[REGION_ALL]);
}
