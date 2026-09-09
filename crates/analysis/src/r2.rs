//! The R2 indicator over the discrete weight simplex, with preference regions.
//!
//! `R2(A; W) = mean over λ ∈ W of  min over a ∈ A of  max_j λ_j (1 − a_j)`
//!
//! in the oriented objective space where every objective lies in `[0, 1]` with
//! higher = better, so the ideal point is `(1, …, 1)`. Smaller is better.
//!
//! Two things make this cheap enough to compute for every cell and every
//! preference region:
//!
//! * The inner minimum is always attained on the Pareto front. If `a` dominates
//!   `b` then `a_j ≥ b_j` for every `j`, hence `λ_j (1 − a_j) ≤ λ_j (1 − b_j)`
//!   and the max over `j` can only shrink. Reducing to the front first is exact.
//! * Every preference region is a *subset* of one enumeration of the simplex, so
//!   the per-weight-vector minimisation runs once and each region is a mean over
//!   its own slice of the result. Ten regions cost barely more than one.
//!
//! Weight vectors are held as integer counts summing to [`Weights::s`], which
//! keeps the region membership tests exact (`l/5` is not representable in
//! binary, so `0.2 + 0.4 > 0.6` and a naive `λ_a + λ_b >= 0.5` would be a coin
//! flip).

use fitting_core::cast::count_to_f64;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::objectives::{oriented_matrix, ObjectiveSpace, Row, FAMILIES, METRIC_PAIRS};
use crate::pareto::pareto_front_mask;
use crate::records::TrialRecord;

/// Name of the region spanning the whole simplex.
pub const REGION_ALL: &str = "all";

// ─── The weight simplex and its preference regions ───────────────────────────

/// A named subset of the enumerated weight vectors.
///
/// `indices` point into [`Weights::vectors`]; a region never owns its vectors,
/// which is what lets one minimisation pass serve every region.
#[derive(Debug, Clone)]
pub struct Region {
    pub name: String,
    pub indices: Vec<usize>,
}

/// The enumerated weight simplex together with the preference regions of the
/// methods chapter.
#[derive(Debug, Clone)]
pub struct Weights {
    /// The space these weights are enumerated over. Fronts scored against them
    /// must be oriented in the same one, which is what makes the width of a
    /// vector and the width of a row agree.
    pub space: ObjectiveSpace,
    /// Granularity of the enumeration: every `λ_j` is a multiple of `1 / s`.
    pub s: usize,
    /// Integer counts summing to `s`, one per weight vector.
    pub counts: Vec<Vec<u8>>,
    /// The same vectors as `λ_j = l_j / s`.
    pub vectors: Vec<Row>,
    /// The preference regions, in report order — see [`build_regions`].
    pub regions: Vec<Region>,
}

impl Weights {
    /// Granularity of the weight simplex: `λ_j = l / s` for integer `l`
    /// (Knowles 2006, eq. 1). The same value the optimizer's `ParEgoOptimizer`
    /// samples with, so the indicator scores fronts under the preferences that
    /// generated them — which is why the analysis never varies it, and why
    /// [`Weights::with_resolution`] exists only for tests and sensitivity
    /// checks.
    pub const DEFAULT_S: usize = 5;

    /// Enumerate the simplex at [`Self::DEFAULT_S`] and build every preference
    /// region.
    ///
    /// For six objectives at `s = 5` this is `C(10, 5) = 252` vectors.
    #[must_use]
    pub fn new(space: ObjectiveSpace) -> Self {
        Self::with_resolution(space, Self::DEFAULT_S)
    }

    /// Enumerate the simplex at an arbitrary granularity.
    ///
    /// Panics unless `1 <= s <= 255`: counts are `u8`, so a larger `s` would
    /// wrap silently in release.
    ///
    /// # Panics
    ///
    /// Panics if `s` is outside `1..=255`.
    #[must_use]
    pub fn with_resolution(space: ObjectiveSpace, s: usize) -> Self {
        assert!(
            (1..=usize::from(u8::MAX)).contains(&s),
            "simplex resolution {s} must be in 1..=255"
        );
        let counts = enumerate_simplex(
            space.len(),
            u8::try_from(s).expect("resolution is asserted 1..=255 above"),
        );
        let vectors: Vec<Row> = counts
            .iter()
            .map(|c| c.iter().map(|&l| f64::from(l) / count_to_f64(s)).collect())
            .collect();
        let regions = build_regions(space, &counts, s);
        Self {
            space,
            s,
            counts,
            vectors,
            regions,
        }
    }

    /// The region named *name*, if it exists.
    #[must_use]
    pub fn region(&self, name: &str) -> Option<&Region> {
        self.regions.iter().find(|r| r.name == name)
    }
}

/// Every vector of `N_OBJECTIVES` non-negative integers summing to `s`.
///
/// Example: for s=2, `N_OBJECTIVES` = 3, it should return
/// (0, 0, 2)
/// (0, 1, 1)
/// (0, 2, 0)
/// (1, 0, 1)
/// (1, 1, 0
/// (2, 0, 0)
fn enumerate_simplex(n: usize, s: u8) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut counts = vec![0u8; n];
    fill(0, s, &mut counts, &mut out);
    out
}

fn fill(dim: usize, remaining: u8, counts: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
    if dim == counts.len() - 1 {
        counts[dim] = remaining;
        out.push(counts.clone());
        return;
    }
    for l in 0..=remaining {
        counts[dim] = l;
        fill(dim + 1, remaining - l, counts, out);
    }
    counts[dim] = 0;
}

/// Name of the region supported entirely on the manifold objectives, and its
/// projected twin. [`ObjectiveSpace::Legacy10`] only — the current space has no
/// manifold axes to separate.
pub const REGION_MANIFOLD: &str = "manifold";
pub const REGION_PROJECTED: &str = "projected";

/// The preference regions of *space*, in report order.
///
/// The two spaces do not share a region set, and cannot: a region is a subset
/// of a simplex whose dimension is the space's.
///
/// * [`ObjectiveSpace::Current6`] — `all`, one per [`FAMILIES`] family, then one
///   per objective. Both kinds use the same "at least half the mass" rule, which
///   at `s = 5` means an integer count of 3 or more: over the 252 vectors of the
///   6-objective simplex that admits 66 for a two-objective family and 21 for a
///   single objective.
/// * [`ObjectiveSpace::Legacy10`] — `all`, one per *metric pair* (the vectors
///   placing at least half their mass on that metric's two objectives, its
///   projected and manifold readings together), then the two **surface**
///   regions: the vectors supported entirely on the manifold objectives, and
///   entirely on the projected ones. This is the set the sweeps under
///   `results/` were reported under, reconstructed exactly — eight regions, not
///   thirteen, because a legacy region is per *metric*, never per objective.
///
/// The surface regions use "supported entirely on" rather than "at least half":
/// each holds five objectives, which is wide enough for that rule to admit a
/// meaningful set. The current space's families hold two, where it would admit
/// six vectors — too thin for a mean to say anything, which is why they use the
/// half-mass rule instead.
fn build_regions(space: ObjectiveSpace, counts: &[Vec<u8>], s: usize) -> Vec<Region> {
    let half = u8::try_from(s.div_ceil(2)).expect("s is at most 255, so half is at most 128"); // 3 of 5: "at least half the mass"
    let mut regions = vec![Region {
        name: REGION_ALL.to_string(),
        indices: (0..counts.len()).collect(),
    }];

    match space {
        ObjectiveSpace::Current6 => {
            for (name, members) in FAMILIES {
                regions.push(Region {
                    name: name.to_string(),
                    // `u16` because a family may hold more than two objectives
                    // and `s` can be up to 255; summing `u8` counts in place
                    // would wrap.
                    indices: select(counts, |c| {
                        members.iter().map(|&j| u16::from(c[j])).sum::<u16>() >= u16::from(half)
                    }),
                });
            }
            for (j, objective) in space.metrics().iter().enumerate() {
                regions.push(Region {
                    name: objective.name().to_string(),
                    indices: select(counts, |c| c[j] >= half),
                });
            }
        }
        ObjectiveSpace::Legacy10 => {
            // The interleaving `LEGACY_OBJECTIVES` is built with: metric `i`
            // owns objectives `2i` (projected) and `2i + 1` (manifold). A
            // region is named for the *projected* member, which is the metric's
            // own name.
            for (i, (projected, _)) in METRIC_PAIRS.iter().enumerate() {
                let (p, m) = (2 * i, 2 * i + 1);
                regions.push(Region {
                    name: projected.name().to_string(),
                    indices: select(counts, |c| {
                        u16::from(c[p]) + u16::from(c[m]) >= u16::from(half)
                    }),
                });
            }
            regions.push(Region {
                name: REGION_MANIFOLD.to_string(),
                indices: select(counts, |c| {
                    c.iter().enumerate().all(|(j, &l)| j % 2 == 1 || l == 0)
                }),
            });
            regions.push(Region {
                name: REGION_PROJECTED.to_string(),
                indices: select(counts, |c| {
                    c.iter().enumerate().all(|(j, &l)| j % 2 == 0 || l == 0)
                }),
            });
        }
    }

    regions
}

/// Every region of *space*, as `(name, axis label)`, in the order
/// [`build_regions`] emits them.
///
/// Lives here rather than in the figure that draws them: `r2_bars` used to keep
/// its own list, and a region added to one and not the other silently mislabels
/// every bar after it. Now the labels are derived from the same match the
/// regions are.
#[must_use]
pub fn region_labels(space: ObjectiveSpace) -> Vec<(String, String)> {
    let mut out = vec![(REGION_ALL.to_string(), "W_all".to_string())];
    match space {
        ObjectiveSpace::Current6 => {
            out.extend(
                FAMILIES
                    .iter()
                    .map(|(family, _)| ((*family).to_string(), format!("W_{}", short(family)))),
            );
            out.extend(
                space
                    .metrics()
                    .iter()
                    .map(|m| (m.name().to_string(), format!("W_{}", m.short()))),
            );
        }
        ObjectiveSpace::Legacy10 => {
            // One region per *metric pair*, named for its projected member,
            // then the two surface regions.
            out.extend(
                METRIC_PAIRS
                    .iter()
                    .map(|(p, _)| (p.name().to_string(), format!("W_{}", p.short()))),
            );
            out.push((REGION_MANIFOLD.to_string(), "W_man".to_string()));
            out.push((REGION_PROJECTED.to_string(), "W_proj".to_string()));
        }
    }
    out
}

/// Abbreviations for the family labels; the full names do not fit an axis.
///
/// Only the families need one — a metric carries its own on
/// `QualityMetric::short`. The fallthrough is a hazard rather than a
/// convenience: an unabbreviated name renders at full width and overlaps its
/// neighbours, so every family needs an arm here.
fn short(family: &str) -> &str {
    match family {
        "structure" => "struct",
        "distance" => "dist",
        "class_separation" => "class",
        other => other,
    }
}

fn select(counts: &[Vec<u8>], pred: impl Fn(&[u8]) -> bool) -> Vec<usize> {
    counts
        .iter()
        .enumerate()
        .filter(|(_, c)| pred(c))
        .map(|(i, _)| i)
        .collect()
}

// ─── The indicator ───────────────────────────────────────────────────────────

/// Per-weight-vector result of the inner minimisation of the R2 indicator.
#[derive(Debug, Clone, Copy)]
pub struct FrontUtility {
    /// `min_a max_j λ_j (1 − a_j)` for this weight vector.
    pub utility: f64,
    /// The front point attaining it, as an index into the front. Ties resolve to
    /// the lowest index, so the result depends only on the front's own order.
    pub best: usize,
}

/// Run the inner minimisation once for every weight vector.
///
/// An empty front is scored as if it held the single worst point `(0, …, 0)`,
/// giving `max_j λ_j`. That keeps the indicator total, and a cell whose front is
/// empty is degenerate anyway.
pub fn front_utilities(front: &[Row], weights: &[Row]) -> Vec<FrontUtility> {
    let mut utilities = Vec::with_capacity(weights.len());

    for lambda in weights {
        if front.is_empty() {
            utilities.push(FrontUtility {
                utility: lambda.iter().copied().fold(0.0, f64::max),
                best: 0,
            });
            continue;
        }
        let mut utility_min = f64::INFINITY;
        let mut arg = 0usize;
        for (i, point) in front.iter().enumerate() {
            let mut utility = 0.0f64;
            for (lambda_j, point_j) in lambda.iter().zip(point.iter()) {
                let new_utility = lambda_j * (1.0 - point_j);
                if new_utility > utility {
                    utility = new_utility;
                }
            }
            if utility < utility_min {
                utility_min = utility;
                arg = i;
            }
        }
        utilities.push(FrontUtility {
            utility: utility_min,
            best: arg,
        });
    }

    utilities
}

/// The R2 indicator of a front under one preference region. Smaller is better.
///
/// `NaN` for an empty region, which [`Weights::new`] never produces.
#[must_use]
pub fn r2(u: &[FrontUtility], region: &Region) -> f64 {
    if region.indices.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = region.indices.iter().map(|&i| u[i].utility).sum();
    sum / count_to_f64(region.indices.len())
}

/// The configuration a preference region recommends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Index into the cell's Pareto front.
    pub front_index: usize,
    /// Fraction of the region's weight vectors that choose it.
    pub share: f64,
}

/// The front point most often chosen across a region's weight vectors.
///
/// Ties resolve to the lowest front index, so the recommendation is a function
/// of the front alone.
#[must_use]
pub fn recommendation(u: &[FrontUtility], region: &Region) -> Option<Recommendation> {
    if region.indices.is_empty() {
        return None;
    }
    let mut votes: BTreeMap<usize, usize> = BTreeMap::new();
    for &i in &region.indices {
        *votes.entry(u[i].best).or_default() += 1;
    }
    // BTreeMap iterates by ascending key and the test is strict, so a tie goes
    // to the lowest front index.
    let mut best = 0usize;
    let mut count = 0usize;
    for (&idx, &c) in &votes {
        if c > count {
            best = idx;
            count = c;
        }
    }
    Some(Recommendation {
        front_index: best,
        share: count_to_f64(count) / count_to_f64(region.indices.len()),
    })
}

// ─── Per-cell summary ────────────────────────────────────────────────────────

/// Everything one experiment cell contributes to the analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellSummary {
    pub n_trials: usize,
    pub n_front: usize,
    /// Indices into the cell's trial records of the Pareto-front members, in
    /// record order. `Recommendation::front_index` indexes into this.
    pub front: Vec<usize>,
    /// Region name → R2 indicator.
    pub r2: BTreeMap<String, f64>,
    /// Region name → recommended configuration.
    pub recommended: BTreeMap<String, Recommendation>,
}

/// The oriented objective values of one record, by objective name.
///
/// Used by the recommendation table, which reports what a recommended
/// configuration attains on all six objectives alongside its hyperparameters.
#[must_use]
pub fn oriented_objectives(record: &TrialRecord, space: ObjectiveSpace) -> BTreeMap<String, f64> {
    let row = crate::objectives::oriented_row(record, space);
    space
        .metrics()
        .iter()
        .zip(row)
        .map(|(metric, v)| (metric.name().to_string(), v))
        .collect()
}

/// Reduce a cell's trials to its front, then score it under every region.
#[must_use]
pub fn cell_summary(records: &[TrialRecord], weights: &Weights) -> CellSummary {
    // The space comes off the weights, so a front can only ever be scored
    // against a simplex of its own dimension.
    let all = oriented_matrix(records, weights.space);
    let keep = pareto_front_mask(&all);
    let front_idx: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter(|(_, k)| **k)
        .map(|(i, _)| i)
        .collect();
    let front: Vec<Row> = front_idx.iter().map(|&i| all[i].clone()).collect();

    let utilities = front_utilities(&front, &weights.vectors);
    let mut r2_by_region = BTreeMap::new();
    let mut rec_by_region = BTreeMap::new();
    for region in &weights.regions {
        r2_by_region.insert(region.name.clone(), r2(&utilities, region));
        if let Some(rec) = recommendation(&utilities, region) {
            rec_by_region.insert(region.name.clone(), rec);
        }
    }

    CellSummary {
        n_trials: all.len(),
        n_front: front.len(),
        front: front_idx,
        r2: r2_by_region,
        recommended: rec_by_region,
    }
}
