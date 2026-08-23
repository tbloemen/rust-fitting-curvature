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
//!   its own slice of the result. Eight regions cost barely more than one.
//!
//! Weight vectors are held as integer counts summing to [`Weights::s`], which
//! keeps the region membership tests exact (`l/5` is not representable in
//! binary, so `0.2 + 0.4 > 0.6` and a naive `λ_a + λ_b >= 0.5` would be a coin
//! flip).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::objectives::{is_manifold, oriented_matrix, N_OBJECTIVES, OBJECTIVES};
pub use crate::objectives::{metric_objectives, METRICS};
use crate::pareto::pareto_front_mask;
use crate::records::TrialRecord;

/// Name of the region spanning the whole simplex.
pub const REGION_ALL: &str = "all";
/// Name of the region supported on the manifold objectives only.
pub const REGION_MANIFOLD: &str = "manifold";
/// Name of the region supported on the projected (2D) objectives only.
pub const REGION_PROJECTED: &str = "projected";

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
    /// Granularity of the enumeration: every `λ_j` is a multiple of `1 / s`.
    pub s: usize,
    /// Integer counts summing to `s`, one per weight vector.
    pub counts: Vec<[u8; N_OBJECTIVES]>,
    /// The same vectors as `λ_j = l_j / s`.
    pub vectors: Vec<[f64; N_OBJECTIVES]>,
    /// `all`, one per metric, `manifold`, `projected` — in that order.
    pub regions: Vec<Region>,
}

impl Default for Weights {
    fn default() -> Self {
        Self::new()
    }
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
    /// For ten objectives at `s = 5` this is `C(14, 9) = 2002` vectors.
    pub fn new() -> Self {
        Self::with_resolution(Self::DEFAULT_S)
    }

    /// Enumerate the simplex at an arbitrary granularity.
    ///
    /// Panics unless `1 <= s <= 255`: counts are `u8`, so a larger `s` would
    /// wrap silently in release.
    pub fn with_resolution(s: usize) -> Self {
        assert!(
            (1..=usize::from(u8::MAX)).contains(&s),
            "simplex resolution {s} must be in 1..=255"
        );
        let counts = enumerate_simplex(s as u8);
        let vectors: Vec<[f64; N_OBJECTIVES]> = counts
            .iter()
            .map(|c| c.map(|l| f64::from(l) / s as f64))
            .collect();
        let regions = build_regions(&counts, s);
        Self {
            s,
            counts,
            vectors,
            regions,
        }
    }

    /// The region named *name*, if it exists.
    pub fn region(&self, name: &str) -> Option<&Region> {
        self.regions.iter().find(|r| r.name == name)
    }
}

/// Every vector of `N_OBJECTIVES` non-negative integers summing to `s`.
///
/// Example: for s=2, N_OBJECTIVES = 3, it should return
/// (0, 0, 2)
/// (0, 1, 1)
/// (0, 2, 0)
/// (1, 0, 1)
/// (1, 1, 0
/// (2, 0, 0)
fn enumerate_simplex(s: u8) -> Vec<[u8; N_OBJECTIVES]> {
    let mut out = Vec::new();
    let mut counts = [0u8; N_OBJECTIVES];
    fill(0, s, &mut counts, &mut out);
    out
}

fn fill(
    dim: usize,
    remaining: u8,
    counts: &mut [u8; N_OBJECTIVES],
    out: &mut Vec<[u8; N_OBJECTIVES]>,
) {
    if dim == N_OBJECTIVES - 1 {
        counts[dim] = remaining;
        out.push(*counts);
        return;
    }
    for l in 0..=remaining {
        counts[dim] = l;
        fill(dim + 1, remaining - l, counts, out);
    }
    counts[dim] = 0;
}

/// The preference regions: the whole simplex, one per metric, and one per
/// evaluation surface.
///
/// A metric's region holds the vectors placing at least half their mass on that
/// metric's two objectives, which at `s = 5` means an integer count of 3 or
/// more. The surface regions hold the vectors supported entirely on the five
/// manifold objectives, respectively the five projected ones.
fn build_regions(counts: &[[u8; N_OBJECTIVES]], s: usize) -> Vec<Region> {
    let half = s.div_ceil(2) as u8; // 3 of 5: "at least half the mass"
    let mut regions = vec![Region {
        name: REGION_ALL.to_string(),
        indices: (0..counts.len()).collect(),
    }];

    for (i, metric) in METRICS.iter().enumerate() {
        let (proj, man) = metric_objectives(i);
        regions.push(Region {
            name: (*metric).to_string(),
            indices: select(counts, |c| c[proj] + c[man] >= half),
        });
    }

    regions.push(Region {
        name: REGION_MANIFOLD.to_string(),
        indices: select(counts, |c| {
            (0..N_OBJECTIVES).all(|j| is_manifold(j) || c[j] == 0)
        }),
    });
    regions.push(Region {
        name: REGION_PROJECTED.to_string(),
        indices: select(counts, |c| {
            (0..N_OBJECTIVES).all(|j| !is_manifold(j) || c[j] == 0)
        }),
    });
    regions
}

fn select(counts: &[[u8; N_OBJECTIVES]], pred: impl Fn(&[u8; N_OBJECTIVES]) -> bool) -> Vec<usize> {
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
pub fn front_utilities(
    front: &[[f64; N_OBJECTIVES]],
    weights: &[[f64; N_OBJECTIVES]],
) -> Vec<FrontUtility> {
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
pub fn r2(u: &[FrontUtility], region: &Region) -> f64 {
    if region.indices.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = region.indices.iter().map(|&i| u[i].utility).sum();
    sum / region.indices.len() as f64
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
        share: count as f64 / region.indices.len() as f64,
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
/// configuration attains on all ten objectives alongside its hyperparameters.
pub fn oriented_objectives(record: &TrialRecord) -> BTreeMap<String, f64> {
    let row = crate::objectives::oriented_row(record);
    OBJECTIVES
        .iter()
        .zip(row)
        .map(|(name, v)| ((*name).to_string(), v))
        .collect()
}

/// Reduce a cell's trials to its front, then score it under every region.
pub fn cell_summary(records: &[TrialRecord], weights: &Weights) -> CellSummary {
    let all = oriented_matrix(records);
    let keep = pareto_front_mask(&all);
    let front_idx: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter(|(_, k)| **k)
        .map(|(i, _)| i)
        .collect();
    let front: Vec<[f64; N_OBJECTIVES]> = front_idx.iter().map(|&i| all[i]).collect();

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
