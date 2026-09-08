//! The metric registry: the trait every metric implements, the handle they are
//! passed around as, and the one hand-maintained list of them.
//!
//! Everything the rest of the workspace needs to know about a metric — its
//! wire name, which distance matrix it reads, which way is better, whether it
//! may serve as a qParEGO objective, how to abbreviate it on an axis, and how
//! to compute it — lives in that metric's own module, in its `impl
//! QualityMetric` block, next to the function that does the arithmetic. Before
//! this, those facts were spread over roughly fourteen hand-maintained lists
//! across four crates, and deleting a single metric left the repository with a
//! runtime panic and three permanently-null JSONL columns.
//!
//! The one list that remains by hand is [`ALL`]. A metric absent from it is
//! simply not measured, which the compiler cannot catch — so `tests/test_metrics.rs`
//! pins the expected wire names, the `ALL`-position round-trip, and the
//! objective eligibility rules.

use super::cluster_density_measure::ClusterDensityMeasure;
use super::continuity::{Continuity, ContinuityManifold};
use super::davies_bouldin_ratio::DaviesBouldinRatio;
use super::dunn_index::DunnIndex;
use super::neighborhood_hit::{NeighborhoodHit, NeighborhoodHitManifold};
use super::normalized_stress::{NormalizedStress, NormalizedStressManifold};
use super::shepard_goodness::{ShepardGoodness, ShepardGoodnessManifold};
use super::trustworthiness::{Trustworthiness, TrustworthinessManifold};
use super::values::MetricValue;
use crate::context::EmbeddingContext;

/// Which distance matrix a metric reads — the before/after-projection
/// distinction, which is the only thing separating a metric from its twin.
///
/// A genuine binary. It briefly had an `Ambient` variant meaning "neither",
/// which existed solely for the spread diagnostics; those are
/// [`crate::spread::SpreadDiagnostics`] now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Space {
    /// Euclidean distances in the projected plane: what the reader sees.
    Projected,
    /// Geodesic distances on the manifold: what the optimiser fits.
    Manifold,
}

/// The preference families the analysis groups objectives into.
///
/// Every variant is a real region `r2::build_regions` emits. It briefly had a
/// `Spread` variant that was not, added to give the spread diagnostics an
/// answer; see [`crate::spread`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    Structure,
    Distance,
    ClassSeparation,
}

impl Family {
    /// The analysis region name, and the grouping key for the web UI.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Family::Structure => "structure",
            Family::Distance => "distance",
            Family::ClassSeparation => "class_separation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Maximize,
    Minimize,
}

/// One embedding-quality metric.
///
/// Implementors are zero-sized and live one per module beside the function
/// they call; the instances live in [`ALL`] and are handed around as
/// [`Metric`].
pub trait QualityMetric: Sync {
    /// Wire name: the JSONL column, the `--metric` value, the pareto-front key.
    /// Changing one invalidates every existing results file.
    fn name(&self) -> &'static str;

    /// The name without the variant suffix. Twins share it, which is how the
    /// analysis pairs manifold against projected readings and how the web UI
    /// puts them in one row.
    fn base(&self) -> &'static str;

    fn space(&self) -> Space;
    fn family(&self) -> Family;
    fn direction(&self) -> Direction;

    /// Whether this may serve as a qParEGO objective.
    ///
    /// The rule is **bounded in `[0, 1]` by construction**, and it is not
    /// cosmetic: `pareto::scalarize_subset` min-max normalises per batch, so one
    /// outlier flattens an axis to ~0 for every real trial, and
    /// `fitting_analysis::oriented_value` clamps to `[0, 1]`, which would peg
    /// most trials at 1.0. That is why the unbounded ratios — `dunn_index`,
    /// `davies_bouldin_ratio`, `cluster_density_measure`, whose upper tails over
    /// `results/` reach 3.0e10, 2.9e11 and 2e24 — answer `false` despite being
    /// perfectly good diagnostics.
    ///
    /// No default: a new metric has to state it.
    fn is_objective(&self) -> bool;

    /// Axis abbreviation. The full names overlap their neighbours on a chart.
    fn short(&self) -> &'static str;

    /// Human-readable label for the web UI.
    fn label(&self) -> &'static str;

    /// This metric's reading.
    ///
    /// Return [`MetricValue::NotApplicable`] where the metric is undefined for
    /// the input — in practice, a label-aware metric on unlabelled data — and
    /// [`MetricValue::measured`] otherwise, which downgrades a non-finite
    /// result to [`MetricValue::Diverged`].
    ///
    /// A metric is not called at all when the distances it reads are not
    /// finite; `MetricValues::compute` records `Diverged` for it instead. So an
    /// implementation never has to defend against a broken distance matrix.
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue;
}

// ─── The handle, and the one list ────────────────────────────────────────────

/// A metric, by reference to its singleton implementation.
///
/// `Copy + Eq + Hash` so it can be passed and keyed exactly as the old
/// fieldless enum was. Equality is by wire name, which [`ALL`] keeps unique.
#[derive(Clone, Copy)]
pub struct Metric(pub(crate) &'static dyn QualityMetric);

impl std::fmt::Debug for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
impl PartialEq for Metric {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}
impl Eq for Metric {}
impl std::hash::Hash for Metric {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

/// Every metric that is measured, **in JSONL column order**.
///
/// This order is the serialisation order of [`super::MetricValues`], so it is
/// pinned to the field order the pre-registry `TrialResult` had: changing it
/// reorders the keys of every newly written results line. `test_metrics.rs`
/// holds the expected list.
///
/// A metric absent from here is not measured at all — the one thing in this
/// design the compiler cannot check, and what the golden-name test is for.
pub const ALL: &[Metric] = &[
    Metric(&Trustworthiness),
    Metric(&TrustworthinessManifold),
    Metric(&Continuity),
    Metric(&ContinuityManifold),
    Metric(&NeighborhoodHit),
    Metric(&NeighborhoodHitManifold),
    Metric(&NormalizedStress),
    Metric(&NormalizedStressManifold),
    Metric(&ShepardGoodness),
    Metric(&ShepardGoodnessManifold),
    Metric(&DaviesBouldinRatio),
    Metric(&DunnIndex),
    Metric(&ClusterDensityMeasure),
];

/// Named handles, for the call sites that name a metric symbolically rather
/// than parsing one: [`OBJECTIVES`], the `ParEGO` tests, the figure axes.
///
/// These are values, not indices, so there is no position to fall out of sync.
pub const TRUSTWORTHINESS: Metric = Metric(&Trustworthiness);
pub const TRUSTWORTHINESS_MANIFOLD: Metric = Metric(&TrustworthinessManifold);
pub const CONTINUITY: Metric = Metric(&Continuity);
pub const CONTINUITY_MANIFOLD: Metric = Metric(&ContinuityManifold);
pub const NEIGHBORHOOD_HIT: Metric = Metric(&NeighborhoodHit);
pub const NEIGHBORHOOD_HIT_MANIFOLD: Metric = Metric(&NeighborhoodHitManifold);
pub const NORMALIZED_STRESS: Metric = Metric(&NormalizedStress);
pub const NORMALIZED_STRESS_MANIFOLD: Metric = Metric(&NormalizedStressManifold);
pub const SHEPARD_GOODNESS: Metric = Metric(&ShepardGoodness);
pub const SHEPARD_GOODNESS_MANIFOLD: Metric = Metric(&ShepardGoodnessManifold);
pub const DAVIES_BOULDIN_RATIO: Metric = Metric(&DaviesBouldinRatio);
pub const DUNN_INDEX: Metric = Metric(&DunnIndex);
pub const CLUSTER_DENSITY_MEASURE: Metric = Metric(&ClusterDensityMeasure);

/// The qParEGO objective set, **grouped by family**.
///
/// This is the single definition behind the optimizer's `default_pareto_metrics`
/// and the analysis's `OBJECTIVES`, an alignment that used to be by hand across
/// crates and is now by construction.
///
/// Two rules fix the membership, both checked by tests rather than asserted in
/// prose: every member is [`Space::Projected`] — what the thesis judges is the
/// 2-D visualisation — and every member is [`QualityMetric::is_objective`],
/// i.e. bounded in `[0, 1]`.
///
/// The *order* is a free choice for the indicators (the weight simplex is
/// enumerated symmetrically and every output table is name-keyed), but it must
/// stay grouped by family: `fitting_analysis::objectives::FAMILIES` indexes into
/// this list by position and relies on each family being contiguous.
pub const OBJECTIVES: &[Metric] = &[
    // structure
    TRUSTWORTHINESS,
    CONTINUITY,
    // distance preservation
    NORMALIZED_STRESS,
    SHEPARD_GOODNESS,
    // class separation
    NEIGHBORHOOD_HIT,
];

impl Metric {
    /// The number of measured metrics — usable as an array length, which is
    /// what [`super::MetricValues`] needs.
    pub const COUNT: usize = ALL.len();

    #[must_use]
    pub fn name(self) -> &'static str {
        self.0.name()
    }
    #[must_use]
    pub fn base(self) -> &'static str {
        self.0.base()
    }
    #[must_use]
    pub fn space(self) -> Space {
        self.0.space()
    }
    #[must_use]
    pub fn family(self) -> Family {
        self.0.family()
    }
    #[must_use]
    pub fn direction(self) -> Direction {
        self.0.direction()
    }
    #[must_use]
    pub fn is_objective(self) -> bool {
        self.0.is_objective()
    }
    #[must_use]
    pub fn short(self) -> &'static str {
        self.0.short()
    }
    #[must_use]
    pub fn label(self) -> &'static str {
        self.0.label()
    }
    pub fn compute(self, c: &EmbeddingContext<'_>) -> MetricValue {
        self.0.compute(c)
    }

    /// Position in [`ALL`] — the slot [`super::MetricValues`] stores at.
    ///
    /// A linear scan of sixteen `&str` pointers, against a metric evaluation
    /// that is `O(n²)` at minimum. `ALL_INDEXED` in the tests pins the
    /// round-trip.
    #[must_use]
    pub fn index(self) -> usize {
        ALL.iter()
            .position(|m| *m == self)
            .expect("every Metric is in ALL")
    }

    /// Look a metric up by wire name. `None` for anything not in [`ALL`] —
    /// which is routine rather than exceptional, since it is also how
    /// deserialisation skips the columns of retired metrics. That is why this
    /// is not `FromStr`: there is no error to report.
    #[must_use]
    pub fn by_name(s: &str) -> Option<Metric> {
        ALL.iter().copied().find(|m| m.name() == s)
    }

    /// The `--metric` help text, so it cannot drift from what parses.
    #[must_use]
    pub fn valid_names() -> String {
        ALL.iter().map(|m| m.name()).collect::<Vec<_>>().join(", ")
    }

    /// Whether this metric's base has a second reading on the other side of the
    /// projection.
    ///
    /// Only a metric with a twin needs its readings told apart — which is why
    /// the browser suffixes `_2d` for those and uses the bare name for
    /// `dunn_index`, `davies_bouldin_ratio` and `cluster_density_measure`,
    /// where there is nothing to disambiguate against.
    #[must_use]
    pub fn has_twin(self) -> bool {
        ALL.iter()
            .any(|o| o.base() == self.base() && o.space() != self.space())
    }

    /// The `(projected, manifold)` pairs, for the figures that compare a
    /// metric's two readings. Derived by matching [`QualityMetric::base`].
    pub fn dual_pairs() -> impl Iterator<Item = (Metric, Metric)> {
        ALL.iter()
            .copied()
            .filter(|m| m.space() == Space::Projected)
            .filter_map(|p| {
                ALL.iter()
                    .copied()
                    .find(|m| m.space() == Space::Manifold && m.base() == p.base())
                    .map(|m| (p, m))
            })
    }
}
