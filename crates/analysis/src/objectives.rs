//! The qParEGO objectives and their orientation into `[0, 1]`-higher-is-better.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::LazyLock;

use fitting_core::metrics::{Direction, Metric, DISTANCE_CONSISTENCY};

use crate::cell::CellFile;
use crate::error::{Error, IoContext, Result};
use crate::records::TrialRecord;

/// The qParEGO objectives, in the order the optimizer writes them.
///
/// This *is* `fitting_core::metrics::OBJECTIVES`, which
/// `optimizer::pareto::default_pareto_metrics` also returns. The two used to be
/// separate lists kept aligned by hand across crates — the alignment this
/// module's doc comment used to ask readers to maintain.
///
/// Every one is measured **after projection to 2D** and is bounded in `[0, 1]`
/// by construction. Both properties are load-bearing:
///
/// - *Projected only.* The manifold (pre-projection, geodesic) variants used to
///   occupy half this list. They are still measured and still present on
///   [`TrialRecord`], and [`METRIC_PAIRS`] still pairs them up — but they no
///   longer steer the search; only Experiment 4's projection-gap figure reads
///   them.
/// - *Bounded only.* [`oriented_value`] clamps to `[0, 1]` and the R2 ideal
///   point is pinned at `(1, …, 1)`, so an unbounded objective would be
///   silently truncated rather than measured. `distance_consistency` qualifies
///   — it is a fraction of points — which is why it joined the set rather than
///   staying a reported-only diagnostic. That is why `dunn_index`,
///   `davies_bouldin_ratio` and `cluster_density_measure` are absent: all three
///   are ratios whose upper tails over `results/` reach 3.0e10, 2.9e11 and 2e24.
///   Admitting one would require estimated ideal/nadir bounds (Karl et al.,
///   *MOHPO — An Overview*, §3.3.3; Grodzevich & Romanko 2006 §4.2), which no
///   longer applies to a set whose limits are all known a priori.
///
/// Both rules are now checked rather than described: `QualityMetric::space` and
/// `is_objective` carry them per metric, and `test_registry.rs` asserts them in
/// both directions.
///
/// Row order is grouped by [`FAMILIES`] and is otherwise a free choice — the
/// indicators are invariant under a relabelling of the axes (the weight simplex
/// is enumerated symmetrically) and nothing on disk is positional, since
/// [`oriented_row`] resolves values by name and every output table is
/// name-keyed. The grouping itself is not free: [`FAMILIES`] indexes into this
/// list by position, and `objectives_are_grouped_by_family` in the core
/// registry tests pins the contiguity.
pub const OBJECTIVES: &[Metric] = fitting_core::metrics::OBJECTIVES;

/// Number of objectives in [`ObjectiveSpace::Current6`]; the arity `FAMILIES`
/// indexes into.
pub const N_OBJECTIVES: usize = OBJECTIVES.len();

// ─── The objective space ─────────────────────────────────────────────────────

/// The objective space results are scored in: the six projected objectives
/// the optimizer searches ([`OBJECTIVES`]).
///
/// Sweeps from before commit c0bad38 (2026-09-08) searched ten objectives —
/// every metric on the manifold and after projection — and none of them
/// measured `distance_consistency`. Scoring such a sweep in this space would
/// add an objective its trials never measured, so the analysis rejects them
/// ([`Self::detect_in_file`]) instead of mixing them in. Every output still
/// carries the space's tag (`obj6`), so outputs stay recognisable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjectiveSpace {
    /// Six projected objectives — what the optimizer searches.
    Current6,
}

impl ObjectiveSpace {
    /// Every space, for exhaustive iteration in tests and CLI parsing.
    pub const ALL: [Self; 1] = [Self::Current6];

    /// The objectives of this space, in scoring order.
    #[must_use]
    pub fn metrics(self) -> &'static [Metric] {
        match self {
            Self::Current6 => OBJECTIVES,
        }
    }

    /// The dimension of this space.
    #[must_use]
    pub fn len(self) -> usize {
        self.metrics().len()
    }

    /// Never true — the space has objectives. Present because clippy asks for
    /// it beside `len`.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.metrics().is_empty()
    }

    /// The filename tag every default output path carries.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Current6 => "obj6",
        }
    }

    /// A caption-ready description.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Current6 => "6 objectives (projected only)",
        }
    }

    /// The space a tag names.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.tag() == tag)
    }

    /// Check that a results file was written in this space, from its first row.
    ///
    /// `distance_consistency` entered the objective set in the same commit that
    /// made the space projected-only, so a row carrying the **column** is a
    /// current sweep and one without it is an older ten-objective sweep. The
    /// test is on the column's *presence*, not its value: a sweep whose first
    /// trials diverged writes the column as `null`.
    ///
    /// # Errors
    ///
    /// Propagates I/O errors, returns [`Error::LegacySweep`] for a file written
    /// in the older ten-objective space, and [`Error::EmptyResults`] for a file
    /// with no parseable row to read the answer off.
    pub fn detect_in_file(path: &Path) -> Result<Self> {
        let file = File::open(path).at(path)?;
        for line in BufReader::new(file).lines() {
            let line = line.at(path)?;
            if line.trim().is_empty() {
                continue;
            }
            let Ok(row) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            if row.get(DISTANCE_CONSISTENCY.name()).is_some() {
                return Ok(Self::Current6);
            }
            return Err(Error::LegacySweep(path.to_path_buf()));
        }
        Err(Error::EmptyResults(path.to_path_buf()))
    }
}

impl std::str::FromStr for ObjectiveSpace {
    type Err = String;

    /// Accepts the tag and the variant name.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "obj6" | "current6" | "current" | "6" => Ok(Self::Current6),
            other => Err(format!(
                "unknown objective space `{other}`; the only space is obj6"
            )),
        }
    }
}

impl std::fmt::Display for ObjectiveSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.tag())
    }
}

/// The three preference families, as `(name, indices into [`OBJECTIVES`])`.
///
/// These replace the `manifold` / `projected` regions, which became vacuous
/// once every objective was projected. The split is the standard DR-quality
/// taxonomy: what the projection preserves of the *neighbourhood* structure, of
/// the *distances*, and of the *labels*.
///
/// `neighborhood_hit` sits in `class_separation`, not `structure`: it is the
/// fraction of a point's k nearest neighbours in the embedding sharing its
/// label, so it reads `labels` and never touches the high-dimensional data. Its
/// resemblance to trustworthiness/continuity is that it is a k-NN statistic at
/// the same `k`, which is a computational similarity, not a semantic one.
///
/// Its partner is `distance_consistency`, which asks whether each point falls
/// nearest to its own class centroid. The pairing is deliberate: neighbourhood
/// hit is purely local and so cannot distinguish classes that are cleanly
/// separated from classes that merely fail to interleave, while distance
/// consistency compares against every class centroid and therefore reads
/// separation across the visualisation as a whole. `class_separation` held only
/// `neighborhood_hit` between the removal of `class_density_measure` and the
/// addition of this one, over which its region was *identical* to the
/// per-objective `neighborhood_hit` region; all three families now hold two
/// objectives each.
///
/// Membership is a **slice, not a fixed-size array**: the `[usize; 2]` this used
/// to be silently outlived the objective set it indexed into, leaving
/// `("class_separation", [4, 5])` reading index 5 of a 5-element row.
/// Variable arity removes the failure mode rather than the one instance.
///
/// Indices rather than names because [`OBJECTIVES`] is ordered by family, so
/// they are contiguous and there is nothing to look up.
/// `families_partition_the_objectives` pins that.
pub const FAMILIES: [(&str, &[usize]); 3] = [
    ("structure", &[0, 1]),
    ("distance", &[2, 3]),
    ("class_separation", &[4, 5]),
];

/// The preference families of a projected-only *space*, as
/// `(name, indices into space.metrics())`, in registry order.
///
/// [`FAMILIES`] restated for any space: the members are grouped by
/// `Metric::family()` rather than listed, so a space with fewer objectives
/// gets the families it can support. A family holding a **single** objective
/// is dropped, because its "at least half the mass" region would be the same
/// set of vectors as that objective's own region. On
/// [`ObjectiveSpace::Current6`] this returns [`FAMILIES`] exactly
/// (`families_agree_with_the_constant` pins it).
#[must_use]
pub fn families(space: ObjectiveSpace) -> Vec<(&'static str, Vec<usize>)> {
    let mut out: Vec<(&'static str, Vec<usize>)> = Vec::new();
    for (j, metric) in space.metrics().iter().enumerate() {
        let name = metric.family().name();
        match out.iter_mut().find(|(n, _)| *n == name) {
            Some((_, members)) => members.push(j),
            None => out.push((name, vec![j])),
        }
    }
    out.retain(|(_, members)| members.len() > 1);
    out
}

/// The metrics that have both a projected and a manifold reading, as
/// `(projected, manifold)`.
///
/// **This is a diagnostic table, not the objective list.** It used to generate
/// [`OBJECTIVES`] by interleaving; it no longer does, and the two are now
/// independent — an objective with no manifold variant would correctly not
/// appear here.
///
/// Its reader is `figures::exp2_proj_gap`, one panel
/// per row comparing the manifold and projected readings of the same metric —
/// the evidence for dropping the manifold objectives. Deleted in `b43c731`
/// once that argument was settled, and since restored.
///
/// Derived by matching `QualityMetric::base` over the registry, so a metric
/// that gains or loses a twin gains or loses a row with no edit here.
pub static METRIC_PAIRS: LazyLock<Vec<(Metric, Metric)>> =
    LazyLock::new(|| Metric::dual_pairs().collect());

/// Length of [`METRIC_PAIRS`], as a `const` because it is used as an array
/// length and `QualityMetric`'s methods are not const-callable.
/// The one number here that is restated rather than derived;
/// `metric_pairs_has_the_declared_length` pins it.
pub const N_METRIC_PAIRS: usize = 5;

/// True when *metric* is minimised, so [`oriented`] flips it.
///
/// Was a `MINIMIZE: [&str; 1]` constant listing `normalized_stress`; the
/// registry states each metric's own orientation, and
/// `only_normalized_stress_is_minimized` pins that this is still the only one.
#[must_use]
pub fn is_minimized_metric(metric: Metric) -> bool {
    metric.direction() == Direction::Minimize
}

/// True when *name* is an objective that is minimised.
///
/// The string form, for callers that genuinely start from one. Everything that
/// already holds a [`Metric`] should use [`is_minimized_metric`] — resolving a
/// name is a linear scan of the registry, and `oriented_row` used to do three
/// of them per objective per record.
pub fn is_minimized(name: &str) -> bool {
    Metric::by_name(name).is_some_and(is_minimized_metric)
}

/// Map one metric's reading into `[0, 1]` with higher = better.
///
/// A reading with no number — a diverged trial, an unwritten column — is the
/// worst case (0.0), matching the optimizer's `metrics_to_vec` substitution so
/// such a trial scores badly rather than being dropped silently.
#[must_use]
pub fn oriented(metric: Metric, v: Option<f64>) -> f64 {
    let Some(x) = v else { return 0.0 };
    if !x.is_finite() {
        return 0.0;
    }
    let x = if is_minimized_metric(metric) {
        1.0 - x
    } else {
        x
    };
    // Every objective is bounded in [0, 1] by construction; clamp defensively.
    x.clamp(0.0, 1.0)
}

/// [`oriented`] by name, for callers that start from a string.
#[must_use]
pub fn oriented_value(name: &str, v: Option<f64>) -> f64 {
    match Metric::by_name(name) {
        Some(metric) => oriented(metric, v),
        // Not a metric at all: nothing to orient against, so pass the value
        // through with the same absent-is-worst rule.
        None => v.filter(|x| x.is_finite()).unwrap_or(0.0).clamp(0.0, 1.0),
    }
}

/// One point of an oriented objective space: every objective in `[0, 1]` with
/// higher = better, as long as its [`ObjectiveSpace`] is wide.
///
/// A `Vec` rather than the `[f64; N_OBJECTIVES]` it used to be, because the
/// width is now a property of the run. Every function taking one takes a
/// `&[f64]`, so a row and a weight vector of the same space line up under
/// `zip` and a mismatched pair is short-circuited by it rather than being
/// silently padded.
pub type Row = Vec<f64>;

/// One record's oriented objective vector, in *space*.
#[must_use]
pub fn oriented_row(r: &TrialRecord, space: ObjectiveSpace) -> Row {
    // Straight off the handle: no name is resolved here. This used to be
    // `oriented_value(metric.name(), r.objective(metric.name()))`, which cost
    // three linear registry scans per objective per record — `by_name` inside
    // `objective`, `index` inside `get`, and `by_name` again inside
    // `is_minimized` — for ~4.3M scans over the sweep set.
    space
        .metrics()
        .iter()
        .map(|metric| oriented(*metric, r.metrics.get(*metric)))
        .collect()
}

/// The `(n, space.len())` oriented-objective matrix for *records*
/// (higher = better).
#[must_use]
pub fn oriented_matrix(records: &[TrialRecord], space: ObjectiveSpace) -> Vec<Row> {
    records.iter().map(|r| oriented_row(r, space)).collect()
}

/// The objective space every cell of a run is scored in.
///
/// *forced* short-circuits the check (the `--objectives` flag). Otherwise
/// every cell is checked with [`ObjectiveSpace::detect_in_file`], so a
/// directory holding an older ten-objective sweep fails rather than producing
/// a table whose rows are in two units.
///
/// # Errors
///
/// Returns [`Error::LegacySweep`] for an older sweep, and propagates I/O
/// errors from checking the cells.
pub fn resolve_space(cells: &[CellFile], forced: Option<ObjectiveSpace>) -> Result<ObjectiveSpace> {
    if let Some(space) = forced {
        return Ok(space);
    }
    for cf in cells {
        ObjectiveSpace::detect_in_file(&cf.path)?;
    }
    Ok(ObjectiveSpace::Current6)
}
