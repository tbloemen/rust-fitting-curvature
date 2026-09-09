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
///   [`TrialRecord`] — [`METRIC_PAIRS`] and `figures/exp4.rs` read them — but
///   they no longer steer the search.
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

/// Number of objectives in the **current** space; the dimension
/// [`ObjectiveSpace::Current6`] scores in.
///
/// No longer an array length — a row is a `Vec<f64>` sized by its
/// [`ObjectiveSpace`], because two spaces now have to coexist in one process.
/// It survives as the arity `FAMILIES` indexes into and as the figure of record
/// for the current search.
pub const N_OBJECTIVES: usize = OBJECTIVES.len();

// ─── The two objective spaces ────────────────────────────────────────────────

/// The ten objectives the sweeps under `results/` were searched and scored in:
/// each of the five paired quality metrics, projected *and* manifold, in
/// [`METRIC_PAIRS`] order with the manifold reading of a pair at the odd index.
///
/// Derived by interleaving [`METRIC_PAIRS`], which is exactly what the deleted
/// `flatten_pairs()` did before commit c0bad38 — so this is the historical list
/// reconstructed from the registry rather than transcribed from it.
static LEGACY_OBJECTIVES: LazyLock<Vec<Metric>> = LazyLock::new(|| {
    METRIC_PAIRS
        .iter()
        .flat_map(|&(projected, manifold)| [projected, manifold])
        .collect()
});

/// Which objective space a set of results is scored in.
///
/// Two spaces have to coexist because the sweeps on disk were not all searched
/// in the same one, and an indicator is only meaningful in the space its front
/// was found in:
///
/// * Until commit c0bad38 (2026-09-08) the optimizer searched **ten**
///   objectives — every quality metric twice, once on the embedding manifold
///   and once after projection to 2D. Half the space therefore rewarded a
///   curved embedding for fitting well *before* projection.
/// * Since then the objectives are **projected only**, and 3597f7b
///   (2026-09-09) added `distance_consistency` as the sixth.
///
/// Scoring a legacy sweep in the current space is not a rescaling: it drops the
/// five axes the search actually optimised, shrinks the Pareto fronts (a curved
/// trial could be non-dominated on a manifold axis alone), and adds a sixth
/// objective that legacy trials never measured, so it orients to the worst case
/// for every one of them and contributes a front-independent constant. That is
/// why this is a choice the analysis makes per run rather than a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjectiveSpace {
    /// Ten objectives: the five paired metrics, projected and manifold.
    Legacy10,
    /// Six projected objectives — what the optimizer searches today.
    Current6,
}

impl ObjectiveSpace {
    /// Both spaces, for exhaustive iteration in tests and CLI parsing.
    pub const ALL: [Self; 2] = [Self::Legacy10, Self::Current6];

    /// The objectives of this space, in scoring order.
    #[must_use]
    pub fn metrics(self) -> &'static [Metric] {
        match self {
            Self::Legacy10 => &LEGACY_OBJECTIVES,
            Self::Current6 => OBJECTIVES,
        }
    }

    /// The dimension of this space.
    #[must_use]
    pub fn len(self) -> usize {
        self.metrics().len()
    }

    /// Never true — both spaces have objectives. Present because clippy asks
    /// for it beside `len`.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.metrics().is_empty()
    }

    /// The filename tag that keeps the two spaces' outputs apart.
    ///
    /// Every default output path carries it, because a table scored in one
    /// space and a table scored in the other are not comparable and must never
    /// overwrite each other.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Legacy10 => "obj10",
            Self::Current6 => "obj6",
        }
    }

    /// A caption-ready description.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Legacy10 => "10 objectives (projected + manifold)",
            Self::Current6 => "6 objectives (projected only)",
        }
    }

    /// The space a tag names.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.tag() == tag)
    }

    /// Which space a results *line* was written in.
    ///
    /// `distance_consistency` entered the objective set in the same commit that
    /// made the space projected-only, and nothing before it measured the metric
    /// at all — so a row carrying the **column** is a post-change sweep and one
    /// without it is a legacy sweep. That is the whole test: it needs no
    /// filename convention and no flag, and it cannot drift out of step with
    /// the data the way either would.
    ///
    /// The test is on the column's *presence*, not its value, which is why it
    /// works on the raw JSON rather than on a [`TrialRecord`]: absent and
    /// `null` both deserialise to absent, and a new sweep whose first trials
    /// diverged writes the column as `null`. One line settles it either way.
    #[must_use]
    pub fn detect_in_line(line: &str) -> Option<Self> {
        let row: serde_json::Value = serde_json::from_str(line).ok()?;
        Some(if row.get(DISTANCE_CONSISTENCY.name()).is_some() {
            Self::Current6
        } else {
            Self::Legacy10
        })
    }

    /// Which space a results file was written in, from its first row.
    ///
    /// # Errors
    ///
    /// Propagates I/O errors, and returns [`Error::EmptyResults`] for a file
    /// with no parseable row to read the answer off.
    pub fn detect_in_file(path: &Path) -> Result<Self> {
        let file = File::open(path).at(path)?;
        for line in BufReader::new(file).lines() {
            let line = line.at(path)?;
            if line.trim().is_empty() {
                continue;
            }
            if let Some(space) = Self::detect_in_line(&line) {
                return Ok(space);
            }
        }
        Err(Error::EmptyResults(path.to_path_buf()))
    }
}

impl std::str::FromStr for ObjectiveSpace {
    type Err = String;

    /// Accepts the tag and the variant name, so `--objectives obj10` and
    /// `--objectives legacy10` both work.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "obj10" | "legacy10" | "legacy" | "10" => Ok(Self::Legacy10),
            "obj6" | "current6" | "current" | "6" => Ok(Self::Current6),
            other => Err(format!(
                "unknown objective space `{other}`; expected obj10 (legacy10) or obj6 (current6)"
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

/// The metrics that have both a projected and a manifold reading, as
/// `(projected, manifold)`.
///
/// **This is a diagnostic table, not the objective list.** It used to generate
/// [`OBJECTIVES`] by interleaving; it no longer does, and the two are now
/// independent — an objective with no manifold variant would correctly not
/// appear here.
///
/// Its remaining consumer is `figures/exp4.rs`, which plots one panel per row
/// to compare the manifold and projected readings of the same metric. That
/// figure is the evidence for dropping the manifold objectives, so it outlives
/// them.
///
/// Derived by matching `QualityMetric::base` over the registry, so a metric
/// that gains or loses a twin gains or loses a panel with no edit here.
pub static METRIC_PAIRS: LazyLock<Vec<(Metric, Metric)>> =
    LazyLock::new(|| Metric::dual_pairs().collect());

/// Length of [`METRIC_PAIRS`], as a `const` because `figures/exp4.rs` uses it
/// as an array length and `QualityMetric`'s methods are not const-callable.
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
/// *forced* short-circuits the scan (the `--objectives` flag). Otherwise every
/// cell is peeked and the answer must be **unanimous**: a directory holding
/// both legacy and re-run sweeps cannot produce one coherent table, because
/// R2 in one space and R2 in the other are not comparable and every table this
/// crate writes differences them (ΔR2 against a baseline cell, Experiment 1's
/// matched-minus-mismatched gain, the ε-indicator pair). Failing here is the
/// point: the alternative is a table whose rows are silently in two units.
///
/// # Errors
///
/// Returns [`Error::MixedObjectiveSpaces`] when the cells disagree, and
/// propagates I/O errors from peeking them.
pub fn resolve_space(cells: &[CellFile], forced: Option<ObjectiveSpace>) -> Result<ObjectiveSpace> {
    if let Some(space) = forced {
        return Ok(space);
    }
    let mut found: Option<(ObjectiveSpace, String)> = None;
    for cf in cells {
        let space = ObjectiveSpace::detect_in_file(&cf.path)?;
        match &found {
            None => found = Some((space, cf.stem.clone())),
            Some((seen, first)) if *seen != space => {
                return Err(Error::MixedObjectiveSpaces {
                    first: first.clone(),
                    first_space: seen.tag(),
                    second: cf.stem.clone(),
                    second_space: space.tag(),
                })
            }
            Some(_) => {}
        }
    }
    // No cells at all is the callers' `NoCells`, not ours; answer conservatively.
    Ok(found.map_or(ObjectiveSpace::Legacy10, |(space, _)| space))
}
