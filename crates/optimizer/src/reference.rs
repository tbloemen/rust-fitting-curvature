//! `--mode reference`: score a dataset's *known* source configuration through
//! the same metric pipeline a t-SNE trial goes through.
//!
//! ## What this measures
//!
//! For a tier-A matched ball the generator's coordinates already are a valid
//! two-dimensional embedding on the manifold the data was built on — the
//! matching geometry can represent the source distances exactly. Scoring that
//! configuration therefore answers "what do the six objectives read when the
//! embedding *is* the ground truth", which separates two things a low trial
//! score otherwise confounds:
//!
//! - **Projection cost.** The objectives are read off `project_to_2d`, so a
//!   curved configuration is scored through a chart (azimuthal-equidistant on
//!   the sphere, Poincaré on the hyperboloid). Whatever the reference loses is
//!   lost to the chart, not to the optimiser.
//! - **Optimisation cost.** Whatever a trial loses *beyond* the reference is
//!   the search failing to recover a configuration that provably exists.
//!
//! It is emphatically **not** a competitor to the Pareto front. See the
//! prohibition in `crates/analysis/CLAUDE.md`: R2 is monotone under set
//! inclusion, so a one-point set can never beat a front containing a comparable
//! point, and every number that scored a singleton against a front has been
//! removed from this repository once already. A reference is a *level* — six
//! objective values on the same bounded, same-oriented axes as the front's —
//! and it must be reported per objective, never collapsed into an indicator.

use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;

use fitting_core::manifolds::create_manifold;
use fitting_core::metrics::MetricValues;
use fitting_core::spread::SpreadDiagnostics;

use crate::cli::Args;
use crate::data::{source_curvature, source_extent};
use crate::evaluate::Evaluator;

/// One scored ground-truth configuration.
///
/// Deliberately not a `TrialResult`: that struct carries the seven
/// hyperparameters as bare `f64`, and a reference has none. Filling them with
/// zeros would make `TrialRecord::param("learning_rate")` return a real-looking
/// `Some(0.0)` to any figure that does not filter. Omitting the fields instead
/// reads back as `None`, because every `TrialRecord` field is
/// `#[serde(default)] Option<_>` and unknown fields are ignored — so
/// `fitting_analysis::records::load_jsonl::<TrialRecord>` parses these lines
/// with no new loader.
#[derive(Debug, Serialize)]
struct ReferenceResult {
    /// Discriminator. Inert to `TrialRecord`, which ignores unknown fields.
    kind: &'static str,
    dataset_name: String,
    n_samples: usize,
    geometry: &'static str,
    curvature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    extent: Option<f64>,

    #[serde(flatten)]
    metrics: MetricValues,
    #[serde(flatten)]
    spread: SpreadDiagnostics,
}

fn geometry_name(curvature: f64) -> &'static str {
    if curvature > 0.0 {
        "spherical"
    } else if curvature < 0.0 {
        "hyperbolic"
    } else {
        "euclidean"
    }
}

pub fn run_reference(dataset_name: &str, args: &Args, evaluator: &Evaluator) {
    let Some(curvature) = source_curvature(dataset_name) else {
        eprintln!(
            "reference '{dataset_name}': skipped — no ground-truth source geometry. \
             Only the tier-A matched balls (ball2_euclidean, ball2_spherical, \
             ball2_hyperbolic) have coordinates that are themselves a valid 2-D embedding."
        );
        return;
    };

    let (points, ambient_dim) = evaluator.source_points();
    let n = evaluator.n_points();
    let expected = create_manifold(curvature).ambient_dim(2);
    if points.is_empty() || ambient_dim != expected {
        eprintln!(
            "reference '{dataset_name}': skipped — source is {ambient_dim}-D ambient, but a \
             2-D target at K={curvature} needs {expected}."
        );
        return;
    }

    let (metrics, spread) = evaluator.score_points(points, ambient_dim, curvature);

    let record = ReferenceResult {
        kind: "reference",
        dataset_name: dataset_name.to_string(),
        n_samples: n,
        geometry: geometry_name(curvature),
        curvature,
        extent: source_extent(dataset_name),
        metrics,
        spread,
    };

    println!(
        "reference '{}' (n={}): {} K={:+.1} — the source configuration scored through the \
         trial pipeline",
        record.dataset_name, n, record.geometry, record.curvature
    );

    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("reference: failed to open {} for append: {e}", args.output);
            return;
        }
    };
    match serde_json::to_string(&record) {
        Ok(json) => {
            writeln!(file, "{json}").ok();
        }
        Err(e) => eprintln!("reference: failed to serialise record: {e}"),
    }
}
