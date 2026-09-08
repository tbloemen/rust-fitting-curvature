//! Loading trial records from the optimizer's JSONL output, and writing this
//! crate's own tables back out in the same format ([`write_jsonl`]).
//!
//! Every field is optional: absent and `null` are the same thing, so a results
//! file written by an older optimizer build still loads, and the optimizer
//! serialises non-finite metrics as `null`, so `Option<f64>` covers diverged
//! trials too.
//!
//! A *missing field* is therefore fine; a *malformed line* is not. See
//! [`load_jsonl`].

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use fitting_core::metrics::{Metric, MetricValues, R_GYRATION, R_RMS};

use crate::error::{Error, IoContext, Result};

/// One line of a results JSONL file. Unknown fields (extra metrics, timings) are
/// ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TrialRecord {
    #[serde(default)]
    pub dataset_name: Option<String>,
    #[serde(default)]
    pub n_samples: Option<usize>,
    #[serde(default)]
    pub geometry: Option<String>,
    #[serde(default)]
    pub curvature: Option<f64>,
    #[serde(default)]
    pub curvature_magnitude: Option<f64>,

    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub perplexity_ratio: Option<f64>,
    #[serde(default)]
    pub momentum_main: Option<f64>,
    #[serde(default)]
    pub centering_weight: Option<f64>,
    #[serde(default)]
    pub global_loss_weight: Option<f64>,
    #[serde(default)]
    pub norm_loss_weight: Option<f64>,
    #[serde(default)]
    pub early_exaggeration_factor: Option<f64>,

    /// Every metric on this line, read by name from the flat JSONL columns.
    ///
    /// Absent, `null` and non-finite all read back as absent through
    /// [`fitting_core::metrics::MetricValues::get`], and a column belonging to
    /// a retired metric is ignored — which is what lets the ~350 MB of
    /// `results/` written by older builds keep loading unchanged.
    #[serde(flatten)]
    pub metrics: MetricValues,

    /// Present only on `--mode scan` sweeps, which are excluded from analysis.
    #[serde(default)]
    pub scan_param: Option<String>,
}

impl TrialRecord {
    /// One metric by wire name, or `None` if this record does not carry it.
    ///
    /// This was a hand-written match arm per metric — a fourth copy of the
    /// name-to-value mapping, and one where a typo reads as permanently
    /// missing, i.e. silently worst-case, rather than as an error.
    pub fn objective(&self, name: &str) -> Option<f64> {
        self.metrics.get(Metric::by_name(name)?)
    }

    pub fn param(&self, name: &str) -> Option<f64> {
        match name {
            "learning_rate" => self.learning_rate,
            "perplexity_ratio" => self.perplexity_ratio,
            "momentum_main" => self.momentum_main,
            "centering_weight" => self.centering_weight,
            "global_loss_weight" => self.global_loss_weight,
            "norm_loss_weight" => self.norm_loss_weight,
            "early_exaggeration_factor" => self.early_exaggeration_factor,
            "curvature_magnitude" => self.curvature_magnitude,
            "curvature" => self.curvature,
            // The spread diagnostics are registry metrics; the figures ask
            // for them through `param` alongside the hyperparameters.
            other => Metric::by_name(other).and_then(|m| self.metrics.get(m)),
        }
    }

    /// Dimensionless embedding curvature κ = |K|·R_rms² for one trial
    /// (thesis `@eq:kappa` at the embedding gauge, `4methods.typ` §gauge-fixing).
    ///
    /// `|K|` prefers `curvature_magnitude` and falls back to `|curvature|`.
    /// The fallback is not cosmetic: **Euclidean sweeps write `curvature: 0.0`
    /// and omit `curvature_magnitude` entirely** (0 of 1032 trials carry it in
    /// `all_off_grid_euclidean.jsonl`, while every hyperbolic trial does), so
    /// without it every Euclidean cell reports no κ at all. Euclidean space has
    /// `K = 0`, hence `κ = 0` exactly on any gauge — that is a value, not a
    /// missing measurement, and a table that prints `---` for it is wrong.
    pub fn kappa(&self) -> Option<f64> {
        let k = self
            .curvature_magnitude
            .or_else(|| self.curvature.map(f64::abs))?;
        let r = self.metrics.get(R_RMS)?;
        if !(k.is_finite() && r.is_finite()) {
            return None;
        }
        Some(k * r * r)
    }

    /// κ gauged by the origin-free radius, `|K|·r_gyration²`.
    ///
    /// The same quantity [`TrialRecord::kappa`] reports, measured without a
    /// pole. On the hyperboloid and in Euclidean space the two agree closely —
    /// `Hyperboloid::center` runs every iteration, so the origin already *is*
    /// the centroid. On the sphere they do not: `Sphere::center` is a no-op and
    /// `lift_pca_to_manifold` puts the constrained coordinate in the last
    /// ambient slot while `distances_from_origin` reads the first, so PCA init
    /// lands every point at ~90° from the pole κ is measured from and
    /// [`TrialRecord::kappa`] sits at `π²/4 ≈ 2.4674` whatever `|K|` is
    /// (68% of the 71,839 spherical trials in `results/`, whole range
    /// `[1.66, 4.93]`).
    ///
    /// `None` for every file written before the fix, which is every file under
    /// `results/`. Callers that need a κ for both sets must say which gauge they
    /// are using rather than silently falling back — the two are not comparable
    /// on the spherical arm.
    pub fn kappa_gyration(&self) -> Option<f64> {
        let k = self
            .curvature_magnitude
            .or_else(|| self.curvature.map(f64::abs))?;
        let r = self.metrics.get(R_GYRATION)?;
        if !(k.is_finite() && r.is_finite()) {
            return None;
        }
        Some(k * r * r)
    }
}

/// Load every JSON object from a JSONL file.
///
/// Strict: a file that cannot be opened and a line that does not deserialise are
/// both errors, so a half-written results file from a killed sweep fails the run
/// instead of quietly contributing a front computed over fewer trials. Blank
/// lines are skipped; the optimizer's writer can leave a trailing newline.
pub fn load_jsonl<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    let path = path.as_ref();
    let file = File::open(path).at(path)?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.at(path)?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        out.push(serde_json::from_str(line).map_err(|e| Error::parse(path, i + 1, e))?);
    }
    Ok(out)
}

/// Trial records from a results JSONL, excluding `--mode scan` sweeps.
pub fn trial_records(path: impl AsRef<Path>) -> Result<Vec<TrialRecord>> {
    let mut recs: Vec<TrialRecord> = load_jsonl(path)?;
    recs.retain(|r| r.scan_param.is_none());
    Ok(recs)
}

/// Write *rows* to *path*, one JSON object per line.
///
/// Every table this crate produces goes through here, which is why the analysis
/// output is the same format as its input — one record per line,
/// self-describing, `null` for a value that does not exist.
///
/// Unlike the optimizer's per-trial writer (which appends, so a killed sweep
/// leaves a valid prefix), this **truncates**: these tables are recomputed
/// whole from the results directory every run.
pub fn write_jsonl<T: Serialize>(
    path: impl AsRef<Path>,
    rows: impl IntoIterator<Item = T>,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).at(parent)?;
        }
    }
    let mut out = BufWriter::new(File::create(path).at(path)?);
    for row in rows {
        let line = serde_json::to_string(&row).map_err(Error::Serialize)?;
        writeln!(out, "{line}").at(path)?;
    }
    out.flush().at(path)
}
