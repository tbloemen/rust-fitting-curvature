//! Loading trial records from the optimizer's JSONL output, and writing this
//! crate's own tables back out in the same format ([`write_jsonl`]).
//!
//! Every field is optional so that a missing column behaves like Python's
//! `dict.get(...)` did: absent and `null` are the same thing, and a results file
//! written by an older optimizer build still loads. The optimizer serialises
//! non-finite metrics as `null`, so `Option<f64>` covers diverged trials too.
//!
//! A *missing field* is therefore fine; a *malformed line* is not. See
//! [`load_jsonl`].

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

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

    #[serde(default)]
    pub trustworthiness: Option<f64>,
    #[serde(default)]
    pub trustworthiness_manifold: Option<f64>,
    #[serde(default)]
    pub continuity: Option<f64>,
    #[serde(default)]
    pub continuity_manifold: Option<f64>,
    #[serde(default)]
    pub normalized_stress: Option<f64>,
    #[serde(default)]
    pub normalized_stress_manifold: Option<f64>,
    #[serde(default)]
    pub shepard_goodness: Option<f64>,
    #[serde(default)]
    pub shepard_goodness_manifold: Option<f64>,
    #[serde(default)]
    pub neighborhood_hit: Option<f64>,
    #[serde(default)]
    pub neighborhood_hit_manifold: Option<f64>,

    #[serde(default)]
    pub r_max: Option<f64>,
    #[serde(default)]
    pub r_rms: Option<f64>,

    /// Present only on `--mode scan` sweeps, which are excluded from analysis.
    #[serde(default)]
    pub scan_param: Option<String>,
}

impl TrialRecord {
    /// One of the 10 objectives by name. Unknown names give `None`, so a caller
    /// that mistypes gets the same "missing" treatment a null would produce.
    pub fn objective(&self, name: &str) -> Option<f64> {
        match name {
            "trustworthiness" => self.trustworthiness,
            "trustworthiness_manifold" => self.trustworthiness_manifold,
            "continuity" => self.continuity,
            "continuity_manifold" => self.continuity_manifold,
            "normalized_stress" => self.normalized_stress,
            "normalized_stress_manifold" => self.normalized_stress_manifold,
            "shepard_goodness" => self.shepard_goodness,
            "shepard_goodness_manifold" => self.shepard_goodness_manifold,
            "neighborhood_hit" => self.neighborhood_hit,
            "neighborhood_hit_manifold" => self.neighborhood_hit_manifold,
            _ => None,
        }
    }

    /// One hyperparameter / geometry scalar by name (for the marginal figures).
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
            "r_max" => self.r_max,
            "r_rms" => self.r_rms,
            _ => None,
        }
    }

    /// Dimensionless embedding curvature κ = |K|·R_rms² for one trial.
    ///
    /// `curvature_magnitude` is |K|; `r_rms` is the RMS geodesic radius. `None`
    /// when either is missing or non-finite (diverged / Euclidean trials).
    pub fn kappa(&self) -> Option<f64> {
        let k = self.curvature_magnitude?;
        let r = self.r_rms?;
        if !(k.is_finite() && r.is_finite()) {
            return None;
        }
        Some(k * r * r)
    }
}

/// Load every JSON object from a JSONL file.
///
/// Strict: a file that cannot be opened and a line that does not deserialise are
/// both errors. The Python this replaced globbed optimistically and skipped
/// whatever it could not read, which made a half-written results file from a
/// killed sweep indistinguishable from a short one — the front would quietly be
/// computed over fewer trials. Blank lines are still skipped; the optimizer's
/// writer can leave a trailing newline.
pub fn load_jsonl(path: impl AsRef<Path>) -> Result<Vec<TrialRecord>> {
    let path = path.as_ref();
    let file = File::open(path).at(path)?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.at(path)?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let rec: TrialRecord =
            serde_json::from_str(line).map_err(|e| Error::parse(path, i + 1, e))?;
        out.push(rec);
    }
    Ok(out)
}

/// Trial records from a results JSONL, excluding `--mode scan` sweeps.
pub fn trial_records(path: impl AsRef<Path>) -> Result<Vec<TrialRecord>> {
    let mut recs = load_jsonl(path)?;
    recs.retain(|r| r.scan_param.is_none());
    Ok(recs)
}

/// Write *rows* to *path*, one JSON object per line.
///
/// Every table this crate produces goes through here, which is why the analysis
/// output is the same format as its input: the optimizer writes JSONL, the
/// figures read JSONL, and a stage-2 table is no different — one record per
/// line, self-describing, `null` for a value that does not exist. A hand-rolled
/// CSV writer used to sit in `bin/r2.rs` doing this job with format strings and
/// an empty field for `None`.
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
