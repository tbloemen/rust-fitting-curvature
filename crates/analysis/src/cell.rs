//! Cell identity: parsing a results-file stem into its experiment coordinates,
//! and finding every results file under a directory.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, IoContext, Result};

/// The loss-weight settings a sweep can be run under.
pub const SETTINGS: [&str; 6] = [
    "all_free",
    "all_off",
    "centering_only",
    "global_only",
    "norm_only",
    "rms_anchored",
];

/// The three constant-curvature geometries.
///
/// Everywhere in this crate `geometry` means the *embedding model a sweep was
/// run under*, never a claim about the data. For the datasets whose geometry is
/// known by construction, see [`SYNTH_TRUTH`].
pub const GEOMETRIES: [&str; 3] = ["euclidean", "hyperbolic", "spherical"];

/// Intrinsic geometry of each synthetic dataset, by construction.
///
/// Experiment 1 asks whether the embedding geometry *matching* the data beats
/// the Euclidean baseline, so it needs a ground truth that no other table does:
/// everywhere else `geometry` is the model that was fitted.
///
/// The origin of these labels is `crates/core/examples/common/mod.rs`
/// (`Fixture::truth`), which names the same generators `"sphere 10D"`,
/// `"tree 10D"`, `"grid 10D"`. The names here are the optimizer's, from
/// `crates/optimizer/src/data.rs::load_synthetic` — that is what appears in
/// results filenames, and the two vocabularies do not join automatically.
///
/// The real datasets are deliberately absent: their geometry is the question,
/// not the given.
pub const SYNTH_TRUTH: [(&str, &str); 11] = [
    // The original suite. Unchanged generators, so the results already under
    // `results/` stay valid for these four.
    //
    // `antipodal_clusters` was the fifth and is **retired**: dropped from the
    // sweep grid in 7ed38aa and from the reported set here, so nothing in the
    // analysis tables or figures carries it any more. Its generator and its
    // cells under `results/` both survive — retiring a dataset stops it being
    // reported, it does not delete anything.
    ("grid", "euclidean"),
    ("sphere", "spherical"),
    ("tree", "hyperbolic"),
    ("hyperbolic_shells", "hyperbolic"),
    // A tree metric, not a manifold. "hyperbolic" here means the geometry that
    // can *represent* the metric — the same footing `wordnet_mammals` sits on —
    // not a claim that the data has constant sectional curvature.
    ("tree_graph", "hyperbolic"),
    // The matched geodesic balls: within a tier these share a sampling scheme,
    // a radius and a label vector, and differ only in curvature. Tier 2 is a
    // 2-D source into a 2-D target; tier 9 shares a 9-D source, so the
    // dimension reduction is matched across the three as well.
    ("ball2_euclidean", "euclidean"),
    ("ball2_spherical", "spherical"),
    ("ball2_hyperbolic", "hyperbolic"),
    ("ball9_euclidean", "euclidean"),
    ("ball9_spherical", "spherical"),
    ("ball9_hyperbolic", "hyperbolic"),
];

/// Datasets no analysis output reports, whatever is still on disk.
///
/// `antipodal_clusters` was dropped from the sweep grid in commit 7ed38aa and
/// retired from the reported set here. Its ~30 cells are still under
/// `results/` and its generator still exists — retiring a dataset stops it
/// being *reported*, it deletes nothing, and pointing `--results-dir` at an
/// archive still finds the files.
///
/// Applied in [`discover_cells`], which is the single door every stage walks
/// through, so one entry retires a dataset from the stage-1 table, ΔR2, the
/// ε-indicator, the recommendations, Experiment 1 and every figure at once.
/// Filtering per output instead is what left the Experiment 4 bar charts
/// drawing a dataset the tables had already dropped.
pub const RETIRED_DATASETS: [&str; 1] = ["antipodal_clusters"];

/// Whether *dataset* is retired from the analysis — see [`RETIRED_DATASETS`].
#[must_use]
pub fn is_retired(dataset: &str) -> bool {
    RETIRED_DATASETS.contains(&dataset)
}

/// The geometry `dataset` is built to have, or `None` for a real dataset (or
/// any name not in [`SYNTH_TRUTH`]).
#[must_use]
pub fn truth_of(dataset: &str) -> Option<&'static str> {
    SYNTH_TRUTH
        .iter()
        .find(|(name, _)| *name == dataset)
        .map(|(_, truth)| *truth)
}

/// A single (setting, dataset, N, geometry) experiment cell.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cell {
    pub setting: String,
    pub dataset: String,
    pub n: usize,
    pub geometry: String,
}

impl Cell {
    #[must_use]
    pub fn new(setting: &str, dataset: &str, n: usize, geometry: &str) -> Self {
        Self {
            setting: setting.to_string(),
            dataset: dataset.to_string(),
            n,
            geometry: geometry.to_string(),
        }
    }
}

/// Which sweep a results file came from, when its stem carries a marker after
/// the geometry token. `None` — no marker — is the original sweeps.
///
/// The marker lives in the *filename* purely as a safety net: the sets are kept
/// in separate directories (`results/` vs `results-rgyr/`), and differing
/// basenames mean a mis-targeted rsync cannot silently overwrite one with the
/// other. It is **not** part of [`Cell`]; see [`parse_cell_stem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Variant {
    /// `_rgyr` — the N=5000 curved re-run that logs the origin-free
    /// `r_gyration` column ([`crate::records::TrialRecord::kappa_gyration`])
    /// alongside the pole-relative `r_rms`.
    Rgyr,
}

impl Variant {
    /// Every variant, for exhaustive matching against a stem.
    pub const ALL: [Variant; 1] = [Variant::Rgyr];

    /// The stem suffix this variant is written as, without the separating `_`.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Variant::Rgyr => "rgyr",
        }
    }

    /// The variant a suffix names, or `None` if it names none.
    #[must_use]
    pub fn from_suffix(suffix: &str) -> Option<Self> {
        Variant::ALL.into_iter().find(|v| v.suffix() == suffix)
    }
}

impl std::fmt::Display for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.suffix())
    }
}

/// Parse a results file stem like `all_off_mnist_n5000_hyperbolic`, returning
/// the cell and the [`Variant`] the stem carried.
///
/// Returns `None` for names that are not a plain trial-results stem (e.g.
/// `*_pareto_*` front files, which contain a second geometry token).
///
/// The setting is anchored at the start and the geometry at the end, so the
/// split is unambiguous even though dataset names contain underscores.
///
/// **The variant is deliberately not part of [`Cell`].** Every figure builds its
/// lookup keys with `Cell::new(...)`, so a variant field would make those
/// lookups miss and `--results-dir results-rgyr` would render nothing. The
/// directory selects the set; the marker only labels the file. Two files in one
/// directory that parse to the same `Cell` are rejected by [`discover_cells`]
/// rather than silently collapsing into one.
#[must_use]
pub fn parse_cell_stem(stem: &str) -> Option<Cell> {
    parse_cell_stem_variant(stem).map(|(cell, _)| cell)
}

/// [`parse_cell_stem`], also returning the [`Variant`] the stem carried.
#[must_use]
pub fn parse_cell_stem_variant(stem: &str) -> Option<(Cell, Option<Variant>)> {
    if stem.contains("_pareto_") {
        return None;
    }
    // Strip the variant before the geometry anchor: it sits *after* the geometry
    // token, so leaving it on would fail the `ends_with(geometry)` test below and
    // the file would be skipped entirely rather than parsed.
    let (stem, variant) = match Variant::ALL.into_iter().find(|v| {
        let s = v.suffix();
        stem.len() > s.len() + 1 && stem.ends_with(s) && stem[..stem.len() - s.len()].ends_with('_')
    }) {
        Some(v) => (&stem[..stem.len() - v.suffix().len() - 1], Some(v)),
        None => (stem, None),
    };
    // Longest match first: the settings list has no shared prefixes today, but
    // matching in descending length order keeps that robust to new settings.
    let mut settings: Vec<&str> = SETTINGS.to_vec();
    settings.sort_by_key(|s| std::cmp::Reverse(s.len()));

    let setting = settings.iter().find(|s| {
        stem.len() > s.len() + 1 && stem.starts_with(**s) && stem[s.len()..].starts_with('_')
    })?;
    let rest = &stem[setting.len() + 1..];

    let geometry = GEOMETRIES.iter().find(|g| {
        rest.len() > g.len() + 1
            && rest.ends_with(**g)
            && rest[..rest.len() - g.len()].ends_with('_')
    })?;
    let middle = &rest[..rest.len() - geometry.len() - 1];

    // `_n5000` is the only sample-size marker the sweeps emit; its absence means
    // the default N=1000 run.
    let (dataset, n) = match middle.strip_suffix("_n5000") {
        Some(ds) => (ds, 5000),
        None => (middle, 1000),
    };
    if dataset.is_empty() {
        return None;
    }
    Some((Cell::new(setting, dataset, n, geometry), variant))
}

/// One results file and the experiment cell its name encodes.
pub struct CellFile {
    pub path: PathBuf,
    /// The file stem, which every downstream table keys by. Carried along
    /// because `parse_cell_stem` already proved it is valid UTF-8.
    pub stem: String,
    pub cell: Cell,
    /// Which sweep this file came from, or `None` for the original ones. Not
    /// part of [`Cell`]: it labels the file, not the experiment.
    pub variant: Option<Variant>,
}

/// Every trial-results JSONL under *`results_dir`*, with its parsed cell.
///
/// Front files (`*_pareto_*.json`), anything whose stem doesn't parse as a
/// cell, and cells of a [`RETIRED_DATASETS`] dataset are skipped. Sorted by
/// stem so the output order is stable.
///
/// # Errors
///
/// Returns `Err` if `read_dir` fails or a directory entry cannot be read.
pub fn discover_cells(results_dir: &Path) -> Result<Vec<CellFile>> {
    let mut out: Vec<CellFile> = Vec::new();
    for entry in std::fs::read_dir(results_dir).at(results_dir)? {
        let path = entry.at(results_dir)?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some((cell, variant)) = parse_cell_stem_variant(stem) {
            if is_retired(&cell.dataset) {
                continue;
            }
            out.push(CellFile {
                stem: stem.to_string(),
                path: path.clone(),
                cell,
                variant,
            });
        }
    }
    out.sort_by(|a, b| a.stem.cmp(&b.stem));

    // Two stems mapping to one cell means two variants of the same experiment
    // landed in one directory — a mis-targeted rsync, most likely. Downstream
    // every cell goes into a `BTreeMap` keyed by `Cell`, so the second insert
    // would silently discard the first and every table would be computed over
    // whichever file happened to sort last. Fail loudly instead.
    //
    // Keyed rather than pairwise on the sorted vec: the sort is by *stem*, and
    // two stems for one cell need not be adjacent under it.
    let mut seen: BTreeMap<&Cell, &str> = BTreeMap::new();
    for cf in &out {
        if let Some(first) = seen.insert(&cf.cell, &cf.stem) {
            return Err(Error::DuplicateCell {
                cell: format!("{:?}", cf.cell),
                first: first.to_string(),
                second: cf.stem.clone(),
            });
        }
    }
    Ok(out)
}
