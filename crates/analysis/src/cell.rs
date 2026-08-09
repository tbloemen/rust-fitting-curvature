//! Cell identity: parsing a results-file stem into its experiment coordinates,
//! and finding every results file under a directory.

use std::path::{Path, PathBuf};

use crate::error::{IoContext, Result};

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
pub const GEOMETRIES: [&str; 3] = ["euclidean", "hyperbolic", "spherical"];

/// A single (setting, dataset, N, geometry) experiment cell.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cell {
    pub setting: String,
    pub dataset: String,
    pub n: usize,
    pub geometry: String,
}

impl Cell {
    pub fn new(setting: &str, dataset: &str, n: usize, geometry: &str) -> Self {
        Self {
            setting: setting.to_string(),
            dataset: dataset.to_string(),
            n,
            geometry: geometry.to_string(),
        }
    }
}

/// Parse a results file stem like `all_off_mnist_n5000_hyperbolic`.
///
/// Returns `None` for names that are not a plain trial-results stem (e.g.
/// `*_pareto_*` front files, which contain a second geometry token).
///
/// The setting is anchored at the start and the geometry at the end, so the
/// split is unambiguous even though dataset names contain underscores.
pub fn parse_cell_stem(stem: &str) -> Option<Cell> {
    if stem.contains("_pareto_") {
        return None;
    }
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
    Some(Cell::new(setting, dataset, n, geometry))
}

/// One results file and the experiment cell its name encodes.
pub struct CellFile {
    pub path: PathBuf,
    /// The file stem, which every downstream table keys by. Carried along
    /// because `parse_cell_stem` already proved it is valid UTF-8.
    pub stem: String,
    pub cell: Cell,
}

/// Every trial-results JSONL under *results_dir*, with its parsed cell.
///
/// Front files (`*_pareto_*.json`) and anything whose stem doesn't parse as a
/// cell are skipped. Sorted by stem so the output order is stable.
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
        if let Some(cell) = parse_cell_stem(stem) {
            out.push(CellFile {
                stem: stem.to_string(),
                path: path.clone(),
                cell,
            });
        }
    }
    out.sort_by(|a, b| a.stem.cmp(&b.stem));
    Ok(out)
}
