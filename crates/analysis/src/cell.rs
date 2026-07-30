//! Cell identity: parsing a results-file stem into its experiment coordinates.

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

impl std::fmt::Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cell({}/{}/N{}/{})",
            self.setting, self.dataset, self.n, self.geometry
        )
    }
}

/// Parse a results file stem like `all_off_mnist_n5000_hyperbolic`.
///
/// Returns `None` for names that are not a plain trial-results stem (e.g.
/// `*_pareto_*` front files, which contain a second geometry token).
///
/// Ports the Python regex `^(setting)_(.+?)(_n5000)?_(geometry)$`. The dataset
/// group is non-greedy and the geometry is anchored at the end, so the split is
/// unambiguous even though dataset names contain underscores.
pub fn parse_cell_stem(stem: &str) -> Option<Cell> {
    if stem.contains("_pareto_") {
        return None;
    }
    // Longest match first so `all_free` is never parsed as setting `all_off`-like
    // prefix of something else; the settings list has no shared prefixes, but
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
