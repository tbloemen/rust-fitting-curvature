//! The thesis datasets, defined once for every curvature-detection example.
//!
//! `three_arm_residuals`, `wilson_residual_curve`, `wilson_residual_curve_real`
//! and `residual_vs_kappa` all report on the same ten datasets, and the numbers
//! are read against one another — the residual table states where each fit
//! lands, the curve plots show the objective it landed on. That only holds if
//! all four build the *same* distance matrices, so the fixture list lives here
//! rather than being copied into each example.
//!
//! Colours are plain RGB triples rather than `plotters::RGBColor` because
//! `three_arm_residuals` builds without the `plot-examples` feature.

#![allow(dead_code)]

use fitting_core::data::{load_fashion_mnist, load_mnist, load_pbmc, load_wordnet_mammals};
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::synthetic_data::{
    generate_hd_antipodal_clusters, generate_hd_hyperbolic_shells, generate_hd_sphere,
    generate_hd_tree, generate_hd_uniform_grid, generate_tree_structured, generate_uniform_grid,
    generate_uniform_hyperbolic, generate_uniform_sphere, DataPoints,
};

/// Target dimension of the fitted model manifold: the two-dimensional
/// embedding space the visualisations use.
pub const DIM: usize = 2;
/// Ambient dimension the `*_10D` synthetics are lifted into, matching
/// `optimizer::Dataset::load_synthetic`.
pub const HD: usize = 10;
/// Sample size the thesis table and plots are generated at.
pub const DEFAULT_N: usize = 1000;
pub const DEFAULT_SEED: u64 = 42;
pub const DEFAULT_DATA_ROOT: &str = "www/public/data";

/// One dataset, resolved to the pairwise distance matrix every fit consumes.
pub struct Fixture {
    pub name: &'static str,
    /// Geometry by construction; `"?"` for the real datasets, whose geometry
    /// is the question rather than the given.
    pub truth: &'static str,
    /// tab10 RGB, paired by family so `sphere 2D` / `sphere 10D` read as two
    /// shades of one colour.
    pub color: (u8, u8, u8),
    pub distances: Vec<f64>,
    pub n: usize,
}

pub fn distances_for(d: &DataPoints) -> Vec<f64> {
    if !d.distances.is_empty() {
        d.distances.clone()
    } else {
        compute_euclidean_distance_matrix(&d.x, d.n_points, d.ambient_dim)
    }
}

pub fn d_max_of(d: &[f64]) -> f64 {
    d.iter().cloned().fold(0.0_f64, f64::max)
}

/// Root-mean-square of the `n(n−1)` off-diagonal pairwise distances — a scale
/// statistic of the *input*, reported alongside the fits. It is not the gauge
/// length of `κ`, which is `R_rms`, the extent of the configuration a fit
/// implies (see `Reconstruction::kappa`).
pub fn d_rms_of(d: &[f64], n: usize) -> f64 {
    let s: f64 = d.iter().map(|x| x * x).sum();
    (s / (n as f64 * (n as f64 - 1.0))).sqrt()
}

fn fixture(
    name: &'static str,
    truth: &'static str,
    color: (u8, u8, u8),
    dp: DataPoints,
) -> Fixture {
    Fixture {
        name,
        truth,
        color,
        distances: distances_for(&dp),
        n: dp.n_points,
    }
}

/// The six synthetic datasets, in thesis-table order.
///
/// The `2D` / `10D` suffixes follow the thesis's dataset labels and are not a
/// single convention: for `sphere` and `tree`, `2D` is the *intrinsic*
/// dimension (S² and a tree in H², both in R³) while `10D` is the *ambient*
/// one (S⁹ and H⁹ in R¹⁰); for `grid` the two coincide.
pub fn synthetic(n: usize, seed: u64) -> Vec<Fixture> {
    vec![
        fixture(
            "grid 10D",
            "euclidean",
            (148, 103, 189),
            generate_hd_uniform_grid(n, HD, seed),
        ),
        fixture(
            "grid 2D",
            "euclidean",
            (197, 176, 213),
            generate_uniform_grid(n, seed),
        ),
        fixture(
            "sphere 2D",
            "spherical",
            (44, 160, 44),
            generate_uniform_sphere(n, seed),
        ),
        fixture(
            "sphere 10D",
            "spherical",
            (152, 223, 138),
            generate_hd_sphere(n, HD, seed),
        ),
        fixture(
            "tree 2D",
            "hyperbolic",
            (214, 39, 40),
            generate_tree_structured(n, seed),
        ),
        fixture(
            "tree 10D",
            "hyperbolic",
            (255, 152, 150),
            generate_hd_tree(n, HD, seed),
        ),
    ]
}

/// The four real datasets, in thesis-table order.  A dataset whose files are
/// missing is reported on stderr and dropped rather than aborting the run.
pub fn real(n: usize, data_root: &str) -> Vec<Fixture> {
    type Loaded = (&'static str, (u8, u8, u8), Result<DataPoints, String>);
    let loaded: [Loaded; 4] = [
        (
            "mnist",
            (31, 119, 180),
            load_mnist(&format!("{data_root}/mnist"), n),
        ),
        (
            "mnist-fashion",
            (255, 127, 14),
            load_fashion_mnist(&format!("{data_root}/fashion-mnist"), n),
        ),
        (
            "wordnet-mammals",
            (127, 127, 127),
            load_wordnet_mammals(&format!("{data_root}/wordnet"), n),
        ),
        (
            "pbmc",
            (140, 86, 75),
            load_pbmc(&format!("{data_root}/pbmc"), n),
        ),
    ];

    let mut out = Vec::new();
    for (name, color, result) in loaded {
        match result {
            Ok(dp) => out.push(fixture(name, "?", color, dp)),
            Err(e) => eprintln!("{name} skipped: {e}"),
        }
    }
    out
}

/// All ten thesis datasets: the six synthetics followed by the four real ones.
pub fn thesis(n: usize, seed: u64, data_root: &str) -> Vec<Fixture> {
    let mut out = synthetic(n, seed);
    out.extend(real(n, data_root));
    out
}

/// Extra fixtures that are not in the thesis table.
///
/// `H2 exact` is the only *exact* constant-curvature hyperbolic manifold in
/// the set and the reference case for what a true hyperbolic fit looks like;
/// the other two are the remaining generators from
/// `optimizer::Dataset::load_synthetic`.
pub fn controls(n: usize, seed: u64) -> Vec<Fixture> {
    vec![
        fixture(
            "H2 exact (ctrl)",
            "hyperbolic",
            (0, 0, 0),
            generate_uniform_hyperbolic(n, seed, 5.0),
        ),
        fixture(
            "hyperbolic_shells",
            "hyperbolic",
            (255, 187, 120),
            generate_hd_hyperbolic_shells(n, HD, seed),
        ),
        fixture(
            "antipodal_clusters",
            "spherical",
            (23, 190, 207),
            generate_hd_antipodal_clusters(n, HD, seed),
        ),
    ]
}

/// Flags shared by the examples: `--n`, `--seed`, `--data-root`, `--all`, plus
/// a bare positional argument still accepted as the data root.
pub struct CommonArgs {
    pub n: usize,
    pub seed: u64,
    pub data_root: String,
    pub all: bool,
    /// Flags the caller is responsible for, in the order they appeared.
    pub rest: Vec<(String, Option<String>)>,
}

impl CommonArgs {
    pub fn parse(known: &[&str]) -> Result<Self, String> {
        let mut out = CommonArgs {
            n: DEFAULT_N,
            seed: DEFAULT_SEED,
            data_root: DEFAULT_DATA_ROOT.to_string(),
            all: false,
            rest: Vec::new(),
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
            match flag.as_str() {
                "--n" => out.n = value("--n")?.parse().map_err(|e| format!("--n: {e}"))?,
                "--seed" => {
                    out.seed = value("--seed")?
                        .parse()
                        .map_err(|e| format!("--seed: {e}"))?
                }
                "--data-root" => out.data_root = value("--data-root")?,
                "--all" => out.all = true,
                other if known.contains(&other) => {
                    let v = value(other)?;
                    out.rest.push((other.to_string(), Some(v)));
                }
                // Backwards compatibility with the original positional form.
                other if !other.starts_with("--") => out.data_root = other.to_string(),
                other => return Err(format!("unknown flag {other}")),
            }
        }
        Ok(out)
    }

    pub fn get(&self, flag: &str) -> Option<&str> {
        self.rest
            .iter()
            .find(|(k, _)| k == flag)
            .and_then(|(_, v)| v.as_deref())
    }
}

/// Run `f` on every fixture in parallel — one thread each.  Every example here
/// is a chain of dense `n³` eigensolves per dataset, independent across
/// datasets, so this turns the total wall time into roughly the slowest one's.
pub fn map_parallel<T: Send>(fixtures: &[Fixture], f: impl Fn(&Fixture) -> T + Sync) -> Vec<T> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = fixtures.iter().map(|fx| scope.spawn(|| f(fx))).collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("fixture thread panicked"))
            .collect()
    })
}
