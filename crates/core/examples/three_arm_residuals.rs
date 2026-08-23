//! The three constant-curvature arms — spherical, hyperbolic, euclidean —
//! fitted to every thesis dataset and reported on one axis.
//!
//! Each arm minimises Wilson's residual-eigenvalue criterion `Σ|λ_res|` over
//! the eigenvalues of its Gram matrix `Z` lying outside the signature block a
//! genuine `dim`-dimensional configuration may occupy, then divides by the
//! dataset constants `n · d_max²` so the three numbers are comparable across
//! arms *and* across datasets.
//!
//! The point of the euclidean arm is that it is the **nested null model**, not
//! a third sibling: `B = −J D∘D J/2` is the `r → ∞` limit of both curved
//! kernels (Wilson et al. 2014, eqs. 24–26). So the curved arms can always
//! match it by running flat-ward, and — carrying a free parameter `r` it lacks
//! — can only do better within their windows. Reading this table is therefore
//! a nested-model question, "does allowing curvature buy a strictly better fit
//! than flat?", and the `Reuc/Rhyp` column is the thing to read, not the bare
//! argmin.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release -p fitting-core --example three_arm_residuals
//! ```
//!
//! Data root defaults to `www/public/data`; pass another as the first argument.

use fitting_core::curvature_detection::{
    detect_hyperbolic, fit_euclidean, fit_hyperbolic, fit_spherical,
};
use fitting_core::data::{load_fashion_mnist, load_mnist, load_pbmc, load_wordnet_mammals};
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::synthetic_data::{
    generate_hd_antipodal_clusters, generate_hd_hyperbolic_shells, generate_hd_sphere,
    generate_hd_tree, generate_uniform_grid, generate_uniform_hyperbolic, DataPoints,
};

const N: usize = 400;
const SEED: u64 = 42;
const DIM: usize = 2;
/// Ambient dimension the curved synthetics are lifted into, matching
/// `optimizer::Dataset::load_synthetic`.
const HD: usize = 10;

fn distances_for(d: &DataPoints) -> Vec<f64> {
    if !d.distances.is_empty() {
        d.distances.clone()
    } else {
        compute_euclidean_distance_matrix(&d.x, d.n_points, d.ambient_dim)
    }
}

fn d_rms_of(d: &[f64], n: usize) -> f64 {
    let s: f64 = d.iter().map(|x| x * x).sum();
    (s / (n as f64 * (n as f64 - 1.0))).sqrt()
}

struct Row {
    name: &'static str,
    /// What the dataset is by construction, for reading the table against.
    truth: &'static str,
    r_sph: f64,
    r_hyp: f64,
    r_euc: f64,
    winner: &'static str,
    /// How much the hyperbolic arm's extra parameter actually buys over flat.
    margin_hyp: f64,
    /// Same for the spherical arm.
    margin_sph: f64,
    kappa_star: f64,
    hyp_pinned: bool,
    sph_pinned: bool,
    gromov: bool,
}

fn build_row(name: &'static str, truth: &'static str, d: &[f64], n: usize) -> Row {
    let sph = fit_spherical(d, n, DIM);
    let hyp = fit_hyperbolic(d, n, DIM);
    let euc = fit_euclidean(d, n, DIM);

    let (r_sph, r_hyp, r_euc) = (
        sph.residual_normalised,
        hyp.residual_normalised,
        euc.residual_normalised,
    );

    // Plain argmin, deliberately with no margin — the question this table
    // exists to answer is whether a margin is needed at all.
    let winner = if r_sph <= r_hyp && r_sph <= r_euc {
        "spherical"
    } else if r_hyp <= r_euc {
        "hyperbolic"
    } else {
        "euclidean"
    };

    let d_rms = d_rms_of(d, n);
    Row {
        name,
        truth,
        r_sph,
        r_hyp,
        r_euc,
        winner,
        margin_hyp: r_euc / r_hyp,
        margin_sph: r_euc / r_sph,
        kappa_star: (d_rms / hyp.radius).powi(2),
        hyp_pinned: hyp.at_upper_bound,
        sph_pinned: sph.at_upper_bound,
        gromov: detect_hyperbolic(d, n).is_hyperbolic,
    }
}

fn main() {
    let data_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "www/public/data".to_string());

    let mut rows: Vec<Row> = Vec::new();

    // Exact-manifold controls: the only cases where a curved model is true.
    let h2 = generate_uniform_hyperbolic(N, SEED, 5.0);
    rows.push(build_row(
        "H2 exact (ctrl)",
        "hyperbolic",
        &distances_for(&h2),
        N,
    ));

    let synth: [(&'static str, &'static str, DataPoints); 5] = [
        ("tree", "hyperbolic", generate_hd_tree(N, HD, SEED)),
        (
            "hyperbolic_shells",
            "hyperbolic",
            generate_hd_hyperbolic_shells(N, HD, SEED),
        ),
        ("sphere", "spherical", generate_hd_sphere(N, HD, SEED)),
        (
            "antipodal_clusters",
            "spherical",
            generate_hd_antipodal_clusters(N, HD, SEED),
        ),
        ("grid", "euclidean", generate_uniform_grid(N, SEED)),
    ];
    for (name, truth, dp) in synth {
        let d = distances_for(&dp);
        let n = dp.n_points;
        rows.push(build_row(name, truth, &d, n));
    }

    let reals: [(&'static str, Result<DataPoints, String>); 4] = [
        ("mnist", load_mnist(&format!("{data_root}/mnist"), N)),
        (
            "fashion_mnist",
            load_fashion_mnist(&format!("{data_root}/fashion-mnist"), N),
        ),
        ("pbmc", load_pbmc(&format!("{data_root}/pbmc"), N)),
        (
            "wordnet_mammals",
            load_wordnet_mammals(&format!("{data_root}/wordnet"), N),
        ),
    ];
    for (name, loaded) in reals {
        match loaded {
            Ok(dp) => {
                let d = distances_for(&dp);
                let n = dp.n_points;
                rows.push(build_row(name, "?", &d, n));
            }
            Err(e) => eprintln!("{name} skipped: {e}"),
        }
    }

    println!(
        "\nThree-arm signature residuals, normalised by n*d_max^2 (n={N}, dim={DIM}, seed={SEED})\n"
    );
    println!(
        "{:<20}{:>12}{:>11}{:>11}{:>11}{:>12}{:>11}{:>11}{:>9}{:>9}  {}",
        "dataset",
        "truth",
        "R_sph",
        "R_hyp",
        "R_euc",
        "winner",
        "Reuc/Rhyp",
        "Reuc/Rsph",
        "kappa*",
        "pinned",
        "gromov"
    );
    println!("{}", "-".repeat(140));
    for r in &rows {
        let pinned = match (r.hyp_pinned, r.sph_pinned) {
            (true, true) => "h+s",
            (true, false) => "hyp",
            (false, true) => "sph",
            (false, false) => "-",
        };
        println!(
            "{:<20}{:>12}{:>11.3e}{:>11.3e}{:>11.3e}{:>12}{:>11.2e}{:>11.2e}{:>9.4}{:>9}  {}",
            r.name,
            r.truth,
            r.r_sph,
            r.r_hyp,
            r.r_euc,
            r.winner,
            r.margin_hyp,
            r.margin_sph,
            r.kappa_star,
            pinned,
            if r.gromov { "hyperbolic" } else { "-" },
        );
    }

    println!("\n  R_*        = Wilson residual-eigenvalue misfit of each arm, same n*d_max^2 gauge");
    println!("  winner     = plain argmin over the three arms, NO margin applied");
    println!("  Reuc/Rhyp  = what the hyperbolic arm's free parameter r buys over the flat null;");
    println!("               >1 means curvature helps, ~1 means it explains nothing");
    println!("  kappa*     = |K|*d_rms^2 of the hyperbolic fit; equals HYPERBOLIC_KAPPA_MIN");
    println!("               (0.01) exactly when pinned, i.e. the bound, not a measurement");
    println!("  pinned     = which curved arm(s) came to rest on their flat-ward window edge");
    println!("  gromov     = the CURRENT production hyperbolicity gate, for comparison");

    let agree = rows.iter().filter(|r| r.truth == r.winner).count();
    let labelled = rows.iter().filter(|r| r.truth != "?").count();
    println!("\n  argmin matches construction on {agree}/{labelled} datasets of known geometry");
}
