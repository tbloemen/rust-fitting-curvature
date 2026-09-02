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
//! Alongside the residuals each curved arm reports its fitted radius `r*`, the
//! signed sectional curvature `K = ±1/r*²` that radius implies, and the
//! dimensionless `κ`. Only `κ` is comparable across datasets: `r*` and `K`
//! carry the units of the input distance matrix, which are pixel distances for
//! MNIST and hop counts for WordNet.
//!
//! `kappa_* = |K|·R_rms²`, where `R_rms` is the RMS geodesic radius of the
//! configuration the fit implies — the one gauge used throughout the
//! repository, so this κ can be set beside a Pareto front's or `--mode
//! detect`'s.
//!
//! Getting `R_rms` means reconstructing each arm, which is cheap and exact
//! rather than a second fit: each `Z(r*)` *is* the Gram matrix of the model it
//! tests for, so its retained eigen-block is the configuration itself.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release -p fitting-core --example three_arm_residuals
//! cargo run --release -p fitting-core --example three_arm_residuals -- --n 200
//! ```
//!
//! Flags: `--n <usize>`, `--seed <u64>`, `--data-root <path>`,
//! `--jsonl <path>`, `--all`. The data root defaults to `www/public/data`
//! (relative to the cwd, so run from the repo root); a bare positional
//! argument is still accepted as the data root.

use std::fmt::Write as _;
use std::io::Write as _;

mod common;
use common::{d_rms_of, map_parallel, CommonArgs, Fixture, DIM};

use fitting_core::curvature_detection::{
    detect_hyperbolic, fit_euclidean, fit_hyperbolic, fit_spherical, reconstruct_hyperbolic,
    reconstruct_spherical,
};

struct Row {
    name: &'static str,
    /// What the dataset is by construction, for reading the table against.
    /// `"?"` for the real datasets, whose geometry is what we are asking.
    truth: &'static str,
    n: usize,
    d_rms: f64,
    d_max: f64,
    r_sph: f64,
    r_hyp: f64,
    r_euc: f64,
    winner: &'static str,
    /// How much the hyperbolic arm's extra parameter actually buys over flat.
    margin_hyp: f64,
    /// Same for the spherical arm.
    margin_sph: f64,
    /// Fitted radius of the hyperbolic arm, in the units of the input
    /// distance matrix, and the curvature quantities it implies.
    r_star_hyp: f64,
    k_hyp: f64,
    /// `|K|·R_rms²`, measured on the reconstruction this fit implies. Same
    /// formula the t-SNE runs report per trial. A pinned hyperbolic arm reads
    /// exactly `HYPERBOLIC_KAPPA_MIN`, since the search cap is a bound on this
    /// same quantity.
    kappa_hyp: f64,
    /// RMS geodesic radius of the hyperbolic reconstruction, the gauge length
    /// behind `kappa_hyp`.
    r_rms_hyp: f64,
    hyp_pinned: bool,
    /// Same for the spherical arm.
    r_star_sph: f64,
    k_sph: f64,
    kappa_sph: f64,
    r_rms_sph: f64,
    sph_pinned: bool,
    gromov: bool,
}

fn build_row(fx: &Fixture) -> Row {
    let (d, n) = (fx.distances.as_slice(), fx.n);
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

    // κ is reported on the embedding gauge, which needs the configuration the
    // fit implies rather than just its radius. Each `Z(r*)` is the Gram matrix
    // of its own model, so reconstructing is an eigendecomposition, not a
    // second fit — and it runs once per arm, at the already-fitted radius.
    let rec_hyp = reconstruct_hyperbolic(d, n, DIM, hyp.radius);
    let rec_sph = reconstruct_spherical(d, n, DIM, sph.radius);
    let (r_rms_hyp, r_rms_sph) = (rec_hyp.r_rms(n), rec_sph.r_rms(n));

    Row {
        name: fx.name,
        truth: fx.truth,
        n,
        d_rms,
        d_max: d.iter().cloned().fold(0.0_f64, f64::max),
        r_sph,
        r_hyp,
        r_euc,
        winner,
        margin_hyp: r_euc / r_hyp,
        margin_sph: r_euc / r_sph,
        r_star_hyp: hyp.radius,
        k_hyp: -1.0 / (hyp.radius * hyp.radius),
        kappa_hyp: rec_hyp.curvature.abs() * r_rms_hyp * r_rms_hyp,
        r_rms_hyp,
        hyp_pinned: hyp.at_upper_bound,
        r_star_sph: sph.radius,
        k_sph: 1.0 / (sph.radius * sph.radius),
        kappa_sph: rec_sph.curvature.abs() * r_rms_sph * r_rms_sph,
        r_rms_sph,
        sph_pinned: sph.at_upper_bound,
        gromov: detect_hyperbolic(d, n).is_hyperbolic,
    }
}

fn pinned_label(row: &Row) -> &'static str {
    match (row.hyp_pinned, row.sph_pinned) {
        (true, true) => "h+s",
        (true, false) => "hyp",
        (false, true) => "sph",
        (false, false) => "-",
    }
}

/// `{:?}` is Rust's shortest round-tripping `f64` representation, so the JSONL
/// carries the fit exactly. Non-finite values have no JSON literal; `null` is
/// the honest encoding and no field here is expected to hit it (the euclidean
/// arm's infinite radius is simply not emitted).
fn json_f64(x: f64) -> String {
    if x.is_finite() {
        format!("{x:?}")
    } else {
        "null".to_string()
    }
}

fn json_line(row: &Row, seed: u64) -> String {
    let mut s = String::new();
    let _ = write!(s, "{{\"dataset\":{:?}", row.name);
    let _ = write!(s, ",\"truth\":{:?}", row.truth);
    let _ = write!(s, ",\"n\":{}", row.n);
    let _ = write!(s, ",\"dim\":{DIM}");
    let _ = write!(s, ",\"seed\":{seed}");
    for (key, value) in [
        ("d_rms", row.d_rms),
        ("d_max", row.d_max),
        ("r_sph", row.r_sph),
        ("r_euc", row.r_euc),
        ("r_hyp", row.r_hyp),
        ("r_star_hyp", row.r_star_hyp),
        ("k_hyp", row.k_hyp),
        ("kappa_hyp", row.kappa_hyp),
        ("r_rms_hyp", row.r_rms_hyp),
    ] {
        let _ = write!(s, ",\"{key}\":{}", json_f64(value));
    }
    let _ = write!(s, ",\"hyp_pinned\":{}", row.hyp_pinned);
    for (key, value) in [
        ("r_star_sph", row.r_star_sph),
        ("k_sph", row.k_sph),
        ("kappa_sph", row.kappa_sph),
        ("r_rms_sph", row.r_rms_sph),
    ] {
        let _ = write!(s, ",\"{key}\":{}", json_f64(value));
    }
    let _ = write!(s, ",\"sph_pinned\":{}", row.sph_pinned);
    let _ = write!(s, ",\"winner\":{:?}", row.winner);
    let _ = write!(s, ",\"gromov\":{}", row.gromov);
    s.push('}');
    s
}

fn main() {
    let args = match CommonArgs::parse(&["--jsonl"]) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let jsonl = args.get("--jsonl").unwrap_or("three_arm_residuals.jsonl");

    let mut fixtures = common::thesis(args.n, args.seed, &args.data_root);
    if args.all {
        fixtures.extend(common::controls(args.n, args.seed));
    }

    let rows: Vec<Row> = map_parallel(&fixtures, build_row);

    println!(
        "\nThree-arm signature residuals, normalised by n*d_max^2 (n={}, dim={DIM}, seed={})\n",
        args.n, args.seed
    );
    println!(
        "{:<20}{:>12}{:>11}{:>11}{:>11}{:>12}{:>11}{:>11}{:>11}{:>11}{:>10}{:>11}{:>11}{:>10}{:>7}  gromov",
        "dataset",
        "truth",
        "R_sph",
        "R_euc",
        "R_hyp",
        "winner",
        "Reuc/Rhyp",
        "Reuc/Rsph",
        "r*_hyp",
        "K_hyp",
        "kappa_hyp",
        "r*_sph",
        "K_sph",
        "kappa_sph",
        "pinned",
    );
    println!("{}", "-".repeat(200));
    for r in &rows {
        println!(
            "{:<20}{:>12}{:>11.3e}{:>11.3e}{:>11.3e}{:>12}{:>11.2e}{:>11.2e}{:>11.4}{:>11.3e}{:>10.4}{:>11.4}{:>11.3e}{:>10.4}{:>7}  {}",
            r.name,
            r.truth,
            r.r_sph,
            r.r_euc,
            r.r_hyp,
            r.winner,
            r.margin_hyp,
            r.margin_sph,
            r.r_star_hyp,
            r.k_hyp,
            r.kappa_hyp,
            r.r_star_sph,
            r.k_sph,
            r.kappa_sph,
            pinned_label(r),
            if r.gromov { "hyperbolic" } else { "-" },
        );
    }

    println!(
        "\n  R_*        = Wilson residual-eigenvalue misfit of each arm, same n*d_max^2 gauge"
    );
    println!("  winner     = plain argmin over the three arms, NO margin applied");
    println!("  Reuc/Rhyp  = what the hyperbolic arm's free parameter r buys over the flat null;");
    println!("               >1 means curvature helps, ~1 means it explains nothing");
    println!("  r*_hyp/sph = fitted radius of each curved arm, in the units of the INPUT distance");
    println!("               matrix — pixel distances for MNIST, hop counts for WordNet, so not");
    println!(
        "               comparable across datasets; K = -1/r*^2 (hyp) or +1/r*^2 (sph) inherits"
    );
    println!("               those units");
    println!(
        "  kappa_*    = |K|*R_rms^2 on the reconstruction each fit implies — the EMBEDDING gauge,"
    );
    println!(
        "               the same one every t-SNE trial reports, so it can be set beside a front's"
    );
    println!("               a pinned hyperbolic arm reads exactly HYPERBOLIC_KAPPA_MIN (0.01),");
    println!("               because the search cap is a bound on this same quantity");
    println!("  R_rms_*    = gauge length behind kappa_*, in input distance units");
    println!("  pinned     = which curved arm(s) came to rest on their flat-ward window edge");
    println!("  gromov     = the CURRENT production hyperbolicity gate, for comparison");

    let agree = rows.iter().filter(|r| r.truth == r.winner).count();
    let labelled = rows.iter().filter(|r| r.truth != "?").count();
    println!("\n  argmin matches construction on {agree}/{labelled} datasets of known geometry");

    let mut out = match std::fs::File::create(jsonl) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => {
            eprintln!("error: cannot write {}: {e}", jsonl);
            std::process::exit(1);
        }
    };
    for r in &rows {
        if let Err(e) = writeln!(out, "{}", json_line(r, args.seed)) {
            eprintln!("error: cannot write {}: {e}", jsonl);
            std::process::exit(1);
        }
    }
    if let Err(e) = out.flush() {
        eprintln!("error: cannot write {}: {e}", jsonl);
        std::process::exit(1);
    }
    println!("  wrote {} rows to {}", rows.len(), jsonl);
}
