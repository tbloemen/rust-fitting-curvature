//! Why the radius search must be bounded *above*.
//!
//! Three experiments:
//!
//! **A — nested models.**  `Euclidean ∈ closure{hyperbolic(r)}`, so
//! `inf_r R_H(r) ≤ L` for *every* dataset, where `L = R_E` is the Euclidean
//! (classical-MDS) residual under the same criterion.  Goodness-of-fit alone
//! can therefore never prefer flat over curved.
//!
//! **B — the two curved families merge.**  `R_S(r)` and `R_H(r)` agree to
//! machine precision once `(d_max/r)²` is small, so at large `r` the criterion
//! cannot separate the two curved models from each other either.
//!
//! **C — the identifiability horizon.**  Hyperbolic data of true curvature
//! radius 1 with shrinking extent `max_rho`.  Once the true radius exceeds
//! `d_max`, does the interior minimum still exist if we widen the window?  If
//! `R_H(r_true) / L → 1` the answer is no — the information is not there, and
//! the bound costs nothing.  Repeated under 0.1% metric noise.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_upper_bound_rationale`

use fitting_core::curvature_detection::{
    eigenvalues_symmetric, hyperbolic_residual_at, spherical_residual_at,
};
use fitting_core::rng::Rng;
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_hyperbolic, generate_uniform_sphere,
};

const N: usize = 200;
const SEED: u64 = 42;
const E_RADIUS: f64 = 5.0;
const DIM: usize = 2;

fn mds_tail(d: &[f64], n: usize, dim: usize) -> f64 {
    let a: Vec<f64> = d.iter().map(|x| x * x).collect();
    let row_mean: Vec<f64> = (0..n)
        .map(|i| a[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64)
        .collect();
    let grand = row_mean.iter().sum::<f64>() / n as f64;
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] = -0.5 * (a[i * n + j] - row_mean[i] - row_mean[j] + grand);
        }
    }
    let lam = eigenvalues_symmetric(&g, n);
    lam[..n - dim].iter().map(|l| l.abs()).sum()
}

fn metric_noise(distances: &[f64], n: usize, sigma: f64, rng: &mut Rng) -> Vec<f64> {
    let mut d = distances.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let p = (d[i * n + j] * (1.0 + sigma * rng.normal())).max(0.0);
            d[i * n + j] = p;
            d[j * n + i] = p;
        }
    }
    d
}

fn d_max_of(d: &[f64]) -> f64 {
    d.iter().cloned().fold(0.0_f64, f64::max)
}

fn main() {
    let cases: [(&str, Vec<f64>); 3] = [
        ("E²", generate_uniform_ball_2d(N, SEED, E_RADIUS).distances),
        ("S²", generate_uniform_sphere(N, SEED).distances),
        ("H²", generate_uniform_hyperbolic(N, SEED, 5.0).distances),
    ];

    // ── A: nested models ───────────────────────────────────────────────────
    //
    // NB: the interior minimum of R_H on genuinely hyperbolic data is a cusp
    // narrower than 1% in r, so a log grid coarse enough to span [0.05, 500]
    // *always* straddles it and reports the flat limit as the infimum.  Do not
    // read the flat-limit columns as global minima — that is what experiment C
    // (which evaluates at the known true radius) is for.  The claim here is
    // only about the limit: R_S(∞) = R_H(∞) = L.
    println!("══ A — the r → ∞ limit of both models is the Euclidean residual L ══");
    println!("  (normalised by n·d_max²; L = Σ_{{i≤n−dim}}|λ(G)|, G the MDS Gram matrix)");
    println!(
        "\n  {:>6}  {:>13}  {:>13}  {:>13}  {:>11}",
        "data", "L (=R_E)", "R_H(200·dm)", "R_S(500·dm)", "R_H/L"
    );
    for (name, d) in &cases {
        let dm = d_max_of(d);
        let gauge = N as f64 * dm * dm;
        let l = mds_tail(d, N, DIM) / gauge;
        let rh = hyperbolic_residual_at(d, N, DIM, 200.0 * dm) / gauge;
        let rs = spherical_residual_at(d, N, DIM, 500.0 * dm) / gauge;
        println!("  {name:>6}  {l:>13.6e}  {rh:>13.6e}  {rs:>13.6e}  {:>11.6}", rh / l);
    }
    println!(
        "\n  E² is exactly 2-D Euclidean, so L = 0 and both residuals decay without bound:"
    );
    {
        let d = &cases[0].1;
        let dm = d_max_of(d);
        let gauge = N as f64 * dm * dm;
        for &ratio in &[10.0_f64, 30.0, 100.0, 200.0] {
            println!(
                "    r={:>4.0}·d_max   R_H = {:.4e}",
                ratio,
                hyperbolic_residual_at(d, N, DIM, ratio * dm) / gauge
            );
        }
    }

    // ── B: the curved families merge ───────────────────────────────────────
    println!("\n══ B — R_S(r) vs R_H(r) at large r (raw Σ|λ_res|, 14 digits) ══");
    for (name, d) in &cases {
        let dm = d_max_of(d);
        println!("\n  {name}:");
        for &ratio in &[3.0_f64, 10.0, 30.0, 100.0] {
            let r = ratio * dm;
            let rs = spherical_residual_at(d, N, DIM, r);
            let rh = hyperbolic_residual_at(d, N, DIM, r);
            println!(
                "    r={:>6.0}·d_max   R_S={rs:.14e}   R_H={rh:.14e}   rel.diff={:.3e}",
                ratio,
                ((rs - rh) / rs).abs()
            );
        }
    }

    // ── C: identifiability horizon ─────────────────────────────────────────
    // True curvature radius is 1 (unit hyperboloid); max_rho sets the extent.
    println!("\n══ C — can the true radius (=1) be found once it exceeds d_max? ══");
    for &sigma in &[0.0_f64, 0.001] {
        println!(
            "\n  ── metric noise σ = {:.1}% ──",
            sigma * 100.0
        );
        println!(
            "  {:>8}  {:>7}  {:>9}  {:>12}  {:>12}  {:>9}  {:>10}",
            "max_rho", "d_max", "r_true/dm", "R_H(r_true)", "L (=R_E)", "ratio", "verdict"
        );
        for &max_rho in &[5.0_f64, 3.0, 2.0, 1.0, 0.7, 0.5, 0.35, 0.25, 0.15] {
            let raw = generate_uniform_hyperbolic(N, SEED, max_rho).distances;
            let d = if sigma > 0.0 {
                let mut rng = Rng::new(7);
                metric_noise(&raw, N, sigma, &mut rng)
            } else {
                raw
            };
            let dm = d_max_of(&d);
            let gauge = N as f64 * dm * dm;
            let l = mds_tail(&d, N, DIM) / gauge;
            // Residual at the TRUE radius, regardless of any search window.
            let at_true = hyperbolic_residual_at(&d, N, DIM, 1.0) / gauge;
            let ratio = at_true / l;
            let verdict = if ratio < 0.1 {
                "resolvable"
            } else if ratio < 0.7 {
                "marginal"
            } else {
                "LOST"
            };
            println!(
                "  {max_rho:>8.2}  {dm:>7.3}  {:>9.3}  {at_true:>12.4e}  {l:>12.4e}  {ratio:>9.4}  {verdict:>10}",
                1.0 / dm
            );
        }
    }
}
