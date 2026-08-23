//! What the Wilson signature residual does as `r → 0` and `r → ∞`.
//!
//! Scans the raw objective `Σ|λ_res|` far outside the search windows of
//! `fit_spherical` / `fit_hyperbolic`, to check the two limits that motivate
//! those windows:
//!
//! - `r → 0`: the spherical residual collapses like `r²` for *any* data
//!   (`|Z^S_ij| = r²|cos(d/r)| ≤ r²`), a degenerate zero of the objective.
//!   The hyperbolic residual instead blows up like `r²cosh(d_max/r)`.
//! - `r → ∞`: both kernels reduce to `±r²J − ½D∘D`, whose non-divergent
//!   spectrum is that of the classical-MDS Gram matrix `G = −½ P (D∘D) P`.
//!   Both residuals therefore converge to the *same* constant
//!   `L = Σ_{i ≤ n−dim} |λ_i(G)|` — the Torgerson misfit of a `dim`-dimensional
//!   Euclidean fit — approached at rate `O(1/r²)`.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_residual_asymptotics`

use fitting_core::curvature_detection::{
    eigenvalues_symmetric, fit_hyperbolic, fit_spherical, hyperbolic_residual_at,
    spherical_residual_at,
};
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_hyperbolic, generate_uniform_sphere,
};

const N: usize = 200;
const SEED: u64 = 42;
const E_RADIUS: f64 = 5.0;
const H_MAX_RHO: f64 = 5.0;
const DIM: usize = 2;

/// Classical-MDS (Torgerson) Gram matrix `G = −½ P (D∘D) P`, by double
/// centring the squared-distance matrix.
fn mds_gram(d: &[f64], n: usize) -> Vec<f64> {
    let a: Vec<f64> = d.iter().map(|x| x * x).collect();
    let mut row_mean = vec![0.0; n];
    for i in 0..n {
        row_mean[i] = a[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64;
    }
    let grand = row_mean.iter().sum::<f64>() / n as f64;
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] = -0.5 * (a[i * n + j] - row_mean[i] - row_mean[j] + grand);
        }
    }
    g
}

/// `L = Σ|λ| over all but the `dim` largest eigenvalues of `G`.  The extra
/// zero eigenvalue `G1 = 0` contributes nothing, so this is exactly the
/// `r → ∞` limit of both residuals.
fn mds_tail(d: &[f64], n: usize, dim: usize) -> f64 {
    let lam = eigenvalues_symmetric(&mds_gram(d, n), n);
    lam[..n - dim].iter().map(|l| l.abs()).sum()
}

fn main() {
    let cases: [(&str, Vec<f64>); 3] = [
        ("E²", generate_uniform_ball_2d(N, SEED, E_RADIUS).distances),
        ("S²", generate_uniform_sphere(N, SEED).distances),
        (
            "H²",
            generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances,
        ),
    ];

    for (name, d) in &cases {
        let d_max = d.iter().cloned().fold(0.0_f64, f64::max);
        let gauge = N as f64 * d_max * d_max;
        let limit = mds_tail(d, N, DIM);

        println!("\n══ {name}  (n={N}, d_max={d_max:.4}) ══");
        println!("  classical-MDS tail L = Σ_{{i≤n−dim}}|λ(G)| = {limit:.6e}   L/(n·d_max²) = {:.4e}", limit / gauge);
        println!(
            "  search windows: spherical r/d_max ∈ [{:.3}, {:.3}], hyperbolic ∈ [{:.3}, {:.3}]",
            1.0 / std::f64::consts::PI,
            1.0 / 2.5,
            1.0 / 20.0,
            1.0
        );
        let fs = fit_spherical(d, N, DIM);
        let fh = fit_hyperbolic(d, N, DIM);
        println!(
            "  fitter lands at: sph r*={:.4} ({:.4e}, pin={})   hyp r*={:.4} ({:.4e}, pin={})",
            fs.radius, fs.residual_normalised, fs.at_upper_bound, fh.radius,
            fh.residual_normalised, fh.at_upper_bound,
        );
        println!();
        println!(
            "  {:>10}  {:>9}  {:>12} {:>11}  {:>12} {:>11}",
            "r/d_max", "d_max/r", "R_S/(n dm²)", "R_S/(n r²)", "R_H/(n dm²)", "R_H/L"
        );

        // Log grid in r/d_max, wide enough to see both limits.
        let n_grid = 25;
        let (lo, hi) = (1e-2_f64, 1e3_f64);
        for i in 0..n_grid {
            let t = i as f64 / (n_grid - 1) as f64;
            let ratio = lo * (hi / lo).powf(t);
            let r = ratio * d_max;
            let x = 1.0 / ratio; // d_max / r

            let rs = spherical_residual_at(d, N, DIM, r);
            // cosh overflows past d_max/r ≈ 700; stay well clear.
            let rh = if x <= 200.0 {
                hyperbolic_residual_at(d, N, DIM, r)
            } else {
                f64::NAN
            };

            let rs_r2 = rs / (N as f64 * r * r);
            print!(
                "  {ratio:>10.3e}  {x:>9.2}  {:>12.4e} {:>11.4e}",
                rs / gauge,
                rs_r2
            );
            if rh.is_nan() {
                println!("  {:>12} {:>11}", "overflow", "-");
            } else {
                println!("  {:>12.4e} {:>11.4e}", rh / gauge, rh / limit);
            }
        }

        // The interior minimum is *narrow* — the coarse decade grid above
        // straddles it.  Rescan at the fitter's own resolution (30 points
        // over the hyperbolic window [d_max/20, d_max]) to see its depth.
        println!("\n  ── fine scan, hyperbolic search window ──");
        println!("  {:>10}  {:>12}", "r/d_max", "R_H/(n dm²)");
        for i in 0..30 {
            let t = i as f64 / 29.0;
            let ratio = (1.0 / 20.0_f64) * (20.0_f64).powf(t);
            let r = ratio * d_max;
            let rh = hyperbolic_residual_at(d, N, DIM, r);
            println!("  {ratio:>10.4}  {:>12.4e}", rh / gauge);
        }
    }
}
