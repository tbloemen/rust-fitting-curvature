//! Did widening the hyperbolic cap lose `tree`'s minimum, or was the old
//! answer a lower-bound artefact?
//!
//! Dense sweep of the dim-2 hyperbolic residual for the `tree` fixture at
//! n = 1000 across both the old window `[d_max/20, d_max]` and the new one
//! `[d_max/20, d_rms/√κ_min]`, so the true argmin is known independently of
//! the 30-point production grid.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_tree_window_check`

use fitting_core::curvature_detection::{hyperbolic_residual_at, HYPERBOLIC_KAPPA_MIN};
use fitting_core::synthetic_data::generate_hd_tree;

const N: usize = 1000;
const SEED: u64 = 42;

fn main() {
    // Matches optimizer::Dataset::load_synthetic (hd dim = 10).
    let d = generate_hd_tree(N, 10, SEED).distances;
    let dm = d.iter().cloned().fold(0.0_f64, f64::max);
    let sum_sq: f64 = d.iter().map(|x| x * x).sum();
    let dr = (sum_sq / (N as f64 * (N as f64 - 1.0))).sqrt();
    let lo = dm / 20.0;
    let cap_old = dm;
    let cap_new = dr / HYPERBOLIC_KAPPA_MIN.sqrt();

    println!("tree, n = {N}:  d_max = {dm:.4}, d_rms = {dr:.4}");
    println!("  lower bound      = d_max/20 = {lo:.4}");
    println!("  old cap (d_max)  = {cap_old:.4}");
    println!("  new cap (10·drms)= {cap_new:.4}\n");

    // Dense log sweep across the widest window, 2000 points.
    const G: usize = 2000;
    let step = (cap_new.ln() - lo.ln()) / (G - 1) as f64;
    let mut best = (f64::INFINITY, 0.0f64);
    let mut best_old = (f64::INFINITY, 0.0f64);
    let mut rows = Vec::new();
    for i in 0..G {
        let r = (lo.ln() + i as f64 * step).exp();
        let v = hyperbolic_residual_at(&d, N, 2, r);
        if v < best.0 {
            best = (v, r);
        }
        if r <= cap_old && v < best_old.0 {
            best_old = (v, r);
        }
        rows.push((r, v));
    }

    println!("  dense argmin over OLD window [{lo:.3}, {cap_old:.3}]: r* = {:.4}  R = {:.4e}", best_old.1, best_old.0);
    println!("  dense argmin over NEW window [{lo:.3}, {cap_new:.3}]: r* = {:.4}  R = {:.4e}", best.1, best.0);
    println!(
        "\n  R at the lower bound  = {:.4e}",
        hyperbolic_residual_at(&d, N, 2, lo)
    );
    println!(
        "  R at the old cap      = {:.4e}",
        hyperbolic_residual_at(&d, N, 2, cap_old)
    );
    println!(
        "  R at the new cap      = {:.4e}",
        hyperbolic_residual_at(&d, N, 2, cap_new)
    );
    println!(
        "  R at the true ρ = 1   = {:.4e}",
        hyperbolic_residual_at(&d, N, 2, 1.0)
    );

    // Coarse profile so the shape is visible.
    println!("\n  {:>10} {:>12}", "r", "R(r)");
    for i in (0..G).step_by(G / 24) {
        println!("  {:>10.4} {:>12.4e}", rows[i].0, rows[i].1);
    }
}
