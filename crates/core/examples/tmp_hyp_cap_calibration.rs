//! Calibrating the hyperbolic upper bound.
//!
//! The `r → ∞` degeneracy says the cap must be finite; it does not say *where*.
//! This program measures the two constants that do:
//!
//! 1. **Signal.** To leading order the normalised residual is linear in the
//!    dimensionless curvature mismatch:
//!    `R_norm ≈ c · d_max²·|K_data − K_model| = c·|(d_max/ρ)² − (d_max/r)²|`.
//!    Measured two independent ways (flat data vs curved model, curved data vs
//!    flat model) — they must give the same `c`.
//! 2. **Noise floor.** Exactly flat data carrying relative distance noise `σ`
//!    has a nonzero residual even under its own correct (flat) model:
//!    `R_floor ≈ a·σ`.
//!
//! A curvature radius `ρ` is resolvable only while signal exceeds floor:
//! `c·(d_max/ρ)² ≳ a·σ`, i.e. `ρ ≲ d_max·√(c/(a·σ))`.  That is the cap.
//!
//! Verified directly by fitting with a deliberately over-wide cap.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_hyp_cap_calibration`

use fitting_core::curvature_detection::{eigenvalues_symmetric, hyperbolic_residual_at};
use fitting_core::rng::Rng;
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_hyperbolic, generate_uniform_sphere,
};

const N: usize = 200;
const SEED: u64 = 42;
const E_RADIUS: f64 = 5.0;
const DIM: usize = 2;

fn d_max_of(d: &[f64]) -> f64 {
    d.iter().cloned().fold(0.0_f64, f64::max)
}

fn d_rms_of(d: &[f64], n: usize) -> f64 {
    let s: f64 = d.iter().map(|x| x * x).sum();
    (s / (n as f64 * (n as f64 - 1.0))).sqrt()
}

/// Classical-MDS tail `L` — the residual of the *flat* model under this
/// criterion.
fn mds_tail(d: &[f64], n: usize, dim: usize) -> f64 {
    let a: Vec<f64> = d.iter().map(|x| x * x).collect();
    let rm: Vec<f64> = (0..n)
        .map(|i| a[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64)
        .collect();
    let g0 = rm.iter().sum::<f64>() / n as f64;
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] = -0.5 * (a[i * n + j] - rm[i] - rm[j] + g0);
        }
    }
    let lam = eigenvalues_symmetric(&g, n);
    lam[..n - dim].iter().map(|l| l.abs()).sum()
}

fn metric_noise(d: &[f64], n: usize, sigma: f64, rng: &mut Rng) -> Vec<f64> {
    let mut out = d.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let p = (out[i * n + j] * (1.0 + sigma * rng.normal())).max(0.0);
            out[i * n + j] = p;
            out[j * n + i] = p;
        }
    }
    out
}

fn golden(a: f64, b: f64, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64) {
    let phi = 0.618_033_988_749_894_9_f64;
    let (mut a, mut b) = (a, b);
    let mut r1 = a + (1.0 - phi) * (b - a);
    let mut r2 = a + phi * (b - a);
    let (mut f1, mut f2) = (f(r1), f(r2));
    for _ in 0..60 {
        if f1 < f2 {
            b = r2;
            r2 = r1;
            f2 = f1;
            r1 = a + (1.0 - phi) * (b - a);
            f1 = f(r1);
        } else {
            a = r1;
            r1 = r2;
            f1 = f2;
            r2 = a + phi * (b - a);
            f2 = f(r2);
        }
        if (b - a) / (a + b).max(1e-12) < 1e-6 {
            break;
        }
    }
    if f1 < f2 { (r1, f1) } else { (r2, f2) }
}

/// Coarse log grid + golden-section refine of every local minimum.  `n_grid` is
/// deliberately large here so grid resolution does not confound the
/// identifiability question this program is about.
fn minimise(lo: f64, hi: f64, n_grid: usize, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64, bool) {
    let step = (hi.ln() - lo.ln()) / (n_grid - 1) as f64;
    let gr: Vec<f64> = (0..n_grid).map(|i| (lo.ln() + i as f64 * step).exp()).collect();
    let gv: Vec<f64> = gr.iter().map(|&r| f(r)).collect();
    let mut mins: Vec<usize> = Vec::new();
    if gv[0] < gv[1] {
        mins.push(0);
    }
    for i in 1..n_grid - 1 {
        if gv[i] < gv[i - 1] && gv[i] < gv[i + 1] {
            mins.push(i);
        }
    }
    if gv[n_grid - 1] < gv[n_grid - 2] {
        mins.push(n_grid - 1);
    }
    if mins.is_empty() {
        mins.push(0);
    }
    let (mut br, mut bv, mut bi) = (gr[mins[0]], gv[mins[0]], mins[0]);
    for &i in &mins {
        let a = gr[i.saturating_sub(1)];
        let b = gr[(i + 1).min(n_grid - 1)];
        let (r, v) = if a < b { golden(a, b, f) } else { (gr[i], gv[i]) };
        if v < bv {
            bv = v;
            br = r;
            bi = i;
        }
    }
    (br, bv, bi == n_grid - 1)
}

fn main() {
    // ── 1. The signal constant c, measured two independent ways ────────────
    println!("══ 1 — signal: R_norm ≈ c·|(d_max/ρ)² − (d_max/r)²| ══");

    let e2 = generate_uniform_ball_2d(N, SEED, E_RADIUS).distances;
    let dm_e = d_max_of(&e2);
    let gauge_e = N as f64 * dm_e * dm_e;
    println!("\n  (a) FLAT data (ρ=∞), curved model at radius r:");
    println!("      {:>10}  {:>12}  {:>12}", "r/d_max", "R_norm", "c = R·(r/dm)²");
    for &q in &[1.0_f64, 2.0, 3.162, 5.109, 8.254] {
        let rn = hyperbolic_residual_at(&e2, N, DIM, q * dm_e) / gauge_e;
        println!("      {q:>10.3}  {rn:>12.4e}  {:>12.4e}", rn * q * q);
    }

    println!("\n  (b) CURVED data (ρ=1), flat model (r=∞), i.e. L:");
    println!(
        "      {:>10}  {:>12}  {:>12}",
        "d_max/ρ", "L_norm", "c = L/(dm/ρ)²"
    );
    for &mr in &[0.15_f64, 0.25, 0.35, 0.5, 0.7, 1.0] {
        let d = generate_uniform_hyperbolic(N, SEED, mr).distances;
        let dm = d_max_of(&d);
        let l = mds_tail(&d, N, DIM) / (N as f64 * dm * dm);
        println!("      {dm:>10.3}  {l:>12.4e}  {:>12.4e}", l / (dm * dm));
    }

    // ── 2. The noise floor constant a ──────────────────────────────────────
    println!("\n══ 2 — noise floor: flat data + σ, residual of its own flat model ══");
    println!("      {:>10}  {:>12}  {:>12}", "sigma", "L_norm", "a = L/σ");
    for &s in &[1e-4_f64, 3e-4, 1e-3, 3e-3, 1e-2] {
        let mut rng = Rng::new(11);
        let d = metric_noise(&e2, N, s, &mut rng);
        let l = mds_tail(&d, N, DIM) / (N as f64 * d_max_of(&d).powi(2));
        println!("      {s:>10.1e}  {l:>12.4e}  {:>12.4e}", l / s);
    }

    // ── 3. Predicted horizon vs direct measurement ─────────────────────────
    println!("\n══ 3 — horizon ρ_max/d_max = √(c/(a·σ)), vs the fit with a 20·d_max cap ══");
    println!("  (100-point grid so grid resolution is not the limiting factor)");
    for &sigma in &[0.0_f64, 1e-4, 1e-3, 1e-2] {
        println!("\n  ── σ = {:.4}% ──", sigma * 100.0);
        println!(
            "  {:>8}  {:>9}  {:>10}  {:>10}  {:>12}  {:>8}",
            "max_rho", "ρ/d_max", "r* found", "r*/ρ", "residual", "pinned"
        );
        for &mr in &[2.0_f64, 1.0, 0.7, 0.5, 0.35, 0.25, 0.15] {
            let raw = generate_uniform_hyperbolic(N, SEED, mr).distances;
            let d = if sigma > 0.0 {
                let mut rng = Rng::new(11);
                metric_noise(&raw, N, sigma, &mut rng)
            } else {
                raw
            };
            let dm = d_max_of(&d);
            let gauge = N as f64 * dm * dm;
            let mut f = |r: f64| hyperbolic_residual_at(&d, N, DIM, r);
            let (rs, val, pin) = minimise(dm / 20.0, 20.0 * dm, 100, &mut f);
            println!(
                "  {mr:>8.2}  {:>9.3}  {rs:>10.4}  {:>10.4}  {:>12.4e}  {:>8}",
                1.0 / dm,
                rs,
                val / gauge,
                pin
            );
        }
    }

    // ── 4. What the cap means in the thesis κ gauge ────────────────────────
    println!("\n══ 4 — the cap in the κ = |K|·d_rms² gauge ══");
    println!("  {:>6}  {:>9}  {:>9}  {:>12}", "data", "d_rms/dm", "at r=dm", "κ floor");
    for (name, d) in [
        ("E²", &e2),
        ("S²", &generate_uniform_sphere(N, SEED).distances),
        ("H²", &generate_uniform_hyperbolic(N, SEED, 5.0).distances),
    ] {
        let dm = d_max_of(d);
        let ratio = d_rms_of(d, N) / dm;
        println!(
            "  {name:>6}  {ratio:>9.4}  {:>9}  {:>12.4}",
            "κ=(drms/dm)²",
            ratio * ratio
        );
    }
}
