//! What should `κ_min` be — i.e. what is the weakest curvature the Wilson
//! hyperbolic fit can actually *resolve*?
//!
//! Since `κ = (d_rms/r)²`, a cap `r ≤ r_upper` is exactly a floor
//! `κ ≥ κ_min = (d_rms/r_upper)²`.  Choosing the cap is therefore choosing the
//! weakest curvature you are willing to report, and the defensible way to set
//! it is to measure where the estimator stops working.
//!
//! Two independent readings, both on EXACT H²(ρ = 1) so `κ_true = d_rms²`
//! (the extent is varied to sweep κ):
//!
//!   1. **Identifiability** — the residual contrast
//!        `C = [R(r→∞) − R(ρ)] / R(r→∞)`,
//!      the fraction of the flat-model misfit that knowing the true curvature
//!      buys you.  `C → 0` means the curved and flat models are the same model.
//!   2. **Recovery** — fit over a window 10³× wider than production's, with a
//!      grid fine enough that the search is not the binding constraint, and
//!      report `r*/ρ`.
//!
//! Both under relative distance noise `d_ij ← d_ij (1 + σ z_ij)`, z symmetric
//! standard normal.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_kappa_min`

use fitting_core::curvature_detection::hyperbolic_residual_at;
use fitting_core::synthetic_data::{generate_uniform_hyperbolic, Rng};

const N: usize = 400;
const DATA_SEED: u64 = 42;
const NOISE_SEEDS: [u64; 3] = [11, 12, 13];

fn d_max_of(d: &[f64]) -> f64 {
    d.iter().cloned().fold(0.0_f64, f64::max)
}
fn d_rms_of(d: &[f64], n: usize) -> f64 {
    let s: f64 = d.iter().map(|x| x * x).sum();
    (s / (n as f64 * (n as f64 - 1.0))).sqrt()
}

/// Symmetric multiplicative noise on the off-diagonal entries.
fn noisy(d: &[f64], n: usize, sigma: f64, seed: u64) -> Vec<f64> {
    if sigma == 0.0 {
        return d.to_vec();
    }
    let mut rng = Rng::new(seed);
    let mut out = d.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let f = 1.0 + sigma * rng.normal();
            let v = (d[i * n + j] * f).max(1e-12);
            out[i * n + j] = v;
            out[j * n + i] = v;
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
    for _ in 0..80 {
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
        if (b - a) / (a + b).max(1e-12) < 1e-10 {
            break;
        }
    }
    if f1 < f2 { (r1, f1) } else { (r2, f2) }
}

/// Fine-grid + golden refinement of every local minimum. `(r*, value, pinned_high)`.
fn fit_wide(lo: f64, hi: f64, g: usize, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64, bool) {
    let step = (hi.ln() - lo.ln()) / (g - 1) as f64;
    let gr: Vec<f64> = (0..g).map(|i| (lo.ln() + i as f64 * step).exp()).collect();
    let gv: Vec<f64> = gr.iter().map(|&r| f(r)).collect();
    let mut mins = Vec::new();
    if gv[0] < gv[1] {
        mins.push(0);
    }
    for i in 1..g - 1 {
        if gv[i] < gv[i - 1] && gv[i] < gv[i + 1] {
            mins.push(i);
        }
    }
    if gv[g - 1] < gv[g - 2] {
        mins.push(g - 1);
    }
    if mins.is_empty() {
        mins.push(g - 1);
    }
    let (mut br, mut bv, mut bi) = (gr[mins[0]], gv[mins[0]], mins[0]);
    for &i in &mins {
        let (a, b) = (gr[i.saturating_sub(1)], gr[(i + 1).min(g - 1)]);
        let (r, v) = if a < b { golden(a, b, f) } else { (gr[i], gv[i]) };
        if v < bv {
            bv = v;
            br = r;
            bi = i;
        }
    }
    (br, bv, bi == g - 1)
}

fn main() {
    // Extents chosen to sweep κ_true = d_rms² over ~4 decades.
    let extents = [0.15_f64, 0.25, 0.4, 0.6, 1.0, 1.6, 2.5, 4.0, 6.0];
    let sigmas = [0.0_f64, 1e-5, 1e-4, 1e-3, 1e-2];

    println!("══ 1 — IDENTIFIABILITY: residual contrast C = [R(∞) − R(ρ)] / R(∞) ══");
    println!("  exact H²(ρ = 1); κ_true = d_rms².  C ≈ 1 → curvature fully explains the misfit;");
    println!("  C ≈ 0 → the curved model buys nothing over flat, so κ is not identified.\n");
    print!("  {:>9} {:>9}", "κ_true", "d_rms");
    for s in sigmas {
        print!("  {:>11}", format!("σ={s:.0e}"));
    }
    println!();
    for &e in &extents {
        let base = generate_uniform_hyperbolic(N, DATA_SEED, e).distances;
        let dr = d_rms_of(&base, N);
        print!("  {:>9.4} {dr:>9.4}", dr * dr);
        for &s in &sigmas {
            let mut acc = 0.0;
            let seeds: &[u64] = if s == 0.0 { &NOISE_SEEDS[..1] } else { &NOISE_SEEDS };
            for &sd in seeds {
                let d = noisy(&base, N, s, sd);
                let dm = d_max_of(&d);
                let r_inf = 1000.0 * dm;
                let r_flat = hyperbolic_residual_at(&d, N, 2, r_inf);
                let r_true = hyperbolic_residual_at(&d, N, 2, 1.0);
                acc += (r_flat - r_true) / r_flat;
            }
            print!("  {:>11.4}", acc / seeds.len() as f64);
        }
        println!();
    }

    println!("\n══ 2 — RECOVERY: r*/ρ from a wide, finely-gridded search ══");
    println!("  window [d_rms/1000, 1000·d_max], 200 log points + golden refinement;");
    println!("  '·' = pinned at the flat end (no interior minimum at all).\n");
    print!("  {:>9} {:>9}", "κ_true", "d_rms");
    for s in sigmas {
        print!("  {:>13}", format!("σ={s:.0e}"));
    }
    println!();
    for &e in &extents {
        let base = generate_uniform_hyperbolic(N, DATA_SEED, e).distances;
        let dr = d_rms_of(&base, N);
        print!("  {:>9.4} {dr:>9.4}", dr * dr);
        for &s in &sigmas {
            let seeds: &[u64] = if s == 0.0 { &NOISE_SEEDS[..1] } else { &NOISE_SEEDS };
            let mut vals = Vec::new();
            let mut pins = 0;
            for &sd in seeds {
                let d = noisy(&base, N, s, sd);
                let dm = d_max_of(&d);
                let mut f = |r: f64| hyperbolic_residual_at(&d, N, 2, r);
                let (r, _, pin) = fit_wide(dr / 1000.0, dm * 1000.0, 200, &mut f);
                if pin {
                    pins += 1;
                } else {
                    vals.push(r);
                }
            }
            if vals.is_empty() {
                print!("  {:>13}", format!("· ({pins}P)"));
            } else {
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = vals[vals.len() / 2];
                let tag = if pins > 0 { format!("{med:.3} [{pins}P]") } else { format!("{med:.4}") };
                print!("  {tag:>13}");
            }
        }
        println!();
    }

    println!("\n══ 3 — implied caps: r_upper = d_rms/√κ_min, as a multiple of d_rms ══");
    println!("  {:>10}  {:>16}", "κ_min", "r_upper / d_rms");
    for k in [0.01_f64, 0.03, 0.1, 0.3, 1.0, 3.0] {
        println!("  {k:>10.2}  {:>16.3}", 1.0 / k.sqrt());
    }
}
