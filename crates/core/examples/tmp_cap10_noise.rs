//! What happens if the hyperbolic cap is raised to `10·d_max`?
//!
//! Fixes the curvature-of-interest first (the user's framing): we are willing to
//! call `r* = 10·d_max` hyperbolic, so the search window becomes
//! `[d_max/20, 10·d_max]`.  Three questions:
//!
//! 1. **Recovery.** For hyperbolic data of true radius `ρ = 1` placed at various
//!    `ρ/d_max`, does the fit return `ρ` under noise `σ`?  Run at the production
//!    grid (30 points) and at a fine grid (100) to separate the grid cost of the
//!    wider window from the noise cost.
//! 2. **False positives.** Exactly Euclidean data + noise: does the wider window
//!    let a *spurious* interior minimum appear, so a finite curvature is reported
//!    for flat data?  With the cap at `d_max` such data pins, which is the
//!    correct "unresolved" signal.
//! 3. **Precision.** `κ ∝ 1/r*²`, so a relative error `e` in `r*` is `2e` in κ.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_cap10_noise`

use fitting_core::curvature_detection::hyperbolic_residual_at;
use fitting_core::rng::Rng;
use fitting_core::synthetic_data::{generate_uniform_ball_2d, generate_uniform_hyperbolic};

const N: usize = 200;
const SEED: u64 = 42;
const E_RADIUS: f64 = 5.0;
const DIM: usize = 2;
const SEEDS: u64 = 3;

fn d_max_of(d: &[f64]) -> f64 {
    d.iter().cloned().fold(0.0_f64, f64::max)
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
    for _ in 0..50 {
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

/// `minimise_log_spaced` semantics: coarse log grid, golden-refine every local
/// minimum, keep the best.  Returns `(r*, value, pinned_at_upper)`.
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

/// max_rho values chosen so the true radius (=1) sits at the listed ρ/d_max.
const CASES: [(f64, f64); 9] = [
    (5.0, 0.10),
    (3.0, 0.17),
    (2.0, 0.25),
    (1.0, 0.5),
    (0.5, 1.0),
    (0.25, 2.0),
    (0.125, 4.0),
    (0.07, 7.0),
    (0.05, 10.0),
];

const SIGMAS: [f64; 5] = [0.0, 1e-5, 1e-4, 3e-4, 1e-3];

fn recovery_table(n_grid: usize, cap_mult: f64) {
    println!(
        "\n  ── grid = {n_grid} points over [d_max/20, {cap_mult:.0}·d_max] ──"
    );
    print!("  {:>10}", "ρ/d_max");
    for s in SIGMAS {
        print!("  {:>16}", format!("σ={:.0e}", s));
    }
    println!("      (cells: median r*/ρ, [min..max] over seeds; P = pinned)");

    for (max_rho, _target) in CASES {
        let raw = generate_uniform_hyperbolic(N, SEED, max_rho).distances;
        let dm0 = d_max_of(&raw);
        print!("  {:>10.2}", 1.0 / dm0);
        for sigma in SIGMAS {
            let mut ratios: Vec<f64> = Vec::new();
            let mut pins = 0;
            let reps = if sigma == 0.0 { 1 } else { SEEDS };
            for s in 0..reps {
                let d = if sigma > 0.0 {
                    let mut rng = Rng::new(1000 + s);
                    metric_noise(&raw, N, sigma, &mut rng)
                } else {
                    raw.clone()
                };
                let dm = d_max_of(&d);
                let mut f = |r: f64| hyperbolic_residual_at(&d, N, DIM, r);
                let (rs, _, pin) = minimise(dm / 20.0, cap_mult * dm, n_grid, &mut f);
                ratios.push(rs); // true ρ = 1
                if pin {
                    pins += 1;
                }
            }
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ratios[ratios.len() / 2];
            let cell = if pins > 0 {
                format!("{med:.2} [{}P]", pins)
            } else {
                format!("{med:.3} [{:.2}..{:.2}]", ratios[0], ratios[ratios.len() - 1])
            };
            print!("  {cell:>16}");
        }
        println!();
    }
}

fn main() {
    println!("══ 1 — recovery of the true radius (ρ = 1) with the cap at 10·d_max ══");
    recovery_table(30, 10.0);
    recovery_table(100, 10.0);

    println!("\n══ 2 — same data, current cap of 1·d_max, for comparison ══");
    recovery_table(30, 1.0);

    println!("\n══ 3 — FALSE POSITIVES: exactly Euclidean data + noise ══");
    println!("  (pinned = correct 'unresolved' signal; interior r* = spurious curvature)");
    let e2 = generate_uniform_ball_2d(N, SEED, E_RADIUS).distances;
    for &cap in &[1.0_f64, 10.0] {
        println!("\n  ── cap = {cap:.0}·d_max, grid = 30 ──");
        println!(
            "  {:>10}  {:>12}  {:>28}  {:>10}",
            "sigma", "pinned/seeds", "interior r*/d_max (if any)", "implied κ"
        );
        for sigma in SIGMAS {
            let reps = if sigma == 0.0 { 1 } else { SEEDS };
            let mut pins = 0;
            let mut interiors: Vec<f64> = Vec::new();
            for s in 0..reps {
                let d = if sigma > 0.0 {
                    let mut rng = Rng::new(2000 + s);
                    metric_noise(&e2, N, sigma, &mut rng)
                } else {
                    e2.clone()
                };
                let dm = d_max_of(&d);
                let mut f = |r: f64| hyperbolic_residual_at(&d, N, DIM, r);
                let (rs, _, pin) = minimise(dm / 20.0, cap * dm, 30, &mut f);
                if pin {
                    pins += 1;
                } else {
                    interiors.push(rs / dm);
                }
            }
            let desc = if interiors.is_empty() {
                "—".to_string()
            } else {
                format!(
                    "{:.3} .. {:.3}",
                    interiors.iter().cloned().fold(f64::INFINITY, f64::min),
                    interiors.iter().cloned().fold(0.0_f64, f64::max)
                )
            };
            // κ implied by the smallest interior r*, using d_rms/d_max ≈ 0.505.
            let kap = if interiors.is_empty() {
                0.0
            } else {
                let rmin = interiors.iter().cloned().fold(f64::INFINITY, f64::min);
                (0.505 / rmin).powi(2)
            };
            println!(
                "  {sigma:>10.0e}  {:>12}  {desc:>28}  {:>10.4}",
                format!("{pins}/{reps}"),
                kap
            );
        }
    }
}
