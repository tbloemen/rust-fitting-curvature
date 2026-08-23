//! The `r → 0` asymptotics of the hyperbolic signature residual.
//!
//! `Z^H(r) = −r² cosh(D/r) ≈ −(r²/2)·exp(D/r)` entrywise, so as `r → 0` the
//! matrix becomes a sum of wildly separated exponential scales.  The eigenvalue
//! asymptotics of such a matrix are governed by **tropical (max-plus) linear
//! algebra** (Akian–Bapat–Gaubert): with `Pₖ` the maximum weight of a
//! `k`-entry selection of `D` with distinct rows and distinct columns (an
//! optimal `k`-assignment), the eigenvalue moduli satisfy
//!
//! ```text
//!     |λₖ(r)| ≍ r² · exp(γₖ / r),      γₖ = Pₖ − Pₖ₋₁   (non-increasing)
//! ```
//!
//! so `r·ln|λₖ(r)| → γₖ`.  This program measures those rates directly and
//! compares them with `Pₖ` computed by brute force, then checks what the
//! residual `R_H` does and whether it stays resolvable against `‖Z‖`.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_small_r_hyperbolic`

use fitting_core::curvature_detection::eigenvalues_symmetric;
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_hyperbolic, generate_uniform_sphere,
};

const N: usize = 200;
const SEED: u64 = 42;
const E_RADIUS: f64 = 5.0;
const DIM: usize = 2;

/// Maximum weight of a `k`-entry selection with distinct rows and distinct
/// columns, i.e. the optimal `k`-assignment value `Pₖ`.  Exact over the `top`
/// heaviest entries, which for small `k` is safe: any optimal selection uses
/// only large entries.  Note `(a,b)` and `(b,a)` are *distinct* entries and may
/// both be used — that 2-cycle is what makes `P₂ = 2·d_max`.
fn assignment_values(d: &[f64], n: usize, kmax: usize, top: usize) -> Vec<f64> {
    let mut entries: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                entries.push((d[i * n + j], i, j));
            }
        }
    }
    entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    entries.truncate(top);

    let mut best = vec![0.0_f64; kmax + 1];
    // Depth-first over subsets of the candidate list.
    fn rec(
        entries: &[(f64, usize, usize)],
        start: usize,
        rows: &mut Vec<usize>,
        cols: &mut Vec<usize>,
        acc: f64,
        depth: usize,
        kmax: usize,
        best: &mut Vec<f64>,
    ) {
        if acc > best[depth] {
            best[depth] = acc;
        }
        if depth == kmax {
            return;
        }
        for t in start..entries.len() {
            let (w, i, j) = entries[t];
            // Optimistic bound: even taking the heaviest remaining entries
            // cannot beat the incumbent.
            if acc + w * (kmax - depth) as f64 <= best[kmax] && depth > 0 {
                break;
            }
            if rows.contains(&i) || cols.contains(&j) {
                continue;
            }
            rows.push(i);
            cols.push(j);
            rec(entries, t + 1, rows, cols, acc + w, depth + 1, kmax, best);
            rows.pop();
            cols.pop();
        }
    }
    rec(
        &entries,
        0,
        &mut Vec::new(),
        &mut Vec::new(),
        0.0,
        0,
        kmax,
        &mut best,
    );
    best
}

/// Golden-section minimisation, mirroring `signature.rs`.
fn golden(a: f64, b: f64, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64) {
    let phi = 0.618_033_988_749_894_9_f64;
    let (mut a, mut b) = (a, b);
    let mut r1 = a + (1.0 - phi) * (b - a);
    let mut r2 = a + phi * (b - a);
    let (mut f1, mut f2) = (f(r1), f(r2));
    for _ in 0..40 {
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
        if (b - a) / (a + b).max(1e-12) < 1e-5 {
            break;
        }
    }
    if f1 < f2 { (r1, f1) } else { (r2, f2) }
}

/// `minimise_log_spaced` from `signature.rs`: coarse log grid, then golden-section
/// refine *every* local minimum.  Returns `(r*, f(r*))`.
fn mini_minimise(lo: f64, hi: f64, n_grid: usize, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64) {
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

    let (mut br, mut bv) = (gr[mins[0]], gv[mins[0]]);
    for &i in &mins {
        let a = gr[i.saturating_sub(1)];
        let b = gr[(i + 1).min(n_grid - 1)];
        let (r, v) = if a < b { golden(a, b, f) } else { (gr[i], gv[i]) };
        if v < bv {
            bv = v;
            br = r;
        }
    }
    (br, bv)
}

fn main() {
    let cases: [(&str, Vec<f64>); 3] = [
        ("E²", generate_uniform_ball_2d(N, SEED, E_RADIUS).distances),
        ("S²", generate_uniform_sphere(N, SEED).distances),
        ("H²", generate_uniform_hyperbolic(N, SEED, 5.0).distances),
    ];

    for (name, d) in &cases {
        let dm = d.iter().cloned().fold(0.0_f64, f64::max);
        println!("\n══ {name}  (n={N}, d_max={dm:.4}) ══");

        // Tropical prediction.
        let p = assignment_values(d, N, 6, 60);
        print!("  Pₖ (optimal k-assignment):");
        for k in 1..=6 {
            print!("  P{k}={:.3}", p[k]);
        }
        println!();
        print!("  γₖ = Pₖ − Pₖ₋₁ (predicted rates):");
        for k in 1..=6 {
            print!("  γ{k}={:.4}", p[k] - p[k - 1]);
        }
        println!();
        println!("  d_max = {dm:.4}   (γ₁ must equal d_max)");

        println!(
            "\n  {:>8}  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>9}  {:>10}  {:>9}",
            "d_max/r", "r·ln|λ1|", "|λ2|", "|λ3|", "|λ4|", "|λ5|", "|λ6|", "r·lnR_H", "R_H/‖Z‖", "signs"
        );
        for &x in &[5.0_f64, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0] {
            let r = dm / x;
            let mut z = vec![0.0; N * N];
            for i in 0..N {
                for j in 0..N {
                    z[i * N + j] = -r * r * (d[i * N + j] / r).cosh();
                }
            }
            let lam = eigenvalues_symmetric(&z, N); // ascending
            // Residual: drop the most negative and the DIM most positive.
            let resid: f64 = lam[1..N - DIM].iter().map(|v| v.abs()).sum();
            let norm = lam[0].abs().max(lam[N - 1].abs());

            let mut by_mag: Vec<f64> = lam.clone();
            by_mag.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
            let signs: String = by_mag[..4]
                .iter()
                .map(|v| if *v < 0.0 { '-' } else { '+' })
                .collect();

            print!("  {x:>8.0}");
            for k in 0..6 {
                print!("  {:>8.4}", r * by_mag[k].abs().ln());
            }
            println!(
                "  {:>9.4}  {:>10.3e}  {:>9}",
                r * resid.ln(),
                resid / norm,
                signs
            );
        }
    }

    // ── D: what does *lowering* the floor cost? ────────────────────────────
    //
    // The objective diverges as r → 0, so a lower bound is not needed to stop a
    // degenerate optimum.  What it buys is grid resolution: `minimise_log_spaced`
    // spends a fixed 30 points on [lo, hi], and the true-radius minimum is a cusp
    // narrower than 1% in r.  Widen the window and the cusp stops registering as
    // a local minimum on the coarse grid, so the refinement never targets it.
    println!("\n══ D — recovered r* on H² (true radius = 1) vs the search floor ══");
    let d = &cases[2].1;
    let dm = d.iter().cloned().fold(0.0_f64, f64::max);
    let gauge = N as f64 * dm * dm;
    println!(
        "  d_max = {dm:.4}, so the true radius sits at r/d_max = {:.4}",
        1.0 / dm
    );
    println!(
        "\n  {:>12}  {:>10}  {:>9}  {:>13}  {:>9}",
        "floor", "step size", "r*", "residual", "verdict"
    );
    for &k in &[20.0_f64, 50.0, 100.0, 1000.0, 10000.0] {
        let lo = dm / k;
        let mut f = |r: f64| -> f64 {
            let mut z = vec![0.0; N * N];
            for i in 0..N {
                for j in 0..N {
                    z[i * N + j] = -r * r * (d[i * N + j] / r).cosh();
                }
            }
            let lam = eigenvalues_symmetric(&z, N);
            lam[1..N - DIM].iter().map(|v| v.abs()).sum()
        };
        let (r_star, val) = mini_minimise(lo, dm, 30, &mut f);
        let step = k.powf(1.0 / 29.0);
        let ok = if (r_star - 1.0).abs() < 0.05 {
            "FOUND"
        } else {
            "MISSED"
        };
        println!(
            "  d_max/{:<7.0}  {:>9.1}%  {r_star:>9.4}  {:>13.4e}  {ok:>9}",
            k,
            (step - 1.0) * 100.0,
            val / gauge
        );
    }
}
