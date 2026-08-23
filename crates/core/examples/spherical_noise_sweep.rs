//! How much noise can spherical data carry before the Wilson signature
//! criterion stops recognising it?
//!
//! [`SPHERICAL_RESIDUAL_MAX`] is calibrated on an *exact* `S²` fixture, which
//! scores ~`1e-8` — so far below the threshold that the calibration says
//! nothing about how much slack real data has.  The question that matters is
//! not where the threshold sits but where the **ordering** breaks: a noisy
//! sphere is only detectable while it still scores better than flat data does
//! under the same spherical model.  Once it scores worse than the Euclidean
//! fixture, no threshold can separate them and tuning the constant is futile.
//!
//! Three models, because they corrupt different things:
//!
//!   - **metric** — multiplicative jitter on the true geodesic distance
//!     matrix, `d'ᵢⱼ = dᵢⱼ·(1 + σ·ε)`, `ε ~ N(0,1)`, symmetric.  The points
//!     never move; only the numbers do.  Models measurement error in a
//!     distance matrix you believe is geodesic.  Note the result is generally
//!     not a metric at all — the triangle inequality can break, and no point
//!     configuration in any space realises it.  It is also an unstructured
//!     full-rank perturbation (`n²/2` independent draws), which is why it
//!     costs more per unit σ than the structured `3n`-draw ambient model.
//!   - **ambient** — Gaussian displacement of the points off the sphere in
//!     `R³` (`σ` in units of the radius), then plain Euclidean distances
//!     between the displaced points.  Models data lying *near* a sphere in a
//!     feature space, the realistic case.  Its `σ = 0` row is not free:
//!     `‖xᵢ − xⱼ‖` is the **chord** through the ball, not the geodesic over
//!     the surface, and Wilson's `Z = r²cos(d/r)` expects geodesics.
//!   - **ambient, geodesic-corrected** — the same displaced points with
//!     `chord → geodesic` undone at the *known* radius.  Separates the two
//!     effects the ambient model conflates.  It recovers the exact fixture's
//!     score at `σ = 0`, which is what identifies the chord as a model
//!     mismatch rather than a noise level.
//!
//! Caveat on comparing columns: σ means a fraction of each *distance* in the
//! metric model and a fraction of the *radius* in the ambient ones.  They are
//! the same order of magnitude on a unit sphere but not the same axis.
//!
//! Each row averages over [`SEEDS`] noise realisations and reports the
//! normalised residual against two reference levels measured in the same run:
//! the detection threshold, and the flat `E²` fixture's score (the level at
//! which the ordering inverts).
//!
//! Run: `cargo run --release -p fitting-core --example spherical_noise_sweep`

use fitting_core::curvature_detection::{detect_geometry, fit_spherical, SPHERICAL_RESIDUAL_MAX};
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::rng::Rng;
use fitting_core::synthetic_data::{generate_uniform_ball_2d, generate_uniform_sphere};

const N: usize = 300;
const DIM: usize = 2;
/// Noise realisations averaged per (model, σ) cell.
const SEEDS: u64 = 3;
const BASE_SEED: u64 = 42;
/// Radius of the Euclidean reference fixture (matches the other examples).
const E_RADIUS: f64 = 5.0;

/// Noise levels, as a fraction of the distance (metric) or of the sphere's
/// radius (ambient).
const SIGMAS: [f64; 15] = [
    0.0, 0.00025, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.04, 0.08, 0.16,
    0.32,
];

/// Candidate values for [`SPHERICAL_RESIDUAL_MAX`], for the tolerance table.
/// `1e-2` is the largest that stays under both the nearest interior impostor
/// (pbmc, `2.2e-2`) and the flat `E²` fixture (`1.4e-2`); `2e-2` clears only
/// the former, so it relies on `at_upper_bound` to reject flat data.
const CANDIDATE_THRESHOLDS: [f64; 4] = [1e-3, 5e-3, 1e-2, 2e-2];

// ── Noise models ────────────────────────────────────────────────────────────

/// Multiplicative jitter on a distance matrix.  Drawn once per unordered pair
/// so the result stays symmetric, and floored at 0 since a negative distance
/// is not a distance.
fn metric_noise(distances: &[f64], n: usize, sigma: f64, rng: &mut Rng) -> Vec<f64> {
    let mut d = distances.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let perturbed = (d[i * n + j] * (1.0 + sigma * rng.normal())).max(0.0);
            d[i * n + j] = perturbed;
            d[j * n + i] = perturbed;
        }
    }
    d
}

/// Displace each point off the unit sphere by `N(0, σ²)` per ambient
/// coordinate, then take Euclidean distances between the displaced points.
///
/// Note what this changes *besides* moving the points: `‖xᵢ − xⱼ‖` is the
/// **chord** through the ball, not the geodesic over the surface.  That swap
/// is deterministic and is already in force at `σ = 0`, which is why the
/// σ = 0 row of this model is not a free reference point.
fn ambient_noise(x: &[f64], n: usize, ambient_dim: usize, sigma: f64, rng: &mut Rng) -> Vec<f64> {
    let mut perturbed = x.to_vec();
    for v in perturbed.iter_mut() {
        *v += sigma * rng.normal();
    }
    compute_euclidean_distance_matrix(&perturbed, n, ambient_dim)
}

/// [`ambient_noise`] with the chord mapped back to the geodesic it came from.
/// On a radius-`r` sphere `chord = 2r·sin(geodesic/2r)`, so the inverse is
/// `geodesic = 2r·asin(chord/2r)`; here `r = 1`, the radius the fixture was
/// generated at.  Isolating this one correction separates the two effects the
/// ambient model conflates — metric mismatch versus off-manifold displacement.
/// Displaced points can sit outside the sphere, making `chord/2 > 1`, so the
/// argument is clamped; that clipping is itself part of the distortion at
/// large σ.
fn ambient_noise_geodesic(
    x: &[f64],
    n: usize,
    ambient_dim: usize,
    sigma: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let mut d = ambient_noise(x, n, ambient_dim, sigma, rng);
    for v in d.iter_mut() {
        *v = 2.0 * (*v / 2.0).clamp(-1.0, 1.0).asin();
    }
    d
}

// ── One (model, σ) cell ─────────────────────────────────────────────────────

struct Cell {
    /// Mean normalised residual of the spherical fit over the seeds.
    residual: f64,
    /// Mean `r*/d_max`.
    r_over_dmax: f64,
    /// How many seeds `detect_geometry` still called spherical.
    n_spherical: u64,
    /// How many seeds the spherical fit landed pinned at the flat-ward edge.
    n_pinned: u64,
    /// Per-seed `(residual_normalised, at_upper_bound)`, kept so the spherical
    /// gate can be replayed against thresholds other than the compiled-in one.
    per_seed: Vec<(f64, bool)>,
}

impl Cell {
    /// How many seeds `detect_geometry` would call spherical if
    /// [`SPHERICAL_RESIDUAL_MAX`] were `threshold`.  The spherical branch is
    /// tested first and is exactly `residual < MAX && !at_upper_bound`, so
    /// replaying that predicate reproduces the verdict without recompiling.
    fn n_spherical_at(&self, threshold: f64) -> usize {
        self.per_seed
            .iter()
            .filter(|(res, pinned)| *res < threshold && !pinned)
            .count()
    }
}

/// Which corruption to apply to the exact `S²` fixture.
#[derive(Clone, Copy, PartialEq)]
enum Model {
    /// Jitter the geodesic distance matrix; points stay on the sphere.
    Metric,
    /// Move the points off the sphere, measure chords.
    Ambient,
    /// As [`Model::Ambient`], with chords converted back to geodesics.
    AmbientGeodesic,
}

fn sweep_cell(sigma: f64, model: Model) -> Cell {
    let mut residual = 0.0;
    let mut r_over_dmax = 0.0;
    let mut n_spherical = 0;
    let mut n_pinned = 0;
    let mut per_seed = Vec::with_capacity(SEEDS as usize);

    for s in 0..SEEDS {
        let data = generate_uniform_sphere(N, BASE_SEED + s);
        // Noise stream kept separate from the point-generation stream so the
        // underlying point set is identical across σ within one seed.
        let mut rng = Rng::new(BASE_SEED.wrapping_mul(1000).wrapping_add(s));
        let d = match model {
            Model::Metric => metric_noise(&data.distances, N, sigma, &mut rng),
            Model::Ambient => ambient_noise(&data.x, N, data.ambient_dim, sigma, &mut rng),
            Model::AmbientGeodesic => {
                ambient_noise_geodesic(&data.x, N, data.ambient_dim, sigma, &mut rng)
            }
        };

        let d_max = d.iter().cloned().fold(0.0_f64, f64::max);
        let fit = fit_spherical(&d, N, DIM);
        residual += fit.residual_normalised;
        r_over_dmax += fit.radius / d_max;
        per_seed.push((fit.residual_normalised, fit.at_upper_bound));
        if fit.at_upper_bound {
            n_pinned += 1;
        }
        if detect_geometry(&d, N, DIM).best_geometry == "spherical" {
            n_spherical += 1;
        }
    }

    let k = SEEDS as f64;
    Cell {
        residual: residual / k,
        r_over_dmax: r_over_dmax / k,
        n_spherical,
        n_pinned,
        per_seed,
    }
}

/// The flat fixture's score under the *spherical* model — the level at which a
/// noisy sphere stops being distinguishable from flat data.
fn euclidean_reference() -> f64 {
    let mut total = 0.0;
    for s in 0..SEEDS {
        let data = generate_uniform_ball_2d(N, BASE_SEED + s, E_RADIUS);
        total += fit_spherical(&data.distances, N, DIM).residual_normalised;
    }
    total / SEEDS as f64
}

fn report(label: &str, model: Model, e_ref: f64) -> Vec<Cell> {
    println!("\n── {label} ──");
    println!(
        "     sigma   residual e/(n dmax^2)   vs threshold   vs E2 ref   r*/d_max   pinned   spherical"
    );

    let mut cells = Vec::with_capacity(SIGMAS.len());
    for sigma in SIGMAS {
        let c = sweep_cell(sigma, model);
        println!(
            "  {:8.5}   {:19.3e}   {:11.2}x   {:8.2}x   {:8.4}   {:2}/{}      {}/{}",
            sigma,
            c.residual,
            c.residual / SPHERICAL_RESIDUAL_MAX,
            c.residual / e_ref,
            c.r_over_dmax,
            c.n_pinned,
            SEEDS,
            c.n_spherical,
            SEEDS,
        );
        cells.push(c);
    }
    cells
}

/// Where the gate stops firing, as a σ rather than as a grid index.
///
/// Reporting the last *sampled* σ that passes would understate the answer by
/// however wide the grid happens to be — at `T = 1e-2` the metric model passes
/// at σ = 0.003 and fails at 0.005, so the grid says 0.3% while the crossing is
/// really near 0.48%.  The residual is smooth and monotone in σ, so the two
/// bracketing cells are interpolated instead.  Linear interpolation is
/// appropriate because the residual is close to proportional to σ (slope ≈2.1
/// for the metric model, ≈0.21 for the geodesic-corrected ambient one), and it
/// needs no extra fits.
///
/// `None` means the gate never fires, even at σ = 0.
fn crossing(cells: &[Cell], threshold: f64) -> Option<f64> {
    // Fits pinned at the flat-ward bound are rejected by `at_upper_bound`
    // whatever the residual does, so no threshold rescues them.
    if cells[0].n_spherical_at(threshold) == 0 {
        return None;
    }
    for i in 1..cells.len() {
        if cells[i].residual >= threshold {
            let (s0, r0) = (SIGMAS[i - 1], cells[i - 1].residual);
            let (s1, r1) = (SIGMAS[i], cells[i].residual);
            return Some(s0 + (threshold - r0) * (s1 - s0) / (r1 - r0));
        }
    }
    // Still passing at the widest σ sampled.
    Some(f64::INFINITY)
}

fn main() {
    let e_ref = euclidean_reference();
    println!("n = {N}, dim = {DIM}, {SEEDS} noise realisations per cell");
    println!("detection threshold SPHERICAL_RESIDUAL_MAX = {SPHERICAL_RESIDUAL_MAX:.0e}");
    println!(
        "flat E² fixture under the spherical model  = {e_ref:.3e}  (ordering-inversion level)"
    );

    let metric = report(
        "metric noise: points stay on S^2, geodesic matrix jittered, d' = d(1 + sigma*N(0,1))",
        Model::Metric,
        e_ref,
    );
    let ambient = report(
        "ambient noise: x' = x + sigma*N(0,1) in R^3, CHORDAL distances ||x'_i - x'_j||",
        Model::Ambient,
        e_ref,
    );
    let ambient_geo = report(
        "ambient noise, chords converted back to geodesics: d = 2*asin(chord/2)",
        Model::AmbientGeodesic,
        e_ref,
    );

    let fmt = |v: Option<f64>| match v {
        Some(s) if s.is_infinite() => format!("> {:.0}%", 100.0 * SIGMAS[SIGMAS.len() - 1]),
        Some(s) => format!("{s:.5} ({:.3}%)", 100.0 * s),
        None => "fails at sigma = 0".to_string(),
    };

    println!("\n\nNoise tolerance vs threshold  —  sigma at which the gate");
    println!("`residual < T && !at_upper_bound` stops firing, interpolated between the");
    println!("bracketing grid points rather than rounded down to the last one that passed.");
    println!(
        "     T      metric                      ambient (chordal)           ambient (geodesic)"
    );
    for t in CANDIDATE_THRESHOLDS {
        println!(
            "  {:6.0e}   {:26}  {:26}  {}",
            t,
            fmt(crossing(&metric, t)),
            fmt(crossing(&ambient, t)),
            fmt(crossing(&ambient_geo, t)),
        );
    }
    println!(
        "\nRaising T buys tolerance linearly (the residual is linear in sigma, slope ~2,\n\
         so sigma_max ~ T/2) and buys nothing on chordal distances, which miss at\n\
         sigma = 0 by more than any usable T. Comparing the last two columns isolates\n\
         why: the same displaced points become detectable again once the chord is\n\
         mapped back to the geodesic, so the failure is a metric mismatch in the model,\n\
         not the displacement. The ceiling on T is the nearest *interior* impostor\n\
         (pbmc, 2.2e-2); the flat E^2 fixture is rejected by at_upper_bound instead, so\n\
         it binds T only if that test fails."
    );
}
