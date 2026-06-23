use crate::curvature_detection::gromov::four_distinct;
use crate::curvature_detection::gromov::median_pairwise_distance;
use crate::curvature_detection::gromov::quad_delta;

// ── Growing-ball δ(k) curve ───────────────────────────────────────────────────

/// Indices of the `k` nodes nearest to `center` (includes `center`
/// itself, which is at distance 0).
fn knn_ball(distances: &[f64], n: usize, center: usize, k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        distances[center * n + i]
            .partial_cmp(&distances[center * n + j])
            .unwrap()
    });
    idx.truncate(k);
    idx
}

/// `C(k, 4)` as `u128` (no overflow for the ball sizes we use).
fn comb4(k: usize) -> u128 {
    if k < 4 {
        return 0;
    }
    let k = k as u128;
    k * (k - 1) * (k - 2) * (k - 3) / 24
}

/// One sample of the δ(k) curve: the mean and worst-case Gromov 4-point
/// δ measured inside balls holding the `k` nearest neighbours of randomly
/// chosen centres.
#[derive(Debug, Clone, Copy)]
pub struct GromovBallPoint {
    /// Ball size — number of nodes in each k-nearest-neighbour ball.
    pub k: usize,
    /// Mean over centres of the per-ball δ (the sup over that ball's
    /// 4-point combinations).  This is the quantity that saturates.
    pub delta_mean: f64,
    /// `delta_mean` normalised by the median pairwise distance of the
    /// whole dataset — scale-free, for comparing geometries/datasets.
    pub delta_mean_normalised: f64,
}

/// The full δ(k) convergence curve plus the dataset's median distance.
#[derive(Debug, Clone)]
pub struct GromovBallCurve {
    pub points: Vec<GromovBallPoint>,
    /// Median pairwise distance used as the normaliser.
    pub median_distance: f64,
}

/// Compute the δ(k) convergence curve of the NeTS-proposal procedure.
///
/// - `ball_sizes`: the k values to probe (each clamped to `[4, n]`).
/// - `n_centers`: how many random ball centres to average over per k
///   (capped at `n`; if `>= n` every node is a centre).
/// - `max_quads_per_ball`: a ball's δ is taken exhaustively over all
///   `C(k, 4)` quadruples when that count is `≤` this bound, otherwise
///   over this many randomly sampled quadruples (the sup is then a lower
///   estimate, so use a generous bound for large balls).
/// - `seed`: makes centre/quadruple sampling reproducible.
pub fn gromov_delta_curve(
    distances: &[f64],
    n: usize,
    ball_sizes: &[usize],
    n_centers: usize,
    max_quads_per_ball: usize,
    seed: u64,
) -> GromovBallCurve {
    let median = median_pairwise_distance(distances, n);

    let mut rng = crate::rng::Rng::new(seed ^ 0x9e3779b97f4a7c15);
    let mut next = || -> usize { rng.next_raw() };

    let mut points = Vec::with_capacity(ball_sizes.len());
    for &k_raw in ball_sizes {
        let k = k_raw.clamp(4, n);

        // Pick distinct centres (or all nodes when n_centers ≥ n).
        let n_c = n_centers.min(n);
        let centers: Vec<usize> = if n_c >= n {
            (0..n).collect()
        } else {
            let mut chosen = vec![false; n];
            let mut out = Vec::with_capacity(n_c);
            while out.len() < n_c {
                let c = next() % n;
                if !chosen[c] {
                    chosen[c] = true;
                    out.push(c);
                }
            }
            out
        };

        let mut sum_delta = 0.0;
        let mut cnt = 0usize;
        for &c in &centers {
            let ball = knn_ball(distances, n, c, k);
            let bk = ball.len();
            if bk < 4 {
                continue;
            }

            let mut ball_delta = 0.0_f64;
            if comb4(bk) <= max_quads_per_ball as u128 {
                for i in 0..bk {
                    for j in (i + 1)..bk {
                        for l in (j + 1)..bk {
                            for m in (l + 1)..bk {
                                let d =
                                    quad_delta(distances, n, ball[i], ball[j], ball[l], ball[m]);
                                if d > ball_delta {
                                    ball_delta = d;
                                }
                            }
                        }
                    }
                }
            } else {
                for _ in 0..max_quads_per_ball {
                    let [p0, p1, p2, p3] = four_distinct(bk, &mut next);
                    let d = quad_delta(distances, n, ball[p0], ball[p1], ball[p2], ball[p3]);
                    if d > ball_delta {
                        ball_delta = d;
                    }
                }
            }

            sum_delta += ball_delta;
            cnt += 1;
        }

        let mean = if cnt > 0 { sum_delta / cnt as f64 } else { 0.0 };
        points.push(GromovBallPoint {
            k,
            delta_mean: mean,
            delta_mean_normalised: if median > 1e-12 { mean / median } else { 0.0 },
        });
    }

    GromovBallCurve {
        points,
        median_distance: median,
    }
}

impl GromovBallCurve {
    /// Mean of `f` over the largest-third of ball sizes — the saturated
    /// plateau of the curve, per the proposal's "δ(n) saturates" rule.
    fn tail_mean(&self, f: impl Fn(&GromovBallPoint) -> f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let len = self.points.len();
        let num_tail_points = len.div_ceil(3);
        let tail = &self.points[len - num_tail_points..];
        tail.iter().map(f).sum::<f64>() / tail.len() as f64
    }

    /// Saturated raw δ of the underlying space (tail-averaged `delta_mean`).
    pub fn saturated_delta(&self) -> f64 {
        self.tail_mean(|p| p.delta_mean)
    }

    /// Saturated normalised δ (tail-averaged `delta_mean_normalised`).
    pub fn saturated_delta_normalised(&self) -> f64 {
        self.tail_mean(|p| p.delta_mean_normalised)
    }

    /// Curvature implied by the saturated δ via `δ = ln(1+√2)/√(−K)`,
    /// i.e. `K = −(ln(1+√2)/δ)²`.  Returns `f64::NEG_INFINITY` for a
    /// degenerate (tree-like) δ → 0.
    pub fn estimated_hyperbolic_curvature(&self) -> f64 {
        let delta = self.saturated_delta();
        if delta < 1e-12 {
            return f64::NEG_INFINITY;
        }
        -((1.0 + 2.0_f64.sqrt()).ln() / delta).powi(2)
    }

    /// Least-squares slope of `ln δ(k)` against `ln k` over the upper half
    /// of the probed ball sizes.  A saturated (hyperbolic) curve has a
    /// slope near 0; a still-growing (flat/spherical) curve has a clearly
    /// positive slope, since there δ ∝ ball diameter ∝ k^(1/d).  The slope
    /// is scale-invariant, so it needs no distance normalisation.
    pub fn tail_loglog_slope(&self) -> f64 {
        let num_points = self.points.len();
        if num_points < 2 {
            return 0.0;
        }
        // Least-squares fit over the upper half of the ball sizes.
        let tail_start = num_points / 2;

        // Accumulators for the OLS slope of ln δ on ln k.
        let mut sum_ln_k = 0.0;
        let mut sum_ln_delta = 0.0;
        let mut sum_ln_k_sq = 0.0;
        let mut sum_ln_k_ln_delta = 0.0;
        let mut count = 0.0;
        for point in &self.points[tail_start..] {
            if point.k == 0 || point.delta_mean <= 0.0 {
                continue;
            }
            let ln_k = (point.k as f64).ln();
            let ln_delta = point.delta_mean.ln();
            sum_ln_k += ln_k;
            sum_ln_delta += ln_delta;
            sum_ln_k_sq += ln_k * ln_k;
            sum_ln_k_ln_delta += ln_k * ln_delta;
            count += 1.0;
        }
        if count < 2.0 {
            return 0.0;
        }
        let denominator = count * sum_ln_k_sq - sum_ln_k * sum_ln_k;
        if denominator.abs() < 1e-12 {
            return 0.0;
        }
        (count * sum_ln_k_ln_delta - sum_ln_k * sum_ln_delta) / denominator
    }
}

// ── Saturation-based hyperbolicity detection ──────────────────────────────────

/// Upper bound on the δ(k) log–log tail slope for a dataset to count as
/// δ-hyperbolic.  Empirically the saturated (hyperbolic) curve sits near
/// ~0.01 while flat/spherical curves are ~0.5–0.6, so `0.15` is a wide
/// gap between the regimes.
pub const SATURATION_SLOPE_THRESHOLD: f64 = 0.15;

const DEFAULT_BALL_CENTERS: usize = 40;
const DEFAULT_MAX_QUADS_PER_BALL: usize = 20_000;
const DEFAULT_CURVE_SEED: u64 = 7;

/// A ball-size schedule spanning the small (rising) regime up to the full
/// dataset, so the curve can reveal saturation if it occurs.
fn default_ball_sizes(n: usize) -> Vec<usize> {
    let mut sizes = vec![4, 6, 8, 10, 15, 20, 30, 40, 60, 80];
    sizes.extend([n / 8, n / 4, n / 2, (3 * n) / 4, n]);
    sizes.retain(|&s| (4..=n).contains(&s));
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// Verdict of the saturation-based hyperbolicity test.
#[derive(Debug, Clone)]
pub struct HyperbolicityVerdict {
    /// `true` when δ(k) saturates (log–log tail slope below
    /// [`SATURATION_SLOPE_THRESHOLD`]).
    pub is_hyperbolic: bool,
    /// Log–log slope of δ(k) over the upper half of ball sizes; the
    /// quantity the decision is based on.
    pub tail_slope: f64,
    /// Saturated raw δ (tail-averaged), the plateau height.
    pub saturated_delta: f64,
    /// `saturated_delta` normalised by the median pairwise distance.
    pub saturated_delta_normalised: f64,
    /// Curvature `K < 0` from `δ = ln(1+√2)/√(−K)` when hyperbolic; `0.0`
    /// otherwise (the formula assumes a negatively-curved space).
    pub curvature: f64,
    /// The full δ(k) curve, e.g. for plotting.
    pub curve: GromovBallCurve,
}

/// Decide whether a distance matrix is δ-hyperbolic via the NeTS-proposal
/// growing-ball saturation test (see the module docs).  Uses a default
/// ball-size schedule, centre count and quadruple budget; for finer
/// control build the curve directly with [`gromov_delta_curve`].
pub fn detect_hyperbolic(distances: &[f64], n: usize) -> HyperbolicityVerdict {
    let ball_sizes = default_ball_sizes(n);
    let curve = gromov_delta_curve(
        distances,
        n,
        &ball_sizes,
        DEFAULT_BALL_CENTERS,
        DEFAULT_MAX_QUADS_PER_BALL,
        DEFAULT_CURVE_SEED,
    );

    let tail_slope = curve.tail_loglog_slope();
    let is_hyperbolic = n >= 4 && tail_slope < SATURATION_SLOPE_THRESHOLD;
    let curvature = if is_hyperbolic {
        curve.estimated_hyperbolic_curvature()
    } else {
        0.0
    };

    HyperbolicityVerdict {
        is_hyperbolic,
        tail_slope,
        saturated_delta: curve.saturated_delta(),
        saturated_delta_normalised: curve.saturated_delta_normalised(),
        curvature,
        curve,
    }
}
