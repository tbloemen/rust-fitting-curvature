//! The handful of statistics the analysis needs, ported from scipy.
//!
//! Spearman's ρ with its t-approximation p-value (`scipy.stats.spearmanr`), the
//! Wilcoxon signed-rank test (`scipy.stats.wilcoxon`), the Friedman test
//! (`scipy.stats.friedmanchisquare`) with Holm's step-down correction for the
//! post-hoc comparisons against a control, and medians. Each is reproduced to
//! match scipy's default options.
//!
//! The Friedman + Holm pair is the procedure Demšar (2006) §3.2.2 recommends for
//! comparing several methods over several datasets, and is what `aggregate.rs`
//! uses on the ΔR2 table.

/// Mean of a slice; `None` when empty.
pub fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().sum::<f64>() / xs.len() as f64)
}

/// Median of a slice (average of the two middle values for even length).
/// `None` when empty. Non-finite values are not filtered — callers do that.
pub fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    })
}

/// Fractional ranks with ties averaged (scipy's `rankdata(method="average")`).
pub fn rankdata(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        // Extend over the run of equal values and give them all the mean rank.
        let mut j = i + 1;
        while j < n && xs[idx[j]] == xs[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0; // mean of ranks i+1 .. j (1-based)
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation of two equal-length slices; `None` if either is constant.
pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n < 2 || y.len() != n {
        return None;
    }
    let mx = mean(x)?;
    let my = mean(y)?;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return None;
    }
    Some(sxy / (sxx * syy).sqrt())
}

/// Spearman rank correlation and its two-sided p-value.
///
/// Matches `scipy.stats.spearmanr`: ρ is Pearson on average-tied ranks, and the
/// p-value uses the t-distribution approximation
/// `t = ρ·sqrt(dof / (1 − ρ²))`, `dof = n − 2` — scipy's default, asymptotic but
/// applied at all n. Returns `None` when n < 3 or a variable is constant.
pub fn spearman(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len();
    if n < 3 || y.len() != n {
        return None;
    }
    let rho = pearson(&rankdata(x), &rankdata(y))?;
    let dof = (n - 2) as f64;
    // ρ = ±1 makes t infinite and p exactly 0.
    let denom = (1.0 + rho) * (1.0 - rho);
    let p = if denom <= 0.0 {
        0.0
    } else {
        let t = rho * (dof / denom).sqrt();
        2.0 * student_t_sf(t.abs(), dof)
    };
    Some((rho, p))
}

/// Which distribution the Wilcoxon p-value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WilcoxonMethod {
    Exact,
    Asymptotic,
}

/// Wilcoxon signed-rank test of `d ≠ 0`, two-sided.
///
/// Matches `scipy.stats.wilcoxon(d)` with its defaults: `zero_method="wilcox"`
/// (zero differences are discarded), `alternative="two-sided"`, statistic
/// `T = min(W⁺, W⁻)`, and `method="auto"` — the exact signed-rank distribution
/// when at most 50 non-zero differences remain and no `|d|` ties are present,
/// the tie-corrected normal approximation otherwise.
///
/// Returns `None` when every difference is zero, where the test is undefined
/// (scipy raises `ValueError` there).
pub fn wilcoxon(d: &[f64]) -> Option<(f64, f64, WilcoxonMethod)> {
    let nonzero: Vec<f64> = d.iter().copied().filter(|v| *v != 0.0).collect();
    let n = nonzero.len();
    if n == 0 {
        return None;
    }

    let abs: Vec<f64> = nonzero.iter().map(|v| v.abs()).collect();
    let ranks = rankdata(&abs);
    let mut r_plus = 0.0;
    let mut r_minus = 0.0;
    for (v, r) in nonzero.iter().zip(&ranks) {
        if *v > 0.0 {
            r_plus += r;
        } else {
            r_minus += r;
        }
    }
    let t = r_plus.min(r_minus);

    let tie_groups = tie_group_sizes(&abs);
    let has_ties = tie_groups.iter().any(|&c| c > 1);

    if n <= 50 && !has_ties {
        // Exact: P(W⁺ ≤ t) by counting subsets of {1..n} whose rank sum is ≤ t.
        let p = (2.0 * signed_rank_cdf(n, t)).min(1.0);
        return Some((t, p, WilcoxonMethod::Exact));
    }

    let nf = n as f64;
    let mn = nf * (nf + 1.0) * 0.25;
    let tie_correction: f64 = tie_groups
        .iter()
        .map(|&c| {
            let c = c as f64;
            c * c * c - c
        })
        .sum();
    let var = (nf * (nf + 1.0) * (2.0 * nf + 1.0) - 0.5 * tie_correction) / 24.0;
    if var <= 0.0 {
        return None;
    }
    let z = (t - mn) / var.sqrt();
    let p = (2.0 * normal_sf(z.abs())).min(1.0);
    Some((t, p, WilcoxonMethod::Asymptotic))
}

/// Sizes of runs of equal values (tie groups) in *xs*.
fn tie_group_sizes(xs: &[f64]) -> Vec<usize> {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i + 1;
        while j < v.len() && v[j] == v[i] {
            j += 1;
        }
        out.push(j - i);
        i = j;
    }
    out
}

/// `P(W⁺ ≤ t)` under the null, for *n* untied non-zero differences.
///
/// W⁺ is the sum of a uniformly random subset of the ranks `1..n`, so the exact
/// pmf is the subset-sum count divided by `2ⁿ`. The DP is over the `n(n+1)/2 + 1`
/// attainable sums; `n ≤ 50` keeps the counts (at most `2⁵⁰`) exact in f64.
fn signed_rank_cdf(n: usize, t: f64) -> f64 {
    let max_sum = n * (n + 1) / 2;
    let mut counts = vec![0.0f64; max_sum + 1];
    counts[0] = 1.0;
    for rank in 1..=n {
        for s in (rank..=max_sum).rev() {
            counts[s] += counts[s - rank];
        }
    }
    let total: f64 = (2.0f64).powi(n as i32);
    // t is an integer sum when there are no ties; floor guards float slop.
    let cut = (t.floor() as isize).clamp(-1, max_sum as isize);
    if cut < 0 {
        return 0.0;
    }
    counts[..=cut as usize].iter().sum::<f64>() / total
}

// ─── Distribution tails ───────────────────────────────────────────────────────

/// Upper tail of the standard normal, `P(Z > z)`.
pub fn normal_sf(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

/// Complementary error function (Numerical Recipes `erfcc`), fractional error
/// below 1.2e-7 everywhere.
///
/// Used only for p-values that are reported to four decimals, so this is far
/// more accuracy than the caller can consume.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let ans = t * (-z * z + poly).exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Upper tail of Student's t, `P(T > t)` for `dof` degrees of freedom.
///
/// `sf(t) = ½·I_x(dof/2, ½)` with `x = dof/(dof + t²)`, for `t ≥ 0`.
pub fn student_t_sf(t: f64, dof: f64) -> f64 {
    if dof <= 0.0 {
        return f64::NAN;
    }
    if !t.is_finite() {
        return if t > 0.0 { 0.0 } else { 1.0 };
    }
    let x = dof / (dof + t * t);
    let half = 0.5 * betai(0.5 * dof, 0.5, x);
    if t >= 0.0 {
        half
    } else {
        1.0 - half
    }
}

/// Regularised incomplete beta `I_x(a, b)` (Numerical Recipes `betai`).
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Continued-fraction expansion for the incomplete beta (Lentz's method).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 200;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        // Even step.
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// `ln Γ(x)` for `x > 0` (Lanczos, g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let mut acc = C[0];
    for (i, &c) in C.iter().enumerate().skip(1) {
        acc += c / (x + i as f64 - 1.0);
    }
    let t = x + G - 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x - 0.5) * t.ln() - t + acc.ln()
}

// ─── Friedman test + Holm post-hoc ────────────────────────────────────────────

/// Result of a Friedman test over `n_blocks` blocks and `k` treatments.
#[derive(Debug, Clone)]
pub struct Friedman {
    /// Tie-corrected χ²_F statistic.
    pub statistic: f64,
    /// Upper tail of χ² with `k − 1` degrees of freedom.
    pub p: f64,
    /// Mean rank per treatment, in input order. Rank 1 is best.
    pub mean_ranks: Vec<f64>,
    pub n_blocks: usize,
    pub k: usize,
}

/// Friedman rank-sum test over complete blocks (`scipy.stats.friedmanchisquare`).
///
/// Each element of *blocks* holds one value per treatment, in a consistent
/// order. **Larger is better**: values are ranked descending within each block,
/// so the best treatment in a block gets rank 1, matching the convention of
/// Demšar (2006). Ties inside a block share the average rank and feed the tie
/// correction `c = 1 − Σ(t³ − t) / (k(k² − 1)n)`.
///
/// `None` when there are fewer than two blocks, fewer than three treatments, a
/// ragged or non-finite block, or when every value in every block is tied (the
/// tie correction would divide by zero).
pub fn friedman(blocks: &[Vec<f64>]) -> Option<Friedman> {
    let n = blocks.len();
    if n < 2 {
        return None;
    }
    let k = blocks.first()?.len();
    // With two treatments this reduces to a sign test and the χ² approximation
    // is not the right tool; Demšar sends that case to Wilcoxon instead.
    if k < 3 {
        return None;
    }
    if blocks
        .iter()
        .any(|b| b.len() != k || b.iter().any(|v| !v.is_finite()))
    {
        return None;
    }

    let mut rank_sums = vec![0.0; k];
    let mut tie_term = 0.0f64;
    for block in blocks {
        // Negate so that `rankdata`'s ascending ranks put the best value first.
        let flipped: Vec<f64> = block.iter().map(|v| -v).collect();
        for (sum, r) in rank_sums.iter_mut().zip(rankdata(&flipped)) {
            *sum += r;
        }
        for t in tie_group_sizes(&flipped) {
            let t = t as f64;
            tie_term += t * (t * t - 1.0);
        }
    }

    let kf = k as f64;
    let nf = n as f64;
    let c = 1.0 - tie_term / (kf * (kf * kf - 1.0) * nf);
    if c <= 0.0 {
        return None;
    }
    let ssbn: f64 = rank_sums.iter().map(|s| s * s).sum();
    let statistic = (12.0 / (kf * nf * (kf + 1.0)) * ssbn - 3.0 * nf * (kf + 1.0)) / c;

    Some(Friedman {
        statistic,
        p: chi2_sf(statistic, kf - 1.0),
        mean_ranks: rank_sums.iter().map(|s| s / nf).collect(),
        n_blocks: n,
        k,
    })
}

/// Holm-adjusted p-values for each treatment against *control*, from a Friedman
/// test's mean ranks (Demšar 2006 §3.2.2).
///
/// The pairwise statistic is `z = (R_i − R_control) / sqrt(k(k+1) / 6n)`, taken
/// two-sided, and the resulting `k − 1` p-values are adjusted by Holm's
/// step-down procedure. Returns one entry per treatment in the same order as
/// `mean_ranks`, with `None` at the control's own index.
///
/// Adjusted p-values are compared against α directly; they already carry the
/// family-wise correction over the `k − 1` comparisons.
pub fn holm_against_control(f: &Friedman, control: usize) -> Vec<Option<f64>> {
    let k = f.k;
    let n = f.n_blocks;
    if control >= k {
        return vec![None; k];
    }
    let se = ((k * (k + 1)) as f64 / (6.0 * n as f64)).sqrt();

    let mut raw: Vec<(usize, f64)> = (0..k)
        .filter(|&i| i != control)
        .map(|i| {
            let z = (f.mean_ranks[i] - f.mean_ranks[control]) / se;
            (i, (2.0 * normal_sf(z.abs())).min(1.0))
        })
        .collect();

    // Holm: sort ascending, scale the j-th smallest by (m − j), and enforce
    // monotonicity so an adjusted p never drops below an earlier one.
    raw.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let m = raw.len();
    let mut running = 0.0f64;
    let mut out = vec![None; k];
    for (j, (i, p)) in raw.into_iter().enumerate() {
        let adjusted = ((m - j) as f64 * p).min(1.0).max(running);
        running = adjusted;
        out[i] = Some(adjusted);
    }
    out
}

/// Upper tail of the χ² distribution, `P(X > x)` with `dof` degrees of freedom.
pub fn chi2_sf(x: f64, dof: f64) -> f64 {
    if dof <= 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(0.5 * dof, 0.5 * x)
}

/// Regularised upper incomplete gamma `Q(a, x)` (Numerical Recipes `gammq`):
/// the series for `x < a + 1`, the continued fraction beyond it.
fn gamma_q(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cf(a, x)
    }
}

/// Series expansion for the regularised lower incomplete gamma `P(a, x)`.
fn gamma_series(a: f64, x: f64) -> f64 {
    const MAXIT: usize = 500;
    const EPS: f64 = 3.0e-14;
    if x <= 0.0 {
        return 0.0;
    }
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..MAXIT {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Continued-fraction expansion for `Q(a, x)` (Lentz's method).
fn gamma_cf(a: f64, x: f64) -> f64 {
    const MAXIT: usize = 500;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAXIT {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h * (-x + a * x.ln() - ln_gamma(a)).exp()
}
