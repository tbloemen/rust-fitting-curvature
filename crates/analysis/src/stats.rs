//! The handful of statistics the analysis needs, ported from scipy and matching
//! its default options: summary statistics, ranks, Pearson's r and Spearman's ρ
//! with its t-approximation p-value (`scipy.stats.spearmanr`).

use fitting_core::cast::{count_to_f64, to_usize};

/// Mean of a slice; `None` when empty.
#[must_use]
pub fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().sum::<f64>() / count_to_f64(xs.len()))
}

/// Median of a slice (average of the two middle values for even length).
/// `None` when empty. Non-finite values are not filtered — callers do that.
#[must_use]
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

/// The *q*-quantile of a slice by linear interpolation between order statistics
/// (numpy's `quantile(method="linear")`, scipy's default). `None` when empty or
/// when *q* is outside [0, 1]. Non-finite values are not filtered — callers do
/// that.
#[must_use]
pub fn quantile(xs: &[f64], q: f64) -> Option<f64> {
    if xs.is_empty() || !(0.0..=1.0).contains(&q) {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Virtual index into the sorted values, then interpolate its neighbours.
    let pos = q * count_to_f64(v.len() - 1);
    let lo = to_usize(pos.floor());
    let hi = to_usize(pos.ceil());
    Some(v[lo] + (v[hi] - v[lo]) * (pos - count_to_f64(lo)))
}

/// Fractional ranks with ties averaged (scipy's `rankdata(method="average")`).
#[must_use]
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
        while j < n && xs[idx[j]].partial_cmp(&xs[idx[i]]) == Some(std::cmp::Ordering::Equal) {
            j += 1;
        }
        let avg = count_to_f64(i + 1 + j) / 2.0; // mean of ranks i+1 .. j (1-based)
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation of two equal-length slices; `None` if either is constant.
#[must_use]
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
#[must_use]
pub fn spearman(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len();
    if n < 3 || y.len() != n {
        return None;
    }
    let rho = pearson(&rankdata(x), &rankdata(y))?;
    let dof = count_to_f64(n - 2);
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

/// Pairwise Spearman ρ between every two of *columns*, each a variable over
/// the same observations.
///
/// Symmetric, with `1.0` on the diagonal and `None` wherever [`spearman`] is
/// undefined for the pair — a constant column, or fewer than three
/// observations. Each pair is computed once and mirrored, and the p-value is
/// discarded: a correlation matrix is read for the ρ, and at the ~1000
/// observations per cell it is drawn from, every ρ past ±0.1 is significant
/// anyway.
///
/// # Panics
///
/// Panics if the columns are not all the same length — the caller has already
/// aligned them by observation, and a ragged input is a bug there, not a
/// value to carry.
#[must_use]
pub fn spearman_matrix(columns: &[Vec<f64>]) -> Vec<Vec<Option<f64>>> {
    let k = columns.len();
    if let Some(first) = columns.first() {
        assert!(
            columns.iter().all(|c| c.len() == first.len()),
            "spearman_matrix: columns must be the same length"
        );
    }
    let mut out = vec![vec![None; k]; k];
    for i in 0..k {
        out[i][i] = Some(1.0);
        for j in (i + 1)..k {
            let rho = spearman(&columns[i], &columns[j]).map(|(rho, _)| rho);
            out[i][j] = rho;
            out[j][i] = rho;
        }
    }
    out
}

// ─── Distribution tails ───────────────────────────────────────────────────────

/// Upper tail of Student's t, `P(T > t)` for `dof` degrees of freedom.
///
/// `sf(t) = ½·I_x(dof/2, ½)` with `x = dof/(dof + t²)`, for `t ≥ 0`.
#[must_use]
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
#[expect(
    clippy::many_single_char_names,
    reason = "Numerical Recipes Lentz continued fraction — names match the reference"
)]
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
        let m_f = count_to_f64(m);
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
        acc += c / (x + count_to_f64(i) - 1.0);
    }
    let t = x + G - 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x - 0.5) * t.ln() - t + acc.ln()
}
