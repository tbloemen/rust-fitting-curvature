/// Binary search for sigma that achieves target perplexity for a single point.
pub fn binary_search_sigma(sq_distances: &[f64], target_perplexity: f64) -> f64 {
    let target_entropy = target_perplexity.ln();
    let tol = 1e-5;
    let max_iter = 50;

    let mut sigma_min = 1e-10_f64;
    let mut sigma_max = 1e4_f64;
    let mut sigma = 1.0_f64;

    for _ in 0..max_iter {
        let two_sigma_sq = 2.0 * sigma * sigma;

        // Compute scaled exponents
        let max_val = sq_distances
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = sq_distances
            .iter()
            .map(|&d| (-d / two_sigma_sq - (-max_val / two_sigma_sq)).exp())
            .collect();

        let sum_exp: f64 = exps.iter().sum();
        if sum_exp < 1e-10 {
            sigma_min = sigma;
            sigma = (sigma_min + sigma_max) / 2.0;
            continue;
        }

        // Probabilities and entropy
        let mut entropy = 0.0_f64;
        for &e in &exps {
            let p = e / sum_exp;
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }

        let diff = entropy - target_entropy;
        if diff.abs() < tol {
            break;
        }

        if diff > 0.0 {
            sigma_max = sigma;
        } else {
            sigma_min = sigma;
        }
        sigma = (sigma_min + sigma_max) / 2.0;
    }

    sigma
}

fn square_distances(data: &[f64], n_points: usize, n_features: usize) -> Vec<f64> {
    let mut sq_dist = vec![0.0; n_points * n_points];
    for i in 0..n_points {
        for j in (i + 1)..n_points {
            let mut d = 0.0;
            for f in 0..n_features {
                let diff = data[i * n_features + f] - data[j * n_features + f];
                d += diff * diff;
            }
            sq_dist[i * n_points + j] = d;
            sq_dist[j * n_points + i] = d;
        }
        sq_dist[i * n_points + i] = f64::INFINITY;
    }
    sq_dist
}

fn compute_conditional_probabilities(
    sq_distances: &[f64],
    perplexity: f64,
    n_points: usize,
) -> Vec<f64> {
    let k = (n_points - 1).min((3.0 * perplexity + 1.0) as usize);
    let mut p_cond = vec![0.0; n_points * n_points];

    for i in 0..n_points {
        // Find k nearest neighbor indices
        let mut indices: Vec<usize> = (0..n_points).filter(|&j| j != i).collect();
        indices.sort_by(|&a, &b| {
            sq_distances[i * n_points + a]
                .partial_cmp(&sq_distances[i * n_points + b])
                .unwrap()
        });
        indices.truncate(k);

        // Get distances to neighbors
        let neighbor_dists: Vec<f64> = indices
            .iter()
            .map(|&j| sq_distances[i * n_points + j])
            .collect();

        // Binary search for sigma
        let sigma = binary_search_sigma(&neighbor_dists, perplexity);
        let two_sigma_sq = 2.0 * sigma * sigma;

        // Compute conditional probabilities
        let exps: Vec<f64> = neighbor_dists
            .iter()
            .map(|&d| (-d / two_sigma_sq).exp())
            .collect();
        let sum_exp: f64 = exps.iter().sum();

        for (idx, &j) in indices.iter().enumerate() {
            p_cond[i * n_points + j] = exps[idx] / sum_exp;
        }
    }
    p_cond
}

fn symmetrize_and_normalize_p(p_cond: &[f64], n_points: usize) -> Vec<f64> {
    let mut p = vec![0.0; n_points * n_points];
    let denom = 2.0 * n_points as f64;
    for i in 0..n_points {
        for j in 0..n_points {
            p[i * n_points + j] = (p_cond[i * n_points + j] + p_cond[j * n_points + i]) / denom;
        }
    }
    // Normalize to sum to 1
    let total: f64 = p.iter().sum();
    if total > 0.0 {
        for val in &mut p {
            *val /= total;
        }
    }
    p
}

/// Compute symmetric perplexity-based affinity matrix P.
///
/// Returns a flat n_points x n_points row-major matrix where P sums to 1.
pub fn compute_perplexity_affinities(
    data: &[f64],
    n_points: usize,
    n_features: usize,
    perplexity: f64,
) -> Vec<f64> {
    // Compute all pairwise squared distances
    let sq_dist = square_distances(data, n_points, n_features);

    // For each point, find k nearest neighbors and compute conditional probabilities
    let p_cond = compute_conditional_probabilities(&sq_dist, perplexity, n_points);

    // Symmetrize: V_ij = (p_j|i + p_i|j) / (2*n)
    symmetrize_and_normalize_p(&p_cond, n_points)
}

/// Compute affinities from a precomputed distance matrix.
pub fn compute_perplexity_affinities_from_distances(
    distance_matrix: &[f64],
    n_points: usize,
    perplexity: f64,
) -> Vec<f64> {
    // Square the distances
    let mut sq_dist: Vec<f64> = distance_matrix.iter().map(|&d| d * d).collect();
    for i in 0..n_points {
        sq_dist[i * n_points + i] = f64::INFINITY;
    }

    let p_cond = compute_conditional_probabilities(&sq_dist, perplexity, n_points);

    symmetrize_and_normalize_p(&p_cond, n_points)
}
