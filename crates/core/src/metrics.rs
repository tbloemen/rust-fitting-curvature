//! Embedding quality metrics.
//!
//! Metrics grouped into:
//! - A. Local structure preservation (trustworthiness, continuity, knn_overlap)
//! - B. Global geometry preservation (geodesic_distortion)
//! - C. Space efficiency (radial_distribution)
//! - D. Perceptual evaluation (class_density_measure, cluster_density_measure,
//!   davies_bouldin, davies_bouldin_ratio)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute ranks: for each point i, ranks[i*n + j] = rank of j sorted by
/// distance from i (0 = self, 1 = nearest neighbor, etc.).
fn compute_ranks(distances: &[f64], n: usize) -> Vec<usize> {
    let mut ranks = vec![0usize; n * n];
    for i in 0..n {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            distances[i * n + a]
                .partial_cmp(&distances[i * n + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (rank, &j) in indices.iter().enumerate() {
            ranks[i * n + j] = rank;
        }
    }
    ranks
}

/// Compute k-nearest neighbor index sets (excluding self).
fn knn_index_sets(dist: &[f64], n: usize, k: usize) -> Vec<Vec<usize>> {
    (0..n)
        .map(|i| {
            let mut indices: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            indices.sort_by(|&a, &b| {
                dist[i * n + a]
                    .partial_cmp(&dist[i * n + b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            indices.truncate(k);
            indices
        })
        .collect()
}

/// Compute pairwise Euclidean distance matrix from 2D points (flat [x,y] pairs).
///
/// Use this to obtain "after-projection" distances from the output of
/// `visualisation::project_to_2d`, so that any metric can be evaluated on
/// what the viewer actually sees rather than on the manifold geometry.
pub fn euclidean_dist_2d(pts_2d: &[f64], n: usize) -> Vec<f64> {
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = pts_2d[i * 2] - pts_2d[j * 2];
            let dy = pts_2d[i * 2 + 1] - pts_2d[j * 2 + 1];
            let d = (dx * dx + dy * dy).sqrt();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// A. Local structure preservation
// ---------------------------------------------------------------------------

/// Trustworthiness (Venna & Kaski 2006).
///
/// Measures whether points that appear as neighbors in the embedding are also
/// neighbors in the original space. Penalizes "false neighbors" in the embedding.
/// Returns a value in [0, 1], higher is better.
pub fn trustworthiness(
    high_dim_distances: &[f64],
    embedded_distances: &[f64],
    n: usize,
    k: usize,
) -> f64 {
    let k = k.min(n - 2);
    if k == 0 || n < 3 {
        return 1.0;
    }

    let ranks_high = compute_ranks(high_dim_distances, n);
    let embed_knn = knn_index_sets(embedded_distances, n, k);

    let denom = n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0);
    if denom < 1e-12 {
        return 1.0;
    }

    let mut penalty = 0.0;
    for i in 0..n {
        for &j in &embed_knn[i] {
            // ranks_high uses 0-based ranks where 0 = self, so rank > k means
            // j is NOT among i's k nearest in high-dim space
            let r = ranks_high[i * n + j];
            if r > k {
                penalty += (r - k) as f64;
            }
        }
    }

    1.0 - (2.0 / denom) * penalty
}

/// Continuity (Venna & Kaski 2006).
///
/// Measures whether points that are neighbors in the original space remain
/// neighbors in the embedding. Penalizes "missed neighbors" from the original.
/// Returns a value in [0, 1], higher is better.
pub fn continuity(
    high_dim_distances: &[f64],
    embedded_distances: &[f64],
    n: usize,
    k: usize,
) -> f64 {
    let k = k.min(n - 2);
    if k == 0 || n < 3 {
        return 1.0;
    }

    let ranks_embed = compute_ranks(embedded_distances, n);
    let high_knn = knn_index_sets(high_dim_distances, n, k);

    let denom = n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0);
    if denom < 1e-12 {
        return 1.0;
    }

    let mut penalty = 0.0;
    for i in 0..n {
        for &j in &high_knn[i] {
            let r = ranks_embed[i * n + j];
            if r > k {
                penalty += (r - k) as f64;
            }
        }
    }

    1.0 - (2.0 / denom) * penalty
}

/// Fraction of k-nearest neighbors preserved between high-dim and embedded spaces.
///
/// Returns a value in [0, 1], higher is better.
pub fn knn_overlap(
    high_dim_distances: &[f64],
    embedded_distances: &[f64],
    n: usize,
    k: usize,
) -> f64 {
    let k = k.min(n - 1);

    let high_knn = knn_index_sets(high_dim_distances, n, k);
    let embed_knn = knn_index_sets(embedded_distances, n, k);

    let mut total_overlap = 0.0;
    for i in 0..n {
        let count = high_knn[i]
            .iter()
            .filter(|idx| embed_knn[i].contains(idx))
            .count();
        total_overlap += count as f64 / k as f64;
    }
    total_overlap / n as f64
}

// ---------------------------------------------------------------------------
// B. Global geometry preservation
// ---------------------------------------------------------------------------

/// Geodesic distortion (Gu et al. 2019): mean ||(d_embed/d_high)^2 - 1||.
pub fn geodesic_distortion_gu2019(
    high_dim_distances: &[f64],
    embedded_distances: &[f64],
    n: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let h = high_dim_distances[i * n + j];
            let e = embedded_distances[i * n + j];
            if h > 1e-12 {
                sum += ((e / h).powi(2) - 1.0).abs();
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Geodesic distortion (MSE): mean (d_embed - d_high)^2.
pub fn geodesic_distortion_mse(
    high_dim_distances: &[f64],
    embedded_distances: &[f64],
    n: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = embedded_distances[i * n + j] - high_dim_distances[i * n + j];
            sum += diff * diff;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

// ---------------------------------------------------------------------------
// C. Space efficiency
// ---------------------------------------------------------------------------

/// Normalized std of radial distances from centroid (coefficient of variation).
/// Lower = more uniform spread.
pub fn radial_distribution(pts_2d: &[f64], n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }

    // Compute centroid
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        cx += pts_2d[i * 2];
        cy += pts_2d[i * 2 + 1];
    }
    cx /= n as f64;
    cy /= n as f64;

    // Compute radii
    let mut radii = Vec::with_capacity(n);
    for i in 0..n {
        let dx = pts_2d[i * 2] - cx;
        let dy = pts_2d[i * 2 + 1] - cy;
        radii.push((dx * dx + dy * dy).sqrt());
    }

    let mean_r: f64 = radii.iter().sum::<f64>() / n as f64;
    if mean_r < 1e-12 {
        return 0.0;
    }

    let var: f64 = radii.iter().map(|&r| (r - mean_r).powi(2)).sum::<f64>() / n as f64;
    var.sqrt() / mean_r
}

// ---------------------------------------------------------------------------
// D. Perceptual evaluation
// ---------------------------------------------------------------------------

/// Class Density Measure (CDM) from Tatu et al. (2009).
///
/// Evaluates class separation by computing per-class density fields on a grid
/// and measuring the sum of absolute differences between all class pairs.
/// Uses Gaussian KDE to estimate smooth density per class.
/// Operates on 2D projected coordinates.
/// Returns a normalized value in [0, 1], higher = better separated classes.
pub fn class_density_measure(pts_2d: &[f64], labels: &[u32], n: usize) -> f64 {
    const GRID_SIZE: usize = 80;

    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort();
    unique_labels.dedup();
    let num_classes = unique_labels.len();
    if num_classes < 2 {
        return 0.0;
    }

    // Bounding box with margin
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for i in 0..n {
        let x = pts_2d[i * 2];
        let y = pts_2d[i * 2 + 1];
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    let extent = (x_max - x_min).max(y_max - y_min);
    if extent < 1e-12 {
        return 0.0;
    }
    let margin = extent * 0.05;
    x_min -= margin;
    x_max += margin;
    y_min -= margin;
    y_max += margin;
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    let cell_x = x_range / GRID_SIZE as f64;
    let cell_y = y_range / GRID_SIZE as f64;

    // Bandwidth for Gaussian KDE (Silverman-like, adapted to data extent)
    let bandwidth = extent / (n as f64).powf(0.2) * 0.5;
    let bw_sq = bandwidth * bandwidth;
    let kernel_radius_x = (3.0 * bandwidth / cell_x).ceil() as isize;
    let kernel_radius_y = (3.0 * bandwidth / cell_y).ceil() as isize;

    // Group points by class
    let class_points: Vec<Vec<(f64, f64)>> = unique_labels
        .iter()
        .map(|&lbl| {
            (0..n)
                .filter(|&i| labels[i] == lbl)
                .map(|i| (pts_2d[i * 2], pts_2d[i * 2 + 1]))
                .collect()
        })
        .collect();

    // Compute density images using Gaussian KDE splatting
    let num_pixels = GRID_SIZE * GRID_SIZE;
    let mut density_images: Vec<Vec<f64>> = Vec::with_capacity(num_classes);

    for class_pts in &class_points {
        let mut density = vec![0.0f64; num_pixels];

        for &(px, py) in class_pts {
            let gcx = ((px - x_min) / cell_x) as isize;
            let gcy = ((py - y_min) / cell_y) as isize;

            for dy in -kernel_radius_y..=kernel_radius_y {
                let ny = gcy + dy;
                if ny < 0 || ny >= GRID_SIZE as isize {
                    continue;
                }
                let grid_y = y_min + (ny as f64 + 0.5) * cell_y;
                let dist_y = py - grid_y;

                for dx in -kernel_radius_x..=kernel_radius_x {
                    let nx = gcx + dx;
                    if nx < 0 || nx >= GRID_SIZE as isize {
                        continue;
                    }
                    let grid_x = x_min + (nx as f64 + 0.5) * cell_x;
                    let dist_x = px - grid_x;
                    let dist_sq = dist_x * dist_x + dist_y * dist_y;
                    density[(ny as usize) * GRID_SIZE + nx as usize] +=
                        (-dist_sq / (2.0 * bw_sq)).exp();
                }
            }
        }
        density_images.push(density);
    }

    // Normalize each density image to [0, 1]
    for density in &mut density_images {
        let max_val = density.iter().cloned().fold(0.0f64, f64::max);
        if max_val > 1e-12 {
            for v in density.iter_mut() {
                *v /= max_val;
            }
        }
    }

    // CDM = sum over class pairs of sum of |density_k - density_l|
    let mut cdm = 0.0;
    let mut num_pairs = 0usize;
    for ci in 0..num_classes {
        for cj in (ci + 1)..num_classes {
            for (a, b) in density_images[ci].iter().zip(&density_images[cj]) {
                cdm += (a - b).abs();
            }
            num_pairs += 1;
        }
    }

    if num_pairs == 0 {
        return 0.0;
    }

    // Normalize to [0, 1]
    cdm / (num_pairs as f64 * num_pixels as f64)
}

/// Cluster Density Measure (ClDM) from Albuquerque et al. (2010).
///
/// Uses the label-based cluster formula on 2D projected coordinates:
/// ClDM = (1/K) * sum_{k<l} d²_{k,l} / (r_k * r_l)
///
/// Measures how well-separated and compact the clusters are.
/// Higher values = better separated clusters.
pub fn cluster_density_measure(pts_2d: &[f64], labels: &[u32], n: usize) -> f64 {
    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort();
    unique_labels.dedup();
    let k = unique_labels.len();
    if k < 2 {
        return 0.0;
    }

    // Compute centroids and radii per cluster
    let mut centroids = vec![(0.0f64, 0.0f64); k];
    let mut counts = vec![0usize; k];
    let mut radii = vec![0.0f64; k];

    for i in 0..n {
        let label_idx = unique_labels.iter().position(|&l| l == labels[i]).unwrap();
        centroids[label_idx].0 += pts_2d[i * 2];
        centroids[label_idx].1 += pts_2d[i * 2 + 1];
        counts[label_idx] += 1;
    }
    for ci in 0..k {
        if counts[ci] > 0 {
            centroids[ci].0 /= counts[ci] as f64;
            centroids[ci].1 /= counts[ci] as f64;
        }
    }

    // Compute average radius per cluster
    for i in 0..n {
        let label_idx = unique_labels.iter().position(|&l| l == labels[i]).unwrap();
        let dx = pts_2d[i * 2] - centroids[label_idx].0;
        let dy = pts_2d[i * 2 + 1] - centroids[label_idx].1;
        radii[label_idx] += (dx * dx + dy * dy).sqrt();
    }
    for ci in 0..k {
        radii[ci] = if counts[ci] > 0 {
            (radii[ci] / counts[ci] as f64).max(1e-12)
        } else {
            1e-12
        };
    }

    let mut cldm = 0.0;
    for ki in 0..k {
        for kj in (ki + 1)..k {
            let dx = centroids[ki].0 - centroids[kj].0;
            let dy = centroids[ki].1 - centroids[kj].1;
            let d_sq = dx * dx + dy * dy;
            cldm += d_sq / (radii[ki] * radii[kj]);
        }
    }
    cldm / k as f64
}

/// Davies-Bouldin index from precomputed distance matrix.
/// Lower = better separated, more compact clusters.
pub fn davies_bouldin(distances: &[f64], labels: &[u32], n: usize) -> f64 {
    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort();
    unique_labels.dedup();
    let k = unique_labels.len();
    if k < 2 {
        return 0.0;
    }

    let cluster_indices: Vec<Vec<usize>> = unique_labels
        .iter()
        .map(|&lbl| (0..n).filter(|&i| labels[i] == lbl).collect())
        .collect();

    // Compute scatters and medoids
    let mut scatters = vec![0.0f64; k];
    let mut medoid_indices = vec![0usize; k];

    for (ci, indices) in cluster_indices.iter().enumerate() {
        if indices.is_empty() {
            continue;
        }
        // Find medoid (point with smallest total distance to others)
        let mut best_total = f64::INFINITY;
        for &candidate in indices {
            let total: f64 = indices.iter().map(|&j| distances[candidate * n + j]).sum();
            if total < best_total {
                best_total = total;
                medoid_indices[ci] = candidate;
            }
        }
        // Scatter: mean distance to medoid
        let medoid = medoid_indices[ci];
        scatters[ci] = indices
            .iter()
            .map(|&j| distances[medoid * n + j])
            .sum::<f64>()
            / indices.len() as f64;
    }

    // DB index
    let mut db = 0.0;
    for i in 0..k {
        let mut max_ratio = 0.0f64;
        for j in 0..k {
            if i == j {
                continue;
            }
            let d_ij = distances[medoid_indices[i] * n + medoid_indices[j]];
            if d_ij < 1e-12 {
                continue;
            }
            let ratio = (scatters[i] + scatters[j]) / d_ij;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        db += max_ratio;
    }
    db / k as f64
}

/// Davies-Bouldin ratio: DB_high / DB_projected.
///
/// Computes the DB index on both the high-dimensional data distances and
/// the 2D projected Euclidean distances. A higher ratio indicates the
/// projection preserves or improves cluster separation relative to the
/// original data (Di Caro et al. 2010).
pub fn davies_bouldin_ratio(
    high_dim_distances: &[f64],
    pts_2d: &[f64],
    labels: &[u32],
    n: usize,
) -> f64 {
    let dist_2d = euclidean_dist_2d(pts_2d, n);
    let db_high = davies_bouldin(high_dim_distances, labels, n);
    let db_proj = davies_bouldin(&dist_2d, labels, n);
    if db_proj < 1e-12 {
        return 0.0;
    }
    db_high / db_proj
}

// ---------------------------------------------------------------------------
// E. Distance-rank preservation
// ---------------------------------------------------------------------------

/// Assign 0-based ranks to `values` (rank 0 = smallest value).
fn rank_vector(values: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0usize; values.len()];
    for (rank, &idx) in indices.iter().enumerate() {
        ranks[idx] = rank;
    }
    ranks
}

/// Normalized stress (Kruskal 1964).
///
/// Measures the global preservation of pairwise distances from the original
/// space D to the projected space P(D):
///
/// `M_s = sqrt( sum_{i<j}(Δ_D(i,j) - Δ_P(i,j))² / sum_{i<j} Δ_D(i,j)² )`
///
/// Returns a value in [0, 1], with **0 being best** (perfect distance
/// preservation).  Pass manifold geodesic distances as `embedded_distances`
/// to measure embedding quality; pass `euclidean_dist_2d` output to measure
/// final visualization quality.
pub fn normalized_stress(high_dim_distances: &[f64], embedded_distances: &[f64], n: usize) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d_h = high_dim_distances[i * n + j];
            let d_e = embedded_distances[i * n + j];
            let diff = d_h - d_e;
            numerator += diff * diff;
            denominator += d_h * d_h;
        }
    }
    if denominator < 1e-12 {
        return 0.0;
    }
    (numerator / denominator).sqrt()
}

/// Neighborhood hit (van der Maaten 2009).
///
/// Measures how well the k-nearest-neighbor structure in the embedding aligns
/// with class labels: for each point, the fraction of its k-NN in the
/// embedding that share the same label, averaged over all points.
///
/// `M_NH = (1/N) * sum_i |{j ∈ kNN_embed(i) : label[j] = label[i]}| / k`
///
/// Returns a value in [0, 1], with **1 being best** (all k-NN same-class).
/// Requires labeled data. Pass manifold geodesic or 2D Euclidean distances
/// for the before/after projection distinction.
pub fn neighborhood_hit(embedded_distances: &[f64], labels: &[u32], n: usize, k: usize) -> f64 {
    let k = k.min(n - 1);
    if k == 0 {
        return 1.0;
    }

    let knn = knn_index_sets(embedded_distances, n, k);
    let mut total = 0.0;
    for i in 0..n {
        let same = knn[i].iter().filter(|&&j| labels[j] == labels[i]).count();
        total += same as f64 / k as f64;
    }
    total / n as f64
}

/// Shepard goodness (Spearman rank correlation of pairwise distances).
///
/// Computes the Spearman rank correlation between all N*(N-1)/2 upper-triangle
/// pairwise distances in the original space and in the embedding, giving a
/// scalar measure of how well the global rank-order of distances is preserved.
///
/// Returns a value in [0, 1], with **1 being best** (perfect rank order
/// preservation).  The result is clipped to 0 from below since negative
/// correlations are meaningless for projection quality.
pub fn shepard_goodness(high_dim_distances: &[f64], embedded_distances: &[f64], n: usize) -> f64 {
    let m = n * (n - 1) / 2;
    if m < 2 {
        return 1.0;
    }

    let mut d_high = Vec::with_capacity(m);
    let mut d_embed = Vec::with_capacity(m);
    for i in 0..n {
        for j in (i + 1)..n {
            d_high.push(high_dim_distances[i * n + j]);
            d_embed.push(embedded_distances[i * n + j]);
        }
    }

    let ranks_high = rank_vector(&d_high);
    let ranks_embed = rank_vector(&d_embed);

    // Spearman: r_s = 1 - 6 * sum(d_i^2) / (m * (m^2 - 1))
    let sum_sq: f64 = ranks_high
        .iter()
        .zip(ranks_embed.iter())
        .map(|(&rh, &re)| {
            let diff = rh as f64 - re as f64;
            diff * diff
        })
        .sum();

    let denom = m as f64 * (m as f64 * m as f64 - 1.0);
    if denom < 1e-12 {
        return 1.0;
    }

    (1.0 - 6.0 * sum_sq / denom).max(0.0)
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// All quality metrics for a completed embedding, in both manifold and
/// 2D-projected variants (the "before" / "after projecting" distinction).
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    // A. Local structure preservation
    pub trustworthiness_manifold: f64,
    pub trustworthiness_2d: f64,
    pub continuity_manifold: f64,
    pub continuity_2d: f64,
    pub knn_overlap_manifold: f64,
    pub knn_overlap_2d: f64,
    // B. Distance preservation
    pub normalized_stress_manifold: f64,
    pub normalized_stress_2d: f64,
    pub shepard_goodness_manifold: f64,
    pub shepard_goodness_2d: f64,
    // C. Label-dependent (None when no labels provided)
    pub neighborhood_hit_manifold: Option<f64>,
    pub neighborhood_hit_2d: Option<f64>,
    // D. Class separation — 2D only, label-dependent
    pub class_density_measure: Option<f64>,
    pub cluster_density_measure: Option<f64>,
    pub davies_bouldin_ratio: Option<f64>,
}

/// Compute a full metrics snapshot.
///
/// `high_dim_dist` — pairwise distances in input space.
/// `embed_dist`    — manifold geodesic distances (before projection).
/// `pts_2d`        — flat (x,y) pairs from `project_to_2d` (after projection).
/// `labels`        — optional class labels; label-dependent metrics are `None` when absent.
/// `k`             — neighbourhood size used for kNN metrics.
pub fn compute_snapshot(
    high_dim_dist: &[f64],
    embed_dist: &[f64],
    pts_2d: &[f64],
    labels: Option<&[u32]>,
    n: usize,
    k: usize,
) -> MetricsSnapshot {
    let dist_2d = euclidean_dist_2d(pts_2d, n);

    let (neighborhood_hit_manifold, neighborhood_hit_2d, class_density, cluster_density, db_ratio) =
        if let Some(lbl) = labels {
            (
                Some(neighborhood_hit(embed_dist, lbl, n, k)),
                Some(neighborhood_hit(&dist_2d, lbl, n, k)),
                Some(class_density_measure(pts_2d, lbl, n)),
                Some(cluster_density_measure(pts_2d, lbl, n)),
                Some(davies_bouldin_ratio(high_dim_dist, pts_2d, lbl, n)),
            )
        } else {
            (None, None, None, None, None)
        };

    MetricsSnapshot {
        trustworthiness_manifold: trustworthiness(high_dim_dist, embed_dist, n, k),
        trustworthiness_2d: trustworthiness(high_dim_dist, &dist_2d, n, k),
        continuity_manifold: continuity(high_dim_dist, embed_dist, n, k),
        continuity_2d: continuity(high_dim_dist, &dist_2d, n, k),
        knn_overlap_manifold: knn_overlap(high_dim_dist, embed_dist, n, k),
        knn_overlap_2d: knn_overlap(high_dim_dist, &dist_2d, n, k),
        normalized_stress_manifold: normalized_stress(high_dim_dist, embed_dist, n),
        normalized_stress_2d: normalized_stress(high_dim_dist, &dist_2d, n),
        shepard_goodness_manifold: shepard_goodness(high_dim_dist, embed_dist, n),
        shepard_goodness_2d: shepard_goodness(high_dim_dist, &dist_2d, n),
        neighborhood_hit_manifold,
        neighborhood_hit_2d,
        class_density_measure: class_density,
        cluster_density_measure: cluster_density,
        davies_bouldin_ratio: db_ratio,
    }
}

/// Dunn index: ratio of minimum inter-cluster distance to maximum intra-cluster diameter.
/// Higher = better clustering.
pub fn dunn_index(embedded_distances: &[f64], labels: &[u32], n: usize) -> f64 {
    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort();
    unique_labels.dedup();
    let k = unique_labels.len();
    if k < 2 {
        return 0.0;
    }

    // Cluster indices
    let cluster_indices: Vec<Vec<usize>> = unique_labels
        .iter()
        .map(|&lbl| (0..n).filter(|&i| labels[i] == lbl).collect())
        .collect();

    // Max intra-cluster diameter
    let mut max_intra = 0.0f64;
    for indices in &cluster_indices {
        for &a in indices {
            for &b in indices {
                let d = embedded_distances[a * n + b];
                if d > max_intra {
                    max_intra = d;
                }
            }
        }
    }
    if max_intra < 1e-12 {
        return 0.0;
    }

    // Min inter-cluster distance
    let mut min_inter = f64::INFINITY;
    for ci in 0..k {
        for cj in (ci + 1)..k {
            for &a in &cluster_indices[ci] {
                for &b in &cluster_indices[cj] {
                    let d = embedded_distances[a * n + b];
                    if d < min_inter {
                        min_inter = d;
                    }
                }
            }
        }
    }

    min_inter / max_intra
}
