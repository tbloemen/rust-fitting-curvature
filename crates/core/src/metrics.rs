//! Embedding quality metrics.
//!
//! Ported from Python `src/metrics.py`. Metrics grouped into:
//! - A. Local structure preservation (knn_overlap)
//! - B. Global geometry preservation (geodesic_distortion)
//! - C. Space efficiency (area_utilisation, radial_distribution)
//! - D. Perceptual evaluation (cluster_density_measure, dunn_index)

// ---------------------------------------------------------------------------
// A. Local structure preservation
// ---------------------------------------------------------------------------

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

    let knn_indices = |dist: &[f64]| -> Vec<Vec<usize>> {
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
    };

    let high_knn = knn_indices(high_dim_distances);
    let embed_knn = knn_indices(embedded_distances);

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

/// Cluster Density Measure (ClDM) from Albuquerque et al. (2010).
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

/// Davies-Bouldin index from precomputed distance matrix.
/// Lower = better separated, more compact clusters.
pub fn davies_bouldin(embedded_distances: &[f64], labels: &[u32], n: usize) -> f64 {
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
            let total: f64 = indices
                .iter()
                .map(|&j| embedded_distances[candidate * n + j])
                .sum();
            if total < best_total {
                best_total = total;
                medoid_indices[ci] = candidate;
            }
        }
        // Scatter: mean distance to medoid
        let medoid = medoid_indices[ci];
        scatters[ci] = indices
            .iter()
            .map(|&j| embedded_distances[medoid * n + j])
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
            let d_ij = embedded_distances[medoid_indices[i] * n + medoid_indices[j]];
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
