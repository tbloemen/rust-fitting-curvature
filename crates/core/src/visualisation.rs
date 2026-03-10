//! Projection utilities for constant-curvature embeddings.
//!
//! Ported from Python `src/visualisation.py`.
//! Supports stereographic, azimuthal equidistant, and orthographic projections
//! for spherical data, Poincaré disk for hyperbolic, and direct coordinates for Euclidean.

use std::f64::consts::PI;

/// Projection method for spherical embeddings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SphericalProjection {
    Stereographic,
    AzimuthalEquidistant,
    Orthographic,
}

/// Result of projecting points to 2D.
pub struct Projection2D {
    /// Flat array of (x, y) pairs, length 2*n_points.
    pub coords: Vec<f64>,
    /// Scale factor applied to Euclidean coordinates (divide projected coords
    /// by this to recover original values). Always 1.0 for curved spaces.
    pub scale: f64,
}

/// Project ambient-space points to 2D for visualization.
///
/// For curved spaces, uses geometry-appropriate projections:
/// - Hyperbolic (k < 0): Poincaré disk model
/// - Spherical (k > 0): Configurable projection method
/// - Euclidean (k = 0): First two coordinates, rescaled to unit range
pub fn project_to_2d(
    points: &[f64],
    n_points: usize,
    ambient_dim: usize,
    curvature: f64,
    projection: SphericalProjection,
) -> Projection2D {
    let mut projected = vec![0.0; n_points * 2];
    let mut scale = 1.0;

    if curvature > 0.0 {
        // Spherical: first align centroid to south pole
        let aligned = align_sphere_to_centroid(points, n_points, ambient_dim);
        let r = 1.0 / curvature.sqrt();

        match projection {
            SphericalProjection::Stereographic => {
                for i in 0..n_points {
                    let offset = i * ambient_dim;
                    let z = aligned[offset]; // pole axis = 0
                    let mut denom = r - z;
                    if denom.abs() < 1e-12 {
                        denom = 1e-12;
                    }
                    let scale = r / denom;
                    let x = scale * aligned[offset + 1] / r;
                    let y = if ambient_dim > 2 {
                        scale * aligned[offset + 2] / r
                    } else {
                        0.0
                    };
                    projected[i * 2] = x;
                    projected[i * 2 + 1] = y;
                }
                // Rescale to fit in unit circle
                let max_dist = (0..n_points)
                    .map(|i| {
                        let x = projected[i * 2];
                        let y = projected[i * 2 + 1];
                        (x * x + y * y).sqrt()
                    })
                    .fold(0.0f64, f64::max);
                if max_dist > 0.0 {
                    for v in &mut projected {
                        *v /= max_dist;
                    }
                }
            }
            SphericalProjection::AzimuthalEquidistant => {
                let r = 1.0 / curvature.sqrt();
                for i in 0..n_points {
                    let offset = i * ambient_dim;
                    let x0 = aligned[offset];
                    let x1 = aligned[offset + 1];
                    let x2 = if ambient_dim > 2 {
                        aligned[offset + 2]
                    } else {
                        0.0
                    };

                    let theta = (-x0 / r).clamp(-1.0, 1.0).acos();
                    let phi = x2.atan2(x1);
                    let sf = 1.0 / (PI * r);
                    projected[i * 2] = theta * phi.cos() * sf * r;
                    projected[i * 2 + 1] = theta * phi.sin() * sf * r;
                }
            }
            SphericalProjection::Orthographic => {
                let r = 1.0 / curvature.sqrt();
                for i in 0..n_points {
                    let offset = i * ambient_dim;
                    projected[i * 2] = aligned[offset + 1] / r;
                    projected[i * 2 + 1] = if ambient_dim > 2 {
                        aligned[offset + 2] / r
                    } else {
                        0.0
                    };
                }
            }
        }
    } else if curvature < 0.0 {
        // Hyperbolic: Poincaré disk
        let r = 1.0 / (-curvature).sqrt();
        for i in 0..n_points {
            let offset = i * ambient_dim;
            let x0 = points[offset];
            let denom = x0 + r;
            if denom.abs() > 1e-10 {
                projected[i * 2] = points[offset + 1] / denom;
                projected[i * 2 + 1] = if ambient_dim > 2 {
                    points[offset + 2] / denom
                } else {
                    0.0
                };
            }
        }
    } else {
        // Euclidean: first 2 coordinates, rescale to unit circle
        for i in 0..n_points {
            projected[i * 2] = points[i * ambient_dim];
            projected[i * 2 + 1] = if ambient_dim > 1 {
                points[i * ambient_dim + 1]
            } else {
                0.0
            };
        }
        let max_dist = (0..n_points)
            .map(|i| {
                let x = projected[i * 2];
                let y = projected[i * 2 + 1];
                (x * x + y * y).sqrt()
            })
            .fold(0.0f64, f64::max);
        if max_dist > 0.0 {
            scale = max_dist;
            for v in &mut projected {
                *v /= max_dist;
            }
        }
    }

    Projection2D {
        coords: projected,
        scale,
    }
}

/// Rotate spherical points so the data centroid aligns with -e_0 (south pole).
fn align_sphere_to_centroid(points: &[f64], n_points: usize, ambient_dim: usize) -> Vec<f64> {
    // Compute centroid
    let mut centroid = vec![0.0; ambient_dim];
    for i in 0..n_points {
        for d in 0..ambient_dim {
            centroid[d] += points[i * ambient_dim + d];
        }
    }
    for d in centroid.iter_mut().take(ambient_dim) {
        *d /= n_points as f64;
    }

    let centroid_norm: f64 = centroid.iter().map(|&c| c * c).sum::<f64>().sqrt();

    if centroid_norm < 1e-10 {
        // Data is roughly uniform — no preferred direction
        return points.to_vec();
    }

    // Normalize centroid
    let c: Vec<f64> = centroid.iter().map(|&v| v / centroid_norm).collect();

    // Householder reflection: H maps c -> -e_0
    // v = c + e_0
    let mut v = c.clone();
    v[0] += 1.0;

    let v_dot: f64 = v.iter().map(|&x| x * x).sum();
    if v_dot < 1e-15 {
        // c ≈ -e_0 already
        return points.to_vec();
    }

    // Apply Householder: H @ x = x - 2 * v * (v^T x) / (v^T v)
    let mut result = vec![0.0; n_points * ambient_dim];
    for i in 0..n_points {
        let vt_x: f64 = (0..ambient_dim)
            .map(|d| v[d] * points[i * ambient_dim + d])
            .sum();
        let coeff = 2.0 * vt_x / v_dot;
        for d in 0..ambient_dim {
            result[i * ambient_dim + d] = points[i * ambient_dim + d] - coeff * v[d];
        }
    }
    result
}

/// Tab10 color palette (matching matplotlib/d3), returns (r, g, b) in 0-255.
pub fn tab10_color(label: u32) -> (u8, u8, u8) {
    match label % 10 {
        0 => (31, 119, 180),  // blue
        1 => (255, 127, 14),  // orange
        2 => (44, 160, 44),   // green
        3 => (214, 39, 40),   // red
        4 => (148, 103, 189), // purple
        5 => (140, 86, 75),   // brown
        6 => (227, 119, 194), // pink
        7 => (127, 127, 127), // gray
        8 => (188, 189, 34),  // olive
        9 => (23, 190, 207),  // cyan
        _ => unreachable!(),
    }
}
