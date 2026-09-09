use fitting_core::synthetic_data::DataPoints;

#[derive(Debug, Clone)]
pub struct Dataset {
    pub x: Vec<f64>,
    pub labels: Vec<u32>,
    pub n_points: usize,
    pub n_features: usize,
    /// Pre-computed pairwise distance matrix (flat n × n, row-major).
    /// Non-empty for datasets like `WordNet` where intrinsic distances drive
    /// affinities and evaluation instead of Euclidean distances in feature space.
    pub precomputed_distances: Vec<f64>,
}

impl From<DataPoints> for Dataset {
    fn from(sd: DataPoints) -> Self {
        Self {
            x: sd.x,
            labels: sd.labels,
            n_points: sd.n_points,
            n_features: sd.ambient_dim,
            precomputed_distances: sd.distances,
        }
    }
}

impl Dataset {
    pub fn load_synthetic(name: &str, n_samples: usize, seed: u64) -> Result<Self, String> {
        use fitting_core::synthetic_data::{
            generate_hd_antipodal_clusters, generate_hd_hyperbolic_shells, generate_hd_sphere,
            generate_hd_tree, generate_hd_uniform_grid, generate_matched_ball, generate_tree_graph,
            MATCHED_BALL_EXTENT,
        };
        let sd = match name {
            "sphere" => generate_hd_sphere(n_samples, 10, seed),
            "antipodal_clusters" => generate_hd_antipodal_clusters(n_samples, 10, seed),
            "tree" => generate_hd_tree(n_samples, 10, seed),
            "hyperbolic_shells" => generate_hd_hyperbolic_shells(n_samples, 10, seed),
            // Euclidean synthetic: a lattice in R^10, matching the ambient
            // dimension of the curved synthetics above.
            "grid" => generate_hd_uniform_grid(n_samples, 10, seed),

            // A real hierarchy: unweighted tree-metric distances, not a
            // hyperbolic point cloud shaped like tree levels (that is `tree`).
            "tree_graph" => generate_tree_graph(n_samples, 2, 3),

            // The matched family: one sampling scheme, three geometries. Tier A
            // is 2-D source into a 2-D target (curvature matching with no
            // dimension reduction); tier B shares a 9-D source across the three
            // so the 9 -> 2 compression is matched too.
            "ball2_euclidean" => {
                generate_matched_ball(n_samples, 2, 0.0, MATCHED_BALL_EXTENT, seed)
            }
            "ball2_spherical" => {
                generate_matched_ball(n_samples, 2, 1.0, MATCHED_BALL_EXTENT, seed)
            }
            "ball2_hyperbolic" => {
                generate_matched_ball(n_samples, 2, -1.0, MATCHED_BALL_EXTENT, seed)
            }
            "ball9_euclidean" => {
                generate_matched_ball(n_samples, 9, 0.0, MATCHED_BALL_EXTENT, seed)
            }
            "ball9_spherical" => {
                generate_matched_ball(n_samples, 9, 1.0, MATCHED_BALL_EXTENT, seed)
            }
            "ball9_hyperbolic" => {
                generate_matched_ball(n_samples, 9, -1.0, MATCHED_BALL_EXTENT, seed)
            }
            _ => {
                return Err(format!(
                    "Unknown dataset '{name}'.\n  \
                 Real: mnist, fashion_mnist, pbmc, wordnet_mammals\n  \
                 Synthetic: sphere, antipodal_clusters, tree, hyperbolic_shells, grid,\n  \
                 \u{20}          tree_graph, ball2_euclidean, ball2_spherical, ball2_hyperbolic,\n  \
                 \u{20}          ball9_euclidean, ball9_spherical, ball9_hyperbolic"
                ));
            }
        };
        Ok(sd.into())
    }

    pub fn load_mnist(path: &str, n_samples: usize) -> Result<Self, String> {
        fitting_core::data::load_mnist(path, n_samples).map(std::convert::Into::into)
    }

    pub fn load_fashion_mnist(path: &str, n_samples: usize) -> Result<Self, String> {
        fitting_core::data::load_fashion_mnist(path, n_samples).map(std::convert::Into::into)
    }

    pub fn load_wordnet_mammals(path: &str, n_samples: usize) -> Result<Self, String> {
        fitting_core::data::load_wordnet_mammals(path, n_samples).map(std::convert::Into::into)
    }

    pub fn load_pbmc(path: &str, n_samples: usize) -> Result<Self, String> {
        fitting_core::data::load_pbmc(path, n_samples).map(std::convert::Into::into)
    }
}

/// The curvature of the manifold a dataset's *source coordinates* live on, when
/// those coordinates are themselves a valid 2-D target embedding.
///
/// `--mode reference` scores that configuration through the trial metric
/// pipeline, which needs to know the manifold it sits on. Only the tier-A
/// matched balls qualify: everything else is either higher-dimensional (the
/// tier-B balls, `sphere`, `tree`, …), graph data with no coordinates
/// (`tree_graph`, `wordnet_mammals`), or a real dataset whose geometry is the
/// question rather than the given.
#[must_use]
pub fn source_curvature(name: &str) -> Option<f64> {
    match name {
        "ball2_euclidean" => Some(0.0),
        "ball2_spherical" => Some(1.0),
        "ball2_hyperbolic" => Some(-1.0),
        _ => None,
    }
}

/// The generator's ball radius, so a reference row is interpretable on its own.
#[must_use]
pub fn source_extent(name: &str) -> Option<f64> {
    source_curvature(name).map(|_| fitting_core::synthetic_data::MATCHED_BALL_EXTENT)
}
