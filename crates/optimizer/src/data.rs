use fitting_core::synthetic_data::DataPoints;

#[derive(Debug, Clone)]
pub struct Dataset {
    pub x: Vec<f64>,
    pub labels: Vec<u32>,
    pub n_points: usize,
    pub n_features: usize,
    /// Pre-computed pairwise distance matrix (flat n × n, row-major).
    /// Non-empty for datasets like WordNet where intrinsic distances drive
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
        use fitting_core::synthetic_data::*;
        let sd = match name {
            "sphere" => generate_hd_sphere(n_samples, 10, seed),
            "antipodal_clusters" => generate_hd_antipodal_clusters(n_samples, 10, seed),
            "tree" => generate_hd_tree(n_samples, 10, seed),
            "hyperbolic_shells" => generate_hd_hyperbolic_shells(n_samples, 10, seed),
            _ => return Err(format!(
                "Unknown dataset '{name}'.\n  \
                 Real: mnist, fashion_mnist, pbmc, wordnet_mammals\n  \
                 Synthetic: sphere, antipodal_clusters, tree, hyperbolic_shells"
            )),
        };
        Ok(sd.into())
    }

    pub fn load_mnist(path: &str, n_samples: usize) -> Result<Self, String> {
        fitting_core::data::load_mnist(path, n_samples).map(|sd| sd.into())
    }

    pub fn load_fashion_mnist(path: &str, n_samples: usize) -> Result<Self, String> {
        fitting_core::data::load_fashion_mnist(path, n_samples).map(|sd| sd.into())
    }

    pub fn load_wordnet_mammals(path: &str, n_samples: usize) -> Result<Self, String> {
        fitting_core::data::load_wordnet_mammals(path, n_samples).map(|sd| sd.into())
    }

    pub fn load_pbmc(path: &str, n_samples: usize) -> Result<Self, String> {
        fitting_core::data::load_pbmc(path, n_samples).map(|sd| sd.into())
    }
}
