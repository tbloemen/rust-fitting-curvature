use fitting_core::synthetic_data::SyntheticData;

#[derive(Debug, Clone)]
pub struct Dataset {
    pub x: Vec<f64>,
    pub labels: Vec<u32>,
    pub n_points: usize,
    pub n_features: usize,
}

impl From<SyntheticData> for Dataset {
    fn from(sd: SyntheticData) -> Self {
        Self {
            x: sd.x,
            labels: sd.labels,
            n_points: sd.n_points,
            n_features: sd.ambient_dim,
        }
    }
}

impl Dataset {
    pub fn load_synthetic(name: &str, n_samples: usize, seed: u64) -> Self {
        use fitting_core::synthetic_data::*;
        let sd = match name {
            "sphere" => generate_hd_sphere(n_samples, 10, seed),
            "antipodal_clusters" => generate_hd_antipodal_clusters(n_samples, 10, seed),
            "tree" => generate_hd_tree(n_samples, 10, seed),
            "hyperbolic_shells" => generate_hd_hyperbolic_shells(n_samples, 10, seed),
            _ => panic!(
                "Unknown synthetic dataset: {name}. Options: sphere, antipodal_clusters, tree, hyperbolic_shells"
            ),
        };
        sd.into()
    }

    pub fn load_mnist(path: &str, n_samples: usize) -> Result<Self, String> {
        fitting_core::data::load_mnist(path, n_samples).map(|sd| sd.into())
    }
}
