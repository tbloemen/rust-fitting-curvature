use std::fs::File;
use std::io::{BufReader, Read};

#[derive(Debug, Clone)]
pub struct Dataset {
    pub x: Vec<f64>,
    pub labels: Vec<u32>,
    pub n_points: usize,
    pub n_features: usize,
}

impl Dataset {
    pub fn load_mnist(path: &str, n_samples: usize) -> Result<Self, String> {
        let images_path = format!("{}/train-images-idx3-ubyte", path);
        let labels_path = format!("{}/train-labels-idx1-ubyte", path);

        let images = Self::read_idx3_ubyte(&images_path)?;
        let labels = Self::read_idx1_ubyte(&labels_path)?;

        let n_features = 28 * 28;
        let total_available = images.len() / n_features;
        let actual_samples = n_samples.min(total_available);

        let mut x = Vec::with_capacity(actual_samples * n_features);
        for i in 0..actual_samples {
            let start = i * n_features;
            let end = start + n_features;
            for &pixel in &images[start..end] {
                x.push(pixel as f64 / 255.0);
            }
        }

        let labels: Vec<u32> = labels[..actual_samples].iter().map(|&l| l as u32).collect();

        Ok(Self {
            x,
            labels,
            n_points: actual_samples,
            n_features,
        })
    }

    fn read_idx3_ubyte(path: &str) -> Result<Vec<u8>, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read magic: {}", e))?;

        let magic_num = u32::from_be_bytes(magic);
        if magic_num != 2051 {
            return Err(format!("Invalid magic number for idx3: {}", magic_num));
        }

        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read num_images: {}", e))?;
        let num_images = u32::from_be_bytes(buf) as usize;

        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read num_rows: {}", e))?;
        let _num_rows = u32::from_be_bytes(buf) as usize;

        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read num_cols: {}", e))?;
        let _num_cols = u32::from_be_bytes(buf) as usize;

        let mut data = vec![0u8; num_images * 28 * 28];
        reader
            .read_exact(&mut data)
            .map_err(|e| format!("Failed to read image data: {}", e))?;

        Ok(data)
    }

    fn read_idx1_ubyte(path: &str) -> Result<Vec<u8>, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read magic: {}", e))?;

        let magic_num = u32::from_be_bytes(magic);
        if magic_num != 2049 {
            return Err(format!("Invalid magic number for idx1: {}", magic_num));
        }

        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read num_items: {}", e))?;
        let num_items = u32::from_be_bytes(buf) as usize;

        let mut data = vec![0u8; num_items];
        reader
            .read_exact(&mut data)
            .map_err(|e| format!("Failed to read label data: {}", e))?;

        Ok(data)
    }

    pub fn load_synthetic(name: &str, n_samples: usize, seed: u64) -> Self {
        match name {
            "gaussian_blob" => Self::synthetic_gaussian_blob(n_samples, seed),
            "concentric_circles" => Self::synthetic_concentric_circles(n_samples, seed),
            "tree" => Self::synthetic_tree(n_samples, seed),
            "grid" => Self::synthetic_grid(n_samples, seed),
            _ => panic!("Unknown synthetic dataset: {}. Options: gaussian_blob, concentric_circles, tree, grid", name),
        }
    }

    fn synthetic_gaussian_blob(n_samples: usize, seed: u64) -> Self {
        let mut rng = fitting_core::synthetic_data::Rng::new(seed);
        let n_features: usize = 10;
        let mut x = Vec::with_capacity(n_samples * n_features);
        let mut radii = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let mut norm_sq = 0.0_f64;
            for _ in 0..n_features {
                let val = rng.normal();
                norm_sq += val * val;
                x.push(val);
            }
            radii.push(norm_sq.sqrt());
        }

        let mut sorted_radii = radii.clone();
        sorted_radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted_radii[n_samples / 2];

        let labels: Vec<u32> = radii
            .iter()
            .map(|&r| if r >= median { 1 } else { 0 })
            .collect();

        Self {
            x,
            labels,
            n_points: n_samples,
            n_features,
        }
    }

    fn synthetic_concentric_circles(n_samples: usize, seed: u64) -> Self {
        let mut rng = fitting_core::synthetic_data::Rng::new(seed);
        let n_features: usize = 10;
        let mut x = Vec::with_capacity(n_samples * n_features);
        let mut labels = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let label = i % 2;
            let mut r = if label == 0 { 1.0 } else { 2.0 };
            r += rng.normal() * 0.05;
            let theta = rng.uniform() * 2.0 * std::f64::consts::PI;

            x.push(r * theta.cos());
            x.push(r * theta.sin());
            for _ in 2..n_features {
                x.push(rng.normal() * 0.1);
            }
            labels.push(label as u32);
        }

        Self {
            x,
            labels,
            n_points: n_samples,
            n_features,
        }
    }

    fn synthetic_tree(n_samples: usize, seed: u64) -> Self {
        use std::f64::consts::PI;
        let mut rng = fitting_core::synthetic_data::Rng::new(seed);
        let n_features: usize = 10;
        let mut x = Vec::with_capacity(n_samples * n_features);
        let mut labels = Vec::with_capacity(n_samples);

        let n_levels: usize = 4;
        let branching: usize = 2;
        let points_per_leaf =
            ((n_samples as f64) / ((branching as f64).powi(n_levels as i32))) as usize;

        fn generate_tree(
            x: &mut Vec<f64>,
            labels: &mut Vec<u32>,
            rng: &mut fitting_core::synthetic_data::Rng,
            center: (f64, f64),
            level: usize,
            max_level: usize,
            n_features: usize,
            points_per_leaf: usize,
            branch: usize,
            branching: usize,
        ) {
            if level >= max_level {
                let ppl = points_per_leaf.max(1);
                for _ in 0..ppl {
                    x.push(center.0 + rng.normal() * 0.3);
                    x.push(center.1 + rng.normal() * 0.3);
                    for _ in 2..n_features {
                        x.push(rng.normal() * 0.1);
                    }
                    labels.push(branch as u32);
                }
                return;
            }

            let angle_offset = if branch % 2 == 0 { 0.0 } else { PI / 6.0 };
            let new_dist = 1.5_f64.powi((max_level - level) as i32) * 0.5;
            for b in 0..branching {
                let angle = angle_offset + (b as f64 - 0.5) * PI / 3.0;
                let new_center = (
                    center.0 + new_dist * angle.cos(),
                    center.1 + new_dist * angle.sin(),
                );
                let new_branch = branch * branching + b;
                generate_tree(
                    x,
                    labels,
                    rng,
                    new_center,
                    level + 1,
                    max_level,
                    n_features,
                    points_per_leaf,
                    new_branch,
                    branching,
                );
            }
        }

        generate_tree(
            &mut x,
            &mut labels,
            &mut rng,
            (0.0, 0.0),
            0,
            n_levels,
            n_features,
            points_per_leaf,
            0,
            branching,
        );

        let actual = x.len() / n_features;
        Self {
            x,
            labels,
            n_points: actual,
            n_features,
        }
    }

    fn synthetic_grid(n_samples: usize, seed: u64) -> Self {
        let mut rng = fitting_core::synthetic_data::Rng::new(seed);
        let n_features: usize = 10;
        let mut x = Vec::with_capacity(n_samples * n_features);
        let mut labels = Vec::with_capacity(n_samples);

        let grid_size = ((n_samples as f64).sqrt() as usize).max(4);
        let cells = grid_size * grid_size;

        for i in 0..n_samples {
            let cell = i % cells;
            let cx = (cell % grid_size) as f64 - (grid_size as f64) / 2.0;
            let cy = (cell / grid_size) as f64 - (grid_size as f64) / 2.0;

            x.push(cx + rng.normal() * 0.2);
            x.push(cy + rng.normal() * 0.2);
            for _ in 2..n_features {
                x.push(rng.normal() * 0.1);
            }
            labels.push((cell as u32) % 4);
        }

        Self {
            x,
            labels,
            n_points: n_samples,
            n_features,
        }
    }
}
