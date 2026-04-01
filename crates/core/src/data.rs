//! Data loading utilities for non-synthetic datasets.
//!
//! Note: filesystem access is not available in WASM; these functions
//! are only compiled for native targets.

use crate::synthetic_data::SyntheticData;

#[cfg(not(target_arch = "wasm32"))]
use std::{
    fs::File,
    io::{BufReader, Read},
};

/// Load MNIST training images and labels from the IDX binary format.
///
/// `path` is the directory containing `train-images-idx3-ubyte` and
/// `train-labels-idx1-ubyte`. Returns a `SyntheticData` with pixel values
/// normalised to [0, 1] and `distances` left empty (not precomputed).
#[cfg(not(target_arch = "wasm32"))]
pub fn load_mnist(path: &str, n_samples: usize) -> Result<SyntheticData, String> {
    let images_path = format!("{}/train-images-idx3-ubyte", path);
    let labels_path = format!("{}/train-labels-idx1-ubyte", path);

    let images = read_idx3_ubyte(&images_path)?;
    let labels_raw = read_idx1_ubyte(&labels_path)?;

    let n_features = 28 * 28;
    let actual_samples = n_samples.min(images.len() / n_features);

    let x = images[..actual_samples * n_features]
        .chunks(n_features)
        .flat_map(|row| row.iter().map(|&p| p as f64 / 255.0))
        .collect();
    let labels = labels_raw[..actual_samples]
        .iter()
        .map(|&l| l as u32)
        .collect();

    Ok(SyntheticData {
        x,
        labels,
        n_points: actual_samples,
        ambient_dim: n_features,
        distances: Vec::new(), // not precomputed; callers compute as needed
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn read_idx3_ubyte(path: &str) -> Result<Vec<u8>, String> {
    let mut reader =
        BufReader::new(File::open(path).map_err(|e| format!("Failed to open {path}: {e}"))?);
    let magic = read_u32(&mut reader)?;
    if magic != 2051 {
        return Err(format!("Invalid magic number for idx3: {magic}"));
    }
    let num_images = read_u32(&mut reader)? as usize;
    let _rows = read_u32(&mut reader)?;
    let _cols = read_u32(&mut reader)?;
    let mut data = vec![0u8; num_images * 28 * 28];
    reader
        .read_exact(&mut data)
        .map_err(|e| format!("Failed to read image data: {e}"))?;
    Ok(data)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_idx1_ubyte(path: &str) -> Result<Vec<u8>, String> {
    let mut reader =
        BufReader::new(File::open(path).map_err(|e| format!("Failed to open {path}: {e}"))?);
    let magic = read_u32(&mut reader)?;
    if magic != 2049 {
        return Err(format!("Invalid magic number for idx1: {magic}"));
    }
    let num_items = read_u32(&mut reader)? as usize;
    let mut data = vec![0u8; num_items];
    reader
        .read_exact(&mut data)
        .map_err(|e| format!("Failed to read label data: {e}"))?;
    Ok(data)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_u32(reader: &mut impl Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read u32: {e}"))?;
    Ok(u32::from_be_bytes(buf))
}
