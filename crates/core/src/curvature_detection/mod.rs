//! Constant-curvature fits of a pairwise distance matrix.
//!
//! - [`signature`] — Wilson et al. (2014) radius-of-curvature fit by a
//!   constant-curvature Gram matrix and its signature residual.
//! - [`reconstruct`] — coordinates from a fitted radius.  Each `Z(r)` the
//!   signature criterion scores is the Gram matrix of the model it tests
//!   for, so its retained eigen-block *is* an embedding; this turns a
//!   [`signature::WilsonFit`] into points that can be measured by the same
//!   DR-quality metrics a t-SNE embedding is.

pub mod reconstruct;
pub mod signature;

pub use reconstruct::{
    reconstruct_euclidean, reconstruct_hyperbolic, reconstruct_spherical, Reconstruction,
};
pub use signature::*;
