//! Numeric conversions whose preconditions are stated once here instead of
//! being re-argued at every call site.
//!
//! Every function in this module is a plain `as` cast. That is deliberate: the
//! point is not to change what the arithmetic does — the sweep results in
//! `results/` have to stay reproducible — but to give each conversion a name,
//! so that `clippy::pedantic`'s cast lints can stay denied everywhere else in
//! the workspace and the handful of genuine suppressions all live in one file.

/// Widen a count to `f64`.
///
/// Exact for values below 2^53. Every `usize` this codebase converts is a
/// count — points, dimensions, iterations, neighbours, histogram bins — and
/// none of them come within many orders of magnitude of that bound, so the
/// "precision loss" the lint warns about cannot occur here.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "counts in this codebase are far below 2^53, so the widening is exact"
)]
pub fn count_to_f64(n: usize) -> f64 {
    n as f64
}

/// Widen a `u64` count to `f64`. Same reasoning as [`count_to_f64`].
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "counts in this codebase are far below 2^53, so the widening is exact"
)]
pub fn u64_to_f64(n: u64) -> f64 {
    n as f64
}

/// Truncate a `f64` toward zero into a `usize`.
///
/// This is Rust's float-to-integer `as`, which has been saturating since 1.45:
/// NaN maps to 0, anything below 0 maps to 0, anything above `usize::MAX` maps
/// to `usize::MAX`, and everything else truncates toward zero. Callers that
/// want a different rounding say so at the call site (`x.round()`, `x.ceil()`).
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "float-to-int `as` saturates; truncation toward zero is the intent"
)]
pub fn to_usize(x: f64) -> usize {
    x as usize
}

/// Truncate a `f64` toward zero into an `i32`, saturating at the bounds.
/// Used for pixel coordinates, where the drawing backend clips anyway.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "float-to-int `as` saturates; truncation toward zero is the intent"
)]
pub fn to_i32(x: f64) -> i32 {
    x as i32
}

/// Truncate a `f64` toward zero into an `i64`, saturating at the bounds.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "float-to-int `as` saturates; truncation toward zero is the intent"
)]
pub fn to_i64(x: f64) -> i64 {
    x as i64
}

/// Truncate a `f64` toward zero into a `u32`, saturating at the bounds.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "float-to-int `as` saturates; truncation toward zero is the intent"
)]
pub fn to_u32(x: f64) -> u32 {
    x as u32
}

#[cfg(test)]
mod tests {
    use super::{count_to_f64, to_i32, to_u32, to_usize, u64_to_f64};

    #[test]
    fn counts_widen_exactly() {
        assert!((count_to_f64(0) - 0.0).abs() < f64::EPSILON);
        assert!((count_to_f64(1) - 1.0).abs() < f64::EPSILON);
        assert!((count_to_f64(1_000_000) - 1e6).abs() < f64::EPSILON);
        // 2^53 is the last integer f64 represents exactly; it round-trips.
        assert_eq!(to_usize(count_to_f64(1 << 53)), 1 << 53);
        assert!((u64_to_f64(1 << 53) - count_to_f64(1 << 53)).abs() < f64::EPSILON);
    }

    #[test]
    fn to_usize_truncates_toward_zero() {
        assert_eq!(to_usize(0.0), 0);
        assert_eq!(to_usize(7.0), 7);
        assert_eq!(to_usize(7.999_999), 7);
        assert_eq!(to_usize(0.999_999), 0);
    }

    #[test]
    fn to_usize_saturates_instead_of_wrapping() {
        assert_eq!(to_usize(-0.5), 0);
        assert_eq!(to_usize(-1e300), 0);
        assert_eq!(to_usize(f64::NAN), 0);
        assert_eq!(to_usize(f64::INFINITY), usize::MAX);
        assert_eq!(to_usize(f64::NEG_INFINITY), 0);
        assert_eq!(to_usize(1e300), usize::MAX);
    }

    #[test]
    fn signed_and_narrow_targets_saturate_too() {
        assert_eq!(to_i32(-3.7), -3);
        assert_eq!(to_i32(1e300), i32::MAX);
        assert_eq!(to_i32(-1e300), i32::MIN);
        assert_eq!(to_i32(f64::NAN), 0);
        assert_eq!(to_u32(-1.0), 0);
        assert_eq!(to_u32(1e300), u32::MAX);
    }
}
