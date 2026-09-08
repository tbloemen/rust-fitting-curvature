//! How far an embedding spreads — the diagnostics `κ = |K|·R²` is gauged
//! against.
//!
//! **These are not quality metrics**, and they used to live in the metric
//! registry, where they answered five of its nine questions with "not
//! applicable". Nothing here is scored against the high-dimensional data, none
//! is ever optimised, and "which direction is better" has no answer — they are
//! three measurements of a configuration's own extent, reported beside the
//! metrics only because the same embedding produces both. Keeping them out of
//! [`crate::metrics`] is what lets `Space` be a genuine
//! projected-or-manifold binary and `Family` hold only real preference
//! families.

use crate::context::EmbeddingContext;
use crate::metrics::{values_mean_of, MetricValue};

/// The three radii, for one embedding.
///
/// Each is a [`MetricValue`], the same element type
/// [`crate::metrics::MetricValues`] holds — so "no reading" carries its reason
/// here too, and `MISSING == MISSING` is true rather than false. The fields are
/// private because the accessors are what callers want.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpreadDiagnostics {
    r_max: MetricValue,
    r_rms: MetricValue,
    r_gyration: MetricValue,
}

impl Default for SpreadDiagnostics {
    fn default() -> Self {
        Self::MISSING
    }
}

impl SpreadDiagnostics {
    /// Nothing measured — what `--mode scan` writes, and what a results line
    /// predating a column reads back as.
    pub const MISSING: Self = Self {
        r_max: MetricValue::Absent,
        r_rms: MetricValue::Absent,
        r_gyration: MetricValue::Absent,
    };

    /// Measure the configuration's extent.
    ///
    /// Gated the same way the metrics are: on a diverged embedding these are
    /// [`MetricValue::Diverged`] rather than numbers. `r_max` is the reason the
    /// gate matters here — it is a `fold(0.0, f64::max)`, and `f64::max`
    /// *ignores* NaN, so a blown-up embedding used to report `r_max: 0.0`
    /// beside `r_rms: null`: a measured-looking zero from garbage.
    pub fn compute(c: &EmbeddingContext<'_>) -> Self {
        if !c.spread_is_finite() {
            return Self {
                r_max: MetricValue::Diverged,
                r_rms: MetricValue::Diverged,
                r_gyration: MetricValue::Diverged,
            };
        }
        let origin = c.origin_dist();
        let (r_max, r_rms) = if origin.is_empty() {
            (0.0, 0.0)
        } else {
            (
                origin.iter().copied().fold(0.0_f64, f64::max),
                (origin.iter().map(|d| d * d).sum::<f64>() / origin.len() as f64).sqrt(),
            )
        };
        Self {
            r_max: MetricValue::measured(r_max),
            r_rms: MetricValue::measured(r_rms),
            r_gyration: MetricValue::measured(gyration_radius(c.manifold_dist(), c.n)),
        }
    }

    /// Component-wise mean over a non-empty slice of samples, with the same
    /// precedence [`crate::metrics::MetricValues::mean`] uses: a seed that
    /// diverged shows in the aggregate rather than being averaged away.
    #[must_use]
    pub fn mean(samples: &[SpreadDiagnostics]) -> SpreadDiagnostics {
        let n = samples.len() as f64;
        let avg =
            |f: fn(&SpreadDiagnostics) -> MetricValue| values_mean_of(samples.iter().map(f), n);
        Self {
            r_max: avg(|s| s.r_max),
            r_rms: avg(|s| s.r_rms),
            r_gyration: avg(|s| s.r_gyration),
        }
    }

    /// Largest geodesic distance from the manifold origin.
    #[must_use]
    pub fn r_max(&self) -> Option<f64> {
        self.r_max.value()
    }

    /// RMS geodesic distance from the manifold origin — the `R_rms` of
    /// `@eq:kappa`.
    ///
    /// Meaningful on the hyperboloid, which is re-centred every iteration, and
    /// **wrong on the sphere**: `Sphere::center` is a no-op and
    /// `lift_pca_to_manifold` writes the constrained coordinate to the last
    /// ambient slot while `Sphere::distances_from_origin` reads the first, so
    /// PCA init lands every point ~90° from the pole κ is gauged against. Use
    /// [`Self::r_gyration`] there.
    #[must_use]
    pub fn r_rms(&self) -> Option<f64> {
        self.r_rms.value()
    }

    /// Origin-free spread: the radius of gyration over the pairwise geodesics.
    ///
    /// The gauge to read κ against, precisely because it needs no pole. See
    /// [`gyration_radius`].
    #[must_use]
    pub fn r_gyration(&self) -> Option<f64> {
        self.r_gyration.value()
    }
}

/// Radius of gyration from a full `n × n` pairwise distance matrix — the
/// embedding's spread, measured without an origin.
///
/// `R_g² = (1 / 2n²) ΣᵢΣⱼ d²ᵢⱼ`, which in flat space is *exactly* the mean
/// squared distance to the centroid. The `2n²` divisor is load-bearing: `dist`
/// is the full matrix, so every pair appears twice, and an `n(n−1)` divisor (or
/// a missing factor of two) still yields plausible-looking numbers while
/// quietly breaking the identity. `test_gyration_matches_centroid_rms` pins it.
///
/// This exists because [`SpreadDiagnostics::r_max`]/[`SpreadDiagnostics::r_rms`]
/// are measured from a *fixed* pole. That is meaningful on the hyperboloid,
/// which `Hyperboloid::center` re-centres on the origin every iteration, and
/// vacuous on the sphere: `Sphere::center` is a no-op, and
/// `lift_pca_to_manifold` writes the constrained coordinate to the last ambient
/// slot while `Sphere::distances_from_origin` reads the first — so PCA init
/// lands every point ~90° from the pole κ is gauged against, and `|K|·r_rms²`
/// sits at `π²/4` however curved the space actually is.
#[must_use]
pub fn gyration_radius(dist: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = dist.iter().map(|d| d * d).sum();
    (sum_sq / (2.0 * (n * n) as f64)).sqrt()
}

// ─── Wire format ─────────────────────────────────────────────────────────────
//
// Three flat columns, so `#[serde(flatten)]` puts them at the top level of a
// trial record exactly where the individual `Option<f64>` fields used to sit.
// Mirrors `MetricValues`: non-finite writes as `null`, `null` and missing both
// read back as absent, and unknown keys are ignored — which is required, since
// a flatten target is handed every key the named fields did not claim.

#[cfg(feature = "serde")]
mod wire {
    use super::{MetricValue, SpreadDiagnostics};
    use serde::de::{IgnoredAny, MapAccess, Visitor};
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::borrow::Cow;

    /// Column order, which is JSONL key order.
    const COLUMNS: [&str; 3] = ["r_max", "r_rms", "r_gyration"];

    impl SpreadDiagnostics {
        fn column(&self, column: &str) -> MetricValue {
            match column {
                "r_max" => self.r_max,
                "r_rms" => self.r_rms,
                _ => self.r_gyration,
            }
        }

        fn set(&mut self, column: &str, v: MetricValue) {
            match column {
                "r_max" => self.r_max = v,
                "r_rms" => self.r_rms = v,
                _ => self.r_gyration = v,
            }
        }
    }

    impl Serialize for SpreadDiagnostics {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let mut map = s.serialize_map(Some(COLUMNS.len()))?;
            for column in COLUMNS {
                // Every reason for not having a number writes `null`, as with
                // the metric block: the format carries one absent, not three.
                map.serialize_entry(column, &self.column(column).value())?;
            }
            map.end()
        }
    }

    impl<'de> Deserialize<'de> for SpreadDiagnostics {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = SpreadDiagnostics;

                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("a map carrying the spread diagnostic columns")
                }

                fn visit_map<A: MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> Result<SpreadDiagnostics, A::Error> {
                    let mut out = SpreadDiagnostics::MISSING;
                    while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                        if COLUMNS.contains(&key.as_ref()) {
                            // `Option<f64>`, not `f64`: a trial that did not
                            // measure writes `"r_rms": null`, and that must not
                            // error. It reads back as `Absent`.
                            let v = map.next_value::<Option<f64>>()?;
                            out.set(&key, v.map_or(MetricValue::Absent, MetricValue::measured));
                        } else {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                    Ok(out)
                }
            }
            d.deserialize_map(V)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::compute_euclidean_distance_matrix;

    /// The flat-space identity the `2n²` divisor exists for: in Euclidean space
    /// the gyration radius *is* the RMS distance to the centroid. A wrong
    /// divisor scales the result by a constant, which no eyeball check on a κ
    /// column would catch.
    #[test]
    fn test_gyration_matches_centroid_rms() {
        const N: usize = 37;
        const D: usize = 3;

        // Deterministic, spread over a few orders of magnitude so a constant
        // factor cannot hide in the noise.
        let mut points = vec![0.0f64; N * D];
        for i in 0..N {
            for d in 0..D {
                let t = (i * D + d) as f64;
                points[i * D + d] = (t * 0.7).sin() * (1.0 + t * 0.31);
            }
        }

        let dist = compute_euclidean_distance_matrix(&points, N, D);
        let got = gyration_radius(&dist, N);

        // Direct definition: RMS distance from the centroid.
        let mut centroid = [0.0f64; D];
        for i in 0..N {
            for (d, c) in centroid.iter_mut().enumerate() {
                *c += points[i * D + d];
            }
        }
        for c in &mut centroid {
            *c /= N as f64;
        }
        let want = {
            let sum_sq: f64 = (0..N)
                .map(|i| {
                    (0..D)
                        .map(|d| (points[i * D + d] - centroid[d]).powi(2))
                        .sum::<f64>()
                })
                .sum();
            (sum_sq / N as f64).sqrt()
        };

        assert!(
            (got - want).abs() < 1e-12 * want.max(1.0),
            "gyration radius {got} != centroid RMS {want}"
        );
    }

    /// A configuration collapsed to a point has zero spread, whatever the
    /// manifold's own radius is — the case `r_rms` reports as the `π²/4` floor
    /// on the sphere rather than as zero.
    #[test]
    fn test_gyration_of_collapsed_configuration_is_zero() {
        let dist = vec![0.0; 16 * 16];
        assert_eq!(gyration_radius(&dist, 16), 0.0);
    }
}
