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
use crate::metrics::gyration_radius;

/// The three radii, for one embedding.
///
/// `f64::NAN` means absent, matching
/// [`crate::metrics::MetricValues`]; the accessors collapse absent and
/// non-finite to `None`, which is what every consumer wants and why the fields
/// are private.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpreadDiagnostics {
    r_max: f64,
    r_rms: f64,
    r_gyration: f64,
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
        r_max: f64::NAN,
        r_rms: f64::NAN,
        r_gyration: f64::NAN,
    };

    pub fn compute(c: &EmbeddingContext<'_>) -> Self {
        let origin = c.origin_dist();
        let (r_max, r_rms) = if origin.is_empty() {
            (0.0, 0.0)
        } else {
            (
                origin.iter().cloned().fold(0.0_f64, f64::max),
                (origin.iter().map(|d| d * d).sum::<f64>() / origin.len() as f64).sqrt(),
            )
        };
        Self {
            r_max,
            r_rms,
            r_gyration: gyration_radius(c.manifold_dist(), c.n),
        }
    }

    /// Component-wise mean over a non-empty slice of samples.
    ///
    /// Arithmetic on the raw values, so a seed that failed to measure
    /// propagates its NaN rather than being averaged out of existence.
    pub fn mean(samples: &[SpreadDiagnostics]) -> SpreadDiagnostics {
        let n = samples.len() as f64;
        let avg = |f: fn(&SpreadDiagnostics) -> f64| samples.iter().map(f).sum::<f64>() / n;
        Self {
            r_max: avg(|s| s.r_max),
            r_rms: avg(|s| s.r_rms),
            r_gyration: avg(|s| s.r_gyration),
        }
    }

    /// Largest geodesic distance from the manifold origin.
    pub fn r_max(&self) -> Option<f64> {
        self.r_max.is_finite().then_some(self.r_max)
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
    pub fn r_rms(&self) -> Option<f64> {
        self.r_rms.is_finite().then_some(self.r_rms)
    }

    /// Origin-free spread: the radius of gyration over the pairwise geodesics.
    ///
    /// The gauge to read κ against, precisely because it needs no pole. See
    /// [`crate::metrics::gyration_radius`].
    pub fn r_gyration(&self) -> Option<f64> {
        self.r_gyration.is_finite().then_some(self.r_gyration)
    }
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
    use super::*;
    use serde::de::{IgnoredAny, MapAccess, Visitor};
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::borrow::Cow;

    /// Column order, which is JSONL key order.
    const COLUMNS: [&str; 3] = ["r_max", "r_rms", "r_gyration"];

    impl SpreadDiagnostics {
        fn raw(&self, column: &str) -> f64 {
            match column {
                "r_max" => self.r_max,
                "r_rms" => self.r_rms,
                _ => self.r_gyration,
            }
        }

        fn set(&mut self, column: &str, v: f64) {
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
                let v = self.raw(column);
                map.serialize_entry(column, &v.is_finite().then_some(v))?;
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
                            // `Option<f64>`, not `f64`: a diverged trial writes
                            // `"r_rms": null` and that must not error.
                            if let Some(v) = map.next_value::<Option<f64>>()? {
                                out.set(&key, v);
                            }
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
