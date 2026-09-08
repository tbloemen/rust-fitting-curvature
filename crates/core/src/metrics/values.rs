//! One embedding's score on every metric.
//!
//! Replaces four parallel per-metric structs — the optimizer's `AllMetrics`,
//! its `TrialResult` columns, its `PriorTrialRecord`, and the analysis's
//! `TrialRecord` columns — with one positional array plus a name-keyed wire
//! format. The array is indexed by position in [`ALL`]; nothing outside this
//! file knows that.

use std::ops::Index;

use super::context::MetricContext;
use super::quality::{Metric, ALL};

/// Every metric's value for one embedding, positional in [`ALL`] order.
///
/// **`f64::NAN` is the single "absent" representation.** It covers a
/// label-aware metric on unlabelled data, a column a results file predates, a
/// column a retired metric left behind, and a diverged trial whose metric came
/// out non-finite — all four were separate shapes before (`f64`, missing key,
/// `Option::None`, `Some(NaN)`), and every consumer already collapsed them to
/// the same worst-case handling. [`MetricValues::get`] is that collapse; it is
/// what callers should use, and [`MetricValues::raw`] is only for serialisation
/// and averaging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricValues([f64; Metric::COUNT]);

impl Default for MetricValues {
    fn default() -> Self {
        Self::MISSING
    }
}

impl MetricValues {
    /// Nothing measured. This is what `--mode scan` writes — it builds a
    /// result without ever scoring one — and it serialises as an all-`null`
    /// block, exactly as the `Option<f64>` columns it replaces did.
    pub const MISSING: Self = Self([f64::NAN; Metric::COUNT]);

    /// Score every metric in [`ALL`]. The context derives each distance matrix
    /// at most once, however many metrics read it.
    pub fn compute(ctx: &MetricContext<'_>) -> Self {
        let mut out = Self::MISSING;
        for (slot, metric) in out.0.iter_mut().zip(ALL) {
            *slot = metric.compute(ctx);
        }
        out
    }

    /// The value, or `None` if it is absent or non-finite.
    ///
    /// Every caller that feeds a plot or a statistic wants this: `figures/exp5`
    /// filters on it, and `oriented_value` maps `None` to the worst case.
    /// Returning `Some(NAN)` instead would put NaN points on a chart.
    pub fn get(&self, m: Metric) -> Option<f64> {
        let v = self.0[m.index()];
        v.is_finite().then_some(v)
    }

    /// The stored value, NaN included. For serialisation and [`Self::mean`].
    pub fn raw(&self, m: Metric) -> f64 {
        self.0[m.index()]
    }

    pub fn set(&mut self, m: Metric, v: f64) {
        self.0[m.index()] = v;
    }

    /// Component-wise mean over a non-empty slice of samples.
    ///
    /// Deliberately arithmetic on `raw`: a NaN in any sample propagates to that
    /// component of the mean, so a seed that failed to score is visible rather
    /// than silently averaged out of existence.
    pub fn mean(samples: &[MetricValues]) -> MetricValues {
        let n = samples.len() as f64;
        let mut out = Self::MISSING;
        for (j, slot) in out.0.iter_mut().enumerate() {
            *slot = samples.iter().map(|s| s.0[j]).sum::<f64>() / n;
        }
        out
    }

    /// `(metric, value)` in [`ALL`] order, NaNs included.
    pub fn iter(&self) -> impl Iterator<Item = (Metric, f64)> + '_ {
        ALL.iter().copied().zip(self.0.iter().copied())
    }
}

impl Index<Metric> for MetricValues {
    type Output = f64;
    fn index(&self, m: Metric) -> &f64 {
        &self.0[m.index()]
    }
}

// ─── Wire format ─────────────────────────────────────────────────────────────
//
// A flat, name-keyed map, so `#[serde(flatten)]` puts the metric columns at the
// top level of a trial record exactly where the individual fields used to sit.
// Behind the `serde` feature: `fitting-core` is otherwise dependency-free and
// the wasm build has no use for this.

#[cfg(feature = "serde")]
mod wire {
    use super::*;
    use serde::de::{IgnoredAny, MapAccess, Visitor};
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::borrow::Cow;

    impl Serialize for MetricValues {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let mut map = s.serialize_map(Some(Metric::COUNT))?;
            for (metric, v) in self.iter() {
                // Non-finite serialises as `null`, which is what the
                // `Option<f64>` columns wrote and what every reader expects.
                map.serialize_entry(metric.name(), &v.is_finite().then_some(v))?;
            }
            map.end()
        }
    }

    impl<'de> Deserialize<'de> for MetricValues {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = MetricValues;

                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("a map of metric names to numbers")
                }

                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<MetricValues, A::Error> {
                    let mut out = MetricValues::MISSING;
                    while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                        match Metric::by_name(&key) {
                            // `Option<f64>`, not `f64`: a diverged trial writes
                            // `"trustworthiness": null` and that must not error.
                            Some(m) => {
                                if let Some(v) = map.next_value::<Option<f64>>()? {
                                    out.set(m, v);
                                }
                            }
                            // Anything else: a non-metric column of the record
                            // this is flattened into, or a retired metric still
                            // present in an old results file. `IgnoredAny`
                            // swallows it whatever its type — which a
                            // `BTreeMap<String, f64>` would not, and the first
                            // non-numeric column added to the optimizer's
                            // output would then break every load.
                            None => {
                                map.next_value::<IgnoredAny>()?;
                            }
                        }
                    }
                    Ok(out)
                }
            }
            d.deserialize_map(V)
        }
    }
}
