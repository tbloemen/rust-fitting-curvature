//! One embedding's score on every metric.
//!
//! Replaces four parallel per-metric structs — the optimizer's `AllMetrics`,
//! its `TrialResult` columns, its `PriorTrialRecord`, and the analysis's
//! `TrialRecord` columns — with one positional array plus a name-keyed wire
//! format. The array is indexed by position in [`ALL`]; nothing outside this
//! file knows that.

use super::quality::{Metric, ALL};
use crate::cast::count_to_f64;
use crate::context::EmbeddingContext;

/// One metric's reading for one embedding.
///
/// The element used to be a bare `f64` with `f64::NAN` standing in for "no
/// reading", which made `MISSING == MISSING` false, ruled out `Eq` and `Hash`,
/// and — more to the point — could not say *why* there was no number. There are
/// three distinct reasons, and telling them apart is what lets a diverged trial
/// be scored as diverged instead of as whatever its rank statistics happened to
/// produce.
///
/// Sixteen bytes: the discriminant packs into the `f64`'s alignment padding, so
/// the reasons cost nothing over an `Option<f64>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricValue {
    /// Measured. **Always finite** — [`MetricValue::measured`] downgrades
    /// anything else, so there is exactly one representation of each state.
    Measured(f64),
    /// Undefined for this input: a label-aware metric on unlabelled data.
    NotApplicable,
    /// Computed against distances that were not finite, so whatever number came
    /// out is meaningless. Set by the divergence gate in
    /// [`MetricValues::compute`], and by `measured` for a reading that went
    /// non-finite on its own.
    Diverged,
    /// No column was written, or the column was `null`.
    ///
    /// The only non-`Measured` state a value read back from disk can be in:
    /// JSON has one `null`, so it cannot carry the distinction the other two
    /// variants make. That is a property of the format, not an oversight — the
    /// reasons are worth having in memory, where the scoring decision happens.
    Absent,
}

impl MetricValue {
    /// A reading straight from a metric function.
    ///
    /// Non-finite becomes [`Self::Diverged`]: a metric that summed its way to a
    /// NaN was reading a broken embedding, which is the same conclusion the gate
    /// reaches, and it must not be storable as a `Measured`.
    #[must_use]
    pub fn measured(v: f64) -> Self {
        if v.is_finite() {
            MetricValue::Measured(v)
        } else {
            MetricValue::Diverged
        }
    }

    /// The number, or `None` for any of the three reasons there isn't one.
    ///
    /// Every caller that feeds a plot or a statistic wants this: `figures/exp5`
    /// filters on it and `oriented_value` maps `None` to the worst case.
    #[must_use]
    pub fn value(self) -> Option<f64> {
        match self {
            MetricValue::Measured(v) => Some(v),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_measured(self) -> bool {
        matches!(self, MetricValue::Measured(_))
    }
}

/// Every metric's reading for one embedding, positional in [`ALL`] order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricValues([MetricValue; Metric::COUNT]);

impl Default for MetricValues {
    fn default() -> Self {
        Self::MISSING
    }
}

impl MetricValues {
    /// Nothing measured. This is what `--mode scan` writes — it builds a result
    /// without ever scoring one — and it serialises as an all-`null` block,
    /// exactly as the `Option<f64>` columns it replaces did.
    pub const MISSING: Self = Self([MetricValue::Absent; Metric::COUNT]);

    /// Score every metric in [`ALL`]. The context derives each distance matrix
    /// at most once, however many metrics read it.
    ///
    /// **The divergence gate.** A metric whose distance matrix is not finite is
    /// reported [`MetricValue::Diverged`] without being run, because running it
    /// produces a number that looks measured and is not: of the fourteen, only
    /// the four that *sum* distances notice a NaN. The four that *compare* fall
    /// into their degenerate branches and report `0.0` (`f64::max` ignores NaN;
    /// `if d > max_intra` and `if d_own < nearest_rival` are both false for
    /// NaN), and the six that *rank* sort NaN to
    /// a defined position via `total_cmp` and return a confident, plausible
    /// score — `trustworthiness` reads 0.93 on an embedding that blew up, and
    /// `pareto::metrics_to_vec` then hands the GP a 0.93.
    ///
    /// The gate is per [`super::Space`], not global: an embedding can be sound on the
    /// manifold and blow up only through the projection, and failing both
    /// readings would throw away a real measurement.
    pub fn compute(ctx: &EmbeddingContext<'_>) -> Self {
        let mut out = Self::MISSING;
        for (slot, metric) in out.0.iter_mut().zip(ALL) {
            *slot = if ctx.distances_are_finite(metric.space()) {
                metric.compute(ctx)
            } else {
                MetricValue::Diverged
            };
        }
        out
    }

    /// This metric's reading, reason included.
    #[must_use]
    pub fn reading(&self, m: Metric) -> MetricValue {
        self.0[m.index()]
    }

    /// This metric's number, or `None` — [`MetricValue::value`] of
    /// [`Self::reading`], which is what almost every caller wants.
    #[must_use]
    pub fn get(&self, m: Metric) -> Option<f64> {
        self.reading(m).value()
    }

    pub fn set(&mut self, m: Metric, v: MetricValue) {
        self.0[m.index()] = v;
    }

    /// Component-wise mean over a non-empty slice of samples.
    ///
    /// All `Measured` gives the mean. Otherwise the strongest reason present —
    /// `Diverged` over `NotApplicable` over `Absent` — so a seed that blew up
    /// is visible in the aggregate rather than averaged out of existence. That
    /// was the job NaN propagation did before, stated explicitly.
    #[must_use]
    pub fn mean(samples: &[MetricValues]) -> MetricValues {
        let n = count_to_f64(samples.len());
        let mut out = Self::MISSING;
        for (j, slot) in out.0.iter_mut().enumerate() {
            *slot = mean_of(samples.iter().map(|s| s.0[j]), n);
        }
        out
    }
}

/// [`MetricValues::mean`] for one component, shared with
/// [`crate::spread::SpreadDiagnostics`], which averages the same way.
pub(crate) fn mean_of(readings: impl Iterator<Item = MetricValue>, n: f64) -> MetricValue {
    let mut total = 0.0;
    let mut reason = None;
    for r in readings {
        match r {
            MetricValue::Measured(v) => total += v,
            // Ordered by severity, so the strongest reason survives.
            other => {
                let rank = |v: MetricValue| match v {
                    MetricValue::Diverged => 3,
                    MetricValue::NotApplicable => 2,
                    _ => 1,
                };
                if reason.is_none_or(|held| rank(other) > rank(held)) {
                    reason = Some(other);
                }
            }
        }
    }
    reason.unwrap_or_else(|| MetricValue::measured(total / n))
}

// ─── Wire format ─────────────────────────────────────────────────────────────
//
// A flat, name-keyed map, so `#[serde(flatten)]` puts the metric columns at the
// top level of a trial record exactly where the individual fields used to sit.
// Behind the `serde` feature: `fitting-core` is otherwise dependency-free and
// the wasm build has no use for this.

#[cfg(feature = "serde")]
mod wire {
    use super::{Metric, MetricValue, MetricValues, ALL};
    use serde::de::{IgnoredAny, MapAccess, Visitor};
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::borrow::Cow;

    impl Serialize for MetricValues {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let mut map = s.serialize_map(Some(Metric::COUNT))?;
            for metric in ALL {
                // Every reason for not having a number writes `null`: JSON has
                // only the one, so the distinction between Diverged,
                // NotApplicable and Absent lives in memory and not on disk.
                map.serialize_entry(metric.name(), &self.get(*metric))?;
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

                fn visit_map<A: MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> Result<MetricValues, A::Error> {
                    let mut out = MetricValues::MISSING;
                    while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                        match Metric::by_name(&key) {
                            // `Option<f64>`, not `f64`: a trial that did not
                            // measure writes `"trustworthiness": null`, and
                            // that must not error. It reads back as `Absent` —
                            // the reason it was not measured does not survive
                            // the format.
                            Some(m) => {
                                let v = map.next_value::<Option<f64>>()?;
                                out.set(m, v.map_or(MetricValue::Absent, MetricValue::measured));
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
