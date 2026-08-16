//! The binary additive ε-indicator (Zitzler, Thiele, Laumanns, Fonseca &
//! da Fonseca 2003, *Performance Assessment of Multiobjective Optimizers*,
//! IEEE TEC 7(2):117–132).
//!
//! `r2.rs` measures a front against a *stated preference model*: the weight
//! resolution `S`, the region definitions and the "at least half the mass"
//! threshold are all choices, and every one of them is arguable. The
//! ε-indicator has no parameters at all, so it is the cross-check that says the
//! ΔR2 conclusions are not an artefact of that model.
//!
//! Two further differences from R2 make it worth reporting alongside rather than
//! instead:
//!
//! * R2 is only **weakly** Pareto compliant; the additive ε-indicator is fully
//!   compliant, so `I ≤ 0` is a statement about domination, not about a mean.
//! * R2 averages over 2002 weight vectors, which can bury a regression confined
//!   to one corner of objective space. ε is a **worst case** and cannot.
//!
//! Both quantities live in the oriented space of [`crate::objectives`] — all ten
//! objectives in `[0, 1]`, higher better — so an ε is directly readable as
//! "objective units", and the NaN-freedom `oriented_value` guarantees carries
//! through: nothing here can produce a NaN from finite input.

use crate::objectives::N_OBJECTIVES;

/// `I_ε+(A, B)`: the smallest ε such that every point of *b* is weakly dominated
/// by some point of *a* shifted up by ε on every objective.
///
/// ```text
/// I_ε+(A, B) = max_{b ∈ B}  min_{a ∈ A}  max_j (b_j − a_j)
/// ```
///
/// Negative means *a* covers *b* with room to spare; `0` means it covers it
/// exactly; positive is the shortfall. The measure is **asymmetric** — one
/// direction alone says nothing when the fronts cross — so callers want
/// [`epsilon_pair`].
///
/// `None` when either front is empty: with no *a* there is nothing to shift, and
/// with no *b* there is nothing to cover, and neither is a value of zero.
pub fn epsilon_additive(a: &[[f64; N_OBJECTIVES]], b: &[[f64; N_OBJECTIVES]]) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut worst = f64::NEG_INFINITY;
    for point_b in b {
        // The cheapest a to cover this b with.
        let mut best = f64::INFINITY;
        for point_a in a {
            let mut shift = f64::NEG_INFINITY;
            for j in 0..N_OBJECTIVES {
                let gap = point_b[j] - point_a[j];
                if gap > shift {
                    shift = gap;
                }
            }
            if shift < best {
                best = shift;
            }
        }
        if best > worst {
            worst = best;
        }
    }
    Some(worst)
}

/// Both directions of the indicator between a treatment front and its control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpsilonPair {
    /// `I_ε+(setting, baseline)`; `≤ 0` iff the setting covers the baseline.
    pub setting_vs_baseline: f64,
    /// `I_ε+(baseline, setting)`; `≤ 0` iff the baseline covers the setting.
    pub baseline_vs_setting: f64,
    /// `I_ε+(baseline, setting) − I_ε+(setting, baseline)`.
    ///
    /// Positive = the setting improved on the baseline, the same reading
    /// direction as `ΔR2` in `aggregate.rs`, which is likewise formed as
    /// baseline-minus-setting because both indicators are costs.
    pub delta: f64,
}

impl EpsilonPair {
    /// The setting's front covers every baseline point outright.
    pub fn setting_covers_baseline(&self) -> bool {
        self.setting_vs_baseline <= 0.0
    }

    /// The baseline's front covers every setting point outright.
    pub fn baseline_covers_setting(&self) -> bool {
        self.baseline_vs_setting <= 0.0
    }
}

/// [`epsilon_additive`] in both directions, with the signed summary.
///
/// `None` when either front is empty, matching [`epsilon_additive`].
pub fn epsilon_pair(
    setting: &[[f64; N_OBJECTIVES]],
    baseline: &[[f64; N_OBJECTIVES]],
) -> Option<EpsilonPair> {
    let setting_vs_baseline = epsilon_additive(setting, baseline)?;
    let baseline_vs_setting = epsilon_additive(baseline, setting)?;
    Some(EpsilonPair {
        setting_vs_baseline,
        baseline_vs_setting,
        delta: baseline_vs_setting - setting_vs_baseline,
    })
}
