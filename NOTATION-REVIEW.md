# Notation review: `crates/analysis` ↔ the literature

Cross-check of the identifiers and symbols used in `crates/analysis` (and the
`docs/thesis` notation they mirror) against the papers each convention comes
from. Each item says what the code calls something, what the literature calls
it, whether it matters, and a suggested resolution.

Scope: naming only. One statistical-method note is included at the end because
it sits inside a cited section. No `.rs` or `.typ` files were edited for this
report.

**Work through this by ticking the decision box on each item.** Resolved items
are deleted, not archived — everything below is live.

---

## 0. What already matches (no action)

Recorded so the same ground is not re-checked later.

| convention | where | verdict |
| --- | --- | --- |
| `λ` for a weight vector | `r2.rs:206`, thesis `4methods.typ:300-304` | Correct. ParEGO/MOEA/D convention; `r2.rs:30` already cites Knowles 2006 eq. 1 for `λ_j = l/s`. |
| `utility` for `min_a max_j λ_j(z*_j − a_j)` | `r2.rs:FrontUtility` | Correct despite smaller-is-better. The R2 literature does call the Tchebycheff aggregation a "utility function" while writing the indicator with `min`. |
| `W` for the weight set, `z*` ideal point, `s = 5` | `r2.rs`, thesis `4methods.typ:299-304` | Matches thesis and the Hansen/Jaszkiewicz lineage. |
| The five metric names | `objectives.rs:7-17` | Match Espadoto et al.'s survey term-for-term, **including "Shepard goodness"**, which is genuinely their term and not a local invention. |
| `block` / `treatment` / `control` | `stats.rs`, `aggregate.rs` | Correct classical-stats vocabulary for what Demšar §3.2.2 calls datasets/algorithms. |

Note for later: the post-2020 R2 papers have drifted to `w`/`W` for weights and
`m` for the objective count. Since this project cites Knowles for the simplex,
`λ` remains the better anchor — do not "modernise" it.

---

## 1. `κ` collides with the constant-curvature literature

- [ ] **Decision:**

- **Code:** `records.rs:119-123` defines `kappa()` as the dimensionless product
  `|K| · R_rms²`. Thesis `3theory.typ:159` (`<eq:kappa>`) defines the same, and
  `3theory.typ:165` calls it "the curvature quantity we use throughout".
  Propagates to `figures/mod.rs:9`, `exp3.rs`, `exp4.rs:125` (`x_desc("median κ (cell)")`).
- **Literature:** in the κ-stereographic line of work — Bachmann et al.
  (Constant Curvature GCNs), Skopek et al. (Mixed-curvature VAEs), and the
  `geoopt` implementation — **κ *is* the signed sectional curvature**, i.e. the
  quantity this repo calls `curvature` / `K`. That is the closest adjacent
  literature to this thesis, which is what makes it the worst collision.
- **Impact:** highest of anything in this report. A reader from that literature
  reads `κ = 0.3` as a curvature, not as a squared radius ratio. Every other item
  here mildly surprises a reader; this one actively misleads.
- **Resolution options:**
  1. Rename to a symbol that is free in this space — `κ̃`, or `τ`.
  2. Keep κ but state the departure explicitly at `3theory.typ:159`, one
     sentence, naming the κ-stereographic usage it conflicts with.
  3. Spell out what it is: `(R_rms / r_curv)²`, the extent in units of the
     curvature radius. `3theory.typ:162` already says this in prose.
- **Cost:** touches `kappa` as a serialized concept (`kappa_data.jsonl`,
  `hyp_kappa` / `sph_kappa` at `figures/mod.rs:268-270`) plus thesis prose and
  figure axes. Option 2 is nearly free; option 1 is not.

## 2. `normalized_stress` is actually *scale-normalized* stress

- [ ] **Decision:**

- **Code:** `crates/core/src/metrics.rs:559` solves for the optimal scale
  `alpha = Σ d_h d_e / Σ d_e²` before computing the residual. That is the
  scale-optimized variant, not plain normalized stress. Exposed as
  `normalized_stress` / `normalized_stress_manifold` in `objectives.rs:11-16`
  and as the only two entries of `objectives.rs::MINIMIZE`.
- **Literature:** Smelser, Miller & Kobourov (IEEE VIS 2024), *"Normalized
  Stress" is Not Normalized*, introduce exactly this variant **because** plain
  normalized stress is not scale-invariant, and name it **scale-normalized
  stress**. A 2025 follow-up (*How Scale Breaks "Normalized Stress" and KL
  Divergence*) extends the argument to other quality metrics.
- **Impact:** the thesis already depends on the distinction the paper draws —
  `4methods.typ:239` argues normalised stress is invariant under the
  `(K, R_rms) → (K/λ², λR_rms)` rescaling precisely because of the α factor. So
  the property is load-bearing for the scale-invariance argument, and the bare
  name is the one the paper is arguing against.
- **Resolution:** rename to `scale_normalized_stress`, **or** — if renaming a
  JSON key present in every results file is too costly — document it in one line
  at `objectives.rs` and cite Smelser et al. in the thesis where §metrics defines
  it. The citation is the part that actually matters.
- **Cost:** rename touches `records.rs` field names, `TrialRecord::objective`,
  `OBJECTIVES`, `METRICS`, `MINIMIZE`, and every `results/*.jsonl` already on
  disk. Documenting is free.

## 3. `r2.rs` never names the scalarization

- [ ] **Decision:**

- **Code:** the `r2.rs:1-20` module doc writes the formula out in full but never
  says "Chebyshev" or "Tchebycheff", and cites neither Hansen & Jaszkiewicz nor
  Brockhoff et al. `stats.rs` cites Demšar three times; `r2.rs` cites only
  Knowles, and only for the simplex granularity.
- **Literature:** thesis `4methods.typ:296` cites both R2 references and
  `4methods.typ:301` names the "weighted Chebyshev utility" — so here the code is
  the weaker link, not the thesis.
- **Impact:** low but cheap to fix. Anyone grepping the crate for the standard
  keyword finds nothing.
- **Resolution:** one sentence in the `r2.rs` header naming the weighted
  Chebyshev aggregation and pointing at Hansen & Jaszkiewicz (1998) and
  Brockhoff, Wagner & Trautmann (2012).
- **Spelling sub-item:** Knowles and MOEA/D (Zhang & Li) both write
  **"Tchebycheff"**; the thesis writes "Chebyshev" (`2related-work.typ:87,89`,
  `4methods.typ:283,301,319`). Both spellings are attested, but the branch this
  project cites uses the former. Pick one and apply it in both places.

## 4. `ρ` carries three meanings

- [ ] **Decision:**

- **Occurrences:**
  - Spearman's ρ — `stats.rs:86-92`, used by Exp 3.
  - `ρ_man-proj` — `exp4.rs:1,33`, the manifold-vs-projected rank agreement.
  - `ρ_perp`, perplexity ratio — thesis `4methods.typ:173`; surfaces here as the
    `perplexity_ratio` column (`records.rs:42`, `TrialRecord::param`).
  - `ρ(r)`, the signature residual — thesis `4methods.typ:383`.
- **Impact:** the first two appear in the same figures, and the collision is
  live inside the analysis crate rather than only in the thesis.
- **Resolution:** ρ for Spearman is non-negotiable, and `ρ_man-proj` inherits
  from it legitimately. `ρ_perp` is the one to move — perplexity has no ρ
  tradition (van der Maaten & Hinton write `Perp`). `f_perp`, or spelling it out
  as "perplexity fraction", both work. The signature residual `ρ(r)` is a
  separate chapter and probably tolerable, but note it.

---

## 5. Non-naming note: the Friedman statistic

- [ ] **Decision:**

- **Code:** `stats.rs:292` implements scipy's `friedmanchisquare`, i.e. the χ²_F
  statistic, and `aggregate.rs:152` reports it as `RankTest.statistic`.
- **Literature:** Demšar (2006) §3.2.2 — cited at `stats.rs:7`,
  `stats.rs:286,343`, `aggregate.rs:17` and thesis `4methods.typ:325` — argues
  specifically that χ²_F is *undesirably conservative* and recommends the
  Iman–Davenport `F_F` statistic instead.
- **Impact:** the vocabulary is right; the statistic is the one the cited section
  advises against. This is a methods question, not a naming one, but it is
  reached through a naming check, so it is recorded here.
- **Resolution:** either switch to `F_F` (it is a closed-form transform of χ²_F
  and needs no new machinery beyond an F-distribution tail — `student_t_sf` and
  `betai` at `stats.rs:161,177` already give the incomplete beta it needs), or
  state in the thesis why χ²_F is used. Separately, rename
  `RankTest.statistic` → `chi2_f` / `f_f` once decided; `statistic` alone does
  not say which.

---

## Sources

- Hansen & Jaszkiewicz (1998), *Evaluating the quality of approximations to the non-dominated set* — original R2.
- Brockhoff, Wagner & Trautmann (2012), *On the Properties of the R2 Indicator* — <https://www.researchgate.net/publication/254462273_On_the_Properties_of_the_R2_Indicator>
- *R2 v2: The Pareto-compliant R2 Indicator* (2024) — <https://arxiv.org/html/2407.01504> (current `w`/`W`/`m` notation)
- *Multiobjective Optimization Using the R2 Utility*, SIAM Review — <https://epubs.siam.org/doi/10.1137/23M1578371>
- Knowles (2006), *ParEGO* — the `λ_j = l/s` simplex, already cited at `r2.rs:30`.
- Espadoto et al. (2019/2021), *Toward a Quantitative Survey of Dimension Reduction Techniques* — <https://pubmed.ncbi.nlm.nih.gov/31567092/>
- Smelser, Miller & Kobourov (2024), *"Normalized Stress" is Not Normalized* — <https://arxiv.org/abs/2408.07724>
- *How Scale Breaks "Normalized Stress" and KL Divergence* (2025) — <https://arxiv.org/html/2510.08660>
- κ-stereographic model, `geoopt` docs — <https://geoopt.readthedocs.io/en/latest/extended/stereographic.html>
- Skopek, Ganea & Bécigneul (2020), *Mixed-curvature Variational Autoencoders* — <https://arxiv.org/pdf/1911.08411>
- Demšar (2006), *Statistical Comparisons of Classifiers over Multiple Data Sets* — §3.2.2, already cited throughout `stats.rs`.
