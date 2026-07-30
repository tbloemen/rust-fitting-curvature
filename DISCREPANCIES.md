# Thesis ↔ results discrepancy report

Cross-check of the claims in `docs/thesis/sections/5results.typ` (and the
methods it cites) against what is actually in `results/`, as of the analysis in
the `figures` / `hv` binaries of `crates/analysis` and `--mode detect`. No thesis `.typ`
files were edited. Each item says what the thesis asserts, what the data shows,
and a suggested resolution.

Verified data inventory (176 trial-results `.jsonl` cells):

| setting | geometries present | datasets | N |
|---|---|---|---|
| `all_off` | euclidean, hyperbolic, **spherical** | all 8 | 1000 & 5000 |
| `centering_only` | euclidean, hyperbolic | all 8 | 1000 & 5000 |
| `global_only` | euclidean, hyperbolic | all 8 | 1000 & 5000 |
| `norm_only` | euclidean, hyperbolic | all 8 | 1000 & 5000 |
| `all_free` | euclidean, hyperbolic | all 8 | 1000 & 5000 |
| `rms_anchored` | — (no runs) | — | — |

Datasets present: `mnist`, `fashion_mnist`, `pbmc`, `wordnet_mammals`,
`sphere`, `antipodal_clusters`, `tree`, `hyperbolic_shells`. (No `grid`.)

128 saved `_pareto_*.json` fronts (40 at N=1000, 88 at N=5000); the other 49
cells have no saved front but are recomputed exactly by `hv front`
(validated to reproduce a saved front bit-for-bit).

---

## A. Hard data gaps (a claimed result cannot be produced from current data)

### A1. Spherical is missing for every non-baseline loss setting — **the "56 Pareto fronts" of Experiment 2 do not exist**
- **Thesis** (§ablation-results, l.103): *"At each sample size this yields fifty-six Pareto fronts: the nominal 5×4×3 grid minus the four spherical `norm_only` cells."* This assumes spherical runs for `all_off`, `centering_only`, `global_only`, `all_free` on the 4 real datasets.
- **Data:** spherical exists **only for `all_off`**. `centering_only`, `global_only`, `all_free` have **no spherical runs at all** (euclidean + hyperbolic only). Real-dataset Exp 2 fronts actually present per N: `all_off` 4×3=12, `{centering,global,all_free}` 4×2×3=24, `norm_only` 4×2=8 → **44 fronts, not 56**. The 12 missing are the spherical `centering_only`/`global_only`/`all_free` × 4-real-dataset cells.
- **Resolution:** either (a) run the 12 missing spherical cells per N (24 total), or (b) restate Exp 2 as a **44-front** design and note spherical is only compared under `all_off` (which is what Exp 5 already does). The ΔH figure/table and `hv aggregate` already handle the reduced grid gracefully (spherical simply has no non-baseline ΔH row).

### A2. `rms_anchored` is specified but never run — **Experiment 3's reproducibility test cannot be done**
- **Thesis** (§tab:loss-ablations l.218 lists it as the 6th setting; §gauge-fixing l.252; §curvature-magnitude-results l.126, l.131): the `rms_anchored` setting fixes R_rms=1 so the search runs over κ directly, and Exp 3 overlays unanchored vs `rms_anchored` κ distributions per hyperbolic dataset.
- **Data:** zero `rms_anchored_*` files. The `TrialConfig::rms_anchored()` code path exists (`common.rs`) but was never executed.
- **Resolution:** run `--experiment rms_anchored --geometry hyperbolic` on the 4 real datasets (+ synthetics if wanted) at N=1000/5000, or drop the reproducibility sub-claim. `figures::exp3::RmsAnchored` already detects the absence and skips with a notice, and will produce the overlay automatically once the runs exist.

### A3. N=5000 κ_data not yet computed
- **Thesis** (§curvature-magnitude-results, §manifold-projection-gap): Exp 3 and Exp 4 are reported at both N=1000 and N=5000.
- **Data:** `results/kappa_data.jsonl` exists for **N=1000 only** (produced locally via `--mode detect`). The Exp 3 scatter needs `kappa_data_n5000.jsonl`.
- **Resolution:** run `slurm/submit_detect.sh` (writes `kappa_data_n5000.jsonl`). `figures::load_kappa_data` already looks for the `_n5000` file and the Exp 3 N=5000 panel will render once it lands. (Exp 4 needs no κ_data file — it derives κ per trial — so it is already produced at both N.)

### A4. No Euclidean synthetic ("Grid") dataset
- **Thesis** (§datasets l.23; Exp 1 figure grid, l.76–79): lists *Grid (Euclidean): a lattice in R²* as one of four synthetic datasets, and the Exp 1 curvature-grid figure has a "Euclidean Grid" row.
- **Data:** no `grid` results. The optimizer's `load_synthetic` (`data.rs`) wires only `sphere`, `antipodal_clusters`, `tree`, `hyperbolic_shells`; `uniform_grid` exists in `synthetic_data.rs` but is not selectable from the sweep binary.
- **Resolution:** add a `"grid" => generate_uniform_grid(...)` arm to `data.rs::load_synthetic` and run it, or remove Grid from §datasets and the Exp 1 figure.

---

## B. Naming / count mismatches (data exists but labels or numbers differ)

### B1. `antipodal_clusters` is used but undocumented
- **Data:** `antipodal_clusters` appears in all sweeps (it is a spherical-class synthetic — clusters at antipodes). Detected geometry at N=1000: **spherical**, κ_data≈3.2.
- **Thesis** (§datasets) documents only `Sphere` as the spherical synthetic; `antipodal_clusters` is never introduced.
- **Resolution:** add it to §datasets (it is arguably a *better* spherical stress-test than the uniform sphere, which the detector under-calls — see C1), or exclude it from the synthetic-data figures. It currently appears in Exp 3/Exp 4 (all-dataset experiments).

### B2. "Six settings" vs "five settings"
- **Thesis** is internally consistent here: methods (l.200, tab:loss-ablations) define **six** settings; the results chapter's Exp 2 uses **five** of them (`all_off` + the four non-baseline) and Exp 3 uses the sixth (`rms_anchored`). This is only a discrepancy against the *data* via A2 (the sixth was never run). No text change needed beyond A2.

### B3. 128 saved fronts, not 56/112
- The "56 fronts" is the intended per-N Exp 2 design (real datasets), not the file count. The 128 saved `_pareto_*.json` span all 8 datasets × 3-or-2 geometries and both N, and 49 further cells have recomputable fronts. Not a contradiction — just note that the saved-front count is not the Exp 2 headline number, and that fronts for figures are recomputed uniformly from the `.jsonl` (so the 40-vs-88 saved split between N=1000 and N=5000 does not bias anything).

---

## C. Method/definition mismatches (fixed in code where the thesis is the spec)

### C1. κ_data d_rms convention — **fixed**
- **Thesis** (§gauge-fixing l.257): κ_data = |K_data|·d_rms² = (d_rms/r*)², with d_rms the *input RMS pairwise distance*.
- **Was:** `--mode detect` first computed d_rms as RMS distance **from the centroid** (`√(S/2n²)`), a factor √(2n/(n-1)) ≈ √2 smaller, so κ_data was ~2× too small.
- **Now:** `detect.rs` uses `rms_pairwise = √(S/n(n-1))` to match the thesis. The N=1000 `kappa_data.jsonl` was **re-run** with the corrected detector (`--mode detect --dataset all --n-samples 1000`, ~10 min locally on 16 cores), which also picked up the `grid` dataset the first run predated. The new values agree with the exact algebraic factor `2n/(n-1)` to ~1e-12, as expected since the curvatures are independent of d_rms. Exp 3 figures were regenerated. (E.g. `tree` κ_data 38.0 → 76.1.)

### C2. Geometry-inference text is stale re: the hyperbolic test
- **Thesis** (§geometry-inference l.315–318, **commented out**) describes the *old* hyperbolic test: 90th-percentile scaled Gromov δ / median distance, threshold 0.18. Line 329 explicitly admits: *"my method/implementation for determining hyperbolic space from Euclidean Space … contained an error and is not properly theoretically backed. This needs to be revisited."*
- **Code (merged branch):** the hyperbolic gate is now the **growing-ball δ(k) saturation test** (`gromov_ball_curve::detect_hyperbolic`, log–log tail slope < `SATURATION_SLOPE_THRESHOLD = 0.15`), curvature via δ = ln(1+√2)/√(−K). The spherical angular-extent test (l.312–313, d_max/r* ≥ 2.5) **does** match current code.
- **Resolution:** rewrite §geometry-inference to describe the growing-ball saturation method and delete the l.329 admission. `--mode detect` now exports the exact diagnostics the new method uses (`delta_tail_slope`, `delta_saturated`, `delta_saturated_normalised`, `delta_curvature`, `delta_is_hyperbolic`) so the text can quote real numbers.

### C3. κ uses R_rms, not R_max — **fixed in analysis**
- **Thesis** (§gauge-fixing l.240, l.246): κ = |K|·R_rms².
- The earlier `analyze_hyperparams.py` used `|K|·r_max²`. The `figures` binary uses `r_rms` throughout (both the per-trial κ and the median-front κ). No data change — both `r_rms` and `r_max` are already logged per trial.

---

## D. Detector-quality caveats (correct code, but worth stating in the results text)

These are behaviours of the (now theoretically-backed) detector on the actual
data, surfaced by `--mode detect` at N=1000. They are not bugs in the export.

- **Synthetic `sphere` → detected euclidean.** Spherical fit pins at the flat-ward bound (ρ≈0.56); the uniform S² sample does not cover enough of the sphere for the angular-extent gate. `antipodal_clusters` (spherical, κ_data≈3.2) is detected correctly, which is why it is the more useful spherical fixture.
- **Synthetic `hyperbolic_shells` → detected euclidean.** δ(k) tail slope ≈0.22, just above the 0.15 saturation threshold, so the hyperbolic gate declines. `tree` (κ_data≈76) and `wordnet_mammals` are detected hyperbolic as expected.
- **Verdicts at N=1000:** mnist→hyperbolic, fashion_mnist→euclidean, pbmc→hyperbolic, wordnet_mammals→hyperbolic, antipodal_clusters→spherical, tree→hyperbolic, sphere→euclidean, hyperbolic_shells→euclidean, grid→euclidean (the flat control, as expected).
- **Spherical embedding κ saturates ≈2.5** on the Pareto front regardless of dataset (Exp 3 figure), whereas hyperbolic embedding κ spreads and tracks κ_data (Spearman ρ≈+0.5). Worth a sentence in the Exp 3 discussion.

---

## Summary of what each experiment can currently deliver

| Exp | Figure | Status with current data |
|---|---|---|
| 1 | curvature grid (embeddings) | needs embedding-image pipeline + Grid dataset (A4); not from sweep JSONL |
| 2 | stacked Pareto fronts | ✅ produced (N=1000/5000); spherical column is `all_off`-only (A1) |
| 2 | ΔH table | ✅ `hv aggregate`; spherical has no non-baseline ΔH (A1) |
| 3 | κ vs κ_data scatter | ✅ N=1000 (κ_data fixed, C1); N=5000 needs A3 |
| 3 | rms_anchored κ overlay | ❌ blocked by A2 |
| 4 | ρ_man-proj(κ) | ✅ produced (both N) |
| 5 | trust-vs-stress front grid | ✅ produced (N=1000/5000) |
| 5 | hyperparameter marginals | ✅ produced (N=1000/5000) |
