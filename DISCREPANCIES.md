# Thesis ↔ results discrepancy report

Cross-check of the claims in `docs/thesis/sections/5results.typ` (and the methods
it cites) against what is actually in `results/`, as of the analysis in the
`figures` / `r2` binaries of `crates/analysis` and `--mode detect`. No thesis
`.typ` files were edited for this report. Each item says what the thesis asserts,
what the data shows, and a suggested resolution.

**Resolved items are deleted, not archived** — everything below is live.

Verified data inventory — **269 trial-results `.jsonl` cells**:

| setting          | geometries present               | datasets      | N           |
| ---------------- | -------------------------------- | ------------- | ----------- |
| `all_off`        | euclidean, hyperbolic, spherical | all 9         | 1000 & 5000 |
| `centering_only` | euclidean, hyperbolic, spherical | all 9         | 1000 & 5000 |
| `global_only`    | euclidean, hyperbolic, spherical | all 9         | 1000 & 5000 |
| `all_free`       | euclidean, hyperbolic, spherical | all 9         | 1000 & 5000 |
| `norm_only`      | euclidean, hyperbolic            | all 9         | 1000 & 5000 |
| `rms_anchored`   | hyperbolic                       | 8 (no `grid`) | 1000 & 5000 |

Datasets present: `mnist`, `fashion_mnist`, `pbmc`, `wordnet_mammals`, `sphere`,
`antipodal_clusters`, `tree`, `hyperbolic_shells`, `grid`.

252 saved `_pareto_*.json` fronts (118 at N=1000, 134 at N=5000); the remaining
cells have no saved front but are recomputed exactly by `r2 front` (validated to
reproduce a saved front bit-for-bit).

---

## A. Hard data gaps (a claimed result cannot be produced from current data)

### A1. N=5000 κ_data not computed

- **Thesis** (§curvature-magnitude-results, §manifold-projection-gap): Exp 3 and Exp 4 are reported at both N=1000 and N=5000.
- **Data:** `results/kappa_data.jsonl` covers **N=1000 only** (9 datasets). There is no `kappa_data_n5000.jsonl`.
- **Resolution:** run `slurm/submit_detect.sh`, or `--mode detect --dataset all --n-samples 5000` locally. `figures::load_kappa_data` already looks for the `_n5000` file and the Exp 3 N=5000 panel renders once it lands. Exp 4 needs no κ_data file (it derives κ per trial) and is already produced at both N.

### A2. Experiment 1's curvature grid is not a sweep product

- **Thesis** (§synthetic-grid-results): the 4×3 figure of representative embeddings, currently commented out in `5results.typ` l.57–93 with two TODOs.
- **Data:** `docs/thesis/assets/experiment_1/` holds 9 `uniform_*_k{-1,0,1}.svg` panels from the old Python pipeline: hyperbolic shells, grid and sphere, but **no tree row**, which the caption itself admits.
- **Resolution:** this figure needs an embedding-image pipeline (render an embedding at fixed hyperparameters), not the sweep JSONL. The `figures` binary does not produce it. Either regenerate all twelve panels under the Rust pipeline or restate the figure as three rows.

---

## B. Naming / count mismatches (data exists but labels or numbers differ)

### B1. `antipodal_clusters` is used but undocumented

- **Data:** `antipodal_clusters` appears in every setting (a spherical-class synthetic: clusters at antipodes). Detected geometry at N=1000: **spherical**, κ_data ≈ 3.21, angular extent 3.04.
- **Thesis** (§datasets) documents only `Sphere` as the spherical synthetic; `antipodal_clusters` is never introduced, yet it carries the all-dataset experiments (3 and 4).
- **Resolution:** add it to §datasets — it is a better spherical fixture than the uniform sphere, which the detector under-calls (see D) — or exclude it from the synthetic-data figures.

### B2. "72 Pareto runs" is a nominal count

- **Thesis** (§loss-ablations l.224): _"The full grid is therefore 6 settings × 4 real datasets × 3 geometries = 72 Pareto runs"_, immediately followed by l.225–226 excluding spherical `norm_only` and restricting `rms_anchored` to hyperbolic.
- **Data:** 60 real-dataset runs per N (48 + 8 + 4), not 72. The two sentences that follow already carve out the difference, so the arithmetic is stated before it is corrected.
- **Resolution:** one-line fix — say 60 and cite the exclusions inline, or label 72 explicitly as the nominal grid.

---

## C. Method/definition mismatches

### C1. Geometry-inference text is stale re: the hyperbolic test

- **Thesis** (§geometry-inference l.315–318, **commented out**) describes the _old_ hyperbolic test: 90th-percentile scaled Gromov δ over median distance, threshold 0.18. Line 329 admits the method _"contained an error and is not properly theoretically backed."_
- **Code:** the hyperbolic gate is the **growing-ball δ(k) saturation test** (`gromov_ball_curve::detect_hyperbolic`, log–log tail slope < `SATURATION_SLOPE_THRESHOLD = 0.15`), curvature via δ = ln(1+√2)/√(−K). The spherical angular-extent test (d_max/r* ≥ 2.5) does match the current code.
- **Resolution:** rewrite §geometry-inference around the growing-ball method and delete the l.329 admission. `--mode detect` exports the exact diagnostics the new method uses (`delta_tail_slope`, `delta_saturated`, `delta_saturated_normalised`, `delta_curvature`, `delta_is_hyperbolic`), so the text can quote real numbers.

### C2. The stated Friedman design silently excludes spherical

- **Thesis** (§front-quality): _"the five loss-weight settings are ranked within each (dataset, geometry, sample size) block, a Friedman test is applied to the resulting ranks."_
- **Data:** Friedman needs complete blocks, and `norm_only` has no spherical runs because the depth-norm loss is undefined on the sphere (§norm-loss). Taking the five settings literally therefore yields **zero complete spherical blocks**: spherical disappears from the test entirely and the pooled row runs on 36 of 54 blocks. Dropping `norm_only` from the treatment list gives all 54 blocks and a spherical row — under which `centering_only` is significant on spherical (Holm p = 0.009), a result the five-setting run cannot produce at all.
- **Resolution:** decide which test the results chapter quotes and say so. Either report the four-setting test as primary (full geometry coverage, `norm_only` handled separately as a hyperbolic/Euclidean-only comparison), or keep five settings and state that spherical is excluded by construction. `r2 aggregate --settings …` runs either, and reports `dropped_blocks` both ways.

### C3. The ε-indicator and dominance ranking are specified but not implemented

- **Thesis** (§front-quality): _"As a secondary summary we report the unary additive ε-indicator … and the dominance ranking of the merged front"_, with agreement between them and ΔR2 _"reported rather than assumed"_.
- **Code:** neither exists. `crates/analysis` computes R2 only.
- **Resolution:** both are short and use machinery already present (`oriented_matrix`, `pareto_front_mask`); add them to `r2.rs` and a column each to the `aggregate` output, or drop the sentence from the methods chapter.

### C4. Per-objective best-attainable table is specified but not implemented

- **Thesis** (§preference-regions): _"Alongside them we report, for each objective separately, the best value attained anywhere on the front and the values that same configuration attains on the other nine."_
- **Code:** `r2 recommend` produces the per-preference recommendation table, but nothing produces the per-objective best-attainable table.
- **Resolution:** a few lines over `CellSummary::front` and the oriented matrix; add a subcommand or a second table to `recommend`.

---

## D. Detector-quality caveats (correct code, but the results text does not state them)

Behaviours of the detector on the actual data, from `--mode detect` at N=1000.
Not bugs in the export, but each one is something a reader of Exp 3 would need.

| dataset              | detected   | κ_data | δ tail slope | angular extent |
| -------------------- | ---------- | ------ | ------------ | -------------- |
| `tree`               | hyperbolic | 76.07  | 0.028        | —              |
| `antipodal_clusters` | spherical  | 3.21   | 0.337        | 3.04           |
| `mnist`              | hyperbolic | 0.47   | 0.053        | —              |
| `wordnet_mammals`    | hyperbolic | 0.46   | 0.000        | —              |
| `pbmc`               | hyperbolic | 0.07   | 0.105        | —              |
| `fashion_mnist`      | euclidean  | 0.00   | 0.194        | 2.50           |
| `sphere`             | euclidean  | 0.00   | 0.188        | 2.50           |
| `hyperbolic_shells`  | euclidean  | 0.00   | 0.225        | 2.50           |
| `grid`               | euclidean  | 0.00   | 0.531        | 2.50           |

- **Synthetic `sphere` → detected euclidean.** The uniform S² sample does not cover enough of the sphere for the angular-extent gate, which pins at its 2.5 bound. `antipodal_clusters` is detected correctly, which is why it is the more useful spherical fixture (B1).
- **Synthetic `hyperbolic_shells` → detected euclidean.** δ(k) tail slope 0.225, above the 0.15 saturation threshold, so the hyperbolic gate declines. `tree` and `wordnet_mammals` are detected hyperbolic as expected.
- **`grid` behaves as the flat control**: the steepest tail slope in the table (0.531, nowhere near saturating) and κ_data = 0.
- **Spherical embedding κ saturates ≈2.5** on the Pareto front regardless of dataset (Exp 3 figure), whereas hyperbolic embedding κ spreads and tracks κ_data (Spearman ρ ≈ +0.5).

---

## Blocked deliverables

Everything not listed here is producible from the current data.

| Exp | Figure / table                        | Blocked by                                               |
| --- | ------------------------------------- | -------------------------------------------------------- |
| 1   | curvature grid (embeddings)           | A2 — needs an embedding-image pipeline; tree row missing |
| 2   | Friedman + Holm / critical-difference | C2 — spherical coverage depends on the treatment list    |
| 2   | ε-indicator + dominance ranking       | C3 — not implemented                                     |
| 3   | κ vs κ_data scatter at N=5000         | A1 — no `kappa_data_n5000.jsonl`                         |
| 5   | per-objective best attainable         | C4 — not implemented                                     |
