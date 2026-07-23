"""
Shared Pareto-front / hypervolume primitives for the fitting-curvature
experiment analysis.

Deliberately **numpy-only** (no scipy / matplotlib / sklearn) so this module and
anything that imports it (notably ``hv_stats.py``) run on a bare compute node
where only numpy is provisioned.  The figure-producing code that needs scipy and
matplotlib lives in ``analyze_experiments.py`` instead.

The ten qParEGO objectives (see ``crates/optimizer/src/pareto.rs`` ::
``default_pareto_metrics``) are, in canonical order, the 2D (post-projection)
and manifold (pre-projection) variants of five DR-quality metrics.  Two of them
(normalised stress) are minimised; the rest are maximised.  Everything here works
in an *oriented* space where every objective is mapped into ``[0, 1]`` with
higher = better, so a fixed hypervolume reference point at the origin and a unit
reference box are exact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

# ─── Objective definitions ────────────────────────────────────────────────────

# The 10 qParEGO objectives, in the order written by the Rust optimizer.
OBJECTIVES: list[str] = [
    "trustworthiness",
    "trustworthiness_manifold",
    "continuity",
    "continuity_manifold",
    "normalized_stress",
    "normalized_stress_manifold",
    "shepard_goodness",
    "shepard_goodness_manifold",
    "neighborhood_hit",
    "neighborhood_hit_manifold",
]

# Objectives where lower is better; oriented as ``1 - value``.
MINIMIZE: set[str] = {"normalized_stress", "normalized_stress_manifold"}

# Config fields carried through to a recomputed Pareto-front entry, matching the
# schema of ``pareto.rs::write_pareto_front``.
_FRONT_CONFIG_FIELDS: list[str] = [
    "n_samples",
    "learning_rate",
    "perplexity_ratio",
    "momentum_main",
    "centering_weight",
    "global_loss_weight",
    "norm_loss_weight",
    "early_exaggeration_factor",
    "curvature_magnitude",
    "r_max",
    "r_rms",
]

_SETTINGS = ("all_free", "all_off", "centering_only", "global_only", "norm_only", "rms_anchored")
_GEOMETRIES = ("euclidean", "hyperbolic", "spherical")
_CELL_RE = re.compile(
    r"^(" + "|".join(_SETTINGS) + r")_(.+?)(_n5000)?_(" + "|".join(_GEOMETRIES) + r")$"
)


# ─── Cell identity ────────────────────────────────────────────────────────────


class Cell:
    """A single (setting, dataset, N, geometry) experiment cell."""

    __slots__ = ("setting", "dataset", "n", "geometry")

    def __init__(self, setting: str, dataset: str, n: int, geometry: str):
        self.setting = setting
        self.dataset = dataset
        self.n = n
        self.geometry = geometry

    def key(self) -> tuple[str, str, int, str]:
        return (self.setting, self.dataset, self.n, self.geometry)

    def __repr__(self) -> str:
        return f"Cell({self.setting}/{self.dataset}/N{self.n}/{self.geometry})"


def parse_cell_stem(stem: str) -> Cell | None:
    """Parse a results file stem like ``all_off_mnist_n5000_hyperbolic``.

    Returns ``None`` for names that are not a plain trial-results stem (e.g.
    ``*_pareto_*`` front files, which contain a second geometry token).
    """
    if "_pareto_" in stem:
        return None
    m = _CELL_RE.match(stem)
    if not m:
        return None
    setting, dataset, n5000, geometry = m.groups()
    return Cell(setting, dataset, 5000 if n5000 else 1000, geometry)


# ─── Loading ──────────────────────────────────────────────────────────────────


def load_jsonl(path: str | Path) -> list[dict]:
    """Load every JSON object from a JSONL file; silently skip bad lines."""
    records: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return records


def trial_records(path: str | Path) -> list[dict]:
    """Trial records from a results JSONL, excluding scan sweeps."""
    return [r for r in load_jsonl(path) if not r.get("scan_param")]


# ─── Orientation to the [0, 1]-higher-is-better space ─────────────────────────


def _oriented_value(name: str, v: object) -> float:
    """Map one raw metric value into ``[0, 1]`` with higher = better.

    A missing / null / non-finite value is the worst case (0.0), matching the
    optimizer's ``metrics_to_vec`` substitution so diverged trials score as bad
    rather than being dropped silently.
    """
    if v is None:
        return 0.0
    x = float(v)
    if not np.isfinite(x):
        return 0.0
    if name in MINIMIZE:
        x = 1.0 - x
    # Metrics are bounded in [0, 1] by construction; clamp defensively.
    return min(1.0, max(0.0, x))


def oriented_matrix(records: list[dict]) -> np.ndarray:
    """``(n, 10)`` oriented-objective matrix for *records* (higher = better)."""
    if not records:
        return np.empty((0, len(OBJECTIVES)))
    return np.array(
        [[_oriented_value(m, r.get(m)) for m in OBJECTIVES] for r in records],
        dtype=float,
    )


# ─── Pareto non-domination ────────────────────────────────────────────────────


def pareto_front_mask(M: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-non-dominated rows of an oriented matrix *M*.

    Row ``i`` is dominated when some row ``j`` is ``>=`` it in every objective
    and strictly greater in at least one.  Exact duplicates are all kept (no row
    strictly dominates an identical one).
    """
    n = M.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(n, dtype=bool)
    for j in range(n):
        if not keep[j]:
            continue
        # Rows weakly dominated by j in every objective ...
        weakly = np.all(M <= M[j], axis=1)
        # ... and strictly worse somewhere (so identical rows are not dropped).
        strictly = np.any(M < M[j], axis=1)
        dominated = weakly & strictly
        dominated[j] = False
        keep[dominated] = False
    return keep


def pareto_front_records(records: list[dict]) -> list[dict]:
    """The non-dominated subset of *records* (in the 10-objective space)."""
    M = oriented_matrix(records)
    mask = pareto_front_mask(M)
    return [r for r, k in zip(records, mask) if k]


def front_entries(records: list[dict]) -> list[dict]:
    """Recompute a Pareto front and serialise it in the optimizer's JSON schema.

    Output matches ``pareto.rs::write_pareto_front``: a list of entries, each a
    flat config dict plus a ``metrics`` sub-dict of the 10 objective values.
    """
    front = pareto_front_records(records)
    entries: list[dict] = []
    for r in front:
        entry: dict = {}
        for field in _FRONT_CONFIG_FIELDS:
            val = r.get(field)
            # Euclidean trials carry no curvature; the optimizer still writes a
            # (fixed) magnitude, so default the missing field to 0.0.
            entry[field] = 0.0 if (field == "curvature_magnitude" and val is None) else val
        entry["metrics"] = {m: r.get(m) for m in OBJECTIVES}
        entries.append(entry)
    return entries


# ─── Monte-Carlo hypervolume ──────────────────────────────────────────────────


def monte_carlo_hypervolume(
    M: np.ndarray,
    n_mc: int = 2_000_000,
    seed: int = 0,
    chunk: int = 200_000,
) -> tuple[float, float]:
    """Seeded Monte-Carlo hypervolume in the oriented unit box ``[0, 1]^10``.

    The reference point is the origin and the reference box has volume 1, so the
    hypervolume dominated by the point set equals the fraction of uniform samples
    that the set dominates.  Dominated points contribute nothing, so passing the
    Pareto front (rather than all trials) gives an identical estimate faster.

    Returns ``(hv, se)`` where ``se`` is the Monte-Carlo standard error of the
    estimate, ``sqrt(p (1 - p) / n_mc)`` from the dominated-fraction binomial.
    This is *estimation* error of the integral, not run-to-run optimizer
    variance — there is a single optimization run per cell, so the latter is not
    estimated here.
    """
    if M.shape[0] == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    d = M.shape[1]
    dominated = 0
    done = 0
    while done < n_mc:
        m = min(chunk, n_mc - done)
        U = rng.random((m, d))
        hit = np.zeros(m, dtype=bool)
        # A sample u is dominated iff some front point f has f >= u in all dims.
        for f in M:
            hit |= np.all(U <= f, axis=1)
        dominated += int(hit.sum())
        done += m
    p = dominated / n_mc
    se = float(np.sqrt(max(p * (1.0 - p), 0.0) / n_mc))
    return float(p), se


def cell_hypervolume(
    records: list[dict],
    n_mc: int = 2_000_000,
    seed: int = 0,
) -> dict:
    """Hypervolume summary for one cell's trial records.

    Reduces to the Pareto front first (HV-invariant, much faster), then runs the
    Monte-Carlo estimate.  Returns a dict with the trial/front counts, the HV and
    its MC standard error.
    """
    M_all = oriented_matrix(records)
    mask = pareto_front_mask(M_all)
    M_front = M_all[mask]
    hv, se = monte_carlo_hypervolume(M_front, n_mc=n_mc, seed=seed)
    return {
        "n_trials": int(M_all.shape[0]),
        "n_front": int(M_front.shape[0]),
        "hv": hv,
        "hv_se": se,
    }
