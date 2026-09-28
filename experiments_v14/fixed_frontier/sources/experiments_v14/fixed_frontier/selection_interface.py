"""Frozen rule for consuming fixed-frontier results on a future select stream.

This module never reads the N14 development outputs by itself. Callers must
provide all eight future selection runs and their phase metadata explicitly.
"""
from __future__ import annotations

import numpy as np

MU = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
METHODS = [f"R{r}" for r in range(1, 9)] + ["full500"]


def choose_fixed_mu(selection_runs):
    """One μ per fixed method by the 8×4 mean ANR; ties favor smaller μ.

    Each result must be a complete N14 select-phase E2 *or* E3 45×4 result.
    No single-run or per-segment envelope is allowed. The caller invokes this
    once for E2 and once for E3 after all eight distinct runs exist.
    """
    if len(selection_runs) != 8:
        raise ValueError("Exactly eight future selection runs required")
    cases = {r.get("case") for r in selection_runs}
    run_ids = {r.get("run") for r in selection_runs}
    if (any(r.get("phase") != "select" for r in selection_runs)
            or len(cases) != 1 or not cases <= {"E2", "E3"}
            or run_ids != set(range(8))):
        raise ValueError("Need one case and select runs 0..7; dev is forbidden")
    values = np.asarray([r["segment_anr_db"] for r in selection_runs], dtype=float)
    if values.shape != (8, 45, 4) or not np.isfinite(values).all():
        raise ValueError("Each run must have finite 45×4 ANR")
    decisions = {}
    for k, name in enumerate(METHODS):
        means = values[:, 5 * k:5 * k + 5, :].mean(axis=(0, 2))
        index = int(np.argmin(means))
        decisions[name] = dict(mu=float(MU[index]), mean_anr_db=float(means[index]),
                               five_mu_mean_anr_db=means.tolist())
    return {"phase": "select", "case": next(iter(cases)), "n_runs": 8,
            "rule": "minimum 8x4 mean ANR; ties smaller mu", "fixed_methods": decisions}


def fixed_branch(result, method, mu):
    """Get one precomputed branch at an explicitly chosen μ (no minimization)."""
    if method not in METHODS or float(mu) not in MU:
        raise ValueError("Unknown fixed method or μ")
    index = METHODS.index(method) * 5 + list(MU).index(float(mu))
    return np.asarray(result["segment_anr_db"][index], dtype=float)
