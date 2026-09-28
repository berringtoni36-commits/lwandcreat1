"""Check a conditional rank-two interpolation bound for sorted coefficients.

Uses only E2/E3 development inputs and the first 20k causal R4 state.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "experiments_v12" / "outputs" / "main_dev_round1"
sys.path.insert(0, str(DEV / "sources" / "experiments_v10"))
import anc_core as ac

D1, D2, D, M, SWITCH = 25, 20, 500, 20, 20000


def row_interpolation_bound(X):
    """Best rank-2 error is at most a discrete-curvature bound.

    The row-wise endpoint interpolant has rank <= 2.  For each row with
    max absolute second difference kappa, the pointwise interpolation error
    is <= kappa*j*(D2-1-j)/2.  This finite-difference inequality applies
    without a smooth continuous interpolation assumption.
    """
    assert X.shape == (D1, D2)
    t = np.arange(D2)/(D2-1)
    L = X[:, :1]*(1-t) + X[:, -1:]*t
    kappa = np.max(np.abs(np.diff(X, n=2, axis=1)), axis=1)
    j = np.arange(D2)
    envelope = .5*kappa[:, None]*j[None, :]*(D2-1-j)[None, :]
    assert np.max(np.abs(X-L)-envelope) < 1e-10
    assert np.linalg.matrix_rank(L, tol=1e-9) <= 2
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    rank2 = (u[:, :2]*s[:2]) @ vh[:2]
    assert np.linalg.norm(X-rank2) <= np.linalg.norm(X-L)+1e-11
    return dict(rank2_fro_error=float(np.linalg.norm(X-rank2)),
                endpoint_linear_fro_error=float(np.linalg.norm(X-L)),
                curvature_fro_upper_bound=float(np.linalg.norm(envelope)),
                relative_rank2_error=float(np.linalg.norm(X-rank2)/np.linalg.norm(X)),
                relative_curvature_bound=float(np.linalg.norm(envelope)/np.linalg.norm(X)),
                max_second_difference=float(np.max(kappa)))


def run(case, run_index):
    path = DEV / f"case{case}_run{run_index:02d}.npz"
    with np.load(path, allow_pickle=False) as f:
        x, d, v, om, ph, a0, b0 = (f[k].copy() for k in
            ("x", "d", "v", "Om", "ph", "initial_A", "initial_B"))
        meta = json.loads(str(f["meta"]))
    assert meta["case"] == case and meta["run"] == run_index
    mu = float(meta["mu"])
    fixed = ac.KronRFF("fixed_R4", 1, D1, D2, 4,
                       mu/2, mu/2, 2., 1e-8, np.random.default_rng(0))
    fixed.A = a0[:, :, :4].copy()
    fixed.B = b0[:, :, :4].copy()
    xb = np.zeros((1, M))
    zh = np.zeros((len(ac.S_PATH), 1, D))
    for n in range(SWITCH):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z = np.sqrt(2/D)*np.cos(np.einsum("rdm,rm->rd", om, xb)+ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2] + .5*zh[3]
        fixed.step(z, q, np.array([d[n]+v[n]]))
    w = (fixed.B[0] @ fixed.A[0].T).T.reshape(D)
    original = w.reshape(D1, D2)
    sorted_X = np.sort(w, kind="stable").reshape(D1, D2)
    result = dict(case=case, run=run_index,
                  original=row_interpolation_bound(original),
                  sorted=row_interpolation_bound(sorted_X))
    return result


if __name__ == "__main__":
    results = [run(case, i) for case in ("E2", "E3") for i in range(5)]
    target = Path(__file__).with_suffix(".json")
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for item in results:
        print(item["case"], item["run"],
              round(item["original"]["relative_rank2_error"], 3),
              round(item["sorted"]["relative_rank2_error"], 3),
              round(item["sorted"]["relative_curvature_bound"], 3))
