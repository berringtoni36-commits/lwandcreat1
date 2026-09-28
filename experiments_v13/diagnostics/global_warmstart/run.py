"""Development-only, one-time causal R4 to R2/R3 compression diagnostic.

The switch is at a predetermined absolute sample, with no segment-boundary
information.  This is deliberately an upper-limit diagnostic: it has no
candidate validation or accepted online structure decision.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEV = ROOT / "experiments_v12" / "outputs" / "main_dev_round1"
sys.path.insert(0, str(DEV / "sources" / "experiments_v10"))
import anc_core as ac

D1, D2, D, M = 25, 20, 500, 20
SWITCH = 20000
SEGMENT = 100000
TAIL = 5000
BETA = .999
EPS = 1e-12


def run(case: str, run: int = 0, methods=("signed_sort", "identity", "fixed_random"),
        mu_override: float | None = None):
    path = DEV / f"case{case}_run{run:02d}.npz"
    with np.load(path, allow_pickle=False) as f:
        x, d, v, om, ph, a0, b0 = (f[k].copy() for k in
            ("x", "d", "v", "Om", "ph", "initial_A", "initial_B"))
        meta = json.loads(str(f["meta"]))
    assert meta["case"] == case and meta["run"] == run
    assert len(x) == 4*SEGMENT
    mu = float(meta["mu"] if mu_override is None else mu_override)
    fixed = ac.KronRFF("fixed_R4", 1, D1, D2, 4,
                       mu/2, mu/2, 2., 1e-8, np.random.default_rng(0))
    fixed.A = a0[:, :, :4].copy()
    fixed.B = b0[:, :, :4].copy()
    branches = {}
    perms = {}
    xb = np.zeros((1, M))
    zh = np.zeros((len(ac.S_PATH), 1, D))
    ad = 0.
    ae = {"fixed_R4": 0.}
    sums = [{"fixed_R4": 0.} for _ in range(4)]
    check = {}
    random_perm = np.random.default_rng(2026092815).permutation(D)
    for n in range(len(x)):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z = np.sqrt(2/D)*np.cos(np.einsum("rdm,rm->rd", om, xb)+ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2] + .5*zh[3]
        dv = np.array([d[n]+v[n]])
        ad = BETA*ad+(1-BETA)*abs(d[n])
        if n < SWITCH:
            # The deployed path is R4 before the switch.  It continues as a
            # separate comparison path afterwards, but is not charged to the
            # hypothetical switched controller.
            ys = float(fixed.step(z, q, dv)[0])
            ae["fixed_R4"] = BETA*ae["fixed_R4"]+(1-BETA)*abs(dv[0]-ys)
        else:
            ys = float(fixed.step(z, q, dv)[0])
            ae["fixed_R4"] = BETA*ae["fixed_R4"]+(1-BETA)*abs(dv[0]-ys)
            for key, branch in branches.items():
                p = perms[key]
                ys = float(branch.step(z[:, p], q[:, p], dv)[0])
                ae[key] = BETA*ae[key]+(1-BETA)*abs(dv[0]-ys)
        segment = n//SEGMENT
        if n%SEGMENT >= SEGMENT-TAIL:
            for key, value in ae.items():
                sums[segment][key] = sums[segment].get(key, 0.) + \
                    20*np.log10((value+EPS)/(ad+EPS))
        if n+1 == SWITCH:
            w = (fixed.B[0] @ fixed.A[0].T).T.reshape(D)
            for method, p in (("signed_sort", np.argsort(w, kind="stable")),
                              ("identity", np.arange(D)),
                              ("fixed_random", random_perm)):
                if method not in methods:
                    continue
                Wp = w[p].reshape(D1, D2).T
                u, s, vh = np.linalg.svd(Wp, full_matrices=False)
                for rank in (2, 3):
                    key = f"{method}_R{rank}"
                    branch = ac.KronRFF(key, 1, D1, D2, rank,
                                        mu/2, mu/2, 2., 1e-8,
                                        np.random.default_rng(0))
                    scale = np.sqrt(s[:rank])
                    branch.A = (vh[:rank].T*scale)[None, :, :]
                    branch.B = (u[:, :rank]*scale)[None, :, :]
                    branch.ybuf = fixed.ybuf.copy()
                    branches[key] = branch
                    perms[key] = p.copy()
                    ae[key] = ae["fixed_R4"]
                    recon = branch.B[0] @ branch.A[0].T
                    check[key] = dict(relative_initial_truncation=float(
                        np.linalg.norm(Wp-recon)/np.linalg.norm(Wp)),
                        full_output_invariance_error=float(abs(z[0, p] @ w[p]-z[0] @ w)),
                        inherited_FIR_error=float(np.max(np.abs(branch.ybuf-fixed.ybuf))))
                    assert check[key]["full_output_invariance_error"] < 1e-12
                    assert check[key]["inherited_FIR_error"] == 0
        if (n+1)%SEGMENT == 0:
            print(case, run, "segment", segment+1, flush=True)
    mean = [{k: float(v/TAIL) for k, v in segment.items()} for segment in sums]
    r4_cost = ac.mults_kron(D1, D2, 4, M, len(ac.S_PATH))+len(ac.S_PATH)
    results = {}
    for key in branches:
        rank = int(key[-1])
        reduced_cost = ac.mults_kron(D1, D2, rank, M, len(ac.S_PATH))+len(ac.S_PATH)
        body = (SWITCH*r4_cost+(len(x)-SWITCH)*reduced_cost)/len(x)
        # Deliberately conventional 1m SVD charge, not a measured operation count.
        conventional = body+(D*4+rank*(D1+D2)+1_000_000)/len(x)
        deltas = [float(s[key]-s["fixed_R4"]) for s in mean]
        results[key] = dict(segment_deltas_db=deltas,
                            body_cost_ratio=body/r4_cost,
                            conventional_cost_ratio=conventional/r4_cost,
                            quality_within_half_db=all(x <= .5 for x in deltas),
                            check=check[key])
    out = dict(case=case, run=run, phase="dev", switch=SWITCH,
               mu=mu,
               known_segment_boundaries_used=False,
               no_candidate_gate=True,
               fixed_R4_segment_anr_db=[s["fixed_R4"] for s in mean],
               branches=results)
    suffix = "" if mu_override is None else f"_mu{mu:g}"
    (HERE / f"{case}_run{run:02d}{suffix}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", nargs="+",
                        choices=("signed_sort", "identity", "fixed_random"),
                        default=["signed_sort", "identity", "fixed_random"])
    parser.add_argument("--mu-e2", type=float)
    parser.add_argument("--mu-e3", type=float)
    args = parser.parse_args()
    for index in args.runs:
        for case in ("E2", "E3"):
            override = args.mu_e2 if case == "E2" else args.mu_e3
            run(case, index, args.methods, override)
