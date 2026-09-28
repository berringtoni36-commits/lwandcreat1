"""Independent dev-only replay of the 20k causal R4 compression diagnostic.

Uses the saved E2/E3 development inputs; never opens selection or confirmation.
The pointwise update kernel is separate from global_warmstart/run.py and is
compared to its published JSON only after replay.  No source file is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

DEPS = Path(tempfile.gettempdir()) / "n13_frontier_deps"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "experiments_v13" / "diagnostics" / "global_warmstart"
DEV = ROOT / "experiments_v12" / "outputs" / "main_dev_round1"
SWITCH = 20_000
T = 400_000
SEGMENT = 100_000
TAIL = 5_000
MU_BY_CASE = {"E2": .2, "E3": .1}
BRANCH_NAMES = [f"{m}_R{r}" for m in ("signed_sort", "identity", "fixed_random")
                for r in (2, 3)]
METHODS = [f"fixed_R{r}" for r in range(1, 9)] + ["full500"] + BRANCH_NAMES


@njit(cache=True)
def step_kron(a, b, rank, mu, z, q, perm, dv, history):
    ub = np.zeros((20, 8), np.float64)
    ua = np.zeros((25, 8), np.float64)
    y = 0.0
    pred1 = 0.0
    nb = 0.0
    for r in range(rank):
        for j in range(20):
            dotz = 0.0
            dotq = 0.0
            for i in range(25):
                index = perm[20 * i + j]
                dotz += z[index] * a[i, r]
                dotq += q[index] * a[i, r]
            ub[j, r] = dotq
            y += b[j, r] * dotz
            pred1 += b[j, r] * dotq
            nb += dotq * dotq
    physical_error = dv - history[1] - .5 * history[2]
    history[2] = history[1]
    history[1] = history[0]
    history[0] = y
    residual1 = dv - pred1
    gainb = (mu / 2.0) * residual1 * np.exp(-residual1 * residual1 / 8.0) / (1e-8 + nb)
    for r in range(rank):
        for j in range(20):
            b[j, r] += gainb * ub[j, r]
    pred2 = 0.0
    na = 0.0
    for r in range(rank):
        for i in range(25):
            dotq = 0.0
            for j in range(20):
                dotq += q[perm[20 * i + j]] * b[j, r]
            ua[i, r] = dotq
            pred2 += a[i, r] * dotq
            na += dotq * dotq
    residual2 = dv - pred2
    gaina = (mu / 2.0) * residual2 * np.exp(-residual2 * residual2 / 8.0) / (1e-8 + na)
    for r in range(rank):
        for i in range(25):
            a[i, r] += gaina * ua[i, r]
    return y, physical_error


@njit(cache=True)
def step_full(w, mu, z, q, dv, history):
    y = 0.0
    prediction = 0.0
    normq = 0.0
    for j in range(500):
        y += w[j] * z[j]
        prediction += w[j] * q[j]
        normq += q[j] * q[j]
    physical_error = dv - history[1] - .5 * history[2]
    history[2] = history[1]
    history[1] = history[0]
    history[0] = y
    residual = dv - prediction
    gain = mu * residual * np.exp(-residual * residual / 8.0) / (1e-8 + normq)
    for j in range(500):
        w[j] += gain * q[j]
    return y, physical_error


@njit(cache=True)
def run_section(start, stop, x, d, v, omega, phase, mu,
                a, b, wfull, ba, bb, perms, histories, ae, ad,
                xb, p1, p2, p3, ytrace, etrace, sums, branches_live):
    identity = np.arange(500)
    z = np.zeros(500, np.float64)
    q = np.zeros(500, np.float64)
    scale = np.sqrt(2.0 / 500.0)
    for n in range(start, stop):
        for k in range(19, 0, -1):
            xb[k] = xb[k - 1]
        xb[0] = x[n]
        for j in range(500):
            dot = phase[j]
            for k in range(20):
                dot += omega[j, k] * xb[k]
            z[j] = scale * np.cos(dot)
            q[j] = p2[j] + .5 * p3[j]
        dv = d[n] + v[n]
        ad[0] = .999 * ad[0] + .001 * abs(d[n])
        last_method = 15 if branches_live else 9
        for method in range(last_method):
            if method < 8:
                y, e = step_kron(a[method], b[method], method + 1, mu,
                                 z, q, identity, dv, histories[method])
            elif method == 8:
                y, e = step_full(wfull, mu, z, q, dv, histories[method])
            else:
                branch = method - 9
                rank = 2 + branch % 2
                y, e = step_kron(ba[branch], bb[branch], rank, mu,
                                 z, q, perms[branch], dv, histories[method])
            ytrace[method, n] = y
            etrace[method, n] = e
            ae[method] = .999 * ae[method] + .001 * abs(e)
            if n % SEGMENT >= SEGMENT - TAIL:
                sums[method, n // SEGMENT] += 20.0 * np.log10(
                    (ae[method] + 1e-12) / (ad[0] + 1e-12))
        for j in range(500):
            p3[j] = p2[j]
            p2[j] = p1[j]
            p1[j] = z[j]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_branches(a, b, last_z):
    # R4 state comes only from x,d,v through sample 19999.  No segment index,
    # later input, or run summary enters this construction.
    w = (b[3, :, :4] @ a[3, :, :4].T).T.reshape(500)
    identity = np.arange(500)
    random = np.random.default_rng(2026092815).permutation(500)
    patterns = (("signed_sort", np.argsort(w, kind="stable")),
                ("identity", identity), ("fixed_random", random))
    ba = np.zeros((6, 25, 8), np.float64)
    bb = np.zeros((6, 20, 8), np.float64)
    perms = np.zeros((6, 500), np.int64)
    checks = {}
    for method_index, (name, perm) in enumerate(patterns):
        target = w[perm].reshape(25, 20).T
        u, singular, vh = np.linalg.svd(target, full_matrices=False)
        for rank in (2, 3):
            k = method_index * 2 + rank - 2
            scales = np.sqrt(singular[:rank])
            ba[k, :, :rank] = vh[:rank].T * scales
            bb[k, :, :rank] = u[:, :rank] * scales
            perms[k] = perm
            reconstructed = bb[k, :, :rank] @ ba[k, :, :rank].T
            checks[f"{name}_R{rank}"] = dict(
                relative_initial_truncation=float(np.linalg.norm(target-reconstructed)
                                                  / np.linalg.norm(target)),
                full_output_invariance_error=float(abs(last_z[perm] @ w[perm] - last_z @ w)),
                permutation_sha256=hashlib.sha256(perm.tobytes()).hexdigest())
    return ba, bb, perms, checks


def replay(case, run):
    if case not in MU_BY_CASE or not 0 <= run <= 4:
        raise ValueError("Only E2/E3 dev runs 0..4")
    source = DEV / f"case{case}_run{run:02d}.npz"
    reported_path = SOURCE / f"{case}_run{run:02d}.json"
    with np.load(source, allow_pickle=False) as f:
        x, d, v, omega, phase, a0, b0 = (f[k].copy() for k in
            ("x", "d", "v", "Om", "ph", "initial_A", "initial_B"))
        meta = json.loads(str(f["meta"]))
    reported = json.loads(reported_path.read_text(encoding="utf-8"))
    if meta["case"] != case or meta["run"] != run or len(x) != T:
        raise RuntimeError("Source identity or length mismatch")
    mu = float(meta["mu"])
    if mu != MU_BY_CASE[case]:
        raise RuntimeError("Source step size differs from frozen case setting")
    a = np.zeros((8, 25, 8), np.float64)
    b = np.zeros((8, 20, 8), np.float64)
    for k in range(8):
        a[k] = a0[0]
        b[k] = b0[0]
    full = np.zeros(500)
    ba = np.zeros((6, 25, 8), np.float64)
    bb = np.zeros((6, 20, 8), np.float64)
    perms = np.zeros((6, 500), np.int64)
    histories = np.zeros((15, 3), np.float64)
    ae = np.zeros(15, np.float64)
    ad = np.zeros(1, np.float64)
    xb = np.zeros(20, np.float64)
    p1 = np.zeros(500, np.float64)
    p2 = np.zeros(500, np.float64)
    p3 = np.zeros(500, np.float64)
    ytrace = np.full((15, T), np.nan, np.float64)
    etrace = np.full((15, T), np.nan, np.float64)
    sums = np.zeros((15, 4), np.float64)
    run_section(0, SWITCH, x, d, v, omega[0], phase[0], mu,
                a, b, full, ba, bb, perms, histories, ae, ad,
                xb, p1, p2, p3, ytrace, etrace, sums, False)
    ba, bb, perms, checks = prepare_branches(a, b, p1)
    for k in range(6):
        histories[9+k] = histories[3]
        ae[9+k] = ae[3]
        ytrace[9+k, :SWITCH] = ytrace[3, :SWITCH]
        etrace[9+k, :SWITCH] = etrace[3, :SWITCH]
    run_section(SWITCH, T, x, d, v, omega[0], phase[0], mu,
                a, b, full, ba, bb, perms, histories, ae, ad,
                xb, p1, p2, p3, ytrace, etrace, sums, True)
    anr = sums / TAIL
    if not np.isfinite(anr).all() or not np.isfinite(ytrace).all() or not np.isfinite(etrace).all():
        raise RuntimeError("Non-finite replay trajectory")
    dv = d + v
    gap = 0.0
    for method in range(15):
        y = ytrace[method].astype(np.float64)
        physical = dv.copy()
        physical[2:] -= y[:-2]
        physical[3:] -= .5 * y[:-3]
        gap = max(gap, float(np.max(np.abs(physical - etrace[method]))))
    if gap > 1e-10:
        raise RuntimeError(f"Physical FIR replay discrepancy {gap}")
    if reported["phase"] != "dev" or reported["switch"] != SWITCH:
        raise RuntimeError("Report phase/switch mismatch")
    comparisons = {}
    comparisons["fixed_R4_max_anr_gap_db"] = float(np.max(np.abs(
        anr[3] - np.asarray(reported["fixed_R4_segment_anr_db"]))))
    for name, row in reported["branches"].items():
        idx = METHODS.index(name)
        delta = anr[idx] - anr[3]
        comparisons[name] = dict(
            max_delta_gap_db=float(np.max(np.abs(delta - row["segment_deltas_db"]))),
            truncation_gap=float(abs(checks[name]["relative_initial_truncation"] -
                                     row["check"]["relative_initial_truncation"])),
            output_invariance_gap=float(abs(checks[name]["full_output_invariance_error"] -
                                            row["check"]["full_output_invariance_error"])))
    if comparisons["fixed_R4_max_anr_gap_db"] > 2e-5 or any(
            row["max_delta_gap_db"] > 2e-5 or row["truncation_gap"] > 2e-8
            for name, row in comparisons.items() if name != "fixed_R4_max_anr_gap_db"):
        raise RuntimeError("Independent numerical replay differs from original JSON")
    output = dict(case=case, run=run, phase="dev", T=T, switch=SWITCH, mu=mu,
                  method_names=METHODS, segment_anr_db=anr.tolist(),
                  checks=checks, comparisons_to_original=comparisons,
                  physical_fir_max_abs_gap=gap,
                  source_inputs_sha256=sha(source), original_json_sha256=sha(reported_path),
                  original_script_sha256=sha(SOURCE / "run.py"),
                  replay_script_sha256=sha(__file__))
    target = HERE / f"{case}_run{run:02d}_audit.json"
    temp = target.with_suffix(".tmp.json")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, target)
    print(json.dumps(dict(case=case, run=run, max_gap=comparisons["fixed_R4_max_anr_gap_db"],
                          fir_gap=gap, signed_R2=anr[9].tolist()), ensure_ascii=False), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=tuple(MU_BY_CASE))
    parser.add_argument("run", type=int, choices=range(5))
    args = parser.parse_args()
    replay(args.case, args.run)


if __name__ == "__main__":
    main()
