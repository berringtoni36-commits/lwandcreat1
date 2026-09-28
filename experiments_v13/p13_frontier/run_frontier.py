"""P13 fixed-rank capacity gate, development streams only.

Run one seed at a time with ``--run 0`` (or ``--run 1`` etc.).  The numerical
kernel is checked against the unchanged v10 implementation by test_frontier.py.
No selector, dynamic controller, or confirmation stream is available here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from time import perf_counter

import numpy as np

DEPS = Path(tempfile.gettempdir()) / "n13_frontier_deps"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
from numba import njit
import numba

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments_v13" / "protocol"))
from generator import STAGES, make_p13  # noqa: E402

MU = np.array([0.05, 0.10, 0.20, 0.40, 0.80], dtype=np.float64)
RANKS = tuple(range(1, 9))
N_METHODS = 45
TAIL = 10_000
SOURCES = (
    ROOT / "experiments_v13" / "protocol" / "PREREGISTRATION.md",
    ROOT / "experiments_v13" / "protocol" / "generator.py",
    ROOT / "experiments_v13" / "p13_frontier" / "run_frontier.py",
    ROOT / "experiments_v13" / "p13_frontier" / "test_frontier.py",
    ROOT / "experiments_v10" / "anc_core.py",
)


@njit(cache=True)
def simulate_fixed(x, d, v, omega, phase, a0, b0, mu, stage_ends, tail):
    """All 8 x 5 nested-factor NKP-MCC and 5 full500 MCC controls.

    Factor updates are B then A, and the microphone uses S=[0,0,1,.5].
    Arrays of every actuator sample and physical error permit FIR replay.
    """
    tmax = x.size
    a = np.zeros((40, 25, 8), np.float64)
    b = np.zeros((40, 20, 8), np.float64)
    for c in range(40):
        for i in range(25):
            for r in range(8):
                a[c, i, r] = a0[i, r]
        for j in range(20):
            for r in range(8):
                b[c, j, r] = b0[j, r]
    w = np.zeros((5, 500), np.float64)
    xb = np.zeros(20, np.float64)
    z = np.zeros(500, np.float64)
    p1 = np.zeros(500, np.float64)
    p2 = np.zeros(500, np.float64)
    p3 = np.zeros(500, np.float64)
    q = np.zeros(500, np.float64)
    y1 = np.zeros(N_METHODS, np.float64)
    y2 = np.zeros(N_METHODS, np.float64)
    y3 = np.zeros(N_METHODS, np.float64)
    ae = np.zeros(N_METHODS, np.float64)
    ad = 0.0
    y_trace = np.empty((N_METHODS, tmax), np.float32)
    e_trace = np.empty((N_METHODS, tmax), np.float32)
    anr_sum = np.zeros((N_METHODS, 3), np.float64)
    anr_count = np.zeros(3, np.int64)
    ub = np.zeros((20, 8), np.float64)
    ua = np.zeros((25, 8), np.float64)
    scale = np.sqrt(2.0 / 500.0)
    for n in range(tmax):
        for k in range(19, 0, -1):
            xb[k] = xb[k - 1]
        xb[0] = x[n]
        for j in range(500):
            dot = phase[j]
            for k in range(20):
                dot += omega[j, k] * xb[k]
            z[j] = scale * np.cos(dot)
            q[j] = p2[j] + 0.5 * p3[j]
        dv = d[n] + v[n]
        ad = 0.999 * ad + 0.001 * abs(d[n])
        stage = 0 if n < stage_ends[0] else 1 if n < stage_ends[1] else 2
        in_tail = n >= stage_ends[stage] - tail
        if in_tail:
            anr_count[stage] += 1
        for c in range(40):
            rank = c // 5 + 1
            m = c % 5
            output = 0.0
            pred1 = 0.0
            norm_b = 0.0
            for r in range(rank):
                for j in range(20):
                    zj = 0.0
                    qj = 0.0
                    for i in range(25):
                        index = 20 * i + j
                        zj += z[index] * a[c, i, r]
                        qj += q[index] * a[c, i, r]
                    ub[j, r] = qj
                    output += b[c, j, r] * zj
                    pred1 += b[c, j, r] * qj
                    norm_b += qj * qj
            ys = y2[c] + 0.5 * y3[c]
            err = dv - ys
            y_trace[c, n] = output
            e_trace[c, n] = err
            ae[c] = 0.999 * ae[c] + 0.001 * abs(err)
            if in_tail:
                anr_sum[c, stage] += 20.0 * np.log10((ae[c] + 1e-12) / (ad + 1e-12))
            y3[c] = y2[c]
            y2[c] = y1[c]
            y1[c] = output
            residual1 = dv - pred1
            psi1 = residual1 * np.exp(-residual1 * residual1 / 8.0)
            gain_b = (mu[m] / 2.0) * psi1 / (1e-8 + norm_b)
            for r in range(rank):
                for j in range(20):
                    b[c, j, r] += gain_b * ub[j, r]
            pred2 = 0.0
            norm_a = 0.0
            for r in range(rank):
                for i in range(25):
                    qi = 0.0
                    for j in range(20):
                        qi += q[20 * i + j] * b[c, j, r]
                    ua[i, r] = qi
                    pred2 += a[c, i, r] * qi
                    norm_a += qi * qi
            residual2 = dv - pred2
            psi2 = residual2 * np.exp(-residual2 * residual2 / 8.0)
            gain_a = (mu[m] / 2.0) * psi2 / (1e-8 + norm_a)
            for r in range(rank):
                for i in range(25):
                    a[c, i, r] += gain_a * ua[i, r]
        normq = 0.0
        for j in range(500):
            normq += q[j] * q[j]
        for m in range(5):
            c = 40 + m
            output = 0.0
            prediction = 0.0
            for j in range(500):
                output += w[m, j] * z[j]
                prediction += w[m, j] * q[j]
            ys = y2[c] + 0.5 * y3[c]
            err = dv - ys
            y_trace[c, n] = output
            e_trace[c, n] = err
            ae[c] = 0.999 * ae[c] + 0.001 * abs(err)
            if in_tail:
                anr_sum[c, stage] += 20.0 * np.log10((ae[c] + 1e-12) / (ad + 1e-12))
            y3[c] = y2[c]
            y2[c] = y1[c]
            y1[c] = output
            residual = dv - prediction
            psi = residual * np.exp(-residual * residual / 8.0)
            gain = mu[m] * psi / (1e-8 + normq)
            for j in range(500):
                w[m, j] += gain * q[j]
        for j in range(500):
            p3[j] = p2[j]
            p2[j] = p1[j]
            p1[j] = z[j]
    return anr_sum, anr_count, y_trace, e_trace


def source_hashes():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCES}


def initial_factors(run):
    rng = np.random.default_rng(np.random.SeedSequence([20260928, 13, 1, 1, run, 3]))
    a = 0.01 * rng.standard_normal((25, 8))
    b = 0.01 * rng.standard_normal((20, 8))
    return a, b


def method_names():
    return [f"R{rank}_mu{mu:.2f}" for rank in RANKS for mu in MU] + [f"full500_mu{mu:.2f}" for mu in MU]


def costs():
    detail = {}
    for rank in RANKS:
        detail[f"R{rank}"] = dict(shared_rff_projection=10_000,
                                 shared_rff_scaling=500, filtered_x=2_000,
                                 factor_output=520 * rank,
                                 factor_filtered_regressors=1_000 * rank,
                                 predictions_norms_increments=135 * rank,
                                 scalar_core=8, physical_output_fir=4,
                                 total_per_sample=12_512 + 1_655 * rank)
    detail["full500"] = dict(shared_rff_projection=10_000,
                             shared_rff_scaling=500, filtered_x=2_000,
                             output_residual_norm_increment=2_000,
                             scalar_core=4, physical_output_fir=4,
                             total_per_sample=14_508)
    return detail


def assess_capacity(anr):
    """Optimistic per-stage, per-rank minimum over the frozen five-step grid."""
    rank_anr = np.min(anr[:40].reshape(8, 5, 3), axis=1)
    best = np.min(rank_anr, axis=0)
    required = []
    for s in range(3):
        feasible = np.flatnonzero((rank_anr[:, s] <= best[s] + 0.5) &
                                  (rank_anr[:, s] <= -6.0))
        required.append(int(feasible[0] + 1) if len(feasible) else None)
    r4_floor = bool(np.all(rank_anr[3] <= -6.0))
    pass_run = bool(r4_floor and required[0] is not None and required[1] is not None
                    and required[2] is not None and required[0] <= 2
                    and 3 <= required[1] <= 4 and required[2] <= 2)
    return dict(required_rank_by_stage=required,
                rank_oracle_anr_db=rank_anr.tolist(), best_anr_db=best.tolist(),
                fixed_R4_absolute_floor=r4_floor, capacity_gate_pass=pass_run)


def atomic_json(path, value):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def ensure_manifest():
    HERE.mkdir(parents=True, exist_ok=True)
    manifest = HERE / "manifest.json"
    content = dict(phase="dev", case="P13", run_ids=list(range(5)), stages=list(STAGES),
                   tail_samples=TAIL, mu=MU.tolist(), ranks=list(RANKS),
                   source_sha256=source_hashes(), python=sys.version,
                   numpy=np.__version__, numba=numba.__version__,
                   platform=platform.platform(), cost_ledger=costs(),
                   interpretation="Development capacity gate only; no selection or confirmation")
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != content:
            raise RuntimeError("Frozen frontier source/environment differs; do not mix results")
    else:
        atomic_json(manifest, content)
        for source in SOURCES:
            target = HERE / "sources" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return content


def run_one(run):
    if not 0 <= run < 5:
        raise ValueError("Only P13 dev runs 0..4 are allowed")
    frozen = ensure_manifest()
    outdir = HERE / f"run{run:02d}"
    outdir.mkdir(exist_ok=True)
    result_path = outdir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    start = perf_counter()
    data = make_p13("dev", run)
    a0, b0 = initial_factors(run)
    input_path = outdir / "inputs.npz"
    if input_path.exists():
        with np.load(input_path) as prior:
            for name, expected in (("x", data.x), ("d", data.d), ("v", data.v),
                                   ("omega", data.omega), ("rff_phase", data.rff_phase),
                                   ("initial_A", a0), ("initial_B", b0)):
                if not np.array_equal(prior[name], expected):
                    raise RuntimeError(f"Saved input {name} differs; refusing resume")
    else:
        temporary = outdir / "inputs.tmp.npz"
        np.savez_compressed(temporary, x=data.x, d=data.d, v=data.v,
                            omega=data.omega, rff_phase=data.rff_phase,
                            initial_A=a0, initial_B=b0,
                            active_interactions=data.active_interactions)
        os.replace(temporary, input_path)
    if data.x.size != sum(STAGES):
        raise RuntimeError("Full pre-registered P13 length required")
    stage_ends = np.array([STAGES[0], STAGES[0] + STAGES[1], sum(STAGES)], dtype=np.int64)
    counts, sample_count, y, error = simulate_fixed(
        data.x, data.d, data.v, data.omega, data.rff_phase, a0, b0, MU,
        stage_ends, TAIL)
    if not np.array_equal(sample_count, np.full(3, TAIL)):
        raise AssertionError("Wrong ANR tail counts")
    if not np.isfinite(counts).all() or not np.isfinite(y).all() or not np.isfinite(error).all():
        raise AssertionError("Non-finite fixed control trajectory")
    # Independent physical FIR replay from saved actuator outputs.
    replay = np.empty_like(error)
    dv = data.d + data.v
    for c in range(N_METHODS):
        yc = y[c].astype(np.float64)
        ys = np.zeros(data.x.size, dtype=np.float64)
        ys[2:] += yc[:-2]
        ys[3:] += 0.5 * yc[:-3]
        replay[c] = dv - ys
    fir_gap = float(np.max(np.abs(replay - error)))
    if fir_gap > 1e-5:
        raise AssertionError(f"Physical FIR replay mismatch {fir_gap}")
    if source_hashes() != frozen["source_sha256"]:
        raise RuntimeError("Source changed during run; refusing result")
    traces_path = outdir / "traces.npz"
    temporary = outdir / "traces.tmp.npz"
    np.savez_compressed(temporary, actuator_output=y, physical_error=error,
                        method_names=np.array(method_names()),
                        segment_anr_db=counts / sample_count)
    os.replace(temporary, traces_path)
    anr = counts / sample_count
    capacity = assess_capacity(anr)
    result = dict(phase="dev", case="P13", run=run, T=data.x.size,
                  method_names=method_names(), mu=MU.tolist(),
                  stage_bounds=[[0, int(stage_ends[0])], [int(stage_ends[0]), int(stage_ends[1])],
                                [int(stage_ends[1]), int(stage_ends[2])]],
                  segment_anr_db=anr.tolist(), capacity=capacity,
                  cost_ledger=frozen["cost_ledger"],
                  costs_total_multiplications={name: int(row["total_per_sample"] * data.x.size)
                                               for name, row in frozen["cost_ledger"].items()},
                  physical_fir_max_abs_gap=fir_gap,
                  source_sha256=frozen["source_sha256"],
                  inputs_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  traces_sha256=hashlib.sha256(traces_path.read_bytes()).hexdigest(),
                  elapsed_seconds=perf_counter() - start)
    atomic_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, required=True, choices=range(5))
    args = parser.parse_args()
    result = run_one(args.run)
    print(json.dumps(dict(run=result["run"],
                          required=result["capacity"]["required_rank_by_stage"],
                          pass_run=result["capacity"]["capacity_gate_pass"],
                          r4_floor=result["capacity"]["fixed_R4_absolute_floor"],
                          seconds=result["elapsed_seconds"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
