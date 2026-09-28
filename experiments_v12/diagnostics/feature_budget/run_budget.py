"""Fixed R2/R3 RFF-budget diagnostic on saved E2/E3 development inputs only.

All controllers start from the saved 500-dimensional factor-row prefixes.
No selection/confirm streams or hindsight choice of step size is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
V12 = HERE.parents[1]
ROOT = V12.parent
SOURCE = V12 / "outputs" / "main_dev_round1"
sys.path.insert(0, str(SOURCE / "sources" / "experiments_v10"))
import anc_core as ac

SHAPES = {180: (15, 12), 256: (16, 16), 320: (20, 16),
          400: (20, 20), 500: (25, 20)}
RANKS = (2, 3)
REFERENCE_RANK = 4
REFERENCE_D = 500
M = 20
LS = len(ac.S_PATH)
TAIL = 5000
BETA = .999
EPS_ANR = 1e-12
SOURCE_RUN = 0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete_cost(d1: int, d2: int, rank: int) -> int:
    """Feature projection/scaling, filtered-x, factor updates, and output FIR."""
    return ac.mults_kron(d1, d2, rank, M, LS) + LS


def load_case(case: str):
    assert case in ("E2", "E3")
    protocol_path = SOURCE / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["phase"] == "dev" and protocol["seed"] == 2026092801
    assert case in protocol["cases"]
    source_path = SOURCE / f"case{case}_run{SOURCE_RUN:02d}.npz"
    summary_path = SOURCE / f"case{case}_run{SOURCE_RUN:02d}_summary.json"
    with np.load(source_path) as f:
        arrays = {key: f[key].copy() for key in
                  ("x", "d", "v", "Om", "ph", "initial_A", "initial_B")}
        meta = json.loads(str(f["meta"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert meta["case"] == case and meta["run"] == SOURCE_RUN
    assert meta["seed"] == protocol["seed"]
    assert len(arrays["x"]) == 400000
    assert arrays["Om"].shape == (1, REFERENCE_D, M)
    assert arrays["ph"].shape == (1, REFERENCE_D)
    assert arrays["initial_A"].shape[1:] == (25, 8)
    assert arrays["initial_B"].shape[1:] == (20, 8)
    assert meta["segment_bounds"] == [[i*100000, (i+1)*100000] for i in range(4)]
    return arrays, meta, summary, source_path, summary_path, protocol_path


def calculate(case: str, max_samples: int | None = None):
    arrays, meta, saved, source_path, summary_path, protocol_path = load_case(case)
    x, d, v = (arrays[key] for key in ("x", "d", "v"))
    om, ph = arrays["Om"], arrays["ph"]
    a0, b0 = arrays["initial_A"], arrays["initial_B"]
    mu = .2 if case == "E2" else .1
    assert meta["mu"] == mu
    T = len(x) if max_samples is None else max_samples
    assert 1 <= T <= len(x)
    controllers = []
    specs = []
    for D, (d1, d2) in SHAPES.items():
        assert D == d1*d2
        for rank in RANKS:
            ctrl = ac.KronRFF(f"D{D}_R{rank}", 1, d1, d2, rank,
                              mu/2, mu/2, 2., 1e-8, np.random.default_rng(0))
            ctrl.A = a0[:, :d1, :rank].copy()
            ctrl.B = b0[:, :d2, :rank].copy()
            controllers.append(ctrl)
            specs.append((D, d1, d2, rank))
    J = len(controllers)
    xb = np.zeros((1, M))
    zh = np.zeros((LS, 1, REFERENCE_D))
    Ae = np.zeros(J)
    Ad = 0.
    sums = np.zeros((4, J))
    counts = np.zeros(4, dtype=int)
    started = perf_counter()
    for n in range(T):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z500 = np.sqrt(2/REFERENCE_D)*np.cos(np.einsum("rdm,rm->rd", om, xb)+ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z500
        q500 = zh[2] + .5*zh[3]
        dv = np.array([d[n] + v[n]])
        Ad = BETA*Ad + (1-BETA)*abs(d[n])
        seg = min(3, 4*n//len(x))
        first, end = meta["segment_bounds"][seg]
        tally = n >= max(first, end-TAIL)
        if tally:
            counts[seg] += 1
        for j, (D, _, _, _) in enumerate(specs):
            # The same 500-basis prefix is used; canonical RFF amplitude is
            # sqrt(2/D), hence the exact factor sqrt(500/D) below.
            scale = np.sqrt(REFERENCE_D/D)
            ys = controllers[j].step(scale*z500[:, :D],
                                     scale*q500[:, :D], dv)[0]
            error = dv[0] - ys
            Ae[j] = BETA*Ae[j] + (1-BETA)*abs(error)
            if tally:
                sums[seg, j] += 20*np.log10((Ae[j]+EPS_ANR)/(Ad+EPS_ANR))
    if max_samples is not None:
        return {"sample_count": T, "segment_tail_counts": counts.tolist(),
                "nonfinite_sums": int(np.sum(~np.isfinite(sums)))}
    assert np.array_equal(counts, [TAIL]*4)
    means = sums/counts[:, None]
    assert np.isfinite(means).all()
    reference = np.array([row["fixed_R4"] for row in saved["segment_anr"]])
    reference_cost = complete_cost(25, 20, REFERENCE_RANK)
    assert reference_cost == 19132
    rows = []
    for j, (D, d1, d2, rank) in enumerate(specs):
        cost = complete_cost(d1, d2, rank)
        anr = means[:, j]
        delta = anr-reference
        rows.append(dict(D=D, D1=d1, D2=d2, R=rank, mu=mu,
                         segment_tail_anr_db=anr.tolist(),
                         segment_delta_vs_R4_db=delta.tolist(),
                         mean_segment_delta_db=float(delta.mean()),
                         worst_segment_delta_db=float(delta.max()),
                         complete_mults_per_sample=cost,
                         cost_ratio_vs_R4=cost/reference_cost,
                         active_factor_coeffs=rank*(d1+d2),
                         all_four_within_half_db=bool(np.all(delta <= .5)),
                         under_90_percent_cost=bool(cost <= .9*reference_cost)))
    # At D=500 and default mu, the original independently executed R2 branch
    # must match the saved float32 ANR trace's segment means.
    r2_index = specs.index((500, 25, 20, 2))
    r2_saved = np.array([row["fixed_R2"] for row in saved["segment_anr"]])
    r2_discrepancy = float(np.max(np.abs(means[:, r2_index]-r2_saved)))
    assert r2_discrepancy < 2e-5, r2_discrepancy
    return dict(case=case, run=SOURCE_RUN, source=str(source_path.relative_to(ROOT)),
                source_sha256=sha(source_path), summary_sha256=sha(summary_path),
                protocol_sha256=sha(protocol_path), T=T, mu=mu,
                R4_D500_segment_tail_anr_db=reference.tolist(),
                R4_D500_complete_mults_per_sample=reference_cost,
                R4_D500_active_factor_coeffs=REFERENCE_RANK*(25+20),
                segment_bounds=meta["segment_bounds"], tail_samples=TAIL,
                rows=rows, R2_D500_saved_max_abs_difference_db=r2_discrepancy,
                seconds=perf_counter()-started)


def write_results(results):
    HERE.mkdir(parents=True, exist_ok=True)
    for result in results:
        case = result["case"]
        (HERE / f"{case}_run00.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    import csv
    with (HERE / "segment_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "run", "D", "D1", "D2", "R", "mu", "segment",
                         "tail_ANR_dB", "delta_vs_R4_dB", "R4_tail_ANR_dB",
                         "complete_mults_per_sample", "cost_ratio_vs_R4",
                         "active_factor_coeffs"])
        for result in results:
            for row in result["rows"]:
                for segment in range(4):
                    writer.writerow([result["case"], result["run"], row["D"],
                                     row["D1"], row["D2"], row["R"], result["mu"],
                                     segment+1, row["segment_tail_anr_db"][segment],
                                     row["segment_delta_vs_R4_db"][segment],
                                     result["R4_D500_segment_tail_anr_db"][segment],
                                     row["complete_mults_per_sample"],
                                     row["cost_ratio_vs_R4"],
                                     row["active_factor_coeffs"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("E2", "E3"), action="append")
    args = parser.parse_args()
    selected = args.case or ["E2", "E3"]
    results = [calculate(case) for case in selected]
    write_results(results)
    for result in results:
        print(result["case"], "seconds", round(result["seconds"], 2),
              "R2 D500 baseline max diff dB",
              result["R2_D500_saved_max_abs_difference_db"])
