"""Causal-at-switch R4-to-R2/R3 permutation diagnostic on E2/E3 dev run00.

Segment boundaries and switch horizons are prescribed in advance. The
permutation at each switch uses only the *current* R4 factor coefficients.
Separate branches then learn online through the remainder of that segment.
This is an offline known-boundary upper-limit diagnostic, not an online gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V12 = ROOT / "experiments_v12"
DEV = V12 / "outputs" / "main_dev_round1"
sys.path.insert(0, str(DEV / "sources" / "experiments_v10"))
import anc_core as ac

CASES = ("E2", "E3")
HORIZONS = (20000, 50000)
RANKS = (2, 3)
METHODS = ("current_signed_sort", "first_segment_signed_sort",
           "identity", "fixed_random")
FIXED_RANDOM_SEED = 2026092814
SEGMENT = 100000
NSEG = 4
D1, D2, D, M = 25, 20, 500, 20
LS = len(ac.S_PATH)
TAIL = 5000
BETA = .999
ANR_EPS = 1e-12
SVD_CONVENTION_MULTS = 1_000_000  # v12's conservative library-SVD charge


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(case: str):
    assert case in CASES
    protocol_path = DEV / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["phase"] == "dev" and case in protocol["cases"]
    source_path = DEV / f"case{case}_run00.npz"
    summary_path = DEV / f"case{case}_run00_summary.json"
    with np.load(source_path, allow_pickle=False) as f:
        x, d, v, om, ph, a0, b0 = (f[key].copy() for key in
                                    ("x", "d", "v", "Om", "ph", "initial_A", "initial_B"))
        meta = json.loads(str(f["meta"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(x) == len(d) == len(v) == NSEG*SEGMENT
    assert om.shape == (1, D, M) and ph.shape == (1, D)
    assert a0.shape[1:] == (D1, 8) and b0.shape[1:] == (D2, 8)
    assert meta["case"] == case and meta["run"] == 0
    assert meta["seed"] == protocol["seed"] == 2026092801
    assert meta["segment_bounds"] == [[s*SEGMENT, (s+1)*SEGMENT] for s in range(NSEG)]
    return dict(x=x, d=d, v=v, Om=om, ph=ph, A0=a0, B0=b0,
                meta=meta, summary=summary,
                provenance=dict(case=case, run=0, phase="dev",
                                source=str(source_path.relative_to(ROOT)),
                                source_sha256=sha(source_path),
                                summary_sha256=sha(summary_path),
                                protocol_sha256=sha(protocol_path),
                                core_sha256=sha(Path(ac.__file__))))


def costs(rank: int, horizon: int):
    r4 = ac.mults_kron(D1, D2, 4, M, LS) + LS
    reduced = ac.mults_kron(D1, D2, rank, M, LS) + LS
    body = (horizon*r4+(SEGMENT-horizon)*reduced)/SEGMENT
    # W=B A^T needs D*4 products. SVD cost is deliberately a convention, not
    # an exact flop count; sort comparisons/gathers and factor remapping remain
    # outside this arithmetic ledger and are reported separately.
    one_time = D*4 + rank*(D1+D2) + SVD_CONVENTION_MULTS
    return dict(R4_complete_mults_per_sample=r4,
                reduced_complete_mults_per_sample=reduced,
                average_R_body_mults_per_sample=body,
                average_R_body_ratio_vs_R4=body/r4,
                W_assembly_mults_once=D*4,
                SVD_convention_mults_once=SVD_CONVENTION_MULTS,
                factor_scale_mults_once=rank*(D1+D2),
                one_time_counted_mults=one_time,
                amortized_counted_mults_per_sample=body+one_time/SEGMENT,
                permutation_indices_uint16_bytes=2*D)


def make_branch(fixed: ac.KronRFF, w: np.ndarray, p: np.ndarray,
                rank: int, mu: float, z: np.ndarray, q: np.ndarray):
    assert np.array_equal(np.sort(p), np.arange(D))
    start = perf_counter()
    full_z_error = float(abs(z[0, p] @ w[p] - z[0] @ w))
    full_q_error = float(abs(q[0, p] @ w[p] - q[0] @ w))
    assert full_z_error < 1e-12 and full_q_error < 1e-12
    Wp = w[p].reshape(D1, D2).T
    u, singular, vh = np.linalg.svd(Wp, full_matrices=False)
    full_permuted_rank = int(np.sum(singular > 1e-10*singular[0]))
    scale = np.sqrt(singular[:rank])
    B = u[:, :rank]*scale
    A = vh[:rank].T*scale
    reconstruction = B @ A.T
    relative_error = float(np.linalg.norm(Wp-reconstruction)/np.linalg.norm(Wp))
    expected_error = float(np.linalg.norm(singular[rank:])/np.linalg.norm(singular))
    assert abs(relative_error-expected_error) < 1e-12
    assert np.linalg.matrix_rank(reconstruction, tol=1e-9) <= rank
    # Frozen output at the switch, before further learning. This checks that
    # factors and feature coordinates agree after the reordering.
    frozen_y = float(np.sum(B*(z[0, p].reshape(D1, D2).T @ A)))
    direct_y = float(z[0, p] @ reconstruction.T.reshape(D))
    assert abs(frozen_y-direct_y) < 1e-12
    elapsed = perf_counter()-start
    branch = ac.KronRFF(f"causal_R{rank}", 1, D1, D2, rank,
                        mu/2, mu/2, 2., 1e-8, np.random.default_rng(0))
    branch.A = A[None, :, :].copy()
    branch.B = B[None, :, :].copy()
    branch.ybuf = fixed.ybuf.copy()
    inherited_fir_error = float(np.max(np.abs(
        branch.ybuf @ ac.S_PATH - fixed.ybuf @ ac.S_PATH)))
    assert inherited_fir_error == 0.
    return branch, dict(relative_frobenius_error=relative_error,
                        full_permuted_weight_matrix_rank=full_permuted_rank,
                        initial_frozen_control_delta=float(frozen_y-z[0] @ w),
                        full_output_invariance_max_abs=max(full_z_error, full_q_error),
                        svd_tail_reconstruction_abs_error=abs(relative_error-expected_error),
                        inherited_secondary_FIR_error=inherited_fir_error,
                        permutation_and_svd_seconds=elapsed)


def run_case(case: str):
    f = load(case)
    x, d, v, om, ph = (f[k] for k in ("x", "d", "v", "Om", "ph"))
    mu = float(f["meta"]["mu"])
    assert mu == (.2 if case == "E2" else .1)
    fixed = ac.KronRFF("continuous_R4", 1, D1, D2, 4,
                       mu/2, mu/2, 2., 1e-8, np.random.default_rng(0))
    fixed.A = f["A0"][:, :, :4].copy()
    fixed.B = f["B0"][:, :, :4].copy()
    branches = {}
    first_permutations = {}
    identity = np.arange(D)
    fixed_random = np.random.default_rng(FIXED_RANDOM_SEED).permutation(D)
    permutation_records = []
    segments = []
    segment_sums = {}
    segment_start_info = {}
    ad = 0.
    fixed_ae = 0.
    xb = np.zeros((1, M))
    zh = np.zeros((LS, 1, D))
    started = perf_counter()
    for n in range(len(x)):
        segment = n//SEGMENT
        within = n%SEGMENT
        if within == 0:
            branches = {}
            segment_sums = {"continuous_R4": 0.}
            segment_start_info = {}
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z = np.sqrt(2/D)*np.cos(np.einsum("rdm,rm->rd", om, xb)+ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2] + .5*zh[3]
        dv = np.array([d[n]+v[n]])
        ad = BETA*ad + (1-BETA)*abs(d[n])
        fixed_ys = float(fixed.step(z, q, dv)[0])
        fixed_ae = BETA*fixed_ae + (1-BETA)*abs(dv[0]-fixed_ys)
        if within >= SEGMENT-TAIL:
            segment_sums["continuous_R4"] += 20*np.log10((fixed_ae+ANR_EPS)/(ad+ANR_EPS))
        for key, state in branches.items():
            p = state["permutation"]
            ys = float(state["controller"].step(z[:, p], q[:, p], dv)[0])
            state["ae"] = BETA*state["ae"] + (1-BETA)*abs(dv[0]-ys)
            if within >= SEGMENT-TAIL:
                segment_sums[key] += 20*np.log10((state["ae"]+ANR_EPS)/(ad+ANR_EPS))
        if within+1 in HORIZONS:
            horizon = within+1
            assembly_started = perf_counter()
            W = fixed.B[0] @ fixed.A[0].T
            w = W.T.reshape(D)
            assembly_seconds = perf_counter()-assembly_started
            sort_started = perf_counter()
            p_current = np.argsort(w, kind="stable")
            sort_seconds = perf_counter()-sort_started
            if segment == 0:
                first_permutations[horizon] = p_current.copy()
            for method in METHODS:
                if method == "current_signed_sort":
                    p = p_current
                elif method == "first_segment_signed_sort":
                    p = first_permutations[horizon]
                elif method == "identity":
                    p = identity
                else:
                    p = fixed_random
                for rank in RANKS:
                    key = f"h{horizon}_{method}_R{rank}"
                    branch, audit = make_branch(fixed, w, p, rank, mu, z, q)
                    branches[key] = {"controller": branch,
                                     "permutation": p.copy(), "ae": fixed_ae}
                    segment_sums[key] = 0.
                    audit.update(case=case, segment=segment+1,
                                 switch_sample=n, horizon=horizon,
                                 method=method, R=rank,
                                 W_assembly_seconds=assembly_seconds,
                                 weight_sort_seconds_once=(sort_seconds if
                                     method == "current_signed_sort" or
                                     (method == "first_segment_signed_sort" and segment == 0)
                                     else 0.),
                                 weight_sort_required=(method == "current_signed_sort" or
                                     (method == "first_segment_signed_sort" and segment == 0)),
                                 start_AE_from_continuous_R4=fixed_ae,
                                 costs=costs(rank, horizon))
                    permutation_records.append(p.astype(np.uint16))
                    segment_start_info[key] = audit
        if within == SEGMENT-1:
            mean = {name: value/TAIL for name, value in segment_sums.items()}
            expected = f["summary"]["segment_anr"][segment]["fixed_R4"]
            reference_error = abs(mean["continuous_R4"]-expected)
            assert reference_error < 2e-5, (case, segment, reference_error)
            rows = []
            for key, audit in segment_start_info.items():
                audit["tail_ANR_db"] = mean[key]
                audit["delta_vs_continuous_R4_db"] = mean[key]-mean["continuous_R4"]
                rows.append(audit)
            segments.append(dict(segment=segment+1,
                                 bounds=[segment*SEGMENT, (segment+1)*SEGMENT],
                                 reference_tail_ANR_db=mean["continuous_R4"],
                                 saved_reference_tail_ANR_db=expected,
                                 reference_replay_abs_difference_db=reference_error,
                                 branches=rows))
            print(case, "segment", segment+1, "done", flush=True)
    permutations = np.asarray(permutation_records, dtype=np.uint16)
    assert permutations.shape == (NSEG*len(HORIZONS)*len(METHODS)*len(RANKS), D)
    np.savez_compressed(HERE / f"{case}_run00_permutations.npz", permutations=permutations)
    result = dict(case=case, run=0, phase="dev", provenance=f["provenance"],
                  horizons=list(HORIZONS), methods=list(METHODS), ranks=list(RANKS),
                  fixed_random_seed=FIXED_RANDOM_SEED,
                  permutations_file=f"{case}_run00_permutations.npz",
                  permutations_sha256=sha(HERE / f"{case}_run00_permutations.npz"),
                  total_seconds=perf_counter()-started, segments=segments)
    (HERE / f"{case}_run00.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(result)
    return result


def write_csv(result: dict):
    target = HERE / f"{result['case']}_run00_segments.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "segment", "horizon", "method", "rank",
                         "continuous_R4_ANR_db", "branch_ANR_db", "delta_db",
                         "relative_frobenius_error", "permuted_weight_matrix_rank",
                         "weight_sort_required", "average_R_body_mults_per_sample",
                         "amortized_counted_mults_per_sample",
                         "full_output_invariance_max_abs",
                         "svd_tail_reconstruction_abs_error"))
        for segment in result["segments"]:
            for branch in segment["branches"]:
                writer.writerow((result["case"], segment["segment"], branch["horizon"],
                                 branch["method"], branch["R"],
                                 segment["reference_tail_ANR_db"], branch["tail_ANR_db"],
                                 branch["delta_vs_continuous_R4_db"],
                                 branch["relative_frobenius_error"],
                                 branch["full_permuted_weight_matrix_rank"],
                                 branch["weight_sort_required"],
                                 branch["costs"]["average_R_body_mults_per_sample"],
                                 branch["costs"]["amortized_counted_mults_per_sample"],
                                 branch["full_output_invariance_max_abs"],
                                 branch["svd_tail_reconstruction_abs_error"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, action="append")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    for case in (args.case or CASES):
        result = run_case(case)
        print(case, "seconds", round(result["total_seconds"], 2), flush=True)
