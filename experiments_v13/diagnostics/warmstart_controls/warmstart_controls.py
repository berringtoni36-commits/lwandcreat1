"""Development-only mechanism controls for a single causal R4 switch at n=20k.

Only saved E2/E3 run00-04 development inputs are used. A continuous online R4
teacher supplies a signed coefficient permutation at the predetermined switch.
All counterfactual branches inherit the physical actuator FIR and ANR state.
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
DEV = ROOT / "experiments_v12" / "outputs" / "main_dev_round1"
sys.path.insert(0, str(DEV / "sources" / "experiments_v10"))
import anc_core as ac

CASES = ("E2", "E3")
RUNS = tuple(range(5))
CASE_INDEX = {"E2": 1, "E3": 2}
SWITCH = 20000  # after samples 0..19999; branch begins at sample 20000
T = 400000
SEGMENT = 100000
TAIL = 5000
D1, D2, D, M = 25, 20, 500, 20
LS = len(ac.S_PATH)
BETA = .999
ANR_EPS = 1e-12
RANDOM_PART = 1301
SVD_CONVENTION_MULTS = 1_000_000
ARMS = ("continuous_R4", "sorted_svd_R2", "sorted_random_R2",
        "sorted_svd_R4", "identity_svd_R2", "sorted_frozen_R2")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(case: str, run: int):
    assert case in CASES and run in RUNS
    protocol_path = DEV / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["phase"] == "dev" and protocol["seed"] == 2026092801
    assert protocol["runs"] >= 5 and case in protocol["cases"]
    source_path = DEV / f"case{case}_run{run:02d}.npz"
    summary_path = DEV / f"case{case}_run{run:02d}_summary.json"
    with np.load(source_path, allow_pickle=False) as f:
        arrays = {key: f[key].copy() for key in
                  ("x", "d", "v", "Om", "ph", "initial_A", "initial_B")}
        meta = json.loads(str(f["meta"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert meta["case"] == case and meta["run"] == run
    assert meta["seed"] == protocol["seed"]
    assert len(arrays["x"]) == len(arrays["d"]) == len(arrays["v"]) == T
    assert arrays["Om"].shape == (1, D, M)
    assert arrays["ph"].shape == (1, D)
    assert arrays["initial_A"].shape[1:] == (D1, 8)
    assert arrays["initial_B"].shape[1:] == (D2, 8)
    assert meta["segment_bounds"] == [[s*SEGMENT, (s+1)*SEGMENT] for s in range(4)]
    return arrays, meta, summary, dict(
        case=case, run=run, phase="dev", source=str(source_path.relative_to(ROOT)),
        source_sha256=sha(source_path), summary_sha256=sha(summary_path),
        protocol_sha256=sha(protocol_path), archived_core_sha256=sha(Path(ac.__file__)))


def full_cost(rank: int) -> int:
    return ac.mults_kron(D1, D2, rank, M, LS) + LS


def cost_record(arm: str):
    r4 = full_cost(4)
    if arm == "sorted_frozen_R2":
        # A frozen branch needs feature projection/scaling and output/FIR only.
        post = D*(M+1) + 2*(D+D2) + LS
    elif arm == "sorted_svd_R4" or arm == "continuous_R4":
        post = r4
    else:
        post = full_cost(2)
    mean = (SWITCH*r4 + (T-SWITCH)*post)/T
    rank = 4 if arm == "sorted_svd_R4" else 2
    if arm == "continuous_R4":
        once = 0
    elif arm == "sorted_random_R2":
        # Form the R4 coefficient vector and scale 90 independent random draws.
        # Generating the random variates themselves is outside this multiply ledger.
        once = D*4 + 2*(D1+D2)
    else:
        once = D*4 + SVD_CONVENTION_MULTS + rank*(D1+D2)
    return dict(pre_switch_R4_complete_mults_per_sample=r4,
                post_switch_complete_mults_per_sample=post,
                average_body_mults_per_sample=mean,
                average_body_ratio_vs_R4=mean/r4,
                one_time_counted_mults=once,
                amortized_counted_mults_per_sample=mean+once/T,
                excluded="500-value sort comparisons/moves, feature gathers, state copy, random-number generation, selector/candidate gate")


def decompose(w: np.ndarray, permutation: np.ndarray, rank: int):
    matrix = w[permutation].reshape(D1, D2).T
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    scale = np.sqrt(singular[:rank])
    B = u[:, :rank]*scale
    A = vh[:rank].T*scale
    projected = B @ A.T
    error = float(np.linalg.norm(matrix-projected)/np.linalg.norm(matrix))
    tail = float(np.linalg.norm(singular[rank:])/np.linalg.norm(singular))
    assert abs(error-tail) < 1e-12
    assert np.linalg.matrix_rank(projected, tol=1e-9) <= rank
    return A, B, dict(relative_frobenius_error=error,
                      singular_tail_check_abs=abs(error-tail),
                      full_permuted_matrix_numeric_rank=int(
                          np.sum(singular > 1e-10*singular[0])))


def kron_branch(name: str, rank: int, mu: float, A: np.ndarray,
                B: np.ndarray, ybuf: np.ndarray):
    ctrl = ac.KronRFF(name, 1, D1, D2, rank,
                      mu/2, mu/2, 2., 1e-8, np.random.default_rng(0))
    ctrl.A = A[None, :, :].copy()
    ctrl.B = B[None, :, :].copy()
    ctrl.ybuf = ybuf.copy()
    return ctrl


class FrozenFactors:
    def __init__(self, A: np.ndarray, B: np.ndarray, ybuf: np.ndarray):
        self.A = A.copy()
        self.B = B.copy()
        self.ybuf = ybuf.copy()

    def step(self, z: np.ndarray) -> float:
        Z = z.reshape(D1, D2).T
        y = float(np.sum(self.B*(Z @ self.A)))
        self.ybuf[:, 1:] = self.ybuf[:, :-1].copy()
        self.ybuf[0, 0] = y
        return float((self.ybuf @ ac.S_PATH)[0])


def init_branches(case: str, run: int, fixed: ac.KronRFF, mu: float,
                  z: np.ndarray, q: np.ndarray):
    started = perf_counter()
    W = fixed.B[0] @ fixed.A[0].T
    w = W.T.reshape(D)
    p = np.argsort(w, kind="stable")
    identity = np.arange(D)
    sort_seconds = perf_counter()-started
    np.testing.assert_array_equal(np.sort(p), identity)
    full_z_error = float(abs(z[0, p] @ w[p] - z[0] @ w))
    full_q_error = float(abs(q[0, p] @ w[p] - q[0] @ w))
    assert max(full_z_error, full_q_error) < 1e-12
    a2, b2, sorted2 = decompose(w, p, 2)
    a4, b4, sorted4 = decompose(w, p, 4)
    ai, bi, identity2 = decompose(w, identity, 2)
    assert sorted2["full_permuted_matrix_numeric_rank"] == 20
    assert identity2["full_permuted_matrix_numeric_rank"] == 4
    random_seed = [2026092801, CASE_INDEX[case], run, RANDOM_PART]
    random_controller = ac.KronRFF("sorted_random_R2", 1, D1, D2, 2,
                                    mu/2, mu/2, 2., 1e-8,
                                    np.random.default_rng(random_seed))
    random_controller.ybuf = fixed.ybuf.copy()
    branches = {
        "sorted_svd_R2": dict(controller=kron_branch("sorted_svd_R2", 2, mu, a2, b2, fixed.ybuf), p=p),
        "sorted_random_R2": dict(controller=random_controller, p=p),
        "sorted_svd_R4": dict(controller=kron_branch("sorted_svd_R4", 4, mu, a4, b4, fixed.ybuf), p=p),
        "identity_svd_R2": dict(controller=kron_branch("identity_svd_R2", 2, mu, ai, bi, fixed.ybuf), p=identity),
        "sorted_frozen_R2": dict(controller=FrozenFactors(a2, b2, fixed.ybuf), p=p),
    }
    inherited_error = max(float(np.max(np.abs(
        state["controller"].ybuf @ ac.S_PATH - fixed.ybuf @ ac.S_PATH)))
        for state in branches.values())
    assert inherited_error == 0.
    projected_control_error = float(abs(
        z[0, p] @ (b2 @ a2.T).T.reshape(D) -
        np.sum(b2*(z[0, p].reshape(D1, D2).T @ a2))))
    assert projected_control_error < 1e-12
    audit = dict(switch_sample=SWITCH-1, first_branch_sample=SWITCH,
                 random_seed_sequence=random_seed, permutation_sort_seconds=sort_seconds,
                 full_output_invariance_max_abs=max(full_z_error, full_q_error),
                 inherited_secondary_FIR_error=inherited_error,
                 projected_control_check_abs=projected_control_error,
                 sorted_R2=sorted2, sorted_R4=sorted4, identity_R2=identity2,
                 sorted_R2_initial_factor_norms=dict(A=float(np.linalg.norm(a2)),
                                                     B=float(np.linalg.norm(b2))),
                 random_R2_initial_factor_norms=dict(
                     A=float(np.linalg.norm(random_controller.A)),
                     B=float(np.linalg.norm(random_controller.B))))
    arrays = dict(permutation=p.astype(np.uint16),
                  R4_original_weights=w, sorted_svd_R2_A=a2,
                  sorted_svd_R2_B=b2, sorted_svd_R4_A=a4,
                  sorted_svd_R4_B=b4, identity_svd_R2_A=ai,
                  identity_svd_R2_B=bi,
                  sorted_random_R2_A=random_controller.A[0].copy(),
                  sorted_random_R2_B=random_controller.B[0].copy())
    return branches, audit, arrays


def run_case(case: str, run: int):
    arrays, meta, saved, provenance = load_case(case, run)
    x, d, v, om, ph = (arrays[k] for k in ("x", "d", "v", "Om", "ph"))
    mu = float(meta["mu"])
    assert mu == (.2 if case == "E2" else .1)
    fixed = ac.KronRFF("continuous_R4", 1, D1, D2, 4,
                       mu/2, mu/2, 2., 1e-8, np.random.default_rng(0))
    fixed.A = arrays["initial_A"][:, :, :4].copy()
    fixed.B = arrays["initial_B"][:, :, :4].copy()
    xb = np.zeros((1, M))
    zh = np.zeros((LS, 1, D))
    ae = {name: 0. for name in ARMS}
    ad = 0.
    sums = np.zeros((4, len(ARMS)))
    branches = {}
    audit = None
    initial_arrays = None
    started = perf_counter()
    for n in range(T):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z = np.sqrt(2/D)*np.cos(np.einsum("rdm,rm->rd", om, xb)+ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2] + .5*zh[3]
        dv = np.array([d[n]+v[n]])
        ad = BETA*ad+(1-BETA)*abs(d[n])
        ys = float(fixed.step(z, q, dv)[0])
        ae["continuous_R4"] = BETA*ae["continuous_R4"]+(1-BETA)*abs(dv[0]-ys)
        if n >= SWITCH:
            for name, state in branches.items():
                perm = state["p"]
                controller = state["controller"]
                if name == "sorted_frozen_R2":
                    branch_ys = controller.step(z[0, perm])
                else:
                    branch_ys = float(controller.step(z[:, perm], q[:, perm], dv)[0])
                ae[name] = BETA*ae[name]+(1-BETA)*abs(dv[0]-branch_ys)
        if n == SWITCH-1:
            branches, audit, initial_arrays = init_branches(case, run, fixed, mu, z, q)
            for name in branches:
                ae[name] = ae["continuous_R4"]
        segment = n//SEGMENT
        if n >= (segment+1)*SEGMENT-TAIL:
            sums[segment] += [20*np.log10((ae[name]+ANR_EPS)/(ad+ANR_EPS))
                              for name in ARMS]
    segment_anr = sums/TAIL
    rows = []
    for segment in range(4):
        ref = float(segment_anr[segment, 0])
        expected = float(saved["segment_anr"][segment]["fixed_R4"])
        discrepancy = abs(ref-expected)
        assert discrepancy < 2e-5, (case, run, segment, discrepancy)
        rows.append(dict(segment=segment+1,
                         bounds=[segment*SEGMENT, (segment+1)*SEGMENT],
                         reference_saved_ANR_db=expected,
                         reference_replay_ANR_db=ref,
                         reference_replay_abs_difference_db=discrepancy,
                         arm_ANR_db={name:float(segment_anr[segment,j])
                                     for j,name in enumerate(ARMS)},
                         delta_vs_R4_db={name:float(segment_anr[segment,j]-ref)
                                         for j,name in enumerate(ARMS) if j}))
    array_path = HERE / f"{case}_run{run:02d}_initialization.npz"
    np.savez_compressed(array_path, **initial_arrays)
    result = dict(case=case, run=run, phase="dev", provenance=provenance,
                  switch_after_samples=SWITCH, arms=list(ARMS),
                  initialization=audit, initialization_file=array_path.name,
                  initialization_file_sha256=sha(array_path),
                  costs={name:cost_record(name) for name in ARMS},
                  segments=rows, total_seconds=perf_counter()-started)
    json_path = HERE / f"{case}_run{run:02d}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    return result


def cache_or_run(case: str, run: int, refresh: bool):
    path = HERE / f"{case}_run{run:02d}.json"
    if path.exists() and not refresh:
        result = json.loads(path.read_text(encoding="utf-8"))
        _, _, _, current = load_case(case, run)
        assert result["provenance"] == current
        assert result["arms"] == list(ARMS)
        assert sha(HERE / result["initialization_file"]) == result["initialization_file_sha256"]
        expected_costs = {name:cost_record(name) for name in ARMS}
        if result["costs"] != expected_costs:
            result["costs"] = expected_costs
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        return result
    return run_case(case, run)


def write_aggregate(results: list[dict]):
    target = HERE / "segment_results.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "run", "segment", "arm", "ANR_db", "delta_vs_R4_db",
                         "body_mults_per_sample", "amortized_counted_mults_per_sample"))
        for result in results:
            for segment in result["segments"]:
                for arm in ARMS:
                    writer.writerow((result["case"], result["run"], segment["segment"], arm,
                                     segment["arm_ANR_db"][arm],
                                     segment["delta_vs_R4_db"].get(arm, 0.),
                                     result["costs"][arm]["average_body_mults_per_sample"],
                                     result["costs"][arm]["amortized_counted_mults_per_sample"]))
    summary = dict(cases=list(CASES), runs=sorted(set(r["run"] for r in results)),
                   results=[dict(case=r["case"], run=r["run"],
                                 source_sha256=r["provenance"]["source_sha256"],
                                 max_reference_abs_difference_db=max(
                                     s["reference_replay_abs_difference_db"]
                                     for s in r["segments"]),
                                 mean_segment_delta_db={arm:float(np.mean([
                                     s["delta_vs_R4_db"][arm] for s in r["segments"]]))
                                     for arm in ARMS[1:]}) for r in results])
    (HERE / "aggregate.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--runs", type=int, nargs="+", choices=RUNS)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    cases = args.case or list(CASES)
    runs = args.runs or list(RUNS)
    results = []
    for case in cases:
        for run in runs:
            result = cache_or_run(case, run, args.refresh)
            results.append(result)
            print(case, run, round(result["total_seconds"], 2), flush=True)
    write_aggregate(results)
