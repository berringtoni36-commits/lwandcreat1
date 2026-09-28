"""E1 development-only capacity geometry and boundary-oracle replay.

The sole feature permutation is derived from a continuously trained R4 at
sample 19,999 of the first low-rank segment. It is frozen for all 300k samples.
Ground-truth teacher weights are reconstructed only for auditing geometry and
the saved disturbance; they never select a permutation or initialize a model.
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
DEV = ROOT / "experiments_v12" / "outputs" / "mechanism_dev_residual_only"
sys.path.insert(0, str(DEV / "sources" / "experiments_v10"))
import anc_core as ac

RUNS = tuple(range(5))
SEED = 2026092801
T, SEGMENT, SWITCH, TAIL = 300000, 100000, 20000, 5000
D1, D2, D, M, LS = 25, 20, 500, 20, len(ac.S_PATH)
BETA, EPS = .999, 1e-12
GROW_PART = 1401
SVD_CONVENTION_MULTS = 1_000_000
ARMS = ("original_R1", "original_R2", "original_R4",
        "sorted_R2", "sorted_R4", "boundary_R2_R4_R2")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(run: int):
    assert run in RUNS
    protocol_path = DEV / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["phase"] == "dev" and protocol["seed"] == SEED
    assert protocol["cases"] == ["E1"] and protocol["lengths"]["E1"] == T
    source = DEV / f"caseE1_run{run:02d}.npz"
    summary_path = DEV / f"caseE1_run{run:02d}_summary.json"
    with np.load(source, allow_pickle=False) as f:
        arrays = {k:f[k].copy() for k in ("x", "d", "v", "Om", "ph", "initial_A", "initial_B")}
        meta = json.loads(str(f["meta"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert meta["case"] == "E1" and meta["run"] == run
    assert meta["seed"] == SEED and meta["T"] == T and meta["mu"] == .2
    assert meta["segment_bounds"] == [[0, SEGMENT], [SEGMENT, 2*SEGMENT], [2*SEGMENT, T]]
    assert len(arrays["x"]) == len(arrays["d"]) == len(arrays["v"]) == T
    assert arrays["Om"].shape == (1, D, M) and arrays["ph"].shape == (1, D)
    assert arrays["initial_A"].shape == (1, D1, 8)
    assert arrays["initial_B"].shape == (1, D2, 8)
    provenance = dict(phase="dev", case="E1", run=run,
                      source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                      summary_sha256=sha(summary_path), protocol_sha256=sha(protocol_path),
                      archived_core_sha256=sha(Path(ac.__file__)))
    return arrays, meta, summary, provenance


def known_teacher(run: int):
    # Exact archived v12 generator: U,V from a separate part=5 seed. Audit only.
    rng = np.random.default_rng([SEED, 0, run, 5])
    U, _ = np.linalg.qr(rng.standard_normal((D2, 4)))
    V, _ = np.linalg.qr(rng.standard_normal((D1, 4)))
    low = 10*np.outer(U[:, 0], V[:, 0])
    high = 5*(U @ V.T)
    assert np.linalg.matrix_rank(low, tol=1e-9) == 1
    assert np.linalg.matrix_rank(high, tol=1e-9) == 4
    return low, high


def controller(name: str, rank: int, A: np.ndarray, B: np.ndarray,
               ybuf: np.ndarray | None = None):
    c = ac.KronRFF(name, 1, D1, D2, rank, .1, .1, 2., 1e-8,
                   np.random.default_rng(0))
    c.A = A[None, :, :].copy()
    c.B = B[None, :, :].copy()
    if ybuf is not None:
        c.ybuf = ybuf.copy()
    return c


def factors(w: np.ndarray, p: np.ndarray, rank: int):
    matrix = w[p].reshape(D1, D2).T
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    A = vh[:rank].T*np.sqrt(s[:rank])
    B = u[:, :rank]*np.sqrt(s[:rank])
    relative = float(np.linalg.norm(matrix-B@A.T)/np.linalg.norm(matrix))
    tail = float(np.linalg.norm(s[rank:])/np.linalg.norm(s))
    assert abs(relative-tail) < 1e-12
    return A, B, relative


def geometry(matrix: np.ndarray, p: np.ndarray):
    reordered = matrix.T.reshape(D)[p].reshape(D1, D2).T
    s = np.linalg.svd(reordered, compute_uv=False)
    norm = np.linalg.norm(s)
    return dict(numeric_rank=int(np.sum(s > 1e-10*s[0])),
                relative_best_R1=float(np.linalg.norm(s[1:])/norm),
                relative_best_R2=float(np.linalg.norm(s[2:])/norm),
                relative_best_R4=float(np.linalg.norm(s[4:])/norm))


def complete_cost(rank: int):
    return ac.mults_kron(D1, D2, rank, M, LS)+LS


def costs():
    c1, c2, c4 = (complete_cost(r) for r in (1, 2, 4))
    plan = {
        "original_R1": ([c1]*3, 0),
        "original_R2": ([c2]*3, 0),
        "original_R4": ([c4]*3, 0),
        "sorted_R2": ([(SWITCH*c4+(SEGMENT-SWITCH)*c2)/SEGMENT, c2, c2],
                      D*4+SVD_CONVENTION_MULTS+2*(D1+D2)),
        "sorted_R4": ([c4]*3, D*4+SVD_CONVENTION_MULTS+4*(D1+D2)),
        "boundary_R2_R4_R2": ([(SWITCH*c4+(SEGMENT-SWITCH)*c2)/SEGMENT, c4, c2],
                              2*(D*4+SVD_CONVENTION_MULTS+2*(D1+D2)) + 2*D1),
    }
    return {name:dict(segment_body_mults_per_sample=parts,
                      mean_body_mults_per_sample=float(np.mean(parts)),
                      one_time_counted_mults=once,
                      amortized_counted_mults_per_sample=float(np.mean(parts)+once/T),
                      excluded="sort comparisons/moves, feature gathers, memory/state copy, RNG generation, online gate and candidate training")
            for name, (parts, once) in plan.items()}


def run_case(run: int):
    arrays, meta, summary, provenance = load_run(run)
    low, high = known_teacher(run)
    x, d, v, om, ph = (arrays[k] for k in ("x", "d", "v", "Om", "ph"))
    originals = {f"original_R{rank}":controller(f"original_R{rank}", rank,
                  arrays["initial_A"][0, :, :rank], arrays["initial_B"][0, :, :rank])
                 for rank in (1, 2, 4)}
    branches = {}
    xb = np.zeros((1, M))
    zh = np.zeros((LS, 1, D))
    teacher_ybuf = np.zeros(LS)
    ad = 0.
    ae = {arm:0. for arm in ARMS}
    segment_sums = np.zeros((3, len(ARMS)))
    boundary_sums = np.zeros((2, len(ARMS)))
    teacher_d_max_abs = 0.
    switch = {}
    transitions = {}
    init_arrays = {}
    started = perf_counter()
    for n in range(T):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z = np.sqrt(2/D)*np.cos(np.einsum("rdm,rm->rd", om, xb)+ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2]+.5*zh[3]
        teacher_matrix = low if n < SEGMENT or n >= 2*SEGMENT else high
        teacher_ybuf[1:] = teacher_ybuf[:-1].copy()
        teacher_ybuf[0] = float(np.sum(teacher_matrix*z.reshape(D1, D2).T))
        teacher_d_max_abs = max(teacher_d_max_abs, abs(float(teacher_ybuf@ac.S_PATH)-d[n]))
        assert teacher_d_max_abs < 1e-10

        if n == SEGMENT:
            old = branches["boundary_R2_R4_R2"]["controller"]
            assert old.R == 2
            rng = np.random.default_rng([SEED, 0, run, GROW_PART])
            added_A = .01*rng.standard_normal((D1, 2))
            A = np.concatenate((old.A[0], added_A), axis=1)
            B = np.concatenate((old.B[0], np.zeros((D2, 2))), axis=1)
            before = old.B[0]@old.A[0].T
            after = B@A.T
            continuity = float(np.max(np.abs(before-after)))
            assert continuity < 1e-14
            grown = controller("boundary_R2_R4_R2", 4, A, B, old.ybuf)
            branches["boundary_R2_R4_R2"]["controller"] = grown
            transitions["growth"] = dict(sample=n, from_R=2, to_R=4,
                                         added_A_seed=[SEED, 0, run, GROW_PART],
                                         immediate_weight_max_abs_change=continuity,
                                         inherited_FIR_max_abs_change=float(np.max(
                                             np.abs(grown.ybuf-old.ybuf))))
        if n == 2*SEGMENT:
            old = branches["boundary_R2_R4_R2"]["controller"]
            assert old.R == 4
            w = (old.B[0]@old.A[0].T).T.reshape(D)
            A, B, relative = factors(w, np.arange(D), 2)
            pruned = controller("boundary_R2_R4_R2", 2, A, B, old.ybuf)
            branches["boundary_R2_R4_R2"]["controller"] = pruned
            transitions["prune"] = dict(sample=n, from_R=4, to_R=2,
                                        relative_frobenius_weight_change=relative,
                                        inherited_FIR_max_abs_change=float(np.max(
                                            np.abs(pruned.ybuf-old.ybuf))))

        dv = np.array([d[n]+v[n]])
        ad = BETA*ad+(1-BETA)*abs(d[n])
        for name, ctrl in originals.items():
            ys = float(ctrl.step(z, q, dv)[0])
            ae[name] = BETA*ae[name]+(1-BETA)*abs(dv[0]-ys)
        if n >= SWITCH:
            for name, state in branches.items():
                p = state["p"]
                ctrl = state["controller"]
                ys = float(ctrl.step(z[:, p], q[:, p], dv)[0])
                ae[name] = BETA*ae[name]+(1-BETA)*abs(dv[0]-ys)
        if n == SWITCH-1:
            c4 = originals["original_R4"]
            W = c4.B[0]@c4.A[0].T
            w = W.T.reshape(D)
            p = np.argsort(w, kind="stable")
            np.testing.assert_array_equal(np.sort(p), np.arange(D))
            a2, b2, e2 = factors(w, p, 2)
            a4, b4, e4 = factors(w, p, 4)
            branches = {
                "sorted_R2":dict(controller=controller("sorted_R2", 2, a2, b2, c4.ybuf), p=p),
                "sorted_R4":dict(controller=controller("sorted_R4", 4, a4, b4, c4.ybuf), p=p),
                "boundary_R2_R4_R2":dict(controller=controller("boundary_R2_R4_R2", 2, a2, b2, c4.ybuf), p=p),
            }
            for name in branches:
                ae[name] = ae["original_R4"]
            inv_z = float(abs(z[0]@w-z[0, p]@w[p]))
            inv_q = float(abs(q[0]@w-q[0, p]@w[p]))
            assert max(inv_z, inv_q) < 1e-12
            switch = dict(sample=n, next_sample=SWITCH, source="original_R4 at first low stage",
                          full_output_invariance_max_abs=max(inv_z, inv_q),
                          inherited_FIR_max_abs_change=max(float(np.max(np.abs(
                              state["controller"].ybuf-c4.ybuf))) for state in branches.values()),
                          R2_initial_relative_frobenius_error=e2,
                          R4_initial_relative_frobenius_error=e4,
                          teacher_geometry_original=dict(low=geometry(low, np.arange(D)),
                                                         high=geometry(high, np.arange(D))),
                          teacher_geometry_sorted=dict(low=geometry(low, p),
                                                       high=geometry(high, p)))
            init_arrays = dict(permutation=p.astype(np.uint16),
                               R4_low_stage_current_weight=w, sorted_R2_A=a2,
                               sorted_R2_B=b2, sorted_R4_A=a4, sorted_R4_B=b4)
        segment = n//SEGMENT
        if n >= (segment+1)*SEGMENT-TAIL:
            segment_sums[segment] += [20*np.log10((ae[name]+EPS)/(ad+EPS))
                                     for name in ARMS]
        for j,boundary in enumerate((SEGMENT, 2*SEGMENT)):
            if boundary <= n < boundary+TAIL:
                boundary_sums[j] += [20*np.log10((ae[name]+EPS)/(ad+EPS))
                                     for name in ARMS]

    segment_anr = segment_sums/TAIL
    boundary_anr = boundary_sums/TAIL
    segments = []
    for j in range(3):
        reference_errors = {}
        for rank in (1, 2, 4):
            name = f"original_R{rank}"
            ref = float(summary["segment_anr"][j][f"fixed_R{rank}"])
            err = abs(float(segment_anr[j, ARMS.index(name)])-ref)
            assert err < 2e-5, (run, j, rank, err)
            reference_errors[name] = err
        segments.append(dict(segment=j+1, bounds=[j*SEGMENT, (j+1)*SEGMENT],
                             teacher_R=1 if j in (0, 2) else 4,
                             arm_ANR_db={name:float(segment_anr[j, k]) for k,name in enumerate(ARMS)},
                             max_reference_replay_abs_difference_db=max(reference_errors.values())))
    initialization_file = HERE / f"E1_run{run:02d}_initialization.npz"
    np.savez_compressed(initialization_file, **init_arrays)
    result = dict(case="E1", run=run, phase="dev", provenance=provenance,
                  switch=switch, transitions=transitions,
                  teacher_d_reconstruction_max_abs=teacher_d_max_abs,
                  initialization_file=initialization_file.name,
                  initialization_file_sha256=sha(initialization_file),
                  costs=costs(), segments=segments,
                  boundary_first_5000_ANR_db={str(boundary):{
                      name:float(boundary_anr[j, k]) for k,name in enumerate(ARMS)}
                      for j,boundary in enumerate((SEGMENT, 2*SEGMENT))},
                  total_seconds=perf_counter()-started)
    (HERE / f"E1_run{run:02d}.json").write_text(json.dumps(result, indent=2),
                                                  encoding="utf-8")
    return result


def cache_or_run(run: int, refresh: bool):
    path = HERE / f"E1_run{run:02d}.json"
    if path.exists() and not refresh:
        result = json.loads(path.read_text(encoding="utf-8"))
        _, _, summary, current = load_run(run)
        assert result["phase"] == "dev" and result["provenance"] == current
        assert sha(HERE / result["initialization_file"]) == result["initialization_file_sha256"]
        assert result["costs"] == costs()
        repaired = False
        for j, segment in enumerate(result["segments"]):
            discrepancy = max(abs(segment["arm_ANR_db"][f"original_R{rank}"] -
                                  summary["segment_anr"][j][f"fixed_R{rank}"])
                              for rank in (1, 2, 4))
            if segment["max_reference_replay_abs_difference_db"] != discrepancy:
                segment["max_reference_replay_abs_difference_db"] = discrepancy
                repaired = True
        if repaired:
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    return run_case(run)


def aggregate(results):
    with (HERE / "segment_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("run", "segment", "teacher_R", "arm", "ANR_db",
                         "delta_vs_original_R4_db", "segment_body_mults_per_sample"))
        for result in results:
            for segment in result["segments"]:
                ref = segment["arm_ANR_db"]["original_R4"]
                for name in ARMS:
                    writer.writerow((result["run"], segment["segment"], segment["teacher_R"],
                                     name, segment["arm_ANR_db"][name],
                                     segment["arm_ANR_db"][name]-ref,
                                     result["costs"][name]["segment_body_mults_per_sample"][segment["segment"]-1]))
    (HERE / "aggregate.json").write_text(json.dumps(dict(
        phase="dev", runs=[r["run"] for r in results],
        max_reference_replay_abs_difference_db=max(
            s["max_reference_replay_abs_difference_db"]
            for r in results for s in r["segments"]),
        max_teacher_d_reconstruction_abs=max(r["teacher_d_reconstruction_max_abs"]
                                            for r in results)), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, nargs="+", choices=RUNS)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    results = []
    for run in args.runs or RUNS:
        result = cache_or_run(run, args.refresh)
        results.append(result)
        print(run, round(result["total_seconds"], 2), flush=True)
    aggregate(results)
