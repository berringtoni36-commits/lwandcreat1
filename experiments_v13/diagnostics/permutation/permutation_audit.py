"""Hindsight-only feature-permutation capacity audit on saved E2/E3 dev run00.

The terminal full500 weight of each segment was learned using the entire
segment, including the frozen evaluation window. No online selection claim is
possible from this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V12 = ROOT / "experiments_v12"
DEV = V12 / "outputs" / "main_dev_round1"
SPECTRUM = V12 / "diagnostics" / "spectrum"
sys.path.insert(0, str(DEV / "sources" / "experiments_v10"))
import anc_core as ac

CASES = ("E2", "E3")
METHODS = ("original", "fixed_random", "first_segment_signed_sort",
           "signed_weight_sort", "magnitude_weight_sort",
           "signed_sort_local_swaps")
RANKS = (2, 3)
N_FEATURES = 500
SHAPE = (20, 25)
SEGMENT_LENGTH = 100000
WARMUP = 5000
SCORE = 5000
BETA = .999
EPS = 1e-12
RANDOM_SEED = 2026092814
SWAP_TRIALS = 600


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(case: str):
    assert case in CASES
    protocol_path = DEV / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["phase"] == "dev" and case in protocol["cases"]
    source_path = DEV / f"case{case}_run00.npz"
    replay_path = SPECTRUM / f"replay_{case}_run00.npz"
    with np.load(source_path, allow_pickle=False) as f:
        x, d, v, om, ph = (f[key].copy() for key in ("x", "d", "v", "Om", "ph"))
        meta = json.loads(str(f["meta"]))
    with np.load(replay_path, allow_pickle=False) as f:
        full = f["terminal_full"].copy()
        fixed = f["terminal_fixed"].copy()
        provenance = json.loads(str(f["provenance"]))
    assert len(x) == len(d) == len(v) == 4*SEGMENT_LENGTH
    assert om.shape == (1, N_FEATURES, 20) and ph.shape == (1, N_FEATURES)
    assert full.shape == fixed.shape == (4, *SHAPE)
    assert meta["case"] == case and meta["run"] == 0
    assert provenance["phase"] == "dev" and provenance["case"] == case
    assert provenance["source_sha256"] == sha(source_path)
    assert provenance["core_sha256"] == sha(Path(ac.__file__))
    return (x, d, v, om, ph, full, fixed,
            {"source": str(source_path.relative_to(ROOT)),
             "source_sha256": sha(source_path),
             "spectrum_replay_sha256": sha(replay_path),
             "protocol_sha256": sha(protocol_path),
             "archived_core_sha256": sha(Path(ac.__file__)),
             "terminal_weight_replay_provenance": provenance})


def truncation(w: np.ndarray, permutation: np.ndarray):
    assert permutation.shape == (N_FEATURES,)
    assert np.array_equal(np.sort(permutation), np.arange(N_FEATURES))
    matrix = w[permutation].reshape(25, 20).T
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    norm = float(np.linalg.norm(singular))
    projections = {}
    errors = {}
    for rank in RANKS:
        low = (u[:, :rank]*singular[:rank]) @ vh[:rank]
        low_perm = low.T.reshape(N_FEATURES)
        low_original = np.empty(N_FEATURES)
        low_original[permutation] = low_perm
        projections[rank] = low_original
        errors[rank] = float(np.linalg.norm(matrix-low)/norm)
        np.testing.assert_allclose(errors[rank],
                                   np.linalg.norm(singular[rank:])/norm,
                                   atol=1e-12, rtol=1e-12)
    return errors, projections


def swap_objective(w: np.ndarray, p: np.ndarray) -> float:
    singular = np.linalg.svd(w[p].reshape(25, 20).T,
                             compute_uv=False)
    denominator = float(np.sum(singular**2))
    return float((np.sum(singular[2:]**2) +
                  np.sum(singular[3:]**2))/(2*denominator))


def refine_signed_sort(w: np.ndarray, initial: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    p = initial.copy()
    initial_objective = current = swap_objective(w, p)
    accepted = 0
    started = perf_counter()
    for _ in range(SWAP_TRIALS):
        i, j = rng.choice(N_FEATURES, 2, replace=False)
        p[i], p[j] = p[j], p[i]
        trial = swap_objective(w, p)
        if trial < current - 1e-14:
            current = trial
            accepted += 1
        else:
            p[i], p[j] = p[j], p[i]
    assert current <= initial_objective + 1e-14
    return p, {"trials": SWAP_TRIALS, "accepted": accepted,
               "initial_objective": initial_objective,
               "final_objective": current,
               "seconds": perf_counter()-started}


def feature_window(x: np.ndarray, om: np.ndarray, ph: np.ndarray,
                   segment: int):
    end = (segment+1)*SEGMENT_LENGTH
    first = end-WARMUP-SCORE
    history = sliding_window_view(x[first-19:end], 20)[:, ::-1]
    z = np.sqrt(2/N_FEATURES)*np.cos(history @ om[0].T + ph[0])
    assert z.shape == (WARMUP+SCORE, N_FEATURES)
    return first, end, z


def frozen_anr(d: np.ndarray, v: np.ndarray, first: int,
               z: np.ndarray, weights: dict[str, np.ndarray]):
    labels = list(weights)
    vectors = np.stack([weights[key] for key in labels], axis=1)
    y = (z @ vectors).T
    ys = ac.fir(ac.S_PATH, y)
    errors = d[first:first+len(z)][None, :] + v[first:first+len(z)][None, :] - ys
    ae = np.zeros(len(labels))
    ad = 0.
    sums = np.zeros(len(labels))
    for n in range(len(z)):
        ae = BETA*ae + (1-BETA)*np.abs(errors[:, n])
        ad = BETA*ad + (1-BETA)*abs(d[first+n])
        if n >= WARMUP:
            sums += 20*np.log10((ae+EPS)/(ad+EPS))
    return {label: float(sums[i]/SCORE) for i, label in enumerate(labels)}


def analyze_case(case: str):
    x, d, v, om, ph, full, fixed, provenance = load_case(case)
    fixed_random = np.random.default_rng(RANDOM_SEED).permutation(N_FEATURES)
    first_segment_signed = np.argsort(full[0].T.reshape(N_FEATURES), kind="stable")
    first_positions = np.empty(N_FEATURES, dtype=int)
    first_positions[first_segment_signed] = np.arange(N_FEATURES)
    permutations = np.empty((4, len(METHODS), N_FEATURES), dtype=np.uint16)
    projections = np.empty((4, len(METHODS), len(RANKS), N_FEATURES))
    full_original = np.empty((4, N_FEATURES))
    segments = []
    started = perf_counter()
    for segment in range(4):
        w = full[segment].T.reshape(N_FEATURES)
        full_original[segment] = w
        identity = np.arange(N_FEATURES)
        signed = np.argsort(w, kind="stable")
        current_positions = np.empty(N_FEATURES, dtype=int)
        current_positions[signed] = np.arange(N_FEATURES)
        magnitude = np.argsort(-np.abs(w), kind="stable")
        local_seed = RANDOM_SEED + (0 if case == "E2" else 100) + segment + 1
        local, swap_info = refine_signed_sort(w, signed, local_seed)
        p_by_method = (identity, fixed_random, first_segment_signed,
                       signed, magnitude, local)
        first, end, z = feature_window(x, om, ph, segment)
        y_full = z @ w
        weights = {"full500": w, "fixed_R4": fixed[segment].T.reshape(N_FEATURES)}
        method_data = []
        for method_index, (method, p) in enumerate(zip(METHODS, p_by_method)):
            permutations[segment, method_index] = p
            # Both physical features and weights are moved by the same p.
            # Without truncation, this must recover the exact full500 output.
            permuted_full_error = float(np.max(np.abs(z[:, p] @ w[p] - y_full)))
            assert permuted_full_error < 1e-10, permuted_full_error
            errors, projected = truncation(w, p)
            for rank_index, rank in enumerate(RANKS):
                name = f"{method}_R{rank}"
                projections[segment, method_index, rank_index] = projected[rank]
                # Inverse permutation puts the low-rank coefficients back in
                # the original feature coordinates for the physical FIR audit.
                np.testing.assert_allclose(z[:64, p] @ projected[rank][p],
                                           z[:64] @ projected[rank],
                                           atol=1e-10, rtol=1e-12)
                weights[name] = projected[rank]
            method_data.append({"method": method,
                                "relative_frobenius_error":
                                    {str(rank): errors[rank] for rank in RANKS},
                                "full500_output_max_abs_difference": permuted_full_error,
                                "swap_search": swap_info if method == METHODS[-1] else None})
        frozen = frozen_anr(d, v, first, z, weights)
        for data in method_data:
            method = data["method"]
            data["frozen_anr_db"] = {str(rank): frozen[f"{method}_R{rank}"]
                                     for rank in RANKS}
            data["delta_vs_full500_db"] = {str(rank):
                frozen[f"{method}_R{rank}"]-frozen["full500"] for rank in RANKS}
            data["delta_vs_fixed_R4_db"] = {str(rank):
                frozen[f"{method}_R{rank}"]-frozen["fixed_R4"] for rank in RANKS}
        segments.append({"segment": segment+1, "bounds": [segment*SEGMENT_LENGTH, end],
                         "frozen_window": [first, end],
                         "signed_sort_same_positions_as_first": int(np.sum(signed == first_segment_signed)),
                         "signed_sort_median_rank_shift_from_first": float(
                             np.median(np.abs(current_positions-first_positions))),
                         "frozen_full500_anr_db": frozen["full500"],
                         "frozen_fixed_R4_anr_db": frozen["fixed_R4"],
                         "methods": method_data})
        print(case, "segment", segment+1, "done", flush=True)
    np.savez_compressed(HERE / f"{case}_run00_arrays.npz",
                        permutations=permutations, projected_weights_original=projections,
                        terminal_full_weights_original=full_original,
                        method_names=np.array(METHODS), ranks=np.array(RANKS))
    result = {"case": case, "run": 0, "phase": "dev",
              "provenance": provenance,
              "method_names": list(METHODS), "ranks": list(RANKS),
              "fixed_random_seed": RANDOM_SEED,
              "local_swap_trials_per_segment": SWAP_TRIALS,
              "array_file": f"{case}_run00_arrays.npz",
              "array_file_sha256": sha(HERE / f"{case}_run00_arrays.npz"),
              "total_seconds": perf_counter()-started,
              "segments": segments}
    (HERE / f"{case}_run00.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, action="append")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    for case in (args.case or CASES):
        result = analyze_case(case)
        print(case, "seconds", round(result["total_seconds"], 2), flush=True)
