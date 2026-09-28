"""Quantify Python/MATLAB short-probe differences without importing selector output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments_v12" / ".deps"))
from scipy.io import loadmat  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def max_abs(left, right):
    x, y = np.asarray(left), np.asarray(right)
    assert x.shape == y.shape
    assert np.array_equal(np.isnan(x), np.isnan(y))
    if np.isfinite(x).any():
        return float(np.max(np.abs(x[np.isfinite(x)]-y[np.isfinite(y)])))
    return 0.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default="E2_run00_provisional_short")
    args = parser.parse_args()
    stem = args.stem
    meta = json.loads((HERE / f"{stem}.json").read_text(encoding="utf-8"))
    fixture_path, result_path = HERE / f"{stem}.mat", HERE / f"{stem}_result.mat"
    assert meta["fixture_sha256"] == sha(fixture_path)
    f = loadmat(fixture_path, squeeze_me=True)
    m = loadmat(result_path, squeeze_me=True)
    traces = {
        "R4_drive": ("active_drive_expected", "active_drive"),
        "R4_secondary": ("active_ys_expected", "active_ys"),
        "candidate_drive": ("candidate_drive_expected", "candidate_drive"),
        "candidate_secondary": ("candidate_ys_expected", "candidate_ys"),
        "actual_drive": ("actual_drive_expected", "actual_drive"),
        "actual_secondary": ("actual_ys_expected", "actual_ys"),
        "actual_physical_error": ("actual_error_expected", "actual_error"),
        "actual_ANR_db": ("actual_anr_expected", "actual_anr"),
        "forced_gamma": ("gamma_forced_expected", "gamma_forced"),
        "RFF_checkpoints": ("z_checks_expected", "z_checks"),
        "filtered_RFF_checkpoints": ("q_checks_expected", "q_checks"),
        "R4_A_checkpoints": ("R4_A_checks_expected", "A_checks"),
        "R4_B_checkpoints": ("R4_B_checks_expected", "B_checks"),
        "R4_A_at_proposal": ("R4_A_at_proposal", "R4_A_at_proposal"),
        "R4_B_at_proposal": ("R4_B_at_proposal", "R4_B_at_proposal"),
        "R4_FIR_at_proposal": ("R4_ybuf_at_proposal", "R4_ybuf_at_proposal"),
        "candidate_A_at_proposal": ("candidate_A_at_proposal", "candidate_A_at_proposal"),
        "candidate_B_at_proposal": ("candidate_B_at_proposal", "candidate_B_at_proposal"),
        "SVD_tail": ("truncation_expected", "truncation"),
    }
    errors = {name:max_abs(f[reference], m[observed])
              for name,(reference,observed) in traces.items()}
    permutation_equal = bool(np.array_equal(
        np.asarray(f["candidate_permutation_one_based"]).ravel(),
        np.asarray(m["perm"]).ravel()))
    counted_equal = bool(int(f["prune_mults_expected"]) == int(m["prune_mults"]))
    sweeps_equal = bool(int(f["jacobi_sweeps_expected"]) == int(m["jacobi_sweeps"]))
    numeric_max = max(errors.values())
    report = dict(phase="dev", fixture=fixture_path.name,
                  fixture_sha256=sha(fixture_path), result_sha256=sha(result_path),
                  selector_sha256=meta["selector_sha256"],
                  frozen_selector=meta["frozen_selector"],
                  errors=errors, max_numeric_abs=numeric_max,
                  permutation_equal=permutation_equal,
                  counted_multiplies_equal=counted_equal,
                  jacobi_sweeps_equal=sweeps_equal,
                  pass_short_probe=bool(numeric_max < 1e-6 and permutation_equal and
                                        counted_equal and sweeps_equal))
    (HERE / f"{stem}_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass_short_probe"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
