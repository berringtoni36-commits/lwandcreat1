"""N14 development-only fixed R1..R8/full500 frontier, five common step sizes.

One case/run is an independently resumable unit. No selection or confirmation
generator is imported or callable from this module.
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
import numba  # noqa: E402

from kernel import simulate_fixed  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = ROOT / "experiments_v14" / "outputs" / "n14_dev_inputs"
MU = np.array([0.05, 0.10, 0.20, 0.40, 0.80], dtype=np.float64)
RANKS = tuple(range(1, 9))
TAIL = 5_000
BOUNDS = np.array([100_000, 200_000, 300_000, 400_000], np.int64)
SOURCES = (
    HERE / "kernel.py",
    HERE / "run_frontier.py",
    HERE / "test_frontier.py",
    HERE / "selection_interface.py",
    ROOT / "experiments_v14" / "protocol" / "PREREGISTRATION.md",
    ROOT / "experiments_v14" / "protocol" / "generator.py",
    ROOT / "experiments_v10" / "anc_core.py",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in SOURCES}


def method_names():
    return [f"R{rank}_mu{mu:.2f}" for rank in RANKS for mu in MU] + [
        f"full500_mu{mu:.2f}" for mu in MU]


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


def atomic_json(path, value):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def ensure_manifest():
    source_sha = source_hashes()
    input_manifest = INPUTS / "manifest.json"
    content = dict(phase="dev", cases=["E2", "E3"], run_ids=list(range(5)),
                   T=400_000, segment_bounds=BOUNDS.tolist(), tail_samples=TAIL,
                   mu=MU.tolist(), ranks=list(RANKS), methods=method_names(),
                   source_sha256=source_sha,
                   input_manifest_sha256=sha256(input_manifest),
                   python=sys.version, numpy=np.__version__,
                   numba=numba.__version__, platform=platform.platform(),
                   cost_ledger=costs(),
                   interpretation="Development fixed frontier; no step-size selection here")
    manifest = HERE / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != content:
            raise RuntimeError("Frontier sources, input manifest, or environment changed")
    else:
        atomic_json(manifest, content)
        for source in SOURCES:
            target = HERE / "sources" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copy2(input_manifest, HERE / "sources" / "input_manifest.json")
    return content


def load_input(case, run):
    if case not in ("E2", "E3") or not 0 <= run < 5:
        raise ValueError("Only N14 dev E2/E3 run 0..4 allowed")
    source = INPUTS / f"case{case}_run{run:02d}.npz"
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest = json.loads((INPUTS / "manifest.json").read_text(encoding="utf-8"))
    entries = [e for e in manifest["entries"] if e["case"] == case and e["run"] == run]
    if len(entries) != 1:
        raise RuntimeError("Missing or duplicate input manifest entry")
    entry = entries[0]
    if sha256(source) != entry["file_sha256"]:
        raise RuntimeError("N14 dev input hash mismatch")
    with np.load(source) as archive:
        data = {key: np.ascontiguousarray(archive[key]) for key in
                ("x", "d", "v", "omega", "rff_phase", "initial_A", "initial_B",
                 "segment_bounds")}
    if (data["x"].shape != (400_000,) or data["d"].shape != (400_000,)
            or data["v"].shape != (400_000,)
            or data["omega"].shape != (500, 20)
            or data["rff_phase"].shape != (500,)
            or data["initial_A"].shape != (25, 8)
            or data["initial_B"].shape != (20, 8)
            or not np.array_equal(data["segment_bounds"],
                                  np.array([[0, 100_000], [100_000, 200_000],
                                            [200_000, 300_000], [300_000, 400_000]]))):
        raise RuntimeError("N14 dev input shape or segment bounds mismatch")
    return source, data


def physical_fir_replay(data, output, error):
    """Independent physical microphone calculation, bounded by f32 trace rounding."""
    dv = data["d"] + data["v"]
    worst = 0.0
    worst_ulp_ratio = 0.0
    for c in range(45):
        yc = output[c].astype(np.float64)
        ec = error[c].astype(np.float64)
        replay = dv.copy()
        replay[2:] -= yc[:-2]
        replay[3:] -= 0.5 * yc[:-3]
        gap = np.abs(replay - ec)
        # Three stored f32 values (output at n-2/n-3 and error at n)
        # account for the entire discrepancy; double arithmetic adds slack.
        scale = (np.spacing(np.abs(output[c]).astype(np.float32)).astype(np.float64))
        err_ulp = np.spacing(np.abs(error[c]).astype(np.float32)).astype(np.float64)
        allowance = 2e-12 + 0.5 * err_ulp
        allowance[2:] += 0.5 * scale[:-2]
        allowance[3:] += 0.25 * scale[:-3]
        worst = max(worst, float(np.max(gap)))
        worst_ulp_ratio = max(worst_ulp_ratio, float(np.max(gap / allowance)))
    if worst_ulp_ratio > 1.001:
        raise AssertionError(f"Physical FIR replay exceeds f32 rounding: {worst_ulp_ratio}")
    return dict(max_abs_gap=worst, max_rounding_bound_ratio=worst_ulp_ratio)


def run_one(case, run):
    frozen = ensure_manifest()
    source, data = load_input(case, run)
    outdir = HERE / f"case{case}_run{run:02d}"
    outdir.mkdir(exist_ok=True)
    result_path = outdir / "result.json"
    if result_path.exists():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if (prior["source_sha256"] != frozen["source_sha256"]
                or prior["input_sha256"] != sha256(source)):
            raise RuntimeError("Existing result is from different source or input")
        return prior
    start = perf_counter()
    sums, counts, y, error = simulate_fixed(
        data["x"], data["d"], data["v"], data["omega"], data["rff_phase"],
        data["initial_A"], data["initial_B"], MU, BOUNDS, TAIL)
    if (not np.array_equal(counts, np.full(4, TAIL))
            or not np.isfinite(sums).all() or not np.isfinite(y).all()
            or not np.isfinite(error).all()):
        raise AssertionError("Invalid frontier trajectory or ANR counts")
    fir = physical_fir_replay(data, y, error)
    if source_hashes() != frozen["source_sha256"]:
        raise RuntimeError("Frontier source changed during computation")
    traces_path = outdir / "traces.npz"
    temporary = outdir / "traces.tmp.npz"
    np.savez_compressed(temporary, actuator_output=y, physical_error=error,
                        method_names=np.array(method_names()),
                        segment_anr_db=sums / counts)
    os.replace(temporary, traces_path)
    result = dict(phase="dev", case=case, run=run, T=400_000,
                  method_names=method_names(), mu=MU.tolist(),
                  segment_bounds=data["segment_bounds"].tolist(),
                  segment_anr_db=(sums / counts).tolist(),
                  anr_tail_counts=counts.tolist(), cost_ledger=frozen["cost_ledger"],
                  total_multiplications={name: int(row["total_per_sample"] * 400_000)
                                         for name, row in frozen["cost_ledger"].items()},
                  physical_fir_replay=fir, input_path=str(source.relative_to(ROOT)),
                  input_sha256=sha256(source),
                  input_manifest_sha256=frozen["input_manifest_sha256"],
                  source_sha256=frozen["source_sha256"],
                  traces_sha256=sha256(traces_path),
                  elapsed_seconds=perf_counter() - start)
    atomic_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("E2", "E3"))
    parser.add_argument("--run", required=True, type=int, choices=range(5))
    args = parser.parse_args()
    result = run_one(args.case, args.run)
    print(json.dumps(dict(case=args.case, run=args.run,
                          r4_mu005=result["segment_anr_db"][15],
                          r4_mu01=result["segment_anr_db"][16],
                          full500_mu01=result["segment_anr_db"][41],
                          seconds=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
