"""Independently replay N14 dev fixed-frontier ANR and check all artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

import numpy as np

DEPS = Path(tempfile.gettempdir()) / "n13_frontier_deps"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
from numba import njit

from run_frontier import HERE, INPUTS, MU, ROOT, costs, method_names, sha256


@njit(cache=True)
def replay_anr(d, e):
    """Independent, scalar-order physical ANR recursion from persisted traces."""
    n_methods, tmax = e.shape
    sums = np.zeros((n_methods, 4), np.float64)
    ae = np.zeros(n_methods, np.float64)
    ad = 0.0
    for n in range(tmax):
        ad = .999 * ad + .001 * abs(d[n])
        segment = n // 100_000
        in_tail = n >= (segment + 1) * 100_000 - 5_000
        for method in range(n_methods):
            ae[method] = .999 * ae[method] + .001 * abs(float(e[method, n]))
            if in_tail:
                sums[method, segment] += 20.0 * np.log10(
                    (ae[method] + 1e-12) / (ad + 1e-12))
    return sums / 5_000


def main():
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == "dev" and manifest["methods"] == method_names()
    assert sha256(INPUTS / "manifest.json") == manifest["input_manifest_sha256"]
    current_costs = costs()
    assert current_costs == manifest["cost_ledger"]
    all_anr = {}
    max_anr_gap = 0.0
    max_fir_bound_ratio = 0.0
    time_seconds = 0.0
    result_hashes = {}
    last_result_mtime = 0.0
    for case in ("E2", "E3"):
        rows = []
        for run in range(5):
            prefix = HERE / f"case{case}_run{run:02d}"
            result_path = prefix / "result.json"
            last_result_mtime = max(last_result_mtime, result_path.stat().st_mtime)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            assert result["phase"] == "dev" and result["case"] == case
            assert result["run"] == run and result["T"] == 400_000
            assert result["source_sha256"] == manifest["source_sha256"]
            assert result["input_manifest_sha256"] == manifest["input_manifest_sha256"]
            assert result["cost_ledger"] == current_costs
            assert result["method_names"] == method_names()
            assert result["anr_tail_counts"] == [5_000] * 4
            source = ROOT / result["input_path"]
            assert sha256(source) == result["input_sha256"]
            traces = prefix / "traces.npz"
            assert sha256(traces) == result["traces_sha256"]
            with np.load(source) as inp, np.load(traces) as saved:
                error = saved["physical_error"]
                output = saved["actuator_output"]
                assert output.shape == (45, 400_000)
                assert error.shape == (45, 400_000)
                assert np.isfinite(output).all() and np.isfinite(error).all()
                anr = replay_anr(inp["d"], error)
            stored = np.asarray(result["segment_anr_db"])
            gap = float(np.max(np.abs(anr - stored)))
            if gap > 1e-4:
                raise AssertionError(f"Independent ANR replay differs {case}/{run}: {gap}")
            max_anr_gap = max(max_anr_gap, gap)
            max_fir_bound_ratio = max(max_fir_bound_ratio,
                                      result["physical_fir_replay"]["max_rounding_bound_ratio"])
            if max_fir_bound_ratio > 1.001:
                raise AssertionError("Physical FIR rounding bound broken")
            rows.append(stored)
            time_seconds += result["elapsed_seconds"]
            result_hashes[f"{case}/{run}"] = sha256(result_path)
        all_anr[case] = np.stack(rows)
    summary = dict(phase="dev", case_runs="E2/E3 each 0..4", T_per_run=400_000,
                   branches_per_run=45, total_branch_samples=180_000_000,
                   completion_utc=datetime.fromtimestamp(last_result_mtime, timezone.utc)
                   .strftime("%Y-%m-%d %H:%M:%S UTC"),
                   sum_per_run_elapsed_seconds=time_seconds,
                   max_independent_anr_replay_abs_gap_db=max_anr_gap,
                   max_physical_fir_rounding_bound_ratio=max_fir_bound_ratio,
                   source_sha256=manifest["source_sha256"],
                   input_manifest_sha256=manifest["input_manifest_sha256"],
                   analyzer_sha256=sha256(Path(__file__)),
                   result_sha256=result_hashes, cost_ledger=current_costs)
    for case, a in all_anr.items():
        a = a.reshape(5, 9, 5, 4)
        summary[case] = dict(
            five_run_four_segment_mean_anr_db=a.mean(axis=(0, 3)).tolist(),
            r4_mu005_stage_mean_db=a[:, 3, 0, :].mean(axis=0).tolist(),
            r4_mu01_stage_mean_db=a[:, 3, 1, :].mean(axis=0).tolist(),
            full500_mu005_stage_mean_db=a[:, 8, 0, :].mean(axis=0).tolist(),
            full500_minus_r4_mu005_min_max_db=[
                float((a[:, 8, 0, :] - a[:, 3, 0, :]).min()),
                float((a[:, 8, 0, :] - a[:, 3, 0, :]).max())],
            # This is descriptive dev data, never the future select-phase μ.
            dev_diagnostic_best_mu=[float(MU[i]) for i in
                                    np.argmin(a.mean(axis=(0, 3)), axis=1)])
    path = HERE / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="complete", max_anr_gap=max_anr_gap,
                          max_fir_rounding_ratio=max_fir_bound_ratio,
                          summary=str(path)), ensure_ascii=False))


if __name__ == "__main__":
    main()
