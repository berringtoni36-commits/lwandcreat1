"""Compare independent full MATLAB replay with frozen 60F6 dev trace/event ledger."""
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


def absolute_max(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    return float(np.max(np.abs(a-b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", required=True)
    args = parser.parse_args()
    stem = args.stem
    meta_path = HERE / f"{stem}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["phase"] == "dev"
    fixture = HERE / meta["fixture"]
    result_path = HERE / f"{stem}_result.mat"
    reference_path = ROOT / meta["reference_trace"]
    summary_path = ROOT / meta["reference_summary"]
    assert sha(fixture) == meta["fixture_sha256"]
    assert sha(reference_path) == meta["reference_trace_sha256"]
    assert sha(summary_path) == meta["reference_summary_sha256"]
    source = ROOT / "experiments_v13" / "online_selector" / "online_selector.py"
    assert sha(source) == meta["selector_sha256"]
    reference_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(reference_path, allow_pickle=False) as f:
        reference = {key:f[key].copy() for key in f.files}
    observed = loadmat(result_path, squeeze_me=True)
    scalar_fields = ("drive", "physical_error", "ys_actual", "anr", "R",
                     "candidate_R", "retiring_R", "resident_factor_coeffs",
                     "gamma", "cost_core", "cost_candidate", "cost_retiring",
                     "cost_management", "cost_reorder", "cost_total")
    reference_names = {"physical_error":"error"}
    errors = {key:absolute_max(observed[key], reference[reference_names.get(key, key)])
              for key in scalar_fields}
    integer_fields = ("R", "candidate_R", "retiring_R", "resident_factor_coeffs",
                      "cost_core", "cost_candidate", "cost_retiring",
                      "cost_management", "cost_reorder", "cost_total")
    integers_equal = all(errors[key] == 0 for key in integer_fields)
    events = reference_summary["events"]
    assert len(events) == 2 and events[1]["accepted"]
    proposal, decision = events
    event_errors = dict(
        proposal_index=abs(int(observed["proposal_n"])-proposal["n"]),
        reorder_mults=abs(int(observed["proposal_reorder"])-proposal["reorder_mults"]),
        jacobi_sweeps=abs(int(observed["proposal_sweeps"])-proposal["jacobi_sweeps"]),
        truncation=abs(float(observed["proposal_tail"])-proposal["truncation_relative_frobenius"]),
        decision_index=abs(int(observed["decision_n"])-decision["n"]),
        accepted=abs(int(observed["decision_accepted"])-int(decision["accepted"])),
        validation_samples=abs(int(observed["validation_count"])-decision["validation_samples"]),
        validation_ratio=abs(float(observed["validation_ratio"])-decision["validation_ratio"]),
        ramp_start=abs(int(observed["ramp_start"])-decision["ramp_start"]),
        ramp_end=abs(int(observed["ramp_end"])-decision["ramp_end"]),
    )
    segment_anr_errors = [abs(float(observed["segment_anr"][j])-
                              float(reference_summary["segments"][j]["anr_db"]))
                          for j in range(4)]
    segment_cost_errors = [abs(float(observed["segment_mults"][j])-
                               float(reference_summary["segments"][j]["mean_mults"]))
                           for j in range(4)]
    mean_cost_error = abs(float(observed["mean_mults"])-reference_summary["mean_mults"])
    checkpoints = (0, 1, 19999, 20000, 20001, 22000, 22001,
                   24000, 24001, 24100, 24101, 99999, 199999, 299999, 399999)
    checkpoint_error = {str(i):float(abs(observed["physical_error"][i]-reference["error"][i]))
                        for i in checkpoints}
    report = dict(phase="dev", stem=stem, selector_sha256=meta["selector_sha256"],
                  fixture_sha256=sha(fixture), reference_trace_sha256=sha(reference_path),
                  matlab_result_sha256=sha(result_path),
                  trace_max_abs=errors, integer_traces_equal=integers_equal,
                  event_differences=event_errors,
                  segment_ANR_abs_differences_db=segment_anr_errors,
                  segment_cost_abs_differences=segment_cost_errors,
                  mean_cost_abs_difference=mean_cost_error,
                  physical_FIR_max_abs=float(observed["physical_fir_max_abs"]),
                  post_ramp_max_abs=float(observed["post_ramp_max_abs"]),
                  checkpoint_physical_error_abs=checkpoint_error)
    report["pass_full"] = bool(integers_equal and errors["drive"] < 1e-9 and
        errors["physical_error"] < 1e-9 and errors["ys_actual"] < 1e-9 and
        errors["anr"] < 1e-5 and errors["gamma"] == 0 and
        max(event_errors.values()) < 1e-8 and max(segment_anr_errors) < 1e-5 and
        max(segment_cost_errors) < 1e-8 and mean_cost_error < 1e-8 and
        report["physical_FIR_max_abs"] < 1e-9 and report["post_ramp_max_abs"] < 1e-9)
    (HERE / f"{stem}_full_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass_full"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
