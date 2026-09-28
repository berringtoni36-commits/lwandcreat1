"""Compare independent full MATLAB periodic-resort replay against dev trace."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments_v12/.deps"))
from scipy.io import loadmat  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def max_abs(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ok = ~np.isnan(a)
    return float(np.max(np.abs(a[ok]-b[ok]))) if np.any(ok) else 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    args = ap.parse_args()
    stem = args.stem
    metadata = json.loads((HERE / f"{stem}.json").read_text(encoding="utf-8"))
    assert metadata["phase"] == "dev" and metadata["resort_interval"] == 50000
    fixture = HERE / metadata["fixture"]
    trace = HERE / metadata["python_trace"]
    result = HERE / f"{stem}_result.mat"
    summary = ROOT / "experiments_v14/resort_diagnostic" / metadata["summary"]
    source = ROOT / "experiments_v14/resort_diagnostic/selector_resort.py"
    frozen = ROOT / "experiments_v13/online_selector/online_selector.py"
    assert sha(fixture) == metadata["fixture_sha256"]
    assert sha(trace) == metadata["python_trace_sha256"]
    assert sha(summary) == metadata["summary_sha256"]
    assert sha(source) == metadata["resort_source_sha256"]
    assert sha(frozen) == metadata["frozen_source_sha256"]
    assert metadata["input"]["phase"] == "dev"
    with np.load(trace, allow_pickle=False) as file:
        ref = {k:file[k].copy() for k in file.files}
    obs = loadmat(result, squeeze_me=True)
    fields = ("drive", "physical_error", "ys_actual", "anr", "R",
              "candidate_R", "retiring_R", "resident_factor_coeffs", "gamma",
              "exp_evals", "post_ramp_output_error", "cost_core", "cost_candidate",
              "cost_retiring", "cost_management", "cost_reorder", "cost_total")
    def reference_name(key):
        if key == "physical_error":
            return "error"
        if key.startswith("cost_"):
            return key[5:]
        return key
    errors = {key:max_abs(obs[key], ref[reference_name(key)]) for key in fields}
    integer_fields = ("R", "candidate_R", "retiring_R", "resident_factor_coeffs",
                      "exp_evals", "cost_core", "cost_candidate", "cost_retiring",
                      "cost_management", "cost_reorder", "cost_total")
    integers_equal = all(errors[key] == 0 for key in integer_fields)
    events = metadata["events"]
    proposals = [e for e in events if e["type"] == "proposal_prune"]
    decisions = [e for e in events if e["type"] == "prune"]
    assert len(proposals) == 8 and len(decisions) == 7 and all(e["accepted"] for e in decisions)
    event_errors = {}
    def compare_vector(name, expected):
        event_errors[name] = max_abs(obs[name], expected)
    compare_vector("proposal_n", [e["n"] for e in proposals])
    compare_vector("proposal_from_R", [e["from_R"] for e in proposals])
    compare_vector("proposal_reorder", [e["reorder_mults"] for e in proposals])
    compare_vector("proposal_sweeps", [e["jacobi_sweeps"] for e in proposals])
    compare_vector("proposal_tail", [e["truncation_relative_frobenius"] for e in proposals])
    compare_vector("proposal_veto", [0]*len(proposals))
    compare_vector("decision_n", [e["n"] for e in decisions])
    compare_vector("decision_from_R", [e["from_R"] for e in decisions])
    compare_vector("decision_accepted", [int(e["accepted"]) for e in decisions])
    compare_vector("validation_ratio", [e["validation_ratio"] for e in decisions])
    compare_vector("validation_count", [e["validation_samples"] for e in decisions])
    compare_vector("ramp_start", [e["ramp_start"] for e in decisions])
    compare_vector("ramp_end", [e["ramp_end"] for e in decisions])
    seg = json.loads(summary.read_text(encoding="utf-8"))["segment_anr_db"]
    seg_anr_errors = [abs(float(obs["segment_anr"][j])-float(seg[j])) for j in range(4)]
    seg_cost_errors = [abs(float(obs["segment_mults"][j])-
                           float(ref["total"][j*100000:(j+1)*100000].mean())) for j in range(4)]
    mean_cost_error = abs(float(obs["mean_mults"])-float(ref["total"].mean()))
    terminal_candidate = int(np.asarray(obs["candidate_R"]).ravel()[-1])
    terminal_candidate_cost = int(np.asarray(obs["cost_candidate"]).ravel()[-1])
    pending_candidate_points = int(np.count_nonzero(np.asarray(obs["candidate_R"]).ravel()[398001:]))
    result_report = dict(phase="dev", stem=stem,
                         resort_source_sha256=sha(source), fixture_sha256=sha(fixture),
                         reference_trace_sha256=sha(trace), matlab_result_sha256=sha(result),
                         frozen_prefix_length=metadata["frozen_prefix_length"],
                         frozen_prefix_max_abs=max(metadata["frozen_prefix_max_abs"].values()),
                         trace_max_abs=errors, integer_traces_equal=integers_equal,
                         event_max_abs=event_errors,
                         segment_ANR_abs_differences_db=seg_anr_errors,
                         segment_cost_abs_differences=seg_cost_errors,
                         mean_cost_abs_difference=mean_cost_error,
                         terminal_candidate_R=terminal_candidate,
                         terminal_candidate_cost=terminal_candidate_cost,
                         pending_candidate_charged_points=pending_candidate_points,
                         physical_FIR_max_abs=float(obs["physical_fir_max_abs"]),
                         post_ramp_max_abs=float(obs["post_ramp_max_abs"]))
    discrete_events = [k for k in event_errors if k not in ("proposal_tail", "validation_ratio")]
    result_report["pass_full"] = bool(
        integers_equal and all(event_errors[k] == 0 for k in discrete_events) and
        event_errors["proposal_tail"] < 1e-8 and event_errors["validation_ratio"] < 1e-8 and
        errors["drive"] < 1e-9 and errors["physical_error"] < 1e-9 and
        errors["ys_actual"] < 1e-9 and errors["anr"] < 1e-5 and
        errors["post_ramp_output_error"] < 1e-9 and errors["gamma"] == 0 and
        max(seg_anr_errors) < 1e-5 and max(seg_cost_errors) < 1e-8 and
        mean_cost_error < 1e-8 and terminal_candidate == 2 and
        pending_candidate_points == 1999 and terminal_candidate_cost > 0 and
        result_report["physical_FIR_max_abs"] < 1e-9 and
        result_report["post_ramp_max_abs"] < 1e-9 and
        result_report["frozen_prefix_max_abs"] == 0)
    (HERE / f"{stem}_comparison.json").write_text(json.dumps(result_report, indent=2), encoding="utf-8")
    print(json.dumps(result_report, indent=2))
    if not result_report["pass_full"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
