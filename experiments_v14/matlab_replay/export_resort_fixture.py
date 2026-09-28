"""Pin N14 dev inputs and independently saved periodic-resort Python trace.

The MATLAB fixture contains only plant input, RFF mapping and initial factors;
Python outputs and events are saved separately for post-run comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "experiments_v14/resort_diagnostic/selector_resort.py"
FROZEN = ROOT / "experiments_v13/online_selector/online_selector.py"
SUMMARY_DIR = ROOT / "experiments_v14/resort_diagnostic"
sys.path.insert(0, str(ROOT / "experiments_v12/.deps"))
from scipy.io import savemat  # noqa: E402
from export_fixture import load_input  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["E2", "E3"], required=True)
    ap.add_argument("--run", type=int, choices=range(5), default=0)
    ap.add_argument("--mu", type=float, default=.05)
    ap.add_argument("--interval", type=int, default=50000)
    args = ap.parse_args()
    assert args.mu == .05 and args.interval == 50000
    arrays, provenance = load_input(args.case, args.run)
    stem = f"case{args.case}_run{args.run:02d}_mu{args.mu:g}_interval{args.interval}"
    summary_path = SUMMARY_DIR / f"{stem}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["phase"] == "dev_diagnostic"
    assert (summary["case"], summary["run"], summary["mu"], summary["interval"]) == (
        args.case, args.run, args.mu, args.interval)
    assert summary["input_sha256"] == provenance["source_sha256"]
    assert summary["simulator_sha256"] == sha(SOURCE)
    assert summary["config"] == dict(arm="once", proposal="signed_sort",
                                     ramp_samples=100, ablate_spectral_veto=False,
                                     resort_interval=args.interval)
    assert sha(FROZEN) == "60f6fa3f0afe5b9ffc03ef4b610bc446efd7a9e5ed391d621fce6f9b1cc78846"
    out_stem = f"{args.case}_run{args.run:02d}_mu{args.mu:g}_resort50k"
    fixture = HERE / f"{out_stem}.mat"
    savemat(fixture, dict(x=arrays["x"], d=arrays["d"], v=arrays["v"],
                          omega=arrays["omega"], rff_phase=arrays["rff_phase"],
                          initial_A=arrays["initial_A"][:, :4],
                          initial_B=arrays["initial_B"][:, :4], mu=args.mu,
                          S_PATH=np.array([0., 0., 1., .5]), first_propose=20000,
                          train_samples=2000, validation_samples=2000,
                          ramp_samples=100, propose_interval=40000,
                          resort_interval=args.interval, cooldown=12000,
                          spectral_threshold=.15, detect_rel=.28, detect_hold=250),
            do_compression=True)
    module = import_source(SOURCE, "n14_resort_reference")
    r = module.simulate(arrays["x"], arrays["d"], arrays["v"],
                        arrays["omega"][None], arrays["rff_phase"][None],
                        arrays["initial_A"][None], arrays["initial_B"][None],
                        args.mu, summary["config"])
    assert len(r["events"]) == len(summary["events"])
    for actual, expected in zip(r["events"], summary["events"]):
        for key, value in expected.items():
            if isinstance(value, float):
                assert abs(actual[key] - value) < 1e-10
            else:
                assert actual[key] == value
    for j, expected in enumerate(summary["segment_anr_db"]):
        assert abs(float(r["anr"][j*100000+95000:(j+1)*100000].mean()) - expected) < 1e-7
    assert abs(float(r["cost"]["total"].mean()/19132) - summary["total_cost_ratio"]) < 1e-12
    trace = HERE / f"{out_stem}_python_trace.npz"
    np.savez_compressed(trace, anr=r["anr"], error=r["error"], drive=r["drive"],
                        R=r["R"], candidate_R=r["candidate_R"],
                        retiring_R=r["retiring_R"], resident_factor_coeffs=r["resident_factor_coeffs"],
                        ys_actual=r["ys_actual"], gamma=r["gamma"],
                        post_ramp_output_error=r["post_ramp_output_error"],
                        exp_evals=r["exp_evals"], **r["cost"])
    frozen = import_source(FROZEN, "n14_frozen_prefix")
    prefix = 74000  # Zero-based samples 0..73999; resort proposes at n=74000.
    base = frozen.simulate(arrays["x"][:prefix], arrays["d"][:prefix],
                           arrays["v"][:prefix], arrays["omega"][None],
                           arrays["rff_phase"][None], arrays["initial_A"][None],
                           arrays["initial_B"][None], args.mu,
                           dict(arm="once", proposal="signed_sort", ramp_samples=100,
                                ablate_spectral_veto=False))
    prefix_diffs = {}
    for key in ("anr", "error", "drive", "R", "candidate_R", "retiring_R",
                "resident_factor_coeffs", "ys_actual", "gamma", "exp_evals"):
        prefix_diffs[key] = float(np.max(np.abs(r[key][:prefix].astype(float)-base[key].astype(float))))
    for key in r["cost"]:
        prefix_diffs[f"cost_{key}"] = float(np.max(np.abs(r["cost"][key][:prefix]-base["cost"][key])))
    assert max(prefix_diffs.values()) == 0, prefix_diffs
    metadata = dict(phase="dev", case=args.case, run=args.run, mu=args.mu,
                    resort_interval=args.interval, input=provenance,
                    resort_source_sha256=sha(SOURCE), frozen_source_sha256=sha(FROZEN),
                    summary=summary_path.name, summary_sha256=sha(summary_path),
                    fixture=fixture.name, fixture_sha256=sha(fixture),
                    python_trace=trace.name, python_trace_sha256=sha(trace),
                    events=r["events"], frozen_prefix_length=prefix,
                    frozen_prefix_max_abs=prefix_diffs)
    (HERE / f"{out_stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(dict(stem=out_stem, events=len(r["events"]),
                          unfinished_candidate=int(r["candidate_R"][-1]),
                          frozen_prefix_max_abs=max(prefix_diffs.values()))), flush=True)


if __name__ == "__main__":
    main()
