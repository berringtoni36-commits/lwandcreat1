"""Export only saved N14 development input/initialization for full MATLAB replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV_EVAL = ROOT / "experiments_v14" / "dev_eval_final"
SOURCE = ROOT / "experiments_v13" / "online_selector" / "online_selector.py"
sys.path.insert(0, str(ROOT / "experiments_v12" / ".deps"))
from scipy.io import savemat  # noqa: E402
from export_fixture import load_input  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("E2", "E3"), required=True)
    parser.add_argument("--run", type=int, choices=range(5), default=0)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--selector-sha", required=True)
    args = parser.parse_args()
    selector_sha = sha(SOURCE)
    assert selector_sha == args.selector_sha
    arrays, provenance = load_input(args.case, args.run)
    stem = f"case{args.case}_run{args.run:02d}_mu{args.mu:g}"
    summary_path = DEV_EVAL / f"{stem}_summary.json"
    trace_path = DEV_EVAL / f"{stem}.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["phase"] == "dev"
    assert summary["case"] == args.case and summary["run"] == args.run
    assert summary["mu"] == args.mu
    assert summary["source_sha256"] == provenance["source_sha256"]
    assert summary["simulator_sha256"] == selector_sha
    assert summary["config"] == dict(arm="once", proposal="signed_sort",
                                     ramp_samples=100, ablate_spectral_veto=False)
    assert summary["accepted_prune"] == 1
    assert sha(trace_path) == summary["result_sha256"]
    assert len(summary["events"]) == 2
    assert summary["events"][0]["type"] == "proposal_prune"
    assert summary["events"][1]["type"] == "prune" and summary["events"][1]["accepted"]
    out_stem = f"{args.case}_run{args.run:02d}_mu{args.mu:g}_60f6_full"
    HERE.mkdir(parents=True, exist_ok=True)
    fixture = HERE / f"{out_stem}.mat"
    savemat(fixture, dict(x=arrays["x"],d=arrays["d"],v=arrays["v"],
                          omega=arrays["omega"],rff_phase=arrays["rff_phase"],
                          initial_A=arrays["initial_A"][:, :4],
                          initial_B=arrays["initial_B"][:, :4],
                          mu=args.mu, S_PATH=np.array([0.,0.,1.,.5]),
                          first_propose=20000, train_samples=2000,
                          validation_samples=2000, ramp_samples=100,
                          propose_interval=40000, cooldown=12000,
                          spectral_threshold=.15, detect_rel=.28,
                          detect_hold=250), do_compression=True)
    metadata = dict(phase="dev", source=provenance,
                    selector_sha256=selector_sha,
                    reference_summary=str(summary_path.relative_to(ROOT)).replace("\\", "/"),
                    reference_summary_sha256=sha(summary_path),
                    reference_trace=str(trace_path.relative_to(ROOT)).replace("\\", "/"),
                    reference_trace_sha256=sha(trace_path),
                    fixture=fixture.name, fixture_sha256=sha(fixture),
                    expected_events=summary["events"],
                    expected_segments=summary["segments"])
    (HERE / f"{out_stem}.json").write_text(json.dumps(metadata, indent=2),
                                             encoding="utf-8")
    print(json.dumps(dict(fixture=fixture.name, selector_sha256=selector_sha,
                          events=summary["events"])))


if __name__ == "__main__":
    main()
