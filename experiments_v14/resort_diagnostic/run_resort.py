"""Exploratory periodic R2 reordering on N14 development streams only.

This is deliberately separate from the frozen one-prune N14 simulator and
cannot materialize select or confirmation streams. Reordering times are a
fixed clock from the first accepted proposal, not supplied stage boundaries.
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
INPUTS = ROOT / "experiments_v14" / "outputs" / "n14_dev_inputs"
SOURCE = HERE / "selector_resort.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model():
    spec = importlib.util.spec_from_file_location("n14_resort_dev", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(case: str, index: int, mu: float, interval: int) -> None:
    if case not in ("E2", "E3") or index not in range(5):
        raise ValueError("Only N14 E2/E3 development runs 0..4 are allowed")
    inp = INPUTS / f"case{case}_run{index:02d}.npz"
    with np.load(inp, allow_pickle=False) as data:
        x, d, v, omega, phase, a0, b0, bounds = (
            data[key].copy() for key in
            ("x", "d", "v", "omega", "rff_phase", "initial_A",
             "initial_B", "segment_bounds")
        )
    assert x.shape == (400000,)
    assert bounds.tolist() == [[j * 100000, (j + 1) * 100000] for j in range(4)]
    config = dict(arm="once", proposal="signed_sort", ramp_samples=100,
                  ablate_spectral_veto=False, resort_interval=interval)
    r = model().simulate(x, d, v, omega[None], phase[None],
                         a0[None], b0[None], mu, config)
    out = dict(phase="dev_diagnostic", case=case, run=index, mu=mu,
               interval=interval, config=config, input_sha256=sha(inp),
               simulator_sha256=sha(SOURCE), events=r["events"],
               segment_anr_db=[float(r["anr"][hi - 5000:hi].mean())
                               for _, hi in bounds],
               total_cost_ratio=float(r["cost"]["total"].mean() / 19132),
               physical_fir_max_abs=r["physical_fir_max_abs"])
    stem = f"case{case}_run{index:02d}_mu{mu:g}_interval{interval}"
    path = HERE / f"{stem}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(case, index, out["segment_anr_db"], out["total_cost_ratio"],
          [(e["n"], e["type"], e.get("accepted")) for e in r["events"]], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("E2", "E3"), required=True)
    parser.add_argument("--runs", type=int, nargs="+", default=[0])
    parser.add_argument("--mu", type=float, default=.05)
    parser.add_argument("--interval", type=int, default=50000)
    args = parser.parse_args()
    for i in args.runs:
        run(args.case, i, args.mu, args.interval)
