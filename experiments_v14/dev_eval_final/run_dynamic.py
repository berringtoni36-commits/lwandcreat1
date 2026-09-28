"""Run the frozen-in-place N14 prototype on NEW development streams only.

This runner is intentionally unable to materialize select/confirm inputs.  It
snapshots the simulator source before importing it, so a later edit cannot be
mistaken for the code that produced an earlier development result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = ROOT / "experiments_v14" / "outputs" / "n14_dev_inputs"
LIVE = ROOT / "experiments_v13" / "online_selector" / "online_selector.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model():
    snapshot = HERE / "sources" / "online_selector.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        if sha(snapshot) != sha(LIVE):
            raise RuntimeError("Simulator changed since this dev evaluation began; use a new output directory")
    else:
        shutil.copy2(LIVE, snapshot)
    # The simulator resolves its repository root relative to __file__, so it
    # must be imported at its live path. The exact bytes are checked against
    # the immutable snapshot before and after import.
    spec = importlib.util.spec_from_file_location("n14_dev_online_selector", LIVE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if sha(snapshot) != sha(LIVE):
        raise RuntimeError("Simulator changed during import")
    return module, snapshot


def evaluate(case: str, run: int, model, snapshot: Path, mu_e2: float, mu_e3: float):
    if case not in ("E2", "E3") or not 0 <= run < 5:
        raise ValueError("Only five N14 development streams per case are allowed")
    source = INPUTS / f"case{case}_run{run:02d}.npz"
    with np.load(source, allow_pickle=False) as f:
        x, d, v, omega, ph, a0, b0, bounds = (f[k].copy() for k in
            ("x", "d", "v", "omega", "rff_phase", "initial_A",
             "initial_B", "segment_bounds"))
    assert x.shape == d.shape == v.shape == (400000,)
    assert bounds.tolist() == [[i*100000, (i+1)*100000] for i in range(4)]
    mu = mu_e2 if case == "E2" else mu_e3
    config = dict(arm="once", proposal="signed_sort", ramp_samples=100,
                  ablate_spectral_veto=False)
    result = model.simulate(x, d, v, omega[None], ph[None],
                            a0[None], b0[None], mu, config)
    seg = []
    for lo, hi in bounds:
        tail = slice(int(hi)-5000, int(hi))
        seg.append(dict(anr_db=float(result["anr"][tail].mean()),
                        mean_mults=float(result["cost"]["total"][int(lo):int(hi)].mean()),
                        mean_R=float(result["R"][int(lo):int(hi)].mean())))
    summary = dict(phase="dev", case=case, run=run, mu=mu, config=config,
                   source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                   simulator_sha256=sha(snapshot),
                   protocol_sha256=sha(ROOT / "experiments_v14" / "protocol" / "PREREGISTRATION.md"),
                   segments=seg,
                   events=result["events"],
                   mean_mults=float(result["cost"]["total"].mean()),
                   mean_cost_ratio_to_R4=float(result["cost"]["total"].mean()/19132),
                   accepted_prune=sum(e.get("accepted", False) for e in result["events"]
                                      if e.get("type") == "prune"),
                   physical_fir_max_abs=result["physical_fir_max_abs"],
                   seconds=result["seconds"])
    target = HERE / f"case{case}_run{run:02d}_mu{mu:g}"
    # The name ends in a decimal step size (for example mu0.1); with_suffix
    # would mistake ".1" for an extension and collide with mu0.2.
    result_path = target.with_name(target.name + ".npz")
    summary_path = target.with_name(target.name + "_summary.json")
    if result_path.exists() or summary_path.exists():
        raise FileExistsError("Development result exists; do not overwrite it")
    arrays = {k: result[k] for k in ("anr", "error", "drive", "R", "candidate_R",
                                      "retiring_R", "resident_factor_coeffs", "ys_actual", "gamma")}
    arrays.update({f"cost_{k}": v for k, v in result["cost"].items()})
    np.savez_compressed(result_path, **arrays)
    summary["result_sha256"] = sha(result_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(case, run, "ANR", [round(s["anr_db"], 3) for s in seg],
          "cost", round(summary["mean_cost_ratio_to_R4"], 4),
          "prune", summary["accepted_prune"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, nargs="+", default=[0])
    parser.add_argument("--cases", nargs="+", choices=("E2", "E3"),
                        default=["E2", "E3"])
    parser.add_argument("--mu-e2", type=float, default=.1)
    parser.add_argument("--mu-e3", type=float, default=.05)
    args = parser.parse_args()
    simulator, snapshot = load_model()
    for index in args.runs:
        for case in args.cases:
            evaluate(case, index, simulator, snapshot, args.mu_e2, args.mu_e3)
