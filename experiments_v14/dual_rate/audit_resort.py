"""Independent N14 development-only replay and same-rank resort ablations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = ROOT / "experiments_v14/outputs/n14_dev_inputs"
PARENT = ROOT / "experiments_v14/resort_diagnostic"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path):
    name = f"audit_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def run(case, index, maintenance):
    assert case in ("E2", "E3") and index in range(5)
    assert maintenance in ("signed_sort", "identity", "fixed_random", "thin_qr", "clone")
    inp = INPUTS / f"case{case}_run{index:02d}.npz"
    with np.load(inp, allow_pickle=False) as data:
        x, d, v, omega, phase, a0, b0 = (
            data[key].copy() for key in
            ("x", "d", "v", "omega", "rff_phase", "initial_A", "initial_B")
        )
    assert x.shape == (400000,)
    source = PARENT / "selector_resort.py" if maintenance == "signed_sort" else HERE / "resort_control.py"
    model = module(source)
    config = dict(arm="once", proposal="signed_sort", ramp_samples=100,
                  ablate_spectral_veto=False, resort_interval=50000)
    if maintenance != "signed_sort":
        config["maintenance_proposal"] = maintenance
    r = model.simulate(x, d, v, omega[None], phase[None],
                       a0[None], b0[None], .05, config)
    cost = r["cost"]
    segments = [float(r["anr"][hi-5000:hi].mean())
                for hi in (100000, 200000, 300000, 400000)]
    stem = f"case{case}_run{index:02d}_mu0.05_interval50000"
    ref = json.loads((PARENT / f"{stem}.json").read_text(encoding="utf-8"))
    assert ref["phase"] == "dev_diagnostic"
    assert ref["input_sha256"] == sha(inp)
    assert ref["simulator_sha256"] == sha(PARENT / "selector_resort.py")
    if maintenance == "signed_sort":
        assert np.allclose(segments, ref["segment_anr_db"], rtol=0, atol=1e-12)
        assert abs(float(cost["total"].mean()/19132)-ref["total_cost_ratio"]) < 1e-12
        assert r["events"] == ref["events"]
    assert r["physical_fir_max_abs"] < 1e-10
    assert r["post_ramp_max_abs"] < 1e-10
    audit = dict(phase="dev_audit", case=case, run=index,
                 maintenance_proposal=maintenance, config=config,
                 input_sha256=sha(inp), controller_sha256=sha(source),
                 parent_controller_sha256=sha(PARENT / "selector_resort.py"),
                 segment_anr_db=segments,
                 segment_cost_ratio=[float(cost["total"][lo:hi].mean()/19132)
                                     for lo,hi in ((0,100000),(100000,200000),
                                                   (200000,300000),(300000,400000))],
                 segment_gap_to_signed_db=[a-b for a,b in zip(segments,ref["segment_anr_db"])],
                 cost_ratio=float(cost["total"].mean()/19132),
                 ledger_mean={k:float(a.mean()) for k,a in cost.items()},
                 events=r["events"],
                 physical_fir_max_abs=r["physical_fir_max_abs"],
                 post_ramp_max_abs=r["post_ramp_max_abs"], seconds=r["seconds"])
    dest = HERE / f"resort_audit_{case}_run{index:02d}_{maintenance}.json"
    dest.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(dict(case=case,run=index,maintenance=maintenance,
                          segments=segments,gap_to_signed=audit["segment_gap_to_signed_db"],
                          cost_ratio=audit["cost_ratio"],
                          accepted=sum(e.get("accepted",False) for e in r["events"]),
                          seconds=r["seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("E2", "E3"), required=True)
    parser.add_argument("--run", type=int, choices=range(5), required=True)
    parser.add_argument("--maintenance", choices=("signed_sort", "identity", "fixed_random", "thin_qr", "clone"), required=True)
    args = parser.parse_args()
    run(args.case, args.run, args.maintenance)
