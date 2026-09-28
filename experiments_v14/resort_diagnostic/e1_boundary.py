"""N14 development E1 known-rank migration boundary for periodic resort."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments_v14" / "protocol"))
from generator import make_case, validate_case

spec = importlib.util.spec_from_file_location("e1_dev_resort", HERE / "selector_resort.py")
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = make_case("E1", 0, phase="dev")
    manifest = validate_case(data)
    args = (data.x, data.d, data.v, data.omega[None], data.rff_phase[None],
            data.initial_A[None], data.initial_B[None], .2)
    fixed = model.simulate(*args, dict(arm="fixed", ramp_samples=100))
    periodic = model.simulate(*args, dict(arm="once", proposal="signed_sort",
                                          ramp_samples=100,
                                          ablate_spectral_veto=False,
                                          resort_interval=50000))
    rows = []
    for lo, hi in data.segment_bounds:
        f = float(fixed["anr"][hi - 5000:hi].mean())
        p = float(periodic["anr"][hi - 5000:hi].mean())
        rows.append(dict(bounds=[lo,hi], fixed_R4_anr_db=f,
                         periodic_anr_db=p, delta_db=p-f))
    out = dict(phase="dev", case="E1", run=0, mu=.2,
               generator_manifest=manifest,
               generator_sha256=checksum(ROOT / "experiments_v14" / "protocol" / "generator.py"),
               simulator_sha256=checksum(HERE / "selector_resort.py"),
               segments=rows, events=periodic["events"],
               total_cost_ratio=float(periodic["cost"]["total"].mean() /
                                      fixed["cost"]["total"].mean()),
               fixed_physical_fir_max_abs=fixed["physical_fir_max_abs"],
               periodic_physical_fir_max_abs=periodic["physical_fir_max_abs"])
    target = HERE / "E1_run00_boundary.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("E1 development migration", rows, out["total_cost_ratio"], flush=True)


if __name__ == "__main__":
    main()
