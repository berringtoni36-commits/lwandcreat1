"""Materialize only E2/E3 run0..4 N14 development input arrays.

Every run is atomic and resumable.  There is no algorithm, selection, or
confirmation call in this script.  The old v12 numerical data path is checked
pointwise on each newly seeded input before a file is admitted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "experiments_v14" / "outputs" / "n14_dev_inputs"
sys.path.insert(0, str(ROOT / "experiments_v10"))
import anc_core as ac  # noqa: E402
from generator import make_case, seed_key, validate_case  # noqa: E402

SOURCES = (HERE / "PREREGISTRATION.md", HERE / "generator.py",
           HERE / "test_generator.py", HERE / "materialize_dev.py",
           ROOT / "experiments_v10" / "anc_core.py")
ARRAYS = ("x", "d", "v", "omega", "rff_phase", "initial_A", "initial_B")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
            for path in SOURCES}


def frozen_context():
    OUT.mkdir(parents=True, exist_ok=True)
    context = dict(phase="dev", cases=["E2", "E3"], runs=list(range(5)),
                   T=400_000, segment_length=100_000, source_sha256=source_hashes(),
                   python=sys.version, numpy=np.__version__, platform=platform.platform(),
                   no_controller_or_select_confirm=True)
    path = OUT / "source_context.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != context:
            raise RuntimeError("Development input source/environment changed; refuse mixed resume")
    else:
        atomic_json(path, context)
        for source in SOURCES:
            target = OUT / "sources" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return context


def v12_path_check(data):
    """Independently reconstruct v12's E2/E3 formula on new N14 streams."""
    if data.case not in ("E2", "E3") or len(data.x) != 400_000:
        raise ValueError("Only full E2/E3 development arrays")
    g = ac.fir(ac.P_PATH, data.x)
    a2 = np.repeat([.02, .08, .16, .02], (len(data.x)+3)//4)[:len(data.x)]
    a3 = np.repeat([.01, .04, .08, .01], (len(data.x)+3)//4)[:len(data.x)]
    expected_d = np.zeros(len(data.x))
    expected_d[2:] = g[:-2] + a2[2:]*g[:-2]**2 - a3[2:]*g[1:-1]**3
    if not np.array_equal(data.d, expected_d):
        raise RuntimeError("N14 primary array differs pointwise from v12 formula")
    noise_rng = np.random.default_rng(np.random.SeedSequence(
        seed_key("dev", data.case, data.run, "measurement_noise")))
    expected_v = (.01*noise_rng.standard_normal(len(data.x)) if data.case == "E2"
                  else .05*ac.sas_cms(noise_rng, 1.6, len(data.x)))
    if not np.array_equal(data.v, expected_v):
        raise RuntimeError("N14 noise array differs pointwise from v12 draw path")
    map_rng = np.random.default_rng(np.random.SeedSequence(
        seed_key("dev", data.case, data.run, "rff")))
    om, ph = ac.draw_rff(map_rng, 1, 500, 20, 3.9)
    if not np.array_equal(data.omega, om[0]) or not np.array_equal(data.rff_phase, ph[0]):
        raise RuntimeError("N14 map differs from v12 draw path")
    factor_rng = np.random.default_rng(np.random.SeedSequence(
        seed_key("dev", data.case, data.run, "initial_factors")))
    expected_a = .01*factor_rng.standard_normal((1, 25, 8))[0]
    expected_b = .01*factor_rng.standard_normal((1, 20, 8))[0]
    if not np.array_equal(data.initial_A, expected_a) or not np.array_equal(data.initial_B, expected_b):
        raise RuntimeError("N14 factor initialization differs from v12 shape/order")
    return dict(primary_pointwise_equal=True, noise_pointwise_equal=True,
                mapping_pointwise_equal=True, initial_factors_pointwise_equal=True,
                max_primary_abs_gap=0., max_noise_abs_gap=0.)


def save_one(case: str, run: int, context: dict) -> dict:
    data = make_case(case, run)
    validation = validate_case(data)
    agreement = v12_path_check(data)
    stem = f"case{case}_run{run:02d}"
    path = OUT / f"{stem}.npz"
    summary_path = OUT / f"{stem}.json"
    if summary_path.exists() and path.exists():
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        if row["file_sha256"] != digest(path) or row["source_sha256"] != context["source_sha256"]:
            raise RuntimeError(f"Saved {stem} checksum or source mismatch")
        with np.load(path, allow_pickle=False) as prior:
            for name in ARRAYS:
                if not np.array_equal(prior[name], getattr(data, name)):
                    raise RuntimeError(f"Saved {stem}/{name} differs on resume")
        return row
    if summary_path.exists() != path.exists():
        raise RuntimeError(f"Incomplete {stem}; inspect before recovery")
    temporary = OUT / f"{stem}.tmp.npz"
    np.savez_compressed(temporary, **{name: getattr(data, name) for name in ARRAYS},
                        segment_bounds=np.asarray(data.segment_bounds, dtype=np.int64))
    os.replace(temporary, path)
    row = dict(case=case, run=run, phase="dev", T=len(data.x),
               smoke_only=data.smoke_only, segment_bounds=data.segment_bounds,
               seed_keys=data.seed_keys, array_sha256=validation["array_sha256"],
               v12_formula_check=agreement, file_sha256=digest(path),
               source_sha256=context["source_sha256"])
    atomic_json(summary_path, row)
    return row


def main():
    context = frozen_context()
    rows = []
    for run in range(5):
        for case in ("E2", "E3"):
            row = save_one(case, run, context)
            rows.append(row)
            print(f"{case} run{run:02d}: {row['file_sha256']}", flush=True)
    if source_hashes() != context["source_sha256"]:
        raise RuntimeError("Source changed during materialization")
    manifest = dict(phase="dev", status="inputs_only", cases=["E2", "E3"],
                    runs_per_case=5, source_context_sha256=digest(OUT / "source_context.json"),
                    source_sha256=context["source_sha256"], entries=rows,
                    no_controller_or_select_confirm=True)
    atomic_json(OUT / "manifest.json", manifest)
    print(f"MANIFEST={OUT / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
