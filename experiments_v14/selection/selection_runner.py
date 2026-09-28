"""Guarded, resumable N14 selection runner. Importing it never makes data.

The CLI requires an explicit, matching frozen grid and --unlock-select before
any select-phase RNG is touched. This module provides no confirm entry point.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter
import uuid

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments_v14" / "protocol"))
sys.path.insert(0, str(ROOT / "experiments_v14" / "fixed_frontier"))
import generator as gen  # noqa: E402
from kernel import simulate_fixed  # noqa: E402
from run_frontier import costs as fixed_costs, physical_fir_replay  # noqa: E402
from selection_interface import choose_fixed_mu  # noqa: E402

MU = np.array([.05, .10, .20, .40, .80], np.float64)
MODEL_PATH = ROOT / "experiments_v14" / "resort_diagnostic" / "selector_resort.py"
SOURCE_RELATIVE = (
    "experiments_v14/protocol/PREREGISTRATION.md",
    "experiments_v14/protocol/generator.py",
    "experiments_v14/fixed_frontier/kernel.py",
    "experiments_v14/fixed_frontier/run_frontier.py",
    "experiments_v14/fixed_frontier/selection_interface.py",
    "experiments_v14/selection/selection_runner.py",
    "experiments_v14/selection/selection_analysis.py",
    "experiments_v14/selection/check_freeze.py",
    "experiments_v14/selection/test_selection.py",
    "experiments_v14/resort_diagnostic/selector_resort.py",
    "experiments_v14/resort_diagnostic/test_resort.py",
    "experiments_v12/counted_linalg.py",
    "experiments_v12/outputs/main_dev_round1/sources/experiments_v10/anc_core.py",
)
CONFIG = dict(arm="once", proposal="signed_sort", ramp_samples=100,
              ablate_spectral_veto=False, resort_interval=50_000)
FIXED_METHODS = [f"R{r}" for r in range(1, 9)] + ["full500"]
DEV_AUDIT_DIR = (ROOT / "experiments_v14" / "resort_diagnostic").resolve()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_RELATIVE}


def model_config_keys():
    """Require the grid to state every config.get key used by pure simulate."""
    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "simulate"]
    if len(functions) != 1:
        raise RuntimeError("Expected exactly one pure simulate function")
    keys = set()
    for node in ast.walk(functions[0]):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "config" and node.func.attr == "get"
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
    if not set(CONFIG) <= keys:
        raise RuntimeError("Pure simulator no longer exposes required config keys")
    return keys


def atomic_json(path: Path, value):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays):
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def _install_traces(path: Path, arrays: dict, expected_sha: str | None = None):
    """Recover either half of an interrupted trace/metadata pair safely."""
    if path.exists():
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(arrays):
                raise RuntimeError("Interrupted trace has different array names")
            for key, value in arrays.items():
                old = prior[key]
                equal = np.array_equal(old, value, equal_nan=True) if np.issubdtype(
                    old.dtype, np.number) else np.array_equal(old, value)
                if not equal:
                    raise RuntimeError(f"Interrupted trace differs: {key}")
        digest = sha(path)
        if expected_sha is not None and digest != expected_sha:
            raise RuntimeError("Interrupted trace SHA differs from metadata")
        return digest
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    digest = sha(temp)
    if expected_sha is not None and digest != expected_sha:
        temp.unlink()
        raise RuntimeError("Regenerated trace SHA differs from metadata")
    os.replace(temp, path)
    return digest


@dataclass(frozen=True)
class FrozenGrid:
    path: Path
    digest: str
    seal_path: Path
    seal_digest: str
    dev_audit_path: Path
    dev_audit_digest: str
    document: dict


def check_freeze(path: Path | None) -> FrozenGrid:
    """Fail closed before any selection RNG, output directory or model run."""
    if path is None or not path.is_file() or path.name != "freeze_candidate_grid.json":
        raise RuntimeError("Missing explicit freeze_candidate_grid.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(status="frozen_for_select", phase="select", version="N14",
                    cases=["E2", "E3"], runs_per_case=8, segment_length=100_000,
                    tail_samples=5_000, mu_grid=MU.tolist())
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(f"Frozen grid mismatch: {key}")
    config = document.get("dynamic_config")
    if (not isinstance(config, dict) or set(config) != model_config_keys()
            or any(config.get(key) != value for key, value in CONFIG.items())
            or any(value == "REVIEW_REQUIRED" for value in config.values())):
        raise RuntimeError("Frozen single-strategy config does not match pure simulate")
    hashes = source_hashes()
    if document.get("source_sha256") != hashes:
        raise RuntimeError("Frozen code/protocol/dependency SHA map mismatch")
    if document.get("protocol_sha256") != hashes[SOURCE_RELATIVE[0]]:
        raise RuntimeError("Frozen protocol SHA mismatch")
    protocol_text = (ROOT / SOURCE_RELATIVE[0]).read_text(encoding="utf-8")
    if re.search(r"resort_interval\s*=\s*50[_ ,]?000", protocol_text) is None:
        raise RuntimeError("N14 protocol has not registered 50000-sample periodic resort")
    if document.get("grid_kind") != "one_signed_sort_periodic_resort_spectral_physical_strategy":
        raise RuntimeError("Unrecognized or non-single-strategy grid")
    if document.get("cost_rule") != "C_R=12512+1655R;full500=14508;physical_FIR=4":
        raise RuntimeError("Frozen scalar multiplication rule mismatch")
    if document.get("selection_rule") != "fixed_8x4_mean_ANR;dynamic_25_joint_J_ge7of8":
        raise RuntimeError("Frozen selection rule mismatch")
    if document.get("freeze_authorization") != "parent_reviewed_after_dev":
        raise RuntimeError("Parent review field absent")
    audit_rel = document.get("dev_audit_relative_path")
    if not isinstance(audit_rel, str) or audit_rel.startswith("FILL_"):
        raise RuntimeError("Reviewed periodic-resort development audit path absent")
    audit_relative = Path(audit_rel)
    if audit_relative.is_absolute() or ".." in audit_relative.parts:
        raise RuntimeError("Development audit path must be workspace-relative")
    audit_path = (ROOT / audit_relative).resolve()
    try:
        audit_path.relative_to(DEV_AUDIT_DIR)
    except ValueError as exc:
        raise RuntimeError("Development audit must belong to periodic-resort candidate") from exc
    if audit_path.suffix.lower() not in (".md", ".json") or not audit_path.is_file():
        raise RuntimeError("Reviewed periodic-resort development audit file absent")
    audit_digest = sha(audit_path)
    grid_digest = sha(path)
    seal_path = path.with_name("freeze_selection_seal.json")
    if not seal_path.is_file():
        raise RuntimeError("Separate parent-reviewed freeze_selection_seal.json absent")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal.get("status") != "reviewed_and_sealed_for_select"
            or seal.get("decision") != "proceed_to_select"
            or seal.get("grid_sha256") != grid_digest
            or seal.get("dev_audit_sha256") != audit_digest
            or not isinstance(seal.get("reason_after_dev_review"), str)
            or len(seal["reason_after_dev_review"].strip()) < 20):
        raise RuntimeError("Selection seal does not bind grid and reviewed dev audit")
    return FrozenGrid(path.resolve(), grid_digest, seal_path.resolve(),
                      sha(seal_path), audit_path, audit_digest, document)


def _rng(case: str, run: int, component: str):
    return np.random.default_rng(np.random.SeedSequence(
        gen.seed_key("select", case, run, component)))


def make_select_input(case: str, run: int, frozen: FrozenGrid):
    """Only call after check_freeze; exact N14 E2/E3 formula and new namespace."""
    if not isinstance(frozen, FrozenGrid) or frozen.document.get("phase") != "select":
        raise RuntimeError("No validated frozen grid")
    if case not in ("E2", "E3") or not 0 <= run < 8:
        raise ValueError("Only paired E2/E3 selection runs 0..7 are defined")
    length = 400_000
    x = _rng(case, run, "reference").standard_normal(length)
    mapping = _rng(case, run, "rff")
    omega = mapping.normal(0., 1./3.9, (500, 20))
    phase = mapping.uniform(0., 2.*np.pi, 500)
    d = gen.e2e3_primary(x, 100_000)
    if case == "E2":
        v = .01 * _rng(case, run, "measurement_noise").standard_normal(length)
    else:
        v = .05 * gen.symmetric_alpha_stable(
            _rng(case, run, "measurement_noise"), 1.6, length)
    factors = _rng(case, run, "initial_factors")
    a0 = .01 * factors.standard_normal((25, 8))
    b0 = .01 * factors.standard_normal((20, 8))
    return dict(x=x, d=d, v=v, omega=omega, rff_phase=phase,
                initial_A=a0, initial_B=b0,
                segment_bounds=np.array([[i*100_000, (i+1)*100_000]
                                         for i in range(4)], np.int64))


def _validate_input(data, *, full):
    t = len(data["x"])
    if full and t != 400_000:
        raise RuntimeError("Selection input must be full length")
    if any(data[key].shape != (t,) for key in ("x", "d", "v")):
        raise RuntimeError("Invalid input signal shape")
    if (data["omega"].shape != (500, 20)
            or data["rff_phase"].shape != (500,)
            or data["initial_A"].shape != (25, 8)
            or data["initial_B"].shape != (20, 8)):
        raise RuntimeError("Invalid RFF or factor shape")
    bounds = data["segment_bounds"]
    if bounds.shape != (4, 2) or not np.array_equal(bounds[:, 0], np.r_[0, bounds[:-1, 1]]) \
            or bounds[-1, 1] != t or np.any(np.diff(bounds[:, 1]) <= 0):
        raise RuntimeError("Invalid four segment bounds")
    if not all(np.isfinite(data[key]).all() for key in data):
        raise RuntimeError("Nonfinite selection input")


def load_model():
    spec = importlib.util.spec_from_file_location("n14_selection_model", MODEL_PATH)
    model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)
    if not callable(getattr(model, "simulate", None)):
        raise RuntimeError("Frozen model lacks pure simulate")
    return model


def _anr_from_error(d, error, bounds, tail):
    ad = ae = 0.0
    sums = np.zeros(4, np.float64)
    counts = np.zeros(4, np.int64)
    stage = 0
    for n in range(len(d)):
        ad = .999 * ad + .001 * abs(d[n])
        ae = .999 * ae + .001 * abs(error[n])
        if n >= bounds[stage, 1] - tail:
            sums[stage] += 20*np.log10((ae+1e-12)/(ad+1e-12))
            counts[stage] += 1
        if n + 1 == bounds[stage, 1] and stage < 3:
            stage += 1
    if not np.array_equal(counts, np.full(4, tail)):
        raise RuntimeError("ANR tail counts wrong")
    return sums / counts


def _physical_gap(data, drive, error):
    expected = gen.physical_error(data["d"], data["v"], drive)
    gap = float(np.max(np.abs(expected - error)))
    if gap > 1e-9:
        raise RuntimeError(f"Physical FIR replay mismatch: {gap}")
    return gap


def _save_input(outdir, data, meta):
    path = outdir / "input.npz"
    prior = outdir / "input.json"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(data) or any(
                    not np.array_equal(saved[key], data[key]) for key in data):
                raise RuntimeError("Existing input disagrees with frozen generator")
    else:
        atomic_npz(path, **data)
    record = dict(**meta, input_sha256=sha(path),
                  array_sha256={key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                for key, value in data.items()})
    if prior.exists() and json.loads(prior.read_text(encoding="utf-8")) != record:
        raise RuntimeError("Input metadata differs on resume")
    if not prior.exists():
        atomic_json(prior, record)
    return record


def _save_fixed(outdir, data, tail, phase):
    path = outdir / "fixed_traces.npz"
    meta_path = outdir / "fixed_result.json"
    if path.exists() and meta_path.exists():
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        if sha(path) != record["traces_sha256"]:
            raise RuntimeError("Fixed trace hash changed on resume")
        return record
    prior_record = (json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta_path.exists() else None)
    start = perf_counter()
    bounds = data["segment_bounds"]
    sums, counts, y, error = simulate_fixed(
        data["x"], data["d"], data["v"], data["omega"], data["rff_phase"],
        data["initial_A"], data["initial_B"], MU, bounds[:, 1], tail)
    if not np.array_equal(counts, np.full(4, tail)) or not np.isfinite(sums).all():
        raise RuntimeError("Invalid fixed ANR or tail counts")
    fir = physical_fir_replay(data, y, error)
    arrays = dict(actuator_output=y, physical_error=error,
                  segment_anr_db=sums / counts,
                  method_names=np.array([f"{name}_mu{m:.2f}"
                                         for name in FIXED_METHODS for m in MU]))
    trace_sha = _install_traces(path, arrays,
                                prior_record["traces_sha256"] if prior_record else None)
    detail = fixed_costs()
    cost = {name: row["total_per_sample"] for name, row in detail.items()}
    record = dict(phase=phase, method_count=45, mu_grid=MU.tolist(),
                  segment_anr_db=(sums / counts).tolist(),
                  costs_per_sample=cost,
                  total_mults_by_method={k: int(v*len(data["x"])) for k, v in cost.items()},
                  tail_counts=counts.tolist(), physical_fir_replay=fir,
                  cost_ledger=detail,
                  traces_sha256=trace_sha, elapsed_seconds=perf_counter()-start)
    if prior_record is not None:
        if (prior_record["segment_anr_db"] != record["segment_anr_db"]
                or prior_record["phase"] != phase):
            raise RuntimeError("Regenerated fixed result differs from metadata")
        return prior_record
    atomic_json(meta_path, record)
    return record


def _save_dynamic(outdir, data, model, mu, tail, config, phase):
    label = f"dynamic_mu{mu:.2f}"
    path = outdir / f"{label}.npz"
    meta_path = outdir / f"{label}.json"
    if path.exists() and meta_path.exists():
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        if sha(path) != record["traces_sha256"]:
            raise RuntimeError("Dynamic trace hash changed on resume")
        return record
    prior_record = (json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta_path.exists() else None)
    result = model.simulate(data["x"], data["d"], data["v"],
                            data["omega"][None], data["rff_phase"][None],
                            data["initial_A"][None], data["initial_B"][None],
                            float(mu), dict(config))
    t = len(data["x"])
    required = ("anr", "error", "drive", "R", "candidate_R", "retiring_R",
                "resident_factor_coeffs", "ys_actual", "gamma", "cost")
    if any(key not in result for key in required):
        raise RuntimeError("Incomplete dynamic simulator result")
    for name in required[:-1]:
        if result[name].shape != (t,):
            raise RuntimeError(f"Wrong dynamic trace shape: {name}")
    if set(result["cost"]) != {"core", "candidate", "retiring", "management", "reorder", "total"}:
        raise RuntimeError("Unexpected dynamic ledger categories")
    ledger = {k: np.asarray(v) for k, v in result["cost"].items()}
    if any(v.shape != (t,) for v in ledger.values()):
        raise RuntimeError("Wrong dynamic ledger length")
    expected_total = sum(ledger[k] for k in ledger if k != "total") + 4
    if not np.array_equal(ledger["total"], expected_total):
        raise RuntimeError("Dynamic scalar multiplication ledger does not sum")
    if np.any(ledger["total"] < 0) or not np.isfinite(result["error"]).all():
        raise RuntimeError("Nonfinite or negative dynamic trajectory")
    fir_gap = _physical_gap(data, result["drive"], result["error"])
    if not np.isfinite(result["post_ramp_max_abs"]) or result["post_ramp_max_abs"] > 1e-10:
        raise RuntimeError("Post-transition physical history check failed")
    bounds = data["segment_bounds"]
    anr = _anr_from_error(data["d"], result["error"], bounds, tail)
    reported = np.array([result["anr"][hi-tail:hi].mean() for lo, hi in bounds])
    if np.max(np.abs(anr-reported)) > 1e-4:
        raise RuntimeError("Dynamic ANR replay differs from simulator")
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays.update({f"cost_{k}": v for k, v in ledger.items()})
    trace_sha = _install_traces(path, arrays,
                                prior_record["traces_sha256"] if prior_record else None)
    events = result.get("events", [])
    accepted = [e for e in events if e.get("type") == "prune" and e.get("accepted")]
    record = dict(phase=phase, mu=float(mu), config=config,
                  segment_anr_db=anr.tolist(),
                  segment_mean_mults=[float(ledger["total"][lo:hi].mean())
                                      for lo, hi in bounds],
                  total_multiplications=int(ledger["total"].sum()),
                  mean_mults=float(ledger["total"].mean()),
                  cost_ledger_sum={k: int(v.sum()) for k, v in ledger.items()},
                  physical_fir_max_abs_gap=fir_gap,
                  post_ramp_max_abs=float(result["post_ramp_max_abs"]),
                  anr_replay_max_abs_gap_db=float(np.max(np.abs(anr-reported))),
                  accepted_prune_count=len(accepted), events=events,
                  rank_at_end=int(result["R"][-1]),
                  physical_integrity_pass=True,
                  traces_sha256=trace_sha, elapsed_seconds=float(result["seconds"]))
    if prior_record is not None:
        if (prior_record["segment_anr_db"] != record["segment_anr_db"]
                or prior_record["cost_ledger_sum"] != record["cost_ledger_sum"]
                or prior_record["phase"] != phase):
            raise RuntimeError("Regenerated dynamic result differs from metadata")
        return prior_record
    atomic_json(meta_path, record)
    return record


def execute_bundle(data, model, outdir: Path, *, case: str, run: int,
                   phase: str, frozen_sha: str, config: dict,
                   tail: int, fail_after: str | None = None,
                   source_guard=None):
    """One recoverable job; smoke tests pass dev arrays, official CLI only select."""
    _validate_input(data, full=phase == "select")
    if source_guard is not None:
        source_guard()
    outdir.mkdir(parents=True, exist_ok=True)
    meta = dict(phase=phase, case=case, run=run, frozen_grid_sha256=frozen_sha,
                seed_keys={part: list(gen.seed_key(phase, case, run, part))
                           for part in gen.COMPONENT_IDS} if phase == "select" else None)
    input_meta = _save_input(outdir, data, meta)
    if fail_after == "input":
        raise RuntimeError("Injected failure after durable input")
    fixed = _save_fixed(outdir, data, tail, phase)
    if source_guard is not None:
        source_guard()
    if fail_after == "fixed":
        raise RuntimeError("Injected failure after durable fixed frontier")
    dynamic = {}
    for mu in MU:
        if source_guard is not None:
            source_guard()
        dynamic[f"{mu:.2f}"] = _save_dynamic(outdir, data, model, mu, tail, config, phase)
        if source_guard is not None:
            source_guard()
        if fail_after == f"dynamic_{mu:.2f}":
            raise RuntimeError(f"Injected failure after durable dynamic {mu:.2f}")
    record = dict(phase=phase, case=case, run=run, frozen_grid_sha256=frozen_sha,
                  input_sha256=input_meta["input_sha256"],
                  fixed_result_sha256=sha(outdir / "fixed_result.json"),
                  fixed_traces_sha256=fixed["traces_sha256"],
                  dynamic_result_sha256={k: sha(outdir / f"dynamic_mu{k}.json")
                                         for k in dynamic},
                  dynamic_traces_sha256={k: v["traces_sha256"]
                                         for k, v in dynamic.items()},
                  fixed_evaluations=45, dynamic_evaluations=5,
                  fixed_core_wall_seconds=fixed["elapsed_seconds"],
                  dynamic_wall_seconds=sum(v["elapsed_seconds"] for v in dynamic.values()))
    final = outdir / "result.json"
    if source_guard is not None:
        source_guard()
    if final.exists() and json.loads(final.read_text(encoding="utf-8")) != record:
        raise RuntimeError("Existing final result differs on resume")
    if not final.exists():
        atomic_json(final, record)
    return record


def prepare_output(frozen: FrozenGrid):
    outdir = ROOT / "experiments_v14" / "outputs" / f"n14_select_{frozen.digest[:16]}"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / "manifest.json"
    value = dict(phase="select", frozen_grid_sha256=frozen.digest,
                 frozen_seal_sha256=frozen.seal_digest,
                 dev_audit_sha256=frozen.dev_audit_digest,
                 source_sha256=frozen.document["source_sha256"],
                 cases=["E2", "E3"], runs_per_case=8, mu_grid=MU.tolist())
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != value:
        raise RuntimeError("Selection manifest mismatch")
    if not manifest.exists():
        atomic_json(manifest, value)
    frozen_copy = outdir / "freeze_candidate_grid.json"
    if frozen_copy.exists() and sha(frozen_copy) != frozen.digest:
        raise RuntimeError("Selection freeze snapshot differs")
    if not frozen_copy.exists():
        shutil.copy2(frozen.path, frozen_copy)
    seal_copy = outdir / "freeze_selection_seal.json"
    if seal_copy.exists() and sha(seal_copy) != frozen.seal_digest:
        raise RuntimeError("Selection seal snapshot differs")
    if not seal_copy.exists():
        shutil.copy2(frozen.seal_path, seal_copy)
    audit_copy = outdir / "reviewed_development_audit" / frozen.dev_audit_path.name
    audit_copy.parent.mkdir(parents=True, exist_ok=True)
    if audit_copy.exists() and sha(audit_copy) != frozen.dev_audit_digest:
        raise RuntimeError("Selection development audit snapshot differs")
    if not audit_copy.exists():
        shutil.copy2(frozen.dev_audit_path, audit_copy)
    for name in SOURCE_RELATIVE:
        target = outdir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != frozen.document["source_sha256"][name]:
            raise RuntimeError(f"Selection source snapshot differs: {name}")
        if not target.exists():
            shutil.copy2(ROOT / name, target)
    return outdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--unlock-select", action="store_true")
    parser.add_argument("--case", choices=("E2", "E3"), required=True)
    parser.add_argument("--run", type=int, choices=range(8), required=True)
    args = parser.parse_args()
    if not args.unlock_select:
        raise RuntimeError("Selection remains locked: explicit --unlock-select required")
    frozen = check_freeze(args.freeze)
    outdir = prepare_output(frozen)
    def guard():
        refreshed = check_freeze(frozen.path)
        if (refreshed.digest != frozen.digest
                or refreshed.seal_digest != frozen.seal_digest
                or refreshed.dev_audit_digest != frozen.dev_audit_digest):
            raise RuntimeError("Frozen grid, seal, or reviewed audit changed during selection")
    guard()
    run_dir = outdir / f"case{args.case}_run{args.run:02d}"
    attempts = run_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    start_utc = datetime.now(timezone.utc).isoformat()
    atomic_json(attempts / f"{attempt_id}.start.json",
                dict(attempt_id=attempt_id, phase="select", case=args.case,
                     run=args.run, frozen_grid_sha256=frozen.digest,
                     started_utc=start_utc))
    attempt_start = perf_counter()
    stage_wall = {}
    status = "failed"
    error_type = None
    try:
        before = perf_counter()
        data = make_select_input(args.case, args.run, frozen)
        stage_wall["input_generation"] = perf_counter() - before
        before = perf_counter()
        model = load_model()
        stage_wall["model_import"] = perf_counter() - before
        guard()
        before = perf_counter()
        record = execute_bundle(data, model, run_dir,
                                case=args.case, run=args.run, phase="select",
                                frozen_sha=frozen.digest,
                                config=frozen.document["dynamic_config"],
                                tail=frozen.document["tail_samples"],
                                source_guard=guard)
        stage_wall["bundle_execution_including_io"] = perf_counter() - before
        guard()
        status = "complete"
        print(json.dumps(record, ensure_ascii=False), flush=True)
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        atomic_json(attempts / f"{attempt_id}.finish.json",
                    dict(attempt_id=attempt_id, phase="select", case=args.case,
                         run=args.run, frozen_grid_sha256=frozen.digest,
                         started_utc=start_utc,
                         finished_utc=datetime.now(timezone.utc).isoformat(),
                         status=status, error_type=error_type,
                         stage_wall_seconds=stage_wall,
                         total_wall_seconds=perf_counter()-attempt_start))


if __name__ == "__main__":
    main()
