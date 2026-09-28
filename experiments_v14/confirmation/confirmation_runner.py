"""Guarded, resumable N14 confirmation runner. No import-time data generation.

The only official entry point first rechecks all 16 completed selection runs,
their unique locked μ, a separate confirmation freeze and parent seal. There
is deliberately no route from a draft freeze to confirm RNG materialization.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter
import uuid
from contextlib import redirect_stdout

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments_v14" / "selection"))
sys.path.insert(0, str(ROOT / "experiments_v14" / "protocol"))
sys.path.insert(0, str(ROOT / "experiments_v14" / "fixed_frontier"))
import generator as gen  # noqa: E402
import selection_runner as sr  # noqa: E402
import selection_analysis as sa  # noqa: E402
from kernel import simulate_fixed  # noqa: E402

CASES = ("E1", "E2", "E3", "C1", "C2", "C3", "C4")
PRIMARY = ("E2", "E3")
BRIDGE = ("E1", "C1", "C2", "C3", "C4")
BRIDGE_MU = {"E1": .2, "C1": .2, "C2": .1, "C3": .2, "C4": .1}
SOURCE_RELATIVE = tuple(dict.fromkeys(sr.SOURCE_RELATIVE + (
    "experiments_v14/confirmation/confirmation_runner.py",
    "experiments_v14/confirmation/confirmation_analysis.py",
    "experiments_v14/confirmation/check_confirmation.py",
    "experiments_v14/confirmation/test_confirmation.py",
    "experiments_v14/protocol/test_generator.py",
)))


def source_hashes():
    return {name: sr.sha(ROOT / name) for name in SOURCE_RELATIVE}


def environment_fingerprint():
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.show_config()
    return dict(python=sys.version, numpy=np.__version__,
                platform=platform.platform(),
                numpy_blas_config_sha256=hashlib.sha256(
                    stream.getvalue().encode("utf-8")).hexdigest())


@dataclass(frozen=True)
class FrozenConfirmation:
    path: Path
    digest: str
    seal_path: Path
    seal_digest: str
    selection_freeze_path: Path
    selection_freeze_digest: str
    selection_output: Path
    selection_analysis_digest: str
    selection_decision: dict
    document: dict


def _selection_integrity(selection_freeze_path: Path):
    """Read-only complete-grid verification. Never creates selection outputs."""
    selected = sr.check_freeze(selection_freeze_path)
    output = ROOT / "experiments_v14" / "outputs" / f"n14_select_{selected.digest[:16]}"
    if not output.is_dir():
        raise RuntimeError("Complete selection output directory absent")
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Selection manifest absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("phase") != "select"
            or manifest.get("frozen_grid_sha256") != selected.digest
            or manifest.get("frozen_seal_sha256") != selected.seal_digest):
        raise RuntimeError("Selection manifest provenance mismatch")
    for name, digest in (("freeze_candidate_grid.json", selected.digest),
                         ("freeze_selection_seal.json", selected.seal_digest)):
        if sr.sha(output / name) != digest:
            raise RuntimeError(f"Selection evidence snapshot mismatch: {name}")
    records = sa._load_all(output, selected.digest, selected.document["dynamic_config"])
    recomputed = sa.decide_selection(records)
    recomputed["frozen_grid_sha256"] = selected.digest
    analysis_path = output / "selection_analysis.json"
    if not analysis_path.is_file():
        raise RuntimeError("Selection analysis absent")
    saved = json.loads(analysis_path.read_text(encoding="utf-8"))
    if saved != recomputed or not saved["selection_pass"]:
        raise RuntimeError("Selection failed or analysis does not reproduce")
    if saved["selected_dynamic"] is None:
        raise RuntimeError("No unique dynamic μ pair selected")
    result_hashes = {f"{case}/{run:02d}": sr.sha(
        output / f"case{case}_run{run:02d}" / "result.json")
        for case in PRIMARY for run in range(8)}
    return selected, output, saved, sr.sha(analysis_path), result_hashes


def _locked_mu(decision):
    fixed = {case: {method: float(decision["fixed_selection"][case]
                                   ["fixed_methods"][method]["mu"])
                    for method in sr.FIXED_METHODS}
             for case in PRIMARY}
    winner = decision["selected_dynamic"]
    dynamic = {"E2": float(winner["mu_E2"]), "E3": float(winner["mu_E3"])}
    return fixed, dynamic


def check_confirmation_ready(freeze_path: Path | None,
                             selection_freeze_path: Path | None) -> FrozenConfirmation:
    """Fail closed before any confirm seed or output directory is touched."""
    if (freeze_path is None or not freeze_path.is_file()
            or freeze_path.name != "freeze_confirmation.json"):
        raise RuntimeError("Missing explicit freeze_confirmation.json")
    if selection_freeze_path is None:
        raise RuntimeError("Missing selection freeze path")
    document = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = dict(status="frozen_for_confirm", phase="confirm", version="N14",
                    primary_cases=list(PRIMARY), bridge_cases=list(BRIDGE),
                    paired_runs=20, k_threshold=17, null_success_probability=.5,
                    alpha_round=.05/12,
                    decision_rule="single_paired_J;K_ge_17_of_20;one_sided_exact_binomial",
                    bridge_mu=BRIDGE_MU, environment=environment_fingerprint())
    for key, expected in required.items():
        if document.get(key) != expected:
            raise RuntimeError(f"Confirmation freeze mismatch: {key}")
    hashes = source_hashes()
    if document.get("source_sha256") != hashes:
        raise RuntimeError("Confirmation source/dependency SHA mismatch")
    if document.get("protocol_sha256") != hashes["experiments_v14/protocol/PREREGISTRATION.md"]:
        raise RuntimeError("Confirmation protocol SHA mismatch")
    selected, output, decision, selection_analysis_sha, result_hashes = _selection_integrity(
        selection_freeze_path)
    fixed_mu, dynamic_mu = _locked_mu(decision)
    if (document.get("selection_freeze_sha256") != selected.digest
            or document.get("selection_output_manifest_sha256")
            != sr.sha(output / "manifest.json")
            or document.get("selection_analysis_sha256") != selection_analysis_sha
            or document.get("selection_result_sha256") != result_hashes
            or document.get("dynamic_config") != selected.document["dynamic_config"]
            or document.get("locked_fixed_mu") != fixed_mu
            or document.get("locked_dynamic_mu") != dynamic_mu):
        raise RuntimeError("Confirmation freeze differs from completed selection")
    if document.get("freeze_authorization") != "parent_reviewed_after_selection":
        raise RuntimeError("Parent confirmation review field absent")
    digest = sr.sha(freeze_path)
    seal_path = freeze_path.with_name("freeze_confirmation_seal.json")
    if not seal_path.is_file():
        raise RuntimeError("Separate parent confirmation seal absent")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal.get("status") != "reviewed_and_sealed_for_confirm"
            or seal.get("decision") != "run_one_locked_confirmation"
            or seal.get("confirmation_freeze_sha256") != digest
            or seal.get("selection_analysis_sha256") != selection_analysis_sha
            or not isinstance(seal.get("reason_after_selection_review"), str)
            or len(seal["reason_after_selection_review"].strip()) < 20):
        raise RuntimeError("Confirmation seal does not bind selection and freeze")
    return FrozenConfirmation(freeze_path.resolve(), digest, seal_path.resolve(),
                              sr.sha(seal_path), selected.path, selected.digest,
                              output, selection_analysis_sha, decision, document)


def make_confirm_input(case: str, run: int, frozen: FrozenConfirmation):
    """Actual N14 confirm arrays; unreachable through CLI before full preflight."""
    if not isinstance(frozen, FrozenConfirmation) or frozen.document["phase"] != "confirm":
        raise RuntimeError("Validated confirmation freeze required")
    if case not in CASES or not 0 <= run < 20:
        raise ValueError("Only the seven N14 cases, confirmation runs 0..19")
    length = gen.DEFAULT_LENGTHS[case]
    def rng(component):
        return np.random.default_rng(np.random.SeedSequence(
            gen.seed_key("confirm", case, run, component)))
    ref_rng = rng("reference")
    if case == "C1":
        x = gen.logistic_delay6(ref_rng, length)
    elif case == "C2":
        x = gen.alpha_reference(ref_rng, length)
    else:
        x = ref_rng.standard_normal(length)
    map_rng = rng("rff")
    omega = map_rng.normal(0., 1./3.9, (500, 20))
    phase = map_rng.uniform(0., 2.*np.pi, 500)
    extra = {}
    if case == "E1":
        d, low, high = gen.e1_teacher_primary(x, omega, phase,
                                               rng("teacher"), 100_000)
        extra = dict(teacher_low=low, teacher_high=high)
    elif case in PRIMARY:
        d = gen.e2e3_primary(x, 100_000)
    else:
        d = gen.static_primary(x)
    if case in ("C1", "C2"):
        v = np.zeros(length, np.float64)
    elif case in ("C4", "E3"):
        v = .05*gen.symmetric_alpha_stable(rng("measurement_noise"), 1.6, length)
    else:
        v = .01*rng("measurement_noise").standard_normal(length)
    factors = rng("initial_factors")
    a0 = .01*factors.standard_normal((25, 8))
    b0 = .01*factors.standard_normal((20, 8))
    if case in ("E1", "E2", "E3"):
        stages = 3 if case == "E1" else 4
        bounds = np.array([[i*100_000, (i+1)*100_000]
                           for i in range(stages)], np.int64)
    else:
        bounds = np.array([[0, length]], np.int64)
    return dict(x=x, d=d, v=v, omega=omega, rff_phase=phase,
                initial_A=a0, initial_B=b0, segment_bounds=bounds, **extra)


def _validate_case_data(data: dict, case: str, *, full: bool):
    t = len(data["x"])
    if full and t != gen.DEFAULT_LENGTHS[case]:
        raise RuntimeError("Confirmation case has wrong formal length")
    if any(data[k].shape != (t,) for k in ("x", "d", "v")):
        raise RuntimeError("Malformed confirmation physical signals")
    if (data["omega"].shape != (500, 20)
            or data["rff_phase"].shape != (500,)
            or data["initial_A"].shape != (25, 8)
            or data["initial_B"].shape != (20, 8)):
        raise RuntimeError("Malformed confirmation map/factors")
    bounds = data["segment_bounds"]
    expected_stages = 3 if case == "E1" else 4 if case in PRIMARY else 1
    if (bounds.shape != (expected_stages, 2)
            or not np.array_equal(bounds[:, 0], np.r_[0, bounds[:-1, 1]])
            or bounds[-1, 1] != t or not all(np.isfinite(v).all()
                                              for v in data.values())):
        raise RuntimeError("Malformed confirmation bounds or nonfinite data")


def _anr_from_error(d, error, bounds, tail):
    ad = ae = 0.0
    result = np.zeros(len(bounds), np.float64)
    count = np.zeros(len(bounds), np.int64)
    stage = 0
    for n in range(len(d)):
        ad = .999*ad + .001*abs(d[n])
        ae = .999*ae + .001*abs(error[n])
        if n >= bounds[stage, 1]-tail:
            result[stage] += 20*np.log10((ae+1e-12)/(ad+1e-12))
            count[stage] += 1
        if n+1 == bounds[stage, 1] and stage+1 < len(bounds):
            stage += 1
    if not np.array_equal(count, np.full(len(bounds), tail)):
        raise RuntimeError("Wrong confirmation ANR tail count")
    return result/count


def _fixed_rounding_check(data, output, error):
    """Independent S=[0,0,1,.5] replay with exact f32 storage bound."""
    dv = data["d"] + data["v"]
    worst_gap = worst_ratio = 0.0
    for c in range(output.shape[0]):
        yc = output[c].astype(np.float64)
        ec = error[c].astype(np.float64)
        replay = dv.copy()
        replay[2:] -= yc[:-2]
        replay[3:] -= .5*yc[:-3]
        gap = np.abs(replay-ec)
        yulp = np.spacing(np.abs(output[c]).astype(np.float32)).astype(np.float64)
        eulp = np.spacing(np.abs(error[c]).astype(np.float32)).astype(np.float64)
        allowance = 2e-12 + .5*eulp
        allowance[2:] += .5*yulp[:-2]
        allowance[3:] += .25*yulp[:-3]
        worst_gap = max(worst_gap, float(gap.max()))
        worst_ratio = max(worst_ratio, float(np.max(gap/allowance)))
    if worst_ratio > 1.001:
        raise RuntimeError("Fixed physical FIR exceeds f32 rounding bound")
    return dict(max_abs_gap=worst_gap, max_rounding_bound_ratio=worst_ratio)


def _save_fixed_selected(outdir, data, selected_mu, case, tail, phase):
    path = outdir / "fixed_selected.npz"
    meta_path = outdir / "fixed_result.json"
    if path.exists() and meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if sr.sha(path) != previous["traces_sha256"]:
            raise RuntimeError("Fixed confirmation trace hash changed")
        return previous
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    start = perf_counter()
    bounds = data["segment_bounds"]
    nstage = len(bounds)
    ends = np.full(4, len(data["x"]), np.int64)
    ends[:nstage] = bounds[:, 1]
    sums, counts, outputs, errors = simulate_fixed(
        data["x"], data["d"], data["v"], data["omega"], data["rff_phase"],
        data["initial_A"], data["initial_B"], sr.MU, ends, tail)
    if not np.array_equal(counts[:nstage], np.full(nstage, tail)):
        raise RuntimeError("Fixed confirmation tail count mismatch")
    indices = [5*k + list(sr.MU).index(float(selected_mu[name]))
               for k, name in enumerate(sr.FIXED_METHODS)]
    y, e = outputs[indices], errors[indices]
    anr = sums[indices, :nstage] / counts[:nstage]
    fir = _fixed_rounding_check(data, y, e)
    arrays = dict(actuator_output=y, physical_error=e,
                  segment_anr_db=anr, method_names=np.array(sr.FIXED_METHODS),
                  selected_mu=np.array([selected_mu[name] for name in sr.FIXED_METHODS]))
    trace_sha = sr._install_traces(path, arrays,
                                    previous["traces_sha256"] if previous else None)
    detail = sr.fixed_costs()
    record = dict(phase=phase, case=case, method_names=sr.FIXED_METHODS,
                  selected_mu={name: float(selected_mu[name]) for name in sr.FIXED_METHODS},
                  segment_anr_db=anr.tolist(), tail_counts=counts[:nstage].tolist(),
                  cost_ledger=detail,
                  total_multiplications={name: int(row["total_per_sample"]*len(data["x"]))
                                         for name, row in detail.items()},
                  physical_fir_replay=fir, traces_sha256=trace_sha,
                  elapsed_seconds=perf_counter()-start,
                  internal_kernel_branches=45, published_locked_branches=9)
    if previous is not None:
        if (previous["segment_anr_db"] != record["segment_anr_db"]
                or previous["selected_mu"] != record["selected_mu"]):
            raise RuntimeError("Regenerated fixed confirmation differs")
        return previous
    sr.atomic_json(meta_path, record)
    return record


def _save_dynamic_confirm(outdir, data, model, mu, config, case, tail, phase):
    path = outdir / "dynamic.npz"
    meta_path = outdir / "dynamic_result.json"
    if path.exists() and meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if sr.sha(path) != previous["traces_sha256"]:
            raise RuntimeError("Dynamic confirmation trace hash changed")
        return previous
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    result = model.simulate(data["x"], data["d"], data["v"],
                            data["omega"][None], data["rff_phase"][None],
                            data["initial_A"][None], data["initial_B"][None],
                            float(mu), dict(config))
    t = len(data["x"])
    required = ("anr", "error", "drive", "R", "candidate_R", "retiring_R",
                "resident_factor_coeffs", "ys_actual", "gamma")
    if any(key not in result or np.shape(result[key]) != (t,) for key in required):
        raise RuntimeError("Incomplete dynamic confirmation trace")
    ledger = {k: np.asarray(v) for k, v in result["cost"].items()}
    if (set(ledger) != {"core", "candidate", "retiring", "management", "reorder", "total"}
            or any(v.shape != (t,) for v in ledger.values())
            or not np.array_equal(ledger["total"],
                                  sum(ledger[k] for k in ledger if k != "total")+4)):
        raise RuntimeError("Dynamic confirmation multiplication ledger broken")
    if not np.isfinite(result["error"]).all() or np.any(ledger["total"] < 0):
        raise RuntimeError("Nonfinite confirmation trajectory or negative cost")
    fir = float(np.max(np.abs(gen.physical_error(data["d"], data["v"], result["drive"])
                              - result["error"])))
    if fir > 1e-9 or result["post_ramp_max_abs"] > 1e-10:
        raise RuntimeError("Dynamic confirmation physical FIR integrity failed")
    bounds = data["segment_bounds"]
    anr = _anr_from_error(data["d"], result["error"], bounds, tail)
    reported = np.array([result["anr"][hi-tail:hi].mean() for _, hi in bounds])
    if np.max(np.abs(anr-reported)) > 1e-4:
        raise RuntimeError("Dynamic confirmation ANR replay mismatch")
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays.update({f"cost_{key}": value for key, value in ledger.items()})
    trace_sha = sr._install_traces(path, arrays,
                                    previous["traces_sha256"] if previous else None)
    events = result.get("events", [])
    record = dict(phase=phase, case=case, mu=float(mu), config=config,
                  segment_anr_db=anr.tolist(),
                  segment_mean_mults=[float(ledger["total"][lo:hi].mean())
                                      for lo, hi in bounds],
                  total_multiplications=int(ledger["total"].sum()),
                  mean_mults=float(ledger["total"].mean()),
                  cost_ledger_sum={key: int(value.sum()) for key, value in ledger.items()},
                  events=events, rank_at_end=int(result["R"][-1]),
                  physical_fir_max_abs_gap=fir,
                  post_ramp_max_abs=float(result["post_ramp_max_abs"]),
                  physical_integrity_pass=True,
                  traces_sha256=trace_sha, elapsed_seconds=float(result["seconds"]))
    if previous is not None:
        if (previous["segment_anr_db"] != record["segment_anr_db"]
                or previous["cost_ledger_sum"] != record["cost_ledger_sum"]
                or previous["config"] != config):
            raise RuntimeError("Regenerated dynamic confirmation differs")
        return previous
    sr.atomic_json(meta_path, record)
    return record


def execute_case(data, model, outdir: Path, *, case: str, run: int, phase: str,
                 frozen_sha: str, fixed_mu: dict, dynamic_mu: float,
                 dynamic_config: dict, tail: int, source_guard=None,
                 fail_after: str | None = None):
    """Durable one-case job; tests may pass dev smoke arrays, CLI only confirm."""
    _validate_case_data(data, case, full=phase == "confirm")
    if source_guard is not None:
        source_guard()
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = dict(phase=phase, case=case, run=run,
                    frozen_confirmation_sha256=frozen_sha,
                    seed_keys={part: list(gen.seed_key("confirm", case, run, part))
                               for part in gen.COMPONENT_IDS}
                    if phase == "confirm" else None)
    input_meta = sr._save_input(outdir, data, metadata)
    if fail_after == "input":
        raise RuntimeError("Injected failure after confirmation input")
    fixed = _save_fixed_selected(outdir, data, fixed_mu, case, tail, phase)
    if source_guard is not None:
        source_guard()
    if fail_after == "fixed":
        raise RuntimeError("Injected failure after selected fixed branches")
    dynamic = _save_dynamic_confirm(outdir, data, model, dynamic_mu,
                                    dynamic_config, case, tail, phase)
    if source_guard is not None:
        source_guard()
    if fail_after == "dynamic":
        raise RuntimeError("Injected failure after locked dynamic branch")
    record = dict(phase=phase, case=case, run=run,
                  frozen_confirmation_sha256=frozen_sha,
                  input_sha256=input_meta["input_sha256"],
                  fixed_result_sha256=sr.sha(outdir / "fixed_result.json"),
                  fixed_traces_sha256=fixed["traces_sha256"],
                  dynamic_result_sha256=sr.sha(outdir / "dynamic_result.json"),
                  dynamic_traces_sha256=dynamic["traces_sha256"],
                  fixed_evaluations=9, dynamic_evaluations=1,
                  fixed_kernel_internal_branches=45)
    if source_guard is not None:
        source_guard()
    final = outdir / "result.json"
    if final.exists() and json.loads(final.read_text(encoding="utf-8")) != record:
        raise RuntimeError("Existing confirmation result differs on resume")
    if not final.exists():
        sr.atomic_json(final, record)
    return record


def prepare_output(frozen: FrozenConfirmation):
    """Only called after confirmation freeze; snapshots all source and evidence."""
    outdir = ROOT / "experiments_v14" / "outputs" / f"n14_confirm_{frozen.digest[:16]}"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = dict(phase="confirm", frozen_confirmation_sha256=frozen.digest,
                    confirmation_seal_sha256=frozen.seal_digest,
                    selection_freeze_sha256=frozen.selection_freeze_digest,
                    selection_analysis_sha256=frozen.selection_analysis_digest,
                    source_sha256=frozen.document["source_sha256"],
                    primary_cases=list(PRIMARY), bridge_cases=list(BRIDGE),
                    runs_per_case=20)
    path = outdir / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Confirmation output manifest changed")
    if not path.exists():
        sr.atomic_json(path, manifest)
    selection_manifest = frozen.selection_output / "manifest.json"
    snapshots = ((frozen.path, outdir / "freeze_confirmation.json", frozen.digest),
                 (frozen.seal_path, outdir / "freeze_confirmation_seal.json",
                  frozen.seal_digest),
                 (selection_manifest, outdir / "selection_manifest.json",
                  frozen.document["selection_output_manifest_sha256"]),
                 (frozen.selection_output / "freeze_candidate_grid.json",
                  outdir / "selection_freeze.json", frozen.selection_freeze_digest),
                 (frozen.selection_output / "freeze_selection_seal.json",
                  outdir / "selection_seal.json",
                  json.loads(selection_manifest.read_text(encoding="utf-8"))
                  ["frozen_seal_sha256"]),
                 (frozen.selection_output / "selection_analysis.json",
                  outdir / "selection_analysis.json", frozen.selection_analysis_digest))
    for source, target, digest in snapshots:
        if target.exists() and sr.sha(target) != digest:
            raise RuntimeError("Confirmation evidence snapshot differs")
        if not target.exists():
            shutil.copy2(source, target)
    for name in SOURCE_RELATIVE:
        target = outdir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sr.sha(target) != frozen.document["source_sha256"][name]:
            raise RuntimeError(f"Confirmation source snapshot differs: {name}")
        if not target.exists():
            shutil.copy2(ROOT / name, target)
    return outdir


def _case_settings(frozen: FrozenConfirmation, case: str):
    if case in PRIMARY:
        fixed = frozen.document["locked_fixed_mu"][case]
        dynamic = frozen.document["locked_dynamic_mu"][case]
        config = frozen.document["dynamic_config"]
    else:
        dynamic = BRIDGE_MU[case]
        fixed = {name: dynamic for name in sr.FIXED_METHODS}
        config = dict(frozen.document["dynamic_config"])
        if case.startswith("C"):
            config["arm"] = "fixed"  # numerical v10 bridge, no rank manager
            config["resort_interval"] = 0
    return fixed, dynamic, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--select-freeze", type=Path, required=True)
    parser.add_argument("--unlock-confirm", action="store_true")
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--run", type=int, choices=range(20), required=True)
    args = parser.parse_args()
    if not args.unlock_confirm:
        raise RuntimeError("Confirmation locked: explicit --unlock-confirm required")
    frozen = check_confirmation_ready(args.freeze, args.select_freeze)
    outdir = prepare_output(frozen)
    def guard():
        refreshed = check_confirmation_ready(frozen.path, frozen.selection_freeze_path)
        if (refreshed.digest != frozen.digest
                or refreshed.seal_digest != frozen.seal_digest
                or refreshed.selection_analysis_digest != frozen.selection_analysis_digest):
            raise RuntimeError("Confirmation freeze/selection changed during run")
    guard()
    run_dir = outdir / f"case{args.case}_run{args.run:02d}"
    attempts = run_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    started_utc = datetime.now(timezone.utc).isoformat()
    sr.atomic_json(attempts / f"{attempt_id}.start.json",
                   dict(attempt_id=attempt_id, case=args.case, run=args.run,
                        phase="confirm", frozen_confirmation_sha256=frozen.digest,
                        started_utc=started_utc))
    start = perf_counter()
    stages = {}
    status = "failed"
    error_type = None
    try:
        before = perf_counter()
        data = make_confirm_input(args.case, args.run, frozen)
        stages["input_generation"] = perf_counter()-before
        before = perf_counter()
        model = sr.load_model()
        stages["model_import"] = perf_counter()-before
        fixed_mu, dynamic_mu, dynamic_config = _case_settings(frozen, args.case)
        guard()
        before = perf_counter()
        result = execute_case(data, model, run_dir, case=args.case, run=args.run,
                              phase="confirm", frozen_sha=frozen.digest,
                              fixed_mu=fixed_mu, dynamic_mu=dynamic_mu,
                              dynamic_config=dynamic_config, tail=5000,
                              source_guard=guard)
        stages["execution_including_io"] = perf_counter()-before
        guard()
        status = "complete"
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        sr.atomic_json(attempts / f"{attempt_id}.finish.json",
                       dict(attempt_id=attempt_id, case=args.case, run=args.run,
                            phase="confirm", frozen_confirmation_sha256=frozen.digest,
                            started_utc=started_utc,
                            finished_utc=datetime.now(timezone.utc).isoformat(),
                            status=status, error_type=error_type,
                            stage_wall_seconds=stages,
                            total_wall_seconds=perf_counter()-start))


if __name__ == "__main__":
    main()
