"""One locked paired N14 confirmation analysis; never picks μ from confirm."""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

import confirmation_runner as cr
import selection_runner as sr


def original_v10_bridge_gap(data: dict, drive: np.ndarray,
                            error: np.ndarray, mu: float) -> dict:
    """Independent original KronRFF.step replay with manager disabled."""
    ac = sr.load_model().ac
    controller = ac.KronRFF("bridge", 1, 25, 20, 4,
                            mu/2, mu/2, 2., 1e-8, np.random.default_rng(0))
    controller.A = data["initial_A"][:, :4][None].copy()
    controller.B = data["initial_B"][:, :4][None].copy()
    xb = np.zeros((1, 20), np.float64)
    zh = np.zeros((4, 1, 500), np.float64)
    output_gap = error_gap = 0.0
    scale = np.sqrt(2./500.)
    for n in range(len(data["x"])):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = data["x"][n]
        z = scale*np.cos(np.einsum("rdm,rm->rd", data["omega"][None], xb)
                         + data["rff_phase"][None])
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2]+.5*zh[3]
        ys = controller.step(z, q, np.array([data["d"][n]+data["v"][n]]))[0]
        output_gap = max(output_gap, abs(float(controller.ybuf[0, 0])-float(drive[n])))
        error_gap = max(error_gap,
                        abs(float(data["d"][n]+data["v"][n]-ys)-float(error[n])))
    return dict(max_actuator_gap=output_gap, max_physical_error_gap=error_gap)


def exact_binomial_upper_tail(k: int, n: int = 20, p: float = .5) -> float:
    if not 0 <= k <= n or not 0 <= p <= 1:
        raise ValueError("Invalid exact binomial arguments")
    return sum(comb(n, j)*p**j*(1-p)**(n-j) for j in range(k, n+1))


def exact_one_sided_lower_bound(k: int, n: int = 20,
                                alpha: float = .05/12) -> float:
    """Clopper-Pearson lower bound by inverting the exact upper tail."""
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo+hi)/2
        if exact_binomial_upper_tail(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def _attempts(folder: Path, case: str, run: int, freeze_sha: str):
    directory = folder / "attempts"
    starts = list(directory.glob("*.start.json"))
    if not starts:
        raise RuntimeError(f"Confirmation case never attempted: {case}/{run}")
    finished = [json.loads(p.read_text(encoding="utf-8"))
                for p in directory.glob("*.finish.json")]
    if any(a.get("case") != case or a.get("run") != run
           or a.get("phase") != "confirm"
           or a.get("frozen_confirmation_sha256") != freeze_sha
           for a in finished):
        raise RuntimeError("Confirmation attempt provenance mismatch")
    return dict(started=len(starts), finished=len(finished),
                failed=sum(a["status"] != "complete" for a in finished),
                unclosed=len(starts)-len(finished),
                logged_wall_seconds=sum(a["total_wall_seconds"] for a in finished))


def _load_case(outdir: Path, frozen: cr.FrozenConfirmation, case: str, run: int):
    folder = outdir / f"case{case}_run{run:02d}"
    attempts = _attempts(folder, case, run, frozen.digest)
    final = folder / "result.json"
    if not final.is_file():
        return dict(case=case, run=run, attempted=True, valid=False,
                    failure="no_complete_result", attempts=attempts)
    try:
        meta = json.loads(final.read_text(encoding="utf-8"))
        if (meta["phase"] != "confirm" or meta["case"] != case
                or meta["run"] != run
                or meta["frozen_confirmation_sha256"] != frozen.digest
                or meta["fixed_evaluations"] != 9
                or meta["dynamic_evaluations"] != 1):
            raise RuntimeError("run manifest mismatch")
        input_meta = json.loads((folder / "input.json").read_text(encoding="utf-8"))
        if (input_meta["phase"] != "confirm" or input_meta["case"] != case
                or input_meta["run"] != run
                or input_meta["frozen_confirmation_sha256"] != frozen.digest
                or input_meta["input_sha256"] != meta["input_sha256"]
                or input_meta["seed_keys"] != {
                    part: list(cr.gen.seed_key("confirm", case, run, part))
                    for part in cr.gen.COMPONENT_IDS}):
            raise RuntimeError("new confirm seed provenance mismatch")
        input_path = folder / "input.npz"
        fixed_path = folder / "fixed_selected.npz"
        dynamic_path = folder / "dynamic.npz"
        fixed_meta_path = folder / "fixed_result.json"
        dynamic_meta_path = folder / "dynamic_result.json"
        for path, digest in ((input_path, meta["input_sha256"]),
                             (fixed_path, meta["fixed_traces_sha256"]),
                             (dynamic_path, meta["dynamic_traces_sha256"]),
                             (fixed_meta_path, meta["fixed_result_sha256"]),
                             (dynamic_meta_path, meta["dynamic_result_sha256"])):
            if sr.sha(path) != digest:
                raise RuntimeError(f"artifact SHA mismatch: {path.name}")
        fixed = json.loads(fixed_meta_path.read_text(encoding="utf-8"))
        dynamic = json.loads(dynamic_meta_path.read_text(encoding="utf-8"))
        chosen_fixed, chosen_dynamic, chosen_config = cr._case_settings(frozen, case)
        if (fixed["phase"] != "confirm" or fixed["selected_mu"] != chosen_fixed
                or dynamic["phase"] != "confirm"
                or dynamic["mu"] != chosen_dynamic
                or dynamic["config"] != chosen_config):
            raise RuntimeError("confirmation locked μ/config mismatch")
        with np.load(input_path, allow_pickle=False) as inp:
            data = {key: inp[key] for key in inp.files}
        cr._validate_case_data(data, case, full=True)
        expected = cr.make_confirm_input(case, run, frozen)
        if set(data) != set(expected) or any(
                not np.array_equal(data[key], expected[key]) for key in data):
            raise RuntimeError("saved input differs from frozen confirm generator")
        with np.load(fixed_path, allow_pickle=False) as trace:
            y = trace["actuator_output"]
            fixed_error = trace["physical_error"]
            if (y.shape != (9, len(data["x"]))
                    or fixed_error.shape != y.shape
                    or not np.array_equal(trace["segment_anr_db"],
                                          np.asarray(fixed["segment_anr_db"]))):
                raise RuntimeError("fixed trace ANR or shape mismatch")
            physical_fixed = cr._fixed_rounding_check(data, y, fixed_error)
            replay_fixed_anr = np.stack([
                cr._anr_from_error(data["d"], row, data["segment_bounds"], 5000)
                for row in fixed_error])
            if np.max(np.abs(replay_fixed_anr-np.asarray(
                    fixed["segment_anr_db"]))) > 1e-4:
                raise RuntimeError("fixed saved ANR differs from physical-error replay")
        expected_fixed_cost = sr.fixed_costs()
        if (fixed["cost_ledger"] != expected_fixed_cost
                or fixed["total_multiplications"] != {
                    name: int(row["total_per_sample"]*len(data["x"]))
                    for name, row in expected_fixed_cost.items()}):
            raise RuntimeError("fixed cost ledger differs from locked formula")
        with np.load(dynamic_path, allow_pickle=False) as trace:
            rank = trace["R"]
            drive, error = trace["drive"], trace["error"]
            total = trace["cost_total"]
            ledger_sum = sum(trace[f"cost_{key}"] for key in
                             ("core", "candidate", "retiring", "management", "reorder"))+4
            if (rank.shape != (len(data["x"]),)
                    or not np.array_equal(total, ledger_sum)
                    or int(total.sum()) != dynamic["total_multiplications"]):
                raise RuntimeError("dynamic rank/cost trace mismatch")
            if (abs(float(total.mean())-dynamic["mean_mults"]) > 1e-10
                    or dynamic["cost_ledger_sum"] != {
                        key: int(trace[f"cost_{key}"].sum()) for key in
                        ("core", "candidate", "retiring", "management", "reorder", "total")}
                    or np.max(np.abs(np.asarray(dynamic["segment_mean_mults"])
                        - np.asarray([total[lo:hi].mean()
                                      for lo, hi in data["segment_bounds"]]))) > 1e-10):
                raise RuntimeError("dynamic cost summaries differ from per-sample ledger")
            fir_gap = float(np.max(np.abs(
                cr.gen.physical_error(data["d"], data["v"], drive)-error)))
            if fir_gap > 1e-9 or not dynamic["physical_integrity_pass"]:
                raise RuntimeError("dynamic physical FIR mismatch")
            anr = cr._anr_from_error(data["d"], error, data["segment_bounds"], 5000)
            if np.max(np.abs(anr-np.asarray(dynamic["segment_anr_db"]))) > 1e-8:
                raise RuntimeError("dynamic saved ANR mismatch")
            accepted = [event for event in dynamic["events"]
                        if event.get("type") == "prune" and event.get("accepted")
                        and event.get("from_R") == 4 and event.get("to_R") == 2
                        and event.get("validation_samples", 0) > 0]
            finish = min((e.get("ramp_end", e["n"]) for e in accepted), default=None)
            retained_r2 = bool(finish is not None and finish < len(rank)
                               and np.all(rank[int(finish)+1:] == 2))
            bridge_gap = None
            if case.startswith("C"):
                if np.any(rank != 4) or dynamic["events"]:
                    raise RuntimeError("Closed-manager C bridge changed rank or proposed")
                bridge_gap = original_v10_bridge_gap(data, drive, error,
                                                     chosen_dynamic)
                if max(bridge_gap.values()) > 1e-10:
                    raise RuntimeError("Original v10 closed-manager bridge exceeds 1e-10")
        return dict(case=case, run=run, attempted=True, valid=True,
                    fixed_anr_db=fixed["segment_anr_db"],
                    dynamic_anr_db=dynamic["segment_anr_db"],
                    dynamic_mean_mults=dynamic["mean_mults"],
                    dynamic_total_mults=dynamic["total_multiplications"],
                    accepted_initial_prune=bool(accepted),
                    prune_finish_sample=int(finish) if finish is not None else None,
                    retained_r2=retained_r2,
                    all_events=dynamic["events"],
                    dynamic_physical_gap=fir_gap,
                    fixed_physical_rounding=physical_fixed,
                    original_v10_bridge_gap=bridge_gap,
                    attempts=attempts)
    except Exception as exc:
        # Preserve numerical/integrity failures as J=0; no post-hoc exclusion.
        return dict(case=case, run=run, attempted=True, valid=False,
                    failure=f"{type(exc).__name__}: {exc}", attempts=attempts)


def score_pair(e2: dict, e3: dict):
    """Frozen single J_i. Every failed condition stays visible by case."""
    reasons = {}
    raw = {}
    for case, row in (("E2", e2), ("E3", e3)):
        failures = []
        if not row.get("valid", False):
            failures.append(row.get("failure", "invalid_run"))
        else:
            fixed_r4 = np.asarray(row["fixed_anr_db"][3], dtype=float)
            dynamic = np.asarray(row["dynamic_anr_db"], dtype=float)
            if fixed_r4.shape != (4,) or dynamic.shape != (4,):
                failures.append("wrong_four_stage_ANR_shape")
            else:
                gap = dynamic-fixed_r4
                raw[case] = dict(fixed_R4_anr_db=fixed_r4.tolist(),
                                 dynamic_anr_db=dynamic.tolist(),
                                 stage_gap_db=gap.tolist())
                if np.any(gap > .5):
                    failures.append("relative_ANR_gt_0.5")
                if np.any(dynamic > -6.):
                    failures.append("dynamic_absolute_ANR_gt_-6")
                if np.any(fixed_r4 > -6.):
                    failures.append("fixed_R4_absolute_ANR_gt_-6")
            ratio = float(row["dynamic_mean_mults"])/19132.
            raw.setdefault(case, {})["online_cost_ratio"] = ratio
            if ratio > .9:
                failures.append("online_cost_gt_0.9")
            if not (row["accepted_initial_prune"] and row["retained_r2"]
                    and row["prune_finish_sample"] is not None
                    and row["prune_finish_sample"] <= 80_000):
                failures.append("no_actual_R4_to_R2_by_80000_retained")
        reasons[case] = failures
    return dict(J=int(not reasons["E2"] and not reasons["E3"]),
                failures=reasons, values=raw)


def summarize(primary: dict, bridge: dict, *, freeze_sha: str):
    if any(len(primary[c]) != 20 for c in ("E2", "E3")):
        raise RuntimeError("Exactly 20 paired primary runs required")
    pairs = []
    for i in range(20):
        scored = score_pair(primary["E2"][i], primary["E3"][i])
        pairs.append(dict(run=i, **scored))
    k = sum(row["J"] for row in pairs)
    alpha = .05/12
    p = exact_binomial_upper_tail(k, 20, .5)
    bridge_rows = {case: [dict(run=row["run"], valid=row["valid"],
                               failure=row.get("failure"),
                               dynamic_anr_db=row.get("dynamic_anr_db"),
                               fixed_anr_db=row.get("fixed_anr_db"),
                               events=row.get("all_events")) for row in rows]
                   for case, rows in bridge.items()}
    bridge_integrity = all(row["valid"] for rows in bridge.values() for row in rows)
    return dict(phase="confirm", frozen_confirmation_sha256=freeze_sha,
                n_paired=20, paired_runs=pairs, K=k, K_threshold=17,
                alpha_round=alpha, exact_one_sided_p=p,
                exact_one_sided_success_probability_lower_bound=
                exact_one_sided_lower_bound(k, 20, alpha),
                primary_statistical_pass=bool(k >= 17 and p <= alpha),
                bridge_implementation_integrity_pass=bridge_integrity,
                overall_claim_permitted=bool(k >= 17 and p <= alpha and bridge_integrity),
                bridge_descriptive=bridge_rows,
                interpretation="One locked paired J and exact binomial test; "
                "bridge failures block implementation integrity but do not alter K. "
                "Failed/invalid attempted runs remain failures; no replacement seeds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--select-freeze", type=Path, required=True)
    parser.add_argument("--unlock-confirm", action="store_true")
    args = parser.parse_args()
    if not args.unlock_confirm:
        raise RuntimeError("Confirmation analysis locked")
    frozen = cr.check_confirmation_ready(args.freeze, args.select_freeze)
    outdir = cr.prepare_output(frozen)
    primary = {case: [_load_case(outdir, frozen, case, i) for i in range(20)]
               for case in cr.PRIMARY}
    bridge = {case: [_load_case(outdir, frozen, case, i) for i in range(20)]
              for case in cr.BRIDGE}
    result = summarize(primary, bridge, freeze_sha=frozen.digest)
    path = outdir / "analysis.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != result:
        raise RuntimeError("Existing confirmation analysis differs")
    if not path.exists():
        sr.atomic_json(path, result)
    print(json.dumps(dict(K=result["K"], p=result["exact_one_sided_p"],
                          primary_pass=result["primary_statistical_pass"],
                          bridge_integrity=result["bridge_implementation_integrity_pass"]),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
