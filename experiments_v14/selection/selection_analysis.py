"""Complete-grid N14 select analysis; no data generator and no confirm path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from selection_runner import (FIXED_METHODS, MU, ROOT, atomic_json,
                              check_freeze, prepare_output, sha)
from selection_interface import choose_fixed_mu


def _load_all(outdir: Path, freeze_sha: str, frozen_config: dict):
    records = {}
    for case in ("E2", "E3"):
        case_rows = []
        for run in range(8):
            folder = outdir / f"case{case}_run{run:02d}"
            final = folder / "result.json"
            if not final.is_file():
                raise RuntimeError(f"Incomplete 800-evaluation selection grid: {case}/{run}")
            manifest = json.loads(final.read_text(encoding="utf-8"))
            if (manifest["phase"] != "select" or manifest["case"] != case
                    or manifest["run"] != run
                    or manifest["frozen_grid_sha256"] != freeze_sha
                    or manifest["fixed_evaluations"] != 45
                    or manifest["dynamic_evaluations"] != 5):
                raise RuntimeError("Selection per-run manifest mismatch")
            if sha(folder / "input.npz") != manifest["input_sha256"]:
                raise RuntimeError("Selection input SHA mismatch")
            attempt_dir = folder / "attempts"
            starts = {p.name.removesuffix(".start.json") for p in
                      attempt_dir.glob("*.start.json")}
            finishes = [json.loads(p.read_text(encoding="utf-8")) for p in
                        attempt_dir.glob("*.finish.json")]
            if not finishes or not any(a["status"] == "complete" for a in finishes):
                raise RuntimeError("Missing complete selection attempt log")
            if any(a["phase"] != "select" or a["case"] != case
                   or a["run"] != run or a["frozen_grid_sha256"] != freeze_sha
                   for a in finishes):
                raise RuntimeError("Selection attempt provenance mismatch")
            unfinished = starts - {a["attempt_id"] for a in finishes}
            input_meta = json.loads((folder / "input.json").read_text(encoding="utf-8"))
            if (input_meta["phase"] != "select" or input_meta["case"] != case
                    or input_meta["run"] != run
                    or input_meta["frozen_grid_sha256"] != freeze_sha
                    or input_meta["input_sha256"] != manifest["input_sha256"]):
                raise RuntimeError("Selection input metadata mismatch")
            fixed_path = folder / "fixed_result.json"
            fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
            if fixed["phase"] != "select" or fixed["method_count"] != 45:
                raise RuntimeError("Selection fixed method metadata mismatch")
            if (sha(fixed_path) != manifest["fixed_result_sha256"]
                    or sha(folder / "fixed_traces.npz") != manifest["fixed_traces_sha256"]):
                raise RuntimeError("Selection fixed SHA mismatch")
            dynamic = {}
            for mu in MU:
                key = f"{mu:.2f}"
                path = folder / f"dynamic_mu{key}.json"
                info = json.loads(path.read_text(encoding="utf-8"))
                if (info["phase"] != "select" or info["mu"] != float(mu)
                        or info["config"] != frozen_config
                        or info["post_ramp_max_abs"] > 1e-10):
                    raise RuntimeError("Selection dynamic configuration mismatch")
                if (sha(path) != manifest["dynamic_result_sha256"][key]
                        or sha(folder / f"dynamic_mu{key}.npz")
                        != manifest["dynamic_traces_sha256"][key]):
                    raise RuntimeError("Selection dynamic SHA mismatch")
                with np.load(folder / f"dynamic_mu{key}.npz", allow_pickle=False) as trace:
                    ranks = trace["R"]
                    accepted = [e for e in info["events"]
                                if e.get("type") == "prune" and e.get("accepted")
                                and e.get("from_R") == 4 and e.get("to_R") == 2]
                    finish = min((e.get("ramp_end", e["n"]) for e in accepted), default=None)
                    stable_r2 = (finish is not None and finish < len(ranks)
                                 and np.all(ranks[int(finish)+1:] == 2))
                info["accepted_R4_to_R2_by_80000"] = bool(
                    finish is not None and finish <= 80_000 and stable_r2)
                info["prune_finish_sample"] = int(finish) if finish is not None else None
                dynamic[key] = info
            case_rows.append(dict(case=case, run=run, phase="select",
                                  segment_anr_db=fixed["segment_anr_db"],
                                  fixed=fixed, dynamic=dynamic,
                                  run_wall_seconds=sum(a["total_wall_seconds"]
                                                       for a in finishes),
                                  finished_attempts=len(finishes),
                                  unfinished_attempts=len(unfinished)))
        records[case] = case_rows
    return records


def decide_selection(records):
    """Select fixed μ by ANR, then scan all 25 paired dynamic combinations."""
    if set(records) != {"E2", "E3"} or any(len(records[c]) != 8 for c in records):
        raise ValueError("Need all eight paired runs per case")
    fixed_selection = {case: choose_fixed_mu(records[case]) for case in ("E2", "E3")}
    r4_index = {case: list(MU).index(fixed_selection[case]["fixed_methods"]["R4"]["mu"])
                for case in ("E2", "E3")}
    r4_by_case = {}
    for case in ("E2", "E3"):
        idx = 15 + r4_index[case]
        r4_by_case[case] = [np.asarray(row["segment_anr_db"][idx], dtype=float)
                            for row in records[case]]
    r4_floor_paired = [bool(all(np.all(r4_by_case[case][i] <= -6.)
                                for case in ("E2", "E3"))) for i in range(8)]
    scans = []
    for mu2 in MU:
        for mu3 in MU:
            choices = {"E2": f"{mu2:.2f}", "E3": f"{mu3:.2f}"}
            successes = []
            reasons = []
            ratios = []
            degradations = []
            for i in range(8):
                pair_pass = True
                pair_reasons = []
                for case in ("E2", "E3"):
                    info = records[case][i]["dynamic"][choices[case]]
                    anr = np.asarray(info["segment_anr_db"], dtype=float)
                    r4 = r4_by_case[case][i]
                    if anr.shape != (4,) or not np.isfinite(anr).all():
                        raise RuntimeError("Malformed dynamic segment ANR")
                    degradation = anr-r4
                    degradations.extend(degradation.tolist())
                    ratio = float(info["mean_mults"])/19132.
                    ratios.append(ratio)
                    failures = []
                    if np.any(degradation > .5):
                        failures.append("relative_ANR")
                    if np.any(anr > -6.):
                        failures.append("absolute_ANR")
                    if np.any(r4 > -6.):
                        failures.append("R4_absolute_ANR")
                    if ratio > .9:
                        failures.append("online_cost")
                    if not info.get("accepted_R4_to_R2_by_80000", False):
                        failures.append("actual_prune_by_80000_and_retained")
                    if not info.get("physical_integrity_pass", False):
                        failures.append("physical_integrity")
                    if failures:
                        pair_pass = False
                        pair_reasons.append({"case": case, "failures": failures})
                successes.append(pair_pass)
                reasons.append(pair_reasons)
            scans.append(dict(mu_E2=float(mu2), mu_E3=float(mu3),
                              paired_J=successes, paired_successes=int(sum(successes)),
                              per_pair_failures=reasons,
                              mean_total_cost_ratio=float(np.mean(ratios)),
                              max_stage_degradation_db=float(max(degradations)),
                              eligible=int(sum(successes)) >= 7
                              and sum(r4_floor_paired) >= 7))
    eligible = sorted((row for row in scans if row["eligible"]),
                      key=lambda row: (row["mean_total_cost_ratio"],
                                       row["max_stage_degradation_db"],
                                       row["mu_E2"], row["mu_E3"]))
    total_eval_mults = 0
    total_wall = 0.
    incomplete_attempts = 0
    finished_attempts = 0
    for case in ("E2", "E3"):
        for row in records[case]:
            t = 400_000
            total_eval_mults += t * sum((12512+1655*r)*5 for r in range(1, 9))
            total_eval_mults += t*14508*5
            total_eval_mults += sum(int(row["dynamic"][f"{mu:.2f}"]["total_multiplications"])
                                    for mu in MU)
            total_wall += row["run_wall_seconds"]
            incomplete_attempts += row.get("unfinished_attempts", 0)
            finished_attempts += row.get("finished_attempts", 1)
    return dict(phase="select", fixed_selection=fixed_selection,
                r4_absolute_floor_paired=r4_floor_paired,
                r4_floor_count=int(sum(r4_floor_paired)),
                dynamic_combinations=scans,
                selected_dynamic=(eligible[0] if eligible else None),
                selection_pass=bool(eligible),
                selection_evaluations=dict(fixed=720, dynamic=80,
                                           paired_joint_decisions=25),
                evaluation_equivalent_total_mults=int(total_eval_mults),
                summed_logged_attempt_wall_seconds=total_wall,
                finished_attempt_count=finished_attempts,
                unfinished_attempt_count=incomplete_attempts,
                wall_time_complete=bool(incomplete_attempts == 0),
                interpretation="Failed combinations and complete 800-evaluation search retained; "
                "logged attempts include input/IO/retries; unfinished attempts leave a wall-time lower bound. "
                "Offline search total is separate from per-deployment online cost")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--unlock-select", action="store_true")
    args = parser.parse_args()
    if not args.unlock_select:
        raise RuntimeError("Selection analysis remains locked")
    frozen = check_freeze(args.freeze)
    outdir = prepare_output(frozen)
    records = _load_all(outdir, frozen.digest, frozen.document["dynamic_config"])
    decision = decide_selection(records)
    decision["frozen_grid_sha256"] = frozen.digest
    path = outdir / "selection_analysis.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != decision:
        raise RuntimeError("Existing selection analysis differs")
    if not path.exists():
        atomic_json(path, decision)
    print(json.dumps(dict(selection_pass=decision["selection_pass"],
                          r4_floor_count=decision["r4_floor_count"],
                          selected_dynamic=decision["selected_dynamic"]),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
