"""N14 selection harness tests. No select/confirm random streams are sampled."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from selection_runner import (CONFIG, MU, ROOT, check_freeze, execute_bundle,
                              load_model, model_config_keys, MODEL_PATH, sha)
from selection_analysis import decide_selection
from selection_runner import gen
from check_freeze import draft_template


def smoke_data():
    case = gen.make_case("E2", 0, phase="dev", smoke_segment=20)
    return {key: np.asarray(getattr(case, key)) for key in
            ("x", "d", "v", "omega", "rff_phase", "initial_A", "initial_B",
             "segment_bounds")}


class SelectionHarnessTests(unittest.TestCase):
    def test_periodic_resort_on_existing_dev_prefix_only(self):
        """The frozen 50k setting really proposes and validates R2→R2."""
        source = (ROOT / "experiments_v14" / "outputs" / "n14_dev_inputs"
                  / "caseE2_run00.npz")
        with np.load(source, allow_pickle=False) as f:
            t = 78_110
            x, d, v = (f[key][:t].copy() for key in ("x", "d", "v"))
            omega = f["omega"][None].copy()
            phase = f["rff_phase"][None].copy()
            a0 = f["initial_A"][None].copy()
            b0 = f["initial_B"][None].copy()
        result = load_model().simulate(x, d, v, omega, phase, a0, b0,
                                       .05, dict(CONFIG))
        events = [(e["n"], e["type"], e["from_R"], e["to_R"],
                   e.get("accepted")) for e in result["events"]]
        self.assertEqual(events, [(20_000, "proposal_prune", 4, 2, None),
                                  (24_000, "prune", 4, 2, True),
                                  (74_000, "proposal_prune", 2, 2, None),
                                  (78_000, "prune", 2, 2, True)])
        self.assertGreater(result["cost"]["reorder"][74_000], 0)
        self.assertTrue(np.all(result["R"][24_101:] == 2))
        self.assertLess(result["physical_fir_max_abs"], 1e-10)

    def test_unfrozen_gate_fails_without_outputs_or_select_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            place = Path(temp)
            with self.assertRaises(RuntimeError):
                check_freeze(None)
            candidate = place / "freeze_candidate_grid.json"
            candidate.write_text(json.dumps({"status": "frozen_for_select"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                check_freeze(candidate)
            plausible = draft_template()
            plausible["status"] = "frozen_for_select"
            plausible["freeze_authorization"] = "parent_reviewed_after_dev"
            candidate.write_text(json.dumps(plausible), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "development audit path absent"):
                check_freeze(candidate)
            self.assertEqual(sorted(p.name for p in place.iterdir()),
                             ["freeze_candidate_grid.json"])

    def test_dev_smoke_and_resume_every_durable_stage(self):
        self.assertEqual(MODEL_PATH.name, "selector_resort.py")
        self.assertEqual(CONFIG["resort_interval"], 50_000)
        self.assertEqual(model_config_keys(), set(CONFIG))
        model = load_model()
        data = smoke_data()
        with tempfile.TemporaryDirectory() as temp:
            place = Path(temp) / "smoke_dev_only"
            common = dict(case="E2", run=0, phase="smoke", frozen_sha="SMOKE_ONLY",
                          config=CONFIG, tail=10)
            with self.assertRaisesRegex(RuntimeError, "Injected failure"):
                execute_bundle(data, model, place, fail_after="input", **common)
            initial_input_sha = sha(place / "input.npz")
            with self.assertRaisesRegex(RuntimeError, "Injected failure"):
                execute_bundle(data, model, place, fail_after="fixed", **common)
            fixed_sha = sha(place / "fixed_traces.npz")
            with self.assertRaisesRegex(RuntimeError, "Injected failure"):
                execute_bundle(data, model, place, fail_after="dynamic_0.05", **common)
            first_dynamic_sha = sha(place / "dynamic_mu0.05.npz")
            result = execute_bundle(data, model, place, **common)
            self.assertEqual(result["phase"], "smoke")
            self.assertEqual(result["fixed_evaluations"], 45)
            self.assertEqual(result["dynamic_evaluations"], 5)
            self.assertEqual(sha(place / "input.npz"), initial_input_sha)
            self.assertEqual(sha(place / "fixed_traces.npz"), fixed_sha)
            self.assertEqual(sha(place / "dynamic_mu0.05.npz"), first_dynamic_sha)
            self.assertEqual(execute_bundle(data, model, place, **common), result)
            self.assertTrue(all(json.loads((place / f"dynamic_mu{mu:.2f}.json")
                                           .read_text(encoding="utf-8"))["phase"] == "smoke"
                                for mu in MU))
            # Also recover a trace whose metadata commit was interrupted.
            (place / "dynamic_mu0.80.json").unlink()
            (place / "result.json").unlink()
            restored = execute_bundle(data, model, place, **common)
            self.assertEqual(restored["dynamic_traces_sha256"]["0.80"],
                             sha(place / "dynamic_mu0.80.npz"))
            # Opposite interruption: metadata survived but trace file did not.
            old_sha = sha(place / "dynamic_mu0.40.npz")
            (place / "dynamic_mu0.40.npz").unlink()
            (place / "result.json").unlink()
            restored = execute_bundle(data, model, place, **common)
            self.assertEqual(sha(place / "dynamic_mu0.40.npz"), old_sha)
            self.assertEqual(restored["dynamic_traces_sha256"]["0.40"], old_sha)

    def test_paired_selection_rule_is_25_combinations_not_dev_oracle(self):
        records = {}
        for case in ("E2", "E3"):
            rows = []
            for run in range(8):
                fixed = np.full((45, 4), -8., float)
                fixed[15] = -9.  # Fair R4 winner is μ=.05 on 8×4 mean.
                dynamic = {f"{mu:.2f}": dict(segment_anr_db=[-9.]*4,
                                             mean_mults=16000.,
                                             total_multiplications=6_400_000_000,
                                             accepted_R4_to_R2_by_80000=True,
                                             physical_integrity_pass=True)
                           for mu in MU}
                rows.append(dict(phase="select", case=case, run=run,
                                 segment_anr_db=fixed.tolist(), dynamic=dynamic,
                                 run_wall_seconds=1.))
            records[case] = rows
        decision = decide_selection(records)
        self.assertEqual(len(decision["dynamic_combinations"]), 25)
        self.assertEqual(decision["r4_floor_count"], 8)
        self.assertTrue(decision["selection_pass"])
        self.assertEqual(decision["selected_dynamic"]["mu_E2"], .05)
        self.assertEqual(decision["selected_dynamic"]["mu_E3"], .05)
        self.assertEqual(decision["selection_evaluations"],
                         dict(fixed=720, dynamic=80, paired_joint_decisions=25))
        for case in records:
            for row in records[case]:
                row["dynamic"]["0.05"]["accepted_R4_to_R2_by_80000"] = False
        changed = decide_selection(records)
        self.assertFalse(next(item for item in changed["dynamic_combinations"]
                              if item["mu_E2"] == item["mu_E3"] == .05)["eligible"])


if __name__ == "__main__":
    unittest.main()
