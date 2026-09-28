"""Confirmation harness tests using dev smoke only; no confirm RNG samples."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import confirmation_runner as cr
import confirmation_analysis as ca
import selection_runner as sr


def dev_smoke(case: str):
    kwargs = ({"smoke_segment": 20} if case in ("E1", "E2", "E3")
              else {"smoke_length": 80})
    item = cr.gen.make_case(case, 0, phase="dev", **kwargs)
    names = ("x", "d", "v", "omega", "rff_phase", "initial_A", "initial_B",
             "segment_bounds")
    data = {name: np.asarray(getattr(item, name)) for name in names}
    if case == "E1":
        data["teacher_low"] = item.teacher_low
        data["teacher_high"] = item.teacher_high
    return data


class ConfirmationHarnessTests(unittest.TestCase):
    def test_no_confirmation_data_without_completed_selection_and_freeze(self):
        with self.assertRaisesRegex(RuntimeError, "freeze_confirmation"):
            cr.check_confirmation_ready(None, None)
        with self.assertRaisesRegex(RuntimeError, "freeze"):
            cr.make_confirm_input("E2", 0, None)
        self.assertNotEqual(cr.gen.seed_key("confirm", "E2", 0, "reference"),
                            cr.gen.seed_key("select", "E2", 0, "reference"))
        with tempfile.TemporaryDirectory() as temp:
            draft = Path(temp) / "freeze_confirmation.json"
            draft.write_text(json.dumps({"status": "draft_only"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Confirmation freeze mismatch"):
                cr.check_confirmation_ready(draft, draft)

    def test_dev_smoke_all_stage_shapes_and_resume(self):
        model = sr.load_model()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            e2 = dev_smoke("E2")
            settings = dict(case="E2", run=0, phase="smoke", frozen_sha="SMOKE_ONLY",
                            fixed_mu={name: .05 for name in sr.FIXED_METHODS},
                            dynamic_mu=.05, dynamic_config=dict(sr.CONFIG), tail=10)
            folder = root / "dev_E2_smoke"
            for stage in ("input", "fixed", "dynamic"):
                with self.assertRaisesRegex(RuntimeError, "Injected failure"):
                    cr.execute_case(e2, model, folder, fail_after=stage, **settings)
            result = cr.execute_case(e2, model, folder, **settings)
            self.assertEqual(result["phase"], "smoke")
            self.assertEqual(result["fixed_evaluations"], 9)
            self.assertEqual(result["dynamic_evaluations"], 1)
            self.assertEqual(cr.execute_case(e2, model, folder, **settings), result)
            with np.load(folder / "fixed_selected.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["segment_anr_db"].shape, (9, 4))
                replay = np.stack([
                    cr._anr_from_error(e2["d"], row, e2["segment_bounds"], 10)
                    for row in saved["physical_error"]])
                self.assertLess(float(np.max(np.abs(
                    replay-saved["segment_anr_db"]))), 1e-4)
            fixed_meta = json.loads((folder / "fixed_result.json").read_text(
                encoding="utf-8"))
            self.assertEqual(fixed_meta["cost_ledger"], sr.fixed_costs())
            e1 = dev_smoke("E1")
            e1_folder = root / "dev_E1_smoke"
            cr.execute_case(e1, model, e1_folder, case="E1", run=0,
                            phase="smoke", frozen_sha="SMOKE_ONLY",
                            fixed_mu={name: .2 for name in sr.FIXED_METHODS},
                            dynamic_mu=.2, dynamic_config=dict(sr.CONFIG), tail=10)
            with np.load(e1_folder / "fixed_selected.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["segment_anr_db"].shape, (9, 3))
            c1 = dev_smoke("C1")
            c_folder = root / "dev_C1_smoke"
            closed = dict(sr.CONFIG, arm="fixed", resort_interval=0)
            cr.execute_case(c1, model, c_folder, case="C1", run=0,
                            phase="smoke", frozen_sha="SMOKE_ONLY",
                            fixed_mu={name: .2 for name in sr.FIXED_METHODS},
                            dynamic_mu=.2, dynamic_config=closed, tail=10)
            with np.load(c_folder / "fixed_selected.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["segment_anr_db"].shape, (9, 1))
            with np.load(c_folder / "dynamic.npz", allow_pickle=False) as saved:
                gap = ca.original_v10_bridge_gap(c1, saved["drive"],
                                                saved["error"], .2)
            self.assertLess(max(gap.values()), 1e-10)

    def test_single_exact_paired_statistic_and_failure_retention(self):
        def row(case, run, *, pass_it=True):
            fixed = np.full((9, 4), -9., float).tolist()
            dynamic = [-9.]*4 if pass_it else [-8.]*4
            return dict(case=case, run=run, valid=True, fixed_anr_db=fixed,
                        dynamic_anr_db=dynamic, dynamic_mean_mults=16000.,
                        accepted_initial_prune=True, retained_r2=True,
                        prune_finish_sample=24100)
        primary = {case: [row(case, i, pass_it=i < 17) for i in range(20)]
                   for case in ("E2", "E3")}
        bridge = {case: [dict(case=case, run=i, valid=True) for i in range(20)]
                  for case in cr.BRIDGE}
        result = ca.summarize(primary, bridge, freeze_sha="SMOKE_ONLY")
        self.assertEqual(result["K"], 17)
        self.assertAlmostEqual(result["exact_one_sided_p"], 1351/1048576)
        self.assertTrue(result["primary_statistical_pass"])
        self.assertGreater(result["exact_one_sided_success_probability_lower_bound"], .5)
        primary["E2"][16] = row("E2", 16, pass_it=False)
        failed = ca.summarize(primary, bridge, freeze_sha="SMOKE_ONLY")
        self.assertEqual(failed["K"], 16)
        self.assertAlmostEqual(failed["exact_one_sided_p"], 6196/1048576)
        self.assertFalse(failed["primary_statistical_pass"])
        primary["E3"][0] = dict(case="E3", run=0, valid=False,
                                failure="algorithmic_nonfinite")
        retained = ca.summarize(primary, bridge, freeze_sha="SMOKE_ONLY")
        self.assertEqual(retained["paired_runs"][0]["J"], 0)
        self.assertIn("algorithmic_nonfinite",
                      retained["paired_runs"][0]["failures"]["E3"])


if __name__ == "__main__":
    unittest.main()
