"""Short, development-only numerical tests for all 45 N14 fixed branches."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments_v10"))
sys.path.insert(0, str(ROOT / "experiments_v14" / "protocol"))

import anc_core as ac  # noqa: E402
from generator import make_case  # noqa: E402
from kernel import simulate_fixed  # noqa: E402
from run_frontier import MU, costs, method_names, physical_fir_replay  # noqa: E402
from selection_interface import choose_fixed_mu, fixed_branch  # noqa: E402


class FrontierTests(unittest.TestCase):
    def test_all_45_trajectories_match_v10_ordering_and_physics(self):
        data = make_case("E2", 0, phase="dev", smoke_segment=20)
        x, d, v = data.x, data.d, data.v
        a0, b0 = data.initial_A, data.initial_B
        ends = np.array([20, 40, 60, 80], np.int64)
        sums, counts, y, error = simulate_fixed(
            x, d, v, data.omega, data.rff_phase, a0, b0, MU, ends, 10)
        np.testing.assert_array_equal(counts, np.full(4, 10))
        controls = []
        for rank in range(1, 9):
            for mu in MU:
                c = ac.KronRFF("fixed", 1, 25, 20, rank, mu / 2, mu / 2,
                               2., 1e-8, np.random.default_rng(0))
                c.A = a0[:, :rank][None, :, :].copy()
                c.B = b0[:, :rank][None, :, :].copy()
                controls.append(c)
        controls += [ac.FullRFF("full", 1, 500, mu, 2., 1e-8) for mu in MU]
        xb = np.zeros((1, 20))
        zh = np.zeros((4, 1, 500))
        ad = 0.
        ae = np.zeros(45)
        ref_anr = np.zeros((45, 4))
        for n in range(80):
            xb[:, 1:] = xb[:, :-1].copy()
            xb[0, 0] = x[n]
            z = np.sqrt(2 / 500) * np.cos(data.omega[None] @ xb[0] + data.rff_phase[None])
            zh[1:] = zh[:-1].copy()
            zh[0] = z
            q = zh[2] + .5 * zh[3]
            dv = np.array([d[n] + v[n]])
            ad = .999 * ad + .001 * abs(d[n])
            stage = n // 20
            for j, c in enumerate(controls):
                ys = c.step(z, q, dv)[0]
                self.assertAlmostEqual(float(y[j, n]), float(c.ybuf[0, 0]), delta=2e-6)
                physical = dv[0] - ys
                self.assertAlmostEqual(float(error[j, n]), float(physical), delta=2e-6)
                ae[j] = .999 * ae[j] + .001 * abs(physical)
                if n >= 10 + 20 * stage:
                    ref_anr[j, stage] += 20 * np.log10((ae[j] + 1e-12) / (ad + 1e-12))
        np.testing.assert_allclose(sums / counts, ref_anr / 10, atol=1e-7, rtol=0)
        replay = physical_fir_replay({"d": d, "v": v}, y, error)
        self.assertLessEqual(replay["max_rounding_bound_ratio"], 1.001)

    def test_accounting_and_selection_boundary(self):
        self.assertEqual(len(method_names()), 45)
        self.assertEqual(costs()["R4"]["total_per_sample"], 19132)
        self.assertEqual(costs()["full500"]["total_per_sample"], 14508)
        dev = [dict(phase="dev", case="E2", run=i,
                    segment_anr_db=np.zeros((45, 4)).tolist()) for i in range(8)]
        with self.assertRaises(ValueError):
            choose_fixed_mu(dev)
        for item in dev:
            item["phase"] = "select"
        dev[0]["segment_anr_db"][16] = [-10.] * 4
        chosen = choose_fixed_mu(dev)
        self.assertEqual(chosen["fixed_methods"]["R4"]["mu"], .1)
        self.assertEqual(fixed_branch(dev[0], "R4", .1).tolist(), [-10.] * 4)


if __name__ == "__main__":
    unittest.main()
