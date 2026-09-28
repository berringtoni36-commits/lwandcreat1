"""Short development-only equivalence checks for the compiled fixed frontier."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments_v10"))
sys.path.insert(0, str(ROOT / "experiments_v13" / "protocol"))

import anc_core as ac  # noqa: E402
from generator import make_p13  # noqa: E402
from run_frontier import MU, assess_capacity, initial_factors, simulate_fixed  # noqa: E402


class FrontierTests(unittest.TestCase):
    def test_every_fixed_trajectory_matches_v10(self):
        tmax = 60
        data = make_p13("dev", 0, length=tmax, switches=(20, 40))
        a0, b0 = initial_factors(0)
        sums, counts, y, error = simulate_fixed(
            data.x, data.d, data.v, data.omega, data.rff_phase,
            a0, b0, MU, np.array([20, 40, tmax], np.int64), 10)
        self.assertTrue(np.array_equal(counts, [10, 10, 10]))
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
        ref_anr = np.zeros((45, 3))
        for n in range(tmax):
            xb[:, 1:] = xb[:, :-1].copy()
            xb[0, 0] = data.x[n]
            z = np.sqrt(2 / 500) * np.cos(data.omega[None] @ xb[0] + data.rff_phase[None])
            zh[1:] = zh[:-1].copy()
            zh[0] = z
            q = zh[2] + .5 * zh[3]
            dv = np.array([data.d[n] + data.v[n]])
            ad = .999 * ad + .001 * abs(data.d[n])
            stage = min(n // 20, 2)
            for j, c in enumerate(controls):
                ys = c.step(z, q, dv)[0]
                self.assertAlmostEqual(float(y[j, n]), float(c.ybuf[0, 0]), delta=2e-6)
                physical = dv[0] - ys
                self.assertAlmostEqual(float(error[j, n]), float(physical), delta=2e-6)
                ae[j] = .999 * ae[j] + .001 * abs(physical)
                if n >= 10 + 20 * stage:
                    ref_anr[j, stage] += 20 * np.log10((ae[j] + 1e-12) / (ad + 1e-12))
        np.testing.assert_allclose(sums / counts, ref_anr / 10, atol=1e-7, rtol=0)

    def test_capacity_rule(self):
        anr = np.zeros((45, 3))
        for rank in range(1, 9):
            anr[(rank - 1) * 5:rank * 5] = [-10.0, -8.0 - rank, -10.0]
        # The middle gets within 0.5 dB of its best only at R8.
        self.assertEqual(assess_capacity(anr)["required_rank_by_stage"], [1, 8, 1])
        self.assertFalse(assess_capacity(anr)["capacity_gate_pass"])


if __name__ == "__main__":
    unittest.main()
