"""Only tiny N14 development/smoke checks; no select or confirm arrays."""
from __future__ import annotations

import sys
import math
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments_v10"))
import anc_core as ac  # noqa: E402
from generator import (CASE_IDS, COMPONENT_IDS, PHASE_IDS, PRIMARY_FIR,
                       SECONDARY_FIR, _dev_rng, causal_fir, e1_teacher_primary,
                       e2e3_primary, make_case, physical_error, rff_features,
                       seed_key, static_primary, validate_case)  # noqa: E402


class GeneratorTests(unittest.TestCase):
    def test_reserved_streams_are_disjoint_without_materializing_them(self):
        keys = [seed_key(phase, case, run, component)
                for phase in PHASE_IDS for case in CASE_IDS for run in (0, 1)
                for component in COMPONENT_IDS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(key[1] == 14 for key in keys))
        self.assertNotEqual(seed_key("dev", "E2", 0, "measurement_noise"),
                            seed_key("dev", "E2", 0, "structure"))
        # Only SeedSequence fingerprints are sampled here, never case arrays.
        phases = [np.random.SeedSequence(seed_key(phase, "E2", 0, "reference"))
                  .generate_state(16).tobytes() for phase in PHASE_IDS]
        self.assertEqual(len(set(phases)), 3)

    def test_e2_e3_actual_arrays_and_causal_primary(self):
        for case in ("E2", "E3"):
            data = make_case(case, 0, smoke_segment=40)
            self.assertTrue(data.smoke_only)
            self.assertEqual(data.x.shape, (160,))
            self.assertEqual(data.omega.shape, (500, 20))
            self.assertEqual(data.initial_A.shape, (25, 8))
            self.assertEqual(data.segment_bounds,
                             ((0, 40), (40, 80), (80, 120), (120, 160)))
            g = ac.fir(PRIMARY_FIR, data.x)
            a2 = np.repeat([.02, .08, .16, .02], 40)
            a3 = np.repeat([.01, .04, .08, .01], 40)
            expected = np.zeros(160)
            expected[2:] = g[:-2] + a2[2:]*g[:-2]**2 - a3[2:]*g[1:-1]**3
            np.testing.assert_array_equal(data.d, expected)
            for n in range(160):
                if n < 2:
                    self.assertEqual(data.d[n], 0.)
                else:
                    stage = n // 40
                    manual = (g[n-2] + (.02, .08, .16, .02)[stage]*g[n-2]**2
                              - (.01, .04, .08, .01)[stage]*g[n-1]**3)
                    self.assertAlmostEqual(data.d[n], manual, delta=1e-14)
            noise_rng = np.random.default_rng(np.random.SeedSequence(
                seed_key("dev", case, 0, "measurement_noise")))
            noise = (.01*noise_rng.standard_normal(160) if case == "E2"
                     else .05*ac.sas_cms(noise_rng, 1.6, 160))
            np.testing.assert_array_equal(data.v, noise)
            changed = data.x.copy()
            changed[73:] += 10.
            np.testing.assert_array_equal(e2e3_primary(changed, 40)[:73], data.d[:73])
            self.assertTrue(np.isfinite(data.v).all())

    def test_prefix_and_component_independence(self):
        short = make_case("E2", 1, smoke_segment=30)
        longer = make_case("E2", 1, smoke_segment=45)
        np.testing.assert_array_equal(short.x, longer.x[:len(short.x)])
        np.testing.assert_array_equal(short.v, longer.v[:len(short.v)])
        np.testing.assert_array_equal(short.omega, longer.omega)
        np.testing.assert_array_equal(short.rff_phase, longer.rff_phase)
        np.testing.assert_array_equal(short.initial_A, longer.initial_A)
        np.testing.assert_array_equal(short.initial_B, longer.initial_B)
        np.testing.assert_array_equal(short.d[:30], longer.d[:30])
        self.assertFalse(np.array_equal(short.x, make_case("E2", 2, smoke_segment=30).x))

    def test_e1_teacher_rank_and_physical_fir(self):
        data = make_case("E1", 0, smoke_segment=30)
        self.assertEqual(data.segment_bounds, ((0, 30), (30, 60), (60, 90)))
        self.assertEqual(np.linalg.matrix_rank(data.teacher_low), 1)
        self.assertEqual(np.linalg.matrix_rank(data.teacher_high), 4)
        z = rff_features(data.x, data.omega, data.rff_phase, 0, len(data.x))
        history = np.zeros(20)
        z_stream = np.zeros_like(z)
        for n, sample in enumerate(data.x):
            history[1:] = history[:-1].copy()
            history[0] = sample
            z_stream[n] = np.sqrt(2/500)*np.cos(data.omega @ history + data.rff_phase)
        np.testing.assert_allclose(z, z_stream, atol=1e-14)
        low = data.teacher_low.T.reshape(500)
        high = data.teacher_high.T.reshape(500)
        raw = np.concatenate((z[:30] @ low, z[30:60] @ high, z[60:] @ low))
        np.testing.assert_allclose(data.d, ac.fir(SECONDARY_FIR, raw), atol=1e-14)
        changed = data.x.copy()
        changed[73:] -= 3.
        d2, _, _ = e1_teacher_primary(
            changed, data.omega, data.rff_phase,
            _dev_rng("E1", 0, "teacher"), 30, block_size=8)
        np.testing.assert_allclose(data.d[:73], d2[:73], atol=1e-14)

    def test_c_bridge_and_physical_impulse(self):
        for case in ("C1", "C2", "C3", "C4"):
            data = make_case(case, 0, smoke_length=1200)
            self.assertEqual(data.segment_bounds, ((0, 1200),))
            np.testing.assert_array_equal(data.d, static_primary(data.x))
            np.testing.assert_array_equal(data.d, ac.primary_disturbance(data.x))
            self.assertTrue(np.isfinite(data.x).all())
            self.assertTrue(np.isfinite(data.v).all())
            if case in ("C1", "C2"):
                self.assertTrue(np.all(data.v == 0))
                same_rng = np.random.default_rng(np.random.SeedSequence(
                    seed_key("dev", case, 0, "reference")))
                reference = (ac.logistic_delay6(same_rng, 1, 1200)[0] if case == "C1"
                             else ac.alpha_stable_reference(same_rng, 1, 1200)[0])
                np.testing.assert_allclose(data.x, reference, atol=1e-14)
        impulse = np.zeros(10)
        impulse[0] = 1.
        np.testing.assert_array_equal(physical_error(np.zeros(10), np.zeros(10), impulse),
                                      -ac.fir(SECONDARY_FIR, impulse))
        np.testing.assert_array_equal(causal_fir(impulse, PRIMARY_FIR),
                                      ac.fir(PRIMARY_FIR, impulse))

    def test_select_and_confirm_are_locked(self):
        for phase in ("select", "confirm"):
            with self.assertRaises(ValueError):
                make_case("E2", 0, phase=phase, smoke_segment=20)
        with self.assertRaises(ValueError):
            make_case("E2", 0, smoke_segment=10)

    def test_bundle_validation_and_hashes(self):
        for case in ("E1", "E2", "E3"):
            data = make_case(case, 0, smoke_segment=30)
            validated = validate_case(data, require_full=False)
            self.assertEqual(validated["primary_max_abs_gap"], 0.)
            self.assertEqual(len(validated["array_sha256"]["omega"]), 64)
            with self.assertRaises(ValueError):
                validate_case(data)

    def test_round_three_binomial_cutoff(self):
        alpha = .05/(3*4)
        tail_16 = sum(math.comb(20, k) for k in range(16, 21))/2**20
        tail_17 = sum(math.comb(20, k) for k in range(17, 21))/2**20
        self.assertEqual(tail_17, 1351/1048576)
        self.assertGreater(tail_16, alpha)
        self.assertLessEqual(tail_17, alpha)


if __name__ == "__main__":
    unittest.main()
