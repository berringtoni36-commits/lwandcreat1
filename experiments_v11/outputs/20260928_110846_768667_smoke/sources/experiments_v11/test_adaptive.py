"""Focused correctness checks for the new selector; no historical files written."""
from copy import deepcopy
import unittest

import numpy as np
from adaptive_core import ac, AdaptiveKron, SelectionConfig, resized_copy, mcc_loss


def initial(R=2):
    return ac.KronRFF("base", 1, 5, 4, R, .1, .1, 2., 1e-8,
                      np.random.default_rng(8))


class AdaptiveTests(unittest.TestCase):
    def test_disabled_matches_original_bitwise(self):
        original = initial()
        selector = AdaptiveKron(original, np.random.default_rng(1), SelectionConfig(enabled=False))
        rng = np.random.default_rng(7)
        for _ in range(200):
            z, q, dv = rng.normal(size=(1, 20)), rng.normal(size=(1, 20)), rng.normal(size=1)
            np.testing.assert_array_equal(original.step(z, q, dv), selector.step(z, q, dv))
        np.testing.assert_array_equal(original.A, selector.active.A)
        np.testing.assert_array_equal(original.B, selector.active.B)
        self.assertEqual(selector.events, [])

    def test_growth_preserves_output_and_new_term_can_learn(self):
        base = initial()
        candidate = resized_copy(base, 1, np.random.default_rng(2))
        np.testing.assert_array_equal(base.B @ base.A.transpose(0, 2, 1),
                                      candidate.B @ candidate.A.transpose(0, 2, 1))
        candidate.step(np.ones((1, 20)), np.ones((1, 20)), np.array([1.]))
        self.assertGreater(np.linalg.norm(candidate.B[:, :, -1]), 0)
        self.assertFalse(np.shares_memory(candidate.A, base.A))

    def test_pruning_removes_exactly_one_term(self):
        base = initial(3)
        candidate = resized_copy(base, -1, np.random.default_rng(2), 1)
        expected = base.B @ base.A.transpose(0, 2, 1) - base.B[:, :, 1:2] @ base.A[:, :, 1:2].transpose(0, 2, 1)
        np.testing.assert_allclose(candidate.B @ candidate.A.transpose(0, 2, 1), expected, atol=1e-18)
        self.assertEqual(candidate.R, 2)
        np.testing.assert_array_equal(candidate.ybuf, base.ybuf)

    def test_growth_scale_matches_learned_factors(self):
        base = initial()
        base.A *= 500
        candidate = resized_copy(base, 1, np.random.default_rng(2))
        target = np.sqrt(np.sum(base.A**2)/base.R)
        self.assertAlmostEqual(np.linalg.norm(candidate.A[:,:,-1])/target, 1., places=7)

    def test_crossfade_filters_actual_output_history(self):
        cfg = SelectionConfig(enabled=False, ramp_samples=7)
        selector = AdaptiveKron(initial(), np.random.default_rng(1), cfg)
        selector.candidate = resized_copy(selector.active, 1, np.random.default_rng(2))
        selector.candidate.B[:, :, -1] = 0.8
        selector.phase = "ramp"
        old, new = deepcopy(selector.active), deepcopy(selector.candidate)
        history, ys, wrong = [], [], []
        rng = np.random.default_rng(6)
        for n in range(15):
            z, q, dv = rng.normal(size=(1, 20)), rng.normal(size=(1, 20)), rng.normal(size=1)
            so, sn = old.step(z, q, dv), new.step(z, q, dv)
            gamma = min((n + 1) / 7, 1.)
            history.append((1-gamma)*old.ybuf[0, 0] + gamma*new.ybuf[0, 0])
            ys.append(selector.step(z, q, dv)[0])
            wrong.append((1-gamma)*so[0] + gamma*sn[0])
        np.testing.assert_allclose(ys, ac.fir(ac.S_PATH, np.array(history)), atol=1e-14)
        self.assertGreater(np.max(np.abs(np.array(ys)-wrong)), 1e-5)
        self.assertEqual(selector.active.R, 3)

    def test_validation_uses_physical_preupdate_error(self):
        selector = AdaptiveKron(initial(), np.random.default_rng(1), SelectionConfig(enabled=False))
        selector.candidate = resized_copy(selector.active, 1, np.random.default_rng(2))
        selector.phase = "validate"
        selector.active.ybuf[0] = [3., 4., 5., 6.]
        selector.candidate.ybuf[0] = [7., 8., 9., 10.]
        # After shift: s*y = 4+0.5*5 and 8+0.5*9; independent of q prediction.
        selector.step(np.zeros((1, 20)), np.zeros((1, 20)), np.array([1.]))
        np.testing.assert_allclose(selector.validation_sum, mcc_loss(np.array([1-6.5, 1-12.5]), 2.))

    def test_decision_penalty_and_hysteresis(self):
        cfg = SelectionConfig(lambda_cost=.01, margin=.001)
        for losses, accepted in [([.1, .05], True), ([.1, .1], False)]:
            s = AdaptiveKron(initial(), np.random.default_rng(1), cfg)
            s.candidate = resized_copy(s.active, 1, np.random.default_rng(2))
            s.validation_sum = np.array(losses)*cfg.validation_samples
            s._decision()
            self.assertEqual(s.events[-1]['event'] == 'accept', accepted)

    def test_cost_counts_candidate_without_duplicating_features(self):
        s = AdaptiveKron(initial(), np.random.default_rng(1), SelectionConfig(enabled=False))
        s.candidate = resized_copy(s.active, 1, np.random.default_rng(2))
        s.phase = "train"
        s.step(np.ones((1,20)), np.ones((1,20)), np.zeros(1))
        self.assertEqual(s.last['candidate_mults'], s.cost(3)-s.shared_cost)
        self.assertEqual(s.last['total_mults'], s.cost(2)+s.cost(3)-s.shared_cost+3*len(ac.S_PATH))
        self.assertEqual(s.last['resident_factor_coeffs'], 5*(5+4))

    def test_causal_prefix_and_rank_bounds(self):
        cfg = SelectionConfig(r_min=1, r_max=3, warmup=5, train_samples=4,
                              validation_samples=4, ramp_samples=2, cooldown=4, screen_every=1)
        rng = np.random.default_rng(2)
        inputs = [(rng.normal(size=(1,20)), rng.normal(size=(1,20)), rng.normal(size=1)) for _ in range(80)]
        a = AdaptiveKron(initial(), np.random.default_rng(1), cfg)
        b = AdaptiveKron(initial(), np.random.default_rng(1), cfg)
        for z,q,dv in inputs:
            np.testing.assert_array_equal(a.step(z,q,dv), b.step(z,q,dv))
            self.assertTrue(1 <= a.active.R <= 3)
            if a.candidate:
                self.assertTrue(1 <= a.candidate.R <= 3)
        self.assertEqual(a.events, b.events)
        self.assertTrue(any(e['event'] in ('accept','reject') for e in a.events))

    def test_validation_is_after_training_and_has_exact_window(self):
        cfg = SelectionConfig(warmup=5, train_samples=4, validation_samples=6,
                              cooldown=4, screen_every=1)
        s = AdaptiveKron(initial(),np.random.default_rng(1),cfg)
        for _ in range(15):
            s.step(np.ones((1,20)),np.ones((1,20)),np.array([.5]))
        self.assertEqual([e['sample'] for e in s.events[:3]], [4,8,14])
        self.assertEqual(s.events[-1]['validation_samples'],6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
