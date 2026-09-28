"""Focused correctness checks for the new selector; no historical files written."""
from copy import deepcopy
import unittest

import numpy as np
from adaptive_core import (ac, AdaptiveKron, SelectionConfig, resized_copy,
                           mcc_loss, AdaptiveKronV12, V12Config, best_rank_prune)


def initial(R=2):
    return ac.KronRFF("base", 1, 5, 4, R, .1, .1, 2., 1e-8,
                      np.random.default_rng(8))


class AdaptiveTests(unittest.TestCase):
    def test_svd_prune_is_best_matrix_rank_approximation(self):
        base = initial(4)
        original = base.B[0] @ base.A[0].T
        candidate, singular = best_rank_prune(base)
        expected_error = singular[-1]
        actual_error = np.linalg.norm(original - candidate.B[0] @ candidate.A[0].T)
        self.assertEqual(candidate.R, 3)
        self.assertAlmostEqual(actual_error, expected_error, places=12)
        np.testing.assert_array_equal(base.B[0] @ base.A[0].T, original)
        np.testing.assert_array_equal(candidate.ybuf, base.ybuf)

    def test_svd_prune_handles_rank_deficient_factors(self):
        base = initial(3)
        base.A[:, :, 2] = base.A[:, :, 0]
        base.B[:, :, 2] = base.B[:, :, 0]
        candidate, singular = best_rank_prune(base)
        self.assertLess(singular[-1], 1e-14)
        np.testing.assert_allclose(candidate.B @ candidate.A.transpose(0, 2, 1),
                                   base.B @ base.A.transpose(0, 2, 1), atol=1e-14)

    def test_v12_backoff_is_direction_specific(self):
        cfg = V12Config(r_max=4, warmup=4, train_samples=4,
                        validation_samples=4, ramp_samples=2, cooldown=4, max_backoff=16,
                        screen_every=1)
        s = AdaptiveKronV12(initial(), np.random.default_rng(1), cfg)
        for _ in range(70):
            s.step(np.zeros((1, 20)), np.zeros((1, 20)), np.zeros(1))
        rejects = [e for e in s.events if e['event'] == 'reject']
        self.assertGreaterEqual(len(rejects), 2)
        self.assertEqual(rejects[0]['rejection_streak'], 1)
        self.assertTrue(all(e['candidate_lifetime'] == 8 for e in rejects))
        self.assertTrue(all(e['cost_penalty_delta'] != 0 for e in rejects))
        self.assertTrue(any(e['decomposition_charge'] == 1_000_000 for e in s.events
                            if e['event'] == 'proposal' and e['new_R'] < e['old_R']))

    def test_v12_disabled_matches_fixed_controller(self):
        base=initial()
        s=AdaptiveKronV12(base,np.random.default_rng(10),V12Config(enabled=False))
        rng=np.random.default_rng(21)
        for _ in range(100):
            z=rng.standard_normal((1,20))
            q=rng.standard_normal((1,20))
            dv=rng.standard_normal(1)
            np.testing.assert_array_equal(base.step(z,q,dv),s.step(z,q,dv))
        np.testing.assert_array_equal(base.A,s.active.A)
        np.testing.assert_array_equal(base.B,s.active.B)

    def test_v12_future_suffix_does_not_change_past_decisions(self):
        cfg=V12Config(warmup=4,train_samples=4,validation_samples=4,
                      ramp_samples=2,cooldown=4,max_backoff=16,screen_every=1)
        a=AdaptiveKronV12(initial(),np.random.default_rng(42),cfg)
        b=AdaptiveKronV12(initial(),np.random.default_rng(42),cfg)
        rng=np.random.default_rng(6)
        prefix=[(rng.standard_normal((1,20)),rng.standard_normal((1,20)),
                 rng.standard_normal(1)) for _ in range(50)]
        for z,q,dv in prefix:
            np.testing.assert_array_equal(a.step(z,q,dv),b.step(z,q,dv))
        self.assertEqual(a.events,b.events)
        for _ in range(20):
            a.step(np.zeros((1,20)),np.zeros((1,20)),np.zeros(1))
            b.step(np.ones((1,20)),np.ones((1,20)),np.ones(1))
        self.assertEqual([e for e in a.events if e['sample']<50],
                         [e for e in b.events if e['sample']<50])

    def test_residual_growth_starts_at_same_output_and_is_charged(self):
        cfg=V12Config(warmup=4,train_samples=4,validation_samples=4,
                      ramp_samples=2,cooldown=4,max_backoff=16,
                      screen_every=1,growth_mode='residual')
        base=initial()
        s=AdaptiveKronV12(base,np.random.default_rng(3),cfg)
        rng=np.random.default_rng(45)
        for _ in range(4):
            s.step(rng.standard_normal((1,20)),rng.standard_normal((1,20)),
                   rng.standard_normal(1))
        self.assertIsNotNone(s.candidate)
        self.assertEqual(s.events[-1]['growth_mode'],'residual')
        self.assertEqual(s.events[-1]['decomposition_charge'],1_000_000)
        np.testing.assert_allclose(s.active.B@s.active.A.transpose(0,2,1),
                                   s.candidate.B@s.candidate.A.transpose(0,2,1))

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

    def test_equal_loss_prune_can_be_accepted_by_cost(self):
        cfg=SelectionConfig(lambda_cost=.1,margin=.0005)
        s=AdaptiveKron(initial(3),np.random.default_rng(1),cfg)
        s.candidate=resized_copy(s.active,-1,np.random.default_rng(2),2)
        s.validation_sum[:]=.1*cfg.validation_samples
        s._decision()
        self.assertEqual(s.events[-1]['event'],'accept')
        self.assertEqual(s.phase,'ramp')

    def test_prune_transition_completes_and_retains_history(self):
        cfg=SelectionConfig(enabled=False,ramp_samples=3)
        s=AdaptiveKron(initial(3),np.random.default_rng(1),cfg)
        s.candidate=resized_copy(s.active,-1,np.random.default_rng(2),2)
        s.phase='ramp'
        outputs=[]; filtered=[]
        for _ in range(12):
            filtered.append(s.step(np.ones((1,20)),np.ones((1,20)),np.array([1.]))[0])
            outputs.append(s.last['y'])
        np.testing.assert_allclose(filtered,ac.fir(ac.S_PATH,np.array(outputs)))
        self.assertEqual(s.active.R,2)
        self.assertIsNone(s.candidate)

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
