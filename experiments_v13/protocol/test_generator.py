"""Tiny development-only invariance tests; never generates confirmation data."""
import unittest

import numpy as np

from generator import (
    SECONDARY_FIR, interaction_count, make_p13, physical_error, physical_primary,
)


class P13GeneratorTests(unittest.TestCase):
    def test_low_high_low_term_count_and_causality(self):
        count = interaction_count(100, (30, 70))
        self.assertEqual([int(count[10]), int(count[50]), int(count[90])], [1, 4, 1])
        x = np.linspace(-1., 1., 100)
        original, _ = physical_primary(x, (30, 70))
        changed = x.copy()
        changed[60:] += 9.
        replay, _ = physical_primary(changed, (30, 70))
        np.testing.assert_array_equal(original[:60], replay[:60])

    def test_rng_is_reproducible_phase_separated_and_mapping_independent(self):
        a = make_p13("dev", 0, 100, (30, 70))
        b = make_p13("dev", 0, 100, (30, 70))
        c = make_p13("select", 0, 100, (30, 70))
        changed_map = make_p13("dev", 0, 100, (30, 70), mapping_component=22)
        for key in ("x", "d", "v", "omega", "rff_phase"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        self.assertFalse(np.array_equal(a.x, c.x))
        self.assertFalse(np.array_equal(a.omega, changed_map.omega))
        np.testing.assert_array_equal(a.d, changed_map.d)
        np.testing.assert_array_equal(a.v, changed_map.v)

    def test_prefix_invariance_and_physical_secondary_path(self):
        short = make_p13("dev", 2, 100, (30, 70))
        long = make_p13("dev", 2, 120, (30, 70))
        for key in ("x", "d", "v"):
            np.testing.assert_array_equal(getattr(short, key), getattr(long, key)[:100])
        impulse = np.zeros(100)
        impulse[5] = 1.
        error = physical_error(np.zeros(100), np.zeros(100), impulse)
        np.testing.assert_array_equal(error[5:9], -SECONDARY_FIR)

    def test_confirmation_is_locked(self):
        with self.assertRaisesRegex(ValueError, "locked"):
            make_p13("confirm", 0, 100, (30, 70))


if __name__ == "__main__":
    unittest.main()
