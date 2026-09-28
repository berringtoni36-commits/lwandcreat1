"""Causality and plant-accounting check for periodic signed reordering."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "selector_resort.py"
spec = importlib.util.spec_from_file_location("resort_test_model", SOURCE)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def test_future_cannot_change_past() -> None:
    path = ROOT / "experiments_v14" / "outputs" / "n14_dev_inputs" / "caseE2_run00.npz"
    with np.load(path, allow_pickle=False) as f:
        x, d, v, omega, phase, a0, b0 = (f[key].copy() for key in
            ("x", "d", "v", "omega", "rff_phase", "initial_A", "initial_B"))
    t, cut = 160000, 110000
    cfg = dict(arm="once", proposal="signed_sort", ramp_samples=100,
               ablate_spectral_veto=False, resort_interval=50000)
    original = model.simulate(x[:t], d[:t], v[:t], omega[None], phase[None],
                              a0[None], b0[None], .05, cfg)
    changed = x[:t].copy(), d[:t].copy(), v[:t].copy()
    rng = np.random.default_rng(9917)
    for arr in changed:
        arr[cut:] = rng.standard_normal(t - cut)
    counterfactual = model.simulate(*changed, omega[None], phase[None],
                                    a0[None], b0[None], .05, cfg)
    for key in ("drive", "error", "R", "candidate_R", "retiring_R", "gamma"):
        np.testing.assert_array_equal(original[key][:cut], counterfactual[key][:cut])
    for name in original["cost"]:
        np.testing.assert_array_equal(original["cost"][name][:cut],
                                      counterfactual["cost"][name][:cut])
    assert [e for e in original["events"] if e["n"] < cut] == [
        e for e in counterfactual["events"] if e["n"] < cut]
    assert original["physical_fir_max_abs"] < 1e-10
    assert counterfactual["physical_fir_max_abs"] < 1e-10


if __name__ == "__main__":
    test_future_cannot_change_past()
    print("periodic resort causal-prefix and physical-FIR checks passed")
