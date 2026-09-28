"""Independent checks of saved development-only warmstart-control outputs."""
from __future__ import annotations

import json

import numpy as np

import warmstart_controls as wc


def main():
    errors = []
    ref_errors = []
    random_projection_errors = []
    transient = json.loads((wc.HERE / "switch_transient.json").read_text(encoding="utf-8"))
    assert len(transient) == 10
    transient_by_key = {(r["case"], r["run"]): r for r in transient}
    transient_mean_errors = []
    for case in wc.CASES:
        for run in wc.RUNS:
            result = json.loads((wc.HERE / f"{case}_run{run:02d}.json").read_text(encoding="utf-8"))
            _, _, _, provenance = wc.load_case(case, run)
            assert result["phase"] == "dev" and result["provenance"] == provenance
            assert result["arms"] == list(wc.ARMS)
            path = wc.HERE / result["initialization_file"]
            assert wc.sha(path) == result["initialization_file_sha256"]
            with np.load(path, allow_pickle=False) as data:
                p = data["permutation"].astype(int)
                np.testing.assert_array_equal(np.sort(p), np.arange(wc.D))
                w = data["R4_original_weights"]
                sorted_w = w[p].reshape(wc.D1, wc.D2).T
                identity_w = w.reshape(wc.D1, wc.D2).T
                assert np.linalg.matrix_rank(sorted_w, tol=1e-9) > 4
                assert np.linalg.matrix_rank(identity_w, tol=1e-9) == 4
                for label, matrix, key in (("sorted_svd_R2", sorted_w, "sorted_R2"),
                                           ("sorted_svd_R4", sorted_w, "sorted_R4"),
                                           ("identity_svd_R2", identity_w, "identity_R2")):
                    approx = data[f"{label}_B"] @ data[f"{label}_A"].T
                    error = float(np.linalg.norm(matrix-approx)/np.linalg.norm(matrix))
                    expected = result["initialization"][key]["relative_frobenius_error"]
                    assert abs(error-expected) < 1e-12
                    errors.append(abs(error-expected))
                random_approx = data["sorted_random_R2_B"] @ data["sorted_random_R2_A"].T
                random_projection_errors.append(float(np.linalg.norm(sorted_w-random_approx)/np.linalg.norm(sorted_w)))
            assert result["initialization"]["full_output_invariance_max_abs"] < 1e-12
            assert result["initialization"]["inherited_secondary_FIR_error"] == 0
            assert result["initialization"]["projected_control_check_abs"] < 1e-12
            assert result["costs"] == {name: wc.cost_record(name) for name in wc.ARMS}
            short = transient_by_key[(case, run)]
            assert short["phase"] == "dev"
            assert short["source_sha256"] == provenance["source_sha256"]
            assert len(short["windows"]) == 10
            for j, window in enumerate(short["windows"]):
                assert window["bounds"] == [wc.SWITCH+j*500, wc.SWITCH+(j+1)*500]
            for arm in wc.ARMS:
                transient_mean_errors.append(abs(short["first_5000_ANR_db"][arm] -
                                                 np.mean([w["arm_ANR_db"][arm]
                                                          for w in short["windows"]])))
            for segment in result["segments"]:
                assert len(segment["arm_ANR_db"]) == len(wc.ARMS)
                ref_errors.append(segment["reference_replay_abs_difference_db"])
                assert ref_errors[-1] < 2e-5
    print(json.dumps(dict(checked_runs=10, checked_segments=40,
                          max_saved_svd_error_discrepancy=max(errors),
                          max_R4_reference_ANR_discrepancy_db=max(ref_errors),
                          max_transient_mean_discrepancy_db=max(transient_mean_errors),
                          random_initial_relative_frobenius_error_range=[
                              min(random_projection_errors), max(random_projection_errors)]), indent=2))


if __name__ == "__main__":
    main()
