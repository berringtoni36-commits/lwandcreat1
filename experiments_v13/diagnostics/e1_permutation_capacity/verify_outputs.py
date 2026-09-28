"""Cross-check saved E1 dev provenance, geometry, and physical replay."""
import json

import numpy as np

import e1_capacity as ec


def read(run, suffix=""):
    return json.loads((ec.HERE / f"E1_run{run:02d}{suffix}.json").read_text(encoding="utf-8"))


def main():
    protocol = json.loads((ec.DEV / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["phase"] == "dev"
    assert ec.sha(ec.DEV / "sources" / "experiments_v12" / "run.py") == protocol["sources_sha256"]["experiments_v12\\run.py"]
    assert ec.sha(ec.DEV / "sources" / "experiments_v10" / "anc_core.py") == protocol["sources_sha256"]["experiments_v10\\anc_core.py"]
    max_reference = max_teacher_d = max_fro = max_inverse = 0.
    for run in ec.RUNS:
        _, _, _, provenance = ec.load_run(run)
        base, realign, delayed = (read(run, suffix) for suffix in ("", "_realign", "_delayed"))
        assert base["phase"] == realign["phase"] == delayed["phase"] == "dev"
        assert base["provenance"] == realign["provenance"] == delayed["provenance"] == provenance
        path = ec.HERE / base["initialization_file"]
        assert ec.sha(path) == base["initialization_file_sha256"]
        with np.load(path, allow_pickle=False) as f:
            p = f["permutation"].astype(int)
            np.testing.assert_array_equal(np.sort(p), np.arange(ec.D))
            w = f["R4_low_stage_current_weight"]
            matrix = w[p].reshape(ec.D1, ec.D2).T
            for rank in (2, 4):
                approx = f[f"sorted_R{rank}_B"] @ f[f"sorted_R{rank}_A"].T
                actual = float(np.linalg.norm(matrix-approx)/np.linalg.norm(matrix))
                reported = base["switch"][f"R{rank}_initial_relative_frobenius_error"]
                max_fro = max(max_fro, abs(actual-reported))
                assert abs(actual-reported) < 1e-12
            low, high = ec.known_teacher(run)
            for stage, teacher in (("low", low), ("high", high)):
                calculated = ec.geometry(teacher, p)
                reported = base["switch"]["teacher_geometry_sorted"][stage]
                assert calculated["numeric_rank"] == reported["numeric_rank"]
                for key in ("relative_best_R1", "relative_best_R2", "relative_best_R4"):
                    assert abs(calculated[key]-reported[key]) < 1e-12
        max_teacher_d = max(max_teacher_d, base["teacher_d_reconstruction_max_abs"])
        assert max_teacher_d < 1e-10
        assert base["costs"] == ec.costs()
        assert base["switch"]["full_output_invariance_max_abs"] < 1e-12
        assert base["switch"]["inherited_FIR_max_abs_change"] == 0
        assert base["transitions"]["growth"]["immediate_weight_max_abs_change"] == 0
        assert base["transitions"]["growth"]["inherited_FIR_max_abs_change"] == 0
        assert base["transitions"]["prune"]["inherited_FIR_max_abs_change"] == 0
        max_reference = max(max_reference, *(s["max_reference_replay_abs_difference_db"]
                                             for s in base["segments"]))
        for j in range(3):
            b = base["segments"][j]["arm_ANR_db"]
            d = delayed["segments"][j]["arm_ANR_db"]
            assert abs(d["original_R4"]-b["original_R4"]) < 1e-9
            assert abs(d["prune_200k"]-realign["segments"][j]["boundary_realign_ANR_db"]) < 1e-9
            if j == 0:
                assert abs(b["sorted_R2"]-b["boundary_R2_R4_R2"]) < 1e-9
                assert abs(b["sorted_R2"]-d["prune_200k"]) < 1e-9
        file = ec.HERE / realign["permutation_file"]
        assert ec.sha(file) == realign["permutation_file_sha256"]
        with np.load(file, allow_pickle=False) as f:
            np.testing.assert_array_equal(f["low_permutation"], p)
            np.testing.assert_array_equal(np.sort(f["final_low_permutation"]), np.arange(ec.D))
        for event in ("high_boundary", "final_low_boundary"):
            assert realign["transitions"][event]["inherited_FIR_max_abs_change"] == 0
        max_inverse = max(max_inverse,
            realign["transitions"]["high_boundary"]["inverse_permutation_output_error_max_abs"],
            realign["transitions"]["final_low_boundary"]["full_output_permutation_error_max_abs"])
        assert max_inverse < 1e-12
        for at in (200000, 220000, 250000):
            event = delayed["transitions"][f"prune_{at//1000}k"]
            assert event["sample"] == at and event["inherited_FIR_error"] == 0
            assert event["full_output_permutation_error_max_abs"] < 1e-12
            assert 0 <= event["R4_to_R2_projection_relative_error"] <= 1
    assert max_reference < 2e-5
    print(json.dumps(dict(checked_dev_runs=5, checked_segment_tails=15,
                          max_original_reference_ANR_difference_db=max_reference,
                          max_teacher_disturbance_reconstruction_abs=max_teacher_d,
                          max_saved_SVD_error_difference=max_fro,
                          max_permutation_output_difference=max_inverse), indent=2))


if __name__ == "__main__":
    main()
