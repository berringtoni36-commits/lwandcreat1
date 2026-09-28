"""Known-boundary upper bound: low sorted R2, high original-order R4, low re-sorted R2.

The 100k and 200k switch times are supplied by the synthetic case design.
They are not inferred online. Only archived E1 phase=dev streams are read.
"""
from __future__ import annotations

import csv
import json

import numpy as np

import e1_capacity as ec


NAME = "boundary_realign_R2_R4_R2"
IDENTITY = np.arange(ec.D)


def run_case(run: int):
    arrays, _, summary, provenance = ec.load_run(run)
    x, d, v, om, ph = (arrays[k] for k in ("x", "d", "v", "Om", "ph"))
    c4 = ec.controller("original_R4", 4, arrays["initial_A"][0, :, :4],
                       arrays["initial_B"][0, :, :4])
    branch = None
    p = None
    xb = np.zeros((1, ec.M))
    zh = np.zeros((ec.LS, 1, ec.D))
    ad = ae4 = aeb = 0.
    sums = np.zeros((3, 2))
    boundary_sums = np.zeros((2, 2))
    transitions = {}
    audit = {}
    p_low = p_final = None
    for n in range(ec.T):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z = np.sqrt(2/ec.D)*np.cos(np.einsum("rdm,rm->rd", om, xb)+ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2]+.5*zh[3]
        if n == ec.SEGMENT:
            assert branch.R == 2
            w_sorted = (branch.B[0]@branch.A[0].T).T.reshape(ec.D)
            w_original = np.empty(ec.D)
            w_original[p] = w_sorted
            inv_z = float(abs(z[0, p]@w_sorted-z[0]@w_original))
            inv_q = float(abs(q[0, p]@w_sorted-q[0]@w_original))
            assert max(inv_z, inv_q) < 1e-12
            A, B, relative = ec.factors(w_original, IDENTITY, 4)
            new = ec.controller(NAME, 4, A, B, branch.ybuf)
            transitions["high_boundary"] = dict(sample=n, from_R=2, to_R=4,
                old_permutation="low-stage signed sort", new_permutation="identity",
                inverse_permutation_output_error_max_abs=max(inv_z, inv_q),
                R4_projection_relative_frobenius_error=relative,
                inherited_FIR_max_abs_change=float(np.max(np.abs(new.ybuf-branch.ybuf))))
            branch, p = new, IDENTITY
        if n == 2*ec.SEGMENT:
            assert branch.R == 4
            w_original = (branch.B[0]@branch.A[0].T).T.reshape(ec.D)
            p_new = np.argsort(w_original, kind="stable")
            np.testing.assert_array_equal(np.sort(p_new), IDENTITY)
            inv_z = float(abs(z[0]@w_original-z[0, p_new]@w_original[p_new]))
            inv_q = float(abs(q[0]@w_original-q[0, p_new]@w_original[p_new]))
            assert max(inv_z, inv_q) < 1e-12
            A, B, relative = ec.factors(w_original, p_new, 2)
            new = ec.controller(NAME, 2, A, B, branch.ybuf)
            transitions["final_low_boundary"] = dict(sample=n, from_R=4, to_R=2,
                old_permutation="identity", new_permutation="current signed sort",
                full_output_permutation_error_max_abs=max(inv_z, inv_q),
                R2_projection_relative_frobenius_error=relative,
                inherited_FIR_max_abs_change=float(np.max(np.abs(new.ybuf-branch.ybuf))))
            branch, p, p_final = new, p_new, p_new.copy()

        dv = np.array([d[n]+v[n]])
        ad = ec.BETA*ad+(1-ec.BETA)*abs(d[n])
        y4 = float(c4.step(z, q, dv)[0])
        ae4 = ec.BETA*ae4+(1-ec.BETA)*abs(dv[0]-y4)
        if n >= ec.SWITCH:
            yb = float(branch.step(z[:, p], q[:, p], dv)[0])
            aeb = ec.BETA*aeb+(1-ec.BETA)*abs(dv[0]-yb)
        if n == ec.SWITCH-1:
            init_result = json.loads((ec.HERE / f"E1_run{run:02d}.json").read_text(encoding="utf-8"))
            assert init_result["phase"] == "dev" and init_result["provenance"] == provenance
            saved_path = ec.HERE / init_result["initialization_file"]
            assert ec.sha(saved_path) == init_result["initialization_file_sha256"]
            with np.load(saved_path, allow_pickle=False) as saved:
                p_low = saved["permutation"].astype(int)
                w4 = (c4.B[0]@c4.A[0].T).T.reshape(ec.D)
                np.testing.assert_allclose(w4, saved["R4_low_stage_current_weight"],
                                           rtol=0, atol=1e-12)
                branch = ec.controller(NAME, 2, saved["sorted_R2_A"],
                                       saved["sorted_R2_B"], c4.ybuf)
            p = p_low
            aeb = ae4
            audit["low_switch"] = dict(sample=n, first_branch_sample=ec.SWITCH,
                                       inherited_FIR_max_abs_change=float(np.max(
                                           np.abs(branch.ybuf-c4.ybuf))))
        seg = n//ec.SEGMENT
        if n >= (seg+1)*ec.SEGMENT-ec.TAIL:
            sums[seg] += [20*np.log10((value+ec.EPS)/(ad+ec.EPS))
                          for value in (ae4, aeb)]
        for j,boundary in enumerate((ec.SEGMENT, 2*ec.SEGMENT)):
            if boundary <= n < boundary+ec.TAIL:
                boundary_sums[j] += [20*np.log10((value+ec.EPS)/(ad+ec.EPS))
                                     for value in (ae4, aeb)]
    result_segments = []
    for j in range(3):
        ref = float(summary["segment_anr"][j]["fixed_R4"])
        ref_error = abs(float(sums[j, 0]/ec.TAIL)-ref)
        assert ref_error < 2e-5
        saved_original = json.loads((ec.HERE / f"E1_run{run:02d}.json").read_text(encoding="utf-8"))
        if j == 0:
            first_low_error = abs(float(sums[j, 1]/ec.TAIL)-
                                  saved_original["segments"][j]["arm_ANR_db"]["sorted_R2"])
            assert first_low_error < 1e-9
            audit["first_low_replay_error_db"] = first_low_error
        result_segments.append(dict(segment=j+1, teacher_R=1 if j in (0, 2) else 4,
            original_R4_ANR_db=float(sums[j, 0]/ec.TAIL),
            boundary_realign_ANR_db=float(sums[j, 1]/ec.TAIL),
            delta_vs_original_R4_db=float((sums[j, 1]-sums[j, 0])/ec.TAIL),
            original_R4_reference_error_db=ref_error))
    assert p_low is not None and p_final is not None
    file = ec.HERE / f"E1_run{run:02d}_realign_permutations.npz"
    np.savez_compressed(file, low_permutation=p_low.astype(np.uint16),
                        final_low_permutation=p_final.astype(np.uint16))
    c2, c4_cost = ec.complete_cost(2), ec.complete_cost(4)
    body = [(ec.SWITCH*c4_cost+(ec.SEGMENT-ec.SWITCH)*c2)/ec.SEGMENT,
            c4_cost, c2]
    # Low SVD + high inverse-permutation R4 SVD + final new-sort R2 SVD.
    once = ((ec.D*4+ec.SVD_CONVENTION_MULTS+2*(ec.D1+ec.D2)) +
            (ec.D*2+ec.SVD_CONVENTION_MULTS+4*(ec.D1+ec.D2)) +
            (ec.D*4+ec.SVD_CONVENTION_MULTS+2*(ec.D1+ec.D2)))
    result = dict(phase="dev", case="E1", run=run, provenance=provenance,
                  transitions=transitions, audit=audit, segments=result_segments,
                  boundary_first_5000_ANR_db={str(boundary):{
                      "original_R4":float(boundary_sums[j, 0]/ec.TAIL),
                      NAME:float(boundary_sums[j, 1]/ec.TAIL)}
                      for j,boundary in enumerate((ec.SEGMENT, 2*ec.SEGMENT))},
                  costs=dict(segment_body_mults_per_sample=body,
                             mean_body_mults_per_sample=float(np.mean(body)),
                             one_time_counted_mults=once,
                             amortized_counted_mults_per_sample=float(np.mean(body)+once/ec.T),
                             excluded="boundary detection, sorting comparisons/moves, feature gathers, memory/state copy, candidate/gate"),
                  permutation_file=file.name, permutation_file_sha256=ec.sha(file))
    (ec.HERE / f"E1_run{run:02d}_realign.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    results = [run_case(run) for run in ec.RUNS]
    with (ec.HERE / "realign_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("run", "segment", "teacher_R", "original_R4_ANR_db",
                         "boundary_realign_ANR_db", "delta_vs_original_R4_db"))
        for result in results:
            for s in result["segments"]:
                writer.writerow((result["run"], s["segment"], s["teacher_R"],
                                 s["original_R4_ANR_db"], s["boundary_realign_ANR_db"],
                                 s["delta_vs_original_R4_db"]))
    print("5-run mean segment ANR (original R4, boundary realign):")
    for j in range(3):
        print(j+1, *(round(float(np.mean([r["segments"][j][key] for r in results])), 3)
                       for key in ("original_R4_ANR_db", "boundary_realign_ANR_db")), flush=True)


if __name__ == "__main__":
    main()
