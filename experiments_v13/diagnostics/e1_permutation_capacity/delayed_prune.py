"""Known-boundary delayed low-stage prune at 200k, 220k or 250k (dev only)."""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

import e1_capacity as ec


TIMES = (200000, 220000, 250000)
ARMS = ("original_R4", "prune_200k", "prune_220k", "prune_250k")
IDENTITY = np.arange(ec.D)


def run_case(run: int):
    arrays, _, summary, provenance = ec.load_run(run)
    teacher_low, _ = ec.known_teacher(run)
    x, d, v, om, ph = (arrays[k] for k in ("x", "d", "v", "Om", "ph"))
    original = ec.controller("original_R4", 4, arrays["initial_A"][0, :, :4],
                             arrays["initial_B"][0, :, :4])
    active = None
    active_p = None
    branches = {}
    ae_branches = {}
    xb = np.zeros((1, ec.M))
    zh = np.zeros((ec.LS, 1, ec.D))
    ad = ae_original = ae_active = 0.
    sums = np.zeros((3, len(ARMS)))
    transient = np.zeros((len(TIMES), 2))
    transitions = {}

    for n in range(ec.T):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z = np.sqrt(2/ec.D)*np.cos(np.einsum("rdm,rm->rd", om, xb)+ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2]+.5*zh[3]
        if n == ec.SEGMENT:
            assert active.R == 2
            w_sorted = (active.B[0]@active.A[0].T).T.reshape(ec.D)
            w_original = np.empty(ec.D)
            w_original[active_p] = w_sorted
            assert abs(z[0, active_p]@w_sorted-z[0]@w_original) < 1e-12
            assert abs(q[0, active_p]@w_sorted-q[0]@w_original) < 1e-12
            A, B, relative = ec.factors(w_original, IDENTITY, 4)
            new = ec.controller("active_high_R4", 4, A, B, active.ybuf)
            assert np.array_equal(new.ybuf, active.ybuf)
            active, active_p = new, IDENTITY
            transitions["high_boundary"] = dict(sample=n,
                inverse_permutation_R4_projection_relative_error=relative,
                inherited_FIR_error=0.)
        if n == 2*ec.SEGMENT:
            assert active.R == 4
            for at in TIMES:
                name = f"prune_{at//1000}k"
                branches[name] = dict(controller=ec.controller(name, 4,
                    active.A[0], active.B[0], active.ybuf), p=IDENTITY)
                ae_branches[name] = ae_active
                np.testing.assert_array_equal(branches[name]["controller"].ybuf,
                                              active.ybuf)
        if n in TIMES:
            name = f"prune_{n//1000}k"
            state = branches[name]
            ctrl = state["controller"]
            assert ctrl.R == 4 and np.array_equal(state["p"], IDENTITY)
            w = (ctrl.B[0]@ctrl.A[0].T).T.reshape(ec.D)
            p = np.argsort(w, kind="stable")
            np.testing.assert_array_equal(np.sort(p), IDENTITY)
            inv_z = float(abs(z[0]@w-z[0, p]@w[p]))
            inv_q = float(abs(q[0]@w-q[0, p]@w[p]))
            assert max(inv_z, inv_q) < 1e-12
            A, B, relative = ec.factors(w, p, 2)
            new = ec.controller(name, 2, A, B, ctrl.ybuf)
            assert np.array_equal(new.ybuf, ctrl.ybuf)
            state["controller"], state["p"] = new, p
            transitions[name] = dict(sample=n, R4_to_R2_projection_relative_error=relative,
                full_output_permutation_error_max_abs=max(inv_z, inv_q),
                inherited_FIR_error=0.,
                true_low_teacher_geometry_under_current_sort=ec.geometry(teacher_low, p))

        dv = np.array([d[n]+v[n]])
        ad = ec.BETA*ad+(1-ec.BETA)*abs(d[n])
        y4 = float(original.step(z, q, dv)[0])
        ae_original = ec.BETA*ae_original+(1-ec.BETA)*abs(dv[0]-y4)
        if ec.SWITCH <= n < 2*ec.SEGMENT:
            yb = float(active.step(z[:, active_p], q[:, active_p], dv)[0])
            ae_active = ec.BETA*ae_active+(1-ec.BETA)*abs(dv[0]-yb)
        if n >= 2*ec.SEGMENT:
            for name, state in branches.items():
                p = state["p"]
                yb = float(state["controller"].step(z[:, p], q[:, p], dv)[0])
                ae_branches[name] = (ec.BETA*ae_branches[name] +
                                     (1-ec.BETA)*abs(dv[0]-yb))
        if n == ec.SWITCH-1:
            previous = json.loads((ec.HERE / f"E1_run{run:02d}.json").read_text(encoding="utf-8"))
            assert previous["phase"] == "dev" and previous["provenance"] == provenance
            file = ec.HERE / previous["initialization_file"]
            assert ec.sha(file) == previous["initialization_file_sha256"]
            with np.load(file, allow_pickle=False) as saved:
                w4 = (original.B[0]@original.A[0].T).T.reshape(ec.D)
                np.testing.assert_allclose(w4, saved["R4_low_stage_current_weight"],
                                           rtol=0, atol=1e-12)
                active_p = saved["permutation"].astype(int)
                active = ec.controller("active_low_R2", 2, saved["sorted_R2_A"],
                                       saved["sorted_R2_B"], original.ybuf)
            ae_active = ae_original
            assert np.array_equal(active.ybuf, original.ybuf)
        seg = n//ec.SEGMENT
        if n >= (seg+1)*ec.SEGMENT-ec.TAIL:
            values = [ae_original]+([ae_active]*3 if seg < 2 else
                                     [ae_branches[name] for name in ARMS[1:]])
            sums[seg] += [20*np.log10((value+ec.EPS)/(ad+ec.EPS)) for value in values]
        for j,at in enumerate(TIMES):
            if at <= n < at+ec.TAIL:
                transient[j] += [20*np.log10((value+ec.EPS)/(ad+ec.EPS))
                                 for value in (ae_original, ae_branches[f"prune_{at//1000}k"])]

    result_segments = []
    prior = json.loads((ec.HERE / f"E1_run{run:02d}.json").read_text(encoding="utf-8"))
    immediate = json.loads((ec.HERE / f"E1_run{run:02d}_realign.json").read_text(encoding="utf-8"))
    assert prior["provenance"] == immediate["provenance"] == provenance
    for j in range(3):
        scores = {arm:float(sums[j, k]/ec.TAIL) for k,arm in enumerate(ARMS)}
        expected = float(summary["segment_anr"][j]["fixed_R4"])
        assert abs(scores["original_R4"]-expected) < 2e-5
        if j == 0:
            assert abs(scores["prune_200k"]-
                       prior["segments"][j]["arm_ANR_db"]["sorted_R2"]) < 1e-9
        else:
            assert abs(scores["prune_200k"]-
                       immediate["segments"][j]["boundary_realign_ANR_db"]) < 1e-9
        result_segments.append(dict(segment=j+1, teacher_R=1 if j in (0, 2) else 4,
                                    arm_ANR_db=scores))
    c2, c4 = ec.complete_cost(2), ec.complete_cost(4)
    first = (ec.SWITCH*c4+(ec.SEGMENT-ec.SWITCH)*c2)/ec.SEGMENT
    once = ((ec.D*4+ec.SVD_CONVENTION_MULTS+2*(ec.D1+ec.D2)) +
            (ec.D*2+ec.SVD_CONVENTION_MULTS+4*(ec.D1+ec.D2)) +
            (ec.D*4+ec.SVD_CONVENTION_MULTS+2*(ec.D1+ec.D2)))
    cost = {}
    for at in TIMES:
        name = f"prune_{at//1000}k"
        low_last = ((at-2*ec.SEGMENT)*c4+(ec.T-at)*c2)/ec.SEGMENT
        body = [first, c4, low_last]
        cost[name] = dict(segment_body_mults_per_sample=body,
                          mean_body_mults_per_sample=float(np.mean(body)),
                          one_time_counted_mults=once,
                          amortized_counted_mults_per_sample=float(np.mean(body)+once/ec.T),
                          excluded="known-boundary detection, sorting comparisons/moves, feature gathers, state copy, candidate/gate")
    result = dict(phase="dev", case="E1", run=run, provenance=provenance,
                  transitions=transitions, segments=result_segments,
                  first_5000_after_prune_ANR_db={str(at):{
                      "original_R4":float(transient[j, 0]/ec.TAIL),
                      f"prune_{at//1000}k":float(transient[j, 1]/ec.TAIL)}
                      for j,at in enumerate(TIMES)},
                  costs=cost)
    (ec.HERE / f"E1_run{run:02d}_delayed.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def cache_or_run(run: int, refresh: bool):
    path = ec.HERE / f"E1_run{run:02d}_delayed.json"
    if path.exists() and not refresh:
        result = json.loads(path.read_text(encoding="utf-8"))
        _, _, _, provenance = ec.load_run(run)
        assert result["phase"] == "dev" and result["provenance"] == provenance
        return result
    return run_case(run)


def aggregate(results):
    with (ec.HERE / "delayed_prune_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("run", "segment", "teacher_R", "arm", "ANR_db",
                         "delta_vs_original_R4_db", "prune_spectral_tail_relative_error",
                         "mean_body_mults_per_sample"))
        for result in results:
            for segment in result["segments"]:
                baseline = segment["arm_ANR_db"]["original_R4"]
                for arm in ARMS:
                    at = arm.split("_")[1] if arm != "original_R4" else None
                    writer.writerow((result["run"], segment["segment"], segment["teacher_R"], arm,
                                     segment["arm_ANR_db"][arm], segment["arm_ANR_db"][arm]-baseline,
                                     result["transitions"][arm]["R4_to_R2_projection_relative_error"] if at else "",
                                     result["costs"][arm]["mean_body_mults_per_sample"] if at else ec.complete_cost(4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, nargs="+", choices=ec.RUNS)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    results = []
    for run in args.runs or ec.RUNS:
        result = cache_or_run(run, args.refresh)
        results.append(result)
        print(run, [round(result["segments"][2]["arm_ANR_db"][a], 3)
                    for a in ARMS], flush=True)
    aggregate(results)
