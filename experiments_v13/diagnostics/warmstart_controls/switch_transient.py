"""Score the first 5,000 physical samples after the predefined 20k switch."""
from __future__ import annotations

import csv
import json

import numpy as np

import warmstart_controls as wc


WINDOW = 500
WINDOWS = 10


def replay(case: str, run: int):
    arrays, meta, _, provenance = wc.load_case(case, run)
    x, d, v, om, ph = (arrays[k] for k in ("x", "d", "v", "Om", "ph"))
    mu = float(meta["mu"])
    fixed = wc.ac.KronRFF("continuous_R4", 1, wc.D1, wc.D2, 4,
                          mu/2, mu/2, 2., 1e-8, np.random.default_rng(0))
    fixed.A = arrays["initial_A"][:, :, :4].copy()
    fixed.B = arrays["initial_B"][:, :, :4].copy()
    xb = np.zeros((1, wc.M))
    zh = np.zeros((wc.LS, 1, wc.D))
    ae = {name: 0. for name in wc.ARMS}
    ad = 0.
    buckets = np.zeros((WINDOWS, len(wc.ARMS)))
    branches = {}
    for n in range(wc.SWITCH+WINDOW*WINDOWS):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z = np.sqrt(2/wc.D)*np.cos(np.einsum("rdm,rm->rd", om, xb)+ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2]+.5*zh[3]
        dv = np.array([d[n]+v[n]])
        ad = wc.BETA*ad+(1-wc.BETA)*abs(d[n])
        ys = float(fixed.step(z, q, dv)[0])
        ae["continuous_R4"] = wc.BETA*ae["continuous_R4"]+(1-wc.BETA)*abs(dv[0]-ys)
        if n >= wc.SWITCH:
            for name, state in branches.items():
                p = state["p"]
                ctrl = state["controller"]
                branch_ys = (ctrl.step(z[0, p]) if name == "sorted_frozen_R2"
                             else float(ctrl.step(z[:, p], q[:, p], dv)[0]))
                ae[name] = wc.BETA*ae[name]+(1-wc.BETA)*abs(dv[0]-branch_ys)
            bucket = (n-wc.SWITCH)//WINDOW
            buckets[bucket] += [20*np.log10((ae[name]+wc.ANR_EPS)/(ad+wc.ANR_EPS))
                                for name in wc.ARMS]
        if n == wc.SWITCH-1:
            branches, _, initialization = wc.init_branches(case, run, fixed, mu, z, q)
            saved_path = wc.HERE / f"{case}_run{run:02d}_initialization.npz"
            with np.load(saved_path, allow_pickle=False) as saved:
                for key, value in initialization.items():
                    np.testing.assert_allclose(saved[key], value, rtol=0, atol=1e-12)
            for name in branches:
                ae[name] = ae["continuous_R4"]
    mean_buckets = buckets/WINDOW
    return dict(case=case, run=run, phase="dev", source_sha256=provenance["source_sha256"],
                switch_after_samples=wc.SWITCH, window_length=WINDOW,
                first_5000_ANR_db={arm:float(mean_buckets[:, i].mean())
                                   for i,arm in enumerate(wc.ARMS)},
                windows=[dict(bounds=[wc.SWITCH+j*WINDOW, wc.SWITCH+(j+1)*WINDOW],
                              arm_ANR_db={arm:float(mean_buckets[j, i])
                                          for i,arm in enumerate(wc.ARMS)})
                         for j in range(WINDOWS)])


def main():
    results = [replay(case, run) for case in wc.CASES for run in wc.RUNS]
    (wc.HERE / "switch_transient.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with (wc.HERE / "switch_transient.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("case", "run", "window", "arm", "ANR_db"))
        for result in results:
            for j, window in enumerate(result["windows"]):
                for arm in wc.ARMS:
                    writer.writerow((result["case"], result["run"], j+1, arm,
                                     window["arm_ANR_db"][arm]))
    for case in wc.CASES:
        subset = [r for r in results if r["case"] == case]
        print(case, {arm:round(float(np.mean([r["first_5000_ANR_db"][arm]
                                              for r in subset])), 3) for arm in wc.ARMS}, flush=True)


if __name__ == "__main__":
    main()
