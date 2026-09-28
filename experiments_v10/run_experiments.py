"""Unified experiment runner for the v10 revision (see SPEC.md).

Usage:
  python3 experiments_v10/run_experiments.py smoke
  python3 experiments_v10/run_experiments.py main      # E1-E4 step-size sweeps
  python3 experiments_v10/run_experiments.py ksweep    # E5
  python3 experiments_v10/run_experiments.py rsweep    # E6
  python3 experiments_v10/run_experiments.py ablation  # E7 sequential vs simultaneous
Raw per-run ANR curves go to data/*.npz, per-run metrics to results/*.csv,
console output to logs/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anc_core as ac  # noqa: E402

DATA, RES, LOGS = HERE / "data", HERE / "results", HERE / "logs"
for p in (DATA, RES, LOGS):
    p.mkdir(exist_ok=True)

BASE = dict(M=20, D=500, D1=25, D2=20, R=4, ell=3.9, sigma=2.0, eps=1e-8,
            beta=0.999, eps_a=1e-12, init=0.01, runs=20, ss_window=5000,
            thresholds=[-5.0, -10.0, -15.0, -20.0])
CONDITIONS = {
    "E1_chaotic":   dict(T=20000, ref="logistic", noise="none"),
    "E2_sas_ref":   dict(T=20000, ref="sas", alpha=1.6, clip=5.0, noise="none"),
    "E3_gauss":     dict(T=30000, ref="gauss", noise="gauss", noise_std=0.01),
    "E4_sas_noise": dict(T=30000, ref="gauss", noise="sas", noise_alpha=1.6, noise_scale=0.05),
}
MU_GRID = [0.05, 0.1, 0.2, 0.4, 0.8]
SEED_BASE = 2026


def code_hash():
    h = hashlib.sha256()
    for f in ("anc_core.py", "run_experiments.py"):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:12]


def make_signals(cond, runs, seed_key):
    """Independent realizations: every run has its own reference and noise."""
    rng = np.random.default_rng([SEED_BASE, *seed_key, 1])
    c = CONDITIONS[cond]
    T = c["T"]
    if c["ref"] == "logistic":
        x = ac.logistic_delay6(rng, runs, T)
    elif c["ref"] == "sas":
        x = ac.alpha_stable_reference(rng, runs, T, c["alpha"], c["clip"])
    else:
        x = rng.standard_normal((runs, T))
    d = ac.primary_disturbance(x)
    if c["noise"] == "none":
        v = np.zeros_like(x)
    elif c["noise"] == "gauss":
        v = c["noise_std"] * rng.standard_normal((runs, T))
    else:
        v = c["noise_scale"] * ac.sas_cms(rng, c["noise_alpha"], (runs, T))
    return x, d, v


def make_map(p, runs, seed_key):
    rng = np.random.default_rng([SEED_BASE, *seed_key, 2])
    return ac.draw_rff(rng, runs, p["D"], p["M"], p["ell"])


def init_rng(seed_key, tag):
    return np.random.default_rng([SEED_BASE, *seed_key, 3, tag])


class Sub:
    """Full controller that uses only the first Dk features (equal trainable budget)."""

    def __init__(self, inner, Dk):
        self.inner, self.Dk, self.name = inner, Dk, inner.name
        self.n_trainable = Dk

    def step(self, z, q, dv):
        return self.inner.step(z[:, : self.Dk], q[:, : self.Dk], dv)


def controllers_main(p, mu, runs, seed_key):
    return [
        ac.FullRFF("RFF-NLMS", runs, p["D"], mu, None, p["eps"]),
        ac.FullRFF("RFF-MCC", runs, p["D"], mu, p["sigma"], p["eps"]),
        Sub(ac.FullRFF("RFF-MCC-180", runs, 180, mu, p["sigma"], p["eps"]), 180),
        ac.KronRFF("NKP-RFF-MCC", runs, p["D1"], p["D2"], p["R"], mu / 2, mu / 2, p["sigma"], p["eps"],
                   init_rng(seed_key, 0), p["init"]),
    ]


def metrics_rows(exp, cond, mu, anr, p, extra=None):
    rows = []
    names = list(anr)
    ss = {k: ac.steady_state(v, p["ss_window"]) for k, v in anr.items()}
    for k in names:
        for r in range(anr[k].shape[0]):
            row = dict(exp=exp, cond=cond, mu=mu, ctrl=k, run=r, ss_anr=float(ss[k][r]),
                       min_anr=float(anr[k][r].min()))
            for tau in p["thresholds"]:
                row[f"t_{int(tau)}dB"] = int(ac.first_crossing(anr[k][r:r + 1], tau)[0])
            if extra:
                row.update(extra)
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_one(exp, cond, p, ctrls_fn, seed_key, tag, runs=None, T=None):
    runs = runs or p["runs"]
    x, d, v = make_signals(cond, runs, seed_key)
    if T is not None:
        x, d, v = x[:, :T], d[:, :T], v[:, :T]
    Om, ph = make_map(p, runs, seed_key)
    ctrls = ctrls_fn(runs)
    t0 = time.time()
    anr = ac.simulate(x, d, v, ctrls, Om, ph, p["beta"], p["eps_a"])
    dt = time.time() - t0
    meta = dict(exp=exp, cond=cond, tag=tag, seed=[SEED_BASE, *seed_key], params=p,
                condition=CONDITIONS[cond], runs=runs, T=x.shape[1], seconds=round(dt, 1),
                code_sha256_12=code_hash(),
                trainable={c.name: c.n_trainable for c in ctrls})
    np.savez_compressed(DATA / f"{tag}.npz", **{k: v for k, v in anr.items()}, meta=json.dumps(meta))
    return anr, meta


def summarize(anr, p):
    out = {}
    for k, v in anr.items():
        ss = ac.steady_state(v, p["ss_window"])
        out[k] = f"ss {ss.mean():7.2f} +- {ss.std(ddof=1) if len(ss) > 1 else 0:5.2f} dB"
    return out


def main(mode):
    p = dict(BASE)
    log = open(LOGS / f"run_{mode}.log", "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say(f"=== {mode} {time.strftime('%Y-%m-%d %H:%M:%S')} code {code_hash()}")
    if mode == "smoke":
        for cond in CONDITIONS:
            for mu in (0.1, 0.4):
                key = (hash(cond) % 1000, int(mu * 1000))
                anr, meta = run_one("smoke", cond, p, lambda r: controllers_main(p, mu, r, key), key,
                                    f"smoke_{cond}_mu{mu}", runs=3, T=6000)
                say(f"{cond} mu={mu} ({meta['seconds']} s, 3 runs x 6000): " + json.dumps(summarize(anr, dict(p, ss_window=1000))))
        return
    if mode == "main":
        rows = []
        for ci, cond in enumerate(CONDITIONS):
            for mi, mu in enumerate(MU_GRID):
                key = (ci + 1, mi + 1)
                tag = f"{cond}_mu{mu}"
                anr, meta = run_one("main", cond, p, lambda r: controllers_main(p, mu, r, key), key, tag)
                rows += metrics_rows("main", cond, mu, anr, p)
                say(f"{tag} ({meta['seconds']} s): " + json.dumps(summarize(anr, p)))
                write_csv(RES / "main_per_run.csv", rows)
        return
    if mode == "ksweep":
        mu = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
        ks = [1, 1.5, 2, 2.8, 4, 5.6, 8, 12, 20]
        key = (5, int(mu * 1000))

        def ctrls(r):
            cs = []
            for k in ks:
                s = p["sigma"] / np.sqrt(k)
                cs.append(ac.FullRFF(f"RFF-MCC_k{k}", r, p["D"], mu, s, p["eps"]))
                cs.append(ac.KronRFF(f"NKP-RFF-MCC_k{k}", r, p["D1"], p["D2"], p["R"], mu / 2, mu / 2, s,
                                     p["eps"], init_rng(key, 0), p["init"]))
            return cs
        anr, meta = run_one("ksweep", "E1_chaotic", p, ctrls, key, f"E5_ksweep_mu{mu}")
        write_csv(RES / f"E5_ksweep_mu{mu}_per_run.csv", metrics_rows("ksweep", "E1_chaotic", mu, anr, p))
        say(f"E5 ksweep mu={mu} ({meta['seconds']} s): " + json.dumps(summarize(anr, p)))
        return
    if mode == "rsweep":
        mu = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
        Rs = [1, 2, 4, 8, 12, 20]
        key = (6, int(mu * 1000))

        def ctrls(r):
            cs = [ac.FullRFF("RFF-MCC", r, p["D"], mu, p["sigma"], p["eps"])]
            for R in Rs:
                cs.append(ac.KronRFF(f"NKP-RFF-MCC_R{R}", r, p["D1"], p["D2"], R, mu / 2, mu / 2, p["sigma"],
                                     p["eps"], init_rng(key, R), p["init"]))
            return cs
        anr, meta = run_one("rsweep", "E1_chaotic", p, ctrls, key, f"E6_rsweep_mu{mu}")
        write_csv(RES / f"E6_rsweep_mu{mu}_per_run.csv", metrics_rows("rsweep", "E1_chaotic", mu, anr, p))
        say(f"E6 rsweep mu={mu} ({meta['seconds']} s): " + json.dumps(summarize(anr, p)))
        return
    if mode == "ablation":
        mu = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
        key = (7, int(mu * 1000))

        def ctrls(r):
            return [
                ac.KronRFF("NKP-sequential", r, p["D1"], p["D2"], p["R"], mu / 2, mu / 2, p["sigma"], p["eps"],
                           init_rng(key, 0), p["init"], sequential=True),
                ac.KronRFF("NKP-simultaneous", r, p["D1"], p["D2"], p["R"], mu / 2, mu / 2, p["sigma"], p["eps"],
                           init_rng(key, 0), p["init"], sequential=False),
            ]
        anr, meta = run_one("ablation", "E1_chaotic", p, ctrls, key, f"E7_ablation_mu{mu}")
        write_csv(RES / f"E7_ablation_mu{mu}_per_run.csv", metrics_rows("ablation", "E1_chaotic", mu, anr, p))
        say(f"E7 ablation mu={mu} ({meta['seconds']} s): " + json.dumps(summarize(anr, p)))
        return
    if mode == "init":
        mu = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
        scales = [0.01, 0.1, 0.3]
        key = (8, int(mu * 1000))

        def ctrls(r):
            cs = [ac.FullRFF("RFF-MCC", r, p["D"], mu, p["sigma"], p["eps"])]
            for s0 in scales:
                cs.append(ac.KronRFF(f"NKP-RFF-MCC_init{s0}", r, p["D1"], p["D2"], p["R"], mu / 2, mu / 2,
                                     p["sigma"], p["eps"], init_rng(key, 0), s0))
            return cs
        anr, meta = run_one("init", "E1_chaotic", p, ctrls, key, f"E8_init_mu{mu}")
        write_csv(RES / f"E8_init_mu{mu}_per_run.csv", metrics_rows("init", "E1_chaotic", mu, anr, p))
        say(f"E8 init mu={mu} ({meta['seconds']} s): " + json.dumps(summarize(anr, p)))
        return
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke")
