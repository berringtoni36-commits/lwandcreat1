"""Statistics and figures of the v10 revision, computed only from data/*.npz and results/*.csv.

Run: python3 experiments_v10/analyze.py
Outputs: results/summary_main.csv, results/summary_stats.txt, figures/*.pdf|png
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anc_core as ac  # noqa: E402

DATA, RES, FIG = HERE / "data", HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)
CONDS = ["E1_chaotic", "E2_sas_ref", "E3_gauss", "E4_sas_noise"]
LABEL = {"E1_chaotic": "Logistic-chaotic reference", "E2_sas_ref": r"$\alpha$-stable reference (clipped)",
         "E3_gauss": "Gaussian reference + Gaussian noise", "E4_sas_noise": r"Gaussian reference + $\alpha$-stable noise"}
CTRLS = ["RFF-NLMS", "RFF-MCC", "RFF-MCC-180", "NKP-RFF-MCC"]
MUS = [0.05, 0.1, 0.2, 0.4, 0.8]
STYLE = {"RFF-NLMS": dict(color="#687480", ls="-."), "RFF-MCC": dict(color="#236A9F", ls="--"),
         "RFF-MCC-180": dict(color="#76539A", ls=":"), "NKP-RFF-MCC": dict(color="#C65D2B", ls="-")}
DISPLAY = {"RFF-NLMS": "RFF-NLMS (D=500)", "RFF-MCC": "RFF-MCC (D=500)",
           "RFF-MCC-180": "RFF-MCC (D=180)", "NKP-RFF-MCC": "NKP-RFF-MCC (500 features, 180 coeff.)"}

mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "savefig.facecolor": "white"})

rows = list(csv.DictReader(open(RES / "main_per_run.csv")))
per = defaultdict(list)
for r in rows:
    per[(r["cond"], float(r["mu"]), r["ctrl"])].append(r)


def arr(cond, mu, ctrl, key, cast=float):
    return np.array([cast(r[key]) for r in sorted(per[(cond, mu, ctrl)], key=lambda r: int(r["run"]))])


# pre-declared operating point: step size that minimizes the mean steady-state ANR of RFF-MCC
main_mu = {c: min(MUS, key=lambda m: arr(c, m, "RFF-MCC", "ss_anr").mean()) for c in CONDS}

out_rows, txt = [], []
for c in CONDS:
    for mu in MUS:
        for k in CTRLS:
            ss = arr(c, mu, k, "ss_anr")
            row = dict(cond=c, mu=mu, ctrl=k, main=int(mu == main_mu[c]), n=len(ss),
                       ss_mean=round(ss.mean(), 2), ss_std=round(ss.std(ddof=1), 2))
            for th in ("t_-5dB", "t_-10dB"):
                t = arr(c, mu, k, th, int)
                ok = t >= 0
                row[th + "_crossed"] = int(ok.sum())
                row[th + "_mean"] = round(t[ok].mean(), 0) if ok.any() else ""
                row[th + "_std"] = round(t[ok].std(ddof=1), 0) if ok.sum() > 1 else ""
            out_rows.append(row)
with open(RES / "summary_main.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

txt.append("Pre-declared operating point (RFF-MCC best mean steady state): " + json.dumps(main_mu))
for c in CONDS:
    mu = main_mu[c]
    txt.append(f"\n== {c} at mu={mu} (structured: mu_a=mu_b=mu/2); 20 independent runs; ss = mean ANR of last 5000 samples")
    for k in CTRLS:
        ss = arr(c, mu, k, "ss_anr")
        t5 = arr(c, mu, k, "t_-5dB", int)
        ok = t5 >= 0
        txt.append(f"  {k:12s} ss {ss.mean():7.2f} +- {ss.std(ddof=1):4.2f} dB | t(-5 dB) "
                   + (f"{t5[ok].mean():7.0f} +- {t5[ok].std(ddof=1):5.0f} ({ok.sum()}/20 crossed)" if ok.any() else "never"))
    for a, b in (("NKP-RFF-MCC", "RFF-MCC"), ("NKP-RFF-MCC", "RFF-MCC-180"), ("RFF-MCC", "RFF-NLMS")):
        x, y = arr(c, mu, a, "ss_anr"), arr(c, mu, b, "ss_anr")
        W, p, n = ac.wilcoxon_signed_rank_exact(x, y)
        txt.append(f"  paired {a} - {b}: mean diff {np.mean(x - y):+.2f} dB, wins(lower ANR) {(x < y).sum()}/20, Wilcoxon p = {p:.2e}")

# asymmetric-step comparison that mimics the old Table 3 (baseline mu=0.05, structured effective 0.2)
c = "E1_chaotic"
txt.append("\n== Asymmetric-step check (E1): RFF-MCC at mu=0.05 vs NKP-RFF-MCC at effective 0.2 (mu_a=mu_b=0.1)")
for lab, mu, k in (("RFF-MCC mu=0.05", 0.05, "RFF-MCC"), ("NKP eff=0.2", 0.2, "NKP-RFF-MCC"), ("RFF-MCC mu=0.2", 0.2, "RFF-MCC")):
    ss = arr(c, mu, k, "ss_anr"); t5 = arr(c, mu, k, "t_-5dB", int); t10 = arr(c, mu, k, "t_-10dB", int)
    txt.append(f"  {lab:16s} ss {ss.mean():6.2f} dB, t(-5) {t5[t5>=0].mean():6.0f}, t(-10) "
               + (f"{t10[t10>=0].mean():6.0f} ({(t10>=0).sum()}/20)" if (t10 >= 0).any() else "never"))

# storage and arithmetic
D, M, Ls = 500, 20, 4
txt.append("\n== Storage and arithmetic per iteration (M=20, Ls=4)")
for lab, trn, fixed, mul in (("RFF-MCC D=500", 500, D * M + D, ac.mults_full(500, M, Ls)),
                             ("NKP-RFF-MCC 25x20 R=4", 180, D * M + D, ac.mults_kron(25, 20, 4, M, Ls)),
                             ("RFF-MCC D=180", 180, 180 * M + 180, ac.mults_full(180, M, Ls))):
    txt.append(f"  {lab:22s} trainable {trn:4d} | fixed map {fixed:6d} | total {trn + fixed:6d} | mults {mul:6d}")

# sweeps
for tag, fname in (("R sweep", "E6_rsweep_mu0.2_per_run.csv"), ("k sweep", "E5_ksweep_mu0.2_per_run.csv"),
                   ("ablation", "E7_ablation_mu0.2_per_run.csv"), ("init", "E8_init_mu0.2_per_run.csv")):
    rr = list(csv.DictReader(open(RES / fname)))
    g = defaultdict(list)
    for r in rr:
        g[r["ctrl"]].append(float(r["ss_anr"]))
    txt.append(f"\n== {tag} (E1, mu=0.2): " + "; ".join(f"{k} {np.mean(v):.2f}+-{np.std(v, ddof=1):.2f}" for k, v in g.items()))

(RES / "summary_stats.txt").write_text("\n".join(txt) + "\n")
print("\n".join(txt))


# ---------------------------------------------------------------- figures
def load(tag):
    z = np.load(DATA / f"{tag}.npz")
    return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# Fig: learning curves at the operating point, 2x2
# y-limits exclude the first samples, where d(n) = 0 because of the primary-path delay
# and the nonzero initial factors of NKP-RFF-MCC make ANR(n) numerically large.
YLIM = {"E1_chaotic": (-18, 3), "E2_sas_ref": (-4, 1), "E3_gauss": (-14, 2), "E4_sas_noise": (-13, 2)}
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
for ax, c in zip(axs.flat, CONDS):
    anr, meta = load(f"{c}_mu{main_mu[c]}")
    T = anr["RFF-MCC"].shape[1]
    n = np.arange(T)
    for k in CTRLS:
        m, s = anr[k].mean(0), anr[k].std(0, ddof=1)
        ax.plot(n, m, lw=1.0, label=DISPLAY[k], **STYLE[k])
        ax.fill_between(n, m - s, m + s, color=STYLE[k]["color"], alpha=0.12, lw=0)
    ax.set_title(f"{LABEL[c]}, $\\mu$ = {main_mu[c]}", fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ANR (dB)")
    ax.grid(alpha=0.3, lw=0.4)
    ax.set_xlim(0, T)
    ax.set_ylim(*YLIM[c])
axs[0, 0].legend(loc="upper right", frameon=False)
save(fig, "fig_learning_curves")

# Fig: steady state vs step size
fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.3), constrained_layout=True)
for ax, c in zip(axs, CONDS):
    for k in CTRLS:
        m = [arr(c, mu, k, "ss_anr").mean() for mu in MUS]
        s = [arr(c, mu, k, "ss_anr").std(ddof=1) for mu in MUS]
        ax.errorbar(MUS, m, yerr=s, marker="o", ms=2.5, lw=0.9, capsize=1.5, label=DISPLAY[k],
                    color=STYLE[k]["color"], ls=STYLE[k]["ls"])
    ax.set_xscale("log")
    ax.set_xticks(MUS)
    ax.set_xticklabels([str(m) for m in MUS])
    ax.set_title(LABEL[c].replace(" + ", " +\n"), fontsize=7)
    ax.set_xlabel(r"Effective step size $\mu$ ($\mu_a=\mu_b=\mu/2$)", fontsize=7)
    ax.grid(alpha=0.3, lw=0.4)
axs[0].set_ylabel("Steady-state ANR (dB)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
save(fig, "fig_steady_vs_mu")

# Fig: R sweep and arithmetic
rr = list(csv.DictReader(open(RES / "E6_rsweep_mu0.2_per_run.csv")))
g = defaultdict(list)
for r in rr:
    g[r["ctrl"]].append(float(r["ss_anr"]))
Rs = [1, 2, 4, 8, 12, 20]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.4), constrained_layout=True)
a1.errorbar(Rs, [np.mean(g[f"NKP-RFF-MCC_R{R}"]) for R in Rs], yerr=[np.std(g[f"NKP-RFF-MCC_R{R}"], ddof=1) for R in Rs],
            marker="o", ms=3, color=STYLE["NKP-RFF-MCC"]["color"], capsize=2, lw=1, label="NKP-RFF-MCC")
a1.axhline(np.mean(g["RFF-MCC"]), **{k: v for k, v in STYLE["RFF-MCC"].items()}, lw=1, label="RFF-MCC (D=500)")
a1.set_xlabel("Number of Kronecker terms R")
a1.set_ylabel("Steady-state ANR (dB)")
a1.set_title(r"Logistic-chaotic reference, $\mu$ = 0.2", fontsize=8)
a1.grid(alpha=0.3, lw=0.4)
a1.legend(frameon=False)
Rg = np.arange(1, 21)
a2.plot(Rg, [ac.mults_kron(25, 20, R, 20, 4) / 1e3 for R in Rg], color=STYLE["NKP-RFF-MCC"]["color"], lw=1,
        label="NKP-RFF-MCC (25x20)")
a2.axhline(ac.mults_full(500, 20, 4) / 1e3, color=STYLE["RFF-MCC"]["color"], ls="--", lw=1, label="RFF-MCC (D=500)")
a2.axhline(ac.mults_full(180, 20, 4) / 1e3, color=STYLE["RFF-MCC-180"]["color"], ls=":", lw=1, label="RFF-MCC (D=180)")
a2.set_xlabel("Number of Kronecker terms R")
a2.set_ylabel("Multiplications per iteration (x1000)")
a2.set_title("Arithmetic cost (includes output y(n))", fontsize=8)
a2.grid(alpha=0.3, lw=0.4)
a2.legend(frameon=False)
save(fig, "fig_R_sweep_cost")

# Fig: k sweep
rr = list(csv.DictReader(open(RES / "E5_ksweep_mu0.2_per_run.csv")))
g = defaultdict(list)
for r in rr:
    g[r["ctrl"]].append(float(r["ss_anr"]))
ks = [1, 1.5, 2, 2.8, 4, 5.6, 8, 12, 20]
fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
for base in ("RFF-MCC", "NKP-RFF-MCC"):
    ax.errorbar(ks, [np.mean(g[f"{base}_k{k}"]) for k in ks], yerr=[np.std(g[f"{base}_k{k}"], ddof=1) for k in ks],
                marker="o", ms=3, capsize=2, lw=1, color=STYLE[base]["color"], ls=STYLE[base]["ls"], label=base)
ax.set_xlabel(r"Correntropy parameter $k$ ($\sigma_{c,\mathrm{eff}} = 2/\sqrt{k}$)")
ax.set_ylabel("Steady-state ANR (dB)")
ax.set_title(r"Logistic-chaotic reference, $\mu$ = 0.2", fontsize=8)
ax.grid(alpha=0.3, lw=0.4)
ax.legend(frameon=False)
save(fig, "fig_k_sweep")
print("figures written to", FIG)
