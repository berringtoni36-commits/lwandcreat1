"""Numerical verification of the algebra used in the revised Sections 3-4.

Run:  python3 experiments_v10/test_math.py   (writes logs/test_math.txt)
Every check prints PASS/FAIL with the largest deviation found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anc_core as ac  # noqa: E402

LOG = Path(__file__).resolve().parent / "logs" / "test_math.txt"
lines = []


def report(name, ok, detail):
    s = f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"
    print(s)
    lines.append(s)


def vec(X):
    return X.flatten(order="F")


rng = np.random.default_rng(20260926)

# T1 identities ---------------------------------------------------------------
D1, D2, R = 5, 4, 3
A = rng.standard_normal((D1, R)); B = rng.standard_normal((D2, R))
q = rng.standard_normal(D1 * D2); Q = q.reshape(D1, D2).T
w_kron = sum(np.kron(A[:, r], B[:, r]) for r in range(R))
w_vec = vec(B @ A.T)
ub, ua = vec(Q @ A), vec(Q.T @ B)
dev = max(np.abs(w_kron - w_vec).max(),
          abs(vec(B) @ ub - w_vec @ q), abs(vec(A) @ ua - w_vec @ q),
          np.abs(vec(Q) - q).max())
report("T1 w=sum a_r(x)b_r=vec(BA^T), b^T u_b = a^T u_a = w^T q", dev < 1e-12, f"max dev {dev:.2e}")

# T2 sequential update is an exact two-stage preconditioned update -------------
mu_a, mu_b, sig, eps = 0.3, 0.7, 2.0, 1e-8
worst = 0.0
for _ in range(2000):
    A = rng.standard_normal((D1, R)) * rng.uniform(0.01, 2)
    B = rng.standard_normal((D2, R)) * rng.uniform(0.01, 2)
    q = rng.standard_normal(D1 * D2); Q = q.reshape(D1, D2).T
    dv = rng.standard_normal() * 3
    w0 = vec(B @ A.T)
    UB = Q @ A; e0 = dv - np.sum(B * UB); psi0, _ = ac.influence(e0, sig)
    cb = mu_b / (eps + np.sum(UB * UB))
    B1 = B + cb * psi0 * UB
    w1 = vec(B1 @ A.T)
    P_b = cb * np.kron(A @ A.T, np.eye(D2))
    UA = Q.T @ B1; e1 = dv - np.sum(A * UA); psi1, _ = ac.influence(e1, sig)
    ca = mu_a / (eps + np.sum(UA * UA))
    A1 = A + ca * psi1 * UA
    w2 = vec(B1 @ A1.T)
    P_a = ca * np.kron(np.eye(D1), B1 @ B1.T)
    worst = max(worst, np.abs((w1 - w0) - psi0 * P_b @ q).max(), np.abs((w2 - w1) - psi1 * P_a @ q).max())
report("T2 sequential step: w1-w0 = psi0 P_b q, w2-w1 = psi1 P_a q (exact)", worst < 1e-9, f"max dev {worst:.2e}")

# T3 sequential a posteriori contraction for 0 < mu_a, mu_b < 2 ----------------
viol, worst_ratio, worst_formula = 0, 0.0, 0.0
N3 = 100000
for _ in range(N3):
    Dd1, Dd2, RR = rng.integers(1, 6), rng.integers(1, 6), rng.integers(1, 4)
    A = rng.standard_normal((Dd1, RR)) * 10 ** rng.uniform(-3, 1)
    B = rng.standard_normal((Dd2, RR)) * 10 ** rng.uniform(-3, 1)
    Q = rng.standard_normal((Dd2, Dd1)) * 10 ** rng.uniform(-2, 1)
    dv = rng.standard_normal() * 10 ** rng.uniform(-2, 1)
    ma, mb = rng.uniform(1e-3, 1.999, 2)
    sg = 10 ** rng.uniform(-1, 1)
    UB = Q @ A; e0 = dv - np.sum(B * UB); psi0, g0 = ac.influence(e0, sg)
    nb = np.sum(UB * UB); B1 = B + mb * psi0 / (eps + nb) * UB
    e1 = dv - np.sum(B1 * UB)
    f1 = (1 - g0 * mb * nb / (eps + nb)) * e0
    UA = Q.T @ B1; psi1, g1 = ac.influence(e1, sg); na = np.sum(UA * UA)
    A1 = A + ma * psi1 / (eps + na) * UA
    e2 = dv - np.sum(B1 * (Q @ A1))
    f2 = (1 - g1 * ma * na / (eps + na)) * e1
    scale = max(1.0, abs(e0))
    worst_formula = max(worst_formula, abs(e1 - f1) / scale, abs(e2 - f2) / scale)
    if abs(e0) > 0:
        ratio = abs(e2) / abs(e0)
        worst_ratio = max(worst_ratio, ratio)
        if ratio > 1 + 1e-9:
            viol += 1
report("T3a sequential residual factors e1=(1-g0 mu_b rho_b)e0, e2=(1-g1 mu_a rho_a)e1", worst_formula < 1e-8,
       f"max rel dev {worst_formula:.2e} over {N3} random cases")
report("T3b sequential |e2| <= |e0| for 0<mu_a,mu_b<2", viol == 0,
       f"violations {viol}/{N3}, max |e2|/|e0| = {worst_ratio:.6f}")

# T4 simultaneous update: exact cross term and conditional bound ---------------
viol_plain, viol_bound, N4 = 0, 0, 100000
worst_x = 0.0
for _ in range(N4):
    Dd1, Dd2, RR = rng.integers(1, 6), rng.integers(1, 6), rng.integers(1, 4)
    A = rng.standard_normal((Dd1, RR)) * 10 ** rng.uniform(-3, 1)
    B = rng.standard_normal((Dd2, RR)) * 10 ** rng.uniform(-3, 1)
    Q = rng.standard_normal((Dd2, Dd1)) * 10 ** rng.uniform(-2, 1)
    dv = rng.standard_normal() * 10 ** rng.uniform(-2, 1)
    ma = mb = rng.uniform(1e-3, 0.999)          # mu_a + mu_b < 2
    sg = 10 ** rng.uniform(-1, 1)
    UB, UA = Q @ A, Q.T @ B
    e = dv - np.sum(B * UB); psi, g = ac.influence(e, sg)
    nb, na = np.sum(UB * UB), np.sum(UA * UA)
    dB = mb * psi / (eps + nb) * UB; dA = ma * psi / (eps + na) * UA
    ep = dv - np.sum((B + dB) * (Q @ (A + dA)))
    xg = g * (ma * na / (eps + na) + mb * nb / (eps + nb))
    cross = np.trace(dB.T @ Q @ dA)
    formula = (1 - xg) * e - cross
    worst_x = max(worst_x, abs(ep - formula) / max(1.0, abs(e)))
    kappa = ma * mb * g * g * abs(e) * np.linalg.norm(Q, 2) * np.sqrt(na * nb) / ((eps + na) * (eps + nb))
    if abs(ep) > (abs(1 - xg) + kappa) * abs(e) * (1 + 1e-9) + 1e-300:
        viol_bound += 1
    if abs(ep) > abs(e) * (1 + 1e-9):
        viol_plain += 1
report("T4a simultaneous: e+ = (1-x)e - tr(dB^T Q dA) (exact)", worst_x < 1e-8, f"max rel dev {worst_x:.2e}")
report("T4b simultaneous: |e+| <= (|1-x| + kappa)|e| (conditional bound)", viol_bound == 0,
       f"violations {viol_bound}/{N4}")
report("T4c simultaneous with mu_a+mu_b<2 is NOT always contracting", viol_plain > 0,
       f"expansions found in {viol_plain}/{N4} random cases (so no unconditional guarantee)")

# T5 reviewer counterexample ---------------------------------------------------
Q = np.array([[1.0]]); A = np.array([[0.01]]); B = np.array([[0.01]]); dv = 1.0
ma = mb = 0.08; sg = 2.0
UB, UA = Q @ A, Q.T @ B
e0 = dv - np.sum(B * UB); psi, g = ac.influence(e0, sg)
dB = mb * psi / (eps + np.sum(UB * UB)) * UB; dA = ma * psi / (eps + np.sum(UA * UA)) * UA
ep = dv - np.sum((B + dB) * (Q @ (A + dA)))
report("T5a counterexample, simultaneous update",
       abs(e0 - 0.9999) < 1e-12 and abs(dA[0, 0] - 7.05874) < 1e-4 and abs(ep + 48.967) < 1e-3,
       f"e0={e0:.4f}, dA=dB={dA[0,0]:.5f}, e+={ep:.3f}")
B1 = B + dB
e1 = dv - np.sum(B1 * UB)
UA1 = Q.T @ B1; psi1, g1 = ac.influence(e1, sg)
A1 = A + ma * psi1 / (eps + np.sum(UA1 * UA1)) * UA1
e2 = dv - np.sum(B1 * (Q @ A1))
report("T5b same data, sequential update", abs(e2) < abs(e1) < abs(e0),
       f"e0={e0:.4f} -> e1={e1:.4f} -> e2={e2:.4f}")

# T6 controller class = formulas ----------------------------------------------
runs, D1, D2, R = 3, 5, 4, 2
crng = np.random.default_rng(1)
ctrl = ac.KronRFF("k", runs, D1, D2, R, 0.2, 0.3, 2.0, eps, crng, init_scale=0.5)
A0, B0 = ctrl.A.copy(), ctrl.B.copy()
z = rng.standard_normal((runs, D1 * D2)); q = rng.standard_normal((runs, D1 * D2)); dv = rng.standard_normal(runs)
ctrl.step(z, q, dv)
dev = 0.0
for r in range(runs):
    Qr = q[r].reshape(D1, D2).T
    UB = Qr @ A0[r]; e0 = dv[r] - np.sum(B0[r] * UB); p0, _ = ac.influence(e0, 2.0)
    B1 = B0[r] + 0.3 * p0 / (eps + np.sum(UB * UB)) * UB
    UA = Qr.T @ B1; e1 = dv[r] - np.sum(A0[r] * UA); p1, _ = ac.influence(e1, 2.0)
    A1 = A0[r] + 0.2 * p1 / (eps + np.sum(UA * UA)) * UA
    dev = max(dev, np.abs(A1 - ctrl.A[r]).max(), np.abs(B1 - ctrl.B[r]).max())
report("T6 KronRFF.step (batched) equals the per-run formulas", dev < 1e-12, f"max dev {dev:.2e}")

# T7 Wilcoxon exact p ---------------------------------------------------------
_, p5, _ = ac.wilcoxon_signed_rank_exact([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
_, p20, _ = ac.wilcoxon_signed_rank_exact(np.arange(1, 21), np.zeros(20))
report("T7 exact Wilcoxon p-values", abs(p5 - 2 / 32) < 1e-12 and abs(p20 - 2 / 2 ** 20) < 1e-15,
       f"p(n=5, all +) = {p5:.6f} (exact 0.0625); p(n=20, all +) = {p20:.3e} (exact {2/2**20:.3e})")

# T8 arithmetic counts ----------------------------------------------------------
full = ac.mults_full(500, 20, 4)
kron = ac.mults_kron(25, 20, 4, 20, 4)
report("T8 multiplication counts D=500, M=20, Ls=4, 25x20, R=4", True,
       f"full {full}, structured {kron}, ratio {kron/full:.3f} (old Table 2: 14,004 vs 17,048 omitted the output y(n))")

LOG.parent.mkdir(exist_ok=True)
LOG.write_text("\n".join(lines) + "\n")
print(f"log written to {LOG}")
