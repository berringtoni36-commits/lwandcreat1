"""Compiled, numerically equivalent 45-branch fixed frontier for N14 dev only."""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
import numpy as np
DEPS = Path(tempfile.gettempdir()) / "n13_frontier_deps"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
from numba import njit
N_METHODS = 45
@njit(cache=True)
def simulate_fixed(x, d, v, omega, phase, a0, b0, mu, stage_ends, tail):
    """All 8 x 5 nested-factor NKP-MCC and 5 full500 MCC controls.

    Factor updates are B then A, and the microphone uses S=[0,0,1,.5].
    Arrays of every actuator sample and physical error permit FIR replay.
    """
    tmax = x.size
    a = np.zeros((40, 25, 8), np.float64)
    b = np.zeros((40, 20, 8), np.float64)
    for c in range(40):
        for i in range(25):
            for r in range(8):
                a[c, i, r] = a0[i, r]
        for j in range(20):
            for r in range(8):
                b[c, j, r] = b0[j, r]
    w = np.zeros((5, 500), np.float64)
    xb = np.zeros(20, np.float64)
    z = np.zeros(500, np.float64)
    p1 = np.zeros(500, np.float64)
    p2 = np.zeros(500, np.float64)
    p3 = np.zeros(500, np.float64)
    q = np.zeros(500, np.float64)
    y1 = np.zeros(N_METHODS, np.float64)
    y2 = np.zeros(N_METHODS, np.float64)
    y3 = np.zeros(N_METHODS, np.float64)
    ae = np.zeros(N_METHODS, np.float64)
    ad = 0.0
    y_trace = np.empty((N_METHODS, tmax), np.float32)
    e_trace = np.empty((N_METHODS, tmax), np.float32)
    anr_sum = np.zeros((N_METHODS, 4), np.float64)
    anr_count = np.zeros(4, np.int64)
    ub = np.zeros((20, 8), np.float64)
    ua = np.zeros((25, 8), np.float64)
    scale = np.sqrt(2.0 / 500.0)
    for n in range(tmax):
        for k in range(19, 0, -1):
            xb[k] = xb[k - 1]
        xb[0] = x[n]
        for j in range(500):
            dot = phase[j]
            for k in range(20):
                dot += omega[j, k] * xb[k]
            z[j] = scale * np.cos(dot)
            q[j] = p2[j] + 0.5 * p3[j]
        dv = d[n] + v[n]
        ad = 0.999 * ad + 0.001 * abs(d[n])
        stage = 0 if n < stage_ends[0] else 1 if n < stage_ends[1] else 2 if n < stage_ends[2] else 3
        in_tail = n >= stage_ends[stage] - tail
        if in_tail:
            anr_count[stage] += 1
        for c in range(40):
            rank = c // 5 + 1
            m = c % 5
            output = 0.0
            pred1 = 0.0
            norm_b = 0.0
            for r in range(rank):
                for j in range(20):
                    zj = 0.0
                    qj = 0.0
                    for i in range(25):
                        index = 20 * i + j
                        zj += z[index] * a[c, i, r]
                        qj += q[index] * a[c, i, r]
                    ub[j, r] = qj
                    output += b[c, j, r] * zj
                    pred1 += b[c, j, r] * qj
                    norm_b += qj * qj
            ys = y2[c] + 0.5 * y3[c]
            err = dv - ys
            y_trace[c, n] = output
            e_trace[c, n] = err
            ae[c] = 0.999 * ae[c] + 0.001 * abs(err)
            if in_tail:
                anr_sum[c, stage] += 20.0 * np.log10((ae[c] + 1e-12) / (ad + 1e-12))
            y3[c] = y2[c]
            y2[c] = y1[c]
            y1[c] = output
            residual1 = dv - pred1
            psi1 = residual1 * np.exp(-residual1 * residual1 / 8.0)
            gain_b = (mu[m] / 2.0) * psi1 / (1e-8 + norm_b)
            for r in range(rank):
                for j in range(20):
                    b[c, j, r] += gain_b * ub[j, r]
            pred2 = 0.0
            norm_a = 0.0
            for r in range(rank):
                for i in range(25):
                    qi = 0.0
                    for j in range(20):
                        qi += q[20 * i + j] * b[c, j, r]
                    ua[i, r] = qi
                    pred2 += a[c, i, r] * qi
                    norm_a += qi * qi
            residual2 = dv - pred2
            psi2 = residual2 * np.exp(-residual2 * residual2 / 8.0)
            gain_a = (mu[m] / 2.0) * psi2 / (1e-8 + norm_a)
            for r in range(rank):
                for i in range(25):
                    a[c, i, r] += gain_a * ua[i, r]
        normq = 0.0
        for j in range(500):
            normq += q[j] * q[j]
        for m in range(5):
            c = 40 + m
            output = 0.0
            prediction = 0.0
            for j in range(500):
                output += w[m, j] * z[j]
                prediction += w[m, j] * q[j]
            ys = y2[c] + 0.5 * y3[c]
            err = dv - ys
            y_trace[c, n] = output
            e_trace[c, n] = err
            ae[c] = 0.999 * ae[c] + 0.001 * abs(err)
            if in_tail:
                anr_sum[c, stage] += 20.0 * np.log10((ae[c] + 1e-12) / (ad + 1e-12))
            y3[c] = y2[c]
            y2[c] = y1[c]
            y1[c] = output
            residual = dv - prediction
            psi = residual * np.exp(-residual * residual / 8.0)
            gain = mu[m] * psi / (1e-8 + normq)
            for j in range(500):
                w[m, j] += gain * q[j]
        for j in range(500):
            p3[j] = p2[j]
            p2[j] = p1[j]
            p1[j] = z[j]
    return anr_sum, anr_count, y_trace, e_trace


