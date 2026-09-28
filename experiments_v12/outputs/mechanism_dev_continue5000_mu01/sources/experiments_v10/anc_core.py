"""Reference implementation of the Kronecker-structured RFF correntropy ANC controller.

Unified conventions (fixed for every experiment of the v10 revision):

* Signal flow: the RFF map is applied to the reference vector first and the
  secondary path then filters every feature sequence, q_n = sum_l s_l z_{n-l}
  (manuscript eq. (7)).  The old scripts FIG6/FIG7/FIG10/untitled.m instead
  filtered the raw reference and then mapped it; that flow is NOT used here.
* Adaptation residual: e_n = d(n) + v(n) - yhat_n with yhat_n = w_n^T q_n
  (full controllers) or b_n^T u_{b,n} (structured controller).  d(n) + v(n) is
  what a modified filtered-x structure reconstructs from the measured error
  and the secondary-path model output; the model is exact in these simulations.
* Physical error: e(n) = d(n) - sum_l s_l y(n-l) + v(n), used only for ANR.
* ANR (manuscript eq. (20)): 20 log10 of exponentially smoothed absolute
  amplitudes, ANR(n) = 20 log10((A_e(n) + eps_a) / (A_d(n) + eps_a)),
  negative values = attenuation.
* Structured controller: factors B (D2 x R) and A (D1 x R), w = vec(B A^T).
  The update is SEQUENTIAL: B is updated first, then the regressor of A is
  recomputed with the updated B, and a fresh residual is used for A.  This is
  the version analyzed in the revised Section 4 (exact preconditioned form and
  exact a posteriori contraction for 0 < mu_a, mu_b < 2).  A simultaneous
  variant is kept only for the counterexample and the ablation test.

All controllers of one experiment run in lockstep over the same batch of
independent Monte Carlo realizations (leading array axis = run index).
"""
from __future__ import annotations

import numpy as np

P_PATH = np.array([0.0, 0.0, 0.0, 1.0, -0.3, 0.2])  # manuscript eq. (3)
S_PATH = np.array([0.0, 0.0, 1.0, 0.5])             # manuscript eq. (3), minimum phase


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------
def fir(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Causal FIR filtering along the last axis (zero initial state)."""
    y = np.zeros_like(x, dtype=float)
    T = x.shape[-1]
    for k, hk in enumerate(h):
        if hk != 0.0:
            y[..., k:] += hk * x[..., : T - k]
    return y


def primary_disturbance(x: np.ndarray) -> np.ndarray:
    """d(n) = g(n-2) + 0.08 g^2(n-2) - 0.04 g^3(n-1), g = P(z) x (eq. (4))."""
    g = fir(P_PATH, x)
    d = np.zeros_like(g)
    d[..., 2:] = g[..., :-2] + 0.08 * g[..., :-2] ** 2 - 0.04 * g[..., 1:-1] ** 3
    return d


def logistic_delay6(rng: np.random.Generator, runs: int, T: int) -> np.ndarray:
    """u(n) = 4 u(n-6) [1 - u(n-6)], six random initial states per run,
    centered and normalized to unit sample standard deviation per run."""
    while True:
        u = np.empty((runs, T))
        u[:, :6] = rng.uniform(0.1, 0.9, size=(runs, 6))
        for n in range(6, T):
            p = u[:, n - 6]
            u[:, n] = 4.0 * p * (1.0 - p)
        # guard against a sub-sequence collapsing onto the fixed point 0
        tail = u[:, -600:]
        ok = all(np.std(tail[r, k::6]) > 1e-3 for r in range(runs) for k in range(6))
        if ok:
            break
    u -= u.mean(axis=1, keepdims=True)
    u /= u.std(axis=1, keepdims=True)
    return u


def sas_cms(rng: np.random.Generator, alpha: float, size) -> np.ndarray:
    """Symmetric alpha-stable samples (beta = 0, unit scale), Chambers-Mallows-Stuck."""
    U = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
    W = rng.exponential(1.0, size=size)
    if abs(alpha - 1.0) < 1e-12:
        return np.tan(U)
    return (np.sin(alpha * U) / np.cos(U) ** (1.0 / alpha)
            * (np.cos((1.0 - alpha) * U) / W) ** ((1.0 - alpha) / alpha))


def alpha_stable_reference(rng, runs, T, alpha=1.6, clip=5.0):
    """Median-amplitude normalized, clipped alpha-stable reference (Table 3)."""
    s = sas_cms(rng, alpha, (runs, T))
    s /= np.median(np.abs(s), axis=1, keepdims=True)
    return np.clip(s, -clip, clip)


# ----------------------------------------------------------------------------
# Random Fourier features
# ----------------------------------------------------------------------------
def draw_rff(rng, runs, D, M, ell):
    """Independent RFF map per run: omega_j ~ N(0, ell^-2 I_M), b_j ~ U[0, 2pi)."""
    Om = rng.normal(0.0, 1.0 / ell, size=(runs, D, M))
    ph = rng.uniform(0.0, 2 * np.pi, size=(runs, D))
    return Om, ph


def influence(e, sigma):
    """psi(e) = e exp(-e^2/(2 sigma^2)); sigma=None gives psi(e) = e (NLMS)."""
    if sigma is None:
        return e, np.ones_like(e)
    g = np.exp(-(e * e) / (2.0 * sigma * sigma))
    return e * g, g


# ----------------------------------------------------------------------------
# Controllers
# ----------------------------------------------------------------------------
class FullRFF:
    """RFF-NLMS (sigma=None) or RFF-MCC: normalized update of the full vector, eq. (9)."""

    def __init__(self, name, runs, D, mu, sigma, eps):
        self.name, self.mu, self.sigma, self.eps = name, mu, sigma, eps
        self.w = np.zeros((runs, D))
        self.ybuf = np.zeros((runs, len(S_PATH)))
        self.n_trainable = D

    def step(self, z, q, dv):
        y = np.einsum("rd,rd->r", self.w, z)
        self.ybuf[:, 1:] = self.ybuf[:, :-1]
        self.ybuf[:, 0] = y
        ys = self.ybuf @ S_PATH
        e = dv - np.einsum("rd,rd->r", self.w, q)          # adaptation residual
        psi, _ = influence(e, self.sigma)
        nq = np.einsum("rd,rd->r", q, q)
        self.w += (self.mu * psi / (self.eps + nq))[:, None] * q
        return ys


class KronRFF:
    """NKP-RFF-MCC: w = vec(B A^T), B: D2 x R, A: D1 x R.

    sequential=True  -> B update, then regressor/residual of A recomputed (default).
    sequential=False -> both regressors from (A_n, B_n), one residual (ablation).
    """

    def __init__(self, name, runs, D1, D2, R, mu_a, mu_b, sigma, eps, rng,
                 init_scale=0.01, sequential=True):
        self.name, self.D1, self.D2, self.R = name, D1, D2, R
        self.mu_a, self.mu_b, self.sigma, self.eps = mu_a, mu_b, sigma, eps
        self.sequential = sequential
        self.A = init_scale * rng.standard_normal((runs, D1, R))
        self.B = init_scale * rng.standard_normal((runs, D2, R))
        self.ybuf = np.zeros((runs, len(S_PATH)))
        self.n_trainable = R * (D1 + D2)

    def mat(self, v):
        """Columnwise reshaping: v = vec(V), V in R^{D2 x D1}."""
        return v.reshape(v.shape[0], self.D1, self.D2).transpose(0, 2, 1)

    def step(self, z, q, dv):
        A, B = self.A, self.B
        Z, Q = self.mat(z), self.mat(q)
        y = np.einsum("rjk,rjk->r", B, Z @ A)               # y(n) = w^T z_n
        self.ybuf[:, 1:] = self.ybuf[:, :-1]
        self.ybuf[:, 0] = y
        ys = self.ybuf @ S_PATH
        UB = Q @ A                                          # mat(u_b)
        e1 = dv - np.einsum("rjk,rjk->r", B, UB)
        psi1, _ = influence(e1, self.sigma)
        nb = np.einsum("rjk,rjk->r", UB, UB)
        if self.sequential:
            B = B + (self.mu_b * psi1 / (self.eps + nb))[:, None, None] * UB
            UA = Q.transpose(0, 2, 1) @ B                   # mat(u_a) with updated B
            e2 = dv - np.einsum("rjk,rjk->r", A, UA)
            psi2, _ = influence(e2, self.sigma)
            na = np.einsum("rjk,rjk->r", UA, UA)
            A = A + (self.mu_a * psi2 / (self.eps + na))[:, None, None] * UA
        else:
            UA = Q.transpose(0, 2, 1) @ B
            na = np.einsum("rjk,rjk->r", UA, UA)
            B = B + (self.mu_b * psi1 / (self.eps + nb))[:, None, None] * UB
            A = A + (self.mu_a * psi1 / (self.eps + na))[:, None, None] * UA
        self.A, self.B = A, B
        return ys


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate(x, d, v, controllers, Om, ph, beta=0.999, eps_a=1e-12, record_every=1):
    """Run all controllers over the same realizations; return ANR curves (dB).

    x, d, v: (runs, T) reference, primary disturbance, measurement noise.
    Returns dict name -> ANR array (runs, T // record_every).
    """
    runs, T = x.shape
    D, M = Om.shape[1], Om.shape[2]
    sc = np.sqrt(2.0 / D)
    s = S_PATH
    Ls = len(s)
    xb = np.zeros((runs, M))
    zh = np.zeros((Ls, runs, D))
    Ad = np.zeros(runs)
    Ae = {c.name: np.zeros(runs) for c in controllers}
    out = {c.name: np.zeros((runs, T // record_every), dtype=np.float32) for c in controllers}
    for n in range(T):
        xb[:, 1:] = xb[:, :-1]
        xb[:, 0] = x[:, n]
        z = sc * np.cos(np.einsum("rdm,rm->rd", Om, xb) + ph)
        zh[1:] = zh[:-1]
        zh[0] = z
        q = s[2] * zh[2] + s[3] * zh[3]                      # s_0 = s_1 = 0
        dn, vn = d[:, n], v[:, n]
        dv = dn + vn
        Ad = beta * Ad + (1 - beta) * np.abs(dn)
        for c in controllers:
            ys = c.step(z, q, dv)
            e_phys = dn - ys + vn
            Ae[c.name] = beta * Ae[c.name] + (1 - beta) * np.abs(e_phys)
            if n % record_every == 0:
                out[c.name][:, n // record_every] = 20 * np.log10((Ae[c.name] + eps_a) / (Ad + eps_a))
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def steady_state(anr, window):
    return anr[:, -window:].mean(axis=1)


def first_crossing(anr, tau):
    """First index with ANR <= tau per run; -1 if never (censored)."""
    hit = anr <= tau
    idx = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    return idx


def wilcoxon_signed_rank_exact(x, y):
    """Two-sided exact Wilcoxon signed-rank test for paired samples (no scipy).

    Zero differences are dropped; ties get average ranks and the exact null
    distribution is computed on doubled ranks by dynamic programming.
    Returns (W_plus, p_value, n_used)."""
    dlt = np.asarray(x, float) - np.asarray(y, float)
    dlt = dlt[dlt != 0]
    n = len(dlt)
    if n == 0:
        return 0.0, 1.0, 0
    absd = np.abs(dlt)
    order = np.argsort(absd)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[order[j + 1]] == absd[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r2 = np.rint(2 * ranks).astype(int)
    total = int(r2.sum())
    dist = np.zeros(total + 1)
    dist[0] = 1.0
    for rk in r2:
        new = dist.copy()
        new[rk:] += dist[: total + 1 - rk]
        dist = new
    dist /= dist.sum()
    wplus2 = int(r2[dlt > 0].sum())
    mean2 = total / 2.0
    dev = abs(wplus2 - mean2)
    ks = np.arange(total + 1)
    p = float(dist[np.abs(ks - mean2) >= dev - 1e-9].sum())
    return wplus2 / 2.0, min(1.0, p), n


# ----------------------------------------------------------------------------
# Arithmetic cost (real multiplications per iteration)
# ----------------------------------------------------------------------------
def mults_full(D, M, Ls):
    """Features (DM projection + D scaling), filtering D*Ls, output w^T z (D),
    residual prediction w^T q (D), ||q||^2 (D), increment (D), 4 scalar."""
    return D * M + D + D * Ls + 4 * D + 4


def mults_kron(D1, D2, R, M, Ls):
    """Features, filtering, output y = tr(B^T Z A) (R D + R D2),
    u_b = vec(Q A) (R D), u_a = vec(Q^T B) (R D), predictions R(D1+D2),
    squared norms R(D1+D2), increments R(D1+D2), 8 scalar."""
    D = D1 * D2
    return D * M + D + D * Ls + (R * D + R * D2) + 2 * R * D + 3 * R * (D1 + D2) + 8
