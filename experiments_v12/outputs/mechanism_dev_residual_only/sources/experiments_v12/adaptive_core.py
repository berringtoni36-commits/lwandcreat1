"""Experimental online term selection around the unchanged v10 controller.

One manager per Monte Carlo run: decisions NEVER pool data across runs.
Costs are reference-model multiplication counts, not hardware instruction counts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments_v10"))
import anc_core as ac


@dataclass(frozen=True)
class SelectionConfig:
    r_min: int = 1
    r_max: int = 8
    warmup: int = 2000
    train_samples: int = 1000
    validation_samples: int = 1000
    ramp_samples: int = 100
    cooldown: int = 2000
    screen_every: int = 50
    screen_beta: float = 0.9
    lambda_cost: float = 0.01
    margin: float = 0.0005
    enabled: bool = True

    def __post_init__(self):
        if not 1 <= self.r_min <= self.r_max:
            raise ValueError("Require 1 <= r_min <= r_max")
        for key in ("warmup", "train_samples", "validation_samples", "ramp_samples",
                    "cooldown", "screen_every"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.train_samples < len(ac.S_PATH) or self.cooldown < len(ac.S_PATH):
            raise ValueError("Train/cooldown must cover secondary-path history")
        if not 0 <= self.screen_beta < 1 or min(self.lambda_cost, self.margin) < 0:
            raise ValueError("Invalid smoothing or penalty")


def mcc_loss(error, sigma):
    """Dimensionless bounded loss, computed accurately near zero."""
    return -np.expm1(-0.5 * np.square(np.asarray(error) / sigma))


def resized_copy(ctrl, direction, rng, remove_index=None):
    """Warm-start candidate, preserving its pre-existing output history.

    Growth adds nonzero A and zero B: initially identical output, nonzero B
    gradient. Pruning history is washed out during the candidate training phase.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    candidate = deepcopy(ctrl)
    if direction == 1:
        # A tiny new factor beside already learned large factors gets almost no
        # share of the JOINT normalization. Match existing A-column RMS norm;
        # B=0 still makes the new term's initial output exactly zero.
        a = rng.standard_normal((1, ctrl.D1, 1))
        target_sq = float(np.sum(ctrl.A * ctrl.A) / ctrl.R)
        a *= np.sqrt(target_sq / (float(np.sum(a * a)) + ctrl.eps))
        candidate.A = np.concatenate((candidate.A, a), axis=2)
        candidate.B = np.concatenate((candidate.B, np.zeros((1, ctrl.D2, 1))), axis=2)
    else:
        if ctrl.R <= 1 or remove_index is None or not 0 <= remove_index < ctrl.R:
            raise ValueError("Invalid prune request")
        candidate.A = np.delete(candidate.A, remove_index, axis=2)
        candidate.B = np.delete(candidate.B, remove_index, axis=2)
    candidate.R += direction
    candidate.n_trainable = candidate.R * (ctrl.D1 + ctrl.D2)
    return candidate


class AdaptiveKron:
    """Sequential factors + alternating growth/prune candidate evaluations.

    A candidate adapts on a training window, then both branches continue online
    learning during a NEW prequential validation window. Scores use outputs
    produced before the current sample's update. Only the selected actuator
    output passes into actual_ybuf; shadow outputs have independent buffers.
    """

    def __init__(self, initial, rng, config=None, M=20):
        if initial.A.shape[0] != 1 or not initial.sequential or initial.sigma is None:
            raise ValueError("Requires one run of sequential KronRFF with MCC")
        self.config = config or SelectionConfig()
        if not self.config.r_min <= initial.R <= self.config.r_max:
            raise ValueError("Initial R outside bounds")
        self.name = "AKT-RFF-MCC"
        self.active = deepcopy(initial)
        self.candidate = None
        self.actual_ybuf = initial.ybuf.copy()
        self.rng, self.M = rng, M
        self.n = 0
        self.phase = "idle"
        self.age = 0
        self.next_probe = self.config.warmup
        self.next_direction = 1
        self.screen = np.zeros(initial.R)
        self.screen_count = 0
        self.events = []
        self.last = {}
        self.validation_sum = np.zeros(2)
        self.c_ref = ac.mults_full(initial.D1 * initial.D2, M, len(ac.S_PATH))

    @property
    def n_trainable(self):
        return self.active.n_trainable

    def cost(self, R):
        return ac.mults_kron(self.active.D1, self.active.D2, R, self.M, len(ac.S_PATH))

    @property
    def shared_cost(self):
        D = self.active.D1 * self.active.D2
        return D * (self.M + 1 + len(ac.S_PATH))

    def _screen(self, q, dv):
        # Screening is explicitly a prediction-residual surrogate, NOT the
        # physical deletion error. Final decisions use full shadow controllers.
        Q = self.active.mat(q)
        terms = np.sum(self.active.B * (Q @ self.active.A), axis=1)[0]
        residual = float(dv[0] - terms.sum())
        importance = mcc_loss(residual + terms, self.active.sigma) - mcc_loss(residual, self.active.sigma)
        if self.screen_count == 0:
            self.screen = importance
        else:
            b = self.config.screen_beta
            self.screen = b * self.screen + (1 - b) * importance
        self.screen_count += 1
        r = self.active.R
        D = self.active.D1 * self.active.D2
        # Q A, term inner products, two scalar multiplications per loss,
        # and two per EWMA element (also charged on first initialization).
        return r * D + r * self.active.D2 + 2 * (r + 1) + 2 * r, r + 1

    def _propose(self):
        cfg, r = self.config, self.active.R
        direction = self.next_direction
        if r == cfg.r_min:
            direction = 1
        elif r == cfg.r_max:
            direction = -1
        if cfg.r_min == cfg.r_max:
            self.next_probe = self.n + 1 + cfg.cooldown
            return 0
        if direction == -1 and self.screen_count == 0:
            self.next_probe = self.n + 1 + cfg.screen_every
            return 0
        removed = int(np.argmin(self.screen)) if direction == -1 else None
        self.candidate = resized_copy(self.active, direction, self.rng, removed)
        self.next_direction = -direction
        self.phase, self.age = "train", 0
        self.validation_sum[:] = 0
        self.events.append(dict(sample=self.n, event="proposal", old_R=r,
                                new_R=self.candidate.R, remove_index=removed))
        return self.active.D1 * (r + 2) if direction == 1 else 0

    def _decision(self):
        cfg = self.config
        losses = self.validation_sum / cfg.validation_samples
        costs = np.array([self.cost(self.active.R), self.cost(self.candidate.R)])
        scores = losses + cfg.lambda_cost * costs / self.c_ref
        accepted = bool(np.isfinite(scores).all() and scores[1] < scores[0] - cfg.margin)
        self.events.append(dict(sample=self.n, event="accept" if accepted else "reject",
                                old_R=self.active.R, new_R=self.candidate.R,
                                old_loss=float(losses[0]), new_loss=float(losses[1]),
                                old_score=float(scores[0]), new_score=float(scores[1]),
                                validation_samples=cfg.validation_samples))
        if accepted:
            self.phase, self.age = "ramp", 0
        else:
            self.candidate = None
            self.phase, self.age = "idle", 0
            self.next_probe = self.n + 1 + cfg.cooldown
        # Selection penalties compare prospective single-controller core costs.
        # Candidate search costs are separately charged in every sample's ledger.

    def step(self, z, q, dv):
        cfg = self.config
        old_r = self.active.R
        candidate_r = self.candidate.R if self.candidate is not None else 0
        phase_at_start = self.phase
        core = self.cost(old_r)
        shadow = self.cost(candidate_r) - self.shared_cost if candidate_r else 0
        # Reference core formula excludes each branch's output FIR. Also charge
        # a separate actual-actuator FIR. Zero taps count, as in v10 convention.
        management = len(ac.S_PATH) * (2 + bool(candidate_r))
        exponentials = 2 * (1 + bool(candidate_r))  # two sequential MCC updates
        if cfg.enabled and self.phase == "idle" and self.n % cfg.screen_every == 0:
            extra, exps = self._screen(q, dv)
            management += extra
            exponentials += exps

        ys_old = self.active.step(z, q, dv)
        y_old = self.active.ybuf[:, 0].copy()
        y_actual, gamma = y_old, 0.0
        if self.candidate is not None:
            ys_new = self.candidate.step(z, q, dv)
            y_new = self.candidate.ybuf[:, 0].copy()
            if phase_at_start == "train":
                self.age += 1
                if self.age >= cfg.train_samples:
                    self.phase, self.age = "validate", 0
                    self.events.append(dict(sample=self.n, event="validation_start"))
            elif phase_at_start == "validate":
                errors = np.array([dv[0] - ys_old[0], dv[0] - ys_new[0]])
                self.validation_sum += mcc_loss(errors, self.active.sigma)
                management += 4
                exponentials += 2
                self.age += 1
                if self.age >= cfg.validation_samples:
                    self._decision()
                    management += 2  # two cost-penalty multiplications
            elif phase_at_start == "ramp":
                self.age += 1
                gamma = self.age / cfg.ramp_samples
                y_actual = (1 - gamma) * y_old + gamma * y_new
                management += 2
                if self.age >= cfg.ramp_samples:
                    self.active = self.candidate
                    self.candidate = None
                    self.phase, self.age = "idle", 0
                    self.next_probe = self.n + 1 + cfg.cooldown
                    self.screen = np.zeros(self.active.R)
                    self.screen_count = 0
                    self.events.append(dict(sample=self.n, event="switch_complete",
                                            old_R=old_r, new_R=self.active.R))

        self.actual_ybuf[:, 1:] = self.actual_ybuf[:, :-1].copy()
        self.actual_ybuf[:, 0] = y_actual
        ys_actual = self.actual_ybuf @ ac.S_PATH
        if cfg.enabled and phase_at_start == "idle" and self.n + 1 >= self.next_probe:
            management += self._propose()
        resident_r = max(old_r + candidate_r,
                         self.active.R + (self.candidate.R if self.candidate else 0))
        self.last = dict(R=old_r, next_R=self.active.R, candidate_R=candidate_r,
                         gamma=gamma, y=float(y_actual[0]), phase=phase_at_start,
                         core_mults=core, candidate_mults=shadow, management_mults=management,
                         total_mults=core + shadow + management, exp_evals=exponentials,
                         resident_factor_coeffs=resident_r * (self.active.D1 + self.active.D2),
                         live_terms=old_r + candidate_r if gamma > 0 else old_r)
        self.n += 1
        return ys_actual


@dataclass(frozen=True)
class V12Config(SelectionConfig):
    """Development configuration; the v11 defaults and results remain immutable."""

    cooldown: int = 4000
    max_backoff: int = 16000
    prune_mode: str = "svd"  # "column" is the v11 ablation.
    growth_mode: str = "random"  # "residual" uses only past prediction gradients.

    def __post_init__(self):
        super().__post_init__()
        if self.max_backoff < self.cooldown:
            raise ValueError("max_backoff must cover cooldown")
        if self.prune_mode not in {"svd", "column"}:
            raise ValueError("Unknown prune mode")
        if self.growth_mode not in {"random", "residual"}:
            raise ValueError("Unknown growth mode")


def best_rank_prune(ctrl):
    """Construct a candidate from the best Frobenius rank-(R-1) approximation.

    This uses a library SVD during development. Its observed cost is charged
    conservatively by the v12 manager, but it is not an exact operation count.
    A counted numerical kernel is required before an exact cost claim.
    """
    if ctrl.R <= 1 or ctrl.A.shape[0] != 1:
        raise ValueError("Pruning requires one run and R > 1")
    A, B = ctrl.A[0], ctrl.B[0]
    qa, ra = np.linalg.qr(A, mode="reduced")
    qb, rb = np.linalg.qr(B, mode="reduced")
    u, singular, vt = np.linalg.svd(rb @ ra.T, full_matrices=False)
    keep = ctrl.R - 1
    scale = np.sqrt(singular[:keep])
    candidate = deepcopy(ctrl)
    candidate.A = ((qa @ vt[:keep].T) * scale).reshape(1, ctrl.D1, keep)
    candidate.B = ((qb @ u[:, :keep]) * scale).reshape(1, ctrl.D2, keep)
    candidate.R = keep
    candidate.n_trainable = keep * (ctrl.D1 + ctrl.D2)
    return candidate, singular


class AdaptiveKronV12(AdaptiveKron):
    """v11 controller plus rejection backoff and basis-independent pruning."""

    def __init__(self, initial, rng, config=None, M=20):
        super().__init__(initial, rng, config or V12Config(), M)
        self.blocked_until = {-1: 0, 1: 0}
        self.reject_streak = {-1: 0, 1: 0}
        self.proposal_start = None
        self.proposal_direction = None
        self.decomposition_calls = 0
        self.decomposition_seconds = 0.0
        self.gradient = np.zeros((initial.D2, initial.D1))
        self.gradient_count = 0

    def _screen(self, q, dv):
        cost, exps = ((0, 0) if self.config.prune_mode == "svd"
                       else super()._screen(q, dv))
        if self.config.growth_mode == "residual":
            Q = self.active.mat(q)[0]
            fitted = float(np.sum(self.active.B[0] * (Q @ self.active.A[0])))
            residual = np.array([dv[0] - fitted])
            psi, _ = ac.influence(residual, self.active.sigma)
            if self.gradient_count == 0:
                self.gradient = float(psi[0]) * Q
            else:
                self.gradient = .9 * self.gradient + .1 * float(psi[0]) * Q
            self.gradient_count += 1
            D = self.active.D1 * self.active.D2
            cost += self.active.R * D + self.active.R * self.active.D2 + 3*D + 2
            exps += 1
        return cost, exps

    def _propose(self):
        cfg, r = self.config, self.active.R
        if cfg.r_min == cfg.r_max:
            self.next_probe = self.n + 1 + cfg.cooldown
            return 0
        legal = [d for d in (self.next_direction, -self.next_direction)
                 if cfg.r_min <= r + d <= cfg.r_max]
        eligible = [d for d in legal if self.n >= self.blocked_until[d]]
        if not eligible:
            self.next_probe = min(self.blocked_until[d] for d in legal)
            return 0
        direction = eligible[0]
        if direction == -1 and cfg.prune_mode == "column" and self.screen_count == 0:
            self.next_probe = self.n + 1 + cfg.screen_every
            return 0
        removed = int(np.argmin(self.screen)) if direction == -1 and cfg.prune_mode == "column" else None
        decomp_charge = 0
        singular = None
        if direction == -1 and cfg.prune_mode == "svd":
            start = perf_counter()
            try:
                self.candidate, singular = best_rank_prune(self.active)
            except np.linalg.LinAlgError as exc:
                self.events.append(dict(sample=self.n, event="decomposition_failure", reason=str(exc)))
                self.blocked_until[-1] = self.n + cfg.cooldown
                self.next_probe = self.n + 1 + cfg.cooldown
                return 0
            self.decomposition_seconds += perf_counter() - start
            self.decomposition_calls += 1
            # An explicit, deliberately high accounting charge for library QR/SVD.
            # It is a budget convention, not a measured hardware instruction count.
            decomp_charge = 1_000_000
        else:
            self.candidate = resized_copy(self.active, direction, self.rng, removed)
            if direction == 1 and cfg.growth_mode == "residual" and self.gradient_count:
                started = perf_counter()
                try:
                    _, s, vt = np.linalg.svd(self.gradient, full_matrices=False)
                except np.linalg.LinAlgError:
                    s = np.zeros(1)
                if s[0] > 1e-12:
                    direction_a = vt[0]
                    target_sq = float(np.sum(self.active.A * self.active.A) / r)
                    self.candidate.A[0, :, -1] = direction_a * np.sqrt(target_sq)
                    decomp_charge = 1_000_000
                    self.decomposition_calls += 1
                    self.decomposition_seconds += perf_counter() - started
        self.next_direction = -direction
        self.proposal_start = self.n
        self.proposal_direction = direction
        self.phase, self.age = "train", 0
        self.validation_sum[:] = 0
        self.events.append(dict(sample=self.n, event="proposal", old_R=r,
                                new_R=self.candidate.R, remove_index=removed,
                                prune_mode=cfg.prune_mode if direction == -1 else None,
                                growth_mode=cfg.growth_mode if direction == 1 else None,
                                smallest_singular=float(singular[-1]) if singular is not None else None,
                                decomposition_charge=decomp_charge))
        return (self.active.D1 * (r + 2) + decomp_charge
                if direction == 1 else decomp_charge)

    def _decision(self):
        direction = self.proposal_direction
        super()._decision()
        event = self.events[-1]
        accepted = event["event"] == "accept"
        if accepted:
            self.reject_streak = {-1: 0, 1: 0}
            self.blocked_until = {-1: 0, 1: 0}
        else:
            self.reject_streak[direction] += 1
            wait = min(self.config.cooldown * 2 ** (self.reject_streak[direction] - 1),
                       self.config.max_backoff)
            self.blocked_until[direction] = self.n + 1 + wait
        event.update(direction=direction,
                     loss_delta=event["new_loss"] - event["old_loss"],
                     cost_penalty_delta=self.config.lambda_cost *
                     (self.cost(event["new_R"]) - self.cost(event["old_R"])) / self.c_ref,
                     margin=self.config.margin,
                     candidate_lifetime=self.n - self.proposal_start,
                     rejection_streak=self.reject_streak[direction])
