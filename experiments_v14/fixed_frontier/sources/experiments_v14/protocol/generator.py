"""N14 physical-ANC data generator prototype; development arrays only.

The select/confirm namespaces are reserved but this module refuses to
materialize their data before the controller and analysis are frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Optional

import numpy as np


PRIMARY_FIR = np.array([0., 0., 0., 1., -0.3, 0.2])
SECONDARY_FIR = np.array([0., 0., 1., 0.5])
CASE_IDS = {"E1": 1, "E2": 2, "E3": 3,
            "C1": 11, "C2": 12, "C3": 13, "C4": 14}
PHASE_IDS = {"dev": 1, "select": 2, "confirm": 3}
COMPONENT_IDS = {"reference": 1, "rff": 2, "initial_factors": 3,
                 "measurement_noise": 4, "teacher": 5, "structure": 41}
DEFAULT_LENGTHS = {"E1": 300_000, "E2": 400_000, "E3": 400_000,
                   "C1": 20_000, "C2": 20_000, "C3": 30_000, "C4": 30_000}
SEGMENT_LENGTH = 100_000


@dataclass(frozen=True)
class CaseData:
    case: str
    run: int
    phase: str
    smoke_only: bool
    x: np.ndarray
    d: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    rff_phase: np.ndarray
    initial_A: np.ndarray
    initial_B: np.ndarray
    segment_bounds: tuple[tuple[int, int], ...]
    seed_keys: dict[str, tuple[int, ...]]
    teacher_low: Optional[np.ndarray] = None
    teacher_high: Optional[np.ndarray] = None


def seed_key(phase: str, case: str, run: int, component: str) -> tuple[int, ...]:
    """Return a reserved namespace key without generating any samples."""
    if phase not in PHASE_IDS or case not in CASE_IDS or component not in COMPONENT_IDS:
        raise ValueError("Unknown phase, case, or component")
    if not isinstance(run, int) or run < 0:
        raise ValueError("run must be a nonnegative integer")
    return (20260928, 14, PHASE_IDS[phase], CASE_IDS[case], run,
            COMPONENT_IDS[component])


def _dev_rng(case: str, run: int, component: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed_key("dev", case, run, component)))


def causal_fir(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Zero-state causal FIR, including zero-valued leading taps."""
    signal = np.asarray(signal, dtype=np.float64)
    taps = np.asarray(taps, dtype=np.float64)
    if signal.ndim != 1 or taps.ndim != 1:
        raise ValueError("FIR expects one-dimensional arrays")
    out = np.zeros_like(signal)
    for delay, tap in enumerate(taps):
        if tap != 0. and delay < len(signal):
            out[delay:] += tap * signal[:len(signal)-delay]
    return out


def static_primary(x: np.ndarray) -> np.ndarray:
    """Original C1-C4 primary path, g=P*x and cubic physical nonlinearity."""
    g = causal_fir(x, PRIMARY_FIR)
    d = np.zeros_like(g)
    d[2:] = g[:-2] + .08*g[:-2]**2 - .04*g[1:-1]**3
    return d


def e2e3_primary(x: np.ndarray, segment_length: int) -> np.ndarray:
    """Original four-stage nonlinear primary path, no RFF teacher."""
    if len(x) != 4*segment_length or segment_length < 1:
        raise ValueError("E2/E3 require four equally long stages")
    g = causal_fir(x, PRIMARY_FIR)
    a2 = np.repeat([.02, .08, .16, .02], segment_length)
    a3 = np.repeat([.01, .04, .08, .01], segment_length)
    d = np.zeros_like(g)
    d[2:] = g[:-2] + a2[2:]*g[:-2]**2 - a3[2:]*g[1:-1]**3
    return d


def symmetric_alpha_stable(rng: np.random.Generator, alpha: float,
                           size: int | tuple[int, ...]) -> np.ndarray:
    """Symmetric unit-scale Chambers-Mallows-Stuck samples."""
    u = rng.uniform(-np.pi/2, np.pi/2, size=size)
    w = rng.exponential(1., size=size)
    if abs(alpha-1.) < 1e-12:
        return np.tan(u)
    return (np.sin(alpha*u)/np.cos(u)**(1./alpha)
            * (np.cos((1.-alpha)*u)/w)**((1.-alpha)/alpha))


def logistic_delay6(rng: np.random.Generator, length: int) -> np.ndarray:
    """Six-interleaved logistic sequences, centered and sample-normalized."""
    if length < 12:
        raise ValueError("Logistic sequence too short")
    while True:
        u = np.empty(length, dtype=np.float64)
        u[:6] = rng.uniform(.1, .9, 6)
        for n in range(6, length):
            p = u[n-6]
            u[n] = 4.*p*(1.-p)
        tail = u[-600:]
        if all(np.std(tail[k::6]) > 1e-3 for k in range(6)):
            break
    u -= u.mean()
    u /= u.std()
    return u


def alpha_reference(rng: np.random.Generator, length: int) -> np.ndarray:
    x = symmetric_alpha_stable(rng, 1.6, length)
    x /= np.median(np.abs(x))
    return np.clip(x, -5., 5.)


def rff_features(x: np.ndarray, omega: np.ndarray, phase: np.ndarray,
                 start: int, stop: int) -> np.ndarray:
    """Dense original 500-D map for a causal block [start, stop)."""
    h = np.zeros((stop-start, 20), dtype=np.float64)
    for lag in range(20):
        first = max(start, lag)
        if first < stop:
            h[first-start:, lag] = x[first-lag:stop-lag]
    return np.sqrt(2./500.) * np.cos(h @ omega.T + phase)


def e1_teacher_primary(x: np.ndarray, omega: np.ndarray, phase: np.ndarray,
                       teacher_rng: np.random.Generator,
                       segment_length: int, block_size: int = 4096
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original E1 low/high/low rank-1/rank-4/rank-1 RFF teacher."""
    if len(x) != 3*segment_length or segment_length < 1:
        raise ValueError("E1 requires three equally long stages")
    u, _ = np.linalg.qr(teacher_rng.standard_normal((20, 4)))
    basis, _ = np.linalg.qr(teacher_rng.standard_normal((25, 4)))
    low = 10.*np.outer(u[:, 0], basis[:, 0])
    high = 5.*(u @ basis.T)
    low_vec = low.T.reshape(500)
    high_vec = high.T.reshape(500)
    raw = np.zeros(len(x), dtype=np.float64)
    for start in range(0, len(x), block_size):
        stop = min(len(x), start+block_size)
        z = rff_features(x, omega, phase, start, stop)
        for part_start, part_stop, weights in ((0, segment_length, low_vec),
                                               (segment_length, 2*segment_length, high_vec),
                                               (2*segment_length, len(x), low_vec)):
            lo, hi = max(start, part_start), min(stop, part_stop)
            if lo < hi:
                raw[lo:hi] = z[lo-start:hi-start] @ weights
    return causal_fir(raw, SECONDARY_FIR), low, high


def physical_error(d: np.ndarray, v: np.ndarray, actuator: np.ndarray) -> np.ndarray:
    if len(d) != len(v) or len(d) != len(actuator):
        raise ValueError("Signal lengths differ")
    return d + v - causal_fir(actuator, SECONDARY_FIR)


def make_case(case: str, run: int, *, phase: str = "dev",
              smoke_segment: int | None = None,
              smoke_length: int | None = None) -> CaseData:
    """Materialize actual arrays only for N14 development runs 0..4.

    Short overrides require a smoke tag and cannot count as full dev runs.
    The select/confirm namespaces exist only in seed_key until formal freeze.
    """
    if phase != "dev":
        raise ValueError("N14 prototype refuses selection/confirmation materialization")
    if case not in CASE_IDS or not isinstance(run, int) or not 0 <= run < 5:
        raise ValueError("Only N14 dev cases and runs 0..4 are available")
    if smoke_segment is not None and smoke_length is not None:
        raise ValueError("Use one short-sequence override")
    if smoke_segment is not None and (case not in ("E1", "E2", "E3") or smoke_segment < 20):
        raise ValueError("smoke_segment is only for E cases and must be >=20")
    if smoke_length is not None and (case not in ("C1", "C2", "C3", "C4")
                                     or smoke_length < 12):
        raise ValueError("smoke_length is only for C cases and must be >=12")
    smoke = smoke_segment is not None or smoke_length is not None
    segment = smoke_segment or SEGMENT_LENGTH
    length = (3 if case == "E1" else 4)*segment if case in ("E1", "E2", "E3") \
        else (smoke_length or DEFAULT_LENGTHS[case])
    ref_rng = _dev_rng(case, run, "reference")
    if case == "C1":
        x = logistic_delay6(ref_rng, length)
    elif case == "C2":
        x = alpha_reference(ref_rng, length)
    else:
        x = ref_rng.standard_normal(length)
    mapping_rng = _dev_rng(case, run, "rff")
    omega = mapping_rng.normal(0., 1./3.9, (500, 20))
    rff_phase = mapping_rng.uniform(0., 2.*np.pi, 500)
    teacher_low = teacher_high = None
    if case == "E1":
        d, teacher_low, teacher_high = e1_teacher_primary(
            x, omega, rff_phase, _dev_rng(case, run, "teacher"), segment)
    elif case in ("E2", "E3"):
        d = e2e3_primary(x, segment)
    else:
        d = static_primary(x)
    if case in ("C1", "C2"):
        v = np.zeros(length)
    elif case in ("C4", "E3"):
        v = .05*symmetric_alpha_stable(_dev_rng(case, run, "measurement_noise"),
                                       1.6, length)
    else:
        v = .01*_dev_rng(case, run, "measurement_noise").standard_normal(length)
    factor_rng = _dev_rng(case, run, "initial_factors")
    initial_a = .01*factor_rng.standard_normal((25, 8))
    initial_b = .01*factor_rng.standard_normal((20, 8))
    bounds = (tuple((k*segment, (k+1)*segment) for k in range(3 if case == "E1" else 4))
              if case in ("E1", "E2", "E3") else ((0, length),))
    keys = {component: seed_key("dev", case, run, component)
            for component in COMPONENT_IDS}
    return CaseData(case, run, phase, smoke, x, d, v, omega, rff_phase,
                    initial_a, initial_b, bounds, keys, teacher_low, teacher_high)


def validate_case(data: CaseData, *, require_full: bool = True) -> dict[str, object]:
    """Check a generated array bundle before it is admitted to a dev run.

    Returns hashes for a future manifest.  This validator does not train a
    controller and refuses short smoke arrays as full development inputs.
    """
    if data.phase != "dev" or data.case not in CASE_IDS or not 0 <= data.run < 5:
        raise ValueError("Only N14 development data can be validated here")
    if require_full and (data.smoke_only or len(data.x) != DEFAULT_LENGTHS[data.case]):
        raise ValueError("Full-length development array required")
    if data.seed_keys != {part: seed_key("dev", data.case, data.run, part)
                          for part in COMPONENT_IDS}:
        raise ValueError("Seed namespace mismatch")
    t = len(data.x)
    shapes = {"x": (t,), "d": (t,), "v": (t,), "omega": (500, 20),
              "rff_phase": (500,), "initial_A": (25, 8), "initial_B": (20, 8)}
    for name, shape in shapes.items():
        arr = getattr(data, name)
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"Malformed or nonfinite {name}")
    if data.segment_bounds[0][0] != 0 or data.segment_bounds[-1][1] != t or any(
            data.segment_bounds[k][1] != data.segment_bounds[k+1][0]
            for k in range(len(data.segment_bounds)-1)):
        raise ValueError("Invalid contiguous segment bounds")
    if data.case in ("E2", "E3"):
        if len(data.segment_bounds) != 4:
            raise ValueError("Expected four E2/E3 stages")
        segment = data.segment_bounds[0][1]
        expected_d = e2e3_primary(data.x, segment)
    elif data.case == "E1":
        if len(data.segment_bounds) != 3:
            raise ValueError("Expected three E1 stages")
        segment = data.segment_bounds[0][1]
        expected_d, low, high = e1_teacher_primary(
            data.x, data.omega, data.rff_phase,
            _dev_rng(data.case, data.run, "teacher"), segment)
        if (data.teacher_low is None or data.teacher_high is None
                or not np.allclose(data.teacher_low, low, atol=1e-12, rtol=0)
                or not np.allclose(data.teacher_high, high, atol=1e-12, rtol=0)):
            raise ValueError("Teacher matrices differ from reserved stream")
    else:
        if data.segment_bounds != ((0, t),):
            raise ValueError("Static bridge must have one segment")
        expected_d = static_primary(data.x)
    primary_gap = float(np.max(np.abs(expected_d - data.d)))
    if primary_gap > 1e-12:
        raise ValueError(f"Primary path mismatch: {primary_gap}")
    if data.case in ("C1", "C2") and np.any(data.v != 0):
        raise ValueError("C1/C2 measurement noise must be zero")
    digests = {name: hashlib.sha256(np.ascontiguousarray(getattr(data, name)).tobytes()).hexdigest()
               for name in shapes}
    return dict(case=data.case, run=data.run, phase=data.phase,
                smoke_only=data.smoke_only, length=t,
                primary_max_abs_gap=primary_gap, array_sha256=digests)
