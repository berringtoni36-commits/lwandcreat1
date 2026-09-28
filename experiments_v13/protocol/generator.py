"""Development/selection-only prototype for the non-dictionary P13 primary path.

This module does not run an ANC controller or expose confirmation samples.
The confirmation runner must be implemented only after protocol/code freeze.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


STAGES = (120_000, 80_000, 120_000)
PRIMARY_FIR = np.array([0., 0., 0., 1., -0.3, 0.2])
SECONDARY_FIR = np.array([0., 0., 1., 0.5])
PAIRS = ((0, 3), (1, 5), (2, 7), (3, 9))
INTERACTION_COEFFS = (0.40, 0.35, -0.35, 0.30)
AR_COEFF = 0.7
RFF_BANDWIDTH = 3.9


@dataclass(frozen=True)
class P13Data:
    x: np.ndarray
    d: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    rff_phase: np.ndarray
    active_interactions: np.ndarray


def _stream(phase: str, run: int, component: int) -> np.random.Generator:
    if phase not in {"dev", "select"}:
        raise ValueError("Prototype is locked against confirmation generation")
    if run < 0:
        raise ValueError("run must be nonnegative")
    phase_id = {"dev": 1, "select": 2}[phase]
    return np.random.default_rng(np.random.SeedSequence(
        [20260928, 13, phase_id, 1, run, component]
    ))


def causal_fir(taps: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Causal zero-state FIR; no wraparound at the beginning."""
    return np.convolve(signal, taps, mode="full")[: len(signal)]


def lag(signal: np.ndarray, delay: int) -> np.ndarray:
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    out = np.zeros_like(signal)
    if delay == 0:
        out[:] = signal
    elif delay < len(signal):
        out[delay:] = signal[:-delay]
    return out


def interaction_count(length: int, switches: tuple[int, int] = (120_000, 200_000)) -> np.ndarray:
    a, b = switches
    if not 0 < a < b < length:
        raise ValueError("Require two ordered transitions inside the series")
    count = np.ones(length, dtype=np.uint8)
    count[a:b] = 4
    return count


def physical_primary(x: np.ndarray,
                     switches: tuple[int, int] = (120_000, 200_000)) -> tuple[np.ndarray, np.ndarray]:
    """Nonstationary, nonlinear primary disturbance independent of the RFF map.

    The first cross term is active throughout; three distinct interactions
    switch on in the middle and off again in the final stage.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("P13 expects one time series per run")
    active = interaction_count(len(x), switches)
    g = causal_fir(PRIMARY_FIR, x)
    h = np.tanh(g)
    d = 0.7 * lag(g, 2) + 0.15 * lag(h, 2)
    for j, ((a, b), coefficient) in enumerate(zip(PAIRS, INTERACTION_COEFFS)):
        term = lag(h, 2 + a) * lag(h, 2 + b)
        if j == 0:
            d += coefficient * term
        else:
            middle = slice(switches[0], switches[1])
            d[middle] += coefficient * term[middle]
    return d, active


def make_p13(phase: str, run: int, length: int = sum(STAGES),
             switches: tuple[int, int] = (120_000, 200_000),
             mapping_component: int = 2) -> P13Data:
    """Generate paired inputs for controller development or selection only."""
    if length < 1:
        raise ValueError("length must be positive")
    reference_rng = _stream(phase, run, 1)
    epsilon = reference_rng.standard_normal(length)
    x = np.empty(length)
    x[0] = epsilon[0]
    innovation_scale = math.sqrt(1. - AR_COEFF**2)
    for n in range(1, length):
        x[n] = AR_COEFF * x[n - 1] + innovation_scale * epsilon[n]
    d, active = physical_primary(x, switches)
    v = .01 * _stream(phase, run, 4).standard_normal(length)
    mapping_rng = _stream(phase, run, mapping_component)
    omega = mapping_rng.normal(0., 1. / RFF_BANDWIDTH, (500, 20))
    rff_phase = mapping_rng.uniform(0., 2. * np.pi, 500)
    return P13Data(x, d, v, omega, rff_phase, active)


def physical_error(d: np.ndarray, v: np.ndarray, actuator_output: np.ndarray) -> np.ndarray:
    """The actual microphone error, with the physical secondary FIR."""
    if not (len(d) == len(v) == len(actuator_output)):
        raise ValueError("signals must have the same length")
    return d + v - causal_fir(SECONDARY_FIR, actuator_output)
