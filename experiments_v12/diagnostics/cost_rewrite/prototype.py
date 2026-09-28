"""Mathematically equivalent fixed NKP-RFF-MCC update with a cheaper multiply DAG.

This is a diagnostic prototype. It does not alter experiments_v12 or v10.
The representation is A_tilde = sqrt(2/D) * A; B remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizedFixed:
    A_tilde: np.ndarray
    B: np.ndarray
    mu_a: float
    mu_b: float
    sigma: float
    eps: float
    D1: int
    D2: int

    def __post_init__(self):
        assert self.A_tilde.shape == (self.D1, self.B.shape[1])
        assert self.B.shape[0] == self.D2
        self.scale_squared = 2.0 / (self.D1*self.D2)
        self.ybuf = np.zeros(4)

    @property
    def R(self):
        return self.B.shape[1]

    @classmethod
    def from_reference(cls, reference):
        scale = np.sqrt(2.0/(reference.D1*reference.D2))
        return cls((scale*reference.A[0]).copy(), reference.B[0].copy(),
                   reference.mu_a, reference.mu_b, reference.sigma,
                   reference.eps, reference.D1, reference.D2)

    def step(self, raw_z, raw_q, dv):
        Z = raw_z.reshape(self.D1,self.D2).T
        Q = raw_q.reshape(self.D1,self.D2).T
        y = float(np.sum(self.B*(Z @ self.A_tilde)))
        self.ybuf[1:] = self.ybuf[:-1].copy()
        self.ybuf[0] = y
        physical_secondary = self.ybuf[2] + .5*self.ybuf[3]

        UB = Q @ self.A_tilde
        e1 = dv - float(np.sum(self.B*UB))
        psi1 = e1*np.exp(-e1*e1/(2*self.sigma*self.sigma))
        norm_b = float(np.sum(UB*UB))
        kb = self.mu_b*psi1/(self.eps+norm_b)
        self.B += kb*UB

        # A:Q^T B_new == B_new:Q A, so the second sequential residual
        # follows from the first without another 25R-element dot product.
        e2 = e1-kb*norm_b
        UA_raw = Q.T @ self.B
        norm_a = self.scale_squared*float(np.sum(UA_raw*UA_raw))
        psi2 = e2*np.exp(-e2*e2/(2*self.sigma*self.sigma))
        ka = self.mu_a*psi2/(self.eps+norm_a)
        self.A_tilde += (self.scale_squared*ka)*UA_raw
        return physical_secondary, y, e1, e2


def cost_breakdown(R):
    """Reference accounting convention: scalar real multiplies per sample."""
    D1,D2,M,D,Ls = 25,20,20,500,4
    original = dict(rff_projection=D*M, feature_scale=D,
                    filtered_x=D*Ls, output_projection=R*D,
                    output_inner=R*D2, ub_projection=R*D,
                    first_residual_inner=R*D2, ub_norm=R*D2,
                    b_update=R*D2, ua_projection=R*D,
                    second_residual_inner=R*D1, ua_norm=R*D1,
                    a_update=R*D1, scalar=8, physical_fir=Ls)
    rewritten = dict(rff_projection=D*M, feature_scale=0,
                     filtered_x=D, output_projection=R*D,
                     output_inner=R*D2, ub_projection=R*D,
                     first_residual_inner=R*D2, ub_norm=R*D2,
                     b_update=R*D2, ua_projection=R*D,
                     second_residual_inner=0, ua_norm=R*D1,
                     a_update=R*D1, scalar=11, physical_fir=1)
    assert sum(original.values()) == 12512+1655*R
    assert sum(rewritten.values()) == 10512+1630*R
    return original, rewritten
