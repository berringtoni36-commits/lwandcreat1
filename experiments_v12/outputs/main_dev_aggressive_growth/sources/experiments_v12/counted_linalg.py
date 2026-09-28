"""Small deterministic QR/Jacobi-SVD with explicit scalar-multiply ledger.

The ledger counts multiplication operands of the mathematical algorithm,
not CPU instructions, vector loads, additions, divisions, or square roots.
The controller dimensions are at most 25 x 8 and 20 x 8.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Count:
    multiplies: int = 0
    divisions: int = 0
    square_roots: int = 0


def product(left,right,counter):
    if left.shape[1]!=right.shape[0]:
        raise ValueError('Matrix dimensions differ')
    counter.multiplies += left.shape[0]*left.shape[1]*right.shape[1]
    return left@right


def qr_mgs2(matrix,counter):
    """Modified Gram-Schmidt with two reorthogonalization passes."""
    matrix=np.asarray(matrix,float)
    m,n=matrix.shape
    q=np.zeros((m,n));r=np.zeros((n,n))
    tol=1e-13*max(1.,float(np.linalg.norm(matrix)))
    for j in range(n):
        v=matrix[:,j].copy()
        for _ in range(2):
            for i in range(j):
                coeff=float(np.dot(q[:,i],v))
                counter.multiplies += m
                r[i,j] += coeff
                v -= coeff*q[:,i]
                counter.multiplies += m
        norm=float(np.sqrt(np.dot(v,v)))
        counter.multiplies += m
        counter.square_roots += 1
        if norm>tol:
            r[j,j]=norm
            q[:,j]=v/norm
            counter.divisions += m
    return q,r


def jacobi_svd(square,counter,max_sweeps=32,tol=1e-12):
    """One-sided Jacobi SVD, sorted by singular value."""
    work=np.array(square,float,copy=True)
    n=work.shape[0]
    if work.shape!=(n,n):
        raise ValueError('Expected square matrix')
    right=np.eye(n)
    converged=False
    for sweep in range(max_sweeps):
        changed=False
        for p in range(n):
            for q in range(p+1,n):
                cp=work[:,p].copy();cq=work[:,q].copy()
                app=float(np.dot(cp,cp));aqq=float(np.dot(cq,cq));apq=float(np.dot(cp,cq))
                counter.multiplies += 3*n
                counter.multiplies += 1
                counter.square_roots += 1
                if abs(apq)<=tol*np.sqrt(app*aqq):
                    continue
                changed=True
                tau=(aqq-app)/(2*apq)
                counter.multiplies += 1
                counter.divisions += 1
                t=(1 if tau>=0 else -1)/(abs(tau)+np.sqrt(1+tau*tau))
                counter.multiplies += 1
                counter.square_roots += 1
                counter.divisions += 1
                c=1/np.sqrt(1+t*t)
                s=c*t
                counter.multiplies += 2
                counter.square_roots += 1
                counter.divisions += 1
                work[:,p]=c*cp-s*cq
                work[:,q]=s*cp+c*cq
                counter.multiplies += 4*n
                vp=right[:,p].copy();vq=right[:,q].copy()
                right[:,p]=c*vp-s*vq
                right[:,q]=s*vp+c*vq
                counter.multiplies += 4*n
        if not changed:
            converged=True
            break
    if not converged:
        raise ValueError(f'Jacobi SVD did not converge after {max_sweeps} sweeps')
    singular=np.sqrt(np.sum(work*work,axis=0))
    counter.multiplies += n*n
    counter.square_roots += n
    left=np.zeros_like(work)
    for j,s in enumerate(singular):
        if s>1e-14:
            left[:,j]=work[:,j]/s
            counter.divisions += n
    order=np.argsort(-singular)
    return left[:,order],singular[order],right[:,order],sweep+1


def counted_prune(ctrl):
    """Return rank-(R-1) candidate and exact modeled multiply count."""
    if ctrl.R<=1 or ctrl.A.shape[0]!=1:
        raise ValueError('Expected one controller with R>1')
    from copy import deepcopy
    count=Count()
    qa,ra=qr_mgs2(ctrl.A[0],count)
    qb,rb=qr_mgs2(ctrl.B[0],count)
    middle=product(rb,ra.T,count)
    u,s,v,sweeps=jacobi_svd(middle,count)
    keep=ctrl.R-1
    scale=np.sqrt(s[:keep]);count.square_roots += keep
    Anew=product(qa,v[:,:keep],count)
    Bnew=product(qb,u[:,:keep],count)
    Anew *= scale
    Bnew *= scale
    count.multiplies += (ctrl.D1+ctrl.D2)*keep
    candidate=deepcopy(ctrl)
    candidate.A=Anew.reshape(1,ctrl.D1,keep)
    candidate.B=Bnew.reshape(1,ctrl.D2,keep)
    candidate.R=keep
    candidate.n_trainable=keep*(ctrl.D1+ctrl.D2)
    return candidate,s,count,sweeps
