"""Development-only numerical check against the unchanged v10 fixed controller."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]/'experiments_v10'))
import anc_core as ac
from prototype import OptimizedFixed, cost_breakdown


def run_case(rank, impulsive, T=4096):
    rng = np.random.default_rng([917,rank,int(impulsive)])
    x = rng.standard_normal(T)
    d = ac.primary_disturbance(x)
    noise = (.03*rng.standard_t(1.6,T) if impulsive
             else .01*rng.standard_normal(T))
    Om = rng.normal(0.,1/3.9,size=(500,20))
    ph = rng.uniform(0.,2*np.pi,size=500)
    ref = ac.KronRFF('reference',1,25,20,rank,.1,.1,2.,1e-8,
                     np.random.default_rng([918,rank]))
    opt = OptimizedFixed.from_reference(ref)
    xb = np.zeros(20)
    hist_raw = np.zeros((4,500))
    hist_scaled = np.zeros((4,500))
    scale = np.sqrt(2/500)
    maxima = dict(physical_secondary=0.,actuator=0.,first_residual=0.,
                  second_residual=0.,factor_A=0.,factor_B=0.)
    for n in range(T):
        xb[1:] = xb[:-1].copy();xb[0]=x[n]
        raw_z = np.cos(Om@xb+ph)
        z = scale*raw_z
        hist_raw[1:] = hist_raw[:-1].copy();hist_raw[0] = raw_z
        hist_scaled[1:] = hist_scaled[:-1].copy();hist_scaled[0] = z
        raw_q = hist_raw[2]+.5*hist_raw[3]
        q = hist_scaled[2]+.5*hist_scaled[3]
        dv = np.array([d[n]+noise[n]])

        # The explicit e1/e2 reference is formed before and after the v10
        # update. It is for checking only; it is not used by the prototype.
        A_before = ref.A[0].copy()
        B_before = ref.B[0].copy()
        Q = q.reshape(25,20).T
        ub_ref = Q@A_before
        e1_ref = float(dv[0]-np.sum(B_before*ub_ref))
        ys_ref = float(ref.step(z[None,:],q[None,:],dv)[0])
        e2_ref = float(dv[0]-np.sum(A_before*(Q.T@ref.B[0])))
        ys_opt,y_opt,e1_opt,e2_opt = opt.step(raw_z,raw_q,float(dv[0]))
        maxima['physical_secondary'] = max(maxima['physical_secondary'],
                                            abs(ys_opt-ys_ref))
        maxima['actuator'] = max(maxima['actuator'],
                                abs(y_opt-ref.ybuf[0,0]))
        maxima['first_residual'] = max(maxima['first_residual'],
                                       abs(e1_opt-e1_ref))
        maxima['second_residual'] = max(maxima['second_residual'],
                                        abs(e2_opt-e2_ref))
        maxima['factor_A'] = max(maxima['factor_A'],
                                 float(np.max(np.abs(opt.A_tilde/scale-ref.A[0]))))
        maxima['factor_B'] = max(maxima['factor_B'],
                                 float(np.max(np.abs(opt.B-ref.B[0]))))
    assert max(maxima.values()) < 1e-8, maxima
    original,rewritten = cost_breakdown(rank)
    assert sum(original.values()) == ac.mults_kron(25,20,rank,20,4)+4
    return dict(rank=rank,impulsive=impulsive,T=T,max_abs=maxima,
                original_multiplies=sum(original.values()),
                rewritten_multiplies=sum(rewritten.values()),
                ratio_to_original_R4=sum(rewritten.values())/19132.,
                original_breakdown=original,rewritten_breakdown=rewritten)


if __name__=='__main__':
    result=[run_case(rank,impulsive) for rank in (2,3,4)
            for impulsive in (False,True)]
    (HERE/'numeric_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('rank','impulsive','max_abs',
                                          'original_multiplies','rewritten_multiplies')}
                      for r in result]))
