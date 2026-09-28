"""Pointwise C1-C4 bridge: fixed simulate mode equals archived v10 R4."""
from __future__ import annotations

import json

import numpy as np

from online_selector import HERE,ac,simulate


def run_case(case):
    rng=np.random.default_rng(2026092892+int(case[-1]))
    T=4000
    if case=='C1':x=ac.logistic_delay6(rng,1,T)[0]
    elif case=='C2':x=ac.alpha_stable_reference(rng,1,T)[0]
    else:x=rng.standard_normal(T)
    d=ac.primary_disturbance(x[None,:])[0]
    if case in ('C1','C2'):v=np.zeros(T)
    elif case=='C3':v=.01*rng.standard_normal(T)
    else:v=.05*ac.sas_cms(rng,1.6,T)
    om,ph=ac.draw_rff(rng,1,500,20,3.9)
    A=.01*rng.standard_normal((1,25,8))
    B=.01*rng.standard_normal((1,20,8))
    mu=.1 if case in ('C2','C4') else .2
    new=simulate(x,d,v,om,ph,A,B,mu,{'arm':'fixed'})
    old=ac.KronRFF('fixed_R4',1,25,20,4,mu/2,mu/2,2.,1e-8,np.random.default_rng(0))
    old.A=A[:,:,:4].copy();old.B=B[:,:,:4].copy()
    xb=np.zeros((1,20));zh=np.zeros((4,1,500))
    old_y=np.zeros(T);old_error=np.zeros(T)
    for n in range(T):
        xb[:,1:]=xb[:,:-1].copy();xb[0,0]=x[n]
        z=np.sqrt(2/500)*np.cos(np.einsum('rdm,rm->rd',om,xb)+ph)
        zh[1:]=zh[:-1].copy();zh[0]=z
        q=zh[2]+.5*zh[3]
        dv=np.array([d[n]+v[n]])
        ys=old.step(z,q,dv)[0]
        old_y[n]=old.ybuf[0,0]
        old_error[n]=d[n]+v[n]-ys
    ydiff=float(np.max(np.abs(old_y-new['drive'])))
    ediff=float(np.max(np.abs(old_error-new['error'])))
    assert ydiff<1e-10 and ediff<1e-10,(ydiff,ediff)
    assert np.all(new['R']==4) and not new['events']
    assert np.all(new['cost']['candidate']==0) and np.all(new['cost']['retiring']==0)
    assert np.all(new['cost']['management']==0) and np.all(new['cost']['reorder']==0)
    assert np.all(new['cost']['total']==ac.mults_kron(25,20,4,20,4)+4)
    return {'case':case,'T':T,'max_actuator_abs_difference':ydiff,
            'max_physical_error_abs_difference':ediff,
            'fixed_cost_mults_per_sample':int(new['cost']['total'][0])}


def main():
    rows=[run_case(c) for c in ('C1','C2','C3','C4')]
    result={'phase':'synthetic_dev_smoke','rows':rows}
    (HERE/'fixed_bridge_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
