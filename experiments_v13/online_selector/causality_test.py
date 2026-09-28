"""Same prefix/different future test for the file-free online simulator."""
from __future__ import annotations

import json

import numpy as np

from online_selector import HERE,simulate


def main():
    rng=np.random.default_rng(2026092891)
    T=26000;cut=23000
    x=rng.standard_normal(T)
    d=.6*np.roll(x,3)+.02*rng.standard_normal(T)
    v=.01*rng.standard_normal(T)
    om=rng.normal(0,1/3.9,(1,500,20))
    ph=rng.uniform(0,2*np.pi,(1,500))
    A=.01*rng.standard_normal((1,25,8))
    B=.01*rng.standard_normal((1,20,8))
    x2=x.copy();d2=d.copy();v2=v.copy()
    x2[cut:]=rng.standard_normal(T-cut)*3
    d2[cut:]=rng.standard_normal(T-cut)*2
    v2[cut:]=rng.standard_normal(T-cut)*.1
    cfg={'arm':'once','proposal':'signed_sort','ramp_samples':100}
    a=simulate(x,d,v,om,ph,A,B,.1,cfg)
    b=simulate(x2,d2,v2,om,ph,A,B,.1,cfg)
    names=('drive','error','anr','R','candidate_R','retiring_R',
           'ys_active','ys_candidate','ys_actual','gamma')
    checks={k:bool(np.array_equal(a[k][:cut],b[k][:cut],equal_nan=True)) for k in names}
    checks['cost']=all(np.array_equal(a['cost'][k][:cut],b['cost'][k][:cut]) for k in a['cost'])
    checks['prefix_events']=([e for e in a['events'] if e['n']<cut]==
                            [e for e in b['events'] if e['n']<cut])
    assert all(checks.values()),checks
    suffix=float(np.max(np.abs(a['drive'][cut:]-b['drive'][cut:])))
    assert suffix>1e-3
    result={'phase':'synthetic_unit_test','same_prefix_samples':cut,
            'different_future_samples':T-cut,'checks':checks,
            'future_drive_max_difference':suffix,
            'events_before_cut':[e for e in a['events'] if e['n']<cut]}
    (HERE/'causality_test.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
