"""Read a saved paired run and scan fixed-rank step sizes without new inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from adaptive_core import ac

MU=(.05,.1,.2,.4,.8)


def sweep(source,ranks=(1,2,3,4,5,6,7,8),max_samples=None):
    source=Path(source)
    with np.load(source) as f:
        x=f['x'];d=f['d'];v=f['v'];Om=f['Om'];ph=f['ph']
        initial_A=f['initial_A'];initial_B=f['initial_B']
        case=json.loads(str(f['meta']))['case']
    T=len(x) if max_samples is None else min(len(x),max_samples)
    controllers=[]
    for rank in ranks:
        for mu in MU:
            c=ac.KronRFF(f'R{rank}_mu{mu}',1,25,20,rank,mu/2,mu/2,2.,1e-8,np.random.default_rng(0))
            c.A=initial_A[:,:,:rank].copy()
            c.B=initial_B[:,:,:rank].copy()
            controllers.append(c)
    nseg=3 if case=='E1' else 4 if case in ('E2','E3') else 1
    sums=np.zeros((nseg,len(controllers)))
    counts=np.zeros(nseg,int)
    Ae=np.zeros(len(controllers));Ad=0.
    xb=np.zeros((1,20));zh=np.zeros((4,1,500))
    start=perf_counter()
    for n in range(T):
        xb[:,1:]=xb[:,:-1].copy();xb[0,0]=x[n]
        z=np.sqrt(2/500)*np.cos(np.einsum('rdm,rm->rd',Om,xb)+ph)
        zh[1:]=zh[:-1].copy();zh[0]=z
        q=zh[2]+.5*zh[3]
        dv=np.array([d[n]+v[n]])
        Ad=.999*Ad+.001*abs(d[n])
        seg=min(nseg-1,nseg*n//T)
        first=seg*T//nseg;end=(seg+1)*T//nseg
        tally=n>=max(first,end-5000)
        if tally:counts[seg]+=1
        for j,c in enumerate(controllers):
            ys=c.step(z,q,dv)[0]
            Ae[j]=.999*Ae[j]+.001*abs(dv[0]-ys)
            if tally:sums[seg,j]+=20*np.log10((Ae[j]+1e-12)/(Ad+1e-12))
    means=sums/counts[:,None]
    result=dict(source=str(source),T=T,ranks=list(ranks),mu=list(MU),
                names=[c.name for c in controllers],segment_anr=means.tolist(),
                seconds=perf_counter()-start)
    target=source.with_name(source.stem+'_fixed_sweep.json')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for rank in ranks:
        choices=[(float(means.mean(axis=0)[j]),controllers[j].name)
                 for j in range(len(controllers)) if controllers[j].R==rank]
        print(rank,min(choices))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('--ranks',type=int,nargs='+',default=[1,2,3,4,5,6,7,8])
    p.add_argument('--max-samples',type=int)
    a=p.parse_args()
    sweep(a.source,a.ranks,a.max_samples)
