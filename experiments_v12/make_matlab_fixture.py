"""Export common arrays and an independently replayed Python R=4 trace."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
from scipy.io import savemat
from adaptive_core import ac


def make(source,dest,n_samples=1000):
    source=Path(source)
    with np.load(source) as data:
        x=data['x'][:n_samples].astype(float)
        d=data['d'][:n_samples].astype(float)
        v=data['v'][:n_samples].astype(float)
        Om=data['Om'].astype(float)
        ph=data['ph'].astype(float)
        A0=data['initial_A'].astype(float)
        B0=data['initial_B'].astype(float)
    ctrl=ac.KronRFF('python_R4',1,25,20,4,.1,.1,2.,1e-8,np.random.default_rng(0))
    ctrl.A=A0[:,:,:4].copy();ctrl.B=B0[:,:,:4].copy()
    xb=np.zeros((1,20));zh=np.zeros((4,1,500))
    z_all=np.zeros((n_samples,500));q_all=np.zeros_like(z_all)
    y_all=np.zeros(n_samples);ys_all=np.zeros(n_samples)
    for n in range(n_samples):
        xb[:,1:]=xb[:,:-1].copy();xb[0,0]=x[n]
        z=np.sqrt(2/500)*np.cos(np.einsum('rdm,rm->rd',Om,xb)+ph)
        zh[1:]=zh[:-1].copy();zh[0]=z
        q=zh[2]+.5*zh[3]
        ys_all[n]=ctrl.step(z,q,np.array([d[n]+v[n]]))[0]
        y_all[n]=ctrl.ybuf[0,0]
        z_all[n]=z[0];q_all[n]=q[0]
    savemat(dest,dict(x=x,d=d,v=v,Om=Om[0],ph=ph[0],A0=A0[0,:,:4],
                      B0=B0[0,:,:4],z_expected=z_all,q_expected=q_all,
                      y_expected=y_all,ys_expected=ys_all,
                      A_final=ctrl.A[0],B_final=ctrl.B[0]),do_compression=True)
    print(dest)


if __name__=='__main__':
    if len(sys.argv) not in (3,4):
        raise SystemExit('Usage: python make_matlab_fixture.py SOURCE.npz DEST.mat [samples]')
    make(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)==4 else 1000)
