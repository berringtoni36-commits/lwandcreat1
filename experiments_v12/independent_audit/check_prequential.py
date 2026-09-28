"""Inspect pre-update physical predictions on a SAVED selection prefix only."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from adaptive_core import ac,AdaptiveKronV12,V12Config

def main():
    folder=HERE.parent/'outputs'/'selection_e1_v500'
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['phase']=='select'
    with np.load(folder/'caseE1_run00.npz') as f:
        A=f['initial_A'];B=f['initial_B'];Om=f['Om'];ph=f['ph']
        x=f['x'][:12000];d=f['d'][:12000];v=f['v'][:12000]
        expected_y=f['actuator_output'][:12000]
    base=ac.KronRFF('audit',1,25,20,2,.1,.1,2.,1e-8,np.random.default_rng(0))
    base.A=A[:,:,:2].copy();base.B=B[:,:,:2].copy()
    ctrl=AdaptiveKronV12(base,np.random.default_rng([protocol['seed'],0,0,4]),
                          V12Config(**protocol['config']))
    xb=np.zeros((1,20));zh=np.zeros((4,1,500));history=np.zeros(4)
    max_physical=max_loss=max_y=0.;validate_samples=0;ramps=0
    for n in range(len(x)):
        xb[:,1:]=xb[:,:-1].copy();xb[0,0]=x[n]
        z=np.sqrt(2/500)*np.cos(np.einsum('rdm,rm->rd',Om,xb)+ph)
        zh[1:]=zh[:-1].copy();zh[0]=z;q=zh[2]+.5*zh[3]
        Z=z.reshape(1,25,20).transpose(0,2,1)
        old_y=float(np.sum(ctrl.active.B*(Z@ctrl.active.A)))
        old_ys=float(ctrl.active.ybuf[0,1]+.5*ctrl.active.ybuf[0,2])
        before=ctrl.validation_sum.copy();phase=ctrl.phase
        actual_y=old_y
        if ctrl.candidate is not None:
            new_y=float(np.sum(ctrl.candidate.B*(Z@ctrl.candidate.A)))
            new_ys=float(ctrl.candidate.ybuf[0,1]+.5*ctrl.candidate.ybuf[0,2])
            if phase=='ramp':
                gamma=(ctrl.age+1)/ctrl.config.ramp_samples
                actual_y=(1-gamma)*old_y+gamma*new_y;ramps+=1
            if phase=='validate':
                physical_errors=np.array([d[n]+v[n]-old_ys,d[n]+v[n]-new_ys])
                loss=-np.expm1(-.5*(physical_errors/2.)**2)
        history[1:]=history[:-1].copy();history[0]=actual_y
        predicted_physical=history[2]+.5*history[3]
        actual=ctrl.step(z,q,np.array([d[n]+v[n]]))[0]
        max_physical=max(max_physical,abs(actual-predicted_physical))
        max_y=max(max_y,abs(ctrl.last['y']-expected_y[n]))
        if phase=='validate':
            max_loss=max(max_loss,float(np.max(np.abs(ctrl.validation_sum-before-loss))))
            validate_samples+=1
    assert max_physical<1e-10 and max_loss<1e-10 and max_y<1e-10
    reused=.01*np.random.default_rng([protocol['seed'],0,0,4]).standard_normal(25)
    result=dict(prefix_samples=len(x),validation_samples_checked=validate_samples,
                ramp_samples_checked=ramps,physical_preupdate_max_abs=max_physical,
                validation_increment_max_abs=max_loss,saved_actuator_replay_max_abs=max_y,
                noise_growth_rng_reuse_confirmed=bool(np.array_equal(reused,v[:25])),
                interpretation='Replays saved selection input only. No new trial or confirmation stream used.')
    (HERE/'prequential_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
