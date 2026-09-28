"""Export one full dynamic trace for independent MATLAB physical/score audit."""
from pathlib import Path
import json
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'.deps'))
from scipy.io import savemat


def make(source,target):
    source=Path(source)
    with np.load(source) as f:
        meta=json.loads(str(f['meta']))
        method=meta['controllers'].index('adaptive')
        payload={name:f[name].copy() for name in
                 ('d','v','actuator_output','R','core_mults',
                  'candidate_mults','management_mults','total_mults')}
        payload['anr_expected']=f['anr'][method].astype(float)
        payload['ys_expected']=f['secondary_output'][method].astype(float)
    events=json.loads(source.with_name(source.stem+'_events.json').read_text(encoding='utf-8'))
    decisions=[e for e in events if e['event'] in ('accept','reject')]
    payload['decision_sample']=np.array([e['sample'] for e in decisions],dtype=np.int64)
    payload['decision_old_R']=np.array([e['old_R'] for e in decisions],dtype=float)
    payload['decision_new_R']=np.array([e['new_R'] for e in decisions],dtype=float)
    payload['decision_old_loss']=np.array([e['old_loss'] for e in decisions],dtype=float)
    payload['decision_new_loss']=np.array([e['new_loss'] for e in decisions],dtype=float)
    payload['decision_accepted']=np.array([e['event']=='accept' for e in decisions],dtype=np.uint8)
    payload['lambda_cost']=float(meta['selection']['lambda_cost'])
    payload['margin']=float(meta['selection']['margin'])
    savemat(target,payload,do_compression=True)
    print(target)


if __name__=='__main__':
    if len(sys.argv)!=3:
        raise SystemExit('Usage: python make_matlab_audit.py RUN.npz FIXTURE.mat')
    make(sys.argv[1],sys.argv[2])
